"""Train the refinement network (Stage 3) -- paper section "Refinement Model".

Operates on the N-sample candidate pool cached by gen_pool_cache.py: every voxel
proposed by at least one of the N draws, plus a 1-ring dilation. Per candidate:

    - own vote fraction            count / N
    - 6-conn neighbour fractions   count(neighbour) / N   (6 values, 0 if absent)
    - number of present neighbours (surface-ness)
    - normalised position          (x,y,z) standardised by dataset Mean/Std
    - height within tower          (z - z_min) / (z_max - z_min)

Target: 1 if the voxel is in the GT set. Loss: class-weighted BCE.

Two arms:
  --use_image   the deployed model -- RefinementNetwork, which cross-attends to
                the same DINOv2 patch memory the transformer saw, so a candidate
                can ask "is there actually a block here in the photos?".
  (default)     a context-only MLP with NO image conditioning -- the ablation
                isolating how much the image contributes over vote count and
                local geometry alone.

Both are scored against the flat vote-threshold baseline, tau swept on val.
Tower IoU is recovered from (label, n_gt) without materialising the GT set:
inter = #kept with label 1, union = |kept| + n_gt - inter.

LR-sensitive: 2e-4 is stable, larger values destabilise training.

    CUDA_VISIBLE_DEVICES=<uuid> uv run python \
        script/train_refinement.py --name <run> --use_image --gpu_id 0
"""
import argparse
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from utils.paths import DATA_ROOT

from os.path import join as pjoin
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from utils.voxel import GRID
from utils.fixseed import fixseed

CHECKPOINTS_DIR = './checkpoints'
DATASET_NAME = 'myarch'
DATA_ROOT = DATA_ROOT

NEIGH6 = np.array([[GRID[0], 0, 0], [-GRID[0], 0, 0],
                   [0, GRID[1], 0], [0, -GRID[1], 0],
                   [0, 0, GRID[2]], [0, 0, -GRID[2]]], dtype=np.int64)
N_FEAT = 1 + 6 + 1 + 3 + 1 + 3   # own + 6 neighbours + n_present + pos(3) + height + sym(3)
#  sym(3) = mirror-X / mirror-Y / rot180 partner vote-fraction (cols 12,13,14).
#  pos stays at cols 8:11 (sym appended at the END) — _collate_c reads pos = vf[...,8:11].


def tower_features(vox, count, N, mean, std):
    """vox (M,3) int, count (M,) → feats (M, N_FEAT) float32."""
    M = len(vox)
    cmap = {tuple(v): c for v, c in zip(vox.tolist(), count.tolist())}
    own = count.astype(np.float32) / N
    nb = np.zeros((M, 6), np.float32)
    npres = np.zeros(M, np.float32)
    vl = vox.tolist()
    for j, d in enumerate(NEIGH6):
        for i, v in enumerate(vl):
            c = cmap.get((v[0] + d[0], v[1] + d[1], v[2] + d[2]))
            if c is not None:
                nb[i, j] = c / N
                npres[i] += 1
    pos = (vox.astype(np.float32) - mean) / std
    z = vox[:, 2].astype(np.float32)
    zr = (z - z.min()) / (z.max() - z.min() + 1e-6)
    # Symmetry: vote fraction of each voxel's mirror partner, reflected about the
    # POOL (count>0) bbox centre so it's computable at inference (no GT). Towers are
    # ~88% self-symmetric; ~64% of zero-coverage voxels have their mirror in the pool
    # → tells C "my mirror is a confirmed block, so I'm probably real" (kills the
    # count=0 ambiguity that caps C; dilation already supplies the candidates).
    pv = vox[count > 0]
    if len(pv) == 0:
        pv = vox
    sx = int(pv[:, 0].min() + pv[:, 0].max())
    sy = int(pv[:, 1].min() + pv[:, 1].max())
    sym = np.zeros((M, 3), np.float32)
    for i, v in enumerate(vl):
        sym[i, 0] = cmap.get((sx - v[0], v[1], v[2]), 0) / N           # mirror-X
        sym[i, 1] = cmap.get((v[0], sy - v[1], v[2]), 0) / N           # mirror-Y
        sym[i, 2] = cmap.get((sx - v[0], sy - v[1], v[2]), 0) / N      # rot180
    return np.concatenate([own[:, None], nb, (npres / 6)[:, None], pos, zr[:, None], sym],
                          axis=1).astype(np.float32)


def load_split(pool_dir, split_file, N, mean, std, limit=0):
    with open(split_file) as f:
        names = sorted(l.strip() for l in f if l.strip())
    towers = []
    for nm in tqdm(names, desc=os.path.basename(split_file)):
        p = pjoin(pool_dir, nm + '.npz')
        if not os.path.exists(p):
            continue
        d = np.load(p)
        vox, count, label = d['vox'].astype(np.int64), d['count'], d['label']
        feat = tower_features(vox, count, N, mean, std)
        towers.append({'feat': feat, 'label': label.astype(np.float32),
                       'count': count.astype(np.float32), 'n_gt': int(d['n_gt']),
                       'name': nm})
        if limit and len(towers) >= limit:
            break
    return towers


class MLP(nn.Module):
    def __init__(self, in_dim, hidden=128, layers=3, dropout=0.1):
        super().__init__()
        h = []
        d = in_dim
        for _ in range(layers):
            h += [nn.Linear(d, hidden), nn.GELU(), nn.Dropout(dropout)]
            d = hidden
        h.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*h)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def iou_from_keep(keep, label, n_gt):
    inter = float(label[keep].sum())
    union = float(keep.sum()) + n_gt - inter
    return inter / union if union else 1.0


def flat_baseline(towers, taus):
    """Baseline A: best flat vote threshold, model-independent (count only)."""
    out = {t: 0.0 for t in taus}
    for tw in towers:
        cf = tw['count'] / tw['count'].max()
        for t in taus:
            out[t] += iou_from_keep(cf >= t, tw['label'], tw['n_gt'])
    n = len(towers)
    return {t: v / n for t, v in out.items()}


def eval_towers(towers, model, device, taus):
    """Returns dict tau→mean classifier IoU, plus best flat-threshold (baseline A)."""
    model.eval()
    cls = {t: 0.0 for t in taus}
    flat = {t: 0.0 for t in taus}
    N = None
    with torch.no_grad():
        for tw in towers:
            x = torch.from_numpy(tw['feat']).to(device)
            p = torch.sigmoid(model(x)).cpu().numpy()
            cf = tw['count'] / tw['count'].max()      # flat uses raw vote fraction
            if N is None:
                N = tw['count'].max()
            for t in taus:
                cls[t] += iou_from_keep(p >= t, tw['label'], tw['n_gt'])
                flat[t] += iou_from_keep(cf >= t, tw['label'], tw['n_gt'])
    n = len(towers)
    return {t: v / n for t, v in cls.items()}, {t: v / n for t, v in flat.items()}


# ─────────────────────────── Model C (image-conditioned) ───────────────────────────

def _collate_c(batch):
    M = max(len(b['label']) for b in batch)
    B = len(batch); F = batch[0]['feat'].shape[1]
    vf = torch.zeros(B, M, F); lab = torch.zeros(B, M)
    pad = torch.ones(B, M, dtype=torch.bool); img = torch.zeros(B, 4, 256, 1024)
    lens, ngt = [], []
    for i, b in enumerate(batch):
        m = len(b['label'])
        vf[i, :m] = torch.from_numpy(b['feat']); lab[i, :m] = torch.from_numpy(b['label'])
        pad[i, :m] = False; img[i] = torch.from_numpy(b['img'].astype(np.float32))
        lens.append(m); ngt.append(b['n_gt'])
    pos = vf[..., 8:11].clone()                         # normalised xyz columns
    return vf, pos, img, pad, lab, torch.tensor(lens), torch.tensor(ngt)


def _eval_c(towers, model, device, taus, batch_towers):
    from torch.utils.data import DataLoader
    model.eval()
    cls = {t: 0.0 for t in taus}
    loader = DataLoader(towers, batch_size=batch_towers, collate_fn=_collate_c)
    with torch.no_grad():
        for vf, pos, img, pad, lab, lens, ngt in loader:
            p = torch.sigmoid(model(vf.to(device), pos.to(device), img.to(device),
                                    pad.to(device))).cpu().numpy()
            for i in range(len(lens)):
                m = lens[i].item(); lb = lab[i, :m].numpy()
                for t in taus:
                    cls[t] += iou_from_keep(p[i, :m] >= t, lb, int(ngt[i]))
    n = len(towers)
    return {t: v / n for t, v in cls.items()}


def train_c(args, device, pool_dir, mean, std, train, val, taus, bestA, bestAt, is_main=True):
    from torch.utils.data import DataLoader
    from models.refinement import RefinementNetwork
    ddp = dist.is_available() and dist.is_initialized()
    feat_dir = args.feat_dir or pjoin(args.data_root, 'image_features_dino_patch')
    for tw in tqdm(train + val, desc='load DINOv2 feats', disable=not is_main):
        tw['img'] = np.load(pjoin(feat_dir, tw['name'] + '.npy')).astype(np.float16)
    val_sub = val[:args.val_limit] if args.val_limit else val
    save_dir = pjoin(CHECKPOINTS_DIR, DATASET_NAME, 'pool_classifier', args.name)
    if is_main:
        os.makedirs(save_dir, exist_ok=True)

    net = RefinementNetwork(N_FEAT, args.latent_dim, args.ff_size, args.c_layers,
                          args.n_heads, args.dropout).to(device)
    if is_main:
        print(f'RefinementNetwork: {sum(p.numel() for p in net.parameters())/1e6:.2f}M params  '
              f'(DDP={ddp})')
    model = DDP(net, device_ids=[device.index], broadcast_buffers=False) if ddp else net
    pos_w = torch.tensor([sum((t['label'] == 0).sum() for t in train) /
                          max(1, sum((t['label'] == 1).sum() for t in train))],
                         device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    sampler = DistributedSampler(train, shuffle=True) if ddp else None
    loader = DataLoader(train, batch_size=args.batch_towers, sampler=sampler,
                        shuffle=(sampler is None), collate_fn=_collate_c, drop_last=True,
                        num_workers=args.num_workers, pin_memory=True)
    best = -1.0
    for ep in range(args.epochs):
        if sampler is not None:
            sampler.set_epoch(ep)
        model.train(); tot = nb = 0.0
        for vf, pos, img, pad, lab, lens, ngt in loader:
            vf, pos, img, pad, lab = (vf.to(device), pos.to(device), img.to(device),
                                      pad.to(device), lab.to(device))
            opt.zero_grad(set_to_none=True)
            logits = model(vf, pos, img, pad)
            valid = ~pad
            if args.loss == 'focal':
                # hard-voxel focus: down-weight easy (well-classified) voxels
                bce_el = F.binary_cross_entropy_with_logits(
                    logits[valid], lab[valid], pos_weight=pos_w, reduction='none')
                pt = torch.exp(-bce_el)
                loss = ((1 - pt) ** args.focal_gamma * bce_el).mean()
            elif args.loss == 'bce_iou':
                # BCE anchor + per-tower soft-Jaccard with TRUE n_gt in the union
                # (= differentiable surrogate of the eval IoU; targets threshold-flippers).
                bce = lossf(logits[valid], lab[valid])
                p = torch.sigmoid(logits) * valid.float()           # (B,M), padded=0
                y = lab * valid.float()
                inter = (p * y).sum(1)                               # (B,)
                ngt_d = ngt.to(device).float()
                union = p.sum(1) + ngt_d - inter
                soft_iou = inter / (union + 1e-6)
                loss = bce + args.iou_weight * (1.0 - soft_iou).mean()
            else:
                loss = lossf(logits[valid], lab[valid])
            loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        if is_main:
            do_eval = ep % 2 == 0 or ep == args.epochs - 1
            if do_eval:
                cls = _eval_c(val_sub, net, device, taus, args.batch_towers)
                bt = max(cls, key=cls.get); bv = cls[bt]; tag = ''
                if bv > best:
                    best = bv
                    torch.save({'model': net.state_dict(), 'tau': bt, 'iou': bv,
                                'args': vars(args)}, pjoin(save_dir, 'best.tar'))
                    tag = ' *saved'
                print(f'ep {ep:3d}  loss {tot/nb:.4f}  C val IoU {bv:.4f} @τ={bt}  '
                      f'(A {bestA:.4f}@{bestAt})  Δ {bv-bestA:+.4f}{tag}', flush=True)
            else:
                print(f'ep {ep:3d}  loss {tot/nb:.4f}', flush=True)
        if ddp:
            dist.barrier()
    if is_main:
        cls = _eval_c(val, net, device, taus, args.batch_towers)
        bt = max(cls, key=cls.get)
        print(f'\nDONE (full val {len(val)})  baseline A {bestA:.4f}  |  Model C {cls[bt]:.4f} '
              f'@τ={bt}  |  Δ {cls[bt]-bestA:+.4f}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', default='poolcls_b_v1')
    ap.add_argument('--pool_dir', default=None,
                    help='default checkpoints/myarch/pool_classifier/pool_<mask>_N<N>')
    ap.add_argument('--mask_name', default='myarch_mask_img_patch_v8_2')
    ap.add_argument('--n_samples', type=int, default=16)
    ap.add_argument('--vq_name', default='ibq_4x_nc1024_nq1')
    ap.add_argument('--data_root', default=DATA_ROOT)
    ap.add_argument('--hidden', type=int, default=128)
    ap.add_argument('--layers', type=int, default=3)
    ap.add_argument('--dropout', type=float, default=0.1)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--batch_vox', type=int, default=262144, help='voxels per step (B)')
    ap.add_argument('--train_limit', type=int, default=0)
    # Model C (image-conditioned cross-attention)
    ap.add_argument('--use_image', action='store_true', help='train Model C, not Baseline B')
    ap.add_argument('--latent_dim', type=int, default=256)
    ap.add_argument('--ff_size', type=int, default=1024)
    ap.add_argument('--c_layers', type=int, default=6)
    ap.add_argument('--n_heads', type=int, default=8)
    ap.add_argument('--batch_towers', type=int, default=12, help='towers per step (C)')
    ap.add_argument('--val_limit', type=int, default=1500, help='val subset during C training')
    ap.add_argument('--feat_dir', default=None)
    ap.add_argument('--loss', default='bce', choices=['bce', 'bce_iou', 'focal'],
                    help='bce_iou = BCE + soft-Jaccard (metric-aligned, uses n_gt); '
                         'focal = hard-voxel-focused BCE')
    ap.add_argument('--iou_weight', type=float, default=1.0, help='weight on soft-IoU term (bce_iou)')
    ap.add_argument('--focal_gamma', type=float, default=2.0)
    ap.add_argument('--num_workers', type=int, default=4, help='DataLoader workers (C path)')
    ap.add_argument('--gpu_id', type=int, default=0)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    fixseed(args.seed)
    # DDP if launched via torchrun (LOCAL_RANK set); else single-GPU --gpu_id.
    if 'LOCAL_RANK' in os.environ:
        local_rank = int(os.environ['LOCAL_RANK'])
        dist.init_process_group(backend='nccl')
        device = torch.device(f'cuda:{local_rank}')
        torch.cuda.set_device(device)
        is_main = dist.get_rank() == 0
    else:
        device = torch.device(f'cuda:{args.gpu_id}')
        is_main = True

    pool_dir = args.pool_dir or pjoin(
        CHECKPOINTS_DIR, DATASET_NAME, 'pool_classifier',
        f'pool_{args.mask_name}_N{args.n_samples}')
    meta = pjoin(CHECKPOINTS_DIR, DATASET_NAME, args.vq_name, 'meta')
    mean = np.load(pjoin(meta, 'mean.npy')).reshape(-1)[:3]
    std = np.load(pjoin(meta, 'std.npy')).reshape(-1)[:3]
    save_dir = pjoin(CHECKPOINTS_DIR, DATASET_NAME, 'pool_classifier', args.name)
    if is_main:
        os.makedirs(save_dir, exist_ok=True)

    train = load_split(pool_dir, pjoin(args.data_root, 'train160k.txt'), args.n_samples,
                       mean, std, limit=args.train_limit)
    val = load_split(pool_dir, pjoin(args.data_root, 'val40k.txt'), args.n_samples, mean, std)
    taus = [round(t, 3) for t in np.arange(0.20, 0.85, 0.05)]

    # baseline A: best flat threshold on val (model-independent)
    flat0 = flat_baseline(val, taus)
    bestA = max(flat0.values()); bestAt = max(flat0, key=flat0.get)
    if is_main:
        print(f'train towers {len(train)}  val towers {len(val)}')
        print(f'baseline A (flat vote) best val IoU {bestA:.4f} @ τ={bestAt}')

    if args.use_image:                       # ---- Model C ----
        train_c(args, device, pool_dir, mean, std, train, val, taus, bestA, bestAt, is_main)
        if dist.is_initialized():
            dist.destroy_process_group()
        return

    # ---- Baseline B: per-voxel context MLP ----
    Xtr = torch.from_numpy(np.concatenate([t['feat'] for t in train])).to(device)
    Ytr = torch.from_numpy(np.concatenate([t['label'] for t in train])).to(device)
    pos_w = (Ytr == 0).sum() / (Ytr == 1).sum().clamp(min=1)
    print(f'train voxels {len(Xtr):,}  pos rate {Ytr.mean():.3f}  pos_weight {pos_w:.2f}')

    model = MLP(N_FEAT, args.hidden, args.layers, args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pos_w)

    n = len(Xtr); best = -1.0
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        tot = 0.0
        for i in range(0, n, args.batch_vox):
            idx = perm[i:i + args.batch_vox]
            opt.zero_grad(set_to_none=True)
            loss = lossf(model(Xtr[idx]), Ytr[idx])
            loss.backward(); opt.step()
            tot += loss.item() * len(idx)
        if ep % 2 == 0 or ep == args.epochs - 1:
            cls, _ = eval_towers(val, model, device, taus)
            bt = max(cls, key=cls.get); bv = cls[bt]
            tag = ''
            if bv > best:
                best = bv
                torch.save({'model': model.state_dict(), 'tau': bt, 'iou': bv,
                            'args': vars(args)}, pjoin(save_dir, 'best.tar'))
                tag = ' *saved'
            print(f'ep {ep:3d}  loss {tot/n:.4f}  cls val IoU {bv:.4f} @τ={bt}  '
                  f'(A {bestA:.4f}@{bestAt})  Δ {bv-bestA:+.4f}{tag}')

    print(f'\nDONE  baseline A {bestA:.4f}  |  classifier B best {best:.4f}  '
          f'|  Δ {best-bestA:+.4f}')


if __name__ == '__main__':
    main()
