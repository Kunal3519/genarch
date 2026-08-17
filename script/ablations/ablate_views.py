"""View-dropping ablation for the patch MaskTransformer.

The cond is (B,4,256,1024) DINOv2 patch tokens, view order [NE,NW,SE,SW]. CFG-drop
zeros ALL views together at training, so the faithful "drop view" intervention is to
ZERO the dropped views' features (-> the learned bias-only / no-info representation),
keeping the 1024-token memory shape intact.

For each tower we build all 15 view-combinations (4×k1, 6×k2, 4×k3, 1×k4), zero the
complementary views, and run them as one batched generate() call. Single-pass per
combo (temp 0.2 by default = the compute_val_iou protocol) isolates conditioning
quality; --n_samples>1 + --vote adds voxel-voting per combo. Towers are the outer loop
so every combo is scored on the SAME towers (paired comparison).

Reports: mean IoU per combo grouped by #views, the single-view ranking, and the
drop-one (k=3) ranking → which view the model leans on most.

    CUDA_VISIBLE_DEVICES=4 uv run python \
        script/ablate_views.py --gpu_id 0 --ckpt latest.tar --n_towers 500 --temperature 0.2
"""
import argparse, os, sys, itertools
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))  # repo root
sys.path.insert(0, os.path.dirname(_HERE))                   # script/
from os.path import join as pjoin
from collections import Counter
import numpy as np
import torch
from tqdm import tqdm

from eval_sampling import (load_mask_model, load_records, CHECKPOINTS_DIR,
                            DATASET_NAME, DATA_ROOT,
                            MASK_TIMESTEPS, MASK_COND_SCALE, MASK_TOPK)
from utils.model_io import load_vq
from utils.vq_decode import decode_windowed, DEFAULT_WINDOW_TOKENS
from utils.voxel import voxel_iou, coords_to_voxel_set
from utils.fixseed import fixseed

VIEW_NAMES = ['NE', 'NW', 'SE', 'SW']


def set_iou(a, b):
    if not a and not b: return 1.0
    if not a or not b:  return 0.0
    return len(a & b) / len(a | b)


def build_combos():
    """ordered list of (label, kept_view_indices)"""
    combos = []
    for k in (1, 2, 3, 4):
        for c in itertools.combinations(range(4), k):
            label = '+'.join(VIEW_NAMES[i] for i in c)
            combos.append((label, list(c)))
    return combos


def stream_records(split_file, n_towers, seed, shard_idx=0, num_shards=1):
    """Lazily yield (name, motion, feat), sharded round-robin over the name list.

    n_towers caps the pool BEFORE sharding, so every shard sees a disjoint slice of
    the same deterministic tower set regardless of how many shards are used.
    """
    motion_dir = pjoin(DATA_ROOT, 'new_joint_vecs_int')
    feat_dir = pjoin(DATA_ROOT, 'image_features_dino_patch')
    names = sorted(l.strip() for l in open(split_file) if l.strip())
    rng = np.random.default_rng(seed)
    rng.shuffle(names)
    keep = []
    for nm in names:
        if os.path.exists(pjoin(motion_dir, nm + '.npy')) and \
           os.path.exists(pjoin(feat_dir, nm + '.npy')):
            keep.append(nm)
        if n_towers and len(keep) >= n_towers:
            break
    keep = keep[shard_idx::num_shards]

    class _Stream:
        def __len__(self):
            return len(keep)

        def __iter__(self):
            for nm in keep:
                yield (nm,
                       np.load(pjoin(motion_dir, nm + '.npy')),
                       np.load(pjoin(feat_dir, nm + '.npy')))
    return _Stream()


def report(scores, combos, args, n_towers):
    m = {lab: float(np.mean(scores[lab])) for lab, _ in combos}
    full = m['+'.join(VIEW_NAMES)]
    mode = (f'refined@tau{args.tau}' if args.refine else
            f'vote@{args.threshold}' if args.vote else 'single')

    print('\n' + '=' * 64)
    print(f'  View-dropping ablation  {args.mask_name}/{args.ckpt}')
    print(f'  {mode} (N={args.n_samples})  temp={args.temperature}  n_towers={n_towers}')
    print('=' * 64)
    print(f"  {'#views':>6}  {'combo':<16} {'mean IoU':>9} {'Δ vs all-4':>11}")
    print('  ' + '-' * 54)
    for k in (4, 3, 2, 1):
        grp = sorted([(lab, c) for lab, c in combos if len(c) == k], key=lambda x: -m[x[0]])
        for lab, c in grp:
            print(f"  {k:>6}  {lab:<16} {m[lab]:>9.4f} {m[lab]-full:>+11.4f}")
        print('  ' + '-' * 54)

    print('\n  Mean IoU by #views (averaged over every subset of that size):')
    for k in (1, 2, 3, 4):
        vals = [m[lab] for lab, c in combos if len(c) == k]
        print(f'    {k} view(s): {np.mean(vals):.4f}   (best {max(vals):.4f}, worst {min(vals):.4f})')

    adj = [m[lab] for lab, c in combos if len(c) == 2 and set(c) not in ({0, 3}, {1, 2})]
    dia = [m[lab] for lab, c in combos if len(c) == 2 and set(c) in ({0, 3}, {1, 2})]
    print(f'\n  2-view adjacent {np.mean(adj):.4f}  vs  diagonal {np.mean(dia):.4f}'
          f'  (Δ {np.mean(adj)-np.mean(dia):+.4f})')

    print('\n  Per-view importance (mean IoU with view − without, across all combos):')
    rank = []
    for vi, vn in enumerate(VIEW_NAMES):
        with_v = np.mean([m[lab] for lab, c in combos if vi in c])
        wo_v = np.mean([m[lab] for lab, c in combos if vi not in c])
        rank.append((vn, with_v - wo_v, m[VIEW_NAMES[vi]]))
    for vn, d, solo in sorted(rank, key=lambda x: -x[1]):
        print(f'    {vn}: +{d:.4f} marginal   (solo-view IoU {solo:.4f})')
    print('=' * 64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mask_name', default='myarch_mask_img_patch_v7')
    ap.add_argument('--ckpt', default='latest.tar')
    ap.add_argument('--vq_name', default='rvq_v2_4x_nc1024_nq3')
    ap.add_argument('--split', default=pjoin(DATA_ROOT, 'val40k.txt'))
    ap.add_argument('--n_towers', type=int, default=500)
    ap.add_argument('--n_samples', type=int, default=1, help='draws per combo per tower')
    ap.add_argument('--vote', action='store_true', help='report voxel-vote@thr instead of single')
    ap.add_argument('--refine', action='store_true',
                    help='report the DEPLOYED pipeline: N samples -> pool -> refinement model')
    ap.add_argument('--cls_name', default='poolcls_c_sym_v1')
    ap.add_argument('--tau', type=float, default=0.45)
    ap.add_argument('--threshold', type=float, default=0.3)
    ap.add_argument('--temperature', type=float, default=0.2)
    ap.add_argument('--decode_mode', choices=['windowed', 'whole'], default='windowed',
                    help='token decode protocol (whole for IBQ)')
    ap.add_argument('--gpu_id', type=int, default=0)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--shard_idx', type=int, default=0)
    ap.add_argument('--num_shards', type=int, default=1)
    ap.add_argument('--out', default='', help='save per-tower per-combo IoUs to this .npz')
    ap.add_argument('--combine', nargs='*', default=None,
                    help='merge shard .npz files and print the table (no GPU work)')
    args = ap.parse_args()

    if args.combine is not None:
        combos = build_combos()
        scores = {lab: [] for lab, _ in combos}
        n = 0
        for p in args.combine:
            d = np.load(p, allow_pickle=True)
            for lab in scores:
                scores[lab].append(d[lab])
            n += len(d[combos[0][0]])
        scores = {lab: np.concatenate(v) for lab, v in scores.items()}
        report(scores, combos, args, n)
        return

    fixseed(args.seed)
    device = torch.device(f'cuda:{args.gpu_id}')
    vq_model, vq_opt = load_vq(CHECKPOINTS_DIR, args.vq_name, device=device,
                                dim_pose=3, dataset_name=DATASET_NAME)
    vq_model.eval().to(device)
    nb_code = vq_model.quantizer.layers[0].nb_code
    meta = pjoin(CHECKPOINTS_DIR, DATASET_NAME, args.vq_name, 'meta')
    mean = np.load(pjoin(meta, 'mean.npy')); std = np.load(pjoin(meta, 'std.npy'))
    model = load_mask_model(vq_opt, args.mask_name, args.ckpt, device)

    refine = None
    if args.refine:
        from script.sweep_sampling_knobs import load_refiner
        refine = load_refiner(args.cls_name, device)
    mean3 = mean.reshape(-1)[:3]; std3 = std.reshape(-1)[:3]

    # Stream rather than preload: each DINOv2 feature is (4,256,1024) fp32 = 4 MB, so
    # load_records() would hold ~160 GB at n_towers=40000, per process.
    records = stream_records(args.split, args.n_towers, args.seed,
                             args.shard_idx, args.num_shards)
    combos = build_combos()
    C, S = len(combos), args.n_samples
    if args.refine:
        mode = f'refined@tau{args.tau} (N={S})'
    elif args.vote:
        mode = f'vote@{args.threshold} (N={S})'
    else:
        mode = f'single (N={S})'
    print(f'{len(records)} towers × {C} combos × {S} draws  |  {mode}  temp={args.temperature}\n')

    scores = {lab: [] for lab, _ in combos}     # per-tower IoU per combo

    @torch.no_grad()
    def run_tower(motion, feat):
        n_cmp = (len(motion) // 4) * 4
        gt_xyz = motion[:n_cmp]; gt_set = coords_to_voxel_set(gt_xyz)
        m_tok = len(motion) // 4
        # build (C*S, 4,256,1024): each combo's masked cond, replicated S times
        base = torch.from_numpy(feat).to(device).float()              # (4,256,1024)
        conds = []
        for _, kept in combos:
            masked = torch.zeros_like(base)
            masked[kept] = base[kept]
            conds.append(masked)
        cond = torch.stack(conds).repeat_interleave(S, dim=0)          # (C*S,4,256,1024)
        m_lens = torch.full((C * S,), m_tok, device=device, dtype=torch.long)
        ids = model.generate(cond, m_lens, timesteps=MASK_TIMESTEPS,
                             cond_scale=MASK_COND_SCALE, temperature=args.temperature,
                             topk_filter_thres=MASK_TOPK, gsample=False).clamp(0, nb_code - 1)
        if args.decode_mode == 'whole':
            coords = vq_model.forward_decoder(ids.unsqueeze(-1)).cpu().numpy()
        else:
            coords = decode_windowed(vq_model, ids.unsqueeze(-1),
                                     window_tokens=DEFAULT_WINDOW_TOKENS).cpu().numpy()
        out = {}
        for ci, (lab, _) in enumerate(combos):
            xyz = [coords[ci * S + s, :n_cmp] * std + mean for s in range(S)]
            if refine is not None:
                # the masked cond for THIS combo is what the refiner must also see
                masked_feat = np.zeros_like(feat)
                masked_feat[combos[ci][1]] = feat[combos[ci][1]]
                out[lab] = set_iou(refine(xyz, gt_set, masked_feat, S,
                                          mean3, std3, args.tau), gt_set)
            elif args.vote:
                cnt = Counter()
                for x in xyz: cnt.update(coords_to_voxel_set(x))
                voted = {v for v, c in cnt.items() if c >= args.threshold * S}
                out[lab] = set_iou(voted, gt_set)
            else:
                out[lab] = float(np.mean([voxel_iou(x, gt_xyz) for x in xyz]))
        return out

    done = 0
    for name, motion, feat in tqdm(records, desc='towers'):
        for lab, v in run_tower(motion, feat).items():
            scores[lab].append(v)
        done += 1
        if args.out and done % 250 == 0:                 # checkpoint long shards
            np.savez(args.out, **{lab: np.asarray(v) for lab, v in scores.items()})

    if args.out:
        np.savez(args.out, **{lab: np.asarray(v) for lab, v in scores.items()})
        print(f'saved {done} towers -> {args.out}')
    report(scores, combos, args, done)


if __name__ == '__main__':
    main()
