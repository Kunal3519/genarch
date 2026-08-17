"""
For each tower, generate under ALL 15 non-empty view-subsets (zeroing dropped
views) and measure whether the BEST subset beats all-4, i.e. does the
"toxic-view" effect found on tower_31734 generalize across the hard tail?

Also tests a NO-GT consensus selector: pick the subset whose N samples agree
most internally (mean pairwise voxel IoU) and check if that blind choice
recovers the oracle gain, the deployability question.

Per tower (whole-seq IBQ decode):
  - 15 subsets x single-pass (temp 0.2)        -> IoU vs GT per subset
  - 15 subsets x K samples (temp 0.8)          -> internal agreement per subset
Reports, over the tower list: all-4 / best-2view / best-3view / oracle-any-subset
single IoU; how many towers the best subset beats all-4 by >thr; and the
consensus-selector IoU vs oracle vs all-4.

Usage:
    CUDA_VISIBLE_DEVICES=<uuid> uv run python \
        script/sweep_view_subsets.py --tower_json /tmp/hard_towers.json --gpu_id 0 \
        --shard_idx 0 --num_shards 1 --tag hard
    # random baseline:
    ... --split val --limit 150 --tag randval
"""
import argparse, os, sys, json, itertools
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))  # repo root
sys.path.insert(0, os.path.dirname(_HERE))                   # script/
from os.path import join as pjoin
from collections import Counter
import numpy as np
import torch
from tqdm import tqdm

from utils.model_io import load_vq
from utils.voxel import voxel_iou, coords_to_voxel_set
from utils.fixseed import fixseed
from eval_sampling import (CHECKPOINTS_DIR, DATASET_NAME, DATA_ROOT,
                           load_mask_model, _resolve_split, set_iou,
                           MASK_TIMESTEPS, MASK_COND_SCALE, MASK_TOPK)

VIEWS = ['NE', 'NW', 'SE', 'SW']
SUBSETS = [s for r in range(1, 5) for s in itertools.combinations(range(4), r)]  # 15
SUBLABEL = ['+'.join(VIEWS[i] for i in s) for s in SUBSETS]
ALL4 = SUBSETS.index((0, 1, 2, 3))


@torch.no_grad()
def gen_batch(gmm, vq, feats, n_tok, n_cmp, temp, mean, std, nb, device):
    """feats: (B,4,256,1024) np -> list of B xyz arrays (whole-seq decode)."""
    cond = torch.from_numpy(feats).to(device).float()
    lens = torch.full((len(feats),), n_tok, device=device, dtype=torch.long)
    ids = gmm.generate(cond, lens, timesteps=MASK_TIMESTEPS, cond_scale=MASK_COND_SCALE,
                       temperature=temp, topk_filter_thres=MASK_TOPK, gsample=False).clamp(0, nb - 1)
    co = vq.forward_decoder(ids.unsqueeze(-1)).cpu().numpy()
    return [co[i, :n_cmp] * std + mean for i in range(len(feats))]


def masked(feat0, sub):
    f = feat0.copy()
    for v in range(4):
        if v not in sub:
            f[v] = 0.0
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tower_json', default=None)
    ap.add_argument('--split', default='val40k')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--tag', default='hard')
    ap.add_argument('--mask_name', default='myarch_mask_img_patch_v8_2')
    ap.add_argument('--mask_ckpt', default='net_best_iou.tar')
    ap.add_argument('--vq_name', default='ibq_4x_nc1024_nq1')
    ap.add_argument('--data_root', default=DATA_ROOT)
    ap.add_argument('--k_agree', type=int, default=6, help='samples/subset for consensus')
    ap.add_argument('--shard_idx', type=int, default=0)
    ap.add_argument('--num_shards', type=int, default=1)
    ap.add_argument('--combine', action='store_true')
    ap.add_argument('--gpu_id', type=int, default=0)
    ap.add_argument('--seed', type=int, default=42)
    a = ap.parse_args()
    tag = f'viewsweep_{a.tag}'
    if a.combine:
        combine(a, tag)
        return
    fixseed(a.seed + a.shard_idx)
    device = torch.device(f'cuda:{a.gpu_id}'); torch.cuda.set_device(device)

    vq, vqo = load_vq(CHECKPOINTS_DIR, a.vq_name, device=device, dim_pose=3, dataset_name=DATASET_NAME)
    vq.eval().to(device); nb = vq.quantizer.layers[0].nb_code
    meta = pjoin(CHECKPOINTS_DIR, DATASET_NAME, a.vq_name, 'meta')
    mean = np.load(pjoin(meta, 'mean.npy')).reshape(-1)[:3]
    std = np.load(pjoin(meta, 'std.npy')).reshape(-1)[:3]
    gmm = load_mask_model(vqo, a.mask_name, a.mask_ckpt, device)

    motion_dir = pjoin(a.data_root, 'new_joint_vecs_int')
    feat_dir = pjoin(a.data_root, 'image_features_dino_patch')
    if a.tower_json:
        names = json.load(open(a.tower_json))
    else:
        with open(_resolve_split(a.split, a.data_root)) as f:
            names = sorted(l.strip() for l in f if l.strip())
    names = [n for n in names if os.path.exists(pjoin(feat_dir, n + '.npy'))]
    names = names[a.shard_idx::a.num_shards]
    if a.limit:
        names = names[:a.limit]
    print(f'{tag} shard {a.shard_idx}/{a.num_shards}: {len(names)} towers, {len(SUBSETS)} subsets')

    rows = {}   # name -> dict(single=[15], agree=[15])
    for nm in tqdm(names, leave=False):
        motion = np.load(pjoin(motion_dir, nm + '.npy'))
        feat0 = np.load(pjoin(feat_dir, nm + '.npy')).astype(np.float32)
        n_tok = len(motion) // 4; n_cmp = n_tok * 4
        gt = motion[:n_cmp]; gt_set = coords_to_voxel_set(gt)
        feats = np.stack([masked(feat0, s) for s in SUBSETS])              # (15,4,256,1024)
        # single-pass IoU per subset
        xs = gen_batch(gmm, vq, feats, n_tok, n_cmp, 0.2, mean, std, nb, device)
        single = np.array([voxel_iou(x, gt) for x in xs])
        # consensus: K temp-0.8 draws per subset -> mean pairwise IoU
        big = np.repeat(feats, a.k_agree, axis=0)                          # (15K,...)
        xb = gen_batch(gmm, vq, big, n_tok, n_cmp, 0.8, mean, std, nb, device)
        sets = [coords_to_voxel_set(x) for x in xb]
        agree = np.zeros(len(SUBSETS))
        for si in range(len(SUBSETS)):
            grp = sets[si * a.k_agree:(si + 1) * a.k_agree]
            ious = [set_iou(grp[i], grp[j]) for i in range(len(grp)) for j in range(i + 1, len(grp))]
            agree[si] = np.mean(ious) if ious else 1.0
        rows[nm] = {'single': single, 'agree': agree}
    out = f'/tmp/{tag}_shard{a.shard_idx}of{a.num_shards}.npy'
    np.save(out, np.array(rows, dtype=object))
    print(f'-> {out}  ({len(rows)} towers)')


def combine(a, tag):
    rows = {}
    for s in range(a.num_shards):
        p = f'/tmp/{tag}_shard{s}of{a.num_shards}.npy'
        if not os.path.exists(p):
            print('MISSING', p); return
        rows.update(np.load(p, allow_pickle=True).item())
    names = list(rows)
    S = np.array([rows[n]['single'] for n in names])     # (T,15)
    A = np.array([rows[n]['agree'] for n in names])       # (T,15)
    T = len(names)
    sizes = np.array([len(s) for s in SUBSETS])
    all4 = S[:, ALL4]
    best_any = S.max(1)
    best2 = S[:, sizes == 2].max(1)
    best3 = S[:, sizes == 3].max(1)
    print(f'\n{"="*64}\n  view-subset sweep, {tag}  ({T} towers, single-pass t0.2)\n{"="*64}')
    print(f'  all-4 (current):        {all4.mean():.4f}')
    print(f'  best 2-view subset:     {best2.mean():.4f}   (Δ {best2.mean()-all4.mean():+.4f})')
    print(f'  best 3-view subset:     {best3.mean():.4f}   (Δ {best3.mean()-all4.mean():+.4f})')
    print(f'  ORACLE best any subset: {best_any.mean():.4f}   (Δ {best_any.mean()-all4.mean():+.4f})')
    for thr in [0.005, 0.02, 0.05, 0.10]:
        print(f'    towers where best-subset beats all-4 by >{thr:.3f}: '
              f'{(best_any-all4>thr).sum():4d} ({100*(best_any-all4>thr).mean():4.1f}%)')
    # which subset is the oracle pick most often (for size<=3)
    cand = [i for i in range(len(SUBSETS)) if sizes[i] <= 3]
    oc = Counter(SUBLABEL[cand[np.argmax(S[t, cand])]] for t in range(T))
    print(f'\n  oracle best subset (size≤3) frequency: '
          + '  '.join(f'{k}:{v}' for k, v in oc.most_common(8)))
    # NO-GT consensus selector: pick highest-agreement subset among size>=2
    cand2 = np.array([i for i in range(len(SUBSETS)) if sizes[i] >= 2])
    sel = cand2[np.argmax(A[:, cand2], 1)]
    sel_iou = S[np.arange(T), sel]
    print(f'\n  -- NO-GT consensus selector (highest internal agreement, size≥2) --')
    print(f'  selector IoU:           {sel_iou.mean():.4f}   (Δ vs all-4 {sel_iou.mean()-all4.mean():+.4f}, '
          f'vs oracle {sel_iou.mean()-best_any.mean():+.4f})')
    print(f'  selector picked all-4:  {100*(sel==ALL4).mean():.1f}% of towers')
    print(f'{"="*64}')


if __name__ == '__main__':
    main()
