"""
Two diagnostics sizing the "add back missing voxels" levers for pushing pool
classifier C past 0.992 (see project_pool_classifier_option2).

Per val tower: GT voxel set (from motion) vs the N=16 sample POOL (count>0 voxels
in the cached pool). "missing" = GT voxels the pool never proposed (zero-coverage).

(1) DILATION upside, bucket each missing voxel by 6-conn ring distance to the pool
    (1 = already in the current 1-ring dilation, 2 = a 2-ring would reach it, ...).
    % missing within k cells caps what a k-ring dilation + C can recover.

(2) SYMMETRY upside, for candidate tower symmetries (mirror-X, mirror-Y, 180° about
    Z, both diagonals) reflected about the GT bbox centre: (a) GT self-symmetry (how
    symmetric the tower is) and (b) % of missing voxels whose symmetric partner IS in
    the pool (recoverable by symmetry completion, the only frozen-feature route to
    occluded blocks). Picks the dominant symmetry per tower.

CPU-only, reads the existing pool cache. Usage:
    uv run python script/diag_missing_voxels.py [--limit N]
"""
import argparse
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from os.path import join as pjoin
import numpy as np
from tqdm import tqdm

from utils.voxel import coords_to_voxel_set, GRID
from eval_sampling import CHECKPOINTS_DIR, DATASET_NAME, DATA_ROOT

NEIGH6 = [(GRID[0], 0, 0), (-GRID[0], 0, 0), (0, GRID[1], 0),
          (0, -GRID[1], 0), (0, 0, GRID[2]), (0, 0, -GRID[2])]


def ring_distances(missing, pool, max_k):
    """Smallest 6-conn ring k (1..max_k) at which each missing voxel is reached by
    dilating the pool; returns dict k->count plus 'far' for beyond max_k."""
    out = {k: 0 for k in range(1, max_k + 1)}
    out['far'] = 0
    remaining = set(missing)
    reached = set(pool)
    frontier = set(pool)
    for k in range(1, max_k + 1):
        newf = set()
        for v in frontier:
            for d in NEIGH6:
                nb = (v[0] + d[0], v[1] + d[1], v[2] + d[2])
                if nb not in reached:
                    newf.add(nb)
        hit = remaining & newf
        out[k] = len(hit)
        remaining -= hit
        reached |= newf
        frontier = newf
        if not remaining:
            break
    out['far'] = len(remaining)
    return out


def sym_transforms(gt_set):
    """Reflections about the GT bounding-box centre (map valid grid cells->cells)."""
    xs = [v[0] for v in gt_set]; ys = [v[1] for v in gt_set]
    sx = min(xs) + max(xs); sy = min(ys) + max(ys)
    return {
        'mirrorX':  lambda v: (sx - v[0], v[1], v[2]),
        'mirrorY':  lambda v: (v[0], sy - v[1], v[2]),
        'rot180':   lambda v: (sx - v[0], sy - v[1], v[2]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mask_name', default='myarch_mask_img_patch_v8_2')
    ap.add_argument('--n_samples', type=int, default=16)
    ap.add_argument('--split', default='val40k')
    ap.add_argument('--data_root', default=DATA_ROOT)
    ap.add_argument('--max_k', type=int, default=4)
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()

    pool_dir = pjoin(CHECKPOINTS_DIR, DATASET_NAME, 'pool_classifier',
                     f'pool_{args.mask_name}_N{args.n_samples}')
    motion_dir = pjoin(args.data_root, 'new_joint_vecs_int')
    with open(pjoin(args.data_root, f'{args.split}.txt')) as f:
        names = sorted(l.strip() for l in f if l.strip())
    if args.limit:
        names = names[:args.limit]

    TRANSFORMS = ['mirrorX', 'mirrorY', 'rot180']
    tot_gt = tot_missing = 0
    ring = {k: 0 for k in range(1, args.max_k + 1)}; ring['far'] = 0
    # per-transform: GT self-symmetry, and missing-partner-in-pool (dominant + each)
    gt_sym = {t: 0 for t in TRANSFORMS}; gt_sym_den = 0
    miss_partner_pool = {t: 0 for t in TRANSFORMS}
    miss_partner_dom = 0          # under each tower's own dominant transform
    n = 0
    for nm in tqdm(names, desc='towers'):
        p = pjoin(pool_dir, nm + '.npz')
        mp = pjoin(motion_dir, nm + '.npy')
        if not (os.path.exists(p) and os.path.exists(mp)):
            continue
        d = np.load(p)
        vox, count = d['vox'].astype(np.int64), d['count']
        pool = {tuple(v) for v, c in zip(vox.tolist(), count.tolist()) if c > 0}
        motion = np.load(mp)
        n_cmp = (len(motion) // 4) * 4
        gt_set = coords_to_voxel_set(motion[:n_cmp])
        missing = gt_set - pool
        tot_gt += len(gt_set); tot_missing += len(missing)

        rd = ring_distances(missing, pool, args.max_k)
        for k in ring:
            ring[k] += rd[k]

        tf = sym_transforms(gt_set)
        gt_sym_den += len(gt_set)
        per_tf_gtsym = {}
        for t in TRANSFORMS:
            f = tf[t]
            sc = sum(1 for v in gt_set if f(v) in gt_set)
            gt_sym[t] += sc; per_tf_gtsym[t] = sc
            if missing:
                miss_partner_pool[t] += sum(1 for v in missing if f(v) in pool)
        # dominant transform for THIS tower = highest GT self-symmetry
        dom = max(TRANSFORMS, key=lambda t: per_tf_gtsym[t])
        fdom = tf[dom]
        miss_partner_dom += sum(1 for v in missing if fdom(v) in pool)
        n += 1

    W = 56
    print(f'\n{"="*W}\n  MISSING-VOXEL DIAGNOSTICS  ({n} {args.split} towers, N={args.n_samples})\n{"="*W}')
    print(f'  GT voxels total           {tot_gt:,}')
    print(f'  zero-coverage (missing)   {tot_missing:,}  = {tot_missing/tot_gt*100:.3f}% of GT')
    print(f'  pool recall (ceiling)     {1 - tot_missing/tot_gt:.4f}')
    print(f'\n  (1) DILATION, ring distance of MISSING voxels to the pool')
    print(f'      (% of the {tot_missing:,} missing; cumulative = what a k-ring reaches)')
    cum = 0
    for k in range(1, args.max_k + 1):
        cum += ring[k]
        print(f'      within {k} cell{"s" if k>1 else " "}   {ring[k]/tot_missing*100:5.1f}%   '
              f'(cumulative {cum/tot_missing*100:5.1f}%)')
    print(f'      beyond {args.max_k}      {ring["far"]/tot_missing*100:5.1f}%')
    print(f'\n  (2) SYMMETRY, about GT bbox centre')
    print(f'      {"transform":10} {"GT self-sym":>12} {"missing partner in pool":>26}')
    for t in TRANSFORMS:
        print(f'      {t:10} {gt_sym[t]/gt_sym_den*100:11.1f}% '
              f'{miss_partner_pool[t]/tot_missing*100:25.1f}%')
    print(f'      {"DOMINANT":10} {"(per-tower)":>12} {miss_partner_dom/tot_missing*100:25.1f}%')
    print('=' * W)
    print('  Reading: dilation cumulative% = max recoverable by k-ring+C;')
    print('  symmetry partner-in-pool% = recoverable by symmetry completion.')


if __name__ == '__main__':
    main()
