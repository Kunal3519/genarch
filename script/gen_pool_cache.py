"""
Cache the N-sample candidate POOL per tower for the voxel add/delete classifier
(option 2, a learned replacement for the vote threshold).

For each tower in --split, draw --n_samples stochastic samples from the frozen
mask transformer (deployment recipe: temp 0.8, whole-seq IBQ decode), snap every
sample to the grid, and accumulate per-voxel VOTE COUNTS over the pool. Optionally
dilate the pool by a 6-connected 1-ring so the classifier can ADD blocks no sample
proposed (these candidates carry count=0). Each pool voxel is labelled by GT
membership. One .npz per tower in --out_dir:
    vox   (M,3) int16   candidate voxel coords (pool ∪ optional dilation)
    count (M,)  uint8   vote count over the N samples (0 = dilation-added)
    label (M,)  uint8   1 if the voxel is in the GT set
    n_gt        int     |GT voxel set|  (GT voxels absent from the pool = FN floor)
    single_iou / vote_iou  float  draft-0 and best-flat-threshold IoU, for sanity

IoU of any kept subset S ⊆ candidates is recoverable from (label, n_gt) alone:
    inter = #{i in S : label=1};  union = |S| + n_gt - inter  → no GT array needed.

Resumable + sharded.

Usage (sharded over 1 GPU):
    CUDA_VISIBLE_DEVICES=<uuid> uv run python \
        script/gen_pool_cache.py --split val --gpu_id 0
    ... --split train --shard_idx 0 --num_shards 3 ...
"""
import argparse
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from os.path import join as pjoin
from collections import Counter
import numpy as np
import torch
from tqdm import tqdm

from utils.model_io import load_vq
from utils.voxel import coords_to_voxel_set, GRID
from utils.fixseed import fixseed
from eval_sampling import (CHECKPOINTS_DIR, DATASET_NAME, DATA_ROOT,
                           load_mask_model, _resolve_split, eval_tower)

# 6-connected face neighbours in grid units (one cell along each axis).
NEIGH6 = [(GRID[0], 0, 0), (-GRID[0], 0, 0),
          (0, GRID[1], 0), (0, -GRID[1], 0),
          (0, 0, GRID[2]), (0, 0, -GRID[2])]


def build_pool(sample_xyz, gt_set, dilate):
    """sample_xyz: list of (n,3) coord arrays → (vox, count, label, n_gt)."""
    cnt = Counter()
    for xyz in sample_xyz:
        cnt.update(coords_to_voxel_set(xyz))
    if dilate:
        for v in list(cnt.keys()):
            for d in NEIGH6:
                nb = (v[0] + d[0], v[1] + d[1], v[2] + d[2])
                if nb not in cnt:
                    cnt[nb] = 0          # candidate with zero votes
    vox = np.array(list(cnt.keys()), dtype=np.int16)
    count = np.array([cnt[tuple(v)] for v in vox.tolist()], dtype=np.uint8)
    label = np.array([1 if tuple(v) in gt_set else 0 for v in vox.tolist()],
                     dtype=np.uint8)
    return vox, count, label, len(gt_set)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mask_name', default='myarch_mask_img_patch_v8_2')
    ap.add_argument('--ckpt', default='net_best_iou.tar')
    ap.add_argument('--vq_name', default='ibq_4x_nc1024_nq1')
    ap.add_argument('--data_root', default=DATA_ROOT)
    ap.add_argument('--split', default='val40k', help='bare name (train160k/val40k) or a .txt path')
    ap.add_argument('--n_samples', type=int, default=16)
    ap.add_argument('--temperature', type=float, default=0.8)
    ap.add_argument('--decode_mode', default='whole', choices=['whole', 'windowed'])
    ap.add_argument('--threshold', type=float, default=0.4375, help='flat-vote ref only')
    ap.add_argument('--dilate', type=int, default=1, help='1 = add 6-conn 1-ring candidates')
    ap.add_argument('--out_dir', default=None,
                    help='default checkpoints/myarch/pool_classifier/pool_<mask_name>_N<N>')
    ap.add_argument('--shard_idx', type=int, default=0)
    ap.add_argument('--num_shards', type=int, default=1)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--gpu_id', type=int, default=0)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    fixseed(args.seed + args.shard_idx)
    device = torch.device(f'cuda:{args.gpu_id}')
    out_dir = args.out_dir or pjoin(
        CHECKPOINTS_DIR, DATASET_NAME, 'pool_classifier',
        f'pool_{args.mask_name}_N{args.n_samples}')
    os.makedirs(out_dir, exist_ok=True)

    vq_model, vq_opt = load_vq(CHECKPOINTS_DIR, args.vq_name, device=device,
                                dim_pose=3, dataset_name=DATASET_NAME)
    vq_model.eval().to(device)
    nb_code = vq_model.quantizer.layers[0].nb_code
    meta = pjoin(CHECKPOINTS_DIR, DATASET_NAME, args.vq_name, 'meta')
    mean = np.load(pjoin(meta, 'mean.npy'))
    std = np.load(pjoin(meta, 'std.npy'))
    model = load_mask_model(vq_opt, args.mask_name, args.ckpt, device)

    with open(_resolve_split(args.split, args.data_root)) as f:
        names = sorted(l.strip() for l in f if l.strip())
    names = names[args.shard_idx::args.num_shards]
    if args.limit:
        names = names[:args.limit]
    motion_dir = pjoin(args.data_root, 'new_joint_vecs_int')
    feat_dir = pjoin(args.data_root, 'image_features_dino_patch')
    thr_cnt = args.threshold * args.n_samples
    print(f'shard {args.shard_idx}/{args.num_shards}: {len(names)} towers -> {out_dir}  '
          f'N={args.n_samples} t={args.temperature} dilate={args.dilate}')

    single_s = vote_s = recall_s = ndone = nskip = 0
    for nm in tqdm(names, desc=f's{args.shard_idx}'):
        out_path = pjoin(out_dir, nm + '.npz')
        if os.path.exists(out_path):
            nskip += 1
            continue
        motion = np.load(pjoin(motion_dir, nm + '.npy'))
        feat = np.load(pjoin(feat_dir, nm + '.npy'))
        ious_gt, sample_xyz = eval_tower(model, vq_model, mean, std, motion, feat,
                                         args.n_samples, args.temperature, device,
                                         nb_code, decode_mode=args.decode_mode)
        n_cmp = (len(motion) // 4) * 4
        gt_set = coords_to_voxel_set(motion[:n_cmp])
        vox, count, label, n_gt = build_pool(sample_xyz, gt_set, args.dilate)
        np.savez(out_path, vox=vox, count=count, label=label, n_gt=n_gt,
                 single_iou=np.float32(ious_gt[0]))
        # running sanity stats (flat vote@thr + pool recall)
        kept = count >= thr_cnt
        inter = int(label[kept].sum()); union = int(kept.sum()) + n_gt - inter
        vote_s += inter / union if union else 1.0
        recall_s += label.sum() / n_gt if n_gt else 1.0   # |cand ∩ GT|/|GT| = ceiling
        single_s += float(ious_gt[0]); ndone += 1

    if ndone:
        print(f'done: {ndone} towers ({nskip} cached)')
        print(f'  single (draft0)          {single_s/ndone:.4f}')
        print(f'  flat vote@{args.threshold}            {vote_s/ndone:.4f}')
        print(f'  pool recall (cls ceiling) {recall_s/ndone:.4f}')
    else:
        print(f'nothing to do ({nskip} cached)')


if __name__ == '__main__':
    main()
