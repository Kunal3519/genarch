"""
Full-split eval of a trained pool classifier (B or C) — reuses train_refinement
functions so the feature build / IoU exactly match training. Reports best-tau Model
IoU vs baseline A (flat vote) and the reference old-C 0.9920.

Usage:
    CUDA_VISIBLE_DEVICES=<uuid> uv run python \
        script/eval_refinement.py --name poolcls_c_sym_v1 --ckpt best.tar
"""
import argparse
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from os.path import join as pjoin
import numpy as np
import torch
from tqdm import tqdm

from models.refinement import RefinementNetwork
from train_refinement import (load_split, _eval_c, flat_baseline, N_FEAT,
                                    CHECKPOINTS_DIR, DATASET_NAME, DATA_ROOT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', default='poolcls_c_sym_v1')
    ap.add_argument('--ckpt', default='best.tar')
    ap.add_argument('--mask_name', default='myarch_mask_img_patch_v8_2')
    ap.add_argument('--n_samples', type=int, default=16)
    ap.add_argument('--vq_name', default='ibq_4x_nc1024_nq1')
    ap.add_argument('--data_root', default=DATA_ROOT)
    ap.add_argument('--split', default='val40k')
    ap.add_argument('--batch_towers', type=int, default=24)
    ap.add_argument('--gpu_id', type=int, default=0)
    args = ap.parse_args()
    device = torch.device(f'cuda:{args.gpu_id}'); torch.cuda.set_device(device)

    pool_dir = pjoin(CHECKPOINTS_DIR, DATASET_NAME, 'pool_classifier',
                     f'pool_{args.mask_name}_N{args.n_samples}')
    meta = pjoin(CHECKPOINTS_DIR, DATASET_NAME, args.vq_name, 'meta')
    mean = np.load(pjoin(meta, 'mean.npy')).reshape(-1)[:3]
    std = np.load(pjoin(meta, 'std.npy')).reshape(-1)[:3]

    val = load_split(pool_dir, pjoin(args.data_root, f'{args.split}.txt'), args.n_samples, mean, std)
    feat_dir = pjoin(args.data_root, 'image_features_dino_patch')
    for tw in tqdm(val, desc='load DINOv2 feats'):
        tw['img'] = np.load(pjoin(feat_dir, tw['name'] + '.npy')).astype(np.float16)

    taus = [round(t, 3) for t in np.arange(0.20, 0.85, 0.05)]
    bestA = max(flat_baseline(val, taus).values())

    ckpt = pjoin(CHECKPOINTS_DIR, DATASET_NAME, 'pool_classifier', args.name, args.ckpt)
    ck = torch.load(ckpt, map_location='cpu', weights_only=False)
    ca = ck.get('args', {})
    m = RefinementNetwork(N_FEAT, ca.get('latent_dim', 256), ca.get('ff_size', 1024),
                        ca.get('c_layers', 6), ca.get('n_heads', 8), ca.get('dropout', 0.1)).to(device)
    m.load_state_dict(ck['model']); m.eval()
    cls = _eval_c(val, m, device, taus, args.batch_towers)
    bt = max(cls, key=cls.get)

    print(f'\n{"="*56}')
    print(f'  {args.name}/{args.ckpt}  full {args.split} ({len(val)} towers, N_FEAT={N_FEAT})')
    print(f'  Model C  {cls[bt]:.4f} @τ={bt}   |  baseline A {bestA:.4f}   |  old C 0.9920')
    print(f'  Δ vs A {cls[bt]-bestA:+.4f}   Δ vs old C {cls[bt]-0.9920:+.4f}')
    print(f'{"="*56}')
    for t in taus:
        print(f'  τ={t}: {cls[t]:.4f}')


if __name__ == '__main__':
    main()
