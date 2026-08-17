"""
Full-val eval of an IBQ/RVQ tokenizer under different decode window sizes.

Loads the VQ once and sweeps decode windows (e.g. full / 32 / 64 / 128). For
each window W: pad each tower to a multiple of W (edge pad), encode+decode in
W-block chunks (win-enc + win-dec), concat, trim, denorm, voxel-IoU vs GT.
"full" = whole-sequence encode+decode (pad only to a multiple of the VQ
downsample factor, single chunk).

Usage:
    uv run python script/eval_ibq_windows.py \
        --vq_name ibq_4x_nc1024_nq1 --ckpt_name net_best_iou.tar \
        --data_root $GENARCH_DATA_ROOT --windows full 32 64 128
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from utils.paths import DATA_ROOT

import argparse
import numpy as np
import torch
from os.path import join as pjoin
from tqdm import tqdm

from utils.model_io import load_vq
from utils.voxel import voxel_iou


@torch.no_grad()
def eval_window(names, vq_model, motion_dir, mean, std, device, window, ds_factor):
    """window: int W for windowed decode, or 'full' for whole-seq. Returns IoU array."""
    ious = []
    for name in names:
        path = pjoin(motion_dir, name + '.npy')
        if not os.path.exists(path):
            continue
        gt = np.load(path).astype(np.float32)
        L = len(gt)
        norm = (gt - mean) / std

        if window == 'full':
            step = max(ds_factor, int(np.ceil(L / ds_factor)) * ds_factor)
        else:
            step = int(window)
        pad = (step - L % step) % step
        if L < step:
            pad = step - L
        if pad > 0:
            norm = np.pad(norm, ((0, pad), (0, 0)), mode='edge')

        chunks = []
        for start in range(0, norm.shape[0], step):
            chunk = torch.from_numpy(norm[start:start + step]).unsqueeze(0).to(device)
            code_idx, _ = vq_model.encode(chunk)
            chunks.append(vq_model.forward_decoder(code_idx).squeeze(0).cpu().numpy())
        pred = np.concatenate(chunks, axis=0)[:L] * std + mean
        ious.append(voxel_iou(pred, gt))
    return np.array(ious)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--vq_name', type=str, default='ibq_4x_nc1024_nq1')
    ap.add_argument('--ckpt_name', type=str, default='net_best_iou.tar')
    ap.add_argument('--data_root', type=str, default=DATA_ROOT)
    ap.add_argument('--split', type=str, default='val40k')
    ap.add_argument('--windows', type=str, nargs='+', default=['full', '32', '64', '128'])
    ap.add_argument('--gpu_id', type=int, default=0)
    ap.add_argument('--limit', type=int, default=0, help='cap #towers (0 = all)')
    opt = ap.parse_args()

    device = torch.device(f'cuda:{opt.gpu_id}' if torch.cuda.is_available() else 'cpu')
    vq_model, vq_opt = load_vq('./checkpoints', opt.vq_name, device=device,
                                dim_pose=3, dataset_name='myarch', ckpt_name=opt.ckpt_name)
    vq_model.eval().to(device)
    ds_factor = vq_opt.stride_t ** vq_opt.down_t

    meta_dir = pjoin('./checkpoints/myarch', opt.vq_name, 'meta')
    mean = np.load(pjoin(meta_dir, 'mean.npy'))
    std = np.load(pjoin(meta_dir, 'std.npy'))

    motion_dir = pjoin(opt.data_root, 'new_joint_vecs_int')
    with open(pjoin(opt.data_root, f'{opt.split}.txt')) as f:
        names = [l.strip() for l in f if l.strip()]
    if opt.limit > 0:
        names = names[:opt.limit]

    print(f'\n  VQ={opt.vq_name} ckpt={opt.ckpt_name}  ds={ds_factor}x  '
          f'train_window={vq_opt.window_size}  num_q={vq_opt.num_quantizers}')
    print(f'  split={opt.split}  towers={len(names)}  data_root={opt.data_root}\n')

    print(f'  {"window":>8} | {"mean IoU":>9} | {"median":>8} | {">=0.99":>7} | {">=0.97":>7}')
    print(f'  {"-"*8}-+-{"-"*9}-+-{"-"*8}-+-{"-"*7}-+-{"-"*7}')
    for w in tqdm(opt.windows, desc='windows', leave=False):
        ious = eval_window(names, vq_model, motion_dir, mean, std, device, w, ds_factor)
        n = len(ious)
        pct99 = 100 * np.mean(ious >= 0.99)
        pct97 = 100 * np.mean(ious >= 0.97)
        tqdm.write(f'  {w:>8} | {ious.mean():9.4f} | {np.median(ious):8.4f} | '
                   f'{pct99:6.1f}% | {pct97:6.1f}%')


if __name__ == '__main__':
    main()
