"""
Smoke-test MaskTransformer.edit() on the deployed 1D pipeline (v8.2 + IBQ).

Three modes, all measured as voxel IoU vs GT:
  keep_prefix  - fix the first f% of tokens (GT), resample the rest  = COMPLETION
                 (a partial tower / floorplan prefix grows into a full tower)
  hole         - fix everything except a middle window                = INPAINTING
  mask_free    - edit_mask=None: gentle whole-sequence re-decode      = REFINEMENT

A useful edit keeps the fixed region EXACT (verified here) and reconstructs the
free region well. Baseline for context = plain generate() from scratch.

  uv run python script/test_edit.py \
      --towers tower_16325 tower_268707 --frac 0.5
"""
import os, sys, argparse
from os.path import join as pjoin
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.paths import DATA_ROOT

import numpy as np
import torch

from utils.voxel import voxel_iou, grid_snap
from utils.model_io import load_vq

DATA = DATA_ROOT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--towers', nargs='+', default=['tower_16325'])
    ap.add_argument('--frac', type=float, default=0.5, help='fraction of tokens kept fixed')
    ap.add_argument('--timesteps', type=int, default=10)
    ap.add_argument('--temperature', type=float, default=0.8)
    ap.add_argument('--gpu_id', type=int, default=0)
    args = ap.parse_args()

    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    from eval_sampling import load_mask_model

    vq, vq_opt = load_vq('checkpoints', 'ibq_4x_nc1024_nq1', device=device,
                          dim_pose=3, dataset_name='myarch')
    vq.eval().to(device)
    model = load_mask_model(vq_opt, 'myarch_mask_img_patch_v8_2', 'net_best_iou.tar', device)
    meta = 'checkpoints/myarch/ibq_4x_nc1024_nq1/meta'
    mean = np.load(pjoin(meta, 'mean.npy')).reshape(-1)[:3]
    std = np.load(pjoin(meta, 'std.npy')).reshape(-1)[:3]

    def decode(ids, L):
        ids = ids[:, :L].clamp(min=0)
        with torch.no_grad():
            xyz = vq.forward_decoder(ids.unsqueeze(-1))[0].cpu().numpy()
        return grid_snap(xyz * std + mean)

    for tw in args.towers:
        motion = np.load(pjoin(DATA, 'new_joint_vecs_int', tw + '.npy')).reshape(-1, 3)
        feat = np.load(pjoin(DATA, 'image_features_dino_patch', tw + '.npy')).astype(np.float32)
        n_cmp = (len(motion) // 4) * 4
        gt = motion[:n_cmp].astype(float)

        with torch.no_grad():
            m = torch.from_numpy((motion[:n_cmp] - mean) / std).float().unsqueeze(0).to(device)
            gt_tokens, _ = vq.encode(m)                    # (1, L, 1)
        gt_ids = gt_tokens[..., 0]                          # (1, L)
        L = gt_ids.shape[1]
        cond = torch.from_numpy(feat).unsqueeze(0).to(device).float()
        m_lens = torch.tensor([L], device=device)

        print(f'\n=== {tw}: {n_cmp} blocks, {L} tokens ===', flush=True)
        # GT-token ceiling + from-scratch baseline for context
        print(f'  GT tokens (tokenizer ceiling) IoU  {voxel_iou(decode(gt_ids, L), gt):.4f}')
        gen = model.generate(cond, m_lens, timesteps=args.timesteps, cond_scale=1.0,
                             temperature=0.2, topk_filter_thres=0.9, gsample=False)
        print(f'  generate() from scratch       IoU  {voxel_iou(decode(gen, L), gt):.4f}')

        for name in ('keep_prefix', 'hole', 'mask_free'):
            if name == 'mask_free':
                em = None
            else:
                em = torch.zeros(1, L, dtype=torch.bool, device=device)
                if name == 'keep_prefix':
                    em[:, int(args.frac * L):] = True             # resample the tail
                else:
                    a = int(0.5 * (1 - args.frac) * L) + int(args.frac * L) // 2
                    lo = max(0, L // 2 - int((1 - args.frac) * L) // 2)
                    hi = min(L, lo + int((1 - args.frac) * L))
                    em[:, lo:hi] = True                          # resample a middle hole
            out = model.edit(cond, gt_ids, m_lens, timesteps=args.timesteps,
                             cond_scale=1.0, temperature=args.temperature,
                             topk_filter_thres=0.9, gsample=False, edit_mask=em)
            iou = voxel_iou(decode(out, L), gt)
            if em is not None:
                kept_ok = bool((out[:, :L][~em] == gt_ids[~em]).all())
                n_free = int(em.sum())
                print(f'  edit/{name:<11} IoU {iou:.4f}   free={n_free}/{L} tokens   '
                      f'fixed-region preserved exactly: {kept_ok}')
            else:
                print(f'  edit/{name:<11} IoU {iou:.4f}   (all {L} tokens re-decoded)')


if __name__ == '__main__':
    main()
