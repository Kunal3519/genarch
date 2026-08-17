"""Side probe: measure CE-grad vs coord-grad norms for ANY mask-transformer ckpt.

Point it at any model via --name/--ckpt (reads vq_name + cond_mode from its opt.txt).
Single-GPU, no DDP, no checkpoint writes, safe to run alongside a live training job.
Reuses MaskTransformerTrainer._compute_losses so the math matches training exactly.

    CUDA_VISIBLE_DEVICES=<i> uv run python \
        script/probe_grads.py --name myarch_mask_img_patch_v7 --ckpt latest.tar
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import argparse
import torch
import numpy as np
from os.path import join as pjoin
from torch.utils.data import DataLoader

from models.mask_transformer.transformer import MaskTransformer
from models.mask_transformer.transformer_trainer import MaskTransformerTrainer
from utils.get_opt import get_opt
from utils.model_io import load_vq
from utils.fixseed import fixseed
from data.tower_dataset import Image2TowerDataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', default='myarch_mask_img_patch_v7')
    ap.add_argument('--ckpt', default='latest.tar')
    ap.add_argument('--n_batches', type=int, default=8)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--seed', type=int, default=12345)
    args = ap.parse_args()

    fixseed(args.seed)
    device = torch.device('cuda:0')  # rely on CUDA_VISIBLE_DEVICES to pick the physical GPU
    torch.cuda.set_device(0)

    opt_path = pjoin('./checkpoints/myarch', args.name, 'opt.txt')
    opt = get_opt(opt_path, device=device)
    # Single-GPU, training-mode trainer with no DDP, no logger, no saves
    opt.use_ddp = False
    opt.is_main_process = True
    opt.local_rank = 0
    opt.is_continue = False
    opt.is_train = False  # skips SummaryWriter; probe only needs forward+backward
    opt.gpu_id = 0
    opt.device = device
    opt.checkpoints_dir = './checkpoints'
    opt.save_root = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name)
    opt.model_dir = pjoin(opt.save_root, 'model')

    # Patch-script defaults
    opt.motion_dir      = pjoin(opt.data_root, 'new_joint_vecs_int')
    opt.image_feat_dir  = pjoin(opt.data_root, 'image_features_dino_patch')
    opt.joints_num      = 1
    opt.max_motion_len  = 512
    opt.max_motion_length = 2000

    print(f'Loading from {opt.model_dir}/{args.ckpt}')
    print(f'ce_loss_weight={opt.ce_loss_weight}, coord_loss_weight={opt.coord_loss_weight}')

    vq_model, vq_opt = load_vq(opt.checkpoints_dir, opt.vq_name,
                                device=device, dim_pose=3, dataset_name=opt.dataset_name)
    opt.num_tokens     = vq_opt.nb_code
    opt.num_quantizers = vq_opt.num_quantizers
    opt.code_dim       = vq_opt.code_dim

    if getattr(opt, 'use_cls_prefix', False):
        cond_mode = 'image_patch_clsprefix'
    else:
        cond_mode = 'image_patch'
    print(f'cond_mode: {cond_mode}')

    t2m = MaskTransformer(
        code_dim=vq_opt.code_dim, cond_mode=cond_mode,
        latent_dim=opt.latent_dim, ff_size=opt.ff_size,
        num_layers=opt.n_layers, num_heads=opt.n_heads,
        dropout=opt.dropout, clip_dim=1024,
        cond_drop_prob=opt.cond_drop_prob, clip_version=None, opt=opt)

    mean = np.load(pjoin(opt.checkpoints_dir, opt.dataset_name, opt.vq_name, 'meta', 'mean.npy'))
    std  = np.load(pjoin(opt.checkpoints_dir, opt.dataset_name, opt.vq_name, 'meta', 'std.npy'))

    val_dataset = Image2TowerDataset(opt, mean, std, pjoin(opt.data_root, 'train160k.txt'))
    loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=2,
                        drop_last=True, shuffle=True, pin_memory=True)

    trainer = MaskTransformerTrainer(opt, t2m, vq_model)
    trainer.t2m_transformer.to(device)
    trainer.vq_model.to(device).eval()
    trainer.opt_t2m_transformer = torch.optim.AdamW(
        trainer.t2m_transformer.parameters(), lr=1e-4, betas=(0.9, 0.99), weight_decay=1e-5)
    trainer.logger = None  # gradient_probe checks `if self.logger is not None` before TB writes

    ckpt_path = pjoin(opt.model_dir, args.ckpt)
    state = torch.load(ckpt_path, map_location=device)
    sd = state['t2m_transformer']
    if any(k.startswith('module.') for k in sd):
        sd = {k.replace('module.', '', 1): v for k, v in sd.items()}
    sd = {k: v for k, v in sd.items() if not k.startswith('sbert_model.')}
    missing, unexpected = trainer.t2m_transformer.load_state_dict(sd, strict=False)
    assert not unexpected, f'unexpected: {unexpected[:5]}'
    print(f'Loaded ckpt at epoch={state.get("ep")} iter={state.get("total_it")}')

    grad_ces, grad_coords = [], []
    ce_losses, coord_losses = [], []
    trainer.t2m_transformer.train()  # match probe semantics (dropout on, as in training)

    def grad_l2():
        total = 0.0
        for p in trainer.t2m_transformer.parameters():
            if p.grad is not None:
                total += p.grad.detach().pow(2).sum().item()
        return total ** 0.5

    for i, batch in enumerate(loader):
        if i >= args.n_batches:
            break
        ce_t, coord_t, _ = trainer._compute_losses(batch)
        ce_losses.append(ce_t.item())
        coord_losses.append(coord_t.item() if torch.is_tensor(coord_t) else float(coord_t))

        trainer.opt_t2m_transformer.zero_grad()
        (opt.ce_loss_weight * ce_t).backward(retain_graph=True)
        g_ce = grad_l2()

        trainer.opt_t2m_transformer.zero_grad()
        (opt.coord_loss_weight * coord_t).backward()
        g_coord = grad_l2()

        trainer.opt_t2m_transformer.zero_grad()
        grad_ces.append(g_ce)
        grad_coords.append(g_coord)
        ratio = g_coord / max(g_ce, 1e-12)
        coord_share = g_coord / (g_ce + g_coord)
        print(f'batch {i:2d}: |∇(0.1·CE)|={g_ce:.4f}  |∇(1.0·coord)|={g_coord:.4f}  '
              f'coord/ce={ratio:.2f}x  coord_share={100*coord_share:.1f}%')

    g_ce_m = float(np.mean(grad_ces))
    g_coord_m = float(np.mean(grad_coords))
    print()
    print(f'=== n={args.n_batches} batches (bs={args.batch_size}) ===')
    print(f'Mean ce_loss      : {np.mean(ce_losses):.4f}  (weighted: {opt.ce_loss_weight * np.mean(ce_losses):.4f})')
    print(f'Mean coord_loss   : {np.mean(coord_losses):.4f}  (weighted: {opt.coord_loss_weight * np.mean(coord_losses):.4f})')
    print(f'Mean |∇(w·CE)|    : {g_ce_m:.4f}')
    print(f'Mean |∇(w·coord)| : {g_coord_m:.4f}')
    print(f'Ratio coord/ce    : {g_coord_m/max(g_ce_m,1e-12):.2f}x')
    print(f'Coord grad share  : {100*g_coord_m/(g_ce_m+g_coord_m):.1f}%   (l2-sum approx; not strict since grads can partially cancel)')


if __name__ == '__main__':
    main()
