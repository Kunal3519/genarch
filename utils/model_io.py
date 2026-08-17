"""Shared loaders for RVQ + Mask/Residual Transformers.

All current scripts loaded these models with the same 25-ish lines of boilerplate
(read opt.txt, construct module, fall back through net_best_iou.tar →
net_best_fid.tar → latest.tar). This module centralises that.
"""
import os
from os.path import join as pjoin

import torch

from models.vq.vqvae import VQVAE
from utils.get_opt import get_opt


_VQ_CKPT_FALLBACKS = ('net_best_iou.tar', 'net_best_fid.tar', 'latest.tar')


def _resolve_ckpt(model_dir, fallbacks=_VQ_CKPT_FALLBACKS, override=None):
    """Pick the first existing checkpoint from `fallbacks` (or `override` if given)."""
    if override is not None:
        path = pjoin(model_dir, override)
        if not os.path.exists(path):
            raise FileNotFoundError(f'Checkpoint not found: {path}')
        return path
    for name in fallbacks:
        path = pjoin(model_dir, name)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        f'No checkpoint in {model_dir} matching any of {fallbacks}')


def load_vq(checkpoints_dir, vq_name, device, dim_pose=3,
             dataset_name='myarch', ckpt_name=None, verbose=True):
    """Construct a VQVAE and load weights.

    Returns (vq_model, vq_opt). Model is on CPU, not in eval mode — callers
    decide when to .to(device) and .eval().
    """
    opt_path = pjoin(checkpoints_dir, dataset_name, vq_name, 'opt.txt')
    vq_opt = get_opt(opt_path, device=device)
    vq_opt.checkpoints_dir = checkpoints_dir

    vq_model = VQVAE(
        vq_opt, dim_pose,
        vq_opt.nb_code, vq_opt.code_dim, vq_opt.output_emb_width,
        vq_opt.down_t, vq_opt.stride_t, vq_opt.width, vq_opt.depth,
        vq_opt.dilation_growth_rate, vq_opt.vq_act, vq_opt.vq_norm,
    )

    model_dir = pjoin(checkpoints_dir, dataset_name, vq_name, 'model')
    ckpt_path = _resolve_ckpt(model_dir, override=ckpt_name)
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state = ckpt.get('vq_model', ckpt.get('net'))
    vq_model.load_state_dict(state)

    if verbose:
        print(f'Loaded VQ "{vq_name}" from {ckpt_path}  '
              f'(nb_code={vq_opt.nb_code}, code_dim={vq_opt.code_dim}, '
              f'num_q={vq_opt.num_quantizers}, down_t={vq_opt.down_t})')
    return vq_model, vq_opt
