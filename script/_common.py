"""Shared helpers for the generation / eval scripts.

Extracted from gen_compare_methods.py (removed 2026-08-10 along with the dead
token-space refiners) so that the surviving scripts share one implementation
instead of each carrying a copy. See REPO_AUDIT.md sec. 1.8.

Deployment protocol constants (MASK_TIMESTEPS / MASK_COND_SCALE / MASK_TOPK) are
imported from eval_sampling, which remains the single source of truth for them.
"""
import os
import sys
from collections import Counter

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from eval_sampling import MASK_TIMESTEPS, MASK_COND_SCALE, MASK_TOPK  # noqa: E402


@torch.no_grad()
def gen_drafts(model, feat, n_tok, n_samples, temp, device, nb_code):
    """Draw n_samples stochastic token sequences from the frozen mask transformer."""
    cond = torch.from_numpy(feat).unsqueeze(0).repeat(n_samples, 1, 1, 1).to(device).float()
    lens = torch.full((n_samples,), n_tok, device=device, dtype=torch.long)
    ids = model.generate(cond, lens, timesteps=MASK_TIMESTEPS, cond_scale=MASK_COND_SCALE,
                         temperature=temp, topk_filter_thres=MASK_TOPK, gsample=False)
    return ids.clamp(0, nb_code - 1)


@torch.no_grad()
def decode_xyz(vq, ids, n_cmp, mean, std):
    """(B, n_tok) ids -> list of B xyz arrays (whole-seq IBQ decode)."""
    coords = vq.forward_decoder(ids.unsqueeze(-1)).cpu().numpy()
    return [coords[i, :n_cmp] * std + mean for i in range(ids.shape[0])]


def vote_set(sets, thr_count):
    """Voxels present in at least thr_count of the given voxel sets."""
    cnt = Counter()
    for vs in sets:
        cnt.update(vs)
    return {v for v, c in cnt.items() if c >= thr_count}
