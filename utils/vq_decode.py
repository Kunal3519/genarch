"""Windowed VQ decode helpers.

The VQ-VAE decoder is a strided conv stack (depth=3, k=3, s=2 by default). At
higher compression (4x+) the receptive field at the sequence ends produces
catastrophic boundary artifacts when you decode a whole token sequence in one
shot. We decode in fixed token windows matching the training window
(typically 64 motion blocks → 16 tokens at 4x compression) and concatenate.

See CLAUDE.md "Mask trainer val IoU bug fix (May 2026)" for the history.
"""
from __future__ import annotations

import torch


DEFAULT_WINDOW_BLOCKS = 64
DEFAULT_WINDOW_TOKENS = 16  # 64 blocks @ 4x compression (down_t=2, stride_t=2)


@torch.no_grad()
def decode_windowed(vq_model, code_idx_full: torch.Tensor,
                    window_tokens: int = DEFAULT_WINDOW_TOKENS) -> torch.Tensor:
    """Decode code indices in fixed-size chunks, then concatenate.

    code_idx_full : ``[B, T, Q]`` long token grid
    Returns        : ``[B, T * down_factor, dim_pose]`` coords
    """
    B, T, Q = code_idx_full.shape
    pieces = []
    for s in range(0, T, window_tokens):
        chunk = code_idx_full[:, s:s + window_tokens, :]
        pieces.append(vq_model.forward_decoder(chunk))
    return torch.cat(pieces, dim=1)


@torch.no_grad()
def probe_window_tokens(vq_model, dim_pose: int, device,
                        window_blocks: int = DEFAULT_WINDOW_BLOCKS) -> int:
    """Encode a zero motion of length ``window_blocks`` to discover tokens-per-window.

    e.g. 64 motion blocks at 4x compression → 16 tokens.
    """
    probe = torch.zeros(1, window_blocks, dim_pose, device=device)
    probe_idx, _ = vq_model.encode(probe)
    return int(probe_idx.shape[1])
