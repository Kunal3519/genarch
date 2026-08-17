"""Refinement network -- Stage 3 of GenArch (paper section "Refinement Model").

The N decoded candidates are snapped to the voxel grid and merged into a
candidate set (union of all samples plus a 1-ring dilation, so voxels missed by
every sample but adjacent to predicted structure stay recoverable). Each
candidate carries pool statistics -- occupancy frequency across the N samples,
normalised position, local neighbourhood -- and cross-attends to the same DINOv2
patch memory the mask transformer was conditioned on, so it can ask "is there
actually a block here in the photos?". Output: one keep/drop logit per candidate
voxel, thresholded at tau.

This is a SET problem, not a sequence one: there is no positional correspondence
between a draft and the ground truth, which is why per-position regression
refiners failed and this formulation works.

Towers are pad-batched on the voxel axis (key_padding_mask); padded queries are
dropped from the loss / IoU.

Deployed checkpoint: checkpoints/myarch/pool_classifier/poolcls_c_sym_v1 @ tau 0.45.
(Historically called "Classifier C" / pool classifier -- never use those names in
the paper.)
"""
import torch
import torch.nn as nn

from models.mask_transformer.transformer import SelfCrossBlock


class RefinementNetwork(nn.Module):
    def __init__(self, n_feat, latent_dim=256, ff_size=1024, num_layers=6,
                 num_heads=8, dropout=0.1):
        super().__init__()
        self.in_proj = nn.Linear(n_feat, latent_dim)
        self.pos_proj = nn.Linear(3, latent_dim)        # emphasise voxel xyz for image lookup

        # Patch memory: identical construction to MaskTransformer's patch path.
        self.patch_proj = nn.Linear(1024, latent_dim)
        self.patch_norm = nn.LayerNorm(latent_dim)
        self.view_emb = nn.Embedding(4, latent_dim)
        self.patch_pos_emb = nn.Parameter(torch.randn(256, latent_dim) * 0.02)

        self.blocks = nn.ModuleList([
            SelfCrossBlock(latent_dim, num_heads, ff_size, dropout)
            for _ in range(num_layers)
        ])
        self.out_norm = nn.LayerNorm(latent_dim)
        self.out_proj = nn.Linear(latent_dim, 1)

    def build_patch_memory(self, feat):
        # feat: (B, 4, 256, 1024) -> mem: [1024, B, latent_dim]
        B = feat.shape[0]
        mem = feat.reshape(B, 4 * 256, 1024)
        mem = self.patch_norm(self.patch_proj(mem))
        view_ids = torch.arange(4, device=mem.device).repeat_interleave(256)
        mem = mem + self.view_emb(view_ids).unsqueeze(0)
        mem = mem + self.patch_pos_emb.unsqueeze(0).repeat(1, 4, 1)
        return mem.permute(1, 0, 2)

    def forward(self, vox_feat, vox_pos, feat, key_padding_mask=None):
        """
        vox_feat: (B, M, n_feat)   per-voxel context features (vote, neighbours, pos, height)
        vox_pos:  (B, M, 3)        normalised voxel xyz
        feat:     (B, 4, 256, 1024) DINOv2 patch features
        key_padding_mask: (B, M) bool, True at padded voxel slots
        returns logits (B, M)
        """
        mem = self.build_patch_memory(feat)                       # [1024, B, d]
        x = self.in_proj(vox_feat) + self.pos_proj(vox_pos)       # (B, M, d)
        x = x.permute(1, 0, 2)                                    # [M, B, d]
        for blk in self.blocks:
            x = blk(x, mem, src_key_padding_mask=key_padding_mask)
        return self.out_proj(self.out_norm(x)).squeeze(-1).permute(1, 0)  # (B, M)
