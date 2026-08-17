import torch
import torch.nn as nn
import numpy as np
# from networks.layers import *
import torch.nn.functional as F
from einops import rearrange, repeat
import math
from random import random
from tqdm.auto import tqdm
from typing import Callable, Optional, List, Dict
from copy import deepcopy
from functools import partial
from models.mask_transformer.tools import *
from torch.distributions.categorical import Categorical

class SelfCrossBlock(nn.Module):
    """Transformer block with self-attention followed by cross-attention to patch memory."""
    def __init__(self, latent_dim, num_heads, ff_size, dropout):
        super().__init__()
        self.self_attn  = nn.MultiheadAttention(latent_dim, num_heads, dropout=dropout)
        self.cross_attn = nn.MultiheadAttention(latent_dim, num_heads, dropout=dropout)
        self.ff = nn.Sequential(
            nn.Linear(latent_dim, ff_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_size, latent_dim),
        )
        self.norm1 = nn.LayerNorm(latent_dim)
        self.norm2 = nn.LayerNorm(latent_dim)
        self.norm3 = nn.LayerNorm(latent_dim)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x, mem, src_key_padding_mask=None):
        # x:   [seqlen, B, d], token sequence (queries)
        # mem: [num_patches, B, d], patch features (keys + values)
        # need_weights=False routes through fused SDPA: without it MHA materializes
        # (B*heads, T, T) attention maps, which OOMs at block-level T (~1800).
        x2, _ = self.self_attn(x, x, x, key_padding_mask=src_key_padding_mask,
                               need_weights=False)
        x = self.norm1(x + self.drop(x2))
        x2, _ = self.cross_attn(x, mem, mem, need_weights=False)
        x = self.norm2(x + self.drop(x2))
        x2 = self.ff(x)
        x = self.norm3(x + self.drop(x2))
        return x



class SelfCrossBlockFFNAdaLN(nn.Module):
    """v5: SelfCrossBlock (v4) with AdaLN-Zero modulation added at the FFN sublayer.
    Self-attn and cross-attn stay post-norm (v4-style); FFN switches to pre-norm
    with affine=False, AdaLN-Zero shift/scale/gate driven by pooled cond vector c.
    Output of adaLN_mod_ffn is zero-initialized → FFN starts as identity at the
    modulation sites, conditioning ramps in. The v5 CLS-prefix mechanism is unchanged.
    Conditioning channels per block: (a) self-attn sees CLS prefix tokens, (b) cross-attn
    over patches (unchanged from v4), (c) FFN modulated by pooled-patch c vector."""
    def __init__(self, latent_dim, num_heads, ff_size, dropout):
        super().__init__()
        self.self_attn  = nn.MultiheadAttention(latent_dim, num_heads, dropout=dropout)
        self.cross_attn = nn.MultiheadAttention(latent_dim, num_heads, dropout=dropout)
        self.ff = nn.Sequential(
            nn.Linear(latent_dim, ff_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_size, latent_dim),
        )
        self.norm1 = nn.LayerNorm(latent_dim)
        self.norm2 = nn.LayerNorm(latent_dim)
        # norm3 = pre-norm + affine=False for AdaLN-Zero on FFN
        self.norm3 = nn.LayerNorm(latent_dim, elementwise_affine=False)
        self.drop = nn.Dropout(dropout)
        self.adaLN_mod_ffn = nn.Sequential(
            nn.SiLU(),
            nn.Linear(latent_dim, 3 * latent_dim),  # shift, scale, gate for FFN
        )
        nn.init.zeros_(self.adaLN_mod_ffn[-1].weight)
        nn.init.zeros_(self.adaLN_mod_ffn[-1].bias)

    def forward(self, x, mem, c, src_key_padding_mask=None):
        # x: [seqlen, B, d], mem: [num_patches, B, d], c: [B, d]
        # Self-attn (post-norm, v4-style)
        x2, _ = self.self_attn(x, x, x, key_padding_mask=src_key_padding_mask)
        x = self.norm1(x + self.drop(x2))
        # Cross-attn (post-norm, v4-style)
        x2, _ = self.cross_attn(x, mem, mem)
        x = self.norm2(x + self.drop(x2))
        # FFN: pre-norm with affine=False + AdaLN-Zero shift/scale/gate
        mod = self.adaLN_mod_ffn(c)
        shift, scale, gate = mod.chunk(3, dim=-1)
        shift = shift.unsqueeze(0); scale = scale.unsqueeze(0); gate = gate.unsqueeze(0)
        x_norm = self.norm3(x) * (1 + scale) + shift
        x = x + gate * self.ff(x_norm)
        return x


class InputProcess(nn.Module):
    def __init__(self, input_feats, latent_dim):
        super().__init__()
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.poseEmbedding = nn.Linear(self.input_feats, self.latent_dim)

    def forward(self, x):
        # [bs, ntokens, input_feats]
        x = x.permute((1, 0, 2)) # [seqen, bs, input_feats]
        # print(x.shape)
        x = self.poseEmbedding(x)  # [seqlen, bs, d]
        return x

class PositionalEncoding(nn.Module):
    #Borrow from MDM, the same as above, but add dropout, exponential may improve precision
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1) #[max_len, 1, d_model]

        self.register_buffer('pe', pe)

    def forward(self, x):
        # not used in the final model
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)

class OutputProcess_Bert(nn.Module):
    def __init__(self, out_feats, latent_dim):
        super().__init__()
        self.dense = nn.Linear(latent_dim, latent_dim)
        self.transform_act_fn = F.gelu
        self.LayerNorm = nn.LayerNorm(latent_dim, eps=1e-12)
        self.poseFinal = nn.Linear(latent_dim, out_feats) #Bias!

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        output = self.poseFinal(hidden_states)  # [seqlen, bs, out_feats]
        output = output.permute(1, 2, 0)  # [bs, c, seqlen]
        return output

class OutputProcess(nn.Module):
    def __init__(self, out_feats, latent_dim):
        super().__init__()
        self.dense = nn.Linear(latent_dim, latent_dim)
        self.transform_act_fn = F.gelu
        self.LayerNorm = nn.LayerNorm(latent_dim, eps=1e-12)
        self.poseFinal = nn.Linear(latent_dim, out_feats) #Bias!

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        output = self.poseFinal(hidden_states)  # [seqlen, bs, out_feats]
        output = output.permute(1, 2, 0)  # [bs, e, seqlen]
        return output


class MaskTransformer(nn.Module):
    def __init__(self, code_dim, cond_mode, latent_dim=256, ff_size=1024, num_layers=8,
                 num_heads=4, dropout=0.1, clip_dim=512, cond_drop_prob=0.1,
                 opt=None, **kargs):
        super(MaskTransformer, self).__init__()
        print(f'latent_dim: {latent_dim}, ff_size: {ff_size}, nlayers: {num_layers}, nheads: {num_heads}, dropout: {dropout}')

        self.code_dim = code_dim
        self.latent_dim = latent_dim
        self.clip_dim = clip_dim
        self.dropout = dropout
        self.opt = opt

        self.cond_mode = cond_mode
        self.cond_drop_prob = cond_drop_prob

        '''
        Preparing Networks
        '''
        self.input_process = InputProcess(self.code_dim, self.latent_dim)
        self.position_enc = PositionalEncoding(self.latent_dim, self.dropout)

        if self.cond_mode == 'image_patch':
            # Cross-attention over DINOv2 patch tokens (4 views × 256 patches × 1024-dim)
            # patch_proj maps each patch from 1024-dim to latent_dim
            self.patch_proj = nn.Linear(1024, self.latent_dim)
            self.patch_norm = nn.LayerNorm(self.latent_dim)
            # Learnable view embedding: which of the 4 views (NE/NW/SE/SW)
            self.view_emb = nn.Embedding(4, self.latent_dim)
            # Learnable 2D spatial position embedding: 16×16 patch grid within each view
            self.patch_pos_emb = nn.Parameter(torch.randn(256, self.latent_dim) * 0.02)
            self.cross_blocks = nn.ModuleList([
                SelfCrossBlock(self.latent_dim, num_heads, ff_size, dropout)
                for _ in range(num_layers)
            ])
            self.cond_emb = nn.Identity()  # unused placeholder for weight init
        elif self.cond_mode == 'image_patch_clsprefix':
            # v5: v4 cross-attn architecture + k=4 CLS-style prefix tokens prepended to
            # the motion sequence (per-view mean-pool of the projected patches) + AdaLN-Zero
            # modulation at the FFN sublayer driven by a pooled-patch cond vector c.
            # Three conditioning channels per block:
            #   (a) self-attn sees CLS prefix tokens (attention-based, content-addressable)
            #   (b) cross-attn over the 1024 patches (unchanged from v4)
            #   (c) FFN modulated by pooled-patch c via AdaLN-Zero shift/scale/gate
            self.patch_proj = nn.Linear(1024, self.latent_dim)
            self.patch_norm = nn.LayerNorm(self.latent_dim)
            self.view_emb = nn.Embedding(4, self.latent_dim)
            self.patch_pos_emb = nn.Parameter(torch.randn(256, self.latent_dim) * 0.02)
            # Projection + per-view tag for the 4 CLS-style prefix tokens
            self.cls_proj = nn.Linear(self.latent_dim, self.latent_dim)
            self.cls_view_emb = nn.Embedding(4, self.latent_dim)
            self.num_cls_prefix = 4
            # Global cond vector c for FFN AdaLN modulation (separate from the prefix tokens)
            self.cond_mlp = nn.Sequential(
                nn.Linear(self.latent_dim, self.latent_dim),
                nn.SiLU(),
                nn.Linear(self.latent_dim, self.latent_dim),
            )
            self.cross_blocks = nn.ModuleList([
                SelfCrossBlockFFNAdaLN(self.latent_dim, num_heads, ff_size, dropout)
                for _ in range(num_layers)
            ])
            self.cond_emb = nn.Identity()
        else:
            seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                              nhead=num_heads,
                                                              dim_feedforward=ff_size,
                                                              dropout=dropout,
                                                              activation='gelu')
            self.seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer,
                                                         num_layers=num_layers)
            if self.cond_mode == 'uncond':
                self.cond_emb = nn.Identity()
            else:
                raise KeyError("Unsupported condition mode!!!")


        _num_tokens = opt.num_tokens + 2  # two dummy tokens, one for masking, one for padding
        self.mask_id = opt.num_tokens
        self.pad_id = opt.num_tokens + 1

        self.output_process = OutputProcess_Bert(out_feats=opt.num_tokens, latent_dim=latent_dim)

        self.token_emb = nn.Embedding(_num_tokens, self.code_dim)

        self.apply(self.__init_weights)

        '''
        Preparing frozen weights
        '''

        self.noise_schedule = cosine_schedule
        # Training masking-ratio distribution (ablation). Default 'cosine' keeps the
        # deployed behaviour EXACTLY. 'uniform' samples r ~ U(alpha, 1) (MMM-style).
        self.train_mask_dist = getattr(opt, 'train_mask_dist', 'cosine') if opt is not None else 'cosine'
        self.train_mask_alpha = getattr(opt, 'train_mask_alpha', 0.0) if opt is not None else 0.0

    def __init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            # AdaLN-Zero blocks use elementwise_affine=False (no learnable weight/bias)
            if module.bias is not None:
                module.bias.data.zero_()
            if module.weight is not None:
                module.weight.data.fill_(1.0)
        # Re-apply AdaLN-Zero output zero-init (overwritten by the .normal_ above)
        if isinstance(module, SelfCrossBlockFFNAdaLN):
            nn.init.zeros_(module.adaLN_mod_ffn[-1].weight)
            nn.init.zeros_(module.adaLN_mod_ffn[-1].bias)

    def mask_cond(self, cond, force_mask=False):
        bs, d =  cond.shape
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.cond_drop_prob > 0.:
            mask = torch.bernoulli(torch.ones(bs, device=cond.device) * self.cond_drop_prob).view(bs, 1)
            return cond * (1. - mask)
        else:
            return cond

    def trans_forward(self, motion_ids, cond, padding_mask, force_mask=False):
        '''
        :param motion_ids: (b, seqlen)
        :padding_mask: (b, seqlen), all pad positions are TRUE else FALSE
        :param cond: (b, clip_dim) for text/image, (b, 4, 256, 1024) for image_patch
        :param force_mask: boolean
        :return:
            -logits: (b, num_token, seqlen)
        '''
        if self.cond_mode == 'image_patch':
            return self._trans_forward_patch(motion_ids, cond, padding_mask, force_mask)
        if self.cond_mode == 'image_patch_clsprefix':
            return self._trans_forward_patch_clsprefix(motion_ids, cond, padding_mask, force_mask)

        cond = self.mask_cond(cond, force_mask=force_mask)

        x = self.token_emb(motion_ids)
        x = self.input_process(x)          # [seqlen, b, latent_dim]
        cond = self.cond_emb(cond).unsqueeze(0)  # [1, b, latent_dim]
        x = self.position_enc(x)
        xseq = torch.cat([cond, x], dim=0)  # [seqlen+1, b, latent_dim]
        padding_mask = torch.cat([torch.zeros_like(padding_mask[:, 0:1]), padding_mask], dim=1)
        output = self.seqTransEncoder(xseq, src_key_padding_mask=padding_mask)[1:]
        logits = self.output_process(output)
        return logits

    def _trans_forward_patch(self, motion_ids, cond, padding_mask, force_mask=False):
        '''
        Cross-attention forward for image_patch mode.
        :param cond: (b, 4, 256, 1024), DINOv2 patch tokens, 4 views × 256 patches × 1024-dim
        '''
        B = motion_ids.shape[0]

        # CFG: zero out all patches for unconditional branch
        if force_mask:
            cond = torch.zeros_like(cond)
        elif self.training and self.cond_drop_prob > 0.:
            # Drop entire tower's patches with prob cond_drop_prob
            drop = torch.bernoulli(
                torch.ones(B, device=cond.device) * self.cond_drop_prob
            ).view(B, 1, 1, 1)
            cond = cond * (1. - drop)

        # Project patches: [B, 4, 256, 1024] → [B, 1024, latent_dim]
        mem = cond.reshape(B, 4 * 256, 1024)   # [B, 1024, 1024]
        mem = self.patch_proj(mem)              # [B, 1024, latent_dim]
        mem = self.patch_norm(mem)              # [B, 1024, latent_dim]

        # Add view embedding (which of the 4 views each patch belongs to)
        view_ids = torch.arange(4, device=mem.device).repeat_interleave(256)  # [1024]
        mem = mem + self.view_emb(view_ids).unsqueeze(0)  # [B, 1024, latent_dim]

        # Add 2D spatial position embedding (patch position within each view)
        patch_pos = self.patch_pos_emb.unsqueeze(0).repeat(1, 4, 1)  # [1, 1024, latent_dim]
        mem = mem + patch_pos                               # [B, 1024, latent_dim]

        mem = mem.permute(1, 0, 2)             # [1024, B, latent_dim]

        # Token sequence
        x = self.token_emb(motion_ids)
        x = self.input_process(x)   # [seqlen, B, latent_dim]
        x = self.position_enc(x)    # [seqlen, B, latent_dim]

        # Self-attention + cross-attention blocks
        for blk in self.cross_blocks:
            x = blk(x, mem, src_key_padding_mask=padding_mask)

        logits = self.output_process(x)  # [B, num_token, seqlen]
        return logits

    def _trans_forward_patch_clsprefix(self, motion_ids, cond, padding_mask, force_mask=False):
        '''
        v5: v4 cross-attn over patches + k=4 CLS-style prefix tokens prepended to the motion
        sequence + AdaLN-Zero modulation at the FFN driven by a pooled global cond vector.
        CFG drop zeros raw patches → patch memory, prefix tokens, AND pooled c all go to
        bias-only ("uncond") together.
        :param cond: (B, 4, 256, 1024), DINOv2 patch tokens
        '''
        B = motion_ids.shape[0]

        # CFG: zero patches for unconditional branch
        if force_mask:
            cond = torch.zeros_like(cond)
        elif self.training and self.cond_drop_prob > 0.:
            drop = torch.bernoulli(
                torch.ones(B, device=cond.device) * self.cond_drop_prob
            ).view(B, 1, 1, 1)
            cond = cond * (1. - drop)

        # Per-view dropout (v8.2): independently zero whole views so partial-view
        # conditioning is in-distribution at inference, not just the all-or-nothing
        # CFG drop above.
        view_drop_prob = getattr(self.opt, 'view_drop_prob', 0.)
        if self.training and not force_mask and view_drop_prob > 0.:
            vdrop = torch.bernoulli(
                torch.ones(B, 4, device=cond.device) * view_drop_prob
            ).view(B, 4, 1, 1)
            cond = cond * (1. - vdrop)

        # Patch memory (identical pipeline to image_patch / v4)
        mem = cond.reshape(B, 4 * 256, 1024)
        mem = self.patch_proj(mem)                       # [B, 1024, latent_dim]
        mem = self.patch_norm(mem)
        view_ids = torch.arange(4, device=mem.device).repeat_interleave(256)
        mem = mem + self.view_emb(view_ids).unsqueeze(0) # [B, 1024, latent_dim]
        patch_pos = self.patch_pos_emb.unsqueeze(0).repeat(1, 4, 1)
        mem = mem + patch_pos                            # [B, 1024, latent_dim]

        # k=4 CLS-style prefix tokens: per-view mean-pool of the projected patches
        mem_per_view = mem.view(B, 4, 256, self.latent_dim)
        cls_tokens = mem_per_view.mean(dim=2)            # [B, 4, latent_dim]
        cls_tokens = self.cls_proj(cls_tokens)           # [B, 4, latent_dim]
        cls_view_ids = torch.arange(4, device=cls_tokens.device)
        cls_tokens = cls_tokens + self.cls_view_emb(cls_view_ids).unsqueeze(0)
        cls_tokens = cls_tokens.permute(1, 0, 2)         # [4, B, latent_dim]

        # Pooled global cond vector c for FFN AdaLN-Zero modulation
        c = mem.mean(dim=1)                              # [B, latent_dim]
        c = self.cond_mlp(c)                             # [B, latent_dim]

        mem = mem.permute(1, 0, 2)                       # [1024, B, latent_dim]

        # Motion token sequence
        x = self.token_emb(motion_ids)
        x = self.input_process(x)   # [seqlen, B, latent_dim]
        x = self.position_enc(x)
        x = torch.cat([cls_tokens, x], dim=0)            # [k+seqlen, B, latent_dim]

        # Pad-mask: prefix tokens are never padded
        prefix_pad = torch.zeros(B, self.num_cls_prefix,
                                 dtype=padding_mask.dtype, device=padding_mask.device)
        padding_mask_full = torch.cat([prefix_pad, padding_mask], dim=1)

        for blk in self.cross_blocks:
            x = blk(x, mem, c, src_key_padding_mask=padding_mask_full)

        # Strip prefix before output head
        x = x[self.num_cls_prefix:]
        logits = self.output_process(x)  # [B, num_token, seqlen]
        return logits

    def forward(self, ids, y, m_lens):
        '''
        :param ids: (b, n)
        :param y: conditioning (pre-encoded features), (b, ) for cond_mode=action
        :m_lens: (b,)
        :return:
        '''

        bs, ntokens = ids.shape
        device = ids.device

        # Positions that are PADDED are ALL FALSE
        non_pad_mask = lengths_to_mask(m_lens, ntokens) #(b, n)
        ids = torch.where(non_pad_mask, ids, self.pad_id)

        force_mask = False
        if self.cond_mode in ('image_patch', 'image_patch_clsprefix'):
            cond_vector = y.to(device).float()  # [B, 4, 256, 1024]
        elif self.cond_mode == 'uncond':
            cond_vector = torch.zeros(bs, self.latent_dim).float().to(device)
            force_mask = True
        else:
            raise NotImplementedError("Unsupported condition mode!!!")


        '''
        Prepare mask
        '''
        rand_time = uniform((bs,), device=device)
        if getattr(self, 'train_mask_dist', 'cosine') == 'uniform':
            # r ~ U(alpha, 1): masking ratio ablation (MMM-style)
            a = self.train_mask_alpha
            rand_mask_probs = a + (1.0 - a) * rand_time
        else:
            rand_mask_probs = self.noise_schedule(rand_time)
        num_token_masked = (ntokens * rand_mask_probs).round().clamp(min=1)

        batch_randperm = torch.rand((bs, ntokens), device=device).argsort(dim=-1)
        # Positions to be MASKED are ALL TRUE
        mask = batch_randperm < num_token_masked.unsqueeze(-1)

        # Positions to be MASKED must also be NON-PADDED
        mask &= non_pad_mask

        # Note this is our training target, not input
        labels = torch.where(mask, ids, self.mask_id)

        x_ids = ids.clone()

        # Further Apply Bert Masking Scheme
        # Step 1: 10% replace with an incorrect token
        mask_rid = get_mask_subset_prob(mask, 0.1)
        rand_id = torch.randint_like(x_ids, high=self.opt.num_tokens)
        x_ids = torch.where(mask_rid, rand_id, x_ids)
        # Step 2: 90% x 10% replace with correct token, and 90% x 88% replace with mask token
        mask_mid = get_mask_subset_prob(mask & ~mask_rid, 0.88)

        # mask_mid = mask

        x_ids = torch.where(mask_mid, self.mask_id, x_ids)

        logits = self.trans_forward(x_ids, cond_vector, ~non_pad_mask, force_mask)
        ce_loss, pred_id, acc = cal_performance(logits, labels, ignore_index=self.mask_id)

        return ce_loss, pred_id, acc, logits

    def forward_with_cond_scale(self,
                                motion_ids,
                                cond_vector,
                                padding_mask,
                                cond_scale=3,
                                force_mask=False):
        # bs = motion_ids.shape[0]
        # if cond_scale == 1:
        if force_mask:
            return self.trans_forward(motion_ids, cond_vector, padding_mask, force_mask=True)

        logits = self.trans_forward(motion_ids, cond_vector, padding_mask)
        if cond_scale == 1:
            return logits

        aux_logits = self.trans_forward(motion_ids, cond_vector, padding_mask, force_mask=True)

        scaled_logits = aux_logits + (logits - aux_logits) * cond_scale
        return scaled_logits

    @torch.no_grad()
    @eval_decorator
    def generate(self,
                 conds,
                 m_lens,
                 timesteps: int,
                 cond_scale: int,
                 temperature=1,
                 topk_filter_thres=0.9,
                 topp_filter_thres=None,
                 gsample=False,
                 force_mask=False,
                 gen_schedule=None,
                 self_correct=False,
                 reveal_order='confidence'
                 ):
        # print(self.opt.num_quantizers)
        # assert len(timesteps) >= len(cond_scales) == self.opt.num_quantizers

        device = next(self.parameters()).device
        seq_len = max(m_lens)
        batch_size = len(m_lens)
        # inference unmasking-cadence schedule; None -> training default (cosine)
        schedule_fn = self.noise_schedule if gen_schedule is None else SCHEDULES[gen_schedule]

        if self.cond_mode in ('image_patch', 'image_patch_clsprefix'):
            cond_vector = conds.to(device).float()  # [B, 4, 256, 1024]
        elif self.cond_mode == 'uncond':
            cond_vector = torch.zeros(batch_size, self.latent_dim).float().to(device)
        else:
            raise NotImplementedError("Unsupported condition mode!!!")

        padding_mask = ~lengths_to_mask(m_lens, seq_len)
        # print(padding_mask.shape, )

        # Start from all tokens being masked
        ids = torch.where(padding_mask, self.pad_id, self.mask_id)
        scores = torch.where(padding_mask, 1e5, 0.)
        starting_temperature = temperature

        for timestep, steps_until_x0 in zip(torch.linspace(0, 1, timesteps, device=device), reversed(range(timesteps))):
            # 0 < timestep < 1
            rand_mask_prob = schedule_fn(timestep)  # Tensor

            '''
            Maskout, and cope with variable length
            '''
            # fix: the ratio regarding lengths, instead of seq_len
            num_token_masked = torch.round(rand_mask_prob * m_lens).clamp(min=1)  # (b, )

            # select num_token_masked tokens with lowest scores to be masked
            sorted_indices = scores.argsort(
                dim=1)  # (b, k), sorted_indices[i, j] = the index of j-th lowest element in scores on dim=1
            ranks = sorted_indices.argsort(dim=1)  # (b, k), rank[i, j] = the rank (0: lowest) of scores[i, j] on dim=1
            is_mask = (ranks < num_token_masked.unsqueeze(-1))
            ids = torch.where(is_mask, self.mask_id, ids)

            '''
            Preparing input
            '''
            # cond_scale may be a scalar (flat CFG) or a (start, end) pair that is
            # linearly interpolated over the decode steps (BAMM-style hybrid CFG)
            if isinstance(cond_scale, (tuple, list)):
                s0, s1 = cond_scale
                scale_t = s0 + (s1 - s0) * float(timestep)
            else:
                scale_t = cond_scale

            # (b, num_token, seqlen)
            logits = self.forward_with_cond_scale(ids, cond_vector=cond_vector,
                                                  padding_mask=padding_mask,
                                                  cond_scale=scale_t,
                                                  force_mask=force_mask)

            logits = logits.permute(0, 2, 1)  # (b, seqlen, ntoken)
            # print(logits.shape, self.opt.num_tokens)
            # clean low prob token
            filtered_logits = top_k(logits, topk_filter_thres, dim=-1)
            if topp_filter_thres is not None:
                filtered_logits = top_p(filtered_logits, topp_filter_thres, dim=-1)

            '''
            Update ids
            '''
            # if force_mask:
            temperature = starting_temperature
            # else:
            # temperature = starting_temperature * (steps_until_x0 / timesteps)
            # temperature = max(temperature, 1e-4)
            # print(filtered_logits.shape)
            # temperature is annealed, gradually reducing temperature as well as randomness
            if gsample:  # use gumbel_softmax sampling
                # print("1111")
                pred_ids = gumbel_sample(filtered_logits, temperature=temperature, dim=-1)  # (b, seqlen)
            else:  # use multinomial sampling
                # print("2222")
                probs = F.softmax(filtered_logits / temperature, dim=-1)  # (b, seqlen, ntoken)
                # print(temperature, starting_temperature, steps_until_x0, timesteps)
                # print(probs / temperature)
                pred_ids = Categorical(probs).sample()  # (b, seqlen)

            # print(pred_ids.max(), pred_ids.min())
            # if pred_ids.
            ids = torch.where(is_mask, pred_ids, ids)

            '''
            Updating scores
            '''
            probs_without_temperature = logits.softmax(dim=-1)  # (b, seqlen, ntoken)
            scores = probs_without_temperature.gather(2, pred_ids.unsqueeze(dim=-1))  # (b, seqlen, 1)
            scores = scores.squeeze(-1)  # (b, seqlen)

            # The two knobs are ORTHOGONAL and compose:
            #   reveal_order='random' -> WHICH still-masked slots are revealed next is
            #       uniform rather than highest-confidence-first (confidence-first is
            #       mode-seeking: it commits to the most typical tokens and conditions
            #       everything else on them). This is the unbiased any-order ancestral
            #       sampler for a masked / absorbing-diffusion model.
            #   self_correct=True    -> already-accepted slots stay eligible for
            #       re-masking instead of being frozen at 1e5.
            if reveal_order == 'random':
                scores = torch.rand_like(scores)
                scores = (scores.masked_fill(padding_mask, 1e5) if self_correct
                          else scores.masked_fill(~is_mask, 1e5))
            elif self_correct:
                # NON-MONOTONIC ("self-correcting") decode: score every slot by the
                # confidence of the token currently occupying it, so a previously
                # accepted but weak token can be re-masked in a later round. The
                # default below instead freezes accepted tokens forever (standard
                # MaskGIT), which is what commits the sample to its first guesses.
                held = ids.clamp(min=0, max=self.opt.num_tokens - 1)
                scores = probs_without_temperature.gather(2, held.unsqueeze(-1)).squeeze(-1)
                scores = scores.masked_fill(padding_mask, 1e5)
            else:
                # We do not want to re-mask the previously kept tokens, or pad tokens
                scores = scores.masked_fill(~is_mask, 1e5)

        ids = torch.where(padding_mask, -1, ids)
        # print("Final", ids.max(), ids.min())
        return ids

    @torch.no_grad()
    @eval_decorator
    def edit(self,
             conds,
             tokens,
             m_lens,
             timesteps: int,
             cond_scale: int,
             temperature=1,
             topk_filter_thres=0.9,
             gsample=False,
             force_mask=False,
             edit_mask=None,
             padding_mask=None,
             self_correct=False,
             reveal_order='confidence',
             ):
        """Inpainting / conditional editing over an EXISTING token sequence.

        Ported from the upstream MoMask edit() (git HEAD) with our image_patch
        conditioning modes added -- the upstream version only handled
        text/action/uncond and would raise for the modes this project deploys.

        tokens    : (b, n) the sequence to edit; positions outside edit_mask are KEPT
        edit_mask : (b, n) bool, True where tokens may be RESAMPLED.
                    None -> "mask-free" mode: every non-pad position is re-decoded
                    from the given tokens with a low fixed re-mask rate (0.16),
                    i.e. gentle refinement of a whole sequence rather than a hole.

        Fixing a prefix (edit_mask False on the prefix, True after) is how a
        partial tower / 2D floorplan is completed; a middle hole gives inpainting.
        """
        assert edit_mask.shape == tokens.shape if edit_mask is not None else True
        device = next(self.parameters()).device
        seq_len = tokens.shape[1]

        if self.cond_mode in ('image_patch', 'image_patch_clsprefix'):
            # patch conditioning is consumed inside trans_forward as cross-attn
            # memory, so it is passed through unchanged: (b, 4, 256, 1024)
            cond_vector = conds.to(device).float()

        elif self.cond_mode == 'uncond':
            cond_vector = torch.zeros(tokens.shape[0], self.latent_dim).float().to(device)
        else:
            raise NotImplementedError("Unsupported condition mode!!!")

        if padding_mask is None:
            padding_mask = ~lengths_to_mask(m_lens, seq_len)

        if edit_mask is None:
            mask_free = True
            ids = torch.where(padding_mask, self.pad_id, tokens)
            edit_mask = torch.ones_like(padding_mask) & ~padding_mask
            edit_len = edit_mask.sum(dim=-1)
            scores = torch.where(edit_mask, 0., 1e5)
        else:
            mask_free = False
            edit_mask = edit_mask & ~padding_mask
            edit_len = edit_mask.sum(dim=-1)
            ids = torch.where(edit_mask, self.mask_id, tokens)
            scores = torch.where(edit_mask, 0., 1e5)
        starting_temperature = temperature

        for timestep, steps_until_x0 in zip(torch.linspace(0, 1, timesteps, device=device),
                                            reversed(range(timesteps))):
            rand_mask_prob = 0.16 if mask_free else self.noise_schedule(timestep)
            num_token_masked = torch.round(rand_mask_prob * edit_len).clamp(min=1)

            sorted_indices = scores.argsort(dim=1)
            ranks = sorted_indices.argsort(dim=1)
            is_mask = (ranks < num_token_masked.unsqueeze(-1))
            ids = torch.where(is_mask, self.mask_id, ids)

            logits = self.forward_with_cond_scale(ids, cond_vector=cond_vector,
                                                  padding_mask=padding_mask,
                                                  cond_scale=cond_scale,
                                                  force_mask=force_mask)
            logits = logits.permute(0, 2, 1)                  # (b, seqlen, ntoken)
            filtered_logits = top_k(logits, topk_filter_thres, dim=-1)

            temperature = starting_temperature
            if gsample:
                pred_ids = gumbel_sample(filtered_logits, temperature=temperature, dim=-1)
            else:
                probs = F.softmax(filtered_logits / temperature, dim=-1)
                pred_ids = Categorical(probs).sample()

            ids = torch.where(is_mask, pred_ids, ids)

            probs_without_temperature = logits.softmax(dim=-1)
            scores = probs_without_temperature.gather(2, pred_ids.unsqueeze(dim=-1)).squeeze(-1)

            # Same two decode-order knobs as generate(), with one crucial difference:
            # eligibility is capped by edit_mask, so a FROZEN token is pinned at 1e5 in
            # every branch and can never be re-masked. Token-exactness of the given
            # region is therefore preserved regardless of these settings.
            if reveal_order == 'random':
                scores = torch.rand_like(scores)
                scores = scores.masked_fill(~edit_mask if self_correct else ~is_mask, 1e5)
            elif self_correct:
                held = ids.clamp(min=0, max=self.opt.num_tokens - 1)
                scores = probs_without_temperature.gather(2, held.unsqueeze(-1)).squeeze(-1)
                scores = scores.masked_fill(~edit_mask, 1e5)
            else:
                # never re-mask kept tokens (or pads)
                scores = scores.masked_fill(~edit_mask, 1e5) if mask_free \
                    else scores.masked_fill(~is_mask, 1e5)

        ids = torch.where(padding_mask, -1, ids)
        return ids

