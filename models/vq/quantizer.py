"""Quantization layer for the 1D tower VQ-VAE.

Two levels live here:
  * the quantizers themselves -- `IBQQuantize` (the released tokenizer) and
    `QuantizeEMAReset` (the residual-VQ ablation arm): vector -> code lookup,
    codebook update, perplexity;
  * `QuantizerStack`, the container that holds `num_quantizers` of them and
    applies them to successive residuals.

**The released GenArch tokenizer is `ibq_4x_nc1024_nq1`: ONE IBQ codebook
(1024x512) plus `logit_scale`.** At num_quantizers=1 the stack's residual loop
runs a single iteration -- there is no Q1/Q2, no residual is ever taken, and the
mask transformer consumes `code_idx[..., 0]` only. The multi-quantizer path is
kept because the paper's tokenizer ablation includes a 3-layer residual-VQ arm
(`rvq_v2_4x_nc1024_nq3`).

Checkpoint keys are `quantizer.layers.<i>.codebook`, from the `self.layers`
ModuleList. Do not rename that attribute: every trained tokenizer -- and every
mask transformer that depends on one -- loads through those exact keys.

(Note: `resnet.py`'s Resnet1D is the conv backbone of the encoder/decoder in
encdec.py. Unrelated to residual quantization despite the similar name.)
"""
import random
from random import randrange

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat, reduce, pack, unpack


#Borrow from vector_quantize_pytorch

def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))

def gumbel_sample(
    logits,
    temperature = 1.,
    stochastic = False,
    dim = -1,
    training = True
):

    if training and stochastic and temperature > 0:
        sampling_logits = (logits / temperature) + gumbel_noise(logits)
    else:
        sampling_logits = logits

    ind = sampling_logits.argmax(dim = dim)

    return ind

class QuantizeEMAReset(nn.Module):
    def __init__(self, nb_code, code_dim, args):
        super(QuantizeEMAReset, self).__init__()
        self.nb_code = nb_code
        self.code_dim = code_dim
        self.mu = args.mu  ##TO_DO
        self.reset_codebook()

    def reset_codebook(self):
        self.init = False
        self.code_sum = None
        self.code_count = None
        self.register_buffer('codebook', torch.zeros(self.nb_code, self.code_dim, requires_grad=False))

    def _tile(self, x):
        nb_code_x, code_dim = x.shape
        if nb_code_x < self.nb_code:
            n_repeats = (self.nb_code + nb_code_x - 1) // nb_code_x
            std = 0.01 / np.sqrt(code_dim)
            out = x.repeat(n_repeats, 1)
            out = out + torch.randn_like(out) * std
        else:
            out = x
        return out

    def init_codebook(self, x):
        out = self._tile(x)
        self.codebook = out[:self.nb_code]
        self.code_sum = self.codebook.clone()
        self.code_count = torch.ones(self.nb_code, device=self.codebook.device)
        self.init = True

    def quantize(self, x, sample_codebook_temp=0.):
        # N X C -> C X N
        k_w = self.codebook.t()
        # x: NT X C
        # NT X N
        distance = torch.sum(x ** 2, dim=-1, keepdim=True) - \
                   2 * torch.matmul(x, k_w) + \
                   torch.sum(k_w ** 2, dim=0, keepdim=True)  # (N * L, b)

        # code_idx = torch.argmin(distance, dim=-1)

        code_idx = gumbel_sample(-distance, dim = -1, temperature = sample_codebook_temp, stochastic=True, training = self.training)

        return code_idx

    def dequantize(self, code_idx):
        x = F.embedding(code_idx, self.codebook)
        return x
    
    def get_codebook_entry(self, indices):
        return self.dequantize(indices).permute(0, 2, 1)

    @torch.no_grad()
    def compute_perplexity(self, code_idx):
        # Calculate new centres
        code_onehot = torch.zeros(self.nb_code, code_idx.shape[0], device=code_idx.device)  # nb_code, N * L
        code_onehot.scatter_(0, code_idx.view(1, code_idx.shape[0]), 1)

        code_count = code_onehot.sum(dim=-1)  # nb_code
        prob = code_count / torch.sum(code_count)
        perplexity = torch.exp(-torch.sum(prob * torch.log(prob + 1e-7)))
        return perplexity

    @torch.no_grad()
    def update_codebook(self, x, code_idx):
        code_onehot = torch.zeros(self.nb_code, x.shape[0], device=x.device) # nb_code, N * L
        code_onehot.scatter_(0, code_idx.view(1, x.shape[0]), 1)

        code_sum = torch.matmul(code_onehot, x) # nb_code, c
        code_count = code_onehot.sum(dim=-1) # nb_code

        out = self._tile(x)
        code_rand = out[:self.nb_code]

        # Update centres
        self.code_sum = self.mu * self.code_sum + (1. - self.mu) * code_sum
        self.code_count = self.mu * self.code_count + (1. - self.mu) * code_count

        usage = (self.code_count.view(self.nb_code, 1) >= 1.0).float()
        code_update = self.code_sum.view(self.nb_code, self.code_dim) / self.code_count.view(self.nb_code, 1)
        self.codebook = usage * code_update + (1-usage) * code_rand


        prob = code_count / torch.sum(code_count)
        perplexity = torch.exp(-torch.sum(prob * torch.log(prob + 1e-7)))

        return perplexity

    def preprocess(self, x):
        # NCT -> NTC -> [NT, C]
        # x = x.permute(0, 2, 1).contiguous()
        # x = x.view(-1, x.shape[-1])
        x = rearrange(x, 'n c t -> (n t) c')
        return x

    def forward(self, x, return_idx=False, temperature=0.):
        N, width, T = x.shape

        x = self.preprocess(x)
        if self.training and not self.init:
            self.init_codebook(x)

        code_idx = self.quantize(x, temperature)
        x_d = self.dequantize(code_idx)

        if self.training:
            perplexity = self.update_codebook(x, code_idx)
        else:
            perplexity = self.compute_perplexity(code_idx)

        commit_loss = F.mse_loss(x, x_d.detach()) # It's right. the t2m-gpt paper is wrong on embed loss and commitment loss.

        # Passthrough
        x_d = x + (x_d - x).detach()

        # Postprocess
        x_d = x_d.view(N, T, -1).permute(0, 2, 1).contiguous()
        code_idx = code_idx.view(N, T).contiguous()
        # print(code_idx[0])
        if return_idx:
            return x_d, code_idx, commit_loss, perplexity
        return x_d, commit_loss, perplexity
    
class IBQQuantize(nn.Module):
    """Index Backpropagation Quantization (arXiv 2412.02692).

    Drop-in replacement for QuantizeEMAReset (same forward signature) that swaps
    NN-lookup + EMA codebook for a *learnable* codebook trained jointly with the
    encoder via a straight-through estimator on the categorical index distribution.

    - logits = cosine similarity between encoder output and every code, scaled by a
      CLIP-style learnable temperature (so the softmax sees all codes).
    - forward pass decodes the hard (argmax) code; backward pass flows gradient
      through the soft softmax to *every* code embedding + the encoder.
    - entropy bonus (MAGVIT2-style): minimise per-sample entropy (sharp picks) and
      maximise aggregate-usage entropy (uniform codebook utilisation, anti-collapse).

    Returns its full, internally-weighted aux loss in the commit slot; the trainer
    adds it with weight 1.0 (no extra opt.commit scaling) when --quantizer ibq.
    """
    def __init__(self, nb_code, code_dim, args):
        super(IBQQuantize, self).__init__()
        self.nb_code = nb_code
        self.code_dim = code_dim
        self.commit = getattr(args, 'commit', 0.02)
        self.entropy_weight = getattr(args, 'ibq_entropy_weight', 0.1)
        # learnable codebook (vs EMA buffer) — jointly optimised by backprop
        self.codebook = nn.Parameter(torch.randn(nb_code, code_dim) * (1.0 / np.sqrt(code_dim)))
        # CLIP-style learnable logit scale for the cosine-sim softmax (init 1/0.07)
        self.logit_scale = nn.Parameter(torch.tensor(float(np.log(1.0 / 0.07))))

    def preprocess(self, x):
        # NCT -> [NT, C]
        return rearrange(x, 'n c t -> (n t) c')

    def _logits(self, x):
        z = F.normalize(x, dim=-1)
        cb = F.normalize(self.codebook, dim=-1)
        scale = self.logit_scale.exp().clamp(max=100.0)
        return (z @ cb.t()) * scale  # (NT, nb_code)

    def quantize(self, x, sample_codebook_temp=0.):
        return self._logits(x).argmax(dim=-1)

    def dequantize(self, code_idx):
        return F.embedding(code_idx, self.codebook)

    def get_codebook_entry(self, indices):
        return self.dequantize(indices).permute(0, 2, 1)

    @torch.no_grad()
    def compute_perplexity(self, code_idx):
        code_onehot = torch.zeros(self.nb_code, code_idx.shape[0], device=code_idx.device)
        code_onehot.scatter_(0, code_idx.view(1, code_idx.shape[0]), 1)
        prob = code_onehot.sum(dim=-1) / code_idx.shape[0]
        return torch.exp(-torch.sum(prob * torch.log(prob + 1e-7)))

    def forward(self, x, return_idx=False, temperature=0.):
        N, width, T = x.shape
        x = self.preprocess(x)

        logits = self._logits(x)
        p = logits.softmax(dim=-1)           # categorical over all codes
        code_idx = p.argmax(dim=-1)          # hard selection
        p_hard = F.one_hot(code_idx, self.nb_code).type_as(p)

        # straight-through on the index distribution: forward = hard code,
        # backward = soft softmax -> gradient reaches all codes + encoder.
        p_ste = p_hard + p - p.detach()
        x_d = p_ste @ self.codebook          # (NT, C); forward == codebook[code_idx]

        # commitment: pull encoder output toward the selected code
        commit_loss = F.mse_loss(x, x_d.detach())

        # entropy bonus (minimise => sharp per-sample, uniform aggregate usage)
        log_p = torch.log(p.clamp(min=1e-10))
        sample_entropy = -(p * log_p).sum(dim=-1).mean()
        avg_p = p.mean(dim=0)
        codebook_entropy = -(avg_p * torch.log(avg_p.clamp(min=1e-10))).sum()
        entropy_loss = sample_entropy - codebook_entropy

        aux_loss = self.commit * commit_loss + self.entropy_weight * entropy_loss

        perplexity = self.compute_perplexity(code_idx)

        # Postprocess
        x_d = x_d.view(N, T, -1).permute(0, 2, 1).contiguous()
        code_idx = code_idx.view(N, T).contiguous()
        if return_idx:
            return x_d, code_idx, aux_loss, perplexity
        return x_d, aux_loss, perplexity


class QuantizeEMA(QuantizeEMAReset):
    @torch.no_grad()
    def update_codebook(self, x, code_idx):
        code_onehot = torch.zeros(self.nb_code, x.shape[0], device=x.device) # nb_code, N * L
        code_onehot.scatter_(0, code_idx.view(1, x.shape[0]), 1)

        code_sum = torch.matmul(code_onehot, x) # nb_code, c
        code_count = code_onehot.sum(dim=-1) # nb_code

        # Update centres
        self.code_sum = self.mu * self.code_sum + (1. - self.mu) * code_sum
        self.code_count = self.mu * self.code_count + (1. - self.mu) * code_count

        usage = (self.code_count.view(self.nb_code, 1) >= 1.0).float()
        code_update = self.code_sum.view(self.nb_code, self.code_dim) / self.code_count.view(self.nb_code, 1)
        self.codebook = usage * code_update + (1-usage) * self.codebook

        prob = code_count / torch.sum(code_count)
        perplexity = torch.exp(-torch.sum(prob * torch.log(prob + 1e-7)))

        return perplexity


class QuantizerStack(nn.Module):
    def __init__(
        self,
        num_quantizers,
        shared_codebook=False,
        quantize_dropout_prob=0.5,
        quantize_dropout_cutoff_index=0,
        **kwargs
    ):
        super().__init__()
        self.num_quantizers = num_quantizers

        quantizer_type = getattr(kwargs.get('args', None), 'quantizer', 'ema')
        QuantClass = IBQQuantize if quantizer_type == 'ibq' else QuantizeEMAReset
        if shared_codebook:
            layer = QuantClass(**kwargs)
            self.layers = nn.ModuleList([layer for _ in range(num_quantizers)])
        else:
            self.layers = nn.ModuleList([QuantClass(**kwargs) for _ in range(num_quantizers)])

        assert quantize_dropout_cutoff_index >= 0 and quantize_dropout_prob >= 0
        self.quantize_dropout_cutoff_index = quantize_dropout_cutoff_index
        # Only meaningful at num_quantizers > 1 (drops trailing residual layers
        # during training so the leading ones stay usable alone).
        self.quantize_dropout_prob = quantize_dropout_prob

    @property
    def codebooks(self):
        return torch.stack([layer.codebook for layer in self.layers], dim=0)  # 'q c d'

    def get_codes_from_indices(self, indices):
        """indices 'b n q' -> all_codes 'q b n d'. Used by VQVAE.forward_decoder."""
        batch, quantize_dim = indices.shape[0], indices.shape[-1]

        # Coarse indices (fewer layers than the stack) are padded with -1 and
        # masked out below, so a dropout-trained stack can decode from a prefix.
        if quantize_dim < self.num_quantizers:
            indices = F.pad(indices, (0, self.num_quantizers - quantize_dim), value=-1)

        codebooks = repeat(self.codebooks, 'q c d -> q b c d', b=batch)
        gather_indices = repeat(indices, 'b n q -> q b n d', d=codebooks.shape[-1])

        mask = gather_indices == -1.
        gather_indices = gather_indices.masked_fill(mask, 0)  # dummy code, masked after gather
        all_codes = codebooks.gather(2, gather_indices)
        return all_codes.masked_fill(mask, 0.)  # 'q b n d'

    def get_codebook_entry(self, indices):
        latent = torch.sum(self.get_codes_from_indices(indices), dim=0)  # 'b n d'
        return latent.permute(0, 2, 1)

    def forward(self, x, return_all_codes=False, sample_codebook_temp=None):
        num_quant, device = self.num_quantizers, x.device

        quantized_out = 0.
        residual = x
        all_losses, all_indices, all_perplexity = [], [], []

        # Training-time dropout of trailing residual layers. Inert at num_quant=1.
        should_quantize_dropout = self.training and random.random() < self.quantize_dropout_prob
        start_drop_quantize_index = num_quant
        if should_quantize_dropout:
            start_drop_quantize_index = randrange(self.quantize_dropout_cutoff_index, num_quant)
            null_indices = torch.full([x.shape[0], x.shape[-1]], -1., device=device, dtype=torch.long)

        for quantizer_index, layer in enumerate(self.layers):
            if should_quantize_dropout and quantizer_index > start_drop_quantize_index:
                all_indices.append(null_indices)
                continue

            quantized, *rest = layer(residual, return_idx=True, temperature=sample_codebook_temp)
            residual -= quantized.detach()
            quantized_out += quantized

            embed_indices, loss, perplexity = rest
            all_indices.append(embed_indices)
            all_losses.append(loss)
            all_perplexity.append(perplexity)

        all_indices = torch.stack(all_indices, dim=-1)
        all_losses = sum(all_losses) / len(all_losses)
        all_perplexity = sum(all_perplexity) / len(all_perplexity)

        ret = (quantized_out, all_indices, all_losses, all_perplexity)
        if return_all_codes:
            ret = (*ret, self.get_codes_from_indices(all_indices))
        return ret

    def quantize(self, x, return_latent=False):
        all_indices, all_codes = [], []
        quantized_out = 0.
        residual = x
        for layer in self.layers:
            quantized, *rest = layer(residual, return_idx=True)
            residual = residual - quantized.detach()
            quantized_out = quantized_out + quantized
            embed_indices, loss, perplexity = rest
            all_indices.append(embed_indices)
            all_codes.append(quantized)

        code_idx = torch.stack(all_indices, dim=-1)
        if return_latent:
            return code_idx, torch.stack(all_codes, dim=0)
        return code_idx
