import time
import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from contextlib import nullcontext
from collections import defaultdict
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from collections import OrderedDict
from utils.utils import print_current_loss
from utils.voxel import grid_snap, voxel_iou
from utils.vq_decode import decode_windowed, probe_window_tokens
from os.path import join as pjoin
from models.mask_transformer.tools import *

from einops import rearrange, repeat

def def_value():
    return 0.0

class MaskTransformerTrainer:
    def __init__(self, args, t2m_transformer, vq_model):
        self.opt = args
        self.t2m_transformer = t2m_transformer
        self.vq_model = vq_model
        self.device = args.device
        self.vq_model.eval()
        # Weight for coordinate reconstruction loss alongside cross-entropy
        self.coord_loss_weight = getattr(args, 'coord_loss_weight', 0.1)
        self.ce_loss_weight = getattr(args, 'ce_loss_weight', 1.0)
        # Rank 0 owns logging, checkpointing, and IoU eval under DDP
        self.is_main = getattr(args, 'is_main_process', True)

        if args.is_train and self.is_main:
            self.logger = SummaryWriter(args.log_dir)
        else:
            self.logger = None


    def update_lr_warm_up(self, nb_iter, warm_up_iter, lr):

        current_lr = lr * (nb_iter + 1) / (warm_up_iter + 1)
        for param_group in self.opt_t2m_transformer.param_groups:
            param_group["lr"] = current_lr

        return current_lr


    # _grid_snap / _voxel_iou moved to utils/voxel.py, use grid_snap / voxel_iou directly.
    _grid_snap = staticmethod(grid_snap)
    _voxel_iou = staticmethod(voxel_iou)

    def compute_val_iou(self, eval_towers, mean, std,
                        num_steps=10, cond_scale=1.0, temperature=0.2,
                        decode_mode='windowed'):
        """
        Run full iterative generation on a fixed set of val towers and
        compute mean voxel IoU vs ground truth.

        decode_mode:
          'windowed', decode in fixed 64-block (window_tokens) chunks to match the
                       VQ training window. Required for RVQ at 4x+: whole-seq decode
                       collapses due to strided-conv boundary artifacts (eval_rvq_ceiling.py).
          'whole', decode the full token sequence in one forward_decoder call.
                       Correct for IBQ, whose decoder generalises to full sequences
                       (and is its deployment decode); collapses for RVQ.

        eval_towers: list of dicts {'cond': Tensor[1,...], 'motion_norm': Tensor[1,T,3], 'm_len': int}
        mean, std:   np.ndarray [3] for unnormalising coordinates
        Returns:     float, mean IoU across all eval towers
        """
        model = (self.t2m_transformer.module
                 if hasattr(self.t2m_transformer, 'module')
                 else self.t2m_transformer)
        model.eval()
        self.vq_model.eval()

        # Build batch
        if model.cond_mode == 'uncond':
            conds = torch.zeros(len(eval_towers), model.latent_dim).to(self.device)
        else:
            conds = torch.cat([t['cond'] for t in eval_towers], dim=0).to(self.device)

        m_lens_blocks = [t['m_len'] for t in eval_towers]
        m_lens_tok = torch.tensor([ml // 4 for ml in m_lens_blocks]).to(self.device)
        motions_norm = torch.cat([t['motion_norm'] for t in eval_towers], dim=0).cpu().numpy()

        with torch.no_grad():
            # Generate token sequences [N, max_tok]
            gen_ids = model.generate(
                conds, m_lens_tok,
                timesteps=num_steps,
                cond_scale=cond_scale,
                temperature=temperature,
            )
            # Clamp to valid codebook range (mask_id / pad_id are out-of-range for the codebook)
            nb_code = self.vq_model.quantizer.layers[0].nb_code
            gen_ids = gen_ids.clamp(0, nb_code - 1)

            if decode_mode == 'whole':
                # Whole-sequence decode (IBQ): single forward_decoder over all tokens.
                coords_norm = self.vq_model.forward_decoder(
                    gen_ids.unsqueeze(-1)).cpu().numpy()
            else:
                # Windowed decode: chunk along token axis to match the VQ training window.
                # Probe the encoder once to derive tokens-per-64-blocks for the tokenizer.
                if not hasattr(self, '_decode_window_tokens'):
                    self._decode_window_tokens = probe_window_tokens(
                        self.vq_model, motions_norm.shape[-1], self.device)
                coords_norm = decode_windowed(
                    self.vq_model, gen_ids.unsqueeze(-1),
                    window_tokens=self._decode_window_tokens,
                ).cpu().numpy()

        ious = []
        for i, m_len in enumerate(m_lens_blocks):
            n_frames = (m_len // 4) * 4           # decoded frame count (tokens × 4)
            n_cmp = min(n_frames, m_len)           # compare only overlapping frames

            gen_coords = coords_norm[i, :n_cmp] * std + mean
            gt_coords  = motions_norm[i, :n_cmp] * std + mean
            ious.append(self._voxel_iou(gen_coords, gt_coords))

        return float(np.mean(ious))

    def compute_coord_loss(self, logits, code_idx, motion_norm, m_lens_blocks):
        """
        Differentiable coordinate reconstruction loss.
        Soft-looks up quantizer-0 codes from logits, adds hard codes from quantizers 1..Q-1,
        decodes through the VQ decoder, and computes MSE vs ground-truth coordinates.

        logits:        [B, nb_code, ntokens]
        code_idx:      [B, ntokens, Q]
        motion_norm:   [B, max_motion_length, 3]  (normalised coordinates)
        m_lens_blocks: [B]  original block counts (before //4)
        """
        vq = self.vq_model
        B, nb_code, ntokens = logits.shape

        # Soft lookup for quantizer 0, gradient flows logits → probs → soft_q0
        cb0 = vq.quantizer.layers[0].codebook  # [nb_code, code_dim]
        probs = F.softmax(logits.permute(0, 2, 1), dim=-1)  # [B, ntokens, nb_code]
        soft_q0 = probs @ cb0                               # [B, ntokens, code_dim]

        # Hard lookup for quantizers 1..Q-1 (no gradient needed)
        sum_codes = soft_q0
        for q in range(1, vq.quantizer.num_quantizers):
            cb_q = vq.quantizer.layers[q].codebook          # [nb_code, code_dim]
            idx_q = code_idx[:, :, q]                       # [B, ntokens]
            hard_q = cb_q[idx_q].detach()                   # [B, ntokens, code_dim]
            sum_codes = sum_codes + hard_q

        # Decode: [B, code_dim, ntokens] → [B, ntokens*4, 3]
        # (Decoder.forward already does the final permute to time-first format)
        coords_pred = vq.decoder(sum_codes.permute(0, 2, 1))  # [B, ntokens*4, 3]

        decoded_len = coords_pred.shape[1]
        coords_gt = motion_norm[:, :decoded_len, :]

        # Valid (non-padded) position mask
        valid = (torch.arange(decoded_len, device=coords_pred.device).unsqueeze(0)
                 < m_lens_blocks.unsqueeze(1))

        if valid.sum() == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        return F.mse_loss(coords_pred[valid], coords_gt[valid])

    def _compute_losses(self, batch_data):
        """Run forward and return loss TENSORS (not floats). Shared by forward() and gradient_probe()."""
        conds, motion, m_lens = batch_data
        motion = motion.detach().float().to(self.device)
        m_lens = m_lens.detach().long().to(self.device)

        # (b, n, q)
        code_idx, _ = self.vq_model.encode(motion)
        m_lens_blocks = m_lens.clone()   # block counts, before downsampling
        m_lens = m_lens // 4             # token counts

        # Pre-encode text (skip in uncond mode, transformer ignores conditioning)
        model = self.t2m_transformer.module if hasattr(self.t2m_transformer, 'module') else self.t2m_transformer
        if getattr(model, 'cond_mode', 'text') == 'uncond':
            cond_vector = torch.zeros(len(m_lens), model.latent_dim).float().to(self.device)
        elif isinstance(conds, (list, tuple)) and isinstance(conds[0], str):
            with torch.no_grad():
                cond_vector = model.encode_text(conds)
        else:
            cond_vector = conds.to(self.device).float()

        ce_loss, _pred_ids, acc, logits = self.t2m_transformer(code_idx[..., 0], cond_vector, m_lens)
        if ce_loss.dim() > 0:
            ce_loss = ce_loss.mean()
        if torch.is_tensor(acc) and acc.dim() > 0:
            acc = acc.mean()

        # Coordinate reconstruction loss (differentiable via soft codebook lookup)
        if self.coord_loss_weight > 0:
            if isinstance(logits, (list, tuple)):
                logits = torch.cat(logits, dim=0)
            coord_loss = self.compute_coord_loss(logits, code_idx, motion, m_lens_blocks)
        else:
            coord_loss = torch.tensor(0.0, device=self.device)

        return ce_loss, coord_loss, acc

    def forward(self, batch_data):
        ce_loss, coord_loss, acc = self._compute_losses(batch_data)
        total_loss = self.ce_loss_weight * ce_loss + self.coord_loss_weight * coord_loss
        return (total_loss,
                acc.item() if torch.is_tensor(acc) else acc,
                ce_loss.item(),
                coord_loss.item() if torch.is_tensor(coord_loss) else coord_loss)

    def _grad_l2_norm(self):
        total = 0.0
        for p in self.t2m_transformer.parameters():
            if p.grad is not None:
                total += p.grad.detach().pow(2).sum().item()
        return total ** 0.5

    def gradient_probe(self, batch_data, it):
        """Backprop CE and coord losses separately to measure their gradient contributions.
        Wrapped in DDP no_sync to avoid wasting NCCL bandwidth on these throwaway gradients.
        Logs grad norms (and ratio) to TensorBoard from rank 0.
        """
        was_training = self.t2m_transformer.training
        self.t2m_transformer.train()  # match the dropout/etc state of normal updates

        no_sync_ctx = self.t2m_transformer.no_sync() if hasattr(self.t2m_transformer, 'no_sync') else nullcontext()
        with no_sync_ctx:
            # Fresh forward, we need separate graphs per loss for separate backwards
            ce_loss_t, coord_loss_t, _ = self._compute_losses(batch_data)

            # CE-only grad (with the actual training weight applied)
            self.opt_t2m_transformer.zero_grad()
            (self.ce_loss_weight * ce_loss_t).backward(retain_graph=True)
            grad_ce = self._grad_l2_norm()

            # Coord-only grad (with the actual training weight applied)
            self.opt_t2m_transformer.zero_grad()
            (self.coord_loss_weight * coord_loss_t).backward()
            grad_coord = self._grad_l2_norm()

            # Don't let probe gradients leak into the next optimizer step
            self.opt_t2m_transformer.zero_grad()

        if not was_training:
            self.t2m_transformer.eval()

        if self.is_main and self.logger is not None:
            ratio = grad_coord / max(grad_ce, 1e-12)
            self.logger.add_scalar('Train/grad_norm_ce', grad_ce, it)
            self.logger.add_scalar('Train/grad_norm_coord', grad_coord, it)
            self.logger.add_scalar('Train/grad_ratio_coord_over_ce', ratio, it)
            print(f'[grad probe @ it {it}] |∇CE|={grad_ce:.4f}  |∇coord|={grad_coord:.4f}  '
                  f'ratio coord/ce = {ratio:.3f}')

    def update(self, batch_data):
        total_loss, acc, ce_loss, coord_loss = self.forward(batch_data)

        self.opt_t2m_transformer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.t2m_transformer.parameters(), max_norm=1.0)
        self.opt_t2m_transformer.step()
        self.scheduler.step()

        return total_loss.item(), acc, ce_loss, coord_loss

    def save(self, file_name, ep, total_it, best_acc=None, best_iou=None):
        if not self.is_main:
            return
        model = self.t2m_transformer.module if hasattr(self.t2m_transformer, 'module') else self.t2m_transformer
        t2m_trans_state_dict = model.state_dict()
        clip_weights = [e for e in t2m_trans_state_dict.keys() if e.startswith('clip_model.')]
        for e in clip_weights:
            del t2m_trans_state_dict[e]
        state = {
            't2m_transformer': t2m_trans_state_dict,
            'opt_t2m_transformer': self.opt_t2m_transformer.state_dict(),
            'scheduler':self.scheduler.state_dict(),
            'ep': ep,
            'total_it': total_it,
            'best_acc': best_acc,
            'best_iou': best_iou,
        }
        torch.save(state, file_name)

    def resume(self, model_dir):
        checkpoint = torch.load(model_dir, map_location=self.device)
        model = self.t2m_transformer.module if hasattr(self.t2m_transformer, 'module') else self.t2m_transformer
        # Handle DataParallel: strip 'module.' prefix from checkpoint keys
        state_dict = checkpoint['t2m_transformer']
        if any(k.startswith('module.') for k in state_dict.keys()):
            state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}
        # Remove SBERT weights from checkpoint (loaded fresh each time)
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith('sbert_model.')}
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        assert len(unexpected_keys) == 0, f"Unexpected keys: {unexpected_keys[:5]}"
        assert all([k.startswith('clip_model.') or k.startswith('sbert_model.') for k in missing_keys])

        try:
            self.opt_t2m_transformer.load_state_dict(checkpoint['opt_t2m_transformer']) # Optimizer

            self.scheduler.load_state_dict(checkpoint['scheduler']) # Scheduler
        except:
            print('Resume wo optimizer')
        best_acc = checkpoint.get('best_acc')
        best_iou = checkpoint.get('best_iou')
        return checkpoint['ep'], checkpoint['total_it'], best_acc, best_iou

    def train(self, train_loader, val_loader,
              eval_towers=None, mean=None, std=None, eval_iou_every=50,
              train_sampler=None):
        """
        eval_towers:    list of dicts {'cond', 'motion_norm', 'm_len'} for periodic IoU eval.
                        Build from the first N val samples in the training script.
        mean, std:      np.ndarray for unnormalising coordinates (required when eval_towers set).
        eval_iou_every: run IoU eval every this many epochs (default 50).
        train_sampler:  DistributedSampler instance (required for DDP); set_epoch() is called each epoch.
        """
        self.t2m_transformer.to(self.device)
        self.vq_model.to(self.device)

        # DDP wrap (only the trainable transformer; VQ stays local in eval mode)
        use_ddp = getattr(self.opt, 'use_ddp', False)
        if use_ddp:
            # broadcast_buffers=False: the only buffer is PositionalEncoding.pe, same on every rank
            # by construction (deterministic init, frozen) and stored as a non-contiguous view that
            # NCCL refuses to broadcast in-place.
            self.t2m_transformer = DDP(self.t2m_transformer,
                                        device_ids=[self.opt.local_rank],
                                        find_unused_parameters=False,
                                        broadcast_buffers=False)
            if self.is_main:
                print(f'Using DDP on {dist.get_world_size()} GPUs')
        elif self.is_main:
            print('Single-GPU training (no DDP)')

        # Resolve fraction-based milestones (e.g. --milestone_fractions 0.75 0.92) into iter targets
        # using the actual iters/epoch, survives wrong iters/epoch estimates that bit v2 and v4.
        fractions = getattr(self.opt, 'milestone_fractions', None)
        if fractions:
            total_iters_plan = self.opt.max_epoch * len(train_loader)
            self.opt.milestones = [int(f * total_iters_plan) for f in fractions]
            if self.is_main:
                print(f'milestone_fractions {fractions} → iter milestones {self.opt.milestones} '
                      f'(total_iters={total_iters_plan}, iters/epoch={len(train_loader)})')

        self.opt_t2m_transformer = optim.AdamW(self.t2m_transformer.parameters(), betas=(0.9, 0.99), lr=self.opt.lr, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.opt_t2m_transformer,
                                                        milestones=self.opt.milestones,
                                                        gamma=self.opt.gamma)

        epoch = 0
        it = 0
        resumed_best_acc = None
        resumed_best_iou = None

        if self.opt.is_continue:
            model_dir = pjoin(self.opt.model_dir, 'latest.tar')  # TODO
            epoch, it, resumed_best_acc, resumed_best_iou = self.resume(model_dir)
            print("Load model epoch:%d iterations:%d"%(epoch, it))
            if resumed_best_iou is not None and self.is_main:
                print(f'Restored best_iou={resumed_best_iou:.4f}, best_acc={resumed_best_acc}')
            if getattr(self.opt, 'reset_optimizer', False):
                # Warm restart: discard stale Adam moments, fresh scheduler at new LR.
                # it is reset so the warmup ramp fires cleanly from 0 → opt.lr.
                # For fraction-based milestones, recompute against REMAINING iters
                # (max_epoch - epoch) × iters/epoch, initial calc used full max_epoch which
                # is wrong when resuming mid-run.
                if fractions:
                    remaining_iters = (self.opt.max_epoch - epoch) * len(train_loader)
                    self.opt.milestones = [int(f * remaining_iters) for f in fractions]
                    if self.is_main:
                        print(f'warm-restart milestones recomputed: {fractions} × remaining {remaining_iters} '
                              f'iters → {self.opt.milestones} (resuming from epoch {epoch})')
                self.opt_t2m_transformer = optim.AdamW(
                    self.t2m_transformer.parameters(), betas=(0.9, 0.99),
                    lr=self.opt.lr, weight_decay=1e-5)
                self.scheduler = optim.lr_scheduler.MultiStepLR(
                    self.opt_t2m_transformer, milestones=self.opt.milestones, gamma=self.opt.gamma)
                it = 0
                print(f'Optimizer reset. Fresh AdamW at lr={self.opt.lr}, milestones={self.opt.milestones}')

        start_time = time.time()
        total_iters = self.opt.max_epoch * len(train_loader)
        if self.is_main:
            print(f'Total Epochs: {self.opt.max_epoch}, Total Iters: {total_iters}')
            print('Iters Per Epoch, Training: %04d, Validation: %03d' % (len(train_loader), len(val_loader)))
            if eval_towers:
                print(f'IoU eval: {len(eval_towers)} fixed val towers every {eval_iou_every} epochs')
        logs = defaultdict(def_value, OrderedDict())

        best_fid, best_div, best_top1, best_top2, best_top3, best_matching = 100, 100, 0, 0, 0, 100
        best_acc = resumed_best_acc if resumed_best_acc is not None else 0.
        best_iou = resumed_best_iou if resumed_best_iou is not None else 0.

        grad_probe_every = getattr(self.opt, 'grad_probe_every', 0)
        while epoch < self.opt.max_epoch:
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            self.t2m_transformer.train()
            self.vq_model.eval()

            for i, batch in enumerate(train_loader):
                it += 1
                if it < self.opt.warm_up_iter:
                    self.update_lr_warm_up(it, self.opt.warm_up_iter, self.opt.lr)

                total_loss, acc, ce_loss, coord_loss = self.update(batch_data=batch)

                # Diagnostic: per-loss gradient norms, runs after the normal optimizer step
                # on the same batch, all ranks must call together (DDP). Adds ~1 extra
                # forward + 2 extra backwards per probed step (no_sync skips NCCL all-reduce).
                if grad_probe_every > 0 and it % grad_probe_every == 0:
                    self.gradient_probe(batch, it)
                logs['loss'] += total_loss
                logs['ce_loss'] += ce_loss
                logs['coord_loss'] += coord_loss
                logs['acc'] += acc
                logs['lr'] += self.opt_t2m_transformer.param_groups[0]['lr']

                if it % self.opt.log_every == 0 and self.is_main:
                    mean_loss = OrderedDict()
                    for tag, value in logs.items():
                        self.logger.add_scalar('Train/%s'%tag, value / self.opt.log_every, it)
                        mean_loss[tag] = value / self.opt.log_every
                    logs = defaultdict(def_value, OrderedDict())
                    print_current_loss(start_time, it, total_iters, mean_loss, epoch=epoch, inner_iter=i)
                elif it % self.opt.log_every == 0:
                    # Non-main ranks: still flush log buffers to keep memory bounded
                    logs = defaultdict(def_value, OrderedDict())

                if it % self.opt.save_latest == 0:
                    self.save(pjoin(self.opt.model_dir, 'latest.tar'), epoch, it, best_acc=best_acc, best_iou=best_iou)

            self.save(pjoin(self.opt.model_dir, 'latest.tar'), epoch, it, best_acc=best_acc, best_iou=best_iou)
            epoch += 1

            if self.is_main:
                print('Validation time:')
            self.vq_model.eval()
            self.t2m_transformer.eval()

            val_loss = []
            val_ce_loss = []
            val_coord_loss = []
            val_acc = []
            with torch.no_grad():
                for i, batch_data in enumerate(val_loader):
                    total_loss, acc, ce_loss, coord_loss = self.forward(batch_data)
                    val_loss.append(total_loss.item() if torch.is_tensor(total_loss) else total_loss)
                    val_ce_loss.append(ce_loss)
                    val_coord_loss.append(coord_loss)
                    val_acc.append(acc)

            # Aggregate val metrics across ranks (DDP) for accurate logging
            val_metrics = torch.tensor(
                [np.mean(val_loss), np.mean(val_ce_loss), np.mean(val_coord_loss), np.mean(val_acc)],
                device=self.device, dtype=torch.float32)
            if use_ddp:
                dist.all_reduce(val_metrics, op=dist.ReduceOp.AVG)
            v_loss, v_ce, v_coord, v_acc = val_metrics.tolist()

            if self.is_main:
                print(f"Validation loss:{v_loss:.3f}  ce:{v_ce:.3f}  coord:{v_coord:.4f}  acc:{v_acc:.3f}")
                self.logger.add_scalar('Val/loss', v_loss, epoch)
                self.logger.add_scalar('Val/ce_loss', v_ce, epoch)
                self.logger.add_scalar('Val/coord_loss', v_coord, epoch)
                self.logger.add_scalar('Val/acc', v_acc, epoch)

            if v_acc > best_acc:
                if self.is_main:
                    print(f"Improved accuracy from {best_acc:.02f} to {v_acc}!!!")
                best_acc = v_acc
                self.save(pjoin(self.opt.model_dir, 'net_best_acc.tar'), epoch, it, best_acc=best_acc, best_iou=best_iou)

            # Periodic IoU evaluation on fixed val towers (rank 0 only, single-batch full inference)
            if eval_towers is not None and epoch % eval_iou_every == 0 and self.is_main:
                # --val_decode_mode: 'windowed' (RVQ default) | 'whole' (IBQ) | 'both'.
                # 'both' logs windowed + whole; net_best_iou tracks the deployment
                # decode = whole. With 'windowed'/'whole', that single metric selects.
                decode_mode = getattr(self.opt, 'val_decode_mode', 'windowed')
                ious = {}
                if decode_mode in ('windowed', 'both'):
                    ious['windowed'] = self.compute_val_iou(eval_towers, mean, std,
                                                            decode_mode='windowed')
                if decode_mode in ('whole', 'both'):
                    ious['whole'] = self.compute_val_iou(eval_towers, mean, std,
                                                         decode_mode='whole')
                select = 'whole' if 'whole' in ious else 'windowed'
                val_iou = ious[select]
                summary = '  '.join(f'{k}={v:.4f}' for k, v in ious.items())
                print(f'Val IoU (epoch {epoch}): {summary}  '
                      f'[select={select}]  (best {select}: {best_iou:.4f})')
                for k, v in ious.items():
                    self.logger.add_scalar(f'Val/iou_{k}', v, epoch)
                self.logger.add_scalar('Val/iou', val_iou, epoch)  # selection metric
                if val_iou > best_iou:
                    print(f'Improved IoU from {best_iou:.4f} to {val_iou:.4f}! Saving net_best_iou.tar')
                    best_iou = val_iou
                    self.save(pjoin(self.opt.model_dir, 'net_best_iou.tar'), epoch, it, best_acc=best_acc, best_iou=best_iou)
                # Put model back in train mode (compute_val_iou sets eval)
                self.t2m_transformer.train()

            # Other ranks wait for rank 0 to finish IoU eval / saving before the next epoch
            if use_ddp:
                dist.barrier()

            # Periodic checkpoint every 50 epochs as failsafe
            if epoch % 50 == 0:
                self.save(pjoin(self.opt.model_dir, f'net_epoch_{epoch:04d}.tar'), epoch, it, best_acc=best_acc, best_iou=best_iou)
                if self.is_main:
                    print(f'Saved periodic checkpoint at epoch {epoch}')

