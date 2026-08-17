import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP
from os.path import join as pjoin
import torch.nn.functional as F

import torch.optim as optim

import time
import numpy as np
from collections import OrderedDict, defaultdict
from utils.utils import print_current_loss
from utils.voxel import voxel_iou as _voxel_iou_pair

import os
import sys

def def_value():
    return 0.0


def compute_val_iou(model, val_ids, motion_dir, mean, std, window_size, device):
    """Compute mean voxel IoU over all val towers (grid-snapped reconstruction vs GT)."""
    model.eval()

    ious = []
    with torch.no_grad():
        for fid in val_ids:
            path = os.path.join(motion_dir, fid + '.npy')
            if not os.path.exists(path):
                continue
            motion = np.load(path).astype(np.float32)
            L = len(motion)

            norm = (motion - mean) / std
            pad = (window_size - L % window_size) % window_size
            if L < window_size:
                pad = window_size - L
            if pad > 0:
                norm = np.pad(norm, ((0, pad), (0, 0)), mode='edge')

            chunks = []
            for start in range(0, norm.shape[0], window_size):
                t = torch.from_numpy(norm[start:start + window_size]).unsqueeze(0).to(device)
                out, _, _ = model(t)
                chunks.append(out.squeeze(0).cpu().numpy())

            recon = np.concatenate(chunks, axis=0)[:L] * std + mean
            ious.append(_voxel_iou_pair(motion, recon))

    return float(np.mean(ious)) if ious else 0.0


class VQTokenizerTrainer:
    def __init__(self, args, vq_model):
        self.opt = args
        self.vq_model = vq_model
        self.device = args.device
        self.is_main = getattr(args, 'is_main_process', True)

        if args.is_train and self.is_main:
            self.logger = SummaryWriter(args.log_dir)
        else:
            self.logger = None

        if args.is_train:
            if args.recons_loss == 'l1':
                self.l1_criterion = torch.nn.L1Loss()
            elif args.recons_loss == 'l1_smooth':
                self.l1_criterion = torch.nn.SmoothL1Loss()

        # self.critic = CriticWrapper(self.opt.dataset_name, self.opt.device)

    def forward(self, batch_data):
        motions = batch_data.detach().to(self.device).float()
        pred_motion, loss_commit, perplexity = self.vq_model(motions)
        
        # DDP/DataParallel may return per-GPU values; reduce them
        loss_commit = loss_commit.mean()
        perplexity = perplexity.mean()

        self.motions = motions
        self.pred_motion = pred_motion

        loss_rec = self.l1_criterion(pred_motion, motions)

        if self.opt.dataset_name == "myarch":
            pred_local_pos = pred_motion
            local_pos = motions
        else:
            pred_local_pos = pred_motion[..., 4 : (self.opt.joints_num - 1) * 3 + 4]
            local_pos = motions[..., 4 : (self.opt.joints_num - 1) * 3 + 4]

        loss_explicit = self.l1_criterion(pred_local_pos, local_pos)

        if getattr(self.opt, 'quantizer', 'ema') == 'ibq':
            # IBQ returns its full (commitment + entropy) aux loss already weighted
            loss = loss_rec + self.opt.loss_vel * loss_explicit + loss_commit
        else:
            loss = loss_rec + self.opt.loss_vel * loss_explicit + self.opt.commit * loss_commit

        # return loss, loss_rec, loss_vel, loss_commit, perplexity
        # return loss, loss_rec, loss_percept, loss_commit, perplexity
        return loss, loss_rec, loss_explicit, loss_commit, perplexity


    # @staticmethod
    def update_lr_warm_up(self, nb_iter, warm_up_iter, lr):

        current_lr = lr * (nb_iter + 1) / (warm_up_iter + 1)
        for param_group in self.opt_vq_model.param_groups:
            param_group["lr"] = current_lr

        return current_lr

    def _unwrap_model(self):
        """Get the underlying model, unwrapping DDP/DataParallel if needed."""
        if isinstance(self.vq_model, (DDP, torch.nn.DataParallel)):
            return self.vq_model.module
        return self.vq_model

    def save(self, file_name, ep, total_it):
        if not self.is_main:
            return
        state = {
            "vq_model": self._unwrap_model().state_dict(),
            "opt_vq_model": self.opt_vq_model.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            'ep': ep,
            'total_it': total_it,
        }
        torch.save(state, file_name)

    def resume(self, model_dir):
        checkpoint = torch.load(model_dir, map_location=self.device)
        self._unwrap_model().load_state_dict(checkpoint['vq_model'])
        self.opt_vq_model.load_state_dict(checkpoint['opt_vq_model'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        return checkpoint['ep'], checkpoint['total_it']

    def train(self, train_loader, val_loader,
              train_sampler=None, val_motion_dir=None, val_ids=None, eval_iou_every=5,
              early_stop_iou=None):
        self.vq_model.to(self.device)

        # DDP wrapping
        if getattr(self.opt, 'use_ddp', False):
            self.vq_model = DDP(self.vq_model, device_ids=[self.opt.local_rank])
            if self.is_main:
                print(f'Using DDP on {dist.get_world_size()} GPUs')

        self.opt_vq_model = optim.AdamW(self.vq_model.parameters(), lr=self.opt.lr, betas=(0.9, 0.99), weight_decay=self.opt.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.opt_vq_model, milestones=self.opt.milestones, gamma=self.opt.gamma)

        epoch = 0
        it = 0
        if self.opt.is_continue:
            model_dir = pjoin(self.opt.model_dir, 'latest.tar')
            epoch, it = self.resume(model_dir)
            print("Load model epoch:%d iterations:%d"%(epoch, it))

        best_iou = 0.0
        min_val_loss = float('inf')
        start_time = time.time()
        total_iters = self.opt.max_epoch * len(train_loader)
        if self.is_main:
            print(f'Total Epochs: {self.opt.max_epoch}, Total Iters: {total_iters}')
            print('Iters Per Epoch, Training: %04d' % len(train_loader))
        current_lr = self.opt.lr
        logs = defaultdict(def_value, OrderedDict())

        while epoch < self.opt.max_epoch:
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            self.vq_model.train()
            for i, batch_data in enumerate(train_loader):
                it += 1
                if it < self.opt.warm_up_iter:
                    current_lr = self.update_lr_warm_up(it, self.opt.warm_up_iter, self.opt.lr)
                loss, loss_rec, loss_vel, loss_commit, perplexity = self.forward(batch_data)
                self.opt_vq_model.zero_grad()
                loss.backward()
                self.opt_vq_model.step()

                if it >= self.opt.warm_up_iter:
                    self.scheduler.step()
                
                logs['loss'] += loss.item()
                logs['loss_rec'] += loss_rec.item()
                # Note it not necessarily velocity, too lazy to change the name now
                logs['loss_vel'] += loss_vel.item()
                logs['loss_commit'] += loss_commit.item()
                logs['perplexity'] += perplexity.item()
                logs['lr'] += self.opt_vq_model.param_groups[0]['lr']

                if it % self.opt.log_every == 0 and self.is_main:
                    mean_loss = OrderedDict()
                    for tag, value in logs.items():
                        self.logger.add_scalar('Train/%s'%tag, value / self.opt.log_every, it)
                        mean_loss[tag] = value / self.opt.log_every
                    logs = defaultdict(def_value, OrderedDict())
                    print_current_loss(start_time, it, total_iters, mean_loss, epoch=epoch, inner_iter=i)

                if it % self.opt.save_latest == 0:
                    self.save(pjoin(self.opt.model_dir, 'latest.tar'), epoch, it)

            self.save(pjoin(self.opt.model_dir, 'latest.tar'), epoch, it)

            epoch += 1
            # if epoch % self.opt.save_every_e == 0:
            #     self.save(pjoin(self.opt.model_dir, 'E%04d.tar' % (epoch)), epoch, total_it=it)

            if self.is_main:
                print('Validation time:')
            self.vq_model.eval()
            val_loss_rec = []
            val_loss_vel = []
            val_loss_commit = []
            val_loss = []
            val_perpexity = []
            val_mae = []
            _mean = torch.tensor(self.opt.mean, device=self.device, dtype=torch.float32) if hasattr(self.opt, 'mean') else None
            _std  = torch.tensor(self.opt.std,  device=self.device, dtype=torch.float32) if hasattr(self.opt, 'std')  else None
            with torch.no_grad():
                for i, batch_data in enumerate(val_loader):
                    loss, loss_rec, loss_vel, loss_commit, perplexity = self.forward(batch_data)
                    val_loss.append(loss.item())
                    val_loss_rec.append(loss_rec.item())
                    val_loss_vel.append(loss_vel.item())
                    val_loss_commit.append(loss_commit.item())
                    val_perpexity.append(perplexity.item())
                    # MAE in world coordinates (single pass)
                    if _mean is not None and _std is not None:
                        gt_world   = self.motions    * _std + _mean
                        pred_world = self.pred_motion * _std + _mean
                    else:
                        gt_world, pred_world = self.motions, self.pred_motion
                    val_mae.append(torch.mean(torch.abs(pred_world - gt_world)).item())

            if self.is_main and self.logger is not None:
                self.logger.add_scalar('Val/loss', sum(val_loss) / len(val_loss), epoch)
                self.logger.add_scalar('Val/loss_rec', sum(val_loss_rec) / len(val_loss_rec), epoch)
                self.logger.add_scalar('Val/loss_vel', sum(val_loss_vel) / len(val_loss_vel), epoch)
                self.logger.add_scalar('Val/loss_commit', sum(val_loss_commit) / len(val_loss), epoch)
                self.logger.add_scalar('Val/loss_perplexity', sum(val_perpexity) / len(val_loss_rec), epoch)
                self.logger.add_scalar('Val/MAE', sum(val_mae) / len(val_mae), epoch)

            if self.is_main:
                print('Validation Loss: %.5f Reconstruction: %.5f, Velocity: %.5f, Commit: %.5f  |  MAE: %.4f' %
                      (sum(val_loss)/len(val_loss), sum(val_loss_rec)/len(val_loss),
                       sum(val_loss_vel)/len(val_loss), sum(val_loss_commit)/len(val_loss),
                       sum(val_mae)/len(val_mae)))

            # Lowest-validation-loss checkpoint, kept alongside net_best_iou.tar and
            # latest.tar: reconstruction IoU and total loss do not always peak at the
            # same epoch, so both selection criteria are preserved.
            _mean_val_loss = sum(val_loss) / len(val_loss)
            if self.is_main and _mean_val_loss < min_val_loss:
                min_val_loss = _mean_val_loss
                self.save(pjoin(self.opt.model_dir, 'net_best_loss.tar'), epoch, it)
                print(f'  -> New lowest val loss ({min_val_loss:.5f}), checkpoint saved')

            if val_motion_dir is not None and val_ids is not None and self.is_main and epoch % eval_iou_every == 0:
                mean_np = np.array(self.opt.mean, dtype=np.float32)
                std_np  = np.array(self.opt.std,  dtype=np.float32)
                iou = compute_val_iou(
                    self._unwrap_model(), val_ids, val_motion_dir,
                    mean_np, std_np, self.opt.window_size, self.device
                )
                self.vq_model.train()
                if self.logger is not None:
                    self.logger.add_scalar('Val/iou', iou, epoch)
                print(f'Val IoU ({len(val_ids)} towers): {iou:.4f}  best: {best_iou:.4f}')
                if iou > best_iou:
                    best_iou = iou
                    self.save(pjoin(self.opt.model_dir, 'net_best_iou.tar'), epoch, it)
                    print('  -> New best IoU checkpoint saved')
                if early_stop_iou is not None and iou >= early_stop_iou:
                    print(f'Early stopping: Val IoU {iou:.4f} >= threshold {early_stop_iou:.4f}')
                    if dist.is_available() and dist.is_initialized():
                        dist.destroy_process_group()
                    return

            # if epoch - min_val_epoch >= self.opt.early_stop_e:
            #     print('Early Stopping!~')


