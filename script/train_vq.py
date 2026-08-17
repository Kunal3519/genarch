"""Train the tower tokenizer (Stage 1) -- see models/vq/vqvae.py.

    torchrun --nproc_per_node=N script/train_vq.py --name <vq> --dataset_name myarch \\
        --down_t 2 --stride_t 2 --window_size 64 --nb_code 1024 --quantizer ibq

Tracks reconstruction voxel-IoU and selects `net_best_iou.tar` on it.
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from os.path import join as pjoin

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from models.vq.vqvae import VQVAE
from models.vq.vq_trainer import VQTokenizerTrainer
from options.vq_option import arg_parse
from data.tower_dataset import MotionDataset
import numpy as np

from utils.get_opt import get_opt
from utils.fixseed import fixseed

os.environ["OMP_NUM_THREADS"] = "1"


if __name__ == "__main__":
    # torch.autograd.set_detect_anomaly(True)
    opt = arg_parse(True)
    fixseed(opt.seed)

    # --- DDP Setup ---
    opt.use_ddp = len(opt.gpu_ids) > 1
    if opt.use_ddp:
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        dist.init_process_group(backend='nccl')
        opt.local_rank = local_rank
        opt.device = torch.device(f'cuda:{local_rank}')
        torch.cuda.set_device(local_rank)
        opt.is_main_process = (local_rank == 0)
    else:
        opt.local_rank = 0
        opt.device = torch.device("cpu" if opt.gpu_ids[0] == -1 else "cuda:" + str(opt.gpu_ids[0]))
        opt.is_main_process = True

    if opt.is_main_process:
        print(f"Using Device: {opt.device}, GPUs: {opt.gpu_ids}, DDP: {opt.use_ddp}")

    opt.save_root = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name)
    opt.model_dir = pjoin(opt.save_root, 'model')
    opt.meta_dir = pjoin(opt.save_root, 'meta')
    opt.eval_dir = pjoin(opt.save_root, 'animation')
    opt.log_dir = pjoin('./log/vq/', opt.dataset_name, opt.name)

    if opt.is_main_process:
        os.makedirs(opt.model_dir, exist_ok=True)
        os.makedirs(opt.meta_dir, exist_ok=True)
        os.makedirs(opt.eval_dir, exist_ok=True)
        os.makedirs(opt.log_dir, exist_ok=True)

    if opt.dataset_name == "myarch":
        # opt.data_root comes from --data_root (default = active 26k dir)
        opt.motion_dir = pjoin(opt.data_root, 'new_joint_vecs_int')
        opt.joints_num = 1
        dim_pose = 3
    else:
        raise KeyError('Dataset Does not Exist')

    mean = np.load(pjoin(opt.data_root, 'Mean.npy'))
    std = np.load(pjoin(opt.data_root, 'Std.npy'))

    train_split_file = pjoin(opt.data_root, 'train160k.txt')
    val_split_file = pjoin(opt.data_root, 'val40k.txt')

    net = VQVAE(opt,
                dim_pose,
                opt.nb_code,
                opt.code_dim,
                opt.code_dim,
                opt.down_t,
                opt.stride_t,
                opt.width,
                opt.depth,
                opt.dilation_growth_rate,
                opt.vq_act,
                opt.vq_norm)

    pc_vq = sum(param.numel() for param in net.parameters())
    if opt.is_main_process:
        print(net)
        print('Total parameters of all models: {}M'.format(pc_vq/1000_000))

    opt.mean = mean.tolist()
    opt.std = std.tolist()
    trainer = VQTokenizerTrainer(opt, vq_model=net)

    train_dataset = MotionDataset(opt, mean, std, train_split_file)
    val_dataset = MotionDataset(opt, mean, std, val_split_file)

    with open(val_split_file) as f:
        val_ids = [l.strip() for l in f if l.strip()]

    # Use DistributedSampler for DDP
    train_sampler = DistributedSampler(train_dataset, shuffle=True) if opt.use_ddp else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if opt.use_ddp else None

    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, drop_last=True, num_workers=4,
                              shuffle=(train_sampler is None), pin_memory=True, sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=opt.batch_size, drop_last=True, num_workers=4,
                            shuffle=(val_sampler is None), pin_memory=True, sampler=val_sampler)

    trainer.train(train_loader, val_loader,
                  train_sampler=train_sampler,
                  val_motion_dir=opt.motion_dir,
                  val_ids=val_ids,
                  eval_iou_every=5,
                  early_stop_iou=getattr(opt, 'early_stop_iou', None))

    if opt.use_ddp:
        dist.destroy_process_group()