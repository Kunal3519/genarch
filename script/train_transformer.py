"""Train the conditional masked tower transformer (Stage 2) -- paper section
"Conditional Masked Tower Transformer".

Conditioning is DINOv2 patch tokens: all 256 spatial patches per view
(4 views x 256 x 1024) enter every block as cross-attention memory, so each
token attends to the image regions it needs, rather than one pooled vector.
`--use_cls_prefix` additionally prepends 4 per-view CLS-style prefix tokens and
modulates the FFN with a pooled conditioning vector -- the deployed variant.

Prerequisite: cached patch features (once per dataset)
    uv run python script/preprocess/precompute_patch_dl.py \\
        --out_root <DATA_ROOT>

Deployed recipe (the released checkpoint, 3-GPU DDP):
    CUDA_VISIBLE_DEVICES=<uuids> setsid nohup uv run torchrun --nproc_per_node=3 \\
        --master_port=29527 script/train_transformer.py \\
        --name <run> --dataset_name myarch --vq_name ibq_4x_nc1024_nq1 \\
        --use_cls_prefix --deterministic_crop --val_decode_mode whole \\
        --batch_size 128 --max_epoch 4950 --lr 1e-4 \\
        --milestone_fractions 0.90 0.97 --warm_up_iter 500 \\
        --latent_dim 384 --n_heads 6 --n_layers 8 --ff_size 1024 \\
        --dropout 0.2 --cond_drop_prob 0.1 --view_drop_prob 0.2 \\
        --coord_loss_weight 1.0 --ce_loss_weight 0.1 < /dev/null &

Notes:
  --coord_loss_weight 1.0 / --ce_loss_weight 0.1 is the coordinate-dominant
    objective; plain CE alone caps around 0.88 because codebook entries are
    context-dependent synonyms.
  --val_decode_mode whole matches IBQ's deployment decode (whole-sequence).
  --view_drop_prob 0.2 makes partial-view inference in-distribution.
  --stream_img_feat streams features from disk instead of preloading them
    (required at 160k towers, and for parallel runs).
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from utils.paths import DATA_ROOT

import torch
import torch.distributed as dist
import numpy as np

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from os.path import join as pjoin

from models.mask_transformer.transformer import MaskTransformer
from models.mask_transformer.transformer_trainer import MaskTransformerTrainer

from options.train_option import TrainT2MOptions
from utils.get_opt import get_opt
from utils.fixseed import fixseed
from utils.model_io import load_vq

from data.tower_dataset import Image2TowerDataset

os.environ.setdefault("OMP_NUM_THREADS", "1")


if __name__ == '__main__':
    parser = TrainT2MOptions()
    opt = parser.parse()
    fixseed(opt.seed)

    if isinstance(opt.gpu_id, str) and ',' in str(opt.gpu_id):
        gpu_ids = [int(g) for g in str(opt.gpu_id).split(',') if g.strip()]
    elif isinstance(opt.gpu_id, int) and opt.gpu_id >= 0:
        gpu_ids = [opt.gpu_id]
    else:
        gpu_ids = []
    opt.gpu_ids = gpu_ids
    torch.autograd.set_detect_anomaly(False)

    # --- DDP setup (driven by torchrun --nproc_per_node=N) ---
    opt.use_ddp = 'LOCAL_RANK' in os.environ and len(gpu_ids) > 1
    if opt.use_ddp:
        local_rank = int(os.environ['LOCAL_RANK'])
        dist.init_process_group(backend='nccl')
        opt.local_rank = local_rank
        opt.device = torch.device(f'cuda:{local_rank}')
        torch.cuda.set_device(local_rank)
        opt.is_main_process = (dist.get_rank() == 0)
    else:
        opt.local_rank = 0
        opt.device = torch.device('cpu' if not gpu_ids else f'cuda:{gpu_ids[0]}')
        if gpu_ids:
            torch.cuda.set_device(gpu_ids[0])
        opt.is_main_process = True

    if opt.is_main_process:
        print(f'Device: {opt.device}, GPUs: {gpu_ids}, DDP: {opt.use_ddp}')

    opt.save_root = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name)
    opt.model_dir = pjoin(opt.save_root, 'model')
    opt.log_dir = pjoin('./log/mask/', opt.dataset_name, opt.name)
    if opt.is_main_process:
        os.makedirs(opt.model_dir, exist_ok=True)
        os.makedirs(opt.log_dir, exist_ok=True)

    assert opt.dataset_name == 'myarch', 'Patch conditioning only implemented for myarch'
    if not getattr(opt, 'data_root', None):
        opt.data_root   = DATA_ROOT
    opt.motion_dir      = pjoin(opt.data_root, 'new_joint_vecs_int')
    opt.image_feat_dir  = pjoin(opt.data_root, 'image_features_dino_patch')
    opt.joints_num      = 1
    opt.max_motion_len  = 512
    opt.max_motion_length = 2000
    # Default coord_loss_weight for patch model if not set via CLI
    if opt.coord_loss_weight == 0.0:
        opt.coord_loss_weight = 1.0
    dim_pose = 3

    assert os.path.isdir(opt.image_feat_dir), (
        f'DINOv2 patch features not found at {opt.image_feat_dir}\n'
        f'Run: python script/preprocess/precompute_patch_dl.py --out_root <DATA_ROOT>')

    vq_model, vq_opt = load_vq(opt.checkpoints_dir, opt.vq_name,
                                 device=opt.device, dim_pose=dim_pose,
                                 dataset_name=opt.dataset_name)
    opt.num_tokens     = vq_opt.nb_code
    opt.num_quantizers = vq_opt.num_quantizers
    opt.code_dim       = vq_opt.code_dim

    if getattr(opt, 'use_cls_prefix', False):
        cond_mode = 'image_patch_clsprefix'
    else:
        cond_mode = 'image_patch'
    if opt.is_main_process:
        print(f'cond_mode: {cond_mode}')

    # clip_dim unused in image_patch mode (patch_proj handles the 1024-dim internally)
    t2m_transformer = MaskTransformer(
        code_dim=vq_opt.code_dim,
        cond_mode=cond_mode,
        latent_dim=opt.latent_dim,
        ff_size=opt.ff_size,
        num_layers=opt.n_layers,
        num_heads=opt.n_heads,
        dropout=opt.dropout,
        clip_dim=1024,            # per-patch dim (used only as placeholder; patch_proj is 1024→latent_dim)
        cond_drop_prob=opt.cond_drop_prob,
        clip_version=None,
        opt=opt)

    pc_transformer = sum(p.numel() for p in t2m_transformer.parameters())
    if opt.is_main_process:
        print(f'Total parameters: {pc_transformer / 1e6:.2f}M')

    mean = np.load(pjoin(opt.checkpoints_dir, opt.dataset_name, opt.vq_name, 'meta', 'mean.npy'))
    std  = np.load(pjoin(opt.checkpoints_dir, opt.dataset_name, opt.vq_name, 'meta', 'std.npy'))

    train_dataset = Image2TowerDataset(opt, mean, std, pjoin(opt.data_root, getattr(opt, 'train_split', 'train160k.txt')))
    val_dataset   = Image2TowerDataset(opt, mean, std, pjoin(opt.data_root, getattr(opt, 'val_split', 'val40k.txt')))

    num_workers = getattr(opt, 'num_workers', 4)

    # DistributedSampler shards the dataset across ranks (no replication)
    train_sampler = DistributedSampler(train_dataset, shuffle=True) if opt.use_ddp else None
    val_sampler   = DistributedSampler(val_dataset,   shuffle=False) if opt.use_ddp else None

    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size,
                              num_workers=num_workers, drop_last=True,
                              shuffle=(train_sampler is None), sampler=train_sampler,
                              pin_memory=True)
    val_loader   = DataLoader(val_dataset, batch_size=opt.batch_size,
                              num_workers=num_workers, drop_last=True,
                              shuffle=(val_sampler is None), sampler=val_sampler,
                              pin_memory=True)

    # Build fixed eval set from 32 val towers for periodic IoU tracking (rank 0 only, runs single-batch).
    # Read directly from the underlying dataset so all ranks see the same towers regardless of sharding.
    eval_towers = None
    if opt.is_main_process:
        eval_towers = []
        for i in range(min(32, len(val_dataset))):
            img_feat, motion, m_len = val_dataset[i]
            eval_towers.append({
                'cond':        torch.from_numpy(np.asarray(img_feat)).unsqueeze(0).float(),
                'motion_norm': torch.from_numpy(np.asarray(motion)).unsqueeze(0).float(),
                'm_len':       int(m_len),
            })
        print(f'Built eval set with {len(eval_towers)} towers for IoU tracking')

    trainer = MaskTransformerTrainer(opt, t2m_transformer, vq_model)
    trainer.train(train_loader, val_loader,
                  eval_towers=eval_towers, mean=mean, std=std, eval_iou_every=50,
                  train_sampler=train_sampler)

    if opt.use_ddp:
        dist.destroy_process_group()
