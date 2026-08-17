"""
Generic full-split voxel-IoU eval for ANY myarch mask transformer (v2..v8+).

Point it at any checkpoint(s) via CLI — cond_mode
(plain / CLS-prefix / AdaLN variants) is auto-detected from the model's opt.txt,
and the matching VQ tokenizer is loaded by name. No new script per version.

For each checkpoint it reports mean voxel-IoU on the val split, a 32-tower
subset (the training-time probe), and the train split (shardable). Optionally
also reports the GT Q0-only and full VQ-reconstruction ceilings (--ceilings).

Decode: windowed-64 for RVQ (default), whole-seq for IBQ (--decode_mode whole).

Usage (single ckpt, val only):
    uv run python script/eval_mask_full.py \
        --mask_name myarch_mask_img_patch_v7 --ckpt net_best_iou.tar \
        --vq_name rvq_v2_4x_nc1024_nq3 --data_root $GENARCH_DATA_ROOT \
        --val_only --gpu_id 0

Usage (multiple ckpts + ceilings):
    ... --mask_name myarch_mask_img_patch_v5 --ckpt latest.tar net_best_iou.tar net_best_acc.tar --ceilings

Usage (sharded train split across 4 GPUs, then combine):
    for sh in 0 1 2 3; do
      uv run ... script/eval_mask_full.py --mask_name <NAME> --ckpt net_best_iou.tar \
          --vq_name <VQ> --data_root <ROOT> --gpu_id $sh \
          --shard_idx $sh --num_shards 4 --out_path eval_${sh}.npy &
    done; wait
    uv run ... script/eval_mask_full.py --combine --num_shards 4 \
        --shard_glob 'eval_{sh}.npy' --out_path eval_combined.npy

Custom split file:
    ... --split val40k.txt
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from utils.paths import DATA_ROOT

import argparse
import torch
import numpy as np
from os.path import join as pjoin
from tqdm import tqdm

from models.mask_transformer.transformer import MaskTransformer
from utils.get_opt import get_opt
from utils.fixseed import fixseed
from utils.model_io import load_vq
from utils.voxel import voxel_iou as _voxel_iou
from utils.vq_decode import decode_windowed as _decode_windowed, DEFAULT_WINDOW_TOKENS


CHECKPOINTS_DIR = './checkpoints'
# Overridden at runtime by --data_root / --vq_name. Defaults to the active 26k set.
DATA_ROOT       = DATA_ROOT
DATASET_NAME    = 'myarch'
VQ_NAME         = 'rvq_v2_4x_nc1024_nq3'
MAX_MOTION_LEN  = 2000
UNIT_LENGTH     = 4
DIM_POSE        = 3
BATCH_SIZE      = 32

MASK_TIMESTEPS   = 10
MASK_COND_SCALE  = 1.0
MASK_TEMPERATURE = 0.2
MASK_TOPK        = 0.9

# Windowed-decode helper lives in utils.vq_decode; alias for readability.
DECODE_WINDOW_TOKENS = DEFAULT_WINDOW_TOKENS
# 'windowed' (RVQ-safe, default) or 'whole' (IBQ's best mode). Set from --decode_mode.
DECODE_MODE = 'windowed'


def _cond_mode_from_opt(opt):
    """Auto-detect the patch-conditioning variant from the model's saved opt.txt."""
    if getattr(opt, 'use_cls_prefix', False):
        return 'image_patch_clsprefix'
    return 'image_patch'


def decode_windowed(vq_model, code_idx_full, window_tokens=DECODE_WINDOW_TOKENS):
    """Decode tokens to coords. Honors DECODE_MODE: windowed-64 (RVQ) or whole-seq (IBQ)."""
    if DECODE_MODE == 'whole':
        x_out = vq_model.forward_decoder(code_idx_full)
        if x_out.dim() == 3 and x_out.shape[1] == DIM_POSE:  # (B, C, T) -> (B, T, C)
            x_out = x_out.transpose(1, 2)
        return x_out
    return _decode_windowed(vq_model, code_idx_full, window_tokens=window_tokens)


def voxel_iou(gen_coords, gt_coords, grid=(4, 4, 8)):
    # Thin wrapper kept for backward compat with any external callers; uses shared utils.voxel.
    return _voxel_iou(gen_coords, gt_coords)


def load_vq_model(device, vq_name=None):
    vq_model, vq_opt = load_vq(CHECKPOINTS_DIR, vq_name or VQ_NAME, device=device,
                                 dim_pose=DIM_POSE, dataset_name=DATASET_NAME)
    vq_model.eval().to(device)
    return vq_model, vq_opt


def load_mask_model(vq_opt, mask_name, ckpt_path, device):
    opt_path = pjoin(CHECKPOINTS_DIR, DATASET_NAME, mask_name, 'opt.txt')
    mask_opt = get_opt(opt_path, device)
    mask_opt.num_tokens        = vq_opt.nb_code
    mask_opt.num_quantizers    = vq_opt.num_quantizers
    mask_opt.code_dim          = vq_opt.code_dim
    mask_opt.max_motion_len    = 512
    mask_opt.max_motion_length = MAX_MOTION_LEN

    cond_mode = _cond_mode_from_opt(mask_opt)
    print(f'  cond_mode: {cond_mode}  vq: {vq_opt.nb_code}x{vq_opt.num_quantizers}')

    model = MaskTransformer(
        code_dim=vq_opt.code_dim,
        cond_mode=cond_mode,
        latent_dim=mask_opt.latent_dim,
        ff_size=mask_opt.ff_size,
        num_layers=mask_opt.n_layers,
        num_heads=mask_opt.n_heads,
        dropout=mask_opt.dropout,
        clip_dim=1024,
        cond_drop_prob=mask_opt.cond_drop_prob,
        clip_version=None,
        opt=mask_opt,
    )

    ckpt = torch.load(ckpt_path, map_location='cpu')
    ep = ckpt.get('ep', '?')
    state_dict = ckpt.get('t2m_transformer', ckpt)
    if any(k.startswith('module.') for k in state_dict):
        state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f'  load_state_dict missing={len(missing)} unexpected={len(unexpected)}')
    model.eval().to(device)
    print(f'  Loaded {os.path.basename(ckpt_path)} (epoch {ep}) from {mask_name}')
    return model


def load_records(split_file):
    motion_dir     = pjoin(DATA_ROOT, 'new_joint_vecs_int')
    patch_feat_dir = pjoin(DATA_ROOT, 'image_features_dino_patch')

    with open(split_file, 'r') as f:
        names = [l.strip() for l in f if l.strip()]

    records = []
    skipped = 0
    for name in tqdm(names, desc=f'Loading {os.path.basename(split_file)}'):
        motion_path = pjoin(motion_dir, name + '.npy')
        feat_path   = pjoin(patch_feat_dir, name + '.npy')
        if not os.path.exists(motion_path) or not os.path.exists(feat_path):
            skipped += 1
            continue
        motion_raw = np.load(motion_path).astype(np.float32)
        m_len = (len(motion_raw) // UNIT_LENGTH) * UNIT_LENGTH
        if m_len == 0:
            skipped += 1
            continue
        patch_feat = np.load(feat_path).astype(np.float32)
        records.append((name, motion_raw[:m_len], patch_feat))

    if skipped:
        print(f'  Skipped {skipped} towers')
    return records


@torch.no_grad()
def eval_mask_model(records, mean, std, vq_model, mask_model, device, desc='  Batches'):
    nb_code = vq_model.quantizer.layers[0].nb_code
    ious, names = [], []
    for batch_start in tqdm(range(0, len(records), BATCH_SIZE), desc=desc, leave=False):
        batch = records[batch_start: batch_start + BATCH_SIZE]
        motions_b = [r[1] for r in batch]
        feats_b   = [r[2] for r in batch]
        m_lens    = [len(m) for m in motions_b]
        cond       = torch.from_numpy(np.stack(feats_b)).to(device)
        m_lens_tok = torch.tensor([ml // 4 for ml in m_lens], device=device, dtype=torch.long)

        gen_ids = mask_model.generate(
            cond, m_lens_tok,
            timesteps=MASK_TIMESTEPS,
            cond_scale=MASK_COND_SCALE,
            temperature=MASK_TEMPERATURE,
            topk_filter_thres=MASK_TOPK,
            gsample=False,
        )
        gen_ids_safe = gen_ids.clamp(0, nb_code - 1)
        coords_pred  = decode_windowed(vq_model, gen_ids_safe.unsqueeze(-1)).cpu().numpy()
        for i, m_len in enumerate(m_lens):
            n_cmp   = (m_len // 4) * 4
            gen_xyz = coords_pred[i, :n_cmp] * std + mean
            gt_xyz  = motions_b[i][:n_cmp]
            ious.append(voxel_iou(gen_xyz, gt_xyz))
            names.append(batch[i][0])
    return ious, names


@torch.no_grad()
def _encode_decode_windowed(motion_norm_1d, vq_model, q0_only, window_blocks=64):
    """Per-tower windowed encode + decode (RVQ ceiling protocol: win-enc + win-dec)."""
    L = motion_norm_1d.shape[0]
    pad = (window_blocks - L % window_blocks) % window_blocks
    if L < window_blocks:
        pad = window_blocks - L
    if pad:
        motion_norm_1d = np.concatenate(
            [motion_norm_1d, np.zeros((pad, motion_norm_1d.shape[1]), dtype=motion_norm_1d.dtype)], axis=0)
    pieces = []
    for s in range(0, motion_norm_1d.shape[0], window_blocks):
        chunk = torch.from_numpy(motion_norm_1d[s:s + window_blocks]).unsqueeze(0).to(vq_model.quantizer.layers[0].codebook.device)
        code_idx, all_codes = vq_model.encode(chunk)
        if q0_only:
            piece = vq_model.decoder(all_codes[0]).squeeze(0).cpu().numpy()
        else:
            piece = vq_model.forward_decoder(code_idx).squeeze(0).cpu().numpy()
        pieces.append(piece)
    return np.concatenate(pieces, axis=0)[:L]


@torch.no_grad()
def eval_gt_q0_only(records, mean, std, vq_model, device):
    ious = []
    for name, motion_raw, _ in tqdm(records, desc='  Towers', leave=False):
        m_len = (len(motion_raw) // UNIT_LENGTH) * UNIT_LENGTH
        motion_norm = (motion_raw[:m_len] - mean) / std
        coords = _encode_decode_windowed(motion_norm, vq_model, q0_only=True)
        gen_xyz = coords * std + mean
        gt_xyz  = motion_raw[:m_len]
        ious.append(voxel_iou(gen_xyz, gt_xyz))
    return ious


@torch.no_grad()
def eval_gt_full(records, mean, std, vq_model, device):
    ious = []
    for name, motion_raw, _ in tqdm(records, desc='  Towers', leave=False):
        m_len = (len(motion_raw) // UNIT_LENGTH) * UNIT_LENGTH
        motion_norm = (motion_raw[:m_len] - mean) / std
        coords = _encode_decode_windowed(motion_norm, vq_model, q0_only=False)
        gen_xyz = coords * std + mean
        gt_xyz  = motion_raw[:m_len]
        ious.append(voxel_iou(gen_xyz, gt_xyz))
    return ious


def summarise(ious, label):
    a = np.asarray(ious)
    print(f'  {label:<35}  mean={a.mean():.4f}  median={np.median(a):.4f}  '
          f'p10={np.percentile(a,10):.4f}  p90={np.percentile(a,90):.4f}  '
          f'≥0.7:{(a>=0.7).mean()*100:.1f}%  ≥0.5:{(a>=0.5).mean()*100:.1f}%')


def _eval_ceilings(val_records, val_first32, train_records, mean, std, vq_model, device):
    out = {}
    print('\n── GT Q0-only ceiling (mask 100% accuracy) ──')
    val_q0  = eval_gt_q0_only(val_records, mean, std, vq_model, device)
    f32_q0  = eval_gt_q0_only(val_first32, mean, std, vq_model, device)
    train_q0 = eval_gt_q0_only(train_records, mean, std, vq_model, device) if train_records else []
    summarise(val_q0,  f'val   (n={len(val_records)})')
    summarise(f32_q0,  'val   first-32 subset')
    if train_q0: summarise(train_q0, f'train (n={len(train_records)})')
    out['GT Q0 only ceiling'] = {'val': val_q0, 'val_first32': f32_q0, 'train': train_q0}

    print('\n── GT Q0-Qn full ceiling (VQ reconstruction) ──')
    val_full   = eval_gt_full(val_records, mean, std, vq_model, device)
    f32_full   = eval_gt_full(val_first32, mean, std, vq_model, device)
    train_full = eval_gt_full(train_records, mean, std, vq_model, device) if train_records else []
    summarise(val_full,  f'val   (n={len(val_records)})')
    summarise(f32_full,  'val   first-32 subset')
    if train_full: summarise(train_full, f'train (n={len(train_records)})')
    out['GT Q0-Q2 (VQ ceiling)'] = {'val': val_full, 'val_first32': f32_full, 'train': train_full}
    return out


def _eval_one_ckpt(label, mask_name, ckpt_file, vq_opt, vq_model, mean, std,
                   val_records, val_first32, train_records, device):
    ckpt_path = pjoin(CHECKPOINTS_DIR, DATASET_NAME, mask_name, 'model', ckpt_file)
    if not os.path.exists(ckpt_path):
        print(f'\nSkipping {label}: not found at {ckpt_path}')
        return None
    print(f'\n── {label} ──')
    mask_model = load_mask_model(vq_opt, mask_name, ckpt_path, device)
    val_iou,   val_names   = eval_mask_model(val_records,  mean, std, vq_model, mask_model, device, desc='  val')
    f32_iou,   _           = eval_mask_model(val_first32,  mean, std, vq_model, mask_model, device, desc='  val[:32]')
    train_iou, train_names = ([], [])
    if train_records:
        train_iou, train_names = eval_mask_model(train_records, mean, std, vq_model, mask_model, device, desc='  train')
    if val_iou: summarise(val_iou,  f'val   (n={len(val_iou)})')
    if f32_iou: summarise(f32_iou,  'val   first-32 subset')
    if train_iou: summarise(train_iou, f'train (n={len(train_iou)})')
    del mask_model
    torch.cuda.empty_cache()
    return {
        'val': val_iou, 'val_names': val_names,
        'val_first32': f32_iou,
        'train': train_iou, 'train_names': train_names,
    }


def _print_summary_table(results, vq_name=VQ_NAME, decode_mode=DECODE_MODE):
    print('\n' + '=' * 90)
    print(f'  SUMMARY — voxel IoU ({vq_name} tokenizer, {decode_mode} decode)')
    print('=' * 90)
    print(f"{'Checkpoint':<32}  {'val':>10}  {'val[:32]':>10}  {'train':>13}")
    print('-' * 90)
    for label, res in results.items():
        v   = np.array(res['val']).mean() if res['val']         else float('nan')
        f32 = np.array(res['val_first32']).mean() if res.get('val_first32') else float('nan')
        t   = np.array(res['train']).mean() if res['train']     else float('nan')
        v_s   = f'{v:.4f}'   if not np.isnan(v)   else '  n/a   '
        f32_s = f'{f32:.4f}' if not np.isnan(f32) else '  n/a   '
        t_s   = f'{t:.4f}'   if not np.isnan(t)   else '   n/a    '
        print(f'  {label:<30}  {v_s:>10}  {f32_s:>10}  {t_s:>13}')
    print('=' * 90)


def _shard(records, shard_idx, num_shards):
    """Deterministic round-robin shard so any subset is representative."""
    if num_shards <= 1:
        return records
    return [r for i, r in enumerate(records) if i % num_shards == shard_idx]


def combine(shard_paths, out_path):
    """Merge per-shard .npy outputs: val taken from shard 0, train concatenated."""
    shard_paths = sorted(shard_paths)
    assert shard_paths, 'no shard files matched'
    base = dict(np.load(shard_paths[0], allow_pickle=True))
    combined = {}
    for label, res0 in base.items():
        train_iou   = list(res0['train'])
        train_names = list(res0['train_names'])
        for p in shard_paths[1:]:
            d = dict(np.load(p, allow_pickle=True)).get(label)
            if d is None:
                continue
            train_iou   += list(d['train'])
            train_names += list(d['train_names'])
        combined[label] = {
            'val': list(res0['val']), 'val_names': res0['val_names'],
            'val_first32': list(res0['val_first32']),
            'train': train_iou, 'train_names': train_names,
        }
    _print_summary_table(combined)
    np.save(out_path, np.array(list(combined.items()), dtype=object))
    print(f'\nSaved combined per-tower results to {out_path}  ({len(shard_paths)} shards)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--mask_name', nargs='+', default=['myarch_mask_img_patch_v7'],
                        help='One or more mask-transformer dirs under checkpoints/myarch/')
    parser.add_argument('--ckpt', nargs='+', default=['net_best_iou.tar'],
                        help='One or more checkpoint files; applied to every --mask_name')
    parser.add_argument('--vq_name', default=VQ_NAME, help='RVQ/IBQ tokenizer dir')
    parser.add_argument('--data_root', default=DATA_ROOT)
    parser.add_argument('--split', default='val40k.txt',
                        help='Val split filename inside data_root')
    parser.add_argument('--train_split', default='train160k.txt',
                        help='Train split filename inside data_root')
    parser.add_argument('--decode_mode', choices=['windowed', 'whole'], default='windowed',
                        help='windowed-64 for RVQ (default), whole-seq for IBQ')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--val_only', action='store_true', help='Skip train split (faster)')
    parser.add_argument('--limit', type=int, default=0, help='Cap #towers per split (0=all); for smoke tests')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--ceilings', action='store_true',
                        help='Also compute GT Q0-only / full VQ-reconstruction ceilings')
    parser.add_argument('--ceilings_only', action='store_true')
    # Train-split sharding across workers/GPUs. Val + ceilings run on shard 0 only.
    parser.add_argument('--shard_idx', type=int, default=0)
    parser.add_argument('--num_shards', type=int, default=1)
    parser.add_argument('--out_path', type=str, default=None)
    # Combine mode: merge sharded outputs into one table.
    parser.add_argument('--combine', action='store_true')
    parser.add_argument('--shard_paths', nargs='+', default=None,
                        help='Explicit shard .npy paths for --combine (else uses out_path glob)')
    args = parser.parse_args()

    # Apply runtime overrides to module globals threaded through the helpers.
    DATA_ROOT = args.data_root
    VQ_NAME = args.vq_name
    DECODE_MODE = args.decode_mode

    if args.combine:
        paths = args.shard_paths or sorted(__import__('glob').glob(
            (args.out_path or 'eval_mask_shard*.npy')))
        combine(paths, args.out_path or 'eval_mask_combined.npy')
        sys.exit(0)

    fixseed(args.seed)
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}  vq: {VQ_NAME}  decode: {DECODE_MODE}\n')

    vq_model, vq_opt = load_vq_model(device, vq_name=VQ_NAME)
    meta = pjoin(CHECKPOINTS_DIR, DATASET_NAME, VQ_NAME, 'meta')
    mean = np.load(pjoin(meta, 'mean.npy'))
    std  = np.load(pjoin(meta, 'std.npy'))

    print('\nLoading data...')
    # Val + ceilings only on shard 0 (avoids duplicate work across shards).
    val_records = load_records(pjoin(DATA_ROOT, args.split)) if args.shard_idx == 0 else []
    train_records = []
    if not args.val_only:
        train_records = _shard(load_records(pjoin(DATA_ROOT, args.train_split)),
                               args.shard_idx, args.num_shards)
    if args.limit > 0:
        val_records = val_records[:args.limit]
        train_records = train_records[:args.limit]
    val_first32 = val_records[:32]
    print(f'\nLoaded {len(val_records)} val, {len(train_records)} train '
          f'(shard {args.shard_idx+1}/{args.num_shards})')

    results = {}
    if (args.ceilings or args.ceilings_only) and args.shard_idx == 0:
        results.update(_eval_ceilings(val_records, val_first32, train_records, mean, std, vq_model, device))

    if not args.ceilings_only:
        for mask_name in args.mask_name:
            for ckpt_file in args.ckpt:
                label = f'{mask_name.replace("myarch_mask_img_patch_", "")} {ckpt_file.replace(".tar", "")}'
                res = _eval_one_ckpt(label, mask_name, ckpt_file, vq_opt, vq_model, mean, std,
                                     val_records, val_first32, train_records, device)
                if res is not None:
                    results[label] = res

    if results:
        _print_summary_table(results, vq_name=VQ_NAME, decode_mode=DECODE_MODE)

    out_path = args.out_path or (f'eval_mask_shard{args.shard_idx}.npy'
                                 if args.num_shards > 1 else 'eval_mask_full.npy')
    np.save(out_path, np.array(list(results.items()), dtype=object))
    print(f'\nSaved per-tower results to {out_path}')
