"""Compare windowed vs whole-sequence decode of *generated* tokens.

Question: GT-token ceilings show windowed > whole-seq, but the user observes
that generated towers look MORE disjointed under windowed decode (with floating
artifact blocks) than under whole-seq decode. Is that effect from the mask
transformer's prediction noise, or from the RVQ decoder's behavior at window
boundaries?

For each (mask transformer, RVQ) pair we evaluate:
  - IoU of generated tokens, decoded both windowed (w=64 blocks) and whole-seq
  - 6-connected component count of the snapped voxel set (proxy for "floating
    artifact blocks") for the same two decode modes
  - Same two metrics on GT-encoded tokens as a control: any boundary artifacts
    visible there are pure RVQ-decoder boundary effects.

Pairs evaluated:
  0: myarch_mask_img_patch_v1  + rvq_newarch            (nb=512,  num_q=6)
  1: myarch_mask_img_patch_v4  + rvq_v2_4x_nc1024_nq3   (nb=1024, num_q=3)

Sharding: --shard_idx K --num_shards N takes val_records[K::N] per worker.
Combine: --combine reads test_decode_artifacts_shard*.npy and prints the 2x2 table.

Usage:
    # 5-way shard across GPUs 2..6
    for K in 0 1 2 3 4; do
      CUDA_VISIBLE_DEVICES=$((K+2)) uv run python \
          script/test_decode_artifacts.py --shard_idx $K --num_shards 5 --gpu_id 0 &
    done; wait

    uv run python script/test_decode_artifacts.py --combine
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from utils.paths import DATA_ROOT

import argparse
import glob
import torch
import numpy as np
from os.path import join as pjoin
from tqdm import tqdm

from models.mask_transformer.transformer import MaskTransformer
from utils.get_opt import get_opt
from utils.fixseed import fixseed
from utils.model_io import load_vq
from utils.voxel import grid_snap, GRID
from utils.vq_decode import decode_windowed


CHECKPOINTS_DIR = './checkpoints'
DATA_ROOT       = DATA_ROOT
DATASET_NAME    = 'myarch'
DIM_POSE        = 3
UNIT_LENGTH     = 4
BATCH_SIZE      = 32

MASK_TIMESTEPS   = 10
MASK_COND_SCALE  = 1.0
MASK_TEMPERATURE = 0.2
MASK_TOPK        = 0.9
MAX_MOTION_LEN   = 2000

PAIRS = [
    # (label, mask_name, mask_ckpt, vq_name)
    ('v1+rvq_newarch',  'myarch_mask_img_patch_v1', 'net_best_iou.tar', 'rvq_newarch'),
    ('v4+rvq_v2_4x',    'myarch_mask_img_patch_v4', 'latest.tar',       'rvq_v2_4x_nc1024_nq3'),
]


# --------------------------------------------------------------------------- #
# Connectivity metric                                                         #
# --------------------------------------------------------------------------- #

def count_components_6conn(voxel_set):
    """6-connected (face-adjacent) component analysis of a voxel set.

    Returns (n_components, largest_component_size, n_floating_blocks).
    n_floating_blocks = total voxels minus the largest component =
      everything not attached (even transitively) to the main mass.
    """
    if not voxel_set:
        return 0, 0, 0
    dx, dy, dz = GRID
    neighbors = [(dx, 0, 0), (-dx, 0, 0),
                 (0, dy, 0), (0, -dy, 0),
                 (0, 0, dz), (0, 0, -dz)]
    visited = set()
    sizes = []
    for start in voxel_set:
        if start in visited:
            continue
        stack = [start]
        size = 0
        while stack:
            v = stack.pop()
            if v in visited:
                continue
            visited.add(v)
            size += 1
            for d in neighbors:
                nb = (v[0] + d[0], v[1] + d[1], v[2] + d[2])
                if nb in voxel_set and nb not in visited:
                    stack.append(nb)
        sizes.append(size)
    n_comp = len(sizes)
    largest = max(sizes)
    return n_comp, largest, sum(sizes) - largest


def voxel_set_and_iou(coords_pred, coords_gt):
    """Snap, dedupe, compute IoU and the predicted voxel set."""
    sp = grid_snap(coords_pred)
    sg = grid_snap(coords_gt)
    pred_set = set(map(tuple, sp.tolist()))
    gt_set   = set(map(tuple, sg.tolist()))
    if not pred_set and not gt_set:
        iou = 1.0
    elif not pred_set or not gt_set:
        iou = 0.0
    else:
        iou = len(pred_set & gt_set) / len(pred_set | gt_set)
    return iou, pred_set, gt_set


# --------------------------------------------------------------------------- #
# Model loading                                                               #
# --------------------------------------------------------------------------- #

def load_vq(vq_name, device):
    vq_model, vq_opt = load_vq(CHECKPOINTS_DIR, vq_name, device=device,
                                 dim_pose=DIM_POSE, dataset_name=DATASET_NAME)
    vq_model.eval().to(device)
    return vq_model, vq_opt


def load_mask(mask_name, ckpt_file, vq_opt, device):
    opt_path = pjoin(CHECKPOINTS_DIR, DATASET_NAME, mask_name, 'opt.txt')
    mask_opt = get_opt(opt_path, device)
    mask_opt.num_tokens        = vq_opt.nb_code
    mask_opt.num_quantizers    = vq_opt.num_quantizers
    mask_opt.code_dim          = vq_opt.code_dim
    mask_opt.max_motion_len    = 512
    mask_opt.max_motion_length = MAX_MOTION_LEN

    model = MaskTransformer(
        code_dim=vq_opt.code_dim,
        cond_mode='image_patch',
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
    ckpt_path = pjoin(CHECKPOINTS_DIR, DATASET_NAME, mask_name, 'model', ckpt_file)
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state_dict = ckpt.get('t2m_transformer', ckpt)
    if any(k.startswith('module.') for k in state_dict):
        state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    model.eval().to(device)
    return model


# --------------------------------------------------------------------------- #
# Data                                                                        #
# --------------------------------------------------------------------------- #

def load_records(split_file):
    motion_dir     = pjoin(DATA_ROOT, 'new_joint_vecs_int')
    patch_feat_dir = pjoin(DATA_ROOT, 'image_features_dino_patch')
    with open(split_file, 'r') as f:
        names = [l.strip() for l in f if l.strip()]
    records = []
    for name in names:
        m_path = pjoin(motion_dir, name + '.npy')
        f_path = pjoin(patch_feat_dir, name + '.npy')
        if not os.path.exists(m_path) or not os.path.exists(f_path):
            continue
        motion_raw = np.load(m_path).astype(np.float32)
        m_len = (len(motion_raw) // UNIT_LENGTH) * UNIT_LENGTH
        if m_len == 0:
            continue
        patch_feat = np.load(f_path).astype(np.float32)
        records.append((name, motion_raw[:m_len], patch_feat))
    return records


# --------------------------------------------------------------------------- #
# Decode + metrics                                                            #
# --------------------------------------------------------------------------- #

@torch.no_grad()
def decode_whole_seq_q0(vq_model, code_idx_q0):
    """Whole-sequence decode of Q0-only tokens. Input [B, T, 1]."""
    return vq_model.forward_decoder(code_idx_q0)


@torch.no_grad()
def encode_full_windowed(vq_model, motion_norm_1d, window_blocks=64):
    """GT-token encode, windowed (matches training protocol). Returns [T_tok] long tensor."""
    L = motion_norm_1d.shape[0]
    pad = (window_blocks - L % window_blocks) % window_blocks
    if L < window_blocks:
        pad = window_blocks - L
    arr = motion_norm_1d
    if pad:
        arr = np.concatenate([arr, np.zeros((pad, arr.shape[1]), dtype=arr.dtype)], axis=0)
    device = vq_model.quantizer.layers[0].codebook.device
    all_tokens = []
    for s in range(0, arr.shape[0], window_blocks):
        chunk = torch.from_numpy(arr[s:s + window_blocks]).unsqueeze(0).to(device)
        code_idx, _ = vq_model.encode(chunk)   # [1, T_tok_chunk, Q]
        all_tokens.append(code_idx[0])
    full = torch.cat(all_tokens, dim=0)        # [T_tok_total, Q]
    return full  # int64


def metrics_for_decode(coords_pred_norm, m_len_blocks, mean, std, gt_raw):
    """Crop predicted coords to gt length, denormalize, compute IoU + components."""
    n_cmp = (m_len_blocks // UNIT_LENGTH) * UNIT_LENGTH
    gen_xyz = coords_pred_norm[:n_cmp] * std + mean
    gt_xyz  = gt_raw[:n_cmp]
    iou, pred_set, gt_set = voxel_set_and_iou(gen_xyz, gt_xyz)
    n_comp_p, _, n_floating_p = count_components_6conn(pred_set)
    n_comp_g, _, n_floating_g = count_components_6conn(gt_set)
    return {
        'iou': iou,
        'n_voxels_pred': len(pred_set),
        'n_voxels_gt':   len(gt_set),
        'n_comp_pred':   n_comp_p,
        'n_floating_pred': n_floating_p,
        'n_comp_gt':     n_comp_g,
        'n_floating_gt': n_floating_g,
    }


@torch.no_grad()
def eval_pair_on_records(records, mean, std, mask_model, vq_model, device):
    """For each record produce: generated/windowed, generated/whole, GT/windowed, GT/whole metrics."""
    nb_code = vq_model.quantizer.layers[0].nb_code
    num_q   = vq_model.num_quantizers if hasattr(vq_model, 'num_quantizers') else len(vq_model.quantizer.layers)

    out = {'gen_windowed': [], 'gen_whole': [], 'gt_windowed': [], 'gt_whole': []}
    names = []

    # ---- Generated tokens (mask transformer) — batched ----
    for batch_start in tqdm(range(0, len(records), BATCH_SIZE), desc='  gen', leave=False):
        batch = records[batch_start: batch_start + BATCH_SIZE]
        motions_b = [r[1] for r in batch]
        feats_b   = [r[2] for r in batch]
        m_lens    = [len(m) for m in motions_b]
        cond      = torch.from_numpy(np.stack(feats_b)).to(device)
        m_lens_tok = torch.tensor([ml // UNIT_LENGTH for ml in m_lens], device=device, dtype=torch.long)

        gen_ids = mask_model.generate(
            cond, m_lens_tok,
            timesteps=MASK_TIMESTEPS,
            cond_scale=MASK_COND_SCALE,
            temperature=MASK_TEMPERATURE,
            topk_filter_thres=MASK_TOPK,
            gsample=False,
        )
        gen_ids = gen_ids.clamp(0, nb_code - 1).unsqueeze(-1)   # [B, T_tok, 1]

        # windowed
        coords_W = decode_windowed(vq_model, gen_ids).cpu().numpy()
        # whole-sequence
        coords_S = decode_whole_seq_q0(vq_model, gen_ids).cpu().numpy()

        for i, m_len in enumerate(m_lens):
            out['gen_windowed'].append(metrics_for_decode(coords_W[i], m_len, mean, std, motions_b[i]))
            out['gen_whole'].append(   metrics_for_decode(coords_S[i], m_len, mean, std, motions_b[i]))
            names.append(batch[i][0])

    # ---- GT tokens (encoded), decoded the same two ways. Per-tower since windowed encode requires per-tower padding. ----
    for (name, motion_raw, _) in tqdm(records, desc='  gt', leave=False):
        m_len = len(motion_raw)
        motion_norm = (motion_raw - mean) / std

        # encode windowed (matches training); use Q0 only for apples-to-apples
        full_tokens = encode_full_windowed(vq_model, motion_norm)   # [T_tok_total, Q]
        q0_tokens   = full_tokens[:, 0:1].unsqueeze(0)              # [1, T_tok, 1]
        T_tok_blocks = (m_len // UNIT_LENGTH)
        # forward_decoder will produce m_tok * down_factor blocks; we crop to m_len in metrics_for_decode
        coords_W_gt = decode_windowed(vq_model, q0_tokens).cpu().numpy()[0]
        coords_S_gt = decode_whole_seq_q0(vq_model, q0_tokens).cpu().numpy()[0]
        out['gt_windowed'].append(metrics_for_decode(coords_W_gt, m_len, mean, std, motion_raw))
        out['gt_whole'].append(   metrics_for_decode(coords_S_gt, m_len, mean, std, motion_raw))

    return names, out


# --------------------------------------------------------------------------- #
# Combine + summary                                                           #
# --------------------------------------------------------------------------- #

def _agg(rows, key):
    return np.array([r[key] for r in rows], dtype=np.float64)


def summarise_pair(label, results):
    print('\n' + '=' * 100)
    print(f'  {label}')
    print('=' * 100)
    header = f"{'mode':<14}  {'IoU':>7}  {'n_voxels':>8}  {'n_comp':>7}  {'n_floating':>10}  {'%towers ≥2 comp':>16}"
    print(header)
    print('-' * 100)
    for mode in ['gen_windowed', 'gen_whole', 'gt_windowed', 'gt_whole']:
        rows = results[mode]
        if not rows:
            continue
        ious  = _agg(rows, 'iou')
        nvox  = _agg(rows, 'n_voxels_pred')
        ncomp = _agg(rows, 'n_comp_pred')
        nfl   = _agg(rows, 'n_floating_pred')
        frac_multi = (ncomp >= 2).mean() * 100
        print(f"{mode:<14}  {ious.mean():>7.4f}  {nvox.mean():>8.1f}  {ncomp.mean():>7.2f}  {nfl.mean():>10.2f}  {frac_multi:>15.1f}%")
    print('-' * 100)
    # Direction: how much does each model gain from whole-seq smoothing on prediction noise?
    ious_W = _agg(results['gen_windowed'], 'iou')
    ious_S = _agg(results['gen_whole'],    'iou')
    nfl_W  = _agg(results['gen_windowed'], 'n_floating_pred')
    nfl_S  = _agg(results['gen_whole'],    'n_floating_pred')
    print(f"  gen IoU  windowed − whole  = {ious_W.mean() - ious_S.mean():+.4f}")
    print(f"  gen floating blocks  windowed − whole = {nfl_W.mean() - nfl_S.mean():+.2f}")
    ious_gW = _agg(results['gt_windowed'], 'iou')
    ious_gS = _agg(results['gt_whole'],    'iou')
    nfl_gW  = _agg(results['gt_windowed'], 'n_floating_pred')
    nfl_gS  = _agg(results['gt_whole'],    'n_floating_pred')
    print(f"  GT  IoU  windowed − whole  = {ious_gW.mean() - ious_gS.mean():+.4f}    (control: pure RVQ effect)")
    print(f"  GT  floating blocks  windowed − whole = {nfl_gW.mean() - nfl_gS.mean():+.2f}")


def combine_shards():
    shard_files = sorted(glob.glob('test_decode_artifacts_shard*.npy'))
    if not shard_files:
        print('No shard files found.')
        return
    print(f'Combining {len(shard_files)} shard(s): {shard_files}')
    merged = {}
    all_names = {}
    for sf in shard_files:
        data = np.load(sf, allow_pickle=True).item()
        for pair_label, payload in data.items():
            merged.setdefault(pair_label, {k: [] for k in ['gen_windowed', 'gen_whole', 'gt_windowed', 'gt_whole']})
            all_names.setdefault(pair_label, [])
            for k in merged[pair_label]:
                merged[pair_label][k].extend(payload['results'][k])
            all_names[pair_label].extend(payload['names'])
    for pair_label, results in merged.items():
        n_unique = len(set(all_names[pair_label]))
        print(f'\n{pair_label}: {len(all_names[pair_label])} rows ({n_unique} unique towers)')
        summarise_pair(pair_label, results)
    np.save('test_decode_artifacts_combined.npy',
            {'results': merged, 'names': all_names}, allow_pickle=True)
    print('\nSaved combined results to test_decode_artifacts_combined.npy')


# --------------------------------------------------------------------------- #
# Entry point                                                                 #
# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu_id', type=int, default=0)
    ap.add_argument('--shard_idx', type=int, default=0)
    ap.add_argument('--num_shards', type=int, default=1)
    ap.add_argument('--pair_idx', type=int, default=-1,
                    help='-1 = both pairs; 0 = v1+rvq_newarch only; 1 = v4+rvq_v2_4x only')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--combine', action='store_true')
    args = ap.parse_args()

    if args.combine:
        combine_shards()
        return

    fixseed(args.seed)
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}  shard {args.shard_idx + 1}/{args.num_shards}')

    print('\nLoading val records...')
    val_records = load_records(pjoin(DATA_ROOT, 'val40k.txt'))
    shard_records = val_records[args.shard_idx::args.num_shards]
    print(f'  total val = {len(val_records)}, this shard = {len(shard_records)}')

    pairs_to_run = PAIRS if args.pair_idx < 0 else [PAIRS[args.pair_idx]]
    out = {}
    for label, mask_name, mask_ckpt, vq_name in pairs_to_run:
        print(f'\n── {label} ──')
        vq_model, vq_opt = load_vq(vq_name, device)
        mean = np.load(pjoin(CHECKPOINTS_DIR, DATASET_NAME, vq_name, 'meta', 'mean.npy'))
        std  = np.load(pjoin(CHECKPOINTS_DIR, DATASET_NAME, vq_name, 'meta', 'std.npy'))
        mask_model = load_mask(mask_name, mask_ckpt, vq_opt, device)
        names, results = eval_pair_on_records(shard_records, mean, std, mask_model, vq_model, device)
        out[label] = {'names': names, 'results': results}
        summarise_pair(label, results)
        del mask_model, vq_model
        torch.cuda.empty_cache()

    out_path = f'test_decode_artifacts_shard{args.shard_idx}of{args.num_shards}.npy'
    np.save(out_path, out, allow_pickle=True)
    print(f'\nSaved shard results to {out_path}')


if __name__ == '__main__':
    main()
