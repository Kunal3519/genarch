"""Image-conditioned tower generation — side-by-side mask-transformer comparison.

Per tower the figure has:
    NE image  |  Ground Truth  |  mask_model_1 (Q0)  |  mask_model_2 (Q0)  |  ...

Default compares ``myarch_mask_img_patch_v2`` and ``myarch_mask_img_patch_v4`` on the
``rvq_v2_4x_nc1024_nq3`` tokenizer. Residual transformer is no longer wired in (its
checkpoint was deleted in the May-14 cleanup); CLAUDE.md notes Δ≈-0.0005 IoU at 4x
anyway, so the comparison stays meaningful with Q0-only generation.

Decoder uses the windowed-chunk protocol (utils.vq_decode.decode_windowed) — required
at 4x+ compression to avoid boundary artifacts.
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from utils.paths import DATA_ROOT, IMAGE_ROOT

import argparse
import random
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from os.path import join as pjoin

from models.mask_transformer.transformer import MaskTransformer
from utils.model_io import load_vq
from utils.voxel import grid_snap, voxel_iou, coords_to_voxel_set
from utils.vq_decode import decode_windowed, probe_window_tokens
from utils.get_opt import get_opt
from utils.fixseed import fixseed


DEFAULT_VQ_NAME = 'rvq_v2_4x_nc1024_nq3'
DEFAULT_MASK_NAMES = ['myarch_mask_img_patch_v2', 'myarch_mask_img_patch_v4']

# Per-model colour palette (cycles if more models passed).
MODEL_COLOURS = ['#FFA500', '#FF6692', '#00CC96', '#AB63FA']

# matplotlib camera azimuth matching each dataset photo viewpoint, fixed against the
# realistic PNGs via a full 4-view check (an earlier guess had East/West flipped).
# Full mapping (90° apart): NE=220, NW=130, SE=310, SW=40. Panels display
# northeast_realistic.png, so render from NE_AZIM.
NE_AZIM = 220
VIEW_AZIM = {'northeast': 220, 'northwest': 130, 'southeast': 310, 'southwest': 40}


def read_vq_name_from_opt(opt_path):
    """Read vq_name from a mask transformer's opt.txt without invoking get_opt."""
    with open(opt_path) as f:
        for line in f:
            if line.startswith('vq_name:'):
                return line.split(':', 1)[1].strip()
    return None


# ---------------------------------------------------------------------------
# Model loaders
# ---------------------------------------------------------------------------

def load_vq_model(checkpoints_dir, device, vq_name=DEFAULT_VQ_NAME):
    vq_model, vq_opt = load_vq(checkpoints_dir, vq_name, device=device,
                                 dim_pose=3, dataset_name='myarch')
    vq_opt.dim_pose = 3
    return vq_model, vq_opt


def load_patch_mask_transformer(checkpoints_dir, name, vq_opt, device, ckpt_name='net_best_iou.tar'):
    mask_opt_path = pjoin(checkpoints_dir, 'myarch', name, 'opt.txt')
    mask_opt = get_opt(mask_opt_path, device=device)
    mask_opt.checkpoints_dir = checkpoints_dir
    mask_opt.num_tokens = vq_opt.nb_code
    mask_opt.num_quantizers = vq_opt.num_quantizers
    mask_opt.code_dim = vq_opt.code_dim
    mask_opt.max_motion_len = 512
    mask_opt.max_motion_length = 2000

    if getattr(mask_opt, 'use_cls_prefix', False):
        cond_mode = 'image_patch_clsprefix'
    else:
        cond_mode = 'image_patch'

    transformer = MaskTransformer(
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
        opt=mask_opt)

    ckpt_path = pjoin(checkpoints_dir, 'myarch', name, 'model', ckpt_name)
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state_dict = ckpt.get('t2m_transformer', ckpt.get('trans'))
    if any(k.startswith('module.') for k in state_dict):
        state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}
    transformer.load_state_dict(state_dict, strict=True)
    print(f'Loaded {name} from {ckpt_path} (epoch {ckpt.get("ep", "?")})')
    return transformer


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def draw_block(ax, cx, cy, cz, color='steelblue', alpha=0.85):
    dx, dy, dz = 4.0, 4.0, 8.0
    x0, y0, z0 = cx - dx/2, cy - dy/2, cz - dz/2
    verts = [
        [(x0,y0,z0),      (x0+dx,y0,z0),      (x0+dx,y0+dy,z0),      (x0,y0+dy,z0)],
        [(x0,y0,z0+dz),   (x0+dx,y0,z0+dz),   (x0+dx,y0+dy,z0+dz),   (x0,y0+dy,z0+dz)],
        [(x0,y0,z0),      (x0+dx,y0,z0),       (x0+dx,y0,z0+dz),      (x0,y0,z0+dz)],
        [(x0+dx,y0,z0),   (x0+dx,y0+dy,z0),   (x0+dx,y0+dy,z0+dz),   (x0+dx,y0,z0+dz)],
        [(x0+dx,y0+dy,z0),(x0,y0+dy,z0),       (x0,y0+dy,z0+dz),      (x0+dx,y0+dy,z0+dz)],
        [(x0,y0+dy,z0),   (x0,y0,z0),          (x0,y0,z0+dz),         (x0,y0+dy,z0+dz)],
    ]
    ax.add_collection3d(Poly3DCollection(
        verts, alpha=alpha, facecolor=color, edgecolor='black', linewidth=0.3))


def render_panel(ax, coords, title, x_range, y_range, z_range, color, azim=NE_AZIM, elev=25, zoom=1.0):
    snapped = grid_snap(coords)
    for cx, cy, cz in snapped:
        draw_block(ax, cx, cy, cz, color=color)
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.set_xlim(x_range); ax.set_ylim(y_range); ax.set_zlim(z_range)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.view_init(elev=elev, azim=azim)
    ax.set_box_aspect([1, 1, 2], zoom=zoom)


def render_diff_panel(ax, gt_coords, pred_coords, title, x_range, y_range, z_range, azim=NE_AZIM, elev=25, zoom=1.0):
    """Render prediction-vs-GT diff: green = intersection, red = pred-only, blue = GT-only."""
    gt_set = coords_to_voxel_set(gt_coords)
    pred_set = coords_to_voxel_set(pred_coords)

    intersection = gt_set & pred_set
    pred_only    = pred_set - gt_set
    gt_only      = gt_set - pred_set

    for cx, cy, cz in intersection:
        draw_block(ax, cx, cy, cz, color='#2ECC40', alpha=0.85)
    for cx, cy, cz in pred_only:
        draw_block(ax, cx, cy, cz, color='#FF4136', alpha=0.70)
    for cx, cy, cz in gt_only:
        draw_block(ax, cx, cy, cz, color='#0074D9', alpha=0.50)

    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.set_xlim(x_range); ax.set_ylim(y_range); ax.set_zlim(z_range)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.view_init(elev=elev, azim=azim)
    ax.set_box_aspect([1, 1, 2], zoom=zoom)
    return len(intersection), len(pred_only), len(gt_only)


def axis_limits(coords_list):
    all_pts = np.vstack([grid_snap(c) for c in coords_list])
    xy_span = max(all_pts[:, 0].max() - all_pts[:, 0].min(),
                  all_pts[:, 1].max() - all_pts[:, 1].min()) + 8
    x_mid = (all_pts[:, 0].min() + all_pts[:, 0].max()) / 2
    y_mid = (all_pts[:, 1].min() + all_pts[:, 1].max()) / 2
    x_range = [x_mid - xy_span/2, x_mid + xy_span/2]
    y_range = [y_mid - xy_span/2, y_mid + xy_span/2]
    z_range = [0, all_pts[:, 2].max() + 8]
    return x_range, y_range, z_range


def short_label(name):
    """e.g. 'myarch_mask_img_patch_v4' -> 'v4'."""
    if name.startswith('myarch_mask_img_patch_'):
        return name[len('myarch_mask_img_patch_'):]
    return name


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    device = torch.device(f'cuda:{args.gpu_id}' if args.gpu_id >= 0 else 'cpu')

    # Each mask transformer brings its own VQ tokenizer (per its opt.txt).
    # Cache VQs by name so v2 + v4 share one rvq_v2_4x instance.
    vq_cache = {}      # vq_name -> (vq_model, vq_opt, mean, std, window_tokens)

    def get_vq(vq_name):
        if vq_name in vq_cache:
            return vq_cache[vq_name]
        vqm, vqo = load_vq_model(args.checkpoints_dir, device, vq_name=vq_name)
        vqm.eval().to(device)
        meta = pjoin(args.checkpoints_dir, 'myarch', vq_name, 'meta')
        m = np.load(pjoin(meta, 'mean.npy'))
        s = np.load(pjoin(meta, 'std.npy'))
        wt = probe_window_tokens(vqm, vqo.dim_pose, device)
        print(f'  → VQ {vq_name}: nb_code={vqo.nb_code}, num_q={vqo.num_quantizers}, '
              f'window_tokens={wt}')
        vq_cache[vq_name] = (vqm, vqo, m, s, wt)
        return vq_cache[vq_name]

    mask_models = []   # list of (label, model, vq_name)
    for name in args.mask_names:
        opt_path = pjoin(args.checkpoints_dir, 'myarch', name, 'opt.txt')
        vq_name = read_vq_name_from_opt(opt_path)
        if vq_name is None:
            raise RuntimeError(f'No vq_name in {opt_path}')
        print(f'\n[{name}] paired tokenizer: {vq_name}')
        vqm, vqo, _, _, _ = get_vq(vq_name)
        m = load_patch_mask_transformer(args.checkpoints_dir, name, vqo, device,
                                        ckpt_name=args.mask_ckpt)
        m.eval().to(device)
        mask_models.append((name, m, vq_name))

    def pick_from_split(split_name, n):
        split_file = pjoin(args.data_root, f'{split_name}.txt')
        with open(split_file) as f:
            names = [l.strip() for l in f if l.strip()]
        rng = random.Random(args.seed)
        rng.shuffle(names)
        out = []
        for name in names:
            if (os.path.exists(pjoin(args.data_root, 'new_joint_vecs_int', name + '.npy')) and
                    os.path.exists(pjoin(args.data_root, 'image_features_dino_patch', name + '.npy'))):
                out.append(name)
            if len(out) >= n:
                break
        return out

    selected = [('val',   n) for n in pick_from_split('val',   args.num_towers)] + \
               [('train', n) for n in pick_from_split('train', args.num_towers)]

    if not selected:
        print('No towers found.')
        return

    print(f'\nSelected {len(selected)} towers ({args.num_towers} val + {args.num_towers} train)')
    os.makedirs(args.output_dir, exist_ok=True)

    patch_feats, gt_coords_list, token_lens_list = [], [], []
    names_list, splits_list = [], []

    for split_tag, name in selected:
        motion     = np.load(pjoin(args.data_root, 'new_joint_vecs_int', name + '.npy'))
        patch_feat = np.load(pjoin(args.data_root, 'image_features_dino_patch', name + '.npy'))
        n_blocks   = (len(motion) // 4) * 4
        patch_feats.append(patch_feat.astype(np.float32))
        gt_coords_list.append(motion[:n_blocks].astype(float))
        token_lens_list.append(n_blocks // 4)
        names_list.append(name)
        splits_list.append(split_tag)

    patch_feat_tensor = torch.FloatTensor(np.stack(patch_feats)).to(device)
    token_lens        = torch.LongTensor(token_lens_list).to(device)

    gen_kwargs = dict(
        timesteps=args.timesteps,
        cond_scale=args.cond_scale,
        temperature=args.temperature,
        topk_filter_thres=args.topk,
        gsample=False,
    )

    # Run each mask model through ITS OWN tokenizer; store per-tower Q0-only decoded coords
    # In --compare_decode mode we keep BOTH decodes per model.
    model_coords_windowed = {}   # name -> [B, T, 3]
    model_coords_whole    = {}
    decode_mode = (
        'compare (both windowed and whole-sequence)' if args.compare_decode
        else ('whole-sequence' if args.whole_seq_decode else 'windowed')
    )
    print(f'\nDecode mode: {decode_mode}')
    for name, m, vq_name in mask_models:
        vqm, vqo, m_mean, m_std, wt = vq_cache[vq_name]
        print(f'\nGenerating with {name} (decoded via {vq_name})...')
        with torch.no_grad():
            q0_ids = m.generate(patch_feat_tensor, token_lens, **gen_kwargs)
            q0_ids_safe = q0_ids.clamp(0, vqo.nb_code - 1)
            tokens_with_q = q0_ids_safe.unsqueeze(-1)
            if args.compare_decode:
                model_coords_windowed[name] = (
                    decode_windowed(vqm, tokens_with_q, window_tokens=wt).cpu().numpy()
                    * m_std + m_mean
                )
                model_coords_whole[name] = (
                    vqm.forward_decoder(tokens_with_q).cpu().numpy()
                    * m_std + m_mean
                )
            else:
                if args.whole_seq_decode:
                    decoded = vqm.forward_decoder(tokens_with_q).cpu().numpy()
                else:
                    decoded = decode_windowed(vqm, tokens_with_q, window_tokens=wt).cpu().numpy()
                model_coords_windowed[name] = decoded * m_std + m_mean  # alias single-mode store
    model_coords = model_coords_windowed  # legacy single-mode alias used below

    # Per-tower figures
    n_models = len(mask_models)
    panel_width = 5.5

    for i, (name_str, split_tag) in enumerate(zip(names_list, splits_list)):
        n_blocks = token_lens_list[i] * 4
        gt = gt_coords_list[i]
        ne_path = pjoin(args.image_root, name_str, 'northeast_realistic.png')

        if args.compare_decode:
            # 3 (or n_models) rows × 4 cols layout:
            #   col 0: NE image (full height)   col 1: GT (full height)
            #   col 2: windowed-decode panel    col 3: whole-seq decode panel
            per_model = [(mn, model_coords_windowed[mn][i, :n_blocks],
                              model_coords_whole[mn][i, :n_blocks])
                         for mn, _, _ in mask_models]

            ranges_inputs = [gt] + [w for _, w, _ in per_model] + [h for _, _, h in per_model]
            x_range, y_range, z_range = axis_limits(ranges_inputs)

            fig = plt.figure(figsize=(panel_width * 4 + 2, panel_width * n_models + 1))
            gs = fig.add_gridspec(n_models, 4, width_ratios=[1.0, 1.0, 1.2, 1.2])

            ax_img = fig.add_subplot(gs[:, 0])
            if os.path.exists(ne_path):
                ax_img.imshow(mpimg.imread(ne_path))
            else:
                ax_img.text(0.5, 0.5, 'image not found', ha='center', va='center',
                            transform=ax_img.transAxes)
            ax_img.set_title(f'{name_str}\n[{split_tag}]', fontsize=12, fontweight='bold')
            ax_img.axis('off')

            ax_gt = fig.add_subplot(gs[:, 1], projection='3d')
            render_panel(ax_gt, gt,
                         f'Ground Truth\n{n_blocks} blocks',
                         x_range, y_range, z_range, color='#636EFA')

            gt_set = coords_to_voxel_set(gt)
            row_iou_strs = []
            for r, (mn, win_pred, whole_pred) in enumerate(per_model):
                lbl = short_label(mn)

                # Windowed
                ax_w = fig.add_subplot(gs[r, 2], projection='3d')
                iou_w = voxel_iou(win_pred, gt)
                pset_w = coords_to_voxel_set(win_pred)
                n_int_w   = len(gt_set & pset_w)
                n_extra_w = len(pset_w - gt_set)
                n_miss_w  = len(gt_set - pset_w)
                title_w = (f'{lbl} — WINDOWED  IoU={iou_w:.3f}\n'
                           f'∩={n_int_w}  +{n_extra_w}  -{n_miss_w}')
                render_diff_panel(ax_w, gt, win_pred, title_w, x_range, y_range, z_range)

                # Whole-seq
                ax_h = fig.add_subplot(gs[r, 3], projection='3d')
                iou_h = voxel_iou(whole_pred, gt)
                pset_h = coords_to_voxel_set(whole_pred)
                n_int_h   = len(gt_set & pset_h)
                n_extra_h = len(pset_h - gt_set)
                n_miss_h  = len(gt_set - pset_h)
                title_h = (f'{lbl} — WHOLE-SEQ  IoU={iou_h:.3f}\n'
                           f'∩={n_int_h}  +{n_extra_h}  -{n_miss_h}')
                render_diff_panel(ax_h, gt, whole_pred, title_h, x_range, y_range, z_range)

                row_iou_strs.append(f'{lbl}: win={iou_w:.3f}  whole={iou_h:.3f}')

            fig.suptitle(f'{name_str} [{split_tag}]   '
                         + '   |   '.join(row_iou_strs)
                         + f'   cond_scale={args.cond_scale}',
                         fontsize=13, fontweight='bold')
            plt.tight_layout()
            out_fig = pjoin(args.output_dir, f'{split_tag}_{name_str}_winVSwhole.png')
            plt.savefig(out_fig, dpi=150, bbox_inches='tight')
            plt.close()
            print(f'  [{split_tag}] {name_str}:  ' + '  '.join(row_iou_strs) + f'  -> {out_fig}')
            continue  # skip the legacy single-mode block below

        # ---- Legacy single-mode figure (image + GT + 1 panel per model) ----
        n_panels = 2 + n_models
        per_model_pred = [(mn, mc[i, :n_blocks]) for mn, mc in model_coords.items()]
        ranges_inputs = [gt] + [p for _, p in per_model_pred]
        x_range, y_range, z_range = axis_limits(ranges_inputs)
        fig = plt.figure(figsize=(panel_width * n_panels, 7))

        ax_img = fig.add_subplot(1, n_panels, 1)
        if os.path.exists(ne_path):
            ax_img.imshow(mpimg.imread(ne_path))
        else:
            ax_img.text(0.5, 0.5, 'image not found', ha='center', va='center',
                        transform=ax_img.transAxes)
        ax_img.set_title(f'{name_str}\n[{split_tag}]', fontsize=11, fontweight='bold')
        ax_img.axis('off')

        ax_gt = fig.add_subplot(1, n_panels, 2, projection='3d')
        render_panel(ax_gt, gt,
                     f'Ground Truth\n{n_blocks} blocks',
                     x_range, y_range, z_range, color='#636EFA')

        ious = []
        gt_set_for_title = coords_to_voxel_set(gt) if args.color_diff else None
        for k, (mn, pred) in enumerate(per_model_pred):
            ax = fig.add_subplot(1, n_panels, 3 + k, projection='3d')
            iou = voxel_iou(pred, gt)
            ious.append((short_label(mn), iou))

            if args.color_diff:
                pred_set = coords_to_voxel_set(pred)
                n_int   = len(gt_set_for_title & pred_set)
                n_extra = len(pred_set - gt_set_for_title)
                n_miss  = len(gt_set_for_title - pred_set)
                title = (f'{short_label(mn)} (Q0)  IoU={iou:.3f}\n'
                         f'∩={n_int}  +{n_extra} extra  -{n_miss} miss')
                render_diff_panel(ax, gt, pred, title, x_range, y_range, z_range)
            else:
                colour = MODEL_COLOURS[k % len(MODEL_COLOURS)]
                render_panel(ax, pred,
                             f'{short_label(mn)} (Q0)\nIoU={iou:.3f}',
                             x_range, y_range, z_range, color=colour)

        iou_str = '  '.join(f'{lbl}={v:.3f}' for lbl, v in ious)
        fig.suptitle(f'{name_str} [{split_tag}]   {iou_str}   '
                     f'cond_scale={args.cond_scale}',
                     fontsize=13, fontweight='bold')
        plt.tight_layout()
        out_fig = pjoin(args.output_dir, f'{split_tag}_{name_str}_compare.png')
        plt.savefig(out_fig, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'  [{split_tag}] {name_str}:  {iou_str}  -> {out_fig}')

    print(f'\nDone. {len(names_list)} figures saved to {args.output_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints')
    parser.add_argument('--data_root',       type=str, default=DATA_ROOT)
    parser.add_argument('--image_root',      type=str, default=IMAGE_ROOT)
    parser.add_argument('--output_dir',      type=str, default='./generation/v2_vs_v4')
    parser.add_argument('--gpu_id',          type=int, default=0)
    parser.add_argument('--num_towers',      type=int, default=8,
                        help='Number per split (16 → 16 val + 16 train = 32 figures).')
    parser.add_argument('--vq_name',         type=str, default=DEFAULT_VQ_NAME)
    parser.add_argument('--mask_names',      nargs='+', default=DEFAULT_MASK_NAMES,
                        help='Mask transformer checkpoint dir names (each will get its own panel).')
    parser.add_argument('--mask_ckpt',       type=str, default='net_best_iou.tar')
    parser.add_argument('--timesteps',       type=int, default=10)
    parser.add_argument('--temperature',     type=float, default=0.2)
    parser.add_argument('--topk',            type=float, default=0.9)
    parser.add_argument('--cond_scale',      type=float, default=1.0)
    parser.add_argument('--seed',            type=int, default=42)
    parser.add_argument('--whole_seq_decode', action='store_true',
                        help='Decode the whole token sequence in one shot (old behaviour). '
                             'Default is windowed-chunk decode which avoids boundary artifacts at 4x+ compression.')
    parser.add_argument('--color_diff', action='store_true',
                        help='Render model panels as GT-vs-prediction diff: '
                             'green = intersection, red = pred-only (extra), blue = GT-only (missed).')
    parser.add_argument('--compare_decode', action='store_true',
                        help='Run BOTH windowed and whole-seq decode per model and produce a single '
                             '"windowed vs whole-seq" figure per tower (rows = models, cols = decode modes).')
    args = parser.parse_args()
    fixseed(args.seed)
    main(args)
