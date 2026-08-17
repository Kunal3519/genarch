"""
VIEW FUSION: condition on two views taken from two DIFFERENT towers and ask what
the model builds.

Setup per pair: tower A supplies its NE view, tower B supplies its SW view, the
other two slots are zeroed. NE and SW are opposite corners, so the two views share
no facade and the evidence they carry is complementary rather than redundant.

The interesting question is not "which IoU is higher" but WHERE each source shows
up, so every predicted block is attributed against the two ground truths:

    A-only   block in A but not in B   -> evidence that A won locally
    B-only   block in B but not in A   -> evidence that B won locally
    shared   block in A and in B       -> ambiguous, both towers agree
    neither  block in no ground truth  -> invented

Only the A-only and B-only regions can distinguish the sources; a model that
simply built the "average tower" would score well on shared and poorly on both
exclusive regions. Length is set to the MEAN of the two towers' token counts so
neither source is favoured by construction.

  CUDA_VISIBLE_DEVICES=<uuid> uv run python \
      script/gen_view_fusion.py --n_pairs 8 --views NE,SW
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.paths import IMAGE_ROOT
from os.path import join as pjoin

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

from utils.model_io import load_vq
from utils.voxel import grid_snap, coords_to_voxel_set
from utils.fixseed import fixseed
from eval_sampling import (load_mask_model, CHECKPOINTS_DIR, DATASET_NAME, DATA_ROOT,
                           MASK_TIMESTEPS, MASK_COND_SCALE, MASK_TOPK)
from _common import gen_drafts, decode_xyz, vote_set
from gen_img_towers import draw_block, VIEW_AZIM

VIEWS = ['NE', 'NW', 'SE', 'SW']
VIEW_FILE = ['northeast', 'northwest', 'southeast', 'southwest']
IMAGE_ROOT = IMAGE_ROOT
VQ_NAME = 'ibq_4x_nc1024_nq1'
MASK_NAME = 'myarch_mask_img_patch_v8_2'

C_A, C_B, C_SHARED, C_NEW = '#0074D9', '#FF851B', '#2ECC40', '#FF4136'


def render_attrib(ax, pred, A, B, xr, yr, zr, azim, zoom=1.0):
    """Colour every predicted block by which ground truth explains it."""
    a_only, b_only = A - B, B - A
    for v in pred:
        c = (C_SHARED if v in A and v in B else
             C_A if v in a_only else
             C_B if v in b_only else C_NEW)
        draw_block(ax, v[0], v[1], v[2], color=c, alpha=0.9)
    ax.set_xlim(xr); ax.set_ylim(yr); ax.set_zlim(zr)
    ax.view_init(elev=25, azim=azim)
    ax.set_box_aspect([1, 1, 2], zoom=zoom)
    ax.set_axis_off()


def plain(ax, coords, xr, yr, zr, azim, color, zoom=1.0):
    for cx, cy, cz in coords_to_voxel_set(coords):
        draw_block(ax, cx, cy, cz, color=color, alpha=0.85)
    ax.set_xlim(xr); ax.set_ylim(yr); ax.set_zlim(zr)
    ax.view_init(elev=25, azim=azim)
    ax.set_box_aspect([1, 1, 2], zoom=zoom)
    ax.set_axis_off()


def photo(ax, tower, v, image_root):
    p = pjoin(image_root, tower, f'{VIEW_FILE[v]}_realistic.png')
    ax.imshow(np.asarray(Image.open(p).convert('RGB').resize((256, 320))))
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor('#2ca02c'); s.set_linewidth(3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n_pairs', type=int, default=8)
    ap.add_argument('--views', default='NE,SW', help='view from A, view from B')
    ap.add_argument('--split', default=pjoin(DATA_ROOT, 'val40k.txt'))
    ap.add_argument('--n_samples', type=int, default=16)
    ap.add_argument('--temperature', type=float, default=0.8)
    ap.add_argument('--threshold', type=float, default=0.4375)
    ap.add_argument('--azim', default='northeast')
    ap.add_argument('--image_root', default=IMAGE_ROOT)
    ap.add_argument('--output_dir', default='./generation/view_fusion')
    ap.add_argument('--gpu_id', type=int, default=0)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()

    fixseed(a.seed)
    device = torch.device(f'cuda:{a.gpu_id}')
    vA, vB = [VIEWS.index(v.strip()) for v in a.views.split(',')]

    vq, vq_opt = load_vq(CHECKPOINTS_DIR, VQ_NAME, device=device, dim_pose=3,
                          dataset_name=DATASET_NAME)
    vq.eval().to(device)
    nb_code = vq_opt.nb_code
    model = load_mask_model(vq_opt, MASK_NAME, 'net_best_iou.tar', device)
    meta = pjoin(CHECKPOINTS_DIR, DATASET_NAME, VQ_NAME, 'meta')
    mean = np.load(pjoin(meta, 'mean.npy')).reshape(-1)[:3]
    std = np.load(pjoin(meta, 'std.npy')).reshape(-1)[:3]

    names = sorted(l.strip() for l in open(a.split) if l.strip())
    rng = np.random.default_rng(a.seed); rng.shuffle(names)
    names = [n for n in names
             if os.path.exists(pjoin(DATA_ROOT, 'image_features_dino_patch', n + '.npy'))
             and os.path.exists(pjoin(a.image_root, n))][:2 * a.n_pairs]
    pairs = [(names[2 * i], names[2 * i + 1]) for i in range(a.n_pairs)]

    def load(nm):
        m = np.load(pjoin(DATA_ROOT, 'new_joint_vecs_int', nm + '.npy')).reshape(-1, 3)
        n = (len(m) // 4) * 4
        f = np.load(pjoin(DATA_ROOT, 'image_features_dino_patch', nm + '.npy')).astype(np.float32)
        return m[:n].astype(float), f, n // 4

    rows, recs = [], []
    for A_nm, B_nm in pairs:
        gtA, fA, LA = load(A_nm)
        gtB, fB, LB = load(B_nm)
        L = int(round((LA + LB) / 2))                 # neither source favoured

        cond = np.zeros_like(fA)
        cond[vA] = fA[vA]
        cond[vB] = fB[vB]
        drafts = gen_drafts(model, cond, L, a.n_samples, a.temperature, device, nb_code)
        xyz = decode_xyz(vq, drafts, L * 4, mean, std)
        pred = vote_set([coords_to_voxel_set(x) for x in xyz],
                        a.threshold * a.n_samples)

        SA, SB = coords_to_voxel_set(gtA), coords_to_voxel_set(gtB)
        a_only, b_only, both = SA - SB, SB - SA, SA & SB
        n = max(len(pred), 1)

        # ── spatial attribution ────────────────────────────────────────────
        # Each camera sits at its render azimuth, so the direction from the
        # building centre towards camera v is (cos, sin) of that azimuth. The
        # axis that separates the two cameras is dirA - dirB; projecting a block
        # onto it gives a signed "towards A / towards B" coordinate, normalised
        # by the footprint half-span so it is comparable across towers.
        # A raw mean is not enough: the exclusive regions are themselves
        # spatially biased, so each recovered set is compared against the mean of
        # the FULL region it came from. A positive shift for A-only and a
        # negative one for B-only means each view drove the side it can see.
        def _dir(vi):
            th = np.deg2rad(VIEW_AZIM[VIEW_FILE[vi]])
            return np.array([np.cos(th), np.sin(th)])
        axis = _dir(vA) - _dir(vB)
        axis = axis / (np.linalg.norm(axis) or 1.0)
        allxy = np.array(sorted(SA | SB), float)[:, :2]
        centre = (allxy.min(0) + allxy.max(0)) / 2
        half = max(np.ptp(allxy[:, 0]), np.ptp(allxy[:, 1])) / 2 or 1.0

        def proj(S):
            if not S:
                return float('nan')
            P = np.array(sorted(S), float)[:, :2]
            return float(((P - centre) @ axis).mean() / half)

        pA, pB = proj(pred & a_only), proj(pred & b_only)
        bA, bB = proj(a_only), proj(b_only)
        rec = dict(
            proj_recovered_A=pA, proj_recovered_B=pB,
            proj_region_A=bA, proj_region_B=bB,
            shift_A=pA - bA, shift_B=pB - bB,
            separation=(pA - pB) - (bA - bB),
            A=A_nm, B=B_nm,
            iou_A=len(pred & SA) / len(pred | SA),
            iou_B=len(pred & SB) / len(pred | SB),
            # what fraction of each EXCLUSIVE region did the prediction recover
            rec_A_only=len(pred & a_only) / max(len(a_only), 1),
            rec_B_only=len(pred & b_only) / max(len(b_only), 1),
            # what is the prediction made of
            frac_A_only=len(pred & a_only) / n,
            frac_B_only=len(pred & b_only) / n,
            frac_shared=len(pred & both) / n,
            frac_neither=len(pred - SA - SB) / n)
        recs.append(rec)
        rows.append((A_nm, B_nm, gtA, gtB, pred, rec))
        print(f'{A_nm[:14]:>14} + {B_nm[:14]:<14} IoU_A={rec["iou_A"]:.3f} '
              f'IoU_B={rec["iou_B"]:.3f}  recall A-only={rec["rec_A_only"]:.3f} '
              f'B-only={rec["rec_B_only"]:.3f}', flush=True)

    # ── figure: one row per pair ──
    allc = np.vstack([grid_snap(np.vstack([r[2], r[3]])) for r in rows] +
                     [np.array(sorted(r[4]), float) for r in rows if r[4]])
    mxy = int(np.ceil(max(allc[:, 0].max(), allc[:, 1].max()) / 10.) * 10)
    mz = int(np.ceil(allc[:, 2].max() / 25.) * 25)
    xr = yr = [0, mxy]; zr = [0, mz]
    az = VIEW_AZIM[a.azim]

    ncols, R = 5, len(rows)
    fig = plt.figure(figsize=(3.3 * ncols, 3.5 * R))
    for r, (A_nm, B_nm, gtA, gtB, pred, rec) in enumerate(rows):
        ax = fig.add_subplot(R, ncols, r * ncols + 1); photo(ax, A_nm, vA, a.image_root)
        if r == 0:
            ax.set_title(f'tower A: {VIEWS[vA]} view', fontsize=13, fontweight='bold')
        ax.set_ylabel(f'IoU$_A$ {rec["iou_A"]:.3f}\nIoU$_B$ {rec["iou_B"]:.3f}',
                      fontsize=12, fontweight='bold')
        ax = fig.add_subplot(R, ncols, r * ncols + 2); photo(ax, B_nm, vB, a.image_root)
        if r == 0:
            ax.set_title(f'tower B: {VIEWS[vB]} view', fontsize=13, fontweight='bold')
        ax = fig.add_subplot(R, ncols, r * ncols + 3, projection='3d')
        plain(ax, gtA, xr, yr, zr, az, C_A)
        if r == 0:
            ax.set_title('ground truth A', fontsize=13, fontweight='bold')
        ax = fig.add_subplot(R, ncols, r * ncols + 4, projection='3d')
        plain(ax, gtB, xr, yr, zr, az, C_B)
        if r == 0:
            ax.set_title('ground truth B', fontsize=13, fontweight='bold')
        ax = fig.add_subplot(R, ncols, r * ncols + 5, projection='3d')
        render_attrib(ax, pred, coords_to_voxel_set(gtA), coords_to_voxel_set(gtB),
                      xr, yr, zr, az)
        if r == 0:
            ax.set_title('generated, attributed', fontsize=13, fontweight='bold')

    m = lambda k: float(np.mean([x[k] for x in recs]))
    fig.suptitle(
        f'Two views from two different towers ({VIEWS[vA]} of A + {VIEWS[vB]} of B, '
        f'other slots zeroed)\n'
        f'blue = only in A   |   orange = only in B   |   green = in both '
        f'(ambiguous)   |   red = in neither (invented)\n'
        f'means over {len(rows)} pairs:  IoU$_A$ {m("iou_A"):.3f}  IoU$_B$ {m("iou_B"):.3f}   '
        f'recall of A-only {m("rec_A_only"):.3f}  of B-only {m("rec_B_only"):.3f}   '
        f'spatial separation {m("separation"):+.3f}',
        fontsize=15, fontweight='bold', y=0.998)
    fig.subplots_adjust(left=.045, right=.995, top=.945, bottom=.005, wspace=.03, hspace=.10)

    os.makedirs(a.output_dir, exist_ok=True)
    out = pjoin(a.output_dir, f'view_fusion_{VIEWS[vA]}_{VIEWS[vB]}.png')
    plt.savefig(out, dpi=110, bbox_inches='tight'); plt.close()

    print(f'\n=== means over {len(rows)} pairs ===')
    for k in ('iou_A', 'iou_B', 'rec_A_only', 'rec_B_only',
              'frac_A_only', 'frac_B_only', 'frac_shared', 'frac_neither'):
        print(f'  {k:<13} {m(k):.4f}')
    print(f'  --- spatial attribution (units of footprint half-span) ---')
    print(f'  A-only region mean {m("proj_region_A"):+.4f} -> recovered '
          f'{m("proj_recovered_A"):+.4f}   shift {m("shift_A"):+.4f}')
    print(f'  B-only region mean {m("proj_region_B"):+.4f} -> recovered '
          f'{m("proj_recovered_B"):+.4f}   shift {m("shift_B"):+.4f}')
    print(f'  separation (positive = each view drives its own side) '
          f'{m("separation"):+.4f}')
    sh = [x['shift_A'] - x['shift_B'] for x in recs]
    print(f'  per-pair shift_A - shift_B: mean {np.mean(sh):+.4f} '
          f'sd {np.std(sh):.4f}  positive in {sum(1 for v in sh if v>0)}/{len(sh)} pairs')
    with open(pjoin(a.output_dir, f'view_fusion_{VIEWS[vA]}_{VIEWS[vB]}.json'), 'w') as fh:
        json.dump({'config': vars(a), 'pairs': recs}, fh, indent=1)
    print(f'-> {out}')


if __name__ == '__main__':
    main()
