"""
Single-tower VIEW-ABLATION render: feed different subsets of the 4 DINOv2 views
(zeroing the dropped views, the in-distribution drop v8.2 was trained with) and
see how the generated tower changes, does the recessed groove get reconstructed
only when the groove-showing views (NW/SE) are present?

View order in the feature tensor is [NE, NW, SE, SW] (idx 0,1,2,3).
Each row = one view subset; columns = GT | GMM single | DiT vote | Classifier C,
rendered from a camera that looks into the groove (diff: green=hit/red=extra/blue=miss).

Usage:
    CUDA_VISIBLE_DEVICES=<uuid> uv run python \
        script/gen_view_ablation.py --tower tower_31734 --gpu_id 0
"""
import argparse, itertools, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from utils.paths import IMAGE_ROOT
from os.path import join as pjoin
from collections import Counter
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from models.refinement import RefinementNetwork
from utils.model_io import load_vq
from utils.voxel import voxel_iou, coords_to_voxel_set
from utils.fixseed import fixseed
from eval_sampling import (CHECKPOINTS_DIR, DATASET_NAME, DATA_ROOT,
                           load_mask_model, set_iou, MASK_TIMESTEPS, MASK_COND_SCALE, MASK_TOPK)
from train_refinement import tower_features, iou_from_keep, N_FEAT
from gen_pool_cache import build_pool
from gen_img_towers import render_panel, render_diff_panel, VIEW_AZIM
from _common import gen_drafts, decode_xyz, vote_set

VIEWS = ['NE', 'NW', 'SE', 'SW']                 # feature-tensor view order
VIEW_FILE = ['northeast', 'northwest', 'southeast', 'southwest']
VIEW_ORDER_FULL = VIEW_FILE                      # camera azimuths, same order as VIEWS
# (label, kept-view-index-set)
SUBSET_PRESETS = {
    # the toxic-view diagnosis: does the groove appear only when NW/SE are given?
    'perception': [
        ('all 4 [NE,NW,SE,SW]', {0, 1, 2, 3}),
        ('NW+SE  (groove views)', {1, 2}),
        ('NE+SW  (groove occluded)', {0, 3}),
        ('NW only', {1}),
        ('SE only', {2}),
        ('NE only', {0}),
    ],
    # the view-count ladder for the paper/supplement: 4 -> 3 -> 2adj -> 2diag -> 1
    'ladder': [
        ('4 views', {0, 1, 2, 3}),
        ('3 views  (drop SW)', {0, 1, 2}),
        ('2 adjacent  NE+NW', {0, 1}),
        ('2 diagonal  NE+SW', {0, 3}),
        ('1 view  NE', {0}),
    ],
    # every non-empty subset, ordered by decreasing view count
    'all15': [('+'.join(VIEWS[i] for i in sorted(c)), set(c))
              for k in (4, 3, 2, 1)
              for c in itertools.combinations(range(4), k)],
}
IMAGE_ROOT = IMAGE_ROOT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tower', default='tower_31734')
    ap.add_argument('--mask_name', default='myarch_mask_img_patch_v8_2')
    ap.add_argument('--mask_ckpt', default='net_best_iou.tar')
    ap.add_argument('--cls_name', default='poolcls_c_sym_v1')
    ap.add_argument('--cls_tau', type=float, default=0.4)
    ap.add_argument('--vq_name', default='ibq_4x_nc1024_nq1')
    ap.add_argument('--data_root', default=DATA_ROOT)
    ap.add_argument('--n_samples', type=int, default=16)
    ap.add_argument('--draft_temp', type=float, default=0.8)
    ap.add_argument('--single_temp', type=float, default=0.2)
    ap.add_argument('--threshold', type=float, default=0.4375)
    ap.add_argument('--azim', default='southeast', help='camera that looks into the groove')
    ap.add_argument('--subsets', default='perception', choices=sorted(SUBSET_PRESETS),
                    help="'perception' = toxic-view diagnosis, 'ladder' = view-count ladder")
    ap.add_argument('--methods', default='single,vote,cls',
                    help='comma list from {single,vote,cls} = the DEPLOYED pipeline '
                         '(what the paper calls GenArch); vote and cls both build on '
                         'the raw N=16 drafts.')
    ap.add_argument('--all_angles', action='store_true',
                    help='render ONE method from all four cameras for every subset, '
                         'tiled into --col_groups blocks (supplement figure)')
    ap.add_argument('--col_groups', type=int, default=2)
    ap.add_argument('--zoom', type=float, default=1.0)
    ap.add_argument('--zooms', default='', help='comma list -> one PNG per zoom, generated once')
    ap.add_argument('--row_h', type=float, default=3.25)
    ap.add_argument('--montage_w', type=float, default=1.30)
    ap.add_argument('--spacer_w', type=float, default=0.34)
    ap.add_argument('--show_views', action='store_true',
                    help='prepend the 4 input photographs, dimmed where the view was dropped')
    ap.add_argument('--image_root', default=IMAGE_ROOT)
    ap.add_argument('--output_dir', default='./generation/perception_diag')
    ap.add_argument('--gpu_id', type=int, default=0)
    ap.add_argument('--seed', type=int, default=42)
    a = ap.parse_args()
    fixseed(a.seed)
    device = torch.device(f'cuda:{a.gpu_id}'); torch.cuda.set_device(device)

    vq, vq_opt = load_vq(CHECKPOINTS_DIR, a.vq_name, device=device, dim_pose=3, dataset_name=DATASET_NAME)
    vq.eval().to(device)
    nb_code = vq.quantizer.layers[0].nb_code; mask_id = nb_code
    meta = pjoin(CHECKPOINTS_DIR, DATASET_NAME, a.vq_name, 'meta')
    mean = np.load(pjoin(meta, 'mean.npy')).reshape(-1)[:3]
    std = np.load(pjoin(meta, 'std.npy')).reshape(-1)[:3]
    gmm = load_mask_model(vq_opt, a.mask_name, a.mask_ckpt, device)
    methods = [m.strip() for m in a.methods.split(',') if m.strip()]
    assert set(methods) <= {'single', 'vote', 'cls'}, methods

    clsf = None
    if 'cls' in methods:
        cck = torch.load(pjoin(CHECKPOINTS_DIR, DATASET_NAME, 'pool_classifier', a.cls_name, 'best.tar'),
                         map_location='cpu', weights_only=False)
        ca = cck.get('args', {})
        clsf = RefinementNetwork(N_FEAT, ca.get('latent_dim', 256), ca.get('ff_size', 1024),
                               ca.get('c_layers', 6), ca.get('n_heads', 8), ca.get('dropout', 0.1)).to(device)
        clsf.load_state_dict(cck['model']); clsf.eval()

    nm = a.tower
    motion = np.load(pjoin(a.data_root, 'new_joint_vecs_int', nm + '.npy'))
    feat0 = np.load(pjoin(a.data_root, 'image_features_dino_patch', nm + '.npy')).astype(np.float32)
    n_tok = len(motion) // 4; n_cmp = n_tok * 4
    gt = motion[:n_cmp].astype(float); gt_set = coords_to_voxel_set(gt)
    az = VIEW_AZIM[a.azim]

    def _as_xyz(vox_set):
        return np.array(sorted(vox_set), float) if vox_set else np.zeros((0, 3))

    def run(feat):
        out = []
        # N=16 raw drafts at temp 0.8 -- shared by vote and by the refinement step,
        # which is how the deployed pipeline is actually wired (eval_pipeline.py).
        drafts = gen_drafts(gmm, feat, n_tok, a.n_samples, a.draft_temp, device, nb_code)
        draft_xyz = decode_xyz(vq, drafts, n_cmp, mean, std)

        if 'single' in methods:
            single = gen_drafts(gmm, feat, n_tok, 1, a.single_temp, device, nb_code)
            gmm_xyz = decode_xyz(vq, single, n_cmp, mean, std)[0]
            out.append(('single-pass', gmm_xyz, voxel_iou(gmm_xyz, gt)))

        if 'vote' in methods:
            v = vote_set([coords_to_voxel_set(x) for x in draft_xyz], a.threshold * a.n_samples)
            out.append((f'vote N={a.n_samples}', _as_xyz(v), set_iou(v, gt_set)))

        if 'cls' in methods:
            vox, count, label, n_gt = build_pool(draft_xyz, gt_set, dilate=1)
            vox = vox.astype(np.int64)
            cf = tower_features(vox, count.astype(np.float32), a.n_samples, mean, std)
            with torch.no_grad():
                vf = torch.from_numpy(cf).unsqueeze(0).to(device)
                p = torch.sigmoid(clsf(vf, vf[..., 8:11].clone(),
                                       torch.from_numpy(feat).unsqueeze(0).to(device).float(),
                                       torch.zeros(1, len(vox), dtype=torch.bool, device=device))).squeeze(0).cpu().numpy()
            keep = p >= a.cls_tau
            c_xyz = vox[keep].astype(float) if keep.any() else np.zeros((0, 3))
            out.append(('+ refinement', c_xyz, iou_from_keep(keep, label.astype(np.float32), n_gt)))
        return out

    rows = []
    for label, keep_idx in SUBSET_PRESETS[a.subsets]:
        feat = feat0.copy()
        for v in range(4):
            if v not in keep_idx:
                feat[v] = 0.0
        res = run(feat)
        rows.append((label, keep_idx, res))
        print(f'{label:28s} ' + '  '.join(f'{t}={iou:.3f}' for t, _, iou in res))

    # ── all-angles layout: ONE method, each subset shown from all 4 cameras ──
    # 15 subsets x 4 cameras is a 1:4 aspect and will not fit a page, so the
    # subsets are tiled into --col_groups blocks of 4 camera columns each.
    if a.all_angles:
        assert len(methods) == 1, '--all_angles renders a single method'
        from utils.voxel import grid_snap as _snap
        allc = np.vstack([gt] + [p for _, _, r in rows for _, p, _ in r if len(p)])
        allc = _snap(allc)
        mxy = int(np.ceil(max(allc[:, 0].max(), allc[:, 1].max()) / 10.) * 10)
        mz = int(np.ceil(allc[:, 2].max() / 25.) * 25)
        xr = yr = [0, mxy]; zr = [0, mz]

        entries = [('Ground truth', None, None, None)] + \
                  [(lab, keep, res[0][1], res[0][2]) for lab, keep, res in rows]
        from matplotlib.gridspec import GridSpec
        from matplotlib.patches import Rectangle
        from PIL import Image

        TW, TH, GUT = 256, 320, 9        # tile size and white gutter

        def montage(keep):
            """2x2 of the four input photographs, faded where the view was withheld.

            The renders leave only ~3% sky above the spire, so butting the tiles
            together makes each tower read as cut off by the tile above it. A white
            gutter separates them into four distinct photographs.
            """
            tiles = []
            for v in range(4):
                p = pjoin(a.image_root, nm, f'{VIEW_FILE[v]}_realistic.png')
                im = np.asarray(Image.open(p).convert('RGB').resize((TW, TH)),
                                dtype=np.float32) / 255.
                if keep is not None and v not in keep:
                    im = 1.0 - 0.10 * (1.0 - im)          # fade toward white
                pad = np.ones((TH + 2 * GUT, TW + 2 * GUT, 3), np.float32)
                pad[GUT:GUT + TH, GUT:GUT + TW] = im
                tiles.append(pad)
            return np.concatenate([np.concatenate(tiles[:2], 1),
                                   np.concatenate(tiles[2:], 1)], 0)

        G = a.col_groups
        R = int(np.ceil(len(entries) / G))
        # per group: 1 montage column + 4 camera columns, plus a spacer between groups
        per = 5
        ncols = G * per + (G - 1)
        wr = []
        for g in range(G):
            if g:
                wr.append(a.spacer_w)                      # spacer
            wr += [a.montage_w] + [1.0] * 4

        def draw(zoom, tag):
            fig = plt.figure(figsize=(2.05 * sum(wr), a.row_h * R))
            gs = GridSpec(R, ncols, figure=fig, width_ratios=wr,
                          wspace=0.03, hspace=0.14)
            axes3d = []
            _body(fig, gs, axes3d, zoom)
            # the zoom is a sweep-only annotation; the shipped figure must not carry it
            ztag = f'   [zoom={zoom}]' if tag else ''
            fig.suptitle(f'{nm}  -  all {len(rows)} view subsets, deployed pipeline '
                         f'(N={a.n_samples} + refinement), each shown from all four '
                         f'cameras{ztag}\n'
                         f'green = correct  |  red = extra  |  blue = missed',
                         fontsize=15, fontweight='bold', y=0.998)
            gs.update(left=.030, right=.996, top=.950, bottom=.006)
            fig.canvas.draw()
            # how much ink, if any, is being clipped by each panel boundary
            img = np.asarray(fig.canvas.buffer_rgba())[..., :3]
            H = img.shape[0]; ink = (img < 245).any(-1)
            touch, fill = 0, []
            for ax in axes3d:
                bb = ax.get_window_extent()
                x0, x1 = int(bb.x0), int(bb.x1); y0, y1 = int(H - bb.y1), int(H - bb.y0)
                sub = ink[max(y0, 0):y1, max(x0, 0):x1]
                if sub.size == 0:
                    continue
                touch += int(sub[0].sum() + sub[-1].sum() + sub[:, 0].sum() + sub[:, -1].sum())
                rws = np.where(sub.any(1))[0]
                if len(rws):
                    fill.append((rws.max() - rws.min()) / sub.shape[0])
            os.makedirs(a.output_dir, exist_ok=True)
            out = pjoin(a.output_dir, f'{nm}_view_ablation_all_angles{tag}.png')
            plt.savefig(out, dpi=115, bbox_inches='tight'); plt.close()
            print(f'  zoom={zoom:<5} clipped-ink-px={touch:<6} '
                  f'mean panel fill={100*np.mean(fill):.0f}%  -> {out}')

        def _body(fig, gs, axes3d, zoom):
            for idx, (lab, keep, pred, iou) in enumerate(entries):
                g, r = idx // R, idx % R
                c0 = g * (per + 1)                             # first column of this group

                axm = fig.add_subplot(gs[r, c0])
                axm.imshow(montage(keep))
                axm.set_xticks([]); axm.set_yticks([])
                for s in axm.spines.values():
                    s.set_visible(False)
                if keep is not None:                           # outline the supplied views
                    for v in range(4):
                        if v in keep:
                            x0 = (v % 2) * (TW + 2 * GUT) + GUT
                            y0 = (v // 2) * (TH + 2 * GUT) + GUT
                            axm.add_patch(Rectangle((x0 - 1, y0 - 1), TW + 1, TH + 1,
                                                    fill=False, edgecolor='#2ca02c',
                                                    linewidth=3.2))
                if r == 0:
                    axm.set_title('input views', fontsize=12.5, fontweight='bold', pad=3)
                txt = lab if iou is None else f'{lab}\nIoU {iou:.3f}'
                axm.set_ylabel(txt, fontsize=11.5, fontweight='bold', linespacing=1.5)

                for v, view in enumerate(VIEW_ORDER_FULL):
                    ax = fig.add_subplot(gs[r, c0 + 1 + v], projection='3d')
                    az = VIEW_AZIM[view]
                    if pred is None:
                        render_panel(ax, gt, "", xr, yr, zr, color="#636EFA", azim=az, zoom=zoom)
                    else:
                        render_diff_panel(ax, gt, pred, '', xr, yr, zr, azim=az, zoom=zoom)
                    ax.set_axis_off(); axes3d.append(ax)
                    if r == 0:
                        ax.set_title(VIEWS[v], fontsize=12.5, fontweight='bold', pad=0)

        zooms = [float(z) for z in a.zooms.split(',')] if a.zooms else [a.zoom]
        multi = len(zooms) > 1
        for z in zooms:
            draw(z, f'_zoom{z:g}'.replace('.', 'p') if multi else '')
        return

    # ── figure: rows = subsets, cols = [4 input views] | GT | <methods> ──
    allc = np.vstack([gt] + [p for _, _, r in rows for _, p, _ in r if len(p)])
    from utils.voxel import grid_snap
    allc = grid_snap(allc)
    mxy = int(np.ceil(max(allc[:, 0].max(), allc[:, 1].max()) / 10.) * 10)
    mz = int(np.ceil(allc[:, 2].max() / 25.) * 25)
    xr = yr = [0, mxy]; zr = [0, mz]
    n_view_cols = 4 if a.show_views else 0
    ncols = n_view_cols + 1 + len(methods); nrows = len(rows)
    fig = plt.figure(figsize=(4.6 * ncols, 5.4 * nrows))
    for r, (label, keep_idx, res) in enumerate(rows):
        # the four conditioning photographs; a dropped view is shown dimmed so the
        # reader can see exactly what the model was and was not given
        for v in range(n_view_cols):
            ax = fig.add_subplot(nrows, ncols, r * ncols + 1 + v)
            ipath = pjoin(a.image_root, nm, f'{VIEW_FILE[v]}_realistic.png')
            if os.path.exists(ipath):
                ax.imshow(mpimg.imread(ipath), alpha=1.0 if v in keep_idx else 0.15)
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_edgecolor('#2ca02c' if v in keep_idx else '#cccccc')
                s.set_linewidth(3 if v in keep_idx else 1)
            if r == 0:
                ax.set_title(VIEWS[v], fontsize=13, fontweight='bold')
            if v == 0:
                ax.set_ylabel(label, fontsize=12, fontweight='bold')

        ax = fig.add_subplot(nrows, ncols, r * ncols + n_view_cols + 1, projection='3d')
        render_panel(ax, gt, 'Ground Truth' if r == 0 else '', xr, yr, zr, color='#636EFA', azim=az, zoom=1.0)
        if not a.show_views:
            ax.text2D(-0.25, 0.5, label, transform=ax.transAxes, fontsize=12, fontweight='bold',
                      rotation=90, va='center')
        for c, (t, pred, iou) in enumerate(res):
            ax = fig.add_subplot(nrows, ncols, r * ncols + n_view_cols + 2 + c, projection='3d')
            render_diff_panel(ax, gt, pred, f'{t}  IoU={iou:.3f}' if r == 0 else f'IoU={iou:.3f}',
                              xr, yr, zr, azim=az, zoom=1.0)
    fig.suptitle(f'{nm}  view-ablation (camera={a.azim})   green=correct · red=extra · blue=missed',
                 fontsize=15, fontweight='bold', y=0.997)
    os.makedirs(a.output_dir, exist_ok=True)
    out = pjoin(a.output_dir, f'{nm}_view_ablation.png')
    fig.subplots_adjust(left=0.06, right=0.99, top=0.97, bottom=0.01, wspace=0.05, hspace=0.18)
    plt.savefig(out, dpi=130, bbox_inches='tight'); plt.close()
    print(f'\n-> {out}')


if __name__ == '__main__':
    main()
