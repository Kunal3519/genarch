"""Teaser + architecture diagrams for the paper (MMM-style landing/arch figures).

Uses REAL assets: raw realistic view PNGs, the GT voxel render, and the actual
GenArch (1D pipeline: v8.2 mask transformer -> N=16 -> pool -> refinement)
reconstruction of one tower.

  CUDA_VISIBLE_DEVICES=<uuid> uv run python \
      script/gen_paper_diagrams.py --tower tower_500
"""
import os, sys, argparse
from os.path import join as pjoin
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.paths import DATA_ROOT, IMAGE_ROOT, EVAL_RESULTS
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

from gen_img_towers import render_panel, render_diff_panel, VIEW_AZIM
from utils.voxel import voxel_iou, grid_snap, coords_to_voxel_set
from utils.fixseed import fixseed

VIEWS = ['northeast', 'northwest', 'southeast', 'southwest']
VLAB = ['NE', 'NW', 'SE', 'SW']
BOX_FC = '#eef2fa'; BOX_EC = '#3b5b92'
FROZ_FC = '#f2f2f2'; FROZ_EC = '#888888'
TOK_COLORS = ['#4c72b0', '#dd8452', '#55a868', '#c44e52', '#8172b3', '#937860']


def load_pipeline(device):
    from eval_pipeline import build_pool
    from eval_sampling import load_mask_model, eval_tower
    from train_refinement import tower_features, N_FEAT
    from models.refinement import RefinementNetwork
    from utils.model_io import load_vq
    vq, vq_opt = load_vq('checkpoints', 'ibq_4x_nc1024_nq1', device=device,
                          dim_pose=3, dataset_name='myarch')
    vq.eval().to(device); nb = vq.quantizer.layers[0].nb_code
    gmm = load_mask_model(vq_opt, 'myarch_mask_img_patch_v8_2', 'net_best_iou.tar', device)
    cck = torch.load('checkpoints/myarch/pool_classifier/poolcls_c_sym_v1/best.tar',
                     map_location='cpu', weights_only=False)
    ca = cck['args']
    clsf = RefinementNetwork(N_FEAT, ca['latent_dim'], ca['ff_size'], ca['c_layers'],
                           ca['n_heads'], ca['dropout']).to(device)
    clsf.load_state_dict(cck['model']); clsf.eval()
    meta = 'checkpoints/myarch/ibq_4x_nc1024_nq1/meta'
    mean = np.load(pjoin(meta, 'mean.npy')); std = np.load(pjoin(meta, 'std.npy'))

    def run(motion, feat, n=16, tau=0.45):
        n_cmp = (len(motion)//4)*4
        gt_set = coords_to_voxel_set(motion[:n_cmp].astype(float))
        _, xyz = eval_tower(gmm, vq, mean, std, motion, feat, n, 0.8, device, nb,
                            decode_mode='whole')
        vox, count, label, n_gt = build_pool(xyz, gt_set, 1)
        cf = tower_features(vox, count, n, mean.reshape(-1)[:3], std.reshape(-1)[:3])
        vf = torch.from_numpy(cf).unsqueeze(0).to(device)
        pos = vf[..., 8:11].clone()
        kpad = torch.zeros(1, len(vox), dtype=torch.bool, device=device)
        img = torch.from_numpy(feat).unsqueeze(0).to(device).float()
        with torch.no_grad():
            p = torch.sigmoid(clsf(vf, pos, img, kpad)).squeeze(0).cpu().numpy()
        return xyz, vox[p >= tau].astype(float)
    return run


def _ranges(coords_list):
    allp = np.vstack([grid_snap(c) for c in coords_list if len(c)])
    m_xy = int(np.ceil(max(allp[:, 0].max(), allp[:, 1].max())/10.0))*10
    m_z = int(np.ceil(allp[:, 2].max()/25.0))*25
    return [0, m_xy], [0, m_xy], [0, m_z]


def _declutter(ax, axis_off=False):
    if axis_off:                       # fully bare: no panes, no axis lines
        ax.set_axis_off()
        return
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_ticklabels([]); axis.set_ticks([])
    ax.set_xlabel(''); ax.set_ylabel(''); ax.set_zlabel('')


def _tower_ax(fig, rect, coords, color, ranges, azim, title='', diff_gt=None, fs=9,
              zoom=1.0, axis_off=False):
    ax = fig.add_axes(rect, projection='3d')
    x_r, y_r, z_r = ranges
    if diff_gt is not None:
        render_diff_panel(ax, diff_gt, coords, title, x_r, y_r, z_r, azim=azim, zoom=zoom)
    else:
        render_panel(ax, coords, title, x_r, y_r, z_r, color=color, azim=azim, zoom=zoom)
    _declutter(ax, axis_off)
    if title:
        ax.set_title(title, fontsize=fs, pad=0)
    return ax


def _box(axf, x, y, w, h, text, fc=BOX_FC, ec=BOX_EC, fs=10, weight='bold', sub=None):
    axf.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.004',
                                 fc=fc, ec=ec, lw=1.4, mutation_scale=1))
    axf.text(x+w/2, y+h/2 + (0.018 if sub else 0), text, ha='center', va='center',
             fontsize=fs, fontweight=weight)
    if sub:
        axf.text(x+w/2, y+h/2-0.028, sub, ha='center', va='center', fontsize=fs-2,
                 color='#444444')


def _arrow(axf, x0, y0, x1, y1=None, **kw):
    """Arrow from (x0,y0) to (x1,y1); y1 defaults to y0 for horizontal arrows."""
    y1 = y0 if y1 is None else y1
    axf.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle='-|>',
                                  mutation_scale=16, lw=1.6, color='#333333', **kw))


def _tokens(axf, x, y, w, h, n, masked=None, rng=None):
    """A strip of n token squares; indices in `masked` drawn as gray [M]."""
    rng = rng or np.random.RandomState(0)
    step = w/n
    for i in range(n):
        m = masked is not None and i in masked
        c = '#cccccc' if m else TOK_COLORS[rng.randint(len(TOK_COLORS))]
        axf.add_patch(Rectangle((x+i*step, y), step*0.82, h, fc=c, ec='#555555', lw=0.6))
        if m:
            axf.text(x+i*step+step*0.41, y+h/2, 'M', ha='center', va='center',
                     fontsize=6.5, color='#555555')


def _montage(image_root, tower, gap=18):
    """The four realistic views of one tower composited into a single 2x2 image."""
    ims = []
    for v in VIEWS:
        p = pjoin(image_root, tower, f'{v}_realistic.png')
        im = mpimg.imread(p)
        if im.ndim == 2:
            im = np.dstack([im]*3)
        ims.append(im[..., :3].astype(np.float32))
    h = min(i.shape[0] for i in ims); w = min(i.shape[1] for i in ims)
    ims = [i[:h, :w] for i in ims]
    vg = np.ones((h, gap, 3), np.float32)
    hg = np.ones((gap, 2*w + gap, 3), np.float32)
    top = np.hstack([ims[0], vg, ims[1]])
    bot = np.hstack([ims[2], vg, ims[3]])
    return np.vstack([top, hg, bot])


def teaser(examples, image_root, out, n_cols=8):
    """Landing figure: the paper's objective only. Four rendered views of a
    building (top) map to its explicit 3D geometry (bottom), one column per
    example. No network internals here, those live in the architecture figure."""
    examples = examples[:n_cols]
    n = len(examples)
    fig = plt.figure(figsize=(16, 5.3))
    axf = fig.add_axes([0, 0, 1, 1]); axf.axis('off'); axf.set_xlim(0, 1); axf.set_ylim(0, 1)

    axf.text(0.5, 0.975, 'GenArch: from four rendered views to explicit 3D architecture',
             ha='center', va='top', fontsize=17, fontweight='bold')

    x0, x1 = 0.072, 0.998
    cw = (x1 - x0)/n
    img_y, img_h = 0.520, 0.395
    tow_y, tow_h = 0.010, 0.455

    # left-hand row labels
    axf.text(x0 - 0.010, img_y + img_h/2, 'input\nfour views', ha='right', va='center',
             fontsize=12.5, fontweight='bold', color='#333333', linespacing=1.5)
    axf.text(x0 - 0.010, tow_y + tow_h/2, 'output\n3D architecture', ha='right', va='center',
             fontsize=12.5, fontweight='bold', color=BOX_EC, linespacing=1.5)

    for i, (tower, gt, pred) in enumerate(examples):
        cx = x0 + i*cw
        axi = fig.add_axes([cx + cw*0.045, img_y, cw*0.91, img_h])
        axi.imshow(_montage(image_root, tower)); axi.axis('off')
        axf.add_patch(FancyArrowPatch((cx + cw/2, img_y - 0.012),
                                      (cx + cw/2, img_y - 0.082),
                                      arrowstyle='-|>', mutation_scale=15, lw=1.6,
                                      color='#9aa5b5'))
        _tower_ax(fig, [cx, tow_y, cw, tow_h], pred, '#3fa34d',
                  _ranges([gt, pred]), VIEW_AZIM['northeast'],
                  zoom=1.30, axis_off=True)

    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('saved ->', out, flush=True)


def _panel(axf, x, y, w, h, label, fc='#fbfcfe', ec='#c3cbd8'):
    axf.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.006',
                                 fc=fc, ec=ec, lw=1.2, mutation_scale=1, zorder=0))
    axf.text(x + 0.012, y + h - 0.030, label, fontsize=11.5, fontweight='bold',
             color='#5a6577', va='center', zorder=2)


def architecture(tower, image_root, gt, samples, pred, iou, out):
    """Three-stage system figure: frozen tokenizer, conditional masked
    transformer, refinement. Each stage sits in its own labelled panel."""
    fig = plt.figure(figsize=(16, 6.6))
    axf = fig.add_axes([0, 0, 1, 1]); axf.axis('off'); axf.set_xlim(0, 1); axf.set_ylim(0, 1)
    ranges = _ranges([gt, pred] + samples)
    rng = np.random.RandomState(3)

    # =================== Stage 1: tokenizer (frozen) ===================
    _panel(axf, 0.025, 0.705, 0.950, 0.280, 'Stage 1  Tower tokenizer   (trained first, then frozen)',
           fc='#f7f7f7', ec='#c9c9c9')
    yc = 0.800
    axf.text(0.100, yc, 'block sequence\n$(x,y,z)_1 \\dots (x,y,z)_B$', ha='center',
             va='center', fontsize=11)
    _arrow(axf, 0.152, yc, 0.186)
    _box(axf, 0.189, yc-0.048, 0.100, 0.096, 'conv encoder', fc=FROZ_FC, ec=FROZ_EC, fs=11)
    _arrow(axf, 0.292, yc, 0.327)
    _tokens(axf, 0.331, yc-0.030, 0.105, 0.060, 8, rng=rng)
    axf.text(0.383, yc+0.062, '$L=\\lceil B/4\\rceil$ tokens', ha='center', fontsize=10.5)
    _arrow(axf, 0.440, yc, 0.476)
    _box(axf, 0.479, yc-0.048, 0.106, 0.096, 'codebook $E$', fc=FROZ_FC, ec=FROZ_EC, fs=11,
         sub='1024 entries (IBQ)')
    _arrow(axf, 0.588, yc, 0.623)
    _box(axf, 0.626, yc-0.048, 0.100, 0.096, 'conv decoder', fc=FROZ_FC, ec=FROZ_EC, fs=11)
    _arrow(axf, 0.729, yc, 0.764)
    axf.text(0.868, yc, 'reconstructed coordinates\nrecon IoU 0.9983', ha='center',
             va='center', fontsize=11)

    # =================== Stage 2: masked transformer ===================
    _panel(axf, 0.025, 0.020, 0.610, 0.645, 'Stage 2  Conditional masked transformer')
    yc = 0.300

    for i, v in enumerate(VIEWS):
        axi = fig.add_axes([0.040 + (i % 2)*0.049, yc + 0.012 - (i//2)*0.190,
                            0.046, 0.178])
        img = pjoin(image_root, tower, f'{v}_realistic.png')
        if os.path.exists(img):
            axi.imshow(mpimg.imread(img))
        axi.axis('off')
    axf.text(0.088, yc + 0.215, 'four views', ha='center', fontsize=11)
    _arrow(axf, 0.140, yc, 0.167)
    _box(axf, 0.170, yc-0.070, 0.074, 0.140, 'DINOv2', fc=FROZ_FC, ec=FROZ_EC, fs=11,
         sub='frozen ViT-L')
    axf.text(0.207, yc-0.098, '$4\\times256$ patch tokens', ha='center', fontsize=10)

    _box(axf, 0.278, yc-0.145, 0.138, 0.360, '')
    axf.text(0.347, yc + 0.180, 'masked transformer', ha='center', fontsize=12,
             fontweight='bold')
    for j, lab in enumerate(['self-attention', 'cross-attention', 'feed-forward']):
        _box(axf, 0.290, yc + 0.088 - j*0.072, 0.114, 0.058, lab, fc='white', ec=BOX_EC,
             fs=10.5, weight='normal')
    axf.text(0.347, yc - 0.118, '$\\times\\,8$ blocks', ha='center', fontsize=10.5)
    _arrow(axf, 0.246, yc, 0.288)                       # patch memory -> cross-attention

    _tokens(axf, 0.281, 0.062, 0.132, 0.050, 10, masked={1, 2, 4, 5, 7, 8, 9}, rng=rng)
    axf.text(0.347, 0.040, 'masked token sequence', ha='center', fontsize=10.5)
    _arrow(axf, 0.347, 0.116, 0.347, y1=yc-0.148)       # tokens -> transformer

    _arrow(axf, 0.419, yc, 0.447)
    axf.text(0.433, yc + 0.062, '$T{=}10$\nparallel\nsteps', ha='center', fontsize=10)
    _tokens(axf, 0.451, yc-0.025, 0.088, 0.050, 8, rng=rng)
    axf.text(0.495, yc + 0.058, 'predicted tokens', ha='center', fontsize=10.5)
    _arrow(axf, 0.543, yc, 0.570)
    _box(axf, 0.573, yc-0.065, 0.056, 0.130, 'VQ\ndecoder', fc=FROZ_FC, ec=FROZ_EC, fs=10.5)

    # =================== Stage 3: refinement ===================
    _panel(axf, 0.650, 0.020, 0.325, 0.645, 'Stage 3  Refinement')
    _arrow(axf, 0.632, yc, 0.663)
    for i, s in enumerate(samples[:3]):
        _tower_ax(fig, [0.662 + i*0.038, yc-0.150, 0.042, 0.340], grid_snap(s),
                  '#3fa34d', ranges, VIEW_AZIM['northeast'], zoom=1.25, axis_off=True)
    axf.text(0.722, yc + 0.205, '$N{=}16$ candidates\n(temperature 0.8)', ha='center',
             fontsize=10.5)
    _arrow(axf, 0.780, yc, 0.806)
    _box(axf, 0.809, yc-0.085, 0.082, 0.170, 'refinement\nnetwork', fs=11,
         sub='pool $+$ image')
    axf.add_patch(FancyArrowPatch((0.868, yc + 0.190), (0.855, yc + 0.092),
                                  arrowstyle='-|>', mutation_scale=13, lw=1.2,
                                  color='#8a93a3', ls='--'))
    axf.text(0.876, yc + 0.212, 'patch tokens', ha='center', fontsize=10,
             color='#6c7583')
    _arrow(axf, 0.894, yc, 0.912)
    _tower_ax(fig, [0.906, yc-0.175, 0.066, 0.390], pred, None, ranges,
              VIEW_AZIM['northeast'], '', diff_gt=gt, zoom=1.25, axis_off=True)
    axf.text(0.939, yc - 0.196, f'final tower\nIoU {iou:.3f}', ha='center', fontsize=10.5)

    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('saved ->', out, flush=True)


RECORDS = pjoin(EVAL_RESULTS,
                'mask_clsC_myarch_mask_img_patch_v8_2_val40k_N16_shard*of6.npy')


def pick_teaser_towers(image_root, n=8, min_iou=0.995, pick_seed=3):
    """Deterministically pick n well-reconstructed val towers spanning the size
    range, so the landing figure shows the variety of geometry the model handles."""
    import glob as _glob
    rec = {}
    for p in sorted(_glob.glob(RECORDS)):
        rec.update(np.load(p, allow_pickle=True).item())
    cand = [(k, v['n_blocks']) for k, v in rec.items()
            if v['byN'][16]['cls'][0.45] >= min_iou
            and os.path.isdir(pjoin(image_root, k))]
    if not cand:
        raise SystemExit('no candidate towers found')
    cand.sort(key=lambda t: t[1])
    # one tower per size bin, at a fixed pseudo-random offset inside the bin so
    # the row spans the size range without picking near-identical neighbours
    rs = np.random.RandomState(pick_seed)
    edges = np.linspace(0, len(cand), n + 1).astype(int)
    return [cand[rs.randint(lo, hi)][0] for lo, hi in zip(edges[:-1], edges[1:])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tower', default='tower_500', help='tower used in the architecture figure')
    ap.add_argument('--teaser_towers', nargs='*', default=None,
                    help='towers shown in the landing figure (default: auto-select)')
    ap.add_argument('--n_teaser', type=int, default=8)
    ap.add_argument('--pick_seed', type=int, default=3)
    ap.add_argument('--data_root', default=DATA_ROOT)
    ap.add_argument('--image_root', default=IMAGE_ROOT)
    ap.add_argument('--out_dir', default='paper-overleaf/Figures')
    ap.add_argument('--only', choices=['teaser', 'arch', 'both'], default='both')
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    fixseed(a.seed)
    device = torch.device('cuda')

    motion_dir = pjoin(a.data_root, 'new_joint_vecs_int')
    feat_dir = pjoin(a.data_root, 'image_features_dino_patch')
    run = load_pipeline(device)

    def reconstruct(name):
        motion = np.load(pjoin(motion_dir, name + '.npy'))
        feat = np.load(pjoin(feat_dir, name + '.npy')).astype(np.float32)
        samples, pred = run(motion, feat)
        gt = motion.astype(float)
        return gt, samples, pred, voxel_iou(pred, gt)

    if a.only in ('teaser', 'both'):
        towers = a.teaser_towers or pick_teaser_towers(a.image_root, a.n_teaser,
                                                       pick_seed=a.pick_seed)
        examples = []
        for t in towers:
            gt, _, pred, iou = reconstruct(t)
            print(f'  teaser {t}: IoU {iou:.4f}', flush=True)
            examples.append((t, gt, pred))
        teaser(examples, a.image_root, pjoin(a.out_dir, 'teaser.png'),
               n_cols=a.n_teaser)

    if a.only in ('arch', 'both'):
        gt, samples, pred, iou = reconstruct(a.tower)
        print(f'{a.tower}: refined IoU {iou:.4f}', flush=True)
        architecture(a.tower, a.image_root, gt, list(samples[:3]), pred, iou,
                     pjoin(a.out_dir, 'architecture.png'))


if __name__ == '__main__':
    main()
