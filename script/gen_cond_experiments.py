"""
Probe WHAT THE CONDITIONING ACTUALLY DOES: feed the model inconsistent or
degenerate view sets and see what geometry comes out.

Cached feature slot order is [NE, NW, SE, SW] (script/preprocess/precompute_patch_dl.py), and
the model adds a learned per-slot view embedding, so slot k carries "this is the
view from corner k" regardless of which tower the pixels came from.

  C1  chimera        - slot k from a DIFFERENT tower k (4 towers, 4 corners)
  C2  same view x4   - one tower's NE view copied into all four slots
                       (+ control: NE in slot 0 only, other slots zeroed)
  C3  two towers     - one view from tower A + one from tower B, others zeroed;
                       run twice: OPPOSITE corners (NE+SW) and ADJACENT (NE+NW)
  C4  freeze + reskin - freeze the bottom of tower A, regenerate everything above
                       it while conditioning on tower B's photos
                       (+ control: same mask, A's own photos)

Length is an input the model requires, so for the from-scratch cases we must supply
one; we use the first source tower's token count and say so.

  uv run python script/gen_cond_experiments.py \
      --towers tower_16325 tower_268707 tower_33872 tower_10621
"""
import os, sys, json, argparse
from os.path import join as pjoin
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.paths import DATA_ROOT, IMAGE_ROOT

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image

from utils.voxel import voxel_iou, grid_snap, GRID
from utils.model_io import load_vq

DATA = DATA_ROOT
RAW = IMAGE_ROOT
VIEWS = ['northeast', 'northwest', 'southeast', 'southwest']
NE_AZIM = 220
GREEN, GREY = '#3fa34d', '#b9bec7'


def cubes(ax, coords, ranges, color=GREEN, edge='#20492b', elev=24, azim=NE_AZIM):
    dx, dy, dz = GRID
    verts = []
    for cx, cy, cz in np.asarray(coords).reshape(-1, 3):
        x0, y0, z0 = cx-dx/2, cy-dy/2, cz-dz/2
        x1, y1, z1 = x0+dx, y0+dy, z0+dz
        verts += [
            [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0)],
            [(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)],
            [(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)],
            [(x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1)],
            [(x1, y1, z0), (x0, y1, z0), (x0, y1, z1), (x1, y1, z1)],
            [(x0, y1, z0), (x0, y0, z0), (x0, y0, z1), (x0, y1, z1)],
        ]
    if verts:
        ax.add_collection3d(Poly3DCollection(verts, facecolors=color, edgecolors=edge,
                                             linewidths=0.18))
    (xr, yr, zr) = ranges
    ax.set_xlim(*xr); ax.set_ylim(*yr); ax.set_zlim(*zr)
    ax.set_box_aspect((xr[1]-xr[0], yr[1]-yr[0], (zr[1]-zr[0])*0.62), zoom=1.28)
    ax.view_init(elev=elev, azim=azim); ax.set_axis_off()


def ranges_of(cl):
    a = np.concatenate([np.asarray(c).reshape(-1, 3) for c in cl if len(c)])
    lo, hi = a.min(0), a.max(0)
    p = (hi-lo).max()*0.06 + 1
    return [(lo[i]-p, hi[i]+p) for i in range(3)]


def montage(specs, gap=6):
    """specs: list of 4 (tower_or_None, view_name). 2x2 montage of the fed views."""
    ims = []
    for tw, v in specs:
        if tw is None:
            ims.append(Image.new('RGB', (256, 256), (238, 238, 238)))
        else:
            ims.append(Image.open(pjoin(RAW, tw, f'{v}_realistic.png')).convert('RGB').resize((256, 256)))
    W = 256*2 + gap
    out = Image.new('RGB', (W, W), (255, 255, 255))
    for i, im in enumerate(ims):
        out.paste(im, ((i % 2)*(256+gap), (i//2)*(256+gap)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--towers', nargs='+',
                    default=['tower_16325', 'tower_268707', 'tower_33872', 'tower_10621'])
    ap.add_argument('--out_dir', default='generation/cond_experiments')
    ap.add_argument('--timesteps', type=int, default=10)
    ap.add_argument('--temperature', type=float, default=0.2)
    ap.add_argument('--edit_temp', type=float, default=0.8)
    ap.add_argument('--gpu_id', type=int, default=0)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    from eval_sampling import load_mask_model

    vq, vq_opt = load_vq('checkpoints', 'ibq_4x_nc1024_nq1', device=device,
                          dim_pose=3, dataset_name='myarch')
    vq.eval().to(device)
    model = load_mask_model(vq_opt, 'myarch_mask_img_patch_v8_2', 'net_best_iou.tar', device)
    meta = 'checkpoints/myarch/ibq_4x_nc1024_nq1/meta'
    mean = np.load(pjoin(meta, 'mean.npy')).reshape(-1)[:3]
    std = np.load(pjoin(meta, 'std.npy')).reshape(-1)[:3]

    T = args.towers
    gt, ids, feats = {}, {}, {}
    for t in T:
        m = np.load(pjoin(DATA, 'new_joint_vecs_int', t + '.npy')).reshape(-1, 3)
        n = (len(m)//4)*4
        gt[t] = m[:n].astype(float)
        with torch.no_grad():
            x = torch.from_numpy((m[:n]-mean)/std).float().unsqueeze(0).to(device)
            ids[t] = vq.encode(x)[0][..., 0]
        feats[t] = np.load(pjoin(DATA, 'image_features_dino_patch', t + '.npy')).astype(np.float32)

    def decode(i, L):
        with torch.no_grad():
            xyz = vq.forward_decoder(i[:, :L].clamp(min=0).unsqueeze(-1))[0].cpu().numpy()
        return grid_snap(xyz*std + mean)

    def gen(cond_np, L):
        c = torch.from_numpy(cond_np).unsqueeze(0).to(device).float()
        out = model.generate(c, torch.tensor([L], device=device), timesteps=args.timesteps,
                             cond_scale=1.0, temperature=args.temperature,
                             topk_filter_thres=0.9, gsample=False)
        return decode(out, L)

    L0 = ids[T[0]].shape[1]
    rows, res = [], []

    # ---- C1 chimera: slot k from tower k ----
    chim = np.stack([feats[T[k]][k] for k in range(4)])           # (4,256,1024)
    out = gen(chim, L0)
    rows.append(('C1  chimera: 4 corners from 4 different towers',
                 [(T[k], VIEWS[k]) for k in range(4)], out,
                 {t: voxel_iou(out, gt[t]) for t in T},
                 f'length taken from {T[0]} ({L0} tokens)'))

    # ---- C2 same view x4 ----
    A = T[0]
    same = np.stack([feats[A][0]]*4)                              # NE copied to all slots
    out = gen(same, ids[A].shape[1])
    rows.append(('C2a  one NE view copied into all 4 slots',
                 [(A, 'northeast')]*4, out, {A: voxel_iou(out, gt[A])},
                 'model is told the same picture is each of the 4 corners'))
    one = np.zeros_like(feats[A]); one[0] = feats[A][0]            # NE only, rest zeroed
    out = gen(one, ids[A].shape[1])
    rows.append(('C2b  control: NE only, other slots zeroed',
                 [(A, 'northeast'), (None, ''), (None, ''), (None, '')], out,
                 {A: voxel_iou(out, gt[A])},
                 'the honest 1-view setting (v8.2 trains with view dropout)'))

    # ---- C3 two towers, two views ----
    B = T[1]
    for tag, (sa, sb) in (('opposite corners NE+SW', (0, 3)), ('adjacent corners NE+NW', (0, 1))):
        c = np.zeros_like(feats[A]); c[sa] = feats[A][sa]; c[sb] = feats[B][sb]
        out = gen(c, L0)
        spec = [(None, '')]*4
        spec[sa] = (A, VIEWS[sa]); spec[sb] = (B, VIEWS[sb])
        rows.append((f'C3  {tag}: one view of {A} + one of {B}', spec, out,
                     {A: voxel_iou(out, gt[A]), B: voxel_iou(out, gt[B])},
                     'other two slots zeroed'))

    # ---- C4 freeze bottom of A, regenerate above with B's photos ----
    zs = gt[A][:, 2]
    col_start = [0] + [i for i in range(1, len(gt[A])) if zs[i] <= zs[i-1]]
    bottom_tok = set(s//4 for s in col_start)
    LA = ids[A].shape[1]
    em = torch.zeros(1, LA, dtype=torch.bool, device=device)
    em[:, [i for i in range(LA) if i not in bottom_tok]] = True     # free everything above
    for tag, src in ((f'reskin with {B} photos', B), (f'control: {A} own photos', A)):
        c = torch.from_numpy(feats[src]).unsqueeze(0).to(device).float()
        o = model.edit(c, ids[A], torch.tensor([LA], device=device),
                       timesteps=args.timesteps, cond_scale=1.0, temperature=args.edit_temp,
                       topk_filter_thres=0.9, gsample=False, edit_mask=em)
        pred = decode(o, LA)
        exact = bool((o[:, :LA][~em] == ids[A][~em]).all())
        # upper-region-only comparison (the part the model was free to change)
        zc = 36.0
        up = lambda c_: np.asarray(c_)[np.asarray(c_)[:, 2] >= zc]
        rows.append((f'C4  freeze bottom of {A}, {tag}',
                     [(src, v) for v in VIEWS], pred,
                     {f'{A} (all)': voxel_iou(pred, gt[A]),
                      f'{A} (above z{int(zc)})': voxel_iou(up(pred), up(gt[A])),
                      f'{B} (above z{int(zc)})': voxel_iou(up(pred), up(gt[B]))},
                     f'bottom {LA-int(em.sum())}/{LA} tokens frozen (exact: {exact})'))

    # ---- figure ----
    rng = ranges_of([gt[t] for t in T] + [r[2] for r in rows])
    n = len(rows)
    fig = plt.figure(figsize=(13.2, 2.9*n + 0.4))
    top = 0.955
    for r, (title, spec, out, ious, note) in enumerate(rows):
        h = top/n; y = top - (r+1)*h
        ax = fig.add_axes([0.01, y+0.006, 0.165, h*0.9])
        ax.imshow(montage(spec)); ax.axis('off')
        ax.set_title('views fed in (NE NW / SE SW)', fontsize=8, pad=2)
        ax = fig.add_axes([0.19, y, 0.16, h*0.93], projection='3d')
        cubes(ax, out, rng); ax.set_title('model output', fontsize=9, pad=-6)
        for j, t in enumerate(T[:2]):
            ax = fig.add_axes([0.355 + j*0.14, y, 0.135, h*0.93], projection='3d')
            cubes(ax, gt[t], rng, color=GREY, edge='#7d838d')
            ax.set_title(f'GT {t}', fontsize=8, pad=-6)
        fig.text(0.65, y + h*0.70, title, fontsize=11, fontweight='bold', va='center')
        fig.text(0.65, y + h*0.52, note, fontsize=8.5, va='center', color='#333')
        txt = '   '.join(f'IoU vs {k}: {v:.3f}' for k, v in ious.items())
        fig.text(0.65, y + h*0.30, txt, fontsize=8.5, va='center', color='#444')
        res.append(dict(case=title, note=note, ious={k: round(float(v), 4) for k, v in ious.items()}))
        print(f'{title}\n    {txt}', flush=True)
    fig.text(0.5, 0.995, 'What does the conditioning actually control?  '
                         '(inconsistent / degenerate view sets)',
             ha='center', va='top', fontsize=12.5, fontweight='bold')
    o = pjoin(args.out_dir, 'cond_experiments.png')
    fig.savefig(o, dpi=145, facecolor='white', bbox_inches='tight'); plt.close(fig)
    json.dump(res, open(pjoin(args.out_dir, 'results.json'), 'w'), indent=2)
    print('->', o)


if __name__ == '__main__':
    main()
