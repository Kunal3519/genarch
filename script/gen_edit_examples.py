"""
Render worked examples of MaskTransformer.edit() -- the applications it unlocks,
plus deliberate edge cases, on the deployed v8.2 + IBQ pipeline.

TOKEN LAYOUT (verified, and it dictates what each mask MEANS -- do not guess):
the block sequence is RASTER over the footprint (y fast, x slow) and each footprint
column is emitted as a full VERTICAL run (z = 4,12,...,top, then the next column).
So for a 10x10 footprint: 100 columns x ~10 blocks, 4 blocks per VQ token.
  * a token PREFIX  = a contiguous slab of the PLAN in x (left part of the building,
                      full height) -- NOT the base of the tower
  * appending tokens = MORE COLUMNS  = growing the footprint sideways, NOT upward
  * true vertical completion needs a STRIDED mask: fix the first token of every
    column (its bottom ~4 storeys) and free the rest -> "podium -> tower"
(An earlier note claiming "2D floorplan -> 3D == fixing a sequence prefix" is wrong
for this layout; a floor plan is one z-slice across ALL columns, i.e. strided.)

Panels per case: GT | what the model is GIVEN | edit() output coloured vs GT
(green = correct, red = extra/false positive, blue = missing).

  uv run python script/gen_edit_examples.py \
      --tower tower_16325 --out_dir generation/edit_examples
"""
import os, sys, json, argparse
from os.path import join as pjoin
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.paths import DATA_ROOT

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from utils.voxel import voxel_iou, grid_snap, GRID
from utils.model_io import load_vq

DATA = DATA_ROOT
NE_AZIM = 220
GREEN, RED, BLUE, GREY = '#3fa34d', '#e03131', '#2f6fd0', '#b9bec7'


# ───────────────────────────── rendering ─────────────────────────────
def cubes(ax, groups, ranges, elev=24, azim=NE_AZIM):
    """groups: list of (coords, facecolor, edgecolor)."""
    dx, dy, dz = GRID
    verts, fcs, ecs = [], [], []
    for coords, fc, ec in groups:
        for cx, cy, cz in np.asarray(coords).reshape(-1, 3):
            x0, y0, z0 = cx - dx/2, cy - dy/2, cz - dz/2
            x1, y1, z1 = x0 + dx, y0 + dy, z0 + dz
            verts += [
                [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0)],
                [(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)],
                [(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)],
                [(x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1)],
                [(x1, y1, z0), (x0, y1, z0), (x0, y1, z1), (x1, y1, z1)],
                [(x0, y1, z0), (x0, y0, z0), (x0, y0, z1), (x0, y1, z1)],
            ]
            fcs += [fc]*6
            ecs += [ec]*6
    if verts:
        ax.add_collection3d(Poly3DCollection(verts, facecolors=fcs, edgecolors=ecs,
                                            linewidths=0.18))
    (xr, yr, zr) = ranges
    ax.set_xlim(*xr); ax.set_ylim(*yr); ax.set_zlim(*zr)
    ax.set_box_aspect((xr[1]-xr[0], yr[1]-yr[0], (zr[1]-zr[0])*0.62), zoom=1.30)
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()


def ranges_of(coord_lists):
    allc = np.concatenate([np.asarray(c).reshape(-1, 3) for c in coord_lists if len(c)])
    lo, hi = allc.min(0), allc.max(0)
    pad = (hi - lo).max()*0.06 + 1
    return [(lo[i]-pad, hi[i]+pad) for i in range(3)]


def diff_groups(pred, gt):
    ps = set(map(tuple, grid_snap(np.asarray(pred, float)).astype(int)))
    gs = set(map(tuple, grid_snap(np.asarray(gt, float)).astype(int)))
    ok = np.array(sorted(ps & gs), float).reshape(-1, 3)
    fp = np.array(sorted(ps - gs), float).reshape(-1, 3)
    fn = np.array(sorted(gs - ps), float).reshape(-1, 3)
    return [(ok, GREEN, '#20492b'), (fp, RED, '#6d1111'), (fn, BLUE, '#123a6b')], len(fp), len(fn)


# ───────────────────────────── main ─────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tower', default='tower_16325')
    ap.add_argument('--alt_tower', default='tower_268707', help='for the wrong-image edge case')
    ap.add_argument('--out_dir', default='generation/edit_examples')
    ap.add_argument('--timesteps', type=int, default=10)
    ap.add_argument('--temperature', type=float, default=0.8)
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

    def tokens_of(name):
        m = np.load(pjoin(DATA, 'new_joint_vecs_int', name + '.npy')).reshape(-1, 3)
        n = (len(m)//4)*4
        with torch.no_grad():
            t = torch.from_numpy((m[:n]-mean)/std).float().unsqueeze(0).to(device)
            ids, _ = vq.encode(t)
        return m[:n].astype(float), ids[..., 0]

    def decode(ids, L):
        with torch.no_grad():
            xyz = vq.forward_decoder(ids[:, :L].clamp(min=0).unsqueeze(-1))[0].cpu().numpy()
        return grid_snap(xyz*std + mean)

    gt, gt_ids = tokens_of(args.tower)
    L = gt_ids.shape[1]
    feat = np.load(pjoin(DATA, 'image_features_dino_patch', args.tower + '.npy')).astype(np.float32)
    cond = torch.from_numpy(feat).unsqueeze(0).to(device).float()
    alt_feat = np.load(pjoin(DATA, 'image_features_dino_patch', args.alt_tower + '.npy')).astype(np.float32)
    alt_cond = torch.from_numpy(alt_feat).unsqueeze(0).to(device).float()

    # column starts -> token index of each column's BOTTOM (for vertical completion)
    zs = gt[:, 2]
    col_start = [0] + [i for i in range(1, len(gt)) if zs[i] <= zs[i-1]]
    bottom_tok = sorted({s//4 for s in col_start})

    def mk(free_idx, n=L):
        em = torch.zeros(1, n, dtype=torch.bool, device=device)
        em[:, list(free_idx)] = True
        return em

    CASES = []
    # ---------- applications (the intended, "easy" uses) ----------
    CASES.append(('A1  complete half the plan',
                  'left half of the building given (full height); model completes the right half',
                  mk(range(L//2, L)), L, cond))
    CASES.append(('A2  inpaint an interior region',
                  'a middle slab of the plan is erased and regenerated; rest held fixed',
                  mk(range(int(.40*L), int(.60*L))), L, cond))
    CASES.append(('A3  refine the whole tower',
                  'no mask: every token re-decoded at a low re-mask rate (cleanup pass)',
                  None, L, cond))
    CASES.append(('A4  podium -> tower (vertical)',
                  'only the bottom ~4 storeys of every column given; model builds upward',
                  mk([i for i in range(L) if i not in set(bottom_tok)]), L, cond))
    ext = int(1.20*L)
    CASES.append(('A5  extend the footprint (outpaint)',
                  f'all {L} GT tokens fixed, sequence grown to {ext}: new columns invented',
                  mk(range(L, ext), n=ext), ext, cond))
    # ---------- edge cases ----------
    CASES.append(('E1  only 10% given',
                  'almost no context: 10% of the plan fixed, 90% regenerated',
                  mk(range(L//10, L)), L, cond))
    CASES.append(('E2  free the sequence TAIL',
                  'last 15% of tokens freed (the far side of the plan; known weak spot)',
                  mk(range(int(.85*L), L)), L, cond))
    CASES.append(('E3  two disjoint holes',
                  'two separate regions erased at once',
                  mk(list(range(int(.20*L), int(.30*L))) + list(range(int(.60*L), int(.70*L)))), L, cond))
    CASES.append(('E4  WRONG images (conflict)',
                  f'half of {args.tower} fixed, but conditioned on {args.alt_tower} views',
                  mk(range(L//2, L)), L, alt_cond))
    CASES.append(('E5  invert: top given, base freed',
                  'the unnatural inverse of A4: upper levels fixed, lower levels regenerated',
                  mk(list(set(bottom_tok))), L, cond))

    rows, results = [], []
    for title, sub, em, n, c in CASES:
        toks = gt_ids
        if n > L:                                  # outpaint: pad the token grid
            toks = torch.cat([gt_ids, torch.zeros(1, n-L, dtype=gt_ids.dtype, device=device)], 1)
        out = model.edit(c, toks, torch.tensor([n], device=device),
                         timesteps=args.timesteps, cond_scale=1.0,
                         temperature=args.temperature, topk_filter_thres=0.9,
                         gsample=False, edit_mask=em)
        pred = decode(out, n)
        iou = voxel_iou(pred, gt)
        kept_ok = True if em is None else bool((out[:, :n][~em] == toks[~em]).all())
        n_free = n if em is None else int(em.sum())
        given = decode(toks, n)          # mask-free: the whole sequence is the input
        if em is not None:                          # show ONLY the fixed blocks
            fixed_tok = torch.where(em, torch.zeros_like(toks), toks)
            keep_pos = (~em)[0].nonzero().flatten()
            given = decode(fixed_tok, n)
            # restrict to blocks coming from kept token positions
            keep_blocks = np.concatenate([np.arange(int(i)*4, int(i)*4+4) for i in keep_pos]) \
                if len(keep_pos) else np.array([], int)
            keep_blocks = keep_blocks[keep_blocks < len(given)]
            given = given[keep_blocks]
        rows.append((title, sub, given, pred, iou, n_free, n, kept_ok))
        results.append(dict(case=title, iou=round(float(iou), 4), free=n_free, total=n,
                            fixed_exact=kept_ok))
        print(f'{title:34s} IoU {iou:.4f}  free {n_free}/{n}  fixed-exact {kept_ok}', flush=True)

    rng = ranges_of([gt] + [r[3] for r in rows])
    for tag, sel in (('applications', rows[:5]), ('edge_cases', rows[5:])):
        fig = plt.figure(figsize=(11.5, 3.05*len(sel) + 0.35))
        for r, (title, sub, given, pred, iou, n_free, n, kept_ok) in enumerate(sel):
            top = 0.955
            h = top/len(sel)
            y = top - (r+1)*h
            ax = fig.add_axes([0.005, y, 0.235, h*0.93], projection='3d')
            cubes(ax, [(gt, GREY, '#7d838d')], rng)
            ax.set_title('ground truth', fontsize=9, pad=-6)
            ax = fig.add_axes([0.245, y, 0.235, h*0.93], projection='3d')
            cubes(ax, [(given, BLUE, '#123a6b')], rng)
            lbl = ('input sequence (all tokens re-decoded)' if n_free == n
                   else f'given to model  ({n-n_free}/{n} tokens fixed)')
            ax.set_title(lbl, fontsize=9, pad=-6)
            ax = fig.add_axes([0.485, y, 0.235, h*0.93], projection='3d')
            g, nfp, nfn = diff_groups(pred, gt)
            cubes(ax, g, rng)
            ax.set_title(f'edit() output   IoU {iou:.3f}', fontsize=9, pad=-6)
            fig.text(0.745, y + h*0.62, title, fontsize=11, fontweight='bold', va='center')
            fig.text(0.745, y + h*0.44, sub, fontsize=8.5, va='center', wrap=True, color='#333')
            fig.text(0.745, y + h*0.22,
                     f'free {n_free}/{n} tokens   fixed region exact: {kept_ok}\n'
                     f'red (extra) {nfp}   blue (missing) {nfn}',
                     fontsize=8.5, va='center', color='#555')
        fig.text(0.5, 0.995, f'edit() {tag.replace("_", " ")} - {args.tower}  '
                             f'(green = correct, red = extra, blue = missing)',
                 ha='center', va='top', fontsize=12, fontweight='bold')
        out = pjoin(args.out_dir, f'edit_{tag}_{args.tower}.png')
        fig.savefig(out, dpi=150, facecolor='white', bbox_inches='tight')
        plt.close(fig)
        print(f'-> {out}', flush=True)
    json.dump(results, open(pjoin(args.out_dir, f'results_{args.tower}.json'), 'w'), indent=2)


if __name__ == '__main__':
    main()
