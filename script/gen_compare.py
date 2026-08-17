"""Side-by-side tower figures comparing ANY set of mask transformers.

Layout per tower: photo | GT | model_1 | model_2 | ... | [voted]. Each 3D panel
is a GT-vs-prediction diff (green=intersection, red=pred-only/extra,
blue=GT-only/missed) with voxel IoU in the title. Adaptable across versions —
pass whatever models you want via --models; no per-version script needed.

All models decode through --vq_name (windowed protocol). If --vote_model is set,
an extra column shows that model's N-draw voxel vote (temp 0.8, keep voxels present
in >= threshold*N samples — the deployable vote_output@0.3 recipe).

    # compare v5 vs v7, with a v7 voting column (the original v5-vs-v7 figure):
    CUDA_VISIBLE_DEVICES=4 uv run python \
        script/gen_compare.py --gpu_id 0 --num_towers 8 \
        --models myarch_mask_img_patch_v5:net_best_iou.tar myarch_mask_img_patch_v7:latest.tar \
        --vote_model myarch_mask_img_patch_v7:latest.tar \
        --output_dir generation/v5_vs_v7_voting

    # single model, no voting column:
    ... --models myarch_mask_img_patch_v7:latest.tar
"""
import os, sys, argparse, random
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from utils.paths import DATA_ROOT
from os.path import join as pjoin
from collections import Counter
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from gen_img_towers import (load_vq_model, load_patch_mask_transformer,
                            render_panel, render_diff_panel, axis_limits, VIEW_AZIM)
from utils.vq_decode import decode_windowed, probe_window_tokens
from utils.voxel import voxel_iou, coords_to_voxel_set
from utils.fixseed import fixseed
from utils.paths import IMAGE_ROOT as IMAGE_ROOT_DEFAULT

CKPTS = './checkpoints'
DATA_ROOT  = DATA_ROOT
IMAGE_ROOT = IMAGE_ROOT_DEFAULT


@torch.no_grad()
def gen_ids(model, feat_t, lens, n, temperature):
    cond = feat_t.repeat(n, 1, 1, 1)
    m_lens = lens.repeat(n)
    return model.generate(cond, m_lens, timesteps=10, cond_scale=1.0,
                          temperature=temperature, topk_filter_thres=0.9, gsample=False)


def _parse_model_spec(spec):
    """'dir:ckpt.tar' -> (label, dir, ckpt). ckpt defaults to net_best_iou.tar.
    Label includes a short ckpt tag so two ckpts of one model are distinguishable."""
    if ':' in spec:
        name, ckpt = spec.split(':', 1)
    else:
        name, ckpt = spec, 'net_best_iou.tar'
    short = name.replace('myarch_mask_img_patch_', '')
    ck = ckpt.replace('.tar', '').replace('net_best_', 'best_').replace('net_', '')
    return f'{short}/{ck}', name, ckpt


def main(a):
    fixseed(a.seed)
    device = torch.device(f'cuda:{a.gpu_id}')
    data_root = a.data_root
    image_roots = a.image_root
    vqm, vqo = load_vq_model(CKPTS, device, vq_name=a.vq_name)
    vqm.eval().to(device)
    nb = vqo.nb_code
    meta = pjoin(CKPTS, 'myarch', a.vq_name, 'meta')
    mean = np.load(pjoin(meta, 'mean.npy')); std = np.load(pjoin(meta, 'std.npy'))
    wt = probe_window_tokens(vqm, vqo.dim_pose, device)
    print(f'decode_mode={a.decode_mode}  vq={a.vq_name}  data_root={data_root}')

    # Single-pass comparison models (one diff column each).
    models = []
    for spec in a.models:
        label, name, ckpt = _parse_model_spec(spec)
        m = load_patch_mask_transformer(CKPTS, name, vqo, device, ckpt_name=ckpt)
        m.eval().to(device)
        models.append((label, m))
    # Optional voting models (one extra column each).
    votes = []
    for spec in (a.vote_model or []):
        vlabel, vname, vckpt = _parse_model_spec(spec)
        vm = load_patch_mask_transformer(CKPTS, vname, vqo, device, ckpt_name=vckpt)
        vm.eval().to(device)
        votes.append((vlabel, vm))

    def pick(split, n):
        with open(pjoin(data_root, f'{split}.txt')) as f:
            names = [l.strip() for l in f if l.strip()]
        random.Random(a.seed).shuffle(names)
        out = []
        for nm in names:
            if (os.path.exists(pjoin(data_root, 'new_joint_vecs_int', nm + '.npy')) and
                os.path.exists(pjoin(data_root, 'image_features_dino_patch', nm + '.npy'))):
                out.append(nm)
            if len(out) >= n: break
        return out

    def find_image(nm, view):
        for root in image_roots:
            ip = pjoin(root, nm, f'{view}_realistic.png')
            if os.path.exists(ip):
                return ip
        return None

    selected = [('val', nm) for nm in pick('val', a.num_towers)] + \
               [('train', nm) for nm in pick('train', a.num_towers)]
    os.makedirs(a.output_dir, exist_ok=True)
    print(f'{len(selected)} towers -> {a.output_dir}')

    @torch.no_grad()
    def decode(ids):
        g = ids.clamp(0, nb - 1).unsqueeze(-1)
        if a.decode_mode == 'whole':          # IBQ: single forward_decoder over full seq
            coords = vqm.forward_decoder(g).cpu().numpy()
        else:                                 # RVQ: windowed-64 chunked decode
            coords = decode_windowed(vqm, g, window_tokens=wt).cpu().numpy()
        return coords * std + mean

    for split, nm in selected:
        motion = np.load(pjoin(data_root, 'new_joint_vecs_int', nm + '.npy'))
        feat   = np.load(pjoin(data_root, 'image_features_dino_patch', nm + '.npy')).astype(np.float32)
        n_blk  = (len(motion) // 4) * 4
        gt = motion[:n_blk].astype(float)
        gt_set = coords_to_voxel_set(gt)
        feat_t = torch.from_numpy(feat).unsqueeze(0).to(device)
        lens = torch.LongTensor([n_blk // 4]).to(device)

        # One single-pass prediction per comparison model (temp 0.2 protocol).
        preds, titles, msg = [], [], []
        for label, m in models:
            pred = decode(gen_ids(m, feat_t, lens, 1, a.temperature))[0, :n_blk]
            iou = voxel_iou(pred, gt)
            preds.append(pred); titles.append(f'{label}  IoU={iou:.3f}')
            msg.append(f'{label}={iou:.3f}')

        # Optional voting columns: N draws at vote_temp -> voxel vote, one col per vote model.
        for vlabel, vm in votes:
            samp = decode(gen_ids(vm, feat_t, lens, a.vote_n, a.vote_temp))
            cnt = Counter()
            for s in range(a.vote_n):
                cnt.update(coords_to_voxel_set(samp[s, :n_blk]))
            voted_set = {v for v, c in cnt.items() if c >= a.threshold * a.vote_n}
            voted = np.array(sorted(voted_set), float) if voted_set else np.zeros((0, 3))
            iou_vote = (len(voted_set & gt_set) / len(voted_set | gt_set)) if (voted_set or gt_set) else 1.0
            preds.append(voted); titles.append(f'{vlabel} VOTED N{a.vote_n}  IoU={iou_vote:.3f}')
            msg.append(f'{vlabel}-voted={iou_vote:.3f}')

        # Axis ranges: all panels start at 0 so X-0/Y-0/Z-0 meet at one shared box
        # corner (consistent origin). X/Y share a max (square, undistorted).
        from utils.voxel import grid_snap
        allp = np.vstack([grid_snap(c) for c in [gt] + [p for p in preds if len(p)]])
        m_xy = int(np.ceil(max(allp[:, 0].max(), allp[:, 1].max()) / 10.0)) * 10
        m_z  = int(np.ceil(allp[:, 2].max() / 25.0)) * 25
        x_r = [0, m_xy]; y_r = [0, m_xy]; z_r = [0, m_z]
        # 4 rows (one per compass view) × ncols: photo | GT | <preds...>
        views = ['northeast', 'northwest', 'southeast', 'southwest']
        ncols = 2 + len(preds)
        fig = plt.figure(figsize=(5.0 * ncols, 5.6 * 4))
        for r, view in enumerate(views):
            az = VIEW_AZIM[view]
            # col0: photo (also carries the row's view label on its y-axis)
            ax_img = fig.add_subplot(4, ncols, r * ncols + 1)
            ip = find_image(nm, view)
            if ip: ax_img.imshow(mpimg.imread(ip))
            else: ax_img.text(0.5, 0.5, 'no image', ha='center', va='center', transform=ax_img.transAxes)
            ax_img.set_xticks([]); ax_img.set_yticks([])
            ax_img.set_ylabel(view.upper(), fontsize=14, fontweight='bold')
            if r == 0: ax_img.set_title('photo', fontsize=13, fontweight='bold')
            # col1: GT
            ax = fig.add_subplot(4, ncols, r * ncols + 2, projection='3d')
            render_panel(ax, gt, 'Ground Truth' if r == 0 else '', x_r, y_r, z_r,
                         color='#636EFA', azim=az, zoom=1.0)
            # cols 2..: prediction diffs
            for c, pred in enumerate(preds):
                ax = fig.add_subplot(4, ncols, r * ncols + 3 + c, projection='3d')
                render_diff_panel(ax, gt, pred, titles[c] if r == 0 else '',
                                  x_r, y_r, z_r, azim=az, zoom=1.0)

        fig.suptitle(f'{nm} [{split}]  ({n_blk} blocks)   '
                     f'green = correct · red = extra · blue = missed',
                     fontsize=16, fontweight='bold', y=0.995)
        out = pjoin(a.output_dir, f'{split}_{nm}_compare_4view.png')
        # extra vertical gap so each row's tick labels don't collide with the row below
        fig.subplots_adjust(left=0.03, right=0.99, top=0.96, bottom=0.02,
                            wspace=0.05, hspace=0.30)
        plt.savefig(out, dpi=150, bbox_inches='tight'); plt.close()
        print(f'  [{split}] {nm}: ' + ' '.join(msg) + f' -> {out}')

    print('Done.')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--gpu_id', type=int, default=0)
    p.add_argument('--num_towers', type=int, default=8, help='per split (val+train)')
    p.add_argument('--vq_name', default='rvq_v2_4x_nc1024_nq3')
    p.add_argument('--decode_mode', choices=['windowed', 'whole'], default='windowed',
                   help='windowed-64 for RVQ (default), whole-seq for IBQ')
    p.add_argument('--data_root', default=DATA_ROOT)
    p.add_argument('--image_root', nargs='+',
                   default=[IMAGE_ROOT_DEFAULT],
                   help='one or more raw-PNG roots; each tower searched in order')
    p.add_argument('--models', nargs='+',
                   default=['myarch_mask_img_patch_v5:net_best_iou.tar',
                            'myarch_mask_img_patch_v7:latest.tar'],
                   help='Models to compare as dir:ckpt.tar (one single-pass diff column each)')
    p.add_argument('--vote_model', nargs='*', default=['myarch_mask_img_patch_v7:latest.tar'],
                   help='dir:ckpt.tar specs to add voted columns for (one each); empty to skip')
    p.add_argument('--temperature', type=float, default=0.2, help='single-pass temp')
    p.add_argument('--vote_n', type=int, default=16)
    p.add_argument('--vote_temp', type=float, default=0.8)
    p.add_argument('--threshold', type=float, default=0.3)
    p.add_argument('--output_dir', default='./generation/model_compare')
    p.add_argument('--seed', type=int, default=42)
    main(p.parse_args())
