"""True end-to-end pipeline from RAW IMAGES (no cached features), validation + leak check.

For each tower: load its 4 PNG views, run DINOv2 ViT-L/14 forward_features LIVE (the
exact precompute_patch_dl.py preprocessing), feed those features to the mask
transformer, generate tokens, decode (whole-seq for IBQ), and score voxel-IoU vs GT.

Two questions this answers:
  1. LEAK CHECK, does the high IoU survive when nothing GT-derived touches generation?
     - view sweep k=0..4: use only the first k real views, ZERO the rest. k=0 = no image
       info at all (unconditional). If IoU stays high at k=0/1, something is leaking GT.
       It should degrade monotonically; k=4 should match the cached-feature full eval.
     - The ONLY GT-derived input to generate() is the token LENGTH (m_lens = n_blocks//4),
       same for every model (incl. v7). IoU itself only uses GT for scoring, never generation.
  2. PIPELINE CHECK, recomputed-from-PNG features vs the cached .npy at k=4 should match
     (within sampling noise), confirming no caching/feature artifact inflates the number.

Runs both checkpoints. This is also the "generate a tower from a brand-new image" path.

    CUDA_VISIBLE_DEVICES=6 uv run python \
        script/eval_e2e_from_images.py --gpu_id 0 --n_towers 8 \
        --mask_name myarch_mask_img_patch_v8_2 --ckpt net_best_iou.tar \
        --vq_name ibq_4x_nc1024_nq1 --decode_mode whole \
        --data_root $GENARCH_DATA_ROOT \
        --image_root $GENARCH_IMAGE_ROOT
"""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from utils.paths import DATA_ROOT, IMAGE_ROOT
from os.path import join as pjoin
from collections import Counter
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from eval_sampling import load_mask_model, CHECKPOINTS_DIR, DATASET_NAME, MASK_TIMESTEPS, MASK_COND_SCALE, MASK_TOPK
from utils.model_io import load_vq
from utils.vq_decode import decode_windowed, DEFAULT_WINDOW_TOKENS
from utils.voxel import voxel_iou, coords_to_voxel_set
from utils.fixseed import fixseed

# EXACT replica of precompute_patch_dl.py preprocessing.
VIEWS = ['northeast_realistic.png', 'northwest_realistic.png',
         'southeast_realistic.png', 'southwest_realistic.png']
DINO_TF = transforms.Compose([
    transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])


def set_iou(a, b):
    if not a and not b: return 1.0
    if not a or not b:  return 0.0
    return len(a & b) / len(a | b)


def find_views(nm, image_roots):
    """Return list of 4 PNG paths (None where missing), searching roots in order."""
    paths = []
    for v in VIEWS:
        hit = None
        for root in image_roots:
            p = pjoin(root, nm, v)
            if os.path.exists(p):
                hit = p; break
        paths.append(hit)
    return paths


@torch.no_grad()
def dino_encode(dino, img_paths, device):
    """4 PNGs -> (4,256,1024) patch features, exactly as precompute does."""
    imgs = torch.stack([DINO_TF(Image.open(p).convert('RGB')) for p in img_paths]).to(device)
    out = dino.forward_features(imgs)
    return out['x_norm_patchtokens'].reshape(4, 256, -1).float()  # (4,256,1024)


@torch.no_grad()
def decode(vqm, ids, nb, mean, std, mode, wt):
    g = ids.clamp(0, nb - 1).unsqueeze(-1)
    coords = (vqm.forward_decoder(g).cpu().numpy() if mode == 'whole'
              else decode_windowed(vqm, g, window_tokens=wt).cpu().numpy())
    return coords * std + mean


@torch.no_grad()
def run_model(model, vqm, feat4, n_views, m_tok, n_cmp, n_samples, temp, device,
              nb, mean, std, mode, wt):
    """feat4=(4,256,1024); use first n_views real views, zero the rest. Returns (single, voted_set)."""
    cond = feat4.clone()
    if n_views < 4:
        cond[n_views:] = 0.0
    cond = cond.unsqueeze(0).repeat(n_samples, 1, 1, 1)
    m_lens = torch.full((n_samples,), m_tok, device=device, dtype=torch.long)
    ids = model.generate(cond, m_lens, timesteps=MASK_TIMESTEPS, cond_scale=MASK_COND_SCALE,
                         temperature=temp, topk_filter_thres=MASK_TOPK, gsample=False)
    coords = decode(vqm, ids, nb, mean, std, mode, wt)
    xyz = [coords[i, :n_cmp] for i in range(n_samples)]
    cnt = Counter()
    for x in xyz:
        cnt.update(coords_to_voxel_set(x))
    voted = {v for v, c in cnt.items() if c >= 0.3 * n_samples}
    return xyz[0], voted


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--mask_name', default='myarch_mask_img_patch_v8_2')
    ap.add_argument('--ckpt', nargs='+', default=['net_best_iou.tar'])
    ap.add_argument('--vq_name', default='ibq_4x_nc1024_nq1')
    ap.add_argument('--decode_mode', choices=['windowed', 'whole'], default='whole')
    ap.add_argument('--data_root', default=DATA_ROOT)
    ap.add_argument('--image_root', nargs='+',
                    default=[IMAGE_ROOT])
    ap.add_argument('--split', default='val40k')
    ap.add_argument('--n_towers', type=int, default=8)
    ap.add_argument('--views', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    ap.add_argument('--n_samples', type=int, default=16)
    ap.add_argument('--temperature', type=float, default=0.8)
    ap.add_argument('--gpu_id', type=int, default=0)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    fixseed(args.seed)
    device = torch.device(f'cuda:{args.gpu_id}')
    vqm, vqo = load_vq(CHECKPOINTS_DIR, args.vq_name, device=device, dim_pose=3, dataset_name=DATASET_NAME)
    vqm.eval().to(device)
    nb = vqm.quantizer.layers[0].nb_code
    meta = pjoin(CHECKPOINTS_DIR, DATASET_NAME, args.vq_name, 'meta')
    mean = np.load(pjoin(meta, 'mean.npy')); std = np.load(pjoin(meta, 'std.npy'))
    wt = DEFAULT_WINDOW_TOKENS
    models = {ck: load_mask_model(vqo, args.mask_name, ck, device) for ck in args.ckpt}

    print('Loading DINOv2 ViT-L/14 ...')
    dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14').eval().to(device)
    for p in dino.parameters(): p.requires_grad = False

    motion_dir = pjoin(args.data_root, 'new_joint_vecs_int')
    feat_dir   = pjoin(args.data_root, 'image_features_dino_patch')
    with open(pjoin(args.data_root, f'{args.split}.txt')) as f:
        names = sorted(l.strip() for l in f if l.strip())
    rng = np.random.default_rng(args.seed); rng.shuffle(names)
    picked = []
    for nm in names:
        if not os.path.exists(pjoin(motion_dir, nm + '.npy')): continue
        vp = find_views(nm, args.image_root)
        if any(p is None for p in vp): continue
        picked.append((nm, vp))
        if len(picked) >= args.n_towers: break
    print(f'{len(picked)} unseen {args.split} towers (DINOv2 recomputed from PNGs)\n')

    # accumulators: sweep[ck][k] = [single...], votes; leak[ck] = (recomp_single, cached_single, recomp_vote, cached_vote, featdiff)
    sweep = {ck: {k: {'s': [], 'v': []} for k in args.views} for ck in args.ckpt}
    leak = {ck: {'rs': [], 'cs': [], 'rv': [], 'cv': []} for ck in args.ckpt}
    featdiff = []

    for nm, vp in picked:
        motion = np.load(pjoin(motion_dir, nm + '.npy'))
        n_cmp = (len(motion) // 4) * 4; m_tok = len(motion) // 4
        gt_xyz = motion[:n_cmp]; gt_set = coords_to_voxel_set(gt_xyz)
        feat_real = dino_encode(dino, vp, device)                       # recomputed
        feat_cached = torch.from_numpy(np.load(pjoin(feat_dir, nm + '.npy')).astype(np.float32)).to(device)
        featdiff.append(float((feat_real - feat_cached).abs().mean()))

        for ck, model in models.items():
            for k in args.views:
                s_xyz, voted = run_model(model, vqm, feat_real, k, m_tok, n_cmp,
                                         args.n_samples, args.temperature, device, nb, mean, std,
                                         args.decode_mode, wt)
                sweep[ck][k]['s'].append(voxel_iou(s_xyz, gt_xyz))
                sweep[ck][k]['v'].append(set_iou(voted, gt_set))
            # leak: cached-feature run at full 4 views (compare to recomputed k=4)
            cs_xyz, cvoted = run_model(model, vqm, feat_cached, 4, m_tok, n_cmp,
                                       args.n_samples, args.temperature, device, nb, mean, std,
                                       args.decode_mode, wt)
            leak[ck]['rs'].append(sweep[ck][4]['s'][-1]); leak[ck]['cs'].append(voxel_iou(cs_xyz, gt_xyz))
            leak[ck]['rv'].append(sweep[ck][4]['v'][-1]); leak[ck]['cv'].append(set_iou(cvoted, gt_set))
        print(f'  {nm}: ' + '  '.join(
            f'{ck.replace("net_best_","").replace(".tar","")} '
            f'v4 single={np.mean(sweep[ck][4]["s"][-1:]):.3f}/vote={np.mean(sweep[ck][4]["v"][-1:]):.3f}'
            for ck in args.ckpt))

    print('\n' + '=' * 78)
    print(f'  END-TO-END FROM IMAGES, {args.mask_name}  ({len(picked)} unseen {args.split} towers, N={args.n_samples} temp={args.temperature})')
    print(f'  DINOv2 recomputed live from PNGs.  mean|recomp − cached| feature diff = {np.mean(featdiff):.2e}')
    print('=' * 78)
    for ck in args.ckpt:
        c = ck.replace('.tar', '')
        print(f'\n  [{c}]  view sweep (single-pass / N={args.n_samples} vote):')
        print(f'    {"#views":>7} | {"single":>8} | {"vote@0.3":>9}')
        print(f'    {"-"*7}-+-{"-"*8}-+-{"-"*9}')
        for k in args.views:
            s = np.mean(sweep[ck][k]['s']); v = np.mean(sweep[ck][k]['v'])
            tag = '  <- no image (unconditional)' if k == 0 else ('  <- full' if k == 4 else '')
            print(f'    {k:>7} | {s:>8.4f} | {v:>9.4f}{tag}')
        rs, cs = np.mean(leak[ck]['rs']), np.mean(leak[ck]['cs'])
        rv, cv = np.mean(leak[ck]['rv']), np.mean(leak[ck]['cv'])
        print(f'    LEAK CHECK @4 views:  recomputed single={rs:.4f} vote={rv:.4f}  |  '
              f'cached single={cs:.4f} vote={cv:.4f}  (Δsingle={rs-cs:+.4f} Δvote={rv-cv:+.4f})')
    print('=' * 78)


if __name__ == '__main__':
    main()
