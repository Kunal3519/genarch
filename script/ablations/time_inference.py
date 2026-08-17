"""Per-tower inference latency for {single | oracle | voting}, two --mode s:

  --mode e2e     TRUE end-to-end: 4 PNGs on disk -> DINOv2 patch encode ->
                 mask transformer -> decode -> coords/voxels. Every stage CUDA-synced.
                 Replicates precompute_patch_dl.py preprocessing:
                 Resize(256,BICUBIC)->CenterCrop(224)->ToTensor->ImageNet-norm,
                 DINOv2 ViT-L/14 forward_features -> x_norm_patchtokens -> (4,256,1024).
  --mode voting  gen-stage breakdown only, from a PRECOMPUTED feature .npy (DINOv2
                 reported as a fixed ~27 ms estimate; skips PNG load + DINOv2 forward).

Replaces time_e2e.py (--mode e2e) and time_voting.py (--mode voting).

    CUDA_VISIBLE_DEVICES=4 uv run python \
        script/time_inference.py --mode e2e --gpu_id 0 --n_samples 16 --temperature 0.8 --reps 20
    ... --mode voting --gpu_id 0 --n_samples 16 --temperature 0.8 --reps 10
"""
import argparse, os, sys, time
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))  # repo root
sys.path.insert(0, os.path.dirname(_HERE))                   # script/
from utils.paths import IMAGE_ROOT
from os.path import join as pjoin
from collections import Counter
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from eval_sampling import (load_mask_model, MASK_TIMESTEPS, MASK_COND_SCALE, MASK_TOPK,
                           CHECKPOINTS_DIR, DATASET_NAME, DATA_ROOT)
from utils.model_io import load_vq
from utils.vq_decode import decode_windowed, DEFAULT_WINDOW_TOKENS
from utils.voxel import coords_to_voxel_set, voxel_iou
from utils.fixseed import fixseed

RAW_ROOT = IMAGE_ROOT
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


def mode_e2e(args, device):
    # ---- models (one-time load, timed separately) ----
    t = time.perf_counter()
    vq_model, vq_opt = load_vq(CHECKPOINTS_DIR, args.vq_name, device=device,
                                dim_pose=3, dataset_name=DATASET_NAME)
    vq_model.eval().to(device)
    nb_code = vq_model.quantizer.layers[0].nb_code
    model = load_mask_model(vq_opt, args.mask_name, args.ckpt, device)
    print('Loading DINOv2 ViT-L/14 ...')
    dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14').eval().to(device)
    for p in dino.parameters(): p.requires_grad = False
    load_s = time.perf_counter() - t
    meta = pjoin(CHECKPOINTS_DIR, DATASET_NAME, args.vq_name, 'meta')
    mean = np.load(pjoin(meta, 'mean.npy')); std = np.load(pjoin(meta, 'std.npy'))

    # ---- pick a tower that has both raw images and a converted motion ----
    motion_dir = pjoin(DATA_ROOT, 'new_joint_vecs_int')
    def ok(n):
        return (os.path.exists(pjoin(motion_dir, n + '.npy')) and
                all(os.path.exists(pjoin(RAW_ROOT, n, v)) for v in VIEWS))
    if args.tower and ok(args.tower):
        name = args.tower
    else:
        name = next(n for n in sorted(os.listdir(RAW_ROOT)) if ok(n))
    motion = np.load(pjoin(motion_dir, name + '.npy'))
    m_tok = len(motion) // 4
    n_cmp = m_tok * 4
    gt_set = coords_to_voxel_set(motion[:n_cmp])
    img_paths = [pjoin(RAW_ROOT, name, v) for v in VIEWS]
    print(f'tower {name}: {len(motion)} blocks -> {m_tok} tokens   N={args.n_samples} temp={args.temperature}\n')

    N = args.n_samples
    thr = args.threshold * N

    # ---- stage fns ----
    def load_imgs():                                  # disk -> CPU tensor [4,3,224,224]
        return torch.stack([DINO_TF(Image.open(p).convert('RGB')) for p in img_paths])

    def dino_encode(batch_cpu):                       # -> feat (1,4,256,1024) on GPU
        x = batch_cpu.to(device)
        out = dino.forward_features(x)
        patch = out['x_norm_patchtokens']             # [4,256,1024]
        return patch.reshape(1, 4, 256, -1)

    def gen(feat, n):                                 # -> list of n coord arrays
        cond = feat.repeat(n, 1, 1, 1).float()
        m_lens = torch.full((n,), m_tok, device=device, dtype=torch.long)
        ids = model.generate(cond, m_lens, timesteps=MASK_TIMESTEPS,
                             cond_scale=MASK_COND_SCALE, temperature=args.temperature,
                             topk_filter_thres=MASK_TOPK, gsample=False).clamp(0, nb_code - 1)
        coords = decode_windowed(vq_model, ids.unsqueeze(-1),
                                 window_tokens=DEFAULT_WINDOW_TOKENS).cpu().numpy()
        return [coords[i, :n_cmp] * std + mean for i in range(n)]

    def post_oracle(samples):
        return max(voxel_iou(x, motion[:n_cmp]) for x in samples)

    def post_vote(samples):
        cnt = Counter()
        for x in samples: cnt.update(coords_to_voxel_set(x))
        return {v for v, c in cnt.items() if c >= thr}

    @torch.no_grad()
    def bench(fn, reps, warmup=3):
        for _ in range(warmup): fn()
        torch.cuda.synchronize(); t0 = time.perf_counter()
        for _ in range(reps): fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / reps * 1000  # ms

    # measure stages
    t_load = bench(load_imgs, args.reps)                       # CPU (no GPU work)
    imgs_cpu = load_imgs()
    t_dino = bench(lambda: dino_encode(imgs_cpu), args.reps)
    feat = dino_encode(imgs_cpu)
    t_gen1 = bench(lambda: gen(feat, 1), args.reps)
    t_genN = bench(lambda: gen(feat, N), max(5, args.reps // 2))
    sN = gen(feat, N)
    t_oracle_post = bench(lambda: post_oracle(sN), args.reps * 3)
    t_vote_post = bench(lambda: post_vote(sN), args.reps * 3)

    shared = t_load + t_dino
    L = '-' * 60
    print('=' * 60)
    print(f'  END-TO-END per-tower latency   {torch.cuda.get_device_name(device)}')
    print(f'  {args.mask_name}/{args.ckpt}  N={N}  temp={args.temperature}')
    print('=' * 60)
    print(f'  [one-time model load: {load_s:.1f}s — excluded below]\n')
    print(f'  SHARED front-end (run once per tower):')
    print(f'    1. load 4 PNGs + preprocess (CPU)   {t_load:7.1f} ms')
    print(f'    2. DINOv2 patch encode (4 views)    {t_dino:7.1f} ms')
    print(f'    shared subtotal                     {shared:7.1f} ms')
    print(L)
    print(f'  3. mask transformer + RVQ decode:')
    print(f'    single (1 draw)                     {t_gen1:7.1f} ms')
    print(f'    N={N} draws (batched, one forward)   {t_genN:7.1f} ms')
    print(f'  4. post-process:')
    print(f'    oracle score (needs GT)             {t_oracle_post:7.1f} ms')
    print(f'    voting (Counter + threshold)        {t_vote_post:7.1f} ms')
    print('=' * 60)
    print(f'  TOTAL  start -> finished tower:')
    print(f'    SINGLE  = {shared + t_gen1:7.1f} ms')
    print(f'    ORACLE  = {shared + t_genN + t_oracle_post:7.1f} ms   (not deployable — needs GT)')
    print(f'    VOTING  = {shared + t_genN + t_vote_post:7.1f} ms')
    print(L)
    print(f'    voting / single = {(shared + t_genN + t_vote_post) / (shared + t_gen1):.2f}x')
    print('=' * 60)


def mode_voting(args, device):
    """Gen-stage breakdown from a precomputed feature .npy (DINOv2 = fixed ~27ms estimate)."""
    vq_model, vq_opt = load_vq(CHECKPOINTS_DIR, args.vq_name, device=device,
                                dim_pose=3, dataset_name=DATASET_NAME)
    vq_model.eval().to(device)
    nb_code = vq_model.quantizer.layers[0].nb_code
    model = load_mask_model(vq_opt, args.mask_name, args.ckpt, device)
    meta = pjoin(CHECKPOINTS_DIR, DATASET_NAME, args.vq_name, 'meta')
    mean = np.load(pjoin(meta, 'mean.npy')); std = np.load(pjoin(meta, 'std.npy'))

    motion_dir = pjoin(DATA_ROOT, 'new_joint_vecs_int')
    feat_dir   = pjoin(DATA_ROOT, 'image_features_dino_patch')
    name = args.tower or sorted(n[:-4] for n in os.listdir(feat_dir) if n.endswith('.npy'))[0]
    motion = np.load(pjoin(motion_dir, name + '.npy'))
    feat   = np.load(pjoin(feat_dir, name + '.npy'))
    m_tok  = len(motion) // 4
    n_cmp  = m_tok * 4
    gt_set = coords_to_voxel_set(motion[:n_cmp])
    print(f'tower {name}: {len(motion)} blocks → {m_tok} tokens, N={args.n_samples}\n')

    def gen_decode(n):
        cond = torch.from_numpy(feat).unsqueeze(0).repeat(n, 1, 1, 1).to(device).float()
        m_lens = torch.full((n,), m_tok, device=device, dtype=torch.long)
        with torch.no_grad():
            ids = model.generate(cond, m_lens, timesteps=MASK_TIMESTEPS,
                                 cond_scale=MASK_COND_SCALE, temperature=args.temperature,
                                 topk_filter_thres=MASK_TOPK, gsample=False)
            coords = decode_windowed(vq_model, ids.clamp(0, nb_code - 1).unsqueeze(-1),
                                     window_tokens=DEFAULT_WINDOW_TOKENS).cpu().numpy()
        return [coords[i, :n_cmp] * std + mean for i in range(n)]

    def post_oracle(samples):
        return max(set_iou(coords_to_voxel_set(x), gt_set) for x in samples)

    def post_vote(samples):
        cnt = Counter()
        for x in samples:
            cnt.update(coords_to_voxel_set(x))
        thr = args.threshold * len(samples)
        return {v for v, c in cnt.items() if c >= thr}

    def bench(fn, reps, warmup=3):
        for _ in range(warmup): fn()
        torch.cuda.synchronize(); t0 = time.perf_counter()
        for _ in range(reps): fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / reps * 1000

    N = args.n_samples
    t_gen1 = bench(lambda: gen_decode(1), args.reps)
    t_genN = bench(lambda: gen_decode(N), args.reps)
    samplesN = gen_decode(N)
    t_oracle_post = bench(lambda: post_oracle(samplesN), args.reps * 3)
    t_vote_post   = bench(lambda: post_vote(samplesN), args.reps * 3)

    print('=' * 62)
    print(f'  Single-tower timing  ({m_tok} tok, N={N}, temp={args.temperature})  H200')
    print('=' * 62)
    print(f'  [shared] DINOv2 image encode (4 views)  ~ 27 ms')
    print('  ' + '-' * 58)
    print(f'  single-pass : gen+decode(1)             = {t_gen1:7.1f} ms')
    print(f'  oracle      : gen+decode({N}) {t_genN:6.1f} + score {t_oracle_post:5.1f} = {t_genN+t_oracle_post:7.1f} ms')
    print(f'  voting      : gen+decode({N}) {t_genN:6.1f} + vote  {t_vote_post:5.1f} = {t_genN+t_vote_post:7.1f} ms')
    print('  ' + '-' * 58)
    print(f'  END-TO-END (incl. DINOv2):')
    print(f'    single-pass = {27+t_gen1:7.1f} ms')
    print(f'    oracle      = {27+t_genN+t_oracle_post:7.1f} ms   (not deployable — needs GT)')
    print(f'    voting      = {27+t_genN+t_vote_post:7.1f} ms')
    print(f'  voting / single-pass end-to-end = {(27+t_genN+t_vote_post)/(27+t_gen1):.2f}x')
    print('=' * 62)


MODES = {'e2e': mode_e2e, 'voting': mode_voting}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--mode', default='e2e', choices=list(MODES))
    ap.add_argument('--mask_name', default='myarch_mask_img_patch_v7')
    ap.add_argument('--ckpt', default='latest.tar')
    ap.add_argument('--vq_name', default='rvq_v2_4x_nc1024_nq3')
    ap.add_argument('--n_samples', type=int, default=16)
    ap.add_argument('--temperature', type=float, default=0.8)
    ap.add_argument('--threshold', type=float, default=0.3)
    ap.add_argument('--tower', default=None, help='tower name; default = first with imgs+motion')
    ap.add_argument('--reps', type=int, default=20)
    ap.add_argument('--gpu_id', type=int, default=0)
    args = ap.parse_args()

    fixseed(0)
    device = torch.device(f'cuda:{args.gpu_id}')
    MODES[args.mode](args, device)


if __name__ == '__main__':
    main()
