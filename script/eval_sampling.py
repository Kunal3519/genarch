"""
Sequence-level inference eval for the MaskTransformer, one tool, many --mode s.

All modes share the same core: draw N stochastic samples per tower (same image
cond, independent temperature sampling), decode via the windowed protocol (or
whole-seq for IBQ), and score voxel-IoU. They differ only in what they sweep/report.
Replaces the old per-experiment scripts (eval_best_of_n, sweep_n, sweep_vote,
eval_selectors, full_eval_voting, eval_v8_ibq_full).

  --mode bestofn    single / mean / consensus / fewest-floating / oracle at fixed N
  --mode sweep_n    sweep N (--ns), report single/consensus/oracle + gen timing
  --mode sweep_thr  N × vote-threshold matrix (--ns × --thresholds) of vote_output IoU
  --mode selectors  GT-free selector bake-off (consensus/floating/vote_pick/vote_output/combined)
  --mode full       full-split single/oracle/vote@thr, shardable + --combine, with timing
  --mode decode_ab  windowed-vs-whole decode A/B + voting (IBQ check), shardable + --combine
  --mode sweep_cfg  classifier-free-guidance scale sweep (--cond_scales, '1:3' = per-step schedule)
  --mode errors     vote-error anatomy: FN/FP pool coverage, height/surface/sequence localization
  --mode len_sens   IoU sensitivity to token-length input error (--len_offsets, single + voted)
  --mode len_jitter spread the N vote samples' lengths around an erroneous center (--jitters,
                    sym + up-biased); centers = GT+offsets and the real predictor (--len_pred_ckpt)

This module is also the importable library (load_mask_model, eval_tower, load_records,
consensus_pick, …) used by ablations/ablate_views.py.

Usage:
    CUDA_VISIBLE_DEVICES=1 uv run python \
        script/eval_sampling.py --mode bestofn --n_samples 16 --n_towers 200 --gpu_id 0
    ... --mode sweep_n   --ns 8 16 32 64 128 --temperature 0.8 --n_towers 100
    ... --mode sweep_thr --ns 8 16 32 --thresholds 0.2 0.3 0.4 --temperature 0.8
    ... --mode selectors --n_samples 32 --temperature 0.8 --n_towers 100
    ... --mode full      --ckpt latest.tar --split val --n_samples 16 --shard_idx 0 --num_shards 4
    ... --mode full      --combine --ckpt latest.tar --split val --n_samples 16
    ... --mode decode_ab --vq_name ibq_4x_nc1024_nq1 --data_root $GENARCH_DATA_ROOT
"""
import argparse
import os
import sys
import time
import glob
import re
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from utils.paths import DATA_ROOT

from os.path import join as pjoin
from collections import Counter
import numpy as np
import torch
from tqdm import tqdm

from models.mask_transformer.transformer import MaskTransformer
from utils.get_opt import get_opt
from utils.model_io import load_vq
from utils.vq_decode import decode_windowed, DEFAULT_WINDOW_TOKENS
from utils.voxel import voxel_iou, coords_to_voxel_set, grid_snap
from utils.fixseed import fixseed
from test_decode_artifacts import count_components_6conn

CHECKPOINTS_DIR = './checkpoints'
DATASET_NAME    = 'myarch'
DATA_ROOT       = DATA_ROOT


def _shard_glob(tag, num_shards):
    """Shard files for an EXACT (tag, num_shards). --combine uses this so a
    leftover run sharded a different way (e.g. an old _shardNof1) can't leak
    into the average."""
    return sorted(glob.glob(f'/tmp/{tag}_shard*of{num_shards}.npy'))


def _purge_stale_shards(tag, num_shards):
    """Before a sharded run, drop this tag's shard files from a run with a
    DIFFERENT num_shards (cross-config leftovers that --combine would sum in).
    Same-num_shards siblings are left alone (they belong to the live run; each
    shard overwrites its own file via np.save)."""
    for p in glob.glob(f'/tmp/{tag}_shard*of*.npy'):
        m = re.search(r'_shard\d+of(\d+)\.npy$', p)
        if m and int(m.group(1)) != num_shards:
            try:
                os.remove(p)
                print(f'  purged stale shard: {os.path.basename(p)}')
            except OSError:
                pass

# Generation hyperparams, identical to compute_val_iou / eval_mask_full.
# (temperature is now a CLI arg; the others match the standard eval protocol.)
MASK_TIMESTEPS  = 10
MASK_COND_SCALE = 1.0
MASK_TOPK       = 0.9


def _cond_mode_from_opt(opt):
    if getattr(opt, 'use_cls_prefix', False):
        return 'image_patch_clsprefix'
    return 'image_patch'


def load_mask_model(vq_opt, mask_name, ckpt_file, device):
    opt_path = pjoin(CHECKPOINTS_DIR, DATASET_NAME, mask_name, 'opt.txt')
    mask_opt = get_opt(opt_path, device)
    mask_opt.num_tokens        = vq_opt.nb_code
    mask_opt.num_quantizers    = vq_opt.num_quantizers
    mask_opt.code_dim          = vq_opt.code_dim
    mask_opt.max_motion_len    = 512
    mask_opt.max_motion_length = 2000
    cond_mode = _cond_mode_from_opt(mask_opt)
    print(f'cond_mode: {cond_mode}  vq: {vq_opt.nb_code}x{vq_opt.num_quantizers}')

    model = MaskTransformer(
        code_dim=vq_opt.code_dim, cond_mode=cond_mode,
        latent_dim=mask_opt.latent_dim, ff_size=mask_opt.ff_size,
        num_layers=mask_opt.n_layers, num_heads=mask_opt.n_heads,
        dropout=mask_opt.dropout, clip_dim=1024,
        cond_drop_prob=mask_opt.cond_drop_prob, clip_version=None, opt=mask_opt)

    ckpt_path = pjoin(CHECKPOINTS_DIR, DATASET_NAME, mask_name, 'model', ckpt_file)
    ckpt = torch.load(ckpt_path, map_location='cpu')
    ep = ckpt.get('ep', '?')
    sd = ckpt.get('t2m_transformer', ckpt)
    if any(k.startswith('module.') for k in sd):
        sd = {k.replace('module.', '', 1): v for k, v in sd.items()}
    sd = {k: v for k, v in sd.items() if not k.startswith('sbert_model.')}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not unexpected, f'unexpected: {unexpected[:5]}'
    model.eval().to(device)
    print(f'Loaded {mask_name}/{ckpt_file} (epoch {ep})')
    return model


def load_records(split_file, n_towers, seed, data_root=DATA_ROOT):
    motion_dir = pjoin(data_root, 'new_joint_vecs_int')
    feat_dir   = pjoin(data_root, 'image_features_dino_patch')
    with open(split_file) as f:
        names = sorted(l.strip() for l in f if l.strip())
    rng = np.random.default_rng(seed)
    rng.shuffle(names)
    records = []
    for name in names:
        mpath = pjoin(motion_dir, name + '.npy')
        fpath = pjoin(feat_dir, name + '.npy')
        if not (os.path.exists(mpath) and os.path.exists(fpath)):
            continue
        records.append((name, np.load(mpath), np.load(fpath)))
        if len(records) >= n_towers:
            break
    return records


@torch.no_grad()
def eval_tower(model, vq_model, mean, std, motion, feat, n_samples, temperature, device, nb_code,
               cond_scale=MASK_COND_SCALE, decode_mode='windowed'):
    """Generate n_samples for one tower, return per-sample IoU vs GT + the decoded coords."""
    m_len = len(motion)
    n_cmp = (m_len // 4) * 4
    gt_xyz = motion[:n_cmp]

    # Replicate condition N times → N independent samples in one generate() call.
    cond = torch.from_numpy(feat).unsqueeze(0).repeat(n_samples, *([1] * feat.ndim)).to(device).float()
    m_lens_tok = torch.full((n_samples,), m_len // 4, device=device, dtype=torch.long)

    gen_ids = model.generate(
        cond, m_lens_tok,
        timesteps=MASK_TIMESTEPS, cond_scale=cond_scale,
        temperature=temperature, topk_filter_thres=MASK_TOPK, gsample=False)
    gen_ids = gen_ids.clamp(0, nb_code - 1)
    if decode_mode == 'whole':
        coords = vq_model.forward_decoder(gen_ids.unsqueeze(-1)).cpu().numpy()
    else:
        coords = decode_windowed(vq_model, gen_ids.unsqueeze(-1),
                                 window_tokens=DEFAULT_WINDOW_TOKENS).cpu().numpy()

    sample_xyz = [coords[i, :n_cmp] * std + mean for i in range(n_samples)]
    ious_gt = np.array([voxel_iou(xyz, gt_xyz) for xyz in sample_xyz])
    return ious_gt, sample_xyz


def consensus_pick(sample_xyz):
    """Pick index of the sample with the highest mean pairwise IoU to the others (no GT)."""
    n = len(sample_xyz)
    if n == 1:
        return 0
    pair = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            iou = voxel_iou(sample_xyz[i], sample_xyz[j])
            pair[i, j] = pair[j, i] = iou
    mean_agree = pair.sum(1) / (n - 1)
    return int(mean_agree.argmax())


def floating_pick(sample_xyz):
    """Pick index of the sample with the fewest floating blocks (GT-free quality proxy).
    Floating blocks = voxels not attached to the largest 6-connected component;
    they come from token errors, so fewer = cleaner structure. Ties broken by
    most total voxels (prefer the more complete tower)."""
    best_i, best_key = 0, None
    for i, xyz in enumerate(sample_xyz):
        vset = coords_to_voxel_set(xyz)
        _, _, n_float = count_components_6conn(vset)
        key = (n_float, -len(vset))   # fewer floats, then more voxels
        if best_key is None or key < best_key:
            best_key, best_i = key, i
    return best_i


# ─────────────────────────── shared helpers ───────────────────────────

def set_iou(a, b):
    if not a and not b: return 1.0
    if not a or not b:  return 0.0
    return len(a & b) / len(a | b)


def zscore(x):
    x = np.asarray(x, float)
    s = x.std()
    return (x - x.mean()) / s if s > 1e-9 else np.zeros_like(x)


def _resolve_split(split, data_root):
    """Bare name -> data_root/<name>.txt; a path or a *.txt value is used as-is."""
    if split.endswith('.txt') or '/' in split:
        return split
    return pjoin(data_root, f'{split}.txt')


def _load_ctx(args, device):
    """Load vq + mask model + mean/std for one (mask, vq) pair."""
    vq_model, vq_opt = load_vq(CHECKPOINTS_DIR, args.vq_name, device=device,
                                dim_pose=3, dataset_name=DATASET_NAME)
    vq_model.eval().to(device)
    nb_code = vq_model.quantizer.layers[0].nb_code
    meta = pjoin(CHECKPOINTS_DIR, DATASET_NAME, args.vq_name, 'meta')
    mean = np.load(pjoin(meta, 'mean.npy')); std = np.load(pjoin(meta, 'std.npy'))
    model = load_mask_model(vq_opt, args.mask_name, args.ckpt, device)
    return vq_model, nb_code, mean, std, model


def _names_for_split(args):
    """Sorted, existing tower names from a split file, sharded names[k::M]."""
    data_root = args.data_root
    split_path = _resolve_split(args.split, data_root)
    motion_dir = pjoin(data_root, 'new_joint_vecs_int')
    feat_dir   = pjoin(data_root, 'image_features_dino_patch')
    with open(split_path) as f:
        names = sorted(l.strip() for l in f if l.strip())
    names = names[args.shard_idx::args.num_shards]
    names = [nm for nm in names
             if os.path.exists(pjoin(motion_dir, nm + '.npy'))
             and os.path.exists(pjoin(feat_dir, nm + '.npy'))]
    return names, motion_dir, feat_dir


# ─────────────────────────── mode: bestofn ───────────────────────────

def mode_bestofn(args, device):
    vq_model, nb_code, mean, std, model = _load_ctx(args, device)
    records = load_records(_resolve_split(args.split, args.data_root),
                           args.n_towers, args.seed, data_root=args.data_root)
    print(f'Loaded {len(records)} towers\n')

    single, mean_n, best_n, consensus, floating = [], [], [], [], []
    per_tower = []
    for name, motion, feat in tqdm(records, desc='towers'):
        ious_gt, sample_xyz = eval_tower(model, vq_model, mean, std, motion, feat,
                                         args.n_samples, args.temperature, device, nb_code)
        ci = consensus_pick(sample_xyz)
        fi = floating_pick(sample_xyz)
        single.append(ious_gt[0])               # first draw = single-pass baseline
        mean_n.append(ious_gt.mean())
        best_n.append(ious_gt.max())
        consensus.append(ious_gt[ci])
        floating.append(ious_gt[fi])
        per_tower.append((name, ious_gt.tolist(), int(ci), int(fi)))

    single, mean_n, best_n, consensus, floating = map(
        np.array, (single, mean_n, best_n, consensus, floating))
    orc = max(best_n.mean() - single.mean(), 1e-9)
    print('\n' + '=' * 68)
    print(f'  {args.mask_name}/{args.ckpt}  |  N={args.n_samples}  '
          f'temp={args.temperature}  n_towers={len(records)}')
    print('=' * 68)
    print(f'  single-pass (1 draw)       mean IoU = {single.mean():.4f}   median = {np.median(single):.4f}')
    print(f'  mean-of-N                  mean IoU = {mean_n.mean():.4f}   median = {np.median(mean_n):.4f}')
    print(f'  consensus  (GT-free pick)  mean IoU = {consensus.mean():.4f}   median = {np.median(consensus):.4f}')
    print(f'  fewest-floating (GT-free)  mean IoU = {floating.mean():.4f}   median = {np.median(floating):.4f}')
    print(f'  best-of-N  (ORACLE)        mean IoU = {best_n.mean():.4f}   median = {np.median(best_n):.4f}')
    print('-' * 68)
    print(f'  oracle uplift     best-of-N − single = {best_n.mean()-single.mean():+.4f}')
    print(f'  consensus uplift  consensus − single = {consensus.mean()-single.mean():+.4f}  '
          f'(recovers {(consensus.mean()-single.mean())/orc:.0%} of oracle)')
    print(f'  floating uplift   floating  − single = {floating.mean()-single.mean():+.4f}  '
          f'(recovers {(floating.mean()-single.mean())/orc:.0%} of oracle)')
    print('=' * 68)

    out = args.out_path or f'eval_bestofN_{args.mask_name}_N{args.n_samples}_t{args.temperature}.npy'
    np.save(out, np.array({'per_tower': per_tower, 'single': single.tolist(),
                           'mean_n': mean_n.tolist(), 'best_n': best_n.tolist(),
                           'consensus': consensus.tolist(), 'floating': floating.tolist(),
                           'n_samples': args.n_samples, 'temperature': args.temperature},
                          dtype=object))
    print(f'Saved {out}')


# ─────────────────────────── mode: sweep_n ───────────────────────────

def mode_sweep_n(args, device):
    vq_model, nb_code, mean, std, model = _load_ctx(args, device)
    records = load_records(_resolve_split(args.split, args.data_root),
                           args.n_towers, args.seed, data_root=args.data_root)
    print(f'Loaded {len(records)} towers  Ns={args.ns}\n')

    # representative tower for timing (median token length)
    tlens = [len(r[1]) // 4 for r in records]
    tref = records[int(np.argsort(tlens)[len(tlens) // 2])]
    ref_feat, ref_tok = tref[2], len(tref[1]) // 4

    def time_gen(n, reps=5, warmup=2):
        cond = torch.from_numpy(ref_feat).unsqueeze(0).repeat(n, 1, 1, 1).to(device).float()
        m_lens = torch.full((n,), ref_tok, device=device, dtype=torch.long)
        def once():
            with torch.no_grad():
                ids = model.generate(cond, m_lens, timesteps=MASK_TIMESTEPS,
                                     cond_scale=MASK_COND_SCALE, temperature=args.temperature,
                                     topk_filter_thres=MASK_TOPK, gsample=False)
                decode_windowed(vq_model, ids.clamp(0, nb_code - 1).unsqueeze(-1),
                                window_tokens=DEFAULT_WINDOW_TOKENS)
        for _ in range(warmup): once()
        torch.cuda.synchronize(); t0 = time.perf_counter()
        for _ in range(reps): once()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / reps * 1000  # ms

    rows = []
    for n in args.ns:
        single, cons, best = [], [], []
        spread, cons_rank, cons_is_best = [], [], []
        for name, motion, feat in tqdm(records, desc=f'N={n}', leave=False):
            ious_gt, sx = eval_tower(model, vq_model, mean, std, motion, feat,
                                     n, args.temperature, device, nb_code)
            ci = consensus_pick(sx)
            oi = int(ious_gt.argmax())
            single.append(ious_gt[0]); best.append(ious_gt.max()); cons.append(ious_gt[ci])
            spread.append(ious_gt.max() - ious_gt.min())
            order = np.argsort(-ious_gt)
            cons_rank.append(int(np.where(order == ci)[0][0]) + 1)
            cons_is_best.append(ci == oi)
        t_ms = time_gen(n)
        rows.append((n, np.mean(single), np.mean(cons), np.mean(best), t_ms,
                     np.mean(spread), np.mean(cons_rank), np.mean(cons_is_best)))
        print(f'  N={n:<4d} single={np.mean(single):.4f}  consensus={np.mean(cons):.4f}  '
              f'oracle={np.mean(best):.4f}  spread={np.mean(spread):.3f}  '
              f'cons_rank={np.mean(cons_rank):.1f}/{n}  cons=best {100*np.mean(cons_is_best):.0f}%  '
              f'gen+decode={t_ms:.0f}ms')

    print('\n' + '=' * 92)
    print(f'  {args.mask_name}/{args.ckpt}  temp={args.temperature}  n_towers={len(records)}  '
          f'(ref tower {ref_tok} tok)')
    print('=' * 92)
    print(f"  {'N':>4} | {'single':>8} {'consensus':>10} {'oracle':>8} | "
          f"{'spread':>7} {'consRank':>9} {'cons=best':>10} | {'gen+dec':>8} {'+DINO':>7}")
    print('  ' + '-' * 88)
    for n, s, c, b, t, sp, cr, cb in rows:
        print(f"  {n:>4} | {s:>8.4f} {c:>10.4f} {b:>8.4f} | {sp:>7.3f} {cr:>6.1f}/{n:<3d}"
              f" {100*cb:>8.0f}% | {t:>6.0f}ms {t+27:>5.0f}ms")
    print('=' * 92)


# ─────────────────────────── mode: sweep_thr ───────────────────────────

def mode_sweep_thr(args, device):
    vq_model, nb_code, mean, std, model = _load_ctx(args, device)
    records = load_records(_resolve_split(args.split, args.data_root),
                           args.n_towers, args.seed, data_root=args.data_root)
    print(f'Loaded {len(records)} towers  Ns={args.ns}  thresholds={args.thresholds}\n')

    gt_sets = {}
    for name, motion, _ in records:
        n_cmp = (len(motion) // 4) * 4
        gt_sets[name] = coords_to_voxel_set(motion[:n_cmp])

    table = {}
    for N in args.ns:
        single, oracle = [], []
        votes = {t: [] for t in args.thresholds}
        for name, motion, feat in tqdm(records, desc=f'N={N}', leave=False):
            ious_gt, sample_xyz = eval_tower(model, vq_model, mean, std, motion, feat,
                                             N, args.temperature, device, nb_code,
                                             decode_mode=args.decode_mode)
            single.append(ious_gt[0]); oracle.append(ious_gt.max())
            cnt = Counter()
            for xyz in sample_xyz:
                cnt.update(coords_to_voxel_set(xyz))
            gt_set = gt_sets[name]
            for t in args.thresholds:
                voted = {v for v, c in cnt.items() if c >= t * N}
                votes[t].append(set_iou(voted, gt_set))
        table[N] = dict(single=np.mean(single), oracle=np.mean(oracle),
                        votes={t: np.mean(votes[t]) for t in args.thresholds})
        best_t = max(args.thresholds, key=lambda t: table[N]['votes'][t])
        print(f'  N={N:<4d} single={table[N]["single"]:.4f}  oracle={table[N]["oracle"]:.4f}  '
              f'best vote@{best_t}={table[N]["votes"][best_t]:.4f}')

    print('\n' + '=' * (30 + 9 * len(args.thresholds)))
    print(f'  vote_output@threshold, {args.mask_name}/{args.ckpt}  temp={args.temperature}  n={len(records)}')
    print('=' * (30 + 9 * len(args.thresholds)))
    hdr = f"  {'N':>4} {'single':>8} {'oracle':>8} |" + ''.join(f'{t:>8}' for t in args.thresholds)
    print(hdr)
    print('  ' + '-' * (len(hdr) - 2))
    for N in args.ns:
        row = f"  {N:>4} {table[N]['single']:>8.4f} {table[N]['oracle']:>8.4f} |"
        best_t = max(args.thresholds, key=lambda t: table[N]['votes'][t])
        for t in args.thresholds:
            row += f"{table[N]['votes'][t]:>7.4f}{'*' if t == best_t else ' '}"
        print(row)
    print('=' * (30 + 9 * len(args.thresholds)))
    print('  * = best threshold for that N.  vote_output > oracle means voting beats best-single-sample.')


# ─────────────────────────── mode: selectors ───────────────────────────

def mode_selectors(args, device):
    vq_model, nb_code, mean, std, model = _load_ctx(args, device)
    records = load_records(_resolve_split(args.split, args.data_root),
                           args.n_towers, args.seed, data_root=args.data_root)
    print(f'Loaded {len(records)} towers\n')

    N = args.n_samples
    res = {k: [] for k in ['single', 'consensus_pair', 'fewest_floating', 'combined', 'oracle']}
    for t in args.vote_thresholds:
        res[f'vote_pick@{t}'] = []
        res[f'vote_output@{t}'] = []

    for name, motion, feat in tqdm(records, desc='towers'):
        ious_gt, sample_xyz = eval_tower(model, vq_model, mean, std, motion, feat,
                                         N, args.temperature, device, nb_code)
        n_cmp = (len(motion) // 4) * 4
        gt_set = coords_to_voxel_set(motion[:n_cmp])
        sets = [coords_to_voxel_set(xyz) for xyz in sample_xyz]
        sizes = np.array([len(s) for s in sets], float)

        pair = np.zeros((N, N))
        for i in range(N):
            for j in range(i + 1, N):
                v = set_iou(sets[i], sets[j])
                pair[i, j] = pair[j, i] = v
        mean_agree = pair.sum(1) / (N - 1)
        floats = np.array([count_components_6conn(s)[2] for s in sets], float)

        res['single'].append(ious_gt[0])
        res['oracle'].append(ious_gt.max())
        res['consensus_pair'].append(ious_gt[int(mean_agree.argmax())])
        res['fewest_floating'].append(ious_gt[int(np.lexsort((-sizes, floats))[0])])

        med = np.median(sizes)
        score = zscore(mean_agree) + zscore(-floats) + zscore(-np.abs(sizes - med))
        res['combined'].append(ious_gt[int(score.argmax())])

        cnt = Counter()
        for s in sets:
            cnt.update(s)
        for t in args.vote_thresholds:
            voted = {v for v, c in cnt.items() if c >= t * N}
            res[f'vote_output@{t}'].append(set_iou(voted, gt_set))
            sims = [set_iou(s, voted) for s in sets]
            res[f'vote_pick@{t}'].append(ious_gt[int(np.argmax(sims))])

    single_m = np.mean(res['single']); oracle_m = np.mean(res['oracle'])
    gap = max(oracle_m - single_m, 1e-9)
    order = ['single', 'consensus_pair', 'fewest_floating', 'combined'] + \
            [f'vote_pick@{t}' for t in args.vote_thresholds] + \
            [f'vote_output@{t}' for t in args.vote_thresholds] + ['oracle']
    print('\n' + '=' * 76)
    print(f'  {args.mask_name}/{args.ckpt}  N={N}  temp={args.temperature}  n={len(records)}')
    print('=' * 76)
    print(f"  {'selector':<20} {'mean IoU':>9} {'median':>8} {'Δ vs single':>12} {'%oracle':>9}")
    print('  ' + '-' * 72)
    for k in order:
        arr = np.array(res[k])
        d = arr.mean() - single_m
        pct = '' if k in ('single', 'oracle') else f'{d/gap:>8.0%}'
        tag = '  <- ceiling' if k == 'oracle' else ('  <- baseline' if k == 'single' else '')
        print(f"  {k:<20} {arr.mean():>9.4f} {np.median(arr):>8.4f} {d:>+12.4f} {pct:>9}{tag}")
    print('=' * 76)


# ─────────────────────────── mode: full ───────────────────────────

def mode_full(args, device):
    tag = (f'fulleval_{args.mask_name}_{args.ckpt.replace(".tar","")}_'
           f'{args.split}_N{args.n_samples}_{args.decode_mode}_thr{args.threshold}')
    if args.combine:
        parts = _shard_glob(tag, args.num_shards)
        assert parts, f'no shards found for {tag} (num_shards={args.num_shards})'
        single = oracle = vote = n = gen_t = 0.0
        for p in parts:
            d = np.load(p, allow_pickle=True).item()
            single += d['single_sum']; oracle += d['oracle_sum']
            vote += d['vote_sum']; n += d['n']; gen_t += d['gen_time_s']
        print('\n' + '=' * 64)
        print(f'  COMBINED  {args.mask_name}/{args.ckpt}  split={args.split}  '
              f'N={args.n_samples}  temp={args.temperature}')
        print(f'  towers={int(n)}  (shards={len(parts)})')
        print('=' * 64)
        print(f'  single-pass        mean IoU = {single/n:.4f}')
        print(f'  best-of-N oracle   mean IoU = {oracle/n:.4f}')
        print(f'  vote@{args.threshold}          mean IoU = {vote/n:.4f}')
        print('-' * 64)
        print(f'  vote − single = {(vote-single)/n:+.4f}   vote − oracle = {(vote-oracle)/n:+.4f}')
        print(f'  total gen wall-clock (summed over shards) = {gen_t:.1f}s')
        print(f'  per-tower gen (N={args.n_samples}) ~ {gen_t/n*1000:.0f} ms   '
              f'(serial est. {gen_t/3600:.2f} h for {int(n)} towers)')
        print('=' * 64)
        return

    _purge_stale_shards(tag, args.num_shards)
    vq_model, nb_code, mean, std, model = _load_ctx(args, device)
    names, motion_dir, feat_dir = _names_for_split(args)
    print(f'shard {args.shard_idx}/{args.num_shards}: {len(names)} towers  '
          f'split={args.split}  ckpt={args.ckpt}')

    thr = args.threshold * args.n_samples
    single_sum = oracle_sum = vote_sum = gen_time = 0.0
    for nm in tqdm(names, desc=f's{args.shard_idx}', leave=False):
        motion = np.load(pjoin(motion_dir, nm + '.npy'))
        feat   = np.load(pjoin(feat_dir, nm + '.npy'))
        torch.cuda.synchronize(); t0 = time.perf_counter()
        ious_gt, sample_xyz = eval_tower(model, vq_model, mean, std, motion, feat,
                                         args.n_samples, args.temperature, device, nb_code,
                                         decode_mode=args.decode_mode)
        torch.cuda.synchronize(); gen_time += time.perf_counter() - t0
        n_cmp = (len(motion) // 4) * 4
        gt_set = coords_to_voxel_set(motion[:n_cmp])
        cnt = Counter()
        for xyz in sample_xyz:
            cnt.update(coords_to_voxel_set(xyz))
        voted = {v for v, c in cnt.items() if c >= thr}
        single_sum += ious_gt[0]; oracle_sum += ious_gt.max()
        vote_sum   += set_iou(voted, gt_set)

    n = len(names)
    out = f'/tmp/{tag}_shard{args.shard_idx}of{args.num_shards}.npy'
    np.save(out, np.array({'single_sum': single_sum, 'oracle_sum': oracle_sum,
                           'vote_sum': vote_sum, 'n': n, 'gen_time_s': gen_time},
                          dtype=object))
    print(f'shard {args.shard_idx}: single={single_sum/n:.4f} oracle={oracle_sum/n:.4f} '
          f'vote={vote_sum/n:.4f}  gen={gen_time:.1f}s ({gen_time/n*1000:.0f}ms/tower)  -> {out}')


# ─────────────────────────── mode: sweep_cfg ───────────────────────────

def _parse_cond_scale(tok):
    """'2' -> 2.0 (flat CFG) | '1:3' -> (1.0, 3.0) linear schedule over decode steps."""
    if ':' in tok:
        a, b = tok.split(':')
        return (float(a), float(b))
    return float(tok)


def mode_sweep_cfg(args, device):
    """Classifier-free-guidance scale sweep. Per --cond_scales config, reports
    single-pass IoU (1 draw at --single_temp) plus first/oracle/vote@thr over
    --n_samples draws at --temperature. Configs mix flat scales ('2') and
    BAMM-style per-step schedules ('1:3' = scale 1 at the first decode step
    rising linearly to 3 at the last). cfg=1 is the no-CFG baseline."""
    vq_model, nb_code, mean, std, model = _load_ctx(args, device)
    records = load_records(_resolve_split(args.split, args.data_root),
                           args.n_towers, args.seed, data_root=args.data_root)
    print(f'Loaded {len(records)} towers  decode={args.decode_mode}  '
          f'single_temp={args.single_temp}  vote: N={args.n_samples} '
          f't={args.temperature} thr={args.threshold}\n')

    thr_count = args.threshold * args.n_samples
    results = {}
    for tok in args.cond_scales:
        scale = _parse_cond_scale(tok)
        single, first, oracle, vote = (np.zeros(len(records)) for _ in range(4))
        t0 = time.perf_counter()
        for i, (name, motion, feat) in enumerate(tqdm(records, desc=f'cfg {tok}', leave=False)):
            s_iou, _ = eval_tower(model, vq_model, mean, std, motion, feat,
                                  1, args.single_temp, device, nb_code,
                                  cond_scale=scale, decode_mode=args.decode_mode)
            ious_gt, sample_xyz = eval_tower(model, vq_model, mean, std, motion, feat,
                                             args.n_samples, args.temperature, device, nb_code,
                                             cond_scale=scale, decode_mode=args.decode_mode)
            n_cmp = (len(motion) // 4) * 4
            gt_set = coords_to_voxel_set(motion[:n_cmp])
            cnt = Counter()
            for xyz in sample_xyz:
                cnt.update(coords_to_voxel_set(xyz))
            voted = {v for v, c in cnt.items() if c >= thr_count}
            single[i] = s_iou[0]
            first[i]  = ious_gt[0]
            oracle[i] = ious_gt.max()
            vote[i]   = set_iou(voted, gt_set)
        results[tok] = dict(single=single.mean(), first=first.mean(),
                            oracle=oracle.mean(), vote=vote.mean(),
                            wall_s=time.perf_counter() - t0)
        r = results[tok]
        print(f'  cfg={tok:>6}  single(t{args.single_temp})={r["single"]:.4f}  '
              f'first(t{args.temperature})={r["first"]:.4f}  oracle={r["oracle"]:.4f}  '
              f'vote@{args.threshold}={r["vote"]:.4f}  [{r["wall_s"]:.0f}s]')

    print('\n' + '=' * 78)
    print(f'  CFG SWEEP  {args.mask_name}/{args.ckpt}  split={args.split}  '
          f'n_towers={len(records)}  decode={args.decode_mode}')
    print('=' * 78)
    print(f'  {"cfg":>8} | {"single t"+str(args.single_temp):>11} | '
          f'{"first t"+str(args.temperature):>10} | {"oracle":>8} | {"vote@"+str(args.threshold):>9}')
    print(f'  {"-"*8}-+-{"-"*11}-+-{"-"*10}-+-{"-"*8}-+-{"-"*9}')
    for tok, r in results.items():
        print(f'  {tok:>8} | {r["single"]:11.4f} | {r["first"]:10.4f} | '
              f'{r["oracle"]:8.4f} | {r["vote"]:9.4f}')
    print('=' * 78)

    out = args.out_path or (f'/tmp/sweep_cfg_{args.mask_name}_{args.ckpt.replace(".tar","")}'
                            f'_{args.split}_n{len(records)}.npy')
    np.save(out, np.array({'results': results, 'cond_scales': args.cond_scales,
                           'n_towers': len(records), 'n_samples': args.n_samples,
                           'temperature': args.temperature, 'single_temp': args.single_temp,
                           'threshold': args.threshold, 'decode_mode': args.decode_mode,
                           'seed': args.seed}, dtype=object))
    print(f'Saved {out}')


# ─────────────────────────── mode: errors ───────────────────────────

_VOX_NEIGHBORS = [(4, 0, 0), (-4, 0, 0), (0, 4, 0), (0, -4, 0), (0, 0, 8), (0, 0, -8)]


def mode_errors(args, device):
    """Error anatomy of the voted output. Per tower: vote@thr over N samples,
    then classify FN (GT blocks the vote missed) and FP (phantom blocks) by
    (a) pool coverage, how many of the N samples contain the voxel; FN with
    coverage 0 appear in NO reconstruction and are unrecoverable by voting,
    (b) relative height in the tower, (c) surface vs interior (FN) / GT-attached
    vs detached (FP), (d) relative position in the token sequence (FN)."""
    vq_model, nb_code, mean, std, model = _load_ctx(args, device)
    records = load_records(_resolve_split(args.split, args.data_root),
                           args.n_towers, args.seed, data_root=args.data_root)
    N = args.n_samples
    thr_count = args.threshold * N
    print(f'Loaded {len(records)} towers  N={N} t={args.temperature} '
          f'thr={args.threshold} (count>={thr_count:g})  decode={args.decode_mode}\n')

    fn_cov  = Counter()   # FN voxel -> appears in c of N samples (0..thr-1)
    fp_cov  = Counter()   # FP voxel -> c (thr..N)
    fn_z    = Counter(); fp_z = Counter(); gt_z = Counter()   # height deciles
    fn_nb   = Counter()   # FN: # of 6-neighbors present in GT (surface-ness)
    gt_nb   = Counter()   # same for all GT voxels (baseline)
    fn_seq  = Counter()   # FN: relative position in token sequence, deciles
    fp_att  = Counter()   # FP: 1 if 6-adjacent to a GT voxel else 0
    tot = dict(fn=0, fp=0, gt=0, fn_union=0, union_miss=0)
    per_tower = []

    for name, motion, feat in tqdm(records, desc='towers', leave=False):
        ious_gt, sample_xyz = eval_tower(model, vq_model, mean, std, motion, feat,
                                         N, args.temperature, device, nb_code,
                                         decode_mode=args.decode_mode)
        n_cmp = (len(motion) // 4) * 4
        gt_arr = grid_snap(motion[:n_cmp])
        gt_set = set(map(tuple, gt_arr.tolist()))
        seq_idx = {}                       # voxel -> first index in GT sequence
        for i, v in enumerate(map(tuple, gt_arr.tolist())):
            seq_idx.setdefault(v, i)

        cnt = Counter()
        for xyz in sample_xyz:
            cnt.update(coords_to_voxel_set(xyz))
        voted = {v for v, c in cnt.items() if c >= thr_count}
        fn = gt_set - voted
        fp = voted - gt_set

        zs = [v[2] for v in gt_set]
        z_lo, z_hi = min(zs), max(zs)
        z_rng = max(z_hi - z_lo, 1)
        zdec = lambda v: min(int(10 * (v[2] - z_lo) / z_rng), 9)
        nnb  = lambda v, s: sum((v[0]+dx, v[1]+dy, v[2]+dz) in s for dx, dy, dz in _VOX_NEIGHBORS)

        for v in gt_set:
            gt_z[zdec(v)] += 1
            gt_nb[nnb(v, gt_set)] += 1
        for v in fn:
            fn_cov[cnt.get(v, 0)] += 1
            fn_z[zdec(v)] += 1
            fn_nb[nnb(v, gt_set)] += 1
            fn_seq[min(int(10 * seq_idx[v] / n_cmp), 9)] += 1
        for v in fp:
            fp_cov[cnt[v]] += 1
            fp_z[zdec(v)] += 1
            fp_att[1 if nnb(v, gt_set) > 0 else 0] += 1

        union = set().union(*(coords_to_voxel_set(xyz) for xyz in sample_xyz))
        fn_in_union = sum(1 for v in fn if v in union)
        tot['fn'] += len(fn); tot['fp'] += len(fp); tot['gt'] += len(gt_set)
        tot['fn_union'] += fn_in_union
        tot['union_miss'] += len(gt_set - union)
        per_tower.append((name, set_iou(voted, gt_set), len(fn), len(fp),
                          fn_in_union, len(gt_set)))

    n_t = len(records)
    print('=' * 76)
    print(f'  VOTE ERROR ANATOMY  {args.mask_name}/{args.ckpt}  '
          f'N={N} t={args.temperature} vote@{args.threshold}  towers={n_t}')
    print('=' * 76)
    print(f'  totals: GT={tot["gt"]}  FN(missed)={tot["fn"]} ({tot["fn"]/n_t:.1f}/tower, '
          f'{tot["fn"]/tot["gt"]:.2%} of GT)  FP(phantom)={tot["fp"]} ({tot["fp"]/n_t:.1f}/tower)')
    fnu = tot['fn_union']
    print(f'\n  Q: do missed blocks appear in ANY of the {N} recons?')
    print(f'     {fnu}/{tot["fn"]} FN = {fnu/max(tot["fn"],1):.1%} appear in >=1 sample '
          f'(recoverable in principle); {tot["fn"]-fnu} ({1-fnu/max(tot["fn"],1):.1%}) in NONE')
    print(f'     GT voxels absent from the whole {N}-sample pool: {tot["union_miss"]} '
          f'({tot["union_miss"]/tot["gt"]:.3%} of GT)')
    print(f'\n  FN pool-coverage (c of {N} samples contain it; vote needs >={thr_count:g}):')
    for c in sorted(fn_cov):
        print(f'     c={c:>2}: {fn_cov[c]:>6}  ({fn_cov[c]/tot["fn"]:.1%})')
    print(f'  FP pool-coverage (phantoms; all >= thr by construction):')
    for c in sorted(fp_cov):
        print(f'     c={c:>2}: {fp_cov[c]:>6}  ({fp_cov[c]/max(tot["fp"],1):.1%})')
    hdr = '  '.join(f'{d/10:.1f}' for d in range(10))
    pct = lambda cn, tt: '  '.join(f'{cn[d]/max(tt,1):4.0%}' for d in range(10))
    print(f'\n  height decile (0=bottom, 0.9=top):   {hdr}')
    print(f'     GT distribution:                  {pct(gt_z, tot["gt"])}')
    print(f'     FN distribution:                  {pct(fn_z, tot["fn"])}')
    print(f'     FP distribution:                  {pct(fp_z, tot["fp"])}')
    print(f'\n  FN sequence-position decile:         {pct(fn_seq, tot["fn"])}')
    print(f'\n  surface-ness (# 6-neighbors in GT; 6=interior):')
    print(f'     GT: ' + '  '.join(f'{k}:{gt_nb[k]/tot["gt"]:.0%}' for k in range(7)))
    print(f'     FN: ' + '  '.join(f'{k}:{fn_nb[k]/max(tot["fn"],1):.0%}' for k in range(7)))
    att = fp_att[1] / max(tot['fp'], 1)
    print(f'\n  FP attachment: {att:.1%} touch a GT voxel (boundary growth); '
          f'{1-att:.1%} detached')
    worst = sorted(per_tower, key=lambda r: r[1])[:8]
    print(f'\n  worst towers (vote IoU | FN FP | FN-in-pool/FN | GT size):')
    for nm, iou, nfn, nfp, fnu_t, ngt in worst:
        print(f'     {nm}: {iou:.4f} | {nfn:>4} {nfp:>4} | {fnu_t}/{nfn} | {ngt}')
    print('=' * 76)

    out = args.out_path or (f'/tmp/errors_{args.mask_name}_{args.ckpt.replace(".tar","")}'
                            f'_n{n_t}.npy')
    np.save(out, np.array({'fn_cov': dict(fn_cov), 'fp_cov': dict(fp_cov),
                           'fn_z': dict(fn_z), 'fp_z': dict(fp_z), 'gt_z': dict(gt_z),
                           'fn_nb': dict(fn_nb), 'gt_nb': dict(gt_nb),
                           'fn_seq': dict(fn_seq), 'fp_att': dict(fp_att),
                           'tot': tot, 'per_tower': per_tower}, dtype=object))
    print(f'Saved {out}')


# ─────────────────────────── mode: decode_ab ───────────────────────────

@torch.no_grad()
def _decode_modes(vq_model, gen_ids, n_cmp, mean, std):
    """Decode the same tokens windowed AND whole-seq. Returns two lists of coords."""
    grid = gen_ids.unsqueeze(-1)
    coords_w = decode_windowed(vq_model, grid, window_tokens=DEFAULT_WINDOW_TOKENS).cpu().numpy()
    coords_f = vq_model.forward_decoder(grid).cpu().numpy()
    N = gen_ids.shape[0]
    xyz_w = [coords_w[i, :n_cmp] * std + mean for i in range(N)]
    xyz_f = [coords_f[i, :n_cmp] * std + mean for i in range(N)]
    return xyz_w, xyz_f


def _ab_score(sample_xyz, gt_xyz, gt_set, thr_count):
    ious = np.array([voxel_iou(xyz, gt_xyz) for xyz in sample_xyz])
    cnt = Counter()
    for xyz in sample_xyz:
        cnt.update(coords_to_voxel_set(xyz))
    voted = {v for v, c in cnt.items() if c >= thr_count}
    return ious[0], ious.max(), set_iou(voted, gt_set)


def mode_decode_ab(args, device):
    tag = (f'abeval_{args.mask_name}_{args.ckpt.replace(".tar","")}_'
           f'{args.split}_N{args.n_samples}')
    if args.combine:
        parts = _shard_glob(tag, args.num_shards)
        assert parts, f'no shards found for {tag} (num_shards={args.num_shards})'
        acc = {k: 0.0 for k in ('w_single', 'w_oracle', 'w_vote',
                                'f_single', 'f_oracle', 'f_vote', 'n', 'gen_t')}
        for p in parts:
            d = np.load(p, allow_pickle=True).item()
            for k in acc: acc[k] += d[k]
        n = acc['n']
        print('\n' + '=' * 70)
        print(f'  COMBINED  {args.mask_name}/{args.ckpt}  split={args.split}  '
              f'N={args.n_samples}  temp={args.temperature}  thr={args.threshold}')
        print(f'  towers={int(n)}  shards={len(parts)}')
        print('=' * 70)
        print(f'  {"decode":>10} | {"single":>8} | {"oracle":>8} | {"vote@"+str(args.threshold):>9}')
        print(f'  {"-"*10}-+-{"-"*8}-+-{"-"*8}-+-{"-"*9}')
        print(f'  {"windowed":>10} | {acc["w_single"]/n:8.4f} | {acc["w_oracle"]/n:8.4f} | {acc["w_vote"]/n:9.4f}')
        print(f'  {"whole-seq":>10} | {acc["f_single"]/n:8.4f} | {acc["f_oracle"]/n:8.4f} | {acc["f_vote"]/n:9.4f}')
        print('-' * 70)
        print(f'  whole−windowed single = {(acc["f_single"]-acc["w_single"])/n:+.4f}   '
              f'vote = {(acc["f_vote"]-acc["w_vote"])/n:+.4f}')
        best_single = max(acc['w_single'], acc['f_single']) / n
        best_vote = max(acc['w_vote'], acc['f_vote']) / n
        print(f'  best single = {best_single:.4f}   best vote = {best_vote:.4f}   '
              f'vote uplift = {best_vote-best_single:+.4f}')
        print(f'  total gen wall-clock = {acc["gen_t"]:.1f}s  '
              f'(~{acc["gen_t"]/n*1000:.0f} ms/tower for N={args.n_samples})')
        print('=' * 70)
        return

    _purge_stale_shards(tag, args.num_shards)
    vq_model, nb_code, mean, std, model = _load_ctx(args, device)
    names, motion_dir, feat_dir = _names_for_split(args)
    if args.n_towers > 0:
        names = names[:args.n_towers]
    print(f'shard {args.shard_idx}/{args.num_shards}: {len(names)} towers  '
          f'split={args.split}  ckpt={args.ckpt}  nb_code={nb_code}')

    thr_count = args.threshold * args.n_samples
    acc = {k: 0.0 for k in ('w_single', 'w_oracle', 'w_vote',
                            'f_single', 'f_oracle', 'f_vote')}
    gen_t = 0.0
    for nm in tqdm(names, desc=f's{args.shard_idx}', leave=False):
        motion = np.load(pjoin(motion_dir, nm + '.npy'))
        feat   = np.load(pjoin(feat_dir, nm + '.npy'))
        m_len = len(motion); n_cmp = (m_len // 4) * 4
        cond = torch.from_numpy(feat).unsqueeze(0).repeat(
            args.n_samples, *([1] * feat.ndim)).to(device).float()
        m_lens_tok = torch.full((args.n_samples,), m_len // 4, device=device, dtype=torch.long)
        torch.cuda.synchronize(); t0 = time.perf_counter()
        with torch.no_grad():
            gen_ids = model.generate(cond, m_lens_tok, timesteps=MASK_TIMESTEPS,
                                     cond_scale=MASK_COND_SCALE, temperature=args.temperature,
                                     topk_filter_thres=MASK_TOPK, gsample=False)
        gen_ids = gen_ids.clamp(0, nb_code - 1)
        torch.cuda.synchronize(); gen_t += time.perf_counter() - t0
        xyz_w, xyz_f = _decode_modes(vq_model, gen_ids, n_cmp, mean, std)
        gt_xyz = motion[:n_cmp]; gt_set = coords_to_voxel_set(gt_xyz)
        ws, wo, wv = _ab_score(xyz_w, gt_xyz, gt_set, thr_count)
        fs, fo, fv = _ab_score(xyz_f, gt_xyz, gt_set, thr_count)
        acc['w_single'] += ws; acc['w_oracle'] += wo; acc['w_vote'] += wv
        acc['f_single'] += fs; acc['f_oracle'] += fo; acc['f_vote'] += fv

    acc['n'] = len(names); acc['gen_t'] = gen_t
    out = f'/tmp/{tag}_shard{args.shard_idx}of{args.num_shards}.npy'
    np.save(out, np.array(acc, dtype=object))
    n = acc['n']
    print(f'shard {args.shard_idx}: win[s={acc["w_single"]/n:.4f} o={acc["w_oracle"]/n:.4f} '
          f'v={acc["w_vote"]/n:.4f}]  whole[s={acc["f_single"]/n:.4f} o={acc["f_oracle"]/n:.4f} '
          f'v={acc["f_vote"]/n:.4f}]  -> {out}')


@torch.no_grad()
def mode_len_sens(args, device):
    """How much IoU does an error in the token-length input cost? For each tower,
    generate at GT length + each --len_offsets value and score vs full GT:
    one single-pass draw (temp --single_temp) and an N-sample vote (temp
    --temperature, threshold --threshold) per offset. Samples are decoded
    individually at their own length so a short sample's tail is never
    contaminated by pad tokens inside the conv decoder's receptive field."""
    vq_model, nb_code, mean, std, model = _load_ctx(args, device)
    records = load_records(_resolve_split(args.split, args.data_root),
                           args.n_towers, args.seed, data_root=args.data_root)
    offsets, N = args.len_offsets, args.n_samples
    assert 0 in offsets, '--len_offsets must include 0 (the baseline)'
    K = len(offsets)
    thr_count = args.threshold * N
    print(f'Loaded {len(records)} towers  offsets={offsets}  '
          f'single t={args.single_temp}  vote N={N} t={args.temperature} '
          f'thr={args.threshold}  decode={args.decode_mode}\n')

    def _decode_one_len(ids, L):
        """ids (B, max_len) all with true length L -> coords (B, 4L, 3) denorm."""
        ids = ids[:, :L].unsqueeze(-1)
        if args.decode_mode == 'whole':
            coords = vq_model.forward_decoder(ids).cpu().numpy()
        else:
            coords = decode_windowed(vq_model, ids,
                                     window_tokens=DEFAULT_WINDOW_TOKENS).cpu().numpy()
        return coords * std + mean

    single = {o: [] for o in offsets}
    voted  = {o: [] for o in offsets}
    base_lens = []
    for name, motion, feat in tqdm(records, desc='towers', leave=False):
        n_cmp = (len(motion) // 4) * 4
        gt_xyz = motion[:n_cmp]
        gt_set = coords_to_voxel_set(gt_xyz)
        base = len(motion) // 4
        base_lens.append(base)
        lens = [int(np.clip(base + o, 1, 512)) for o in offsets]

        cond1 = torch.from_numpy(feat).unsqueeze(0).repeat(
            K, *([1] * feat.ndim)).to(device).float()
        m_lens1 = torch.tensor(lens, device=device, dtype=torch.long)
        ids1 = model.generate(cond1, m_lens1, timesteps=MASK_TIMESTEPS,
                              cond_scale=MASK_COND_SCALE, temperature=args.single_temp,
                              topk_filter_thres=MASK_TOPK, gsample=False)
        ids1 = ids1.clamp(0, nb_code - 1)

        condN = torch.from_numpy(feat).unsqueeze(0).repeat(
            K * N, *([1] * feat.ndim)).to(device).float()
        m_lensN = m_lens1.repeat_interleave(N)
        idsN = model.generate(condN, m_lensN, timesteps=MASK_TIMESTEPS,
                              cond_scale=MASK_COND_SCALE, temperature=args.temperature,
                              topk_filter_thres=MASK_TOPK, gsample=False)
        idsN = idsN.clamp(0, nb_code - 1)

        for k, o in enumerate(offsets):
            L = lens[k]
            xyz1 = _decode_one_len(ids1[k:k + 1], L)
            single[o].append(voxel_iou(xyz1[0], gt_xyz))
            xyzN = _decode_one_len(idsN[k * N:(k + 1) * N], L)
            cnt = Counter()
            for i in range(N):
                cnt.update(coords_to_voxel_set(xyzN[i]))
            voted_set = {v for v, c in cnt.items() if c >= thr_count}
            voted[o].append(set_iou(voted_set, gt_set))

    base_lens = np.array(base_lens)
    print('=' * 78)
    print(f'  LENGTH SENSITIVITY  {args.mask_name}/{args.ckpt}  towers={len(records)}  '
          f'decode={args.decode_mode}')
    print(f'  token length: mean={base_lens.mean():.0f}  min={base_lens.min()}  '
          f'max={base_lens.max()}  (1 token = 4 blocks)')
    print('=' * 78)
    s0, v0 = np.mean(single[0]), np.mean(voted[0])
    print(f'  {"offset":>8} {"rel err":>8} {"single":>8} {"Δsingle":>9} '
          f'{"vote@" + format(args.threshold, "g"):>10} {"Δvote":>9}')
    for o in offsets:
        s, v = np.mean(single[o]), np.mean(voted[o])
        print(f'  {o:>+8d} {100 * o / base_lens.mean():>+7.1f}% {s:>8.4f} {s - s0:>+9.4f} '
              f'{v:>10.4f} {v - v0:>+9.4f}')
    print('=' * 78)
    if args.out_path:
        np.save(args.out_path, np.array({'offsets': offsets,
                                         'single': {o: single[o] for o in offsets},
                                         'voted': {o: voted[o] for o in offsets},
                                         'base_lens': base_lens}, dtype=object))
        print(f'Saved {args.out_path}')


@torch.no_grad()
def mode_len_jitter(args, device):
    """Can voting absorb a length-prediction error if the N samples get
    DIFFERENT lengths spread around the (possibly wrong) center, instead of
    all sharing it? Centers: GT+offset for each --len_offsets, plus the real
    grid length-predictor when --len_pred_ckpt is given. Spreads: J0 (all at
    center, = len_sens baseline) and, for each J in --jitters, 'sym'
    (linspace center-J..center+J) and 'up' (center-J/2..center+3J/2, biased
    long, since the vote trims over-built tails but cannot invent blocks
    beyond the longest samples). Each sample decoded at its own length."""
    vq_model, nb_code, mean, std, model = _load_ctx(args, device)
    records = load_records(_resolve_split(args.split, args.data_root),
                           args.n_towers, args.seed, data_root=args.data_root)
    N = args.n_samples
    thr_count = args.threshold * N

    pred_model = None
    if args.len_pred_ckpt:
        import train_length_predictor as lp
        pred_model, pred_ck = lp.load_length_predictor(args.len_pred_ckpt, device)
        print(f'Loaded length predictor {args.len_pred_ckpt} (val MAE {pred_ck["val_mae"]:.2f})')

    configs = [('J0', 0, 0.0)]
    for J in args.jitters:
        configs += [(f'J{J}sym', J, 0.0), (f'J{J}up', J, 0.5)]

    def jitter_lens(center, J, shift):
        if J == 0:
            lens = np.full(N, center, dtype=float)
        else:
            lens = center + np.linspace(-J, J, N) + 2 * J * shift
        return np.clip(np.round(lens).astype(int), 1, 512)

    center_tags = [f'gt{o:+d}' for o in args.len_offsets] + (['pred'] if pred_model else [])
    results = {t: {c[0]: [] for c in configs} for t in center_tags}
    pred_errs = []
    print(f'Loaded {len(records)} towers  centers={center_tags}  '
          f'configs={[c[0] for c in configs]}  N={N} t={args.temperature} '
          f'thr={args.threshold}  decode={args.decode_mode}\n')

    for name, motion, feat in tqdm(records, desc='towers', leave=False):
        n_cmp = (len(motion) // 4) * 4
        gt_xyz = motion[:n_cmp]
        gt_set = coords_to_voxel_set(gt_xyz)
        base = len(motion) // 4
        centers = {f'gt{o:+d}': base + o for o in args.len_offsets}
        if pred_model:
            import train_length_predictor as lp
            p = lp.predict_tokens(pred_model, pred_ck, feat, device)
            centers['pred'] = p
            pred_errs.append(p - base)

        cond = torch.from_numpy(feat).unsqueeze(0).repeat(
            N, *([1] * feat.ndim)).to(device).float()
        for tag, c in centers.items():
            for cfg_name, J, shift in configs:
                lens = jitter_lens(c, J, shift)
                m_lens = torch.from_numpy(lens).to(device)
                ids = model.generate(cond, m_lens, timesteps=MASK_TIMESTEPS,
                                     cond_scale=MASK_COND_SCALE,
                                     temperature=args.temperature,
                                     topk_filter_thres=MASK_TOPK, gsample=False)
                ids = ids.clamp(0, nb_code - 1)
                cnt = Counter()
                for L in np.unique(lens):
                    rows = np.where(lens == L)[0]
                    sub = ids[torch.from_numpy(rows).to(device), :L].unsqueeze(-1)
                    if args.decode_mode == 'whole':
                        coords = vq_model.forward_decoder(sub).cpu().numpy()
                    else:
                        coords = decode_windowed(vq_model, sub,
                                                 window_tokens=DEFAULT_WINDOW_TOKENS).cpu().numpy()
                    coords = coords * std + mean
                    for i in range(len(rows)):
                        cnt.update(coords_to_voxel_set(coords[i]))
                voted = {v for v, k in cnt.items() if k >= thr_count}
                results[tag][cfg_name].append(set_iou(voted, gt_set))

    print('=' * 90)
    print(f'  LENGTH-JITTER VOTING  {args.mask_name}/{args.ckpt}  towers={len(records)}  '
          f'N={N} t={args.temperature} thr={args.threshold}  decode={args.decode_mode}')
    if pred_errs:
        pe = np.array(pred_errs)
        print(f'  predictor on this subset: MAE {np.abs(pe).mean():.2f} tok  '
              f'bias {pe.mean():+.2f}  P<=2 {np.mean(np.abs(pe) <= 2):.1%}  '
              f'P<=8 {np.mean(np.abs(pe) <= 8):.1%}')
    print('=' * 90)
    hdr = '  {:>8}'.format('center') + ''.join(f' {c[0]:>9}' for c in configs)
    print(hdr)
    for tag in center_tags:
        row = f'  {tag:>8}'
        for cfg_name, _, _ in configs:
            row += f' {np.mean(results[tag][cfg_name]):>9.4f}'
        print(row)
    print('=' * 90)
    if args.out_path:
        np.save(args.out_path, np.array({'results': results, 'pred_errs': pred_errs,
                                         'configs': [c[0] for c in configs]}, dtype=object))
        print(f'Saved {args.out_path}')


MODES = {'bestofn': mode_bestofn, 'sweep_n': mode_sweep_n, 'sweep_thr': mode_sweep_thr,
         'selectors': mode_selectors, 'full': mode_full, 'decode_ab': mode_decode_ab,
         'sweep_cfg': mode_sweep_cfg, 'errors': mode_errors, 'len_sens': mode_len_sens,
         'len_jitter': mode_len_jitter}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--mode', default='bestofn', choices=list(MODES))
    ap.add_argument('--mask_name', default='myarch_mask_img_patch_v7')
    ap.add_argument('--ckpt', default='net_best_iou.tar')
    ap.add_argument('--vq_name', default='rvq_v2_4x_nc1024_nq3')
    ap.add_argument('--data_root', default=DATA_ROOT)
    ap.add_argument('--split', default='val40k', help='bare name (val40k/train160k) or a .txt path')
    ap.add_argument('--n_samples', type=int, default=16)
    ap.add_argument('--ns', type=int, nargs='+', default=[8, 16, 32, 64, 128],
                    help='sweep_n / sweep_thr: N values')
    ap.add_argument('--n_towers', type=int, default=200,
                    help='cap for shuffled-subset modes; for decode_ab 0=all in shard')
    ap.add_argument('--temperature', type=float, default=None,
                    help='default 0.2 for bestofn, 0.8 for the voting/sweep modes')
    ap.add_argument('--threshold', type=float, default=0.3, help='full / decode_ab vote threshold')
    ap.add_argument('--thresholds', type=float, nargs='+',
                    default=[0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45], help='sweep_thr columns')
    ap.add_argument('--vote_thresholds', type=float, nargs='+', default=[0.3, 0.4, 0.5, 0.6],
                    help='selectors vote thresholds')
    ap.add_argument('--cond_scales', nargs='+', default=['1', '2', '3', '4'],
                    help="sweep_cfg: flat scales ('2') and/or start:end schedules ('1:3')")
    ap.add_argument('--single_temp', type=float, default=0.2,
                    help='sweep_cfg / len_sens: temperature for the 1-draw single-pass measurement')
    ap.add_argument('--len_offsets', type=int, nargs='+',
                    default=[-8, -4, -2, -1, 0, 1, 2, 4, 8],
                    help='len_sens / len_jitter: token-length offsets to test '
                         '(len_sens: must include 0 = baseline)')
    ap.add_argument('--jitters', type=int, nargs='+', default=[4, 8, 16],
                    help='len_jitter: half-width J of the per-sample length spread')
    ap.add_argument('--len_pred_ckpt', default=None,
                    help='len_jitter: grid length-predictor checkpoint; adds a "pred" center row')
    ap.add_argument('--decode_mode', choices=['windowed', 'whole'], default='windowed',
                    help='sweep_cfg / sweep_thr / full: token decode protocol (whole for IBQ)')
    ap.add_argument('--shard_idx', type=int, default=0)
    ap.add_argument('--num_shards', type=int, default=1)
    ap.add_argument('--combine', action='store_true', help='full / decode_ab: assemble shards')
    ap.add_argument('--gpu_id', type=int, default=0)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out_path', default=None)
    args = ap.parse_args()

    if args.temperature is None:
        args.temperature = 0.2 if args.mode == 'bestofn' else 0.8

    fixseed(args.seed)
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}  mode={args.mode}  N={args.n_samples}  temp={args.temperature}')
    MODES[args.mode](args, device)


if __name__ == '__main__':
    main()
