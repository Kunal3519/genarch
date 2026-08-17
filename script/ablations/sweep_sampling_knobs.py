"""
Cheap inference-knob sweeps for the mask transformer, on a val shard.

Two sweeps (never explored before):
  --sweep timesteps : number of iterative decode steps  {8,10,15,20,30}
  --sweep topk_topp : top-k fraction threshold x top-p nucleus grid

For every config we report the deployment metrics on the SAME tower subset:
  single  = 1 draw at --single_temp (0.2), whole-seq decode, IoU vs GT
  vote    = N draws at --temperature (0.8), whole-seq decode, vote@--threshold, IoU

Sharded across GPUs: launch one process per GPU with --shard_idx k --num_shards M
(all shards evaluate a disjoint slice of the SAME fixed n_towers subset), then
run once with --combine to assemble the table.

  # per-GPU (4 free GPUs), e.g. timesteps sweep:
  CUDA_VISIBLE_DEVICES=<uuid> uv run python script/sweep_sampling_knobs.py \
      --sweep timesteps --shard_idx k --num_shards 4 --n_towers 400
  # then:
  uv run python script/sweep_sampling_knobs.py --sweep timesteps --num_shards 4 --combine
"""
import os, sys, time, argparse, glob
from os.path import join as pjoin
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))  # repo root
sys.path.insert(0, os.path.dirname(_HERE))                   # script/
from utils.paths import EVAL_RESULTS

import numpy as np
import torch
from collections import Counter
from tqdm import tqdm

from utils.model_io import load_vq
from utils.voxel import voxel_iou, coords_to_voxel_set
from utils.vq_decode import decode_windowed, DEFAULT_WINDOW_TOKENS
from script.eval_sampling import (load_mask_model, _resolve_split,
                                  set_iou, CHECKPOINTS_DIR, DATASET_NAME, DATA_ROOT)

MASK_COND_SCALE = 1.0
OUT_DIR = EVAL_RESULTS


def parse_cfgs(sweep):
    """Return list of dicts: {label, timesteps, topk, topp, schedule}."""
    if sweep == 'timesteps':
        return [dict(label=f'ts{ts}', timesteps=ts, topk=0.9, topp=None, schedule=None)
                for ts in (8, 10, 15, 20, 30)]
    if sweep == 'topk_topp':
        cfgs = []
        # top-k only (fraction thres; 0.0 keeps all, higher keeps fewer)
        for tk in (0.0, 0.8, 0.9, 0.95, 0.98):
            cfgs.append(dict(label=f'tk{tk}', timesteps=10, topk=tk, topp=None, schedule=None))
        # top-p only (no top-k), nucleus
        for tp in (0.8, 0.9, 0.95, 0.99):
            cfgs.append(dict(label=f'tp{tp}', timesteps=10, topk=0.0, topp=tp, schedule=None))
        return cfgs
    if sweep == 'schedule':
        # inference unmasking-cadence shape (no retrain); cosine = current default
        return [dict(label=s, timesteps=10, topk=0.9, topp=None, schedule=s)
                for s in ('cosine', 'linear', 'square', 'cubic', 'sqrt')]
    raise ValueError(sweep)


def gen_samples(model, vq_model, mean, std, motion, feat, n_samples, temperature,
                cfg, device, nb_code):
    """N samples for one tower -> list of decoded (snapped-later) xyz arrays."""
    m_len = len(motion)
    n_cmp = (m_len // 4) * 4
    cond = torch.from_numpy(feat).unsqueeze(0).repeat(
        n_samples, *([1] * feat.ndim)).to(device).float()
    m_lens_tok = torch.full((n_samples,), m_len // 4, device=device, dtype=torch.long)
    gen_ids = model.generate(
        cond, m_lens_tok, timesteps=cfg['timesteps'], cond_scale=MASK_COND_SCALE,
        temperature=temperature, topk_filter_thres=cfg['topk'],
        topp_filter_thres=cfg['topp'], gsample=False,
        gen_schedule=cfg.get('schedule'))
    gen_ids = gen_ids.clamp(0, nb_code - 1)
    coords = vq_model.forward_decoder(gen_ids.unsqueeze(-1)).cpu().numpy()  # whole-seq (IBQ)
    return [coords[i, :n_cmp] * std + mean for i in range(n_samples)], n_cmp


def names_for(args):
    split_path = _resolve_split(args.split, args.data_root)
    motion_dir = pjoin(args.data_root, 'new_joint_vecs_int')
    feat_dir   = pjoin(args.data_root, 'image_features_dino_patch')
    with open(split_path) as f:
        names = sorted(l.strip() for l in f if l.strip())
    names = [nm for nm in names
             if os.path.exists(pjoin(motion_dir, nm + '.npy'))
             and os.path.exists(pjoin(feat_dir, nm + '.npy'))]
    rng = np.random.default_rng(args.seed)
    rng.shuffle(names)
    names = names[:args.n_towers]                 # fixed subset (same across shards)
    names = sorted(names)
    names = names[args.shard_idx::args.num_shards]  # disjoint slice for this shard
    return names, motion_dir, feat_dir


def tag_for(args):
    return f'knobsweep_{args.sweep}_{args.mask_name}_{args.ckpt.replace(".tar","")}_{args.split}_n{args.n_towers}'


def combine(args):
    tag = tag_for(args)
    parts = sorted(glob.glob(pjoin(OUT_DIR, f'{tag}_shard*of{args.num_shards}.npy')))
    assert parts, f'no shards for {tag}'
    agg = {}
    n_total = 0
    for p in parts:
        d = np.load(p, allow_pickle=True).item()
        n_total += d['n']
        for lab, v in d['cfgs'].items():
            a = agg.setdefault(lab, dict(single=0.0, vote=0.0, oracle=0.0, refined=0.0, gen_s=0.0))
            a['single'] += v['single']; a['vote'] += v['vote']
            a['oracle'] += v['oracle']; a['gen_s'] += v['gen_s']
            a['refined'] += v.get('refined', 0.0)
    print('\n' + '=' * 72)
    print(f'  KNOB SWEEP [{args.sweep}]  {args.mask_name}/{args.ckpt}  '
          f'split={args.split}  towers={n_total}  (shards={len(parts)})')
    print(f'  single t={args.single_temp} | vote N={args.n_samples} t={args.temperature} '
          f'@{args.threshold} | whole-seq decode')
    print('=' * 72)
    print(f'  {"config":<10} {"single":>8} {"vote":>8} {"refined":>9} {"oracle":>8} {"gen ms/tow":>11}')
    print('-' * 72)
    order = [c['label'] for c in parse_cfgs(args.sweep)]
    base_single = base_vote = base_ref = None
    for lab in order:
        a = agg[lab]
        s, v, o = a['single']/n_total, a['vote']/n_total, a['oracle']/n_total
        rf = a['refined']/n_total
        ms = a['gen_s']/n_total*1000
        ds = dv = ''
        if base_single is None:
            base_single, base_vote, base_ref = s, v, rf
        else:
            ds = f'({s-base_single:+.4f})'; dv = f'({rf-base_ref:+.4f})'
        print(f'  {lab:<10} {s:>8.4f} {v:>8.4f} {rf:>9.4f} {o:>8.4f} {ms:>11.0f}   {ds} {dv}')
    print('=' * 72)
    print('  Δ shown vs the first row (deployment default ts10 / tk0.9).')


def load_refiner(cls_name, device):
    """The deployed refinement model (pool classifier C). Returns a callable
    (samples, gt_set) -> refined voxel set, or None if --cls_name ''."""
    if not cls_name:
        return None
    from script.eval_pipeline import build_pool
    from script.train_refinement import tower_features, N_FEAT
    from models.refinement import RefinementNetwork
    ck = torch.load(pjoin(CHECKPOINTS_DIR, DATASET_NAME, 'pool_classifier', cls_name, 'best.tar'),
                    map_location='cpu', weights_only=False)
    ca = ck['args']
    net = RefinementNetwork(N_FEAT, ca['latent_dim'], ca['ff_size'], ca['c_layers'],
                          ca['n_heads'], ca['dropout']).to(device)
    net.load_state_dict(ck['model']); net.eval()

    def refine(samples, gt_set, feat, n_samples, mean3, std3, tau):
        vox, count, _lab, _n_gt = build_pool(samples, gt_set, 1)
        cf = tower_features(vox, count, n_samples, mean3, std3)
        vf = torch.from_numpy(cf).unsqueeze(0).to(device)
        pos = vf[..., 8:11].clone()
        kpad = torch.zeros(1, len(vox), dtype=torch.bool, device=device)
        img = torch.from_numpy(feat).unsqueeze(0).to(device).float()
        with torch.no_grad():
            p = torch.sigmoid(net(vf, pos, img, kpad)).squeeze(0).cpu().numpy()
        return set(map(tuple, vox[p >= tau].astype(int).tolist()))
    return refine


def run_shard(args, device):
    cfgs = parse_cfgs(args.sweep)
    vq_model, vq_opt = load_vq(CHECKPOINTS_DIR, args.vq_name, device=device,
                                dim_pose=3, dataset_name=DATASET_NAME)
    vq_model.eval().to(device)
    nb_code = vq_model.quantizer.layers[0].nb_code
    meta = pjoin(CHECKPOINTS_DIR, DATASET_NAME, args.vq_name, 'meta')
    mean = np.load(pjoin(meta, 'mean.npy')); std = np.load(pjoin(meta, 'std.npy'))
    model = load_mask_model(vq_opt, args.mask_name, args.ckpt, device)
    refine = load_refiner(args.cls_name, device)
    mean3 = mean.reshape(-1)[:3]; std3 = std.reshape(-1)[:3]

    names, motion_dir, feat_dir = names_for(args)
    print(f'shard {args.shard_idx}/{args.num_shards}: {len(names)} towers, '
          f'{len(cfgs)} configs', flush=True)

    res = {c['label']: dict(single=0.0, vote=0.0, oracle=0.0, refined=0.0, gen_s=0.0) for c in cfgs}
    thr = args.threshold * args.n_samples
    for nm in tqdm(names, desc=f's{args.shard_idx}', leave=False):
        motion = np.load(pjoin(motion_dir, nm + '.npy'))
        feat   = np.load(pjoin(feat_dir, nm + '.npy'))
        n_cmp = (len(motion) // 4) * 4
        gt_set = coords_to_voxel_set(motion[:n_cmp])
        for c in cfgs:
            r = res[c['label']]
            # vote: N draws at t=temperature
            torch.cuda.synchronize(); t0 = time.perf_counter()
            samples, _ = gen_samples(model, vq_model, mean, std, motion, feat,
                                     args.n_samples, args.temperature, c, device, nb_code)
            torch.cuda.synchronize(); r['gen_s'] += time.perf_counter() - t0
            ious = np.array([voxel_iou(xyz, motion[:n_cmp]) for xyz in samples])
            cnt = Counter()
            for xyz in samples:
                cnt.update(coords_to_voxel_set(xyz))
            voted = {v for v, cc in cnt.items() if cc >= thr}
            r['vote'] += set_iou(voted, gt_set); r['oracle'] += ious.max()
            if refine is not None:
                r['refined'] += set_iou(refine(samples, gt_set, feat, args.n_samples,
                                               mean3, std3, args.tau), gt_set)
            # single: 1 draw at single_temp
            s1, _ = gen_samples(model, vq_model, mean, std, motion, feat,
                                1, args.single_temp, c, device, nb_code)
            r['single'] += voxel_iou(s1[0], motion[:n_cmp])

    out = pjoin(OUT_DIR, f'{tag_for(args)}_shard{args.shard_idx}of{args.num_shards}.npy')
    np.save(out, np.array({'n': len(names), 'cfgs': res}, dtype=object))
    print(f'shard {args.shard_idx} done -> {out}', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sweep', choices=['timesteps', 'topk_topp', 'schedule'], required=True)
    ap.add_argument('--mask_name', default='myarch_mask_img_patch_v8_2')
    ap.add_argument('--ckpt', default='net_best_iou.tar')
    ap.add_argument('--vq_name', default='ibq_4x_nc1024_nq1')
    ap.add_argument('--data_root', default=DATA_ROOT)
    ap.add_argument('--split', default='val40k')
    ap.add_argument('--n_towers', type=int, default=400)
    ap.add_argument('--n_samples', type=int, default=16)
    ap.add_argument('--temperature', type=float, default=0.8)
    ap.add_argument('--single_temp', type=float, default=0.2)
    ap.add_argument('--threshold', type=float, default=0.4375)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--shard_idx', type=int, default=0)
    ap.add_argument('--num_shards', type=int, default=1)
    ap.add_argument('--cls_name', default='poolcls_c_sym_v1',
                    help="refinement model; '' disables the refined column")
    ap.add_argument('--tau', type=float, default=0.45)
    ap.add_argument('--combine', action='store_true')
    args = ap.parse_args()

    if args.combine:
        combine(args); return
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with torch.no_grad():
        run_shard(args, device)


if __name__ == '__main__':
    main()
