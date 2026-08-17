"""Evaluate the full deployed GenArch pipeline, per tower.

    4 views -> DINOv2 patch features -> masked transformer
      -> single pass (temp 0.2)                              [reference number]
      -> N=16 draws (temp 0.8) -> whole-sequence IBQ decode
      -> grid snap -> candidate pool (union + 1-ring dilate)
      -> refinement network -> threshold tau                 [deployed output]

The 16 draws are generated INLINE, so no pool cache is needed. Reports
single-pass, flat-vote and refined IoU at N in {4, 8, 16}, plus CD / F-score and
per-tower timings. Records every tower's IoU; --combine merges shards and ranks
the worst.

Released configuration: transformer `net_best_iou`, tokenizer
`ibq_4x_nc1024_nq1` (whole-sequence decode), refinement `poolcls_c_sym_v1` at
tau 0.45 (tau is read from the checkpoint). Resumable per shard: saves every
--save_every towers and skips already-recorded names on restart.
"""

import os, sys, argparse, time, numpy as np, torch
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))   # repo root
sys.path.insert(0, _HERE)                    # script/ (sibling imports)
from utils.paths import DATA_ROOT, EVAL_RESULTS
from os.path import join as pjoin
from collections import Counter
from tqdm import tqdm
from eval_sampling import load_mask_model, eval_tower, _resolve_split, CHECKPOINTS_DIR, DATASET_NAME
from models.refinement import RefinementNetwork
from train_refinement import tower_features, iou_from_keep, N_FEAT
from utils.model_io import load_vq
from utils.voxel import coords_to_voxel_set, voxel_iou, GRID, OFFSET
from utils.fixseed import fixseed
from scipy.spatial import cKDTree

NEIGH6 = [(GRID[0],0,0),(-GRID[0],0,0),(0,GRID[1],0),(0,-GRID[1],0),(0,0,GRID[2]),(0,0,-GRID[2])]

def _cells(a):
    """Snapped raw coords (M,3) -> occupied-cell INDEX points (units = cells)."""
    a = np.asarray(a, np.float64)
    if len(a) == 0:
        return a.reshape(0, 3)
    return np.stack([(a[:, 0]-OFFSET[0])/GRID[0],
                     (a[:, 1]-OFFSET[1])/GRID[1],
                     (a[:, 2]-OFFSET[2])/GRID[2]], 1)

def cell_metrics(pred_tuples_arr, gt_set):
    """(cd, f0, f1) on cell-index point sets."""
    P = _cells(pred_tuples_arr)
    G = _cells(np.array(sorted(gt_set), dtype=np.int32)) if gt_set else np.zeros((0, 3))
    if len(P) == 0 or len(G) == 0:
        same = (len(P) == 0 and len(G) == 0)
        return (0.0, 1.0, 1.0) if same else (float('nan'), 0.0, 0.0)
    tp_, tg_ = cKDTree(P), cKDTree(G)
    dp, _ = tg_.query(P); dg, _ = tp_.query(G)
    cd = float(dp.mean() + dg.mean())
    inter = len(gt_set & {tuple(v) for v in np.asarray(pred_tuples_arr).tolist()})
    pr0, rc0 = inter/len(P), inter/len(G)
    f0 = 0.0 if pr0+rc0 == 0 else 2*pr0*rc0/(pr0+rc0)
    pr1, rc1 = float((dp <= 1.0).mean()), float((dg <= 1.0).mean())
    f1 = 0.0 if pr1+rc1 == 0 else 2*pr1*rc1/(pr1+rc1)
    return cd, f0, f1
TAUS = [0.30, 0.35, 0.40, 0.45, 0.50]
N_SUB = [4, 8, 16]   # sub-pool sizes (must be <= n_samples); reuse first-n of the 16 draws

def build_pool(sample_xyz, gt_set, dilate=1):
    cnt = Counter()
    for xyz in sample_xyz:
        cnt.update(coords_to_voxel_set(xyz))
    if dilate:
        for v in list(cnt.keys()):
            for d in NEIGH6:
                nb = (v[0]+d[0], v[1]+d[1], v[2]+d[2])
                if nb not in cnt:
                    cnt[nb] = 0
    vox = np.array(list(cnt.keys()), dtype=np.int16)
    count = np.array([cnt[tuple(v)] for v in vox.tolist()], dtype=np.uint8)
    label = np.array([1 if tuple(v) in gt_set else 0 for v in vox.tolist()], dtype=np.uint8)
    return vox, count, label, len(gt_set)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mask_name', default='myarch_mask_img_patch_v8_2')
    ap.add_argument('--mask_ckpt', default='net_best_iou.tar')
    ap.add_argument('--vq_name', default='ibq_4x_nc1024_nq1')
    ap.add_argument('--cls_name', default='poolcls_c_sym_v1')
    ap.add_argument('--cls_ckpt', default='best.tar')
    ap.add_argument('--data_root', default=DATA_ROOT)
    ap.add_argument('--split', default='val40k')
    ap.add_argument('--n_samples', type=int, default=16)
    ap.add_argument('--single_temp', type=float, default=0.2)
    ap.add_argument('--draft_temp', type=float, default=0.8)
    ap.add_argument('--decode_mode', default='whole')
    ap.add_argument('--dilate', type=int, default=1)
    ap.add_argument('--threshold', type=float, default=0.4375, help='flat-vote fraction (per-N)')
    ap.add_argument('--shard_idx', type=int, default=0)
    ap.add_argument('--num_shards', type=int, default=1)
    ap.add_argument('--gpu_id', type=int, default=0)
    ap.add_argument('--save_every', type=int, default=200)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out_dir', default=EVAL_RESULTS)
    ap.add_argument('--combine', action='store_true')
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    tag = f'mask_clsC_{a.mask_name}_{a.split.replace(".txt","")}_N{a.n_samples}'
    shard_path = pjoin(a.out_dir, f'{tag}_shard{a.shard_idx}of{a.num_shards}.npy')

    if a.combine:
        return combine(a, tag)

    fixseed(a.seed + a.shard_idx)
    device = torch.device(f'cuda:{a.gpu_id}')
    # mean/std (voxel-position normalisation for the refinement features) come from the VQ
    # meta -- the refinement network was trained with these stats.
    meta = pjoin(CHECKPOINTS_DIR, DATASET_NAME, a.vq_name, 'meta')
    mean = np.load(pjoin(meta, 'mean.npy')); std = np.load(pjoin(meta, 'std.npy'))
    mean3 = mean.reshape(-1)[:3]; std3 = std.reshape(-1)[:3]
    vq, vq_opt = load_vq(CHECKPOINTS_DIR, a.vq_name, device=device, dim_pose=3, dataset_name=DATASET_NAME)
    vq.eval().to(device); nb_code = vq.quantizer.layers[0].nb_code
    gmm = load_mask_model(vq_opt, a.mask_name, a.mask_ckpt, device)
    cck = torch.load(pjoin(CHECKPOINTS_DIR, DATASET_NAME, 'pool_classifier', a.cls_name, a.cls_ckpt),
                     map_location='cpu', weights_only=False)
    ca = cck['args']
    clsf = RefinementNetwork(N_FEAT, ca['latent_dim'], ca['ff_size'], ca['c_layers'], ca['n_heads'], ca['dropout']).to(device)
    clsf.load_state_dict(cck['model']); clsf.eval()
    print(f'[s{a.shard_idx}] classifier {a.cls_name} loaded (saved tau={cck.get("tau")})', flush=True)

    motion_dir = pjoin(a.data_root, 'new_joint_vecs_int'); feat_dir = pjoin(a.data_root, 'image_features_dino_patch')
    with open(_resolve_split(a.split, a.data_root)) as f:
        names = sorted(l.strip() for l in f if l.strip())
    names = names[a.shard_idx::a.num_shards]
    names = [n for n in names if os.path.exists(pjoin(motion_dir, n+'.npy')) and os.path.exists(pjoin(feat_dir, n+'.npy'))]
    if a.limit: names = names[:a.limit]

    res = {}
    if os.path.exists(shard_path):
        res = np.load(shard_path, allow_pickle=True).item()
        print(f'[s{a.shard_idx}] resuming: {len(res)} already done', flush=True)
    todo = [n for n in names if n not in res]
    print(f'[s{a.shard_idx}/{a.num_shards}] {len(todo)} towers to eval ({len(names)} total)', flush=True)

    tt = {'single': 0., 'draft16': 0., 'cls': 0., 'total': 0.}; ntimed = 0
    sync = lambda: torch.cuda.synchronize(device)
    for i, nm in enumerate(tqdm(todo, desc=f's{a.shard_idx}', leave=False)):
        motion = np.load(pjoin(motion_dir, nm+'.npy')); feat = np.load(pjoin(feat_dir, nm+'.npy')).astype(np.float32)
        n_cmp = (len(motion)//4)*4                      # token-alignment truncation
        gt = motion[:n_cmp].astype(float)
        gt_set = coords_to_voxel_set(gt)
        sync(); s0 = time.perf_counter()
        # single-pass temp 0.2
        ious1, _ = eval_tower(gmm, vq, mean, std, motion, feat, 1, a.single_temp, device, nb_code, decode_mode=a.decode_mode)
        single = float(ious1[0])
        sync(); s1 = time.perf_counter()
        # 16 drafts temp 0.8
        _, sample_xyz = eval_tower(gmm, vq, mean, std, motion, feat, a.n_samples, a.draft_temp, device, nb_code, decode_mode=a.decode_mode)
        sync(); s2 = time.perf_counter()
        # pool -> classifier C, ablated over sub-pool sizes N in N_SUB (reuse first-n draws)
        img = torch.from_numpy(feat).unsqueeze(0).to(device).float()
        byN = {}
        for n in N_SUB:
            vox, count, label, n_gt = build_pool(sample_xyz[:n], gt_set, a.dilate)
            cf = tower_features(vox, count, n, mean3, std3)
            vf = torch.from_numpy(cf).unsqueeze(0).to(device)
            pos = vf[..., 8:11].clone()
            kpad = torch.zeros(1, len(vox), dtype=torch.bool, device=device)
            with torch.no_grad():
                p = torch.sigmoid(clsf(vf, pos, img, kpad)).squeeze(0).cpu().numpy()
            lab = label.astype(np.float32)
            cls = {t: float(iou_from_keep(p >= t, lab, n_gt)) for t in TAUS}
            vote = float(iou_from_keep(count >= a.threshold * n, lab, n_gt))
            byN[n] = {'cls': cls, 'vote': vote}
        sync(); s3 = time.perf_counter()
        # CD / F-score for the deployment output (N=16 pool, clsC @ tau 0.45); p/vox
        # are from the last N_SUB iteration, i.e. n = 16.
        cdf45 = cell_metrics(vox[p >= 0.45], gt_set)
        res[nm] = {'single': single, 'n_blocks': int(n_cmp), 'byN': byN, 'cdf45': cdf45}
        tt['single'] += s1-s0; tt['draft16'] += s2-s1; tt['cls'] += s3-s2; tt['total'] += s3-s0; ntimed += 1
        if (i+1) % a.save_every == 0:
            np.save(shard_path, np.array(res, dtype=object))
    if ntimed:
        print(f'[s{a.shard_idx}] TIMING/tower (n={ntimed}): single={tt["single"]/ntimed*1000:.0f}ms  '
              f'16-draft={tt["draft16"]/ntimed*1000:.0f}ms  pool+clsC={tt["cls"]/ntimed*1000:.0f}ms  '
              f'TOTAL={tt["total"]/ntimed*1000:.0f}ms', flush=True)
    np.save(shard_path, np.array(res, dtype=object))
    sv = np.mean([r['single'] for r in res.values()])
    line = ' '.join(f'N{n}:clsC@.45={np.mean([r["byN"][n]["cls"][0.45] for r in res.values()]):.4f}'
                    f'/vote={np.mean([r["byN"][n]["vote"] for r in res.values()]):.4f}' for n in N_SUB)
    print(f'[s{a.shard_idx}] DONE n={len(res)} single={sv:.4f}  {line} -> {shard_path}', flush=True)

def combine(a, tag):
    res = {}
    for s in range(a.num_shards):
        p = pjoin(a.out_dir, f'{tag}_shard{s}of{a.num_shards}.npy')
        if os.path.exists(p): res.update(np.load(p, allow_pickle=True).item())
    n = len(res); assert n, 'no shard results'
    names = list(res.keys())
    single = np.array([res[k]['single'] for k in names])
    # per-N classifier (per tau) + vote
    clsN = {nn: {t: np.array([res[k]['byN'][nn]['cls'][t] for k in names]) for t in TAUS} for nn in N_SUB}
    voteN = {nn: np.array([res[k]['byN'][nn]['vote'] for k in names]) for nn in N_SUB}
    gen_label = 'masked transformer, IBQ whole-seq'
    print('='*72); print(f'  {a.split} EVAL  n={n} towers   ({gen_label})'); print('='*72)
    print(f'  single-pass  mean IoU = {single.mean():.4f}\n')
    print('  N-ABLATION  (Classifier C, classifier trained on N=16 mask-transformer pools):')
    print('    N    clsC@.30  @.35   @.40   @.45*  @.50   | flat-vote   gen-ms/tower')
    genms = {4: 75+46, 8: 75+92, 16: 75+185}  # single(75) + N*~11.6ms draft, from timing test
    for nn in N_SUB:
        row = '  '.join(f'{clsN[nn][t].mean():.4f}' for t in TAUS)
        best_t = max(TAUS, key=lambda t: clsN[nn][t].mean())
        print(f'   {nn:3d}   {row}  | {voteN[nn].mean():.4f}    ~{genms[nn]}ms   (best tau {best_t})')
    print('  (*0.45 = deployment tau)')
    # deltas vs N=16
    print(f'\n  clsC@.45  N16={clsN[16][0.45].mean():.4f}  '
          f'N8={clsN[8][0.45].mean():.4f} (Δ{clsN[8][0.45].mean()-clsN[16][0.45].mean():+.4f})  '
          f'N4={clsN[4][0.45].mean():.4f} (Δ{clsN[4][0.45].mean()-clsN[16][0.45].mean():+.4f})')
    print('='*72)
    # per-tower CSV (single + classifier@0.45 at each N)
    csv = pjoin(a.out_dir, 'per_tower_iou.csv')
    with open(csv, 'w') as f:
        f.write('tower,single_iou,clsC_N16,clsC_N8,clsC_N4,n_blocks\n')
        for k in names:
            b = res[k]['byN']
            f.write(f"{k},{res[k]['single']:.5f},{b[16]['cls'][0.45]:.5f},"
                    f"{b[8]['cls'][0.45]:.5f},{b[4]['cls'][0.45]:.5f},{res[k]['n_blocks']}\n")
    print(f'  per-tower CSV ({n} rows) -> {csv}')
    c16 = clsN[16][0.45]; order = np.argsort(c16)
    print('\n  === 30 WORST towers by Classifier C IoU (N=16 @0.45) ===')
    print('  tower                  clsC16  clsC8   single   blocks')
    for idx in order[:30]:
        k = names[idx]; b = res[k]['byN']
        print(f"  {k:22s} {c16[idx]:.4f}  {b[8]['cls'][0.45]:.4f}  {res[k]['single']:.4f}   {res[k]['n_blocks']}")
    np.save(pjoin(a.out_dir, 'per_tower_full.npy'), np.array(res, dtype=object))
    print(f'\n  full per-tower dict -> {pjoin(a.out_dir,"per_tower_full.npy")}')

if __name__ == '__main__':
    main()
