"""
Token-length predictor: pooled DINOv2 patch features -> tower token count.

Removes the GT-length input from the generation pipeline: at deployment the
MaskTransformer's generate() needs m_lens (tokens, = blocks/4); this small MLP
predicts it from the same (4,256,1024) DINOv2 patch features the mask model
conditions on. len_sens sweep (June 2026) says +-2 tokens costs <=0.002 voted
IoU, +-8 tokens ~0.013 -> target MAE <= 2-3 tokens.

Stages (--stage all = both):
  cache  read every tower's (4,256,1024) patch features, pool per --pool, plus
         token count from new_joint_vecs_int. One .npz holding train+val.
         IO-bound (~210 GB read), threaded.
  train  MLP on the cached features, Huber loss on standardized token count.
         Reports val MAE + the P(|err|<=k) error budget from the len_sens
         sweep. Saves best-MAE checkpoint + norm stats to
         checkpoints/myarch/length_predictor/.

Pooling (--pool):
  mean   per-view mean over the 256 patches -> flat 4096. Plain MLP.
         RESULT June 2026: val MAE 9.94 tok — too lossy; global mean throws
         away spatial extent (how much of the frame the tower fills), which
         is the main length signal.
  grid   per-view 4x4 region means over the 16x16 patch grid (16 tokens) +
         global max (1 token) -> (4,17,1024) fp16. Token-wise Linear(1024->128)
         then MLP. Keeps coarse silhouette extent.

Usage:
    uv run python script/train_length_predictor.py \
        --stage all --gpu_id 0
"""
import argparse
import os
import sys
import time
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))  # repo root
sys.path.insert(0, os.path.dirname(_HERE))                   # script/
from utils.paths import DATA_ROOT

from concurrent.futures import ThreadPoolExecutor
from os.path import join as pjoin
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

DATA_ROOT = DATA_ROOT
OUT_DIR   = './checkpoints/myarch/length_predictor'


def read_split(data_root, split):
    with open(pjoin(data_root, split + '.txt')) as f:
        return sorted(l.strip() for l in f if l.strip())


def pool_grid(feat):
    """(4,256,1024) patch features -> (4,17,1024): 4x4 region means + global max."""
    g = feat.reshape(4, 4, 4, 4, 4, 1024).mean(axis=(2, 4))      # (4,4,4,1024)
    regions = g.reshape(4, 16, 1024)
    gmax = feat.max(axis=1, keepdims=True)                       # (4,1,1024)
    return np.concatenate([regions, gmax], axis=1)               # (4,17,1024)


def load_length_predictor(ckpt_path, device):
    """Returns (model, ckpt_dict) for inference; use predict_tokens()."""
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)  # own ckpt (numpy scalars)
    assert ck.get('pool') == 'grid', 'only grid-pool checkpoints supported'
    model = GridLengthMLP(68, 1024, ck['proj_dim'], tuple(ck['hidden']), ck['dropout'])
    model.load_state_dict(ck['model'])
    model.eval().to(device)
    ck['x_mean'] = ck['x_mean'].to(device); ck['x_std'] = ck['x_std'].to(device)
    ck['y_mean'] = ck['y_mean'].to(device); ck['y_std'] = ck['y_std'].to(device)
    return model, ck


@torch.no_grad()
def predict_tokens(model, ck, feat, device):
    """(4,256,1024) numpy patch features -> predicted token count (int)."""
    x = torch.from_numpy(pool_grid(feat)).reshape(1, 68, 1024).float().to(device)
    x = (x - ck['x_mean']) / ck['x_std']
    p = model(x) * ck['y_std'] + ck['y_mean']
    return int(round(float(p)))


def build_cache(args):
    feat_dir   = pjoin(args.data_root, 'image_features_dino_patch')
    motion_dir = pjoin(args.data_root, 'new_joint_vecs_int')
    os.makedirs(OUT_DIR, exist_ok=True)

    def pool_one(name):
        feat = np.load(pjoin(feat_dir, name + '.npy'))          # (4,256,1024)
        if args.pool == 'mean':
            x = feat.mean(axis=1).reshape(-1).astype(np.float32)   # (4096,)
        else:  # grid: 16x16 patch grid -> 4x4 region means + per-view global max
            x = pool_grid(feat).astype(np.float16)                 # (4,17,1024)
        y = len(np.load(pjoin(motion_dir, name + '.npy'))) // 4  # tokens
        return x, y

    out = {}
    for split in ('train', 'val'):
        names = read_split(args.data_root, split)
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            rows = list(tqdm(ex.map(pool_one, names), total=len(names), desc=f'cache {split}'))
        out[f'X_{split}'] = np.stack([r[0] for r in rows])
        out[f'y_{split}'] = np.array([r[1] for r in rows], dtype=np.float32)
        out[f'names_{split}'] = np.array(names)
        print(f'{split}: X {out[f"X_{split}"].shape}  tokens '
              f'min={out[f"y_{split}"].min():.0f} max={out[f"y_{split}"].max():.0f}')
    np.savez(args.cache_path, **out)
    print(f'Saved {args.cache_path} ({os.path.getsize(args.cache_path)/1e9:.2f} GB)')


class LengthMLP(nn.Module):
    def __init__(self, in_dim=4096, hidden=(512, 256), dropout=0.1):
        super().__init__()
        layers, d = [], in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.GELU(), nn.Dropout(dropout)]
            d = h
        layers += [nn.Linear(d, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class GridLengthMLP(nn.Module):
    """(B, 4*17, 1024) region tokens -> token-wise projection -> flat MLP."""
    def __init__(self, n_tokens=68, dim=1024, proj=128, hidden=(1024, 256), dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(dim, proj)
        self.act = nn.GELU()
        layers, d = [], n_tokens * proj
        for h in hidden:
            layers += [nn.Linear(d, h), nn.GELU(), nn.Dropout(dropout)]
            d = h
        layers += [nn.Linear(d, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):                       # (B, n_tokens, 1024)
        h = self.act(self.proj(x)).flatten(1)
        return self.net(h).squeeze(-1)


def _metrics(pred_tok, y_tok):
    """pred_tok already rounded to int tokens."""
    err = pred_tok - y_tok
    out = {'mae': np.abs(err).mean(), 'rmse': np.sqrt((err ** 2).mean()),
           'bias': err.mean()}
    for k in (1, 2, 4, 8):
        out[f'p{k}'] = (np.abs(err) <= k).mean()
    out['worst'] = np.abs(err).max()
    return out


def _feat_stats(X, chunk=4096):
    """fp32 per-feature mean/std of a (possibly fp16) tensor, chunked."""
    s = torch.zeros(X.shape[1:], dtype=torch.float32, device=X.device)
    ss = torch.zeros_like(s)
    for i in range(0, len(X), chunk):
        c = X[i:i + chunk].float()
        s += c.sum(0); ss += (c * c).sum(0)
    mean = s / len(X)
    std = (ss / len(X) - mean ** 2).clamp_min(1e-12).sqrt()
    return mean.unsqueeze(0), std.clamp_min(1e-6).unsqueeze(0)


def train(args, device):
    d = np.load(args.cache_path, allow_pickle=True)
    Xtr = torch.from_numpy(d['X_train']).to(device)
    ytr = torch.from_numpy(d['y_train']).to(device)
    Xva = torch.from_numpy(d['X_val']).to(device)
    yva_np = d['y_val']

    if Xtr.ndim == 4:                       # grid: (N,4,17,1024) -> (N,68,1024)
        Xtr = Xtr.flatten(1, 2)
        Xva = Xva.flatten(1, 2)

    x_mean, x_std = _feat_stats(Xtr)
    y_mean, y_std = ytr.mean(), ytr.std()

    def norm(x):
        return (x.float() - x_mean) / x_std

    if Xtr.ndim == 3:
        model = GridLengthMLP(Xtr.shape[1], Xtr.shape[2], args.proj_dim,
                              tuple(args.hidden), args.dropout).to(device)
    else:
        Xtr, Xva = norm(Xtr), norm(Xva)     # small enough to normalize once
        model = LengthMLP(Xtr.shape[1], tuple(args.hidden), args.dropout).to(device)
    n_par = sum(p.numel() for p in model.parameters())
    print(f'LengthMLP params: {n_par/1e6:.2f}M  train={len(Xtr)}  val={len(Xva)}  '
          f'tokens mean={y_mean:.0f} std={y_std:.0f}')
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)
    loss_fn = nn.HuberLoss(delta=1.0)   # delta in std units (~1 std of token count)

    is_grid = Xtr.ndim == 3
    best = {'mae': 1e9}
    n = len(Xtr)
    for ep in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(n, device=device)
        tot = 0.0
        for i in range(0, n, args.batch_size):
            idx = perm[i:i + args.batch_size]
            xb = norm(Xtr[idx]) if is_grid else Xtr[idx]
            pred = model(xb)
            loss = loss_fn(pred, (ytr[idx] - y_mean) / y_std)
            optim.zero_grad(); loss.backward(); optim.step()
            tot += loss.item() * len(idx)
        sched.step()

        if ep % args.eval_every == 0 or ep == args.epochs:
            model.eval()
            with torch.no_grad():
                xv = norm(Xva) if is_grid else Xva
                pv = (model(xv) * y_std + y_mean).cpu().numpy()
            m = _metrics(np.round(pv), yva_np)
            tag = ''
            if m['mae'] < best['mae']:
                best = {**m, 'ep': ep}
                torch.save({'model': model.state_dict(),
                            'x_mean': x_mean.cpu(), 'x_std': x_std.cpu(),
                            'y_mean': y_mean.cpu(), 'y_std': y_std.cpu(),
                            'hidden': list(args.hidden), 'dropout': args.dropout,
                            'pool': args.pool, 'proj_dim': args.proj_dim,
                            'ep': ep, 'val_mae': m['mae']},
                           pjoin(OUT_DIR, f'model_{args.pool}.tar'))
                tag = '  <- saved'
            print(f'ep {ep:4d}  train_loss {tot/n:.4f}  val MAE {m["mae"]:.2f} tok  '
                  f'bias {m["bias"]:+.2f}  P<=1 {m["p1"]:.1%}  P<=2 {m["p2"]:.1%}  '
                  f'P<=4 {m["p4"]:.1%}  P<=8 {m["p8"]:.1%}  worst {m["worst"]:.0f}{tag}')

    # naive baseline for reference: always predict the train-mean length
    base = _metrics(np.full_like(yva_np, np.round(float(y_mean.cpu()))), yva_np)
    print('=' * 78)
    print(f'  best (ep {best["ep"]}): val MAE {best["mae"]:.2f} tok ({best["mae"]*4:.1f} blocks)  '
          f'rmse {best["rmse"]:.2f}  bias {best["bias"]:+.2f}')
    print(f'  P(|err|<=1) {best["p1"]:.1%}   <=2 {best["p2"]:.1%}   <=4 {best["p4"]:.1%}   '
          f'<=8 {best["p8"]:.1%}   worst {best["worst"]:.0f}')
    print(f'  predict-mean baseline: MAE {base["mae"]:.2f} tok  P<=2 {base["p2"]:.1%}')
    print(f'  len_sens budget: +-2 tok <= 0.002 voted IoU, +-8 tok ~ 0.013')
    print('=' * 78)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--stage', choices=['cache', 'train', 'all'], default='all')
    ap.add_argument('--data_root', default=DATA_ROOT)
    ap.add_argument('--pool', choices=['mean', 'grid'], default='grid')
    ap.add_argument('--cache_path', default=None,
                    help='default: checkpoints/myarch/length_predictor/pooled_feats[_grid].npz')
    ap.add_argument('--workers', type=int, default=16, help='cache stage IO threads')
    ap.add_argument('--proj_dim', type=int, default=128, help='grid: token-wise projection dim')
    ap.add_argument('--hidden', type=int, nargs='+', default=[512, 256])
    ap.add_argument('--dropout', type=float, default=0.1)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--weight_decay', type=float, default=1e-4)
    ap.add_argument('--batch_size', type=int, default=1024)
    ap.add_argument('--epochs', type=int, default=300)
    ap.add_argument('--eval_every', type=int, default=10)
    ap.add_argument('--gpu_id', type=int, default=0)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    if args.cache_path is None:
        args.cache_path = pjoin(OUT_DIR, 'pooled_feats.npz' if args.pool == 'mean'
                                else f'pooled_feats_{args.pool}.npz')
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    if args.stage in ('cache', 'all'):
        build_cache(args)
    if args.stage in ('train', 'all'):
        device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
        print(f'Device: {device}')
        train(args, device)


if __name__ == '__main__':
    main()
