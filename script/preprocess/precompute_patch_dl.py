"""
DINOv2 patch precompute (DataLoader-parallel; ~15x faster than a single-threaded loop
(bit-identical features) but parallelized via a torch DataLoader with many CPU
workers, so the 374ms/img CPU bicubic resize runs across many cores instead of
one. GPU just does the (trivial) forward. Resumable; shardable across processes.
"""
import os, argparse, numpy as np, torch
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from os.path import join as pjoin

VIEWS = ['northeast_realistic.png', 'northwest_realistic.png',
         'southeast_realistic.png', 'southwest_realistic.png']
# DINOv2 preprocessing: Resize(256, bicubic) -> CenterCrop(224) -> ImageNet norm.
T = transforms.Compose([
    transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

class TowerDS(Dataset):
    def __init__(self, names, image_root):
        self.names = names; self.image_root = image_root
    def __len__(self): return len(self.names)
    def __getitem__(self, i):
        name = self.names[i]; td = pjoin(self.image_root, name)
        imgs = []
        for v in VIEWS:
            ip = pjoin(td, v)
            try:
                im = Image.open(ip).convert('RGB'); im.load()
                imgs.append(T(im))
            except Exception as e:
                print(f'  [skip] {name}/{v}: {type(e).__name__}: {e}', flush=True)
                return name, torch.empty(0)
        return name, torch.stack(imgs)  # [4,3,224,224]

def collate(batch):
    return [(n, t) for n, t in batch if t.numel() > 0]

def main(a):
    device = f'cuda:{a.gpu}'
    out_dir = pjoin(a.out_root, 'image_features_dino_patch'); os.makedirs(out_dir, exist_ok=True)
    with open(a.split_file) as f:
        names = sorted({l.strip() for l in f if l.strip()})
    names = names[a.shard::a.num_shards]
    names = [n for n in names if not os.path.exists(pjoin(out_dir, n + '.npy'))]  # resume
    print(f'[GPU {a.gpu} | shard {a.shard}/{a.num_shards}] {len(names)} towers to encode, {a.workers} workers', flush=True)
    if not names:
        print(f'[GPU {a.gpu}] nothing to do'); return
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14'); model.eval().to(device)
    for p in model.parameters(): p.requires_grad = False
    dl = DataLoader(TowerDS(names, a.image_root), batch_size=a.batch, num_workers=a.workers,
                    collate_fn=collate, prefetch_factor=4, persistent_workers=True)
    done = 0
    for good in dl:
        if not good: continue
        ts = torch.stack([t for _, t in good]).reshape(-1, 3, 224, 224).to(device, non_blocking=True)
        with torch.no_grad():
            patch = model.forward_features(ts)['x_norm_patchtokens'].reshape(len(good), 4, 256, -1)
        for (n, _), f in zip(good, patch.cpu().numpy()):
            np.save(pjoin(out_dir, n + '.npy'), f.astype(np.float32))
        done += len(good)
        if done % 1024 < a.batch:
            print(f'[GPU {a.gpu}] {done}/{len(names)}', flush=True)
    print(f'[GPU {a.gpu}] DONE {done}/{len(names)}', flush=True)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--num_shards', type=int, default=1)
    p.add_argument('--workers', type=int, default=64)
    p.add_argument('--batch', type=int, default=64)
    p.add_argument('--image_root', required=True)
    p.add_argument('--out_root', required=True)
    p.add_argument('--split_file', required=True)
    main(p.parse_args())
