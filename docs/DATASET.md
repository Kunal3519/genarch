# Gen3D dataset

200,000 procedurally generated architectural towers. Each tower is an ordered
sequence of block centre coordinates on a fixed voxel grid, rendered from four
fixed camera azimuths.

> Placed in `docs/` rather than `data/` on purpose: `data/` is the Python package
> holding the `Dataset` classes, not a data directory.

## Grid convention

Blocks sit on a regular lattice. X and Y have step 4 / offset 2; Z has step 8 /
offset 4, so snapping a float coordinate is

```
X = round((x - 2) / 4) * 4 + 2
Y = round((y - 2) / 4) * 4 + 2
Z = round((z - 4) / 8) * 8 + 4
```

`utils/voxel.py` is the single source of truth for this (`GRID`, `OFFSET`,
`grid_snap`, `voxel_iou`). Never re-derive it in a script.

Towers run roughly **820–1780 blocks** (mean ~1230). At the tokenizer's 4x
downsampling that is ~205–445 tokens.

## Layout

Two directories: the raw release and the converted form the code reads.

```
NewArchDataRaw/                 1.2 TB
└── tower_<id>/
    ├── params.json             generator parameters + cell_centers
    ├── northeast_realistic.png
    ├── northwest_realistic.png
    ├── southeast_realistic.png
    └── southwest_realistic.png

NewArchDataConverted/           790 GB
├── new_joint_vecs_int/         3.2 GB   (N,3) int32   <- training target
├── new_joint_vecs/             3.2 GB   (N,3) float32
├── image_features_dino_patch/  783 GB   (4,256,1024) float32  cached DINOv2
├── texts/                      828 MB   generator metadata (unused by the paper)
├── Mean.npy, Std.npy           (1,3) coordinate normalisation
└── *.txt                       split files
```

**The DINOv2 feature cache is 783 GB, 99% of the converted set.** It is
regenerable from the raw PNGs in a few GPU-hours, so mirror the raw release plus
the 3.2 GB of coordinates and rebuild features locally rather than transferring
it.

## Splits

An 80/20 split of all 200,000 towers:

| file | towers | use |
|---|---:|---|
| `train160k.txt` | 160,000 | training |
| `val40k.txt` | 40,000 | all reported validation numbers |

Verified disjoint: overlap 0, union exactly 200,000. Every number in the paper
is measured on `val40k`.

## Building the converted set

```bash
# 1. raw -> coordinate arrays + splits
python script/preprocess/convert_new_towers.py \
    --src  $GENARCH_IMAGE_ROOT \
    --dst  $GENARCH_DATA_ROOT

# 2. cache DINOv2 ViT-L patch features (4 x 256 x 1024 per tower)
python script/preprocess/precompute_patch_dl.py \
    --out_root $GENARCH_DATA_ROOT
```

Step 2 is the expensive one. Two notes that cost real time to learn:

- It is **CPU-bound on the bicubic resize**, not the GPU. A naive per-image loop
  leaves the GPU at 0% utilisation and runs ~15x slower; `precompute_patch_dl.py`
  uses a `DataLoader` with many workers for exactly this reason. If you fan out,
  check `nvidia-smi` utilisation before adding GPUs.
- Preprocessing is `Resize(256, bicubic) -> CenterCrop(224) -> ImageNet
  normalise`. Any re-encoding must match it exactly, or cached and fresh features
  will disagree.

## Pointing the code at it

See the Configuration section of the top-level `README.md`. In short, either

```bash
export GENARCH_DATA_ROOT=/path/to/NewArchDataConverted
export GENARCH_IMAGE_ROOT=/path/to/NewArchDataRaw
```

or put the same lines in a `.genarch_paths` file at the repo root.

## Sanity check

```python
import numpy as np
c = np.load(f'{DATA_ROOT}/new_joint_vecs_int/tower_003.npy')   # (1272, 3) int32
f = np.load(f'{DATA_ROOT}/image_features_dino_patch/tower_003.npy')  # (4,256,1024) float32
```

Coordinates should already be grid-aligned: `(grid_snap(c) == c).all()`.
