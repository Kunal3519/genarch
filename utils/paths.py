"""Single resolution point for every dataset / artifact root.

Resolution order, highest first:

  1. environment variable          e.g. GENARCH_DATA_ROOT=/mnt/towers
  2. `.genarch_paths` at the repo root -- a gitignored KEY=VALUE file for
     machine-specific paths, so nobody's absolute paths live in the repo:

         GENARCH_DATA_ROOT=/data/me/datasets/NewArchDataConverted
         GENARCH_IMAGE_ROOT=/data/me/datasets/NewArchDataRaw

  3. the repo-relative default below

Import these instead of writing a literal path:

    from utils.paths import DATA_ROOT, IMAGE_ROOT
"""
import os

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_LOCAL_FILE = os.path.join(_REPO, '.genarch_paths')


def _load_local(path=_LOCAL_FILE):
    """Parse the optional gitignored KEY=VALUE overrides file."""
    out = {}
    if not os.path.isfile(path):
        return out
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            k, v = line.split('=', 1)
            out[k.strip()] = os.path.expanduser(v.strip())
    return out


_LOCAL = _load_local()


def _resolve(key, default):
    return os.environ.get(key) or _LOCAL.get(key) or default


# Converted dataset: new_joint_vecs_int/, image_features_dino_patch/, split files.
DATA_ROOT = _resolve('GENARCH_DATA_ROOT', os.path.join(_REPO, 'data_root'))

# Raw towers: tower_<id>/params.json + 4 rendered PNG views. Only needed to
# re-render or re-encode features.
IMAGE_ROOT = _resolve('GENARCH_IMAGE_ROOT', os.path.join(_REPO, 'data_raw'))

# Where eval scripts persist per-tower records / shards.
EVAL_RESULTS = _resolve('GENARCH_EVAL_RESULTS', os.path.join(_REPO, 'eval_results'))

# Model weights (checkpoints/<dataset_name>/<run>/...).
CHECKPOINTS_DIR = _resolve('GENARCH_CHECKPOINTS', './checkpoints')


