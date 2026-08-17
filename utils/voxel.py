"""Grid-snap + voxel-IoU helpers for myarch towers.

The NewArchData tower grid: X/Y step=4 offset=2, Z step=8 offset=4.
    X = round((x - 2) / 4) * 4 + 2
    Y = round((y - 2) / 4) * 4 + 2
    Z = round((z - 4) / 8) * 8 + 4

Single source of truth, every script / trainer that did its own copy now imports from here.
"""
from __future__ import annotations

import numpy as np

GRID = (4, 4, 8)          # (dx, dy, dz)
OFFSET = (2, 2, 4)        # (ox, oy, oz)


def grid_snap(coords: np.ndarray, grid=GRID, offset=OFFSET) -> np.ndarray:
    """Snap an (N, 3) float coord array to the tower grid. Returns int (N, 3)."""
    dx, dy, dz = grid
    ox, oy, oz = offset
    s = np.empty_like(coords, dtype=np.int64)
    s[:, 0] = np.round((coords[:, 0] - ox) / dx) * dx + ox
    s[:, 1] = np.round((coords[:, 1] - oy) / dy) * dy + oy
    s[:, 2] = np.round((coords[:, 2] - oz) / dz) * dz + oz
    return s


def coords_to_voxel_set(coords: np.ndarray):
    """Snap and de-duplicate to a Python set of (x, y, z) int tuples."""
    s = grid_snap(coords)
    return set(map(tuple, s.tolist()))


def voxel_iou(coords_a: np.ndarray, coords_b: np.ndarray) -> float:
    """Voxel-set IoU between two coord arrays after grid snapping.

    Both empty → 1.0; one empty → 0.0.
    """
    a = coords_to_voxel_set(coords_a)
    b = coords_to_voxel_set(coords_b)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)
