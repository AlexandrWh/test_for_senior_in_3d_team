"""Binary mask helpers: threshold -> all points (no connected-component filtering)."""

from __future__ import annotations

import numpy as np

# Shared threshold for axial mask -> PCA (windowed [0, 1] scale).
THR_01 = 20.0 / 255.0


def threshold_mask(img: np.ndarray, *, thr: float = THR_01) -> np.ndarray:
    return (img.astype(np.float32, copy=False) > float(thr)).astype(np.bool_)


def points_above_threshold(img: np.ndarray, *, thr: float = THR_01) -> np.ndarray | None:
    """All (x, y) pixels above threshold on a 2D slice [y, x]."""
    ys, xs = np.nonzero(threshold_mask(img, thr=thr))
    if xs.size == 0:
        return None
    return np.stack([xs.astype(np.float64), ys.astype(np.float64)], axis=1)
