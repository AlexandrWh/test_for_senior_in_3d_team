"""Axial PCA on central slice: threshold mask -> median center + e1/e2."""

from __future__ import annotations

import numpy as np

THR_01 = 20.0 / 255.0
MIN_POINTS = 30


def threshold_mask(img: np.ndarray, *, thr: float = THR_01) -> np.ndarray:
    return (img.astype(np.float32, copy=False) > float(thr)).astype(np.bool_)


def points_above_threshold(img: np.ndarray, *, thr: float = THR_01) -> np.ndarray | None:
    """All (x, y) pixels above threshold on a 2D slice [y, x]."""
    ys, xs = np.nonzero(threshold_mask(img, thr=thr))
    if xs.size == 0:
        return None
    return np.stack([xs.astype(np.float64), ys.astype(np.float64)], axis=1)


def _pca_median_center_from_points(pts: np.ndarray, *, thr: float) -> dict | None:
    if pts.shape[0] < MIN_POINTS:
        return None

    origin = pts.mean(axis=0)
    centered = pts - origin
    cov = (centered.T @ centered) / max(1.0, float(len(pts)))
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]
    e1 = evecs[:, order[0]]
    e2 = evecs[:, order[1]]

    t1 = centered @ e1
    t2 = centered @ e2
    m1 = float(np.median(t1))
    m2 = float(np.median(t2))
    center = origin + m1 * e1 + m2 * e2

    return {
        "center": center,
        "origin": origin,
        "e1": e1,
        "e2": e2,
        "m1": m1,
        "m2": m2,
        "n_points": int(pts.shape[0]),
        "evals": evals[order],
        "thr": float(thr),
    }


def central_slice_axial_pca(
    vol_zyx: np.ndarray,
    *,
    z_index: int | None = None,
    thr: float = THR_01,
) -> dict | None:
    """PCA on thresholded mask of a single axial slice (default: geometric Z center)."""
    vol = vol_zyx.astype(np.float32, copy=False)
    if vol.ndim != 3:
        return None

    zi = int(vol.shape[0] // 2) if z_index is None else int(z_index)
    zi = max(0, min(zi, int(vol.shape[0]) - 1))

    pts = points_above_threshold(vol[zi], thr=thr)
    if pts is None:
        return None

    out = _pca_median_center_from_points(pts, thr=thr)
    if out is not None:
        out["z_index"] = zi
    return out


def shift_xy_to_center(shape_yx: tuple[int, int], center_xy: np.ndarray) -> tuple[float, float]:
    h, w = shape_yx
    return float(w / 2 - center_xy[0]), float(h / 2 - center_xy[1])
