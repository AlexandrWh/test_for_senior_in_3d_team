"""Axial head center: threshold + PCA median projections on all mask points."""

from __future__ import annotations

import numpy as np

from head_align.mask_utils import THR_01, points_above_threshold

MIN_POINTS = 30
AXIAL_PCA_N_SLICES_UP = 3  # slices at z_center + i*z_step, i=0..n-1 (toward crown)
AXIAL_PCA_Z_STEP = 10


def _axial_slice_indices_upward(
    z_len: int,
    *,
    n_slices: int = AXIAL_PCA_N_SLICES_UP,
    z_step: int = AXIAL_PCA_Z_STEP,
) -> np.ndarray:
    """Z indices from geometric center upward (higher z toward crown)."""
    z_len = int(z_len)
    if z_len <= 0:
        return np.array([0], dtype=int)
    cz = z_len // 2
    n_slices = max(1, int(n_slices))
    z_step = max(1, int(z_step))
    indices: list[int] = []
    for i in range(n_slices):
        zi = cz + i * z_step
        if zi >= z_len:
            break
        indices.append(zi)
    if not indices:
        indices = [min(cz, z_len - 1)]
    return np.asarray(indices, dtype=int)


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
        "thr": float(thr),
    }


def volume_axial_pca_median_center(
    vol_zyx: np.ndarray,
    *,
    n_slices_up: int = AXIAL_PCA_N_SLICES_UP,
    z_step: int = AXIAL_PCA_Z_STEP,
    thr: float = THR_01,
) -> dict | None:
    """Multi-slice axial PCA: threshold masks on slices above center, all points pooled."""
    vol = vol_zyx.astype(np.float32, copy=False)
    if vol.ndim != 3:
        return None

    indices = _axial_slice_indices_upward(int(vol.shape[0]), n_slices=n_slices_up, z_step=z_step)
    parts: list[np.ndarray] = []
    for zi in indices:
        pts = points_above_threshold(vol[int(zi)], thr=thr)
        if pts is not None:
            parts.append(pts)

    if not parts:
        return None

    pts = np.concatenate(parts, axis=0)
    if pts.shape[0] < MIN_POINTS:
        return None
    out = _pca_median_center_from_points(pts, thr=thr)
    if out is not None:
        out["z_indices"] = indices
        out["n_slices"] = int(len(indices))
    return out


def shift_xy_to_center(shape_yx: tuple[int, int], center_xy: np.ndarray) -> tuple[float, float]:
    h, w = shape_yx
    return float(w / 2 - center_xy[0]), float(h / 2 - center_xy[1])


def pca_e1_tilt_rad(ax_det: dict) -> float:
    """Inclination of e1 from +Y on axial (x, y) slice."""
    e1 = np.asarray(ax_det["e1"], dtype=np.float64)
    return float(np.arctan2(e1[0], e1[1]))


def pca_axis_segments(
    center_xy: np.ndarray,
    e1: np.ndarray,
    e2: np.ndarray,
    shape_yx: tuple[int, int],
    *,
    scale: float = 0.48,
) -> tuple[tuple[tuple[float, float], tuple[float, float]], tuple[tuple[float, float], tuple[float, float]]]:
    h, w = (int(shape_yx[0]), int(shape_yx[1]))
    half = float(scale) * float(max(h, w))
    c = np.asarray(center_xy, dtype=np.float64)
    v1 = np.asarray(e1, dtype=np.float64)
    v2 = np.asarray(e2, dtype=np.float64)
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 > 1e-8:
        v1 = v1 / n1
    if n2 > 1e-8:
        v2 = v2 / n2
    seg1 = (tuple(c - half * v1), tuple(c + half * v1))
    seg2 = (tuple(c - half * v2), tuple(c + half * v2))
    return seg1, seg2
