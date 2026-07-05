"""Angle conventions for head pose (CW-positive).

Convention (viewed along +axis toward origin):
  +angle = clockwise (CW)
  -angle = counter-clockwise (CCW) by |angle|

Measured tilt (guide line, PCA): positive CW misalignment from canonical.
Apply to volume (scipy Euler ZYX):
  PreAlign Z: -rz_pca
  Pose aug:   -rz_pca + rz_aug,  ry_aug,  rx_aug
  Full align: -(rz_pca + rz_pose),  ry_pose, -rx_pose
"""

from __future__ import annotations

import math

Point = list[float]


def segment_tilt_png_rad(p0: Point, p1: Point) -> float:
    """Raw PNG tilt: atan2(dx, dy), + = leans right (origin upper)."""
    y0, y1 = float(p0[1]), float(p1[1])
    x0, x1 = float(p0[0]), float(p1[0])
    if y0 <= y1:
        top, bottom = (x0, y0), (x1, y1)
    else:
        top, bottom = (x1, y1), (x0, y0)
    dx = float(bottom[0]) - float(top[0])
    dy = float(bottom[1]) - float(top[1])
    if abs(dx) < 1e-9 and abs(dy) < 1e-9:
        return 0.0
    return float(math.atan2(dx, dy))


def segment_tilt_cw_rad(p0: Point, p1: Point) -> float:
    """Guide-line tilt in CW-positive convention."""
    return -float(segment_tilt_png_rad(p0, p1))


def pca_e1_tilt_cw_rad(e1: object) -> float:
    """PCA e1 axis tilt in CW-positive convention (axial x,y components)."""
    import numpy as np

    e = np.asarray(e1, dtype=np.float64)
    return -float(math.atan2(float(e[0]), float(e[1])))


def prealign_apply_euler_cw(rz_pca_cw: float) -> tuple[float, float, float]:
    """CCW scipy correction for measured CW tilt rz_pca."""
    return (-float(rz_pca_cw), 0.0, 0.0)


def pose_apply_euler_cw(
    rz_pca_cw: float,
    rz_aug_cw: float,
    ry_aug_cw: float,
    rx_aug_cw: float,
) -> tuple[float, float, float]:
    """PreAlign CCW correction + aug (CW-positive) added in euler space."""
    rz_s = -float(rz_pca_cw) + float(rz_aug_cw)
    return (rz_s, float(ry_aug_cw), float(rx_aug_cw))


def full_align_apply_euler_cw(
    rz_pca_cw: float,
    rz_pose_cw: float,
    ry_pose_cw: float,
    rx_pose_cw: float,
) -> tuple[float, float, float]:
    """Single correction on 1 mm: undo measured tilt (rz_pca + rz_pose, ry, rx)."""
    rz_total = float(rz_pca_cw) + float(rz_pose_cw)
    return (-rz_total, float(ry_pose_cw), -float(rx_pose_cw))
