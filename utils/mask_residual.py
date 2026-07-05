"""Residual head-pose error from eye/ear masks after HeadAligner transform."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from models.head_aligner import apply_full_align
from models.pre_aligner import PreAlignParams
from paths import APPLY_SPACING_MM
from utils import center_crop_yx_to_square, prepare_isotropic_ct, read_nifti, sitk_image_to_numpy
from utils.angles import Point

HEAD_STRUCTURES_DIR = "head_glands_cavities"
EYE_LEFT = "eye_left.nii.gz"
EYE_RIGHT = "eye_right.nii.gz"
EAR_LEFT = "auditory_canal_left.nii.gz"
EAR_RIGHT = "auditory_canal_right.nii.gz"


def _resample_mask_to_ref(mask_img: sitk.Image, ref_img: sitk.Image) -> sitk.Image:
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(ref_img)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetDefaultPixelValue(0)
    resampler.SetOutputPixelType(sitk.sitkUInt8)
    return resampler.Execute(mask_img)


def prepare_mask_on_volume_grid(
    mask_path: str | Path,
    volume_path: str | Path,
    *,
    spacing_mm: float = APPLY_SPACING_MM,
) -> np.ndarray:
    """Mask on isotropic grid + square YX crop (matches PreAligner.prepare_volume)."""
    ref_ct = prepare_isotropic_ct(volume_path, spacing_mm=float(spacing_mm))
    m_img = _resample_mask_to_ref(read_nifti(mask_path), ref_ct)
    m = sitk_image_to_numpy(m_img) > 0
    return center_crop_yx_to_square(m.astype(np.float32)) > 0


def prealign_params_from_meta(meta: dict) -> PreAlignParams:
    pre = meta.get("prealign", {})
    return PreAlignParams(
        z_min=float(pre["z_min"]),
        z_max=float(pre["z_max"]),
        dx=float(pre["dx"]),
        dy=float(pre["dy"]),
        rz=float(pre.get("rz_pca_rad", pre.get("rz_rad", 0.0))),
        has_head=bool(meta.get("has_head", True)),
    )


def pose_angles_from_meta(meta: dict) -> tuple[float, float, float]:
    pose = meta.get("pose", {})
    return (
        float(pose.get("rz_rad", 0.0)),
        float(pose.get("ry_rad", 0.0)),
        float(pose.get("rx_rad", 0.0)),
    )


def apply_align_to_mask(
    mask_path: str | Path,
    volume_path: str | Path,
    meta: dict,
    *,
    spacing_mm: float = APPLY_SPACING_MM,
) -> np.ndarray:
    """Apply HeadAligner full transform to a structure mask @ spacing_mm."""
    mask = prepare_mask_on_volume_grid(mask_path, volume_path, spacing_mm=spacing_mm)
    params = prealign_params_from_meta(meta)
    rz_pose, ry_pose, rx_pose = pose_angles_from_meta(meta)
    aligned, out_meta = apply_full_align(
        mask.astype(np.float32),
        params,
        rz_pose=rz_pose,
        ry_pose=ry_pose,
        rx_pose=rx_pose,
        spacing_mm=float(spacing_mm),
        interpolator=sitk.sitkNearestNeighbor,
    )
    if not out_meta.get("ok") or aligned.size == 0:
        raise ValueError(str(out_meta.get("reason", "align_failed")))
    return aligned > 0.5


def mask_centroid_zyx(mask: np.ndarray) -> tuple[float, float, float] | None:
    zz, yy, xx = np.nonzero(mask > 0)
    if zz.size == 0:
        return None
    return float(zz.mean()), float(yy.mean()), float(xx.mean())


def abs_deviation_from_horizontal_rad(p0: Point, p1: Point) -> float:
    """|angle| of segment vs image horizontal (X axis), folded to [-90°, 90°]."""
    dx = float(p1[0]) - float(p0[0])
    dy = float(p1[1]) - float(p0[1])
    if abs(dx) < 1e-9 and abs(dy) < 1e-9:
        return 0.0
    a = math.atan2(dy, dx)
    if a > math.pi / 2:
        a -= math.pi
    elif a < -math.pi / 2:
        a += math.pi
    return abs(float(a))


def _lr_points_axial(
    left: tuple[float, float, float],
    right: tuple[float, float, float],
) -> tuple[Point, Point]:
    """Axial PNG: x=volume X, y=volume Y."""
    if left[2] <= right[2]:
        l, r = left, right
    else:
        l, r = right, left
    return [l[2], l[1]], [r[2], r[1]]


def _lr_points_coronal(
    left: tuple[float, float, float],
    right: tuple[float, float, float],
) -> tuple[Point, Point]:
    """Coronal PNG: x=volume X, y=volume Z."""
    if left[2] <= right[2]:
        l, r = left, right
    else:
        l, r = right, left
    return [l[2], l[0]], [r[2], r[0]]


def _sagittal_eye_ear_points(
    eye: tuple[float, float, float],
    ear: tuple[float, float, float],
) -> tuple[Point, Point]:
    """Sagittal PNG: x=volume Y (A-P), y=volume Z."""
    return [eye[1], eye[0]], [ear[1], ear[0]]


def compute_mask_residuals(
    volume_path: str | Path,
    masks_case_dir: str | Path,
    meta: dict,
    *,
    spacing_mm: float = APPLY_SPACING_MM,
) -> dict[str, object]:
    """
    Apply align predictions to eye/ear masks; measure residual |tilt| vs horizontal.

    rz proxy: axial L-R eye/ear segment deviation from horizontal.
    ry proxy: coronal L-R eye/ear segment deviation from horizontal.
    rx proxy: sagittal eye-mid → ear-mid deviation from horizontal.
    """
    volume_path = Path(volume_path)
    masks_case_dir = Path(masks_case_dir)
    head_dir = masks_case_dir / HEAD_STRUCTURES_DIR
    if not head_dir.is_dir():
        raise FileNotFoundError(f"missing {head_dir}")

    paths = {
        "eye_l": head_dir / EYE_LEFT,
        "eye_r": head_dir / EYE_RIGHT,
        "ear_l": head_dir / EAR_LEFT,
        "ear_r": head_dir / EAR_RIGHT,
    }
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"missing mask: {path}")

    aligned: dict[str, np.ndarray] = {}
    cents: dict[str, tuple[float, float, float] | None] = {}
    for key, path in paths.items():
        aligned[key] = apply_align_to_mask(path, volume_path, meta, spacing_mm=spacing_mm)
        cents[key] = mask_centroid_zyx(aligned[key])
        if cents[key] is None:
            raise ValueError(f"empty aligned mask: {key}")

    el, er = cents["eye_l"], cents["eye_r"]
    al, ar = cents["ear_l"], cents["ear_r"]
    assert el is not None and er is not None and al is not None and ar is not None

    eye_mid = (
        0.5 * (el[0] + er[0]),
        0.5 * (el[1] + er[1]),
        0.5 * (el[2] + er[2]),
    )
    ear_mid = (
        0.5 * (al[0] + ar[0]),
        0.5 * (al[1] + ar[1]),
        0.5 * (al[2] + ar[2]),
    )

    rz_eyes = abs_deviation_from_horizontal_rad(*_lr_points_axial(el, er))
    rz_ears = abs_deviation_from_horizontal_rad(*_lr_points_axial(al, ar))
    ry_eyes = abs_deviation_from_horizontal_rad(*_lr_points_coronal(el, er))
    ry_ears = abs_deviation_from_horizontal_rad(*_lr_points_coronal(al, ar))
    rx_om = abs_deviation_from_horizontal_rad(*_sagittal_eye_ear_points(eye_mid, ear_mid))

    def _deg(r: float) -> float:
        return float(math.degrees(r))

    out: dict[str, object] = {
        "rz_eyes_rad": rz_eyes,
        "rz_ears_rad": rz_ears,
        "ry_eyes_rad": ry_eyes,
        "ry_ears_rad": ry_ears,
        "rx_om_rad": rx_om,
        "rz_eyes_deg": _deg(rz_eyes),
        "rz_ears_deg": _deg(rz_ears),
        "ry_eyes_deg": _deg(ry_eyes),
        "ry_ears_deg": _deg(ry_ears),
        "rx_om_deg": _deg(rx_om),
        "centroids_zyx": {
            "eye_l": list(el),
            "eye_r": list(er),
            "ear_l": list(al),
            "ear_r": list(ar),
            "eye_mid": list(eye_mid),
            "ear_mid": list(ear_mid),
        },
        "spacing_mm": float(spacing_mm),
    }
    return out
