"""CT I/O, resampling, volume prep for Z-slice classifier."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import SimpleITK as sitk

from utils.labels import collect_case_ids, case_id_from_name  # noqa: F401 — re-export

__all__ = [
    "read_nifti",
    "sitk_image_to_numpy",
    "apply_brain_ct_window",
    "resample_ct_to_isotropic",
    "prepare_isotropic_ct",
    "center_crop_yx_to_square",
    "center_crop_pad_yx",
    "center_slice_np",
    "collect_case_ids",
    "case_id_from_name",
]


def read_nifti(path: str | Path) -> sitk.Image:
    return canonicalize_cq500_orientation(sitk.ReadImage(str(path)))


def canonicalize_cq500_orientation(image: sitk.Image) -> sitk.Image:
    if image.GetDirection()[4] > 0:
        return sitk.Flip(image, flipAxes=(False, True, False))
    return image


def sitk_image_to_numpy(image: sitk.Image) -> np.ndarray:
    return sitk.GetArrayFromImage(image)


def apply_brain_ct_window(
    hu: np.ndarray,
    window_level: float = 40.0,
    window_width: float = 80.0,
    output_range: tuple[float, float] = (0.0, 1.0),
) -> np.ndarray:
    hu = hu.astype(np.float32)
    lower = window_level - window_width / 2.0
    upper = window_level + window_width / 2.0
    clipped = np.clip(hu, lower, upper)
    out_min, out_max = output_range
    windowed = (clipped - lower) / (upper - lower)
    windowed = windowed * (out_max - out_min) + out_min
    return windowed.astype(np.float32)


def resample_to_spacing(
    image: sitk.Image,
    target_spacing: tuple[float, float, float],
    *,
    interpolator: int = sitk.sitkLinear,
    default_value: float = -1024.0,
) -> sitk.Image:
    original_spacing = np.array(image.GetSpacing(), dtype=np.float64)
    original_size = np.array(image.GetSize(), dtype=np.int64)
    target_spacing_np = np.array(target_spacing, dtype=np.float64)
    target_size = np.round(original_size * original_spacing / target_spacing_np)
    target_size = np.maximum(target_size, 1).astype(np.int64)

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(tuple(float(v) for v in target_spacing_np))
    resampler.SetSize([int(v) for v in target_size])
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    resampler.SetInterpolator(interpolator)
    resampler.SetDefaultPixelValue(float(default_value))
    resampler.SetOutputPixelType(image.GetPixelID())
    return resampler.Execute(image)


def resample_ct_to_isotropic(
    ct_image: sitk.Image,
    target_spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> sitk.Image:
    return resample_to_spacing(
        image=ct_image,
        target_spacing=target_spacing,
        interpolator=sitk.sitkLinear,
        default_value=-1024.0,
    )


def prepare_isotropic_ct(path: str | Path, *, spacing_mm: float = 1.0) -> sitk.Image:
    s = float(spacing_mm)
    if s <= 0:
        raise ValueError(f"spacing_mm must be > 0, got {spacing_mm!r}")
    return resample_ct_to_isotropic(read_nifti(path), target_spacing=(s, s, s))


def center_crop_yx_to_square(hu_zyx: np.ndarray) -> np.ndarray:
    if hu_zyx.ndim != 3:
        raise ValueError(f"Expected hu_zyx with shape [z,y,x], got {hu_zyx.shape}")
    _z, y, x = (int(hu_zyx.shape[0]), int(hu_zyx.shape[1]), int(hu_zyx.shape[2]))
    if y == x:
        return hu_zyx
    side = min(y, x)
    y0 = max(0, (y - side) // 2)
    x0 = max(0, (x - side) // 2)
    return hu_zyx[:, y0 : y0 + side, x0 : x0 + side].copy()


def center_crop_pad_yx(
    vol_zyx: np.ndarray,
    size: int,
    *,
    pad_value: float = 0.0,
) -> np.ndarray:
    """Center crop or pad each axial slice to size x size (Z unchanged)."""
    if vol_zyx.ndim != 3:
        raise ValueError(f"Expected vol_zyx with shape [z,y,x], got {vol_zyx.shape}")
    side = int(size)
    if side <= 0:
        raise ValueError(f"size must be > 0, got {size!r}")

    z, y, x = (int(vol_zyx.shape[0]), int(vol_zyx.shape[1]), int(vol_zyx.shape[2]))
    out = np.full((z, side, side), pad_value, dtype=vol_zyx.dtype)

    copy_y = min(y, side)
    copy_x = min(x, side)
    y0_src = max(0, (y - side) // 2)
    x0_src = max(0, (x - side) // 2)
    y0_dst = max(0, (side - y) // 2)
    x0_dst = max(0, (side - x) // 2)
    out[:, y0_dst : y0_dst + copy_y, x0_dst : x0_dst + copy_x] = vol_zyx[
        :, y0_src : y0_src + copy_y, x0_src : x0_src + copy_x
    ]
    return out


def center_slice_np(vol_zyx: np.ndarray, plane: str) -> np.ndarray:
    z, y, x = (int(vol_zyx.shape[0]), int(vol_zyx.shape[1]), int(vol_zyx.shape[2]))
    if plane == "axial":
        return vol_zyx[z // 2]
    if plane == "coronal":
        return vol_zyx[:, y // 2, :]
    if plane == "sagittal":
        return vol_zyx[:, :, x // 2]
    raise ValueError(f"plane must be axial|coronal|sagittal, got {plane!r}")
