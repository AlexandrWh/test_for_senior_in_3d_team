from __future__ import annotations

from pathlib import Path

import numpy as np
import SimpleITK as sitk

from utils import read_nifti, resample_ct_to_isotropic, sitk_image_to_numpy


def prepare_isotropic_ct(path: str | Path, *, spacing_mm: float = 1.0) -> sitk.Image:
    s = float(spacing_mm)
    if s <= 0:
        raise ValueError(f"spacing_mm must be > 0, got {spacing_mm!r}")
    return resample_ct_to_isotropic(read_nifti(path), target_spacing=(s, s, s))


def center_crop_yx_to_square(hu_zyx: np.ndarray) -> np.ndarray:
    """Make Y/X square by center-cropping the larger axis ([z, y, x])."""
    if hu_zyx.ndim != 3:
        raise ValueError(f"Expected hu_zyx with shape [z,y,x], got {hu_zyx.shape}")
    _z, y, x = (int(hu_zyx.shape[0]), int(hu_zyx.shape[1]), int(hu_zyx.shape[2]))
    if y == x:
        return hu_zyx
    side = min(y, x)
    y0 = max(0, (y - side) // 2)
    x0 = max(0, (x - side) // 2)
    return hu_zyx[:, y0 : y0 + side, x0 : x0 + side].copy()


def array_to_sitk_roi(
    hu: np.ndarray,
    ref: sitk.Image,
    *,
    x_index: int,
    y_index: int,
    z_index: int,
) -> sitk.Image:
    """Build sitk image from [z,y,x] HU crop; index offsets are in ref volume (ITK x,y,z)."""
    out = sitk.GetImageFromArray(hu.astype(np.float32))
    spacing = np.array(ref.GetSpacing(), dtype=np.float64)
    direction = np.array(ref.GetDirection(), dtype=np.float64).reshape(3, 3)
    origin = np.array(ref.GetOrigin(), dtype=np.float64)
    index_shift = np.array([float(x_index), float(y_index), float(z_index)], dtype=np.float64)
    origin = origin + direction @ (index_shift * spacing)
    out.SetSpacing([float(v) for v in spacing])
    out.SetDirection([float(v) for v in ref.GetDirection()])
    out.SetOrigin(tuple(origin.tolist()))
    return out


def square_crop_offsets_zyx(shape_zyx: tuple[int, int, int]) -> tuple[int, int, int]:
    """Return y0, x0, side for center square crop on Y/X."""
    _z, h, w = (int(shape_zyx[0]), int(shape_zyx[1]), int(shape_zyx[2]))
    side = min(h, w)
    y0 = max(0, (h - side) // 2)
    x0 = max(0, (w - side) // 2)
    return y0, x0, side


def extract_infer_slab_hu(
    hu_zyx: np.ndarray,
    *,
    bottom_n: int,
) -> tuple[np.ndarray, dict[str, int]]:
    """Infer slab [z,y,x] HU (same geometry as prepare_pre_detector_volume, without window)."""
    y0, x0, side = square_crop_offsets_zyx(hu_zyx.shape)
    vol_sq = hu_zyx[:, y0 : y0 + side, x0 : x0 + side]
    z_len = int(vol_sq.shape[0])
    n = min(int(bottom_n), z_len)
    z0 = max(0, z_len - n)
    slab = vol_sq[z0 : z0 + n].astype(np.float32, copy=True)
    return slab, {"x_index": x0, "y_index": y0, "z_index": z0, "side": side}


def extract_infer_slab_ct(ct: sitk.Image, *, bottom_n: int) -> sitk.Image:
    """Bottom infer slab as sitk subvolume with parent CT geometry."""
    hu = sitk_image_to_numpy(ct).astype(np.float32)
    slab, off = extract_infer_slab_hu(hu, bottom_n=bottom_n)
    return array_to_sitk_roi(
        slab,
        ct,
        x_index=off["x_index"],
        y_index=off["y_index"],
        z_index=off["z_index"],
    )


def vol_zyx_to_sitk(vol_zyx: np.ndarray, *, spacing_mm: float = 4.0) -> sitk.Image:
    img = sitk.GetImageFromArray(vol_zyx.astype(np.float32))
    s = float(spacing_mm)
    img.SetSpacing((s, s, s))
    img.SetOrigin((0.0, 0.0, 0.0))
    img.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
    return img
