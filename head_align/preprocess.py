"""4mm preprocessing: bottom crop, detector align, cls resize."""

from __future__ import annotations

from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F

from head_align.axial_detector import (
    pca_e1_tilt_rad,
    shift_xy_to_center,
    volume_axial_pca_median_center,
)
from head_align.rigid import apply_rigid_correction, voxel_shift_zyx_to_trans_mm_post_rotate
from head_align.volume import vol_zyx_to_sitk
from utils import sitk_image_to_numpy

SPACING_MM = 4.0
BOTTOM_CROP_Z = 48  # inferior slab height in Z slices @ 4mm (= 192mm); matches OUT_SHAPE Z
CROP_Z = 48
CROP_XY = 56
OUT_SHAPE = (CROP_Z, CROP_XY, CROP_XY)


def extract_bottom_slab_z(vol_zyx: np.ndarray, *, n: int = BOTTOM_CROP_Z) -> np.ndarray:
    """Keep inferior Z slab: last n slices along Z (from bottom end of volume)."""
    z = int(vol_zyx.shape[0])
    n = int(n)
    if n <= 0 or z <= n:
        return vol_zyx.astype(np.float32, copy=False)
    return vol_zyx[-n:].astype(np.float32, copy=True)


def pad_z_min(vol_zyx: np.ndarray, *, min_z: int = CROP_Z) -> np.ndarray:
    """Symmetric zero-pad along Z when shorter than min_z (no tail-only crop)."""
    z = int(vol_zyx.shape[0])
    min_z = int(min_z)
    if z >= min_z:
        return vol_zyx.astype(np.float32, copy=False)
    need = min_z - z
    pad_before = need // 2
    pad_after = need - pad_before
    return np.pad(
        vol_zyx,
        ((pad_before, pad_after), (0, 0), (0, 0)),
        mode="constant",
        constant_values=0.0,
    ).astype(np.float32)


def center_crop_zyx(
    vol_zyx: np.ndarray,
    *,
    z_side: int = CROP_Z,
    xy_side: int = CROP_XY,
) -> np.ndarray:
    z, h, w = vol_zyx.shape
    z_side, xy_side = int(z_side), int(xy_side)
    cz, cy, cx = z // 2, h // 2, w // 2
    hz, hxy = z_side // 2, xy_side // 2
    z0, y0, x0 = cz - hz, cy - hxy, cx - hxy
    z1, y1, x1 = z0 + z_side, y0 + xy_side, x0 + xy_side

    out = np.zeros((z_side, xy_side, xy_side), dtype=np.float32)
    sz0, sy0, sx0 = max(0, z0), max(0, y0), max(0, x0)
    sz1, sy1, sx1 = min(z, z1), min(h, y1), min(w, x1)
    dz0, dy0, dx0 = sz0 - z0, sy0 - y0, sx0 - x0
    dz1, dy1, dx1 = dz0 + (sz1 - sz0), dy0 + (sy1 - sy0), dx0 + (sx1 - sx0)
    out[dz0:dz1, dy0:dy1, dx0:dx1] = vol_zyx[sz0:sz1, sy0:sy1, sx0:sx1]
    return out


def resize_trilinear(vol_zyx: np.ndarray, out_shape: tuple[int, int, int]) -> np.ndarray:
    t = torch.from_numpy(vol_zyx.astype(np.float32, copy=False)).unsqueeze(0).unsqueeze(0)
    out = F.interpolate(
        t,
        size=(int(out_shape[0]), int(out_shape[1]), int(out_shape[2])),
        mode="trilinear",
        align_corners=False,
    )
    return out[0, 0].numpy().astype(np.float32)


def apply_rigid_volume_zyx(
    vol_zyx: np.ndarray,
    rotvec_rad: np.ndarray,
    shift_zyx_vox: np.ndarray | tuple[float, float, float] = (0.0, 0.0, 0.0),
    *,
    spacing_mm: float = SPACING_MM,
    expand_to_fit: bool = False,
    default_value: float = 0.0,
) -> np.ndarray:
    """One trilinear resample: voxel shift (z,y,x) then rotation about geometric center."""
    rotvec_rad = np.asarray(rotvec_rad, dtype=np.float32)
    shift_zyx_vox = np.asarray(shift_zyx_vox, dtype=np.float32)
    trans_mm = voxel_shift_zyx_to_trans_mm_post_rotate(rotvec_rad, shift_zyx_vox, spacing_mm=spacing_mm)
    if float(np.linalg.norm(rotvec_rad)) < 1e-7 and float(np.linalg.norm(trans_mm)) < 1e-4:
        return vol_zyx.astype(np.float32, copy=False)
    ct = vol_zyx_to_sitk(vol_zyx, spacing_mm=spacing_mm)
    ct_out = apply_rigid_correction(
        ct,
        rotvec_rad,
        trans_mm,
        default_value=default_value,
        expand_to_fit=expand_to_fit,
    )
    return sitk_image_to_numpy(ct_out).astype(np.float32)


def _center_crop_offsets(
    vol_shape_zyx: tuple[int, int, int],
    out_shape: tuple[int, int, int] = OUT_SHAPE,
) -> tuple[int, int, int]:
    z, h, w = (int(vol_shape_zyx[0]), int(vol_shape_zyx[1]), int(vol_shape_zyx[2]))
    oz, oy, ox = (int(out_shape[0]), int(out_shape[1]), int(out_shape[2]))
    z0 = max(0, (z - oz) // 2)
    y0 = max(0, (h - oy) // 2)
    x0 = max(0, (w - ox) // 2)
    return z0, y0, x0


def _place_crop(
    vol_zyx: np.ndarray,
    out_shape: tuple[int, int, int],
    *,
    z0: int,
    y0: int,
    x0: int,
) -> np.ndarray:
    oz, oy, ox = out_shape
    out = np.zeros(out_shape, dtype=np.float32)
    z, h, w = vol_zyx.shape
    sz0, sy0, sx0 = max(0, z0), max(0, y0), max(0, x0)
    sz1 = min(z, z0 + oz)
    sy1 = min(h, y0 + oy)
    sx1 = min(w, x0 + ox)
    dz0, dy0, dx0 = sz0 - z0, sy0 - y0, sx0 - x0
    dz1, dy1, dx1 = dz0 + (sz1 - sz0), dy0 + (sy1 - sy0), dx0 + (sx1 - sx0)
    if sz1 > sz0 and sy1 > sy0 and sx1 > sx0:
        out[dz0:dz1, dy0:dy1, dx0:dx1] = vol_zyx[sz0:sz1, sy0:sy1, sx0:sx1]
    return out


def center_crop_to_shape(vol_zyx: np.ndarray, out_shape: tuple[int, int, int] = OUT_SHAPE) -> np.ndarray:
    z0, y0, x0 = _center_crop_offsets(vol_zyx.shape, out_shape)
    return _place_crop(vol_zyx, out_shape, z0=z0, y0=y0, x0=x0)


def random_crop_to_shape(
    vol_zyx: np.ndarray,
    out_shape: tuple[int, int, int],
    rng: np.random.Generator,
) -> np.ndarray:
    z, h, w = vol_zyx.shape
    oz, oy, ox = out_shape
    z0 = int(rng.integers(0, max(1, z - oz + 1))) if z > oz else 0
    y0 = int(rng.integers(0, max(1, h - oy + 1))) if h > oy else 0
    x0 = int(rng.integers(0, max(1, w - ox + 1))) if w > ox else 0
    return _place_crop(vol_zyx, out_shape, z0=z0, y0=y0, x0=x0)


ClsMode = Literal["resize", "center", "random"]


def cls_preprocess(
    vol_zyx: np.ndarray,
    rng: np.random.Generator,
    *,
    out_shape: tuple[int, int, int] = OUT_SHAPE,
    mode: ClsMode | None = None,
) -> np.ndarray:
    """Map pre-detector volume to fixed cls input (no detector)."""
    if mode is None:
        mode = rng.choice(["resize", "center", "random"])
    if mode == "resize":
        return resize_trilinear(vol_zyx, out_shape)
    if mode == "center":
        return center_crop_to_shape(vol_zyx, out_shape)
    return random_crop_to_shape(vol_zyx, out_shape, rng)


def apply_axial_pca_align(
    vol_zyx: np.ndarray,
    ax_det: dict | None = None,
    *,
    spacing_mm: float = SPACING_MM,
) -> tuple[np.ndarray, dict[str, object]]:
    """
    Axial PCA align: (1) shift center to frame center, (2) Z-rot by e1 tilt.

    Estimated on multi-slice axial mask around Z center; applied to full volume.
    """
    vol = vol_zyx.astype(np.float32, copy=False)
    if ax_det is None:
        ax_det = volume_axial_pca_median_center(vol)
    if ax_det is None:
        raise ValueError("volume_axial_pca_median_center failed")

    z_idx = int(vol.shape[0] // 2)
    axial = vol[z_idx]

    dx, dy = shift_xy_to_center(axial.shape, ax_det["center"])
    tilt_rad = pca_e1_tilt_rad(ax_det)
    rotz_rad = tilt_rad

    vol_out = apply_rigid_volume_zyx(
        vol,
        np.array([0.0, 0.0, rotz_rad], dtype=np.float32),
        (0.0, dy, dx),
        spacing_mm=spacing_mm,
        expand_to_fit=False,
    )

    meta: dict[str, object] = {
        "e1_tilt_rad": np.float32(tilt_rad),
        "e1_tilt_deg": np.float32(np.rad2deg(tilt_rad)),
        "rotz_detector_rad": np.float32(rotz_rad),
        "rotz_detector_deg": np.float32(np.rad2deg(rotz_rad)),
        "shift_zyx": np.array([0.0, dy, dx], dtype=np.float32),
        "center_xy": np.asarray(ax_det["center"], dtype=np.float32),
        "pca_e1": np.asarray(ax_det["e1"], dtype=np.float32),
        "pca_e2": np.asarray(ax_det["e2"], dtype=np.float32),
        "pca_n_slices": int(ax_det.get("n_slices", 1)),
        "pca_n_points": int(ax_det.get("n_points", 0)),
    }
    return vol_out, meta


def detector_crop_after_axial_pca(vol_axial: np.ndarray) -> tuple[np.ndarray | None, dict[str, object]]:
    """After axial PCA align: center crop to pose input shape."""
    vol = vol_axial.astype(np.float32, copy=False)
    if vol.ndim != 3 or min(vol.shape) < 8:
        return None, {}

    cropped = center_crop_to_shape(vol, OUT_SHAPE)
    z0, y0, x0 = _center_crop_offsets(vol.shape, OUT_SHAPE)
    meta: dict[str, object] = {
        "shift_dz": np.float32(0.0),
        "crop_z0": int(z0),
        "crop_y0": int(y0),
        "crop_x0": int(x0),
    }
    return cropped, meta


def detector_align_slab_pca_zrot(
    vol_zyx: np.ndarray,
) -> tuple[np.ndarray | None, np.ndarray | None, dict[str, object]]:
    """
    Axial PCA align on full slab + center crop for pose input.

    Returns (pose_crop, vol_axial_aligned, meta).
    """
    vol = vol_zyx.astype(np.float32, copy=False)
    if vol.ndim != 3 or min(vol.shape) < 8:
        return None, None, {}

    ax_det = volume_axial_pca_median_center(vol)
    if ax_det is None:
        return None, None, {}

    vol_axial, align_meta = apply_axial_pca_align(vol, ax_det, spacing_mm=SPACING_MM)
    shift_zyx = np.asarray(align_meta["shift_zyx"], dtype=np.float32)
    dx, dy = float(shift_zyx[2]), float(shift_zyx[1])

    cropped, crop_meta = detector_crop_after_axial_pca(vol_axial)
    if cropped is None:
        return None, None, {}

    meta: dict[str, object] = {
        "shift_zyx": np.array([float(crop_meta["shift_dz"]), dy, dx], dtype=np.float32),
        "rotz_detector_rad": align_meta["rotz_detector_rad"],
        "rotz_detector_deg": align_meta["rotz_detector_deg"],
        "center_xy": ax_det["center"].astype(np.float32),
        "pca_e1": ax_det["e1"].astype(np.float32),
        "pca_e2": ax_det["e2"].astype(np.float32),
        "crop_z0": crop_meta["crop_z0"],
        "crop_y0": crop_meta["crop_y0"],
        "crop_x0": crop_meta["crop_x0"],
        "detector_mode": "pca_zrot",
        "slab_shape_zyx": tuple(int(x) for x in vol.shape),
    }
    return cropped, vol_axial.astype(np.float32), meta
