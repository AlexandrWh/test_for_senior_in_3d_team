"""Rigid resampling for axial PCA align (center + Z-rot, one transform)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy.spatial.transform import Rotation

from utils import sitk_image_to_numpy


def vol_zyx_to_sitk(vol_zyx: np.ndarray, *, spacing_mm: float = 4.0) -> sitk.Image:
    img = sitk.GetImageFromArray(vol_zyx.astype(np.float32))
    s = float(spacing_mm)
    img.SetSpacing((s, s, s))
    img.SetOrigin((0.0, 0.0, 0.0))
    img.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
    return img


def image_center_physical(image: sitk.Image) -> tuple[float, float, float]:
    size = np.array(image.GetSize(), dtype=np.float64)
    center_index = (size - 1.0) / 2.0
    return image.TransformContinuousIndexToPhysicalPoint([float(c) for c in center_index])


def rotation_matrix_to_transform(
    matrix: np.ndarray,
    center: tuple[float, float, float] | None = None,
    invert: bool = False,
) -> sitk.Transform:
    matrix = np.asarray(matrix, dtype=np.float64)
    transform = sitk.AffineTransform(3)
    if matrix.shape == (3, 3):
        transform.SetMatrix(matrix.flatten())
        if center is not None:
            transform.SetCenter(center)
    elif matrix.shape == (4, 4):
        transform.SetMatrix(matrix[:3, :3].flatten())
        transform.SetTranslation(matrix[:3, 3].tolist())
    else:
        raise ValueError(f"Expected 3x3 or 4x4 matrix, got {matrix.shape}")
    return transform.GetInverse() if invert else transform


def resample_image_with_transform(
    image: sitk.Image,
    transform: sitk.Transform,
    *,
    default_value: float = 0.0,
    expand_to_fit: bool = False,
    interpolator: int = sitk.sitkLinear,
) -> sitk.Image:
    spacing = np.array(image.GetSpacing(), dtype=np.float64)
    direction = np.array(image.GetDirection(), dtype=np.float64).reshape(3, 3)
    out_direction = image.GetDirection()
    out_origin = image.GetOrigin()
    out_size = list(image.GetSize())

    if expand_to_fit:
        size = np.array(image.GetSize(), dtype=np.float64)
        corners_idx = [
            (x, y, z)
            for x in (0.0, size[0])
            for y in (0.0, size[1])
            for z in (0.0, size[2])
        ]
        inv = transform.GetInverse()
        phys = np.array(
            [
                inv.TransformPoint(image.TransformContinuousIndexToPhysicalPoint(idx))
                for idx in corners_idx
            ],
            dtype=np.float64,
        )
        proj = phys @ direction
        min_proj = proj.min(axis=0)
        max_proj = proj.max(axis=0)
        out_size = [
            max(int(np.ceil((max_proj[a] - min_proj[a]) / spacing[a])), 1)
            for a in range(3)
        ]
        out_origin = tuple((direction @ min_proj).tolist())

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing([float(v) for v in spacing])
    resampler.SetSize([int(v) for v in out_size])
    resampler.SetOutputDirection(out_direction)
    resampler.SetOutputOrigin(out_origin)
    resampler.SetTransform(transform)
    resampler.SetInterpolator(interpolator)
    resampler.SetDefaultPixelValue(float(default_value))
    resampler.SetOutputPixelType(image.GetPixelID())
    return resampler.Execute(image)


def build_affine_4x4(
    rotvec_rad: np.ndarray,
    trans_mm: np.ndarray,
    center_xyz: np.ndarray,
) -> np.ndarray:
    r = Rotation.from_rotvec(np.asarray(rotvec_rad, dtype=np.float64)).as_matrix()
    c = np.asarray(center_xyz, dtype=np.float64)
    t = np.asarray(trans_mm, dtype=np.float64)
    mat = np.eye(4, dtype=np.float64)
    mat[:3, :3] = r
    mat[:3, 3] = c - r @ c + t
    return mat


def voxel_shift_zyx_to_trans_mm_post_rotate(
    rotvec_rad: np.ndarray,
    shift_zyx_vox: np.ndarray,
    *,
    spacing_mm: float,
) -> np.ndarray:
    dz, dy, dx = (float(x) for x in np.asarray(shift_zyx_vox, dtype=np.float64))
    s = float(spacing_mm)
    t_shift = np.array([dx * s, dy * s, dz * s], dtype=np.float64)
    rotvec_rad = np.asarray(rotvec_rad, dtype=np.float64)
    if float(np.linalg.norm(rotvec_rad)) < 1e-8:
        return t_shift.astype(np.float32)
    return Rotation.from_rotvec(rotvec_rad).apply(t_shift).astype(np.float32)


def apply_rigid_correction(
    ct: sitk.Image,
    rotvec_rad: np.ndarray,
    trans_mm: np.ndarray,
    *,
    default_value: float = 0.0,
    expand_to_fit: bool = False,
    interpolator: int = sitk.sitkLinear,
) -> sitk.Image:
    center = np.array(image_center_physical(ct), dtype=np.float64)
    mat = build_affine_4x4(rotvec_rad, trans_mm, center)
    tf = rotation_matrix_to_transform(mat, invert=False)
    return resample_image_with_transform(
        ct,
        tf.GetInverse(),
        default_value=default_value,
        expand_to_fit=expand_to_fit,
        interpolator=interpolator,
    )


def apply_rigid_volume_zyx(
    vol_zyx: np.ndarray,
    rotvec_rad: np.ndarray,
    shift_zyx_vox: np.ndarray | tuple[float, float, float] = (0.0, 0.0, 0.0),
    *,
    spacing_mm: float = 4.0,
    expand_to_fit: bool = False,
    default_value: float = 0.0,
    interpolator: int = sitk.sitkLinear,
) -> np.ndarray:
    rotvec_rad = np.asarray(rotvec_rad, dtype=np.float32)
    shift_zyx_vox = np.asarray(shift_zyx_vox, dtype=np.float32)
    trans_mm = voxel_shift_zyx_to_trans_mm_post_rotate(
        rotvec_rad, shift_zyx_vox, spacing_mm=spacing_mm
    )
    if float(np.linalg.norm(rotvec_rad)) < 1e-7 and float(np.linalg.norm(trans_mm)) < 1e-4:
        return vol_zyx.astype(np.float32, copy=False)
    ct = vol_zyx_to_sitk(vol_zyx, spacing_mm=spacing_mm)
    ct_out = apply_rigid_correction(
        ct,
        rotvec_rad,
        trans_mm,
        default_value=default_value,
        expand_to_fit=expand_to_fit,
        interpolator=interpolator,
    )
    return sitk_image_to_numpy(ct_out).astype(np.float32)


def save_volume_nifti(vol_zyx: np.ndarray, path: Path | str, *, spacing_mm: float) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(vol_zyx_to_sitk(vol_zyx, spacing_mm=spacing_mm), str(path), useCompression=True)


def load_volume_nifti(path: Path | str) -> np.ndarray:
    return sitk_image_to_numpy(sitk.ReadImage(str(path))).astype(np.float32)


def volume_to_nifti_bytes(vol_zyx: np.ndarray, *, spacing_mm: float) -> bytes:
    """Serialize [Z,Y,X] volume to gzipped NIfTI bytes."""
    import tempfile

    img = vol_zyx_to_sitk(vol_zyx, spacing_mm=spacing_mm)
    with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        sitk.WriteImage(img, tmp_path, useCompression=True)
        return Path(tmp_path).read_bytes()
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def compute_full_align_affine_4x4(
    vol_zyx: np.ndarray,
    params: object,
    *,
    rz_pose: float,
    ry_pose: float,
    rx_pose: float,
    spacing_mm: float,
) -> list[list[float]]:
    """
  4x4 affine (homogeneous): physical point in prepared input volume -> aligned output.

  Input/output space: isotropic spacing, origin (0,0,0), identity direction (SimpleITK XYZ).
  Includes Z-crop offset and the same rigid as apply_full_align.
    """
    from models.pre_aligner import PreAlignParams
    from utils.angles import full_align_apply_euler_cw

    if not isinstance(params, PreAlignParams):
        raise TypeError(f"expected PreAlignParams, got {type(params)!r}")

    z_lo, z_hi = params.z_span_voxels(spacing_mm)
    crop_z = int(z_hi) - int(z_lo) + 1
    crop_shape = (crop_z, int(vol_zyx.shape[1]), int(vol_zyx.shape[2]))
    dx_vox, dy_vox = params.shift_xy_voxels(spacing_mm)
    rz_s, ry_s, rx_s = full_align_apply_euler_cw(params.rz, rz_pose, ry_pose, rx_pose)
    rotvec = Rotation.from_euler("ZYX", [rz_s, ry_s, rx_s]).as_rotvec()
    trans_mm = voxel_shift_zyx_to_trans_mm_post_rotate(
        rotvec, (0.0, dy_vox, dx_vox), spacing_mm=float(spacing_mm)
    )
    ct = vol_zyx_to_sitk(np.zeros(crop_shape, dtype=np.float32), spacing_mm=float(spacing_mm))
    center = np.array(image_center_physical(ct), dtype=np.float64)
    a_rigid = build_affine_4x4(rotvec, trans_mm, center)
    t_crop = np.eye(4, dtype=np.float64)
    t_crop[2, 3] = -float(z_lo) * float(spacing_mm)
    return (a_rigid @ t_crop).tolist()
