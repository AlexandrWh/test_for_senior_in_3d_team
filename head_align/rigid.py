"""Rigid transform parameterization shared by augment, training GT, and inference."""

from __future__ import annotations

import numpy as np
import SimpleITK as sitk
import torch
from scipy.spatial.transform import Rotation

from utils import image_center_physical, resample_image_with_transform, rotation_matrix_to_transform


def correction_translation_mm(rotation_aug: Rotation, translation_aug_mm: np.ndarray) -> np.ndarray:
    """
    Translation parameter for the inverse rigid transform.

    Forward misalign uses x' = R x + (c - R c + t_aug).
    Correction uses R_corr = R^T and t_corr = -R^T t_aug (same center parameterization).
    """
    t = np.asarray(translation_aug_mm, dtype=np.float64)
    return (-rotation_aug.inv().as_matrix() @ t).astype(np.float64)


def correction_params(
    rotation_aug: Rotation,
    translation_aug_mm: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """GT pose correction as rotvec (rad) + translation mm for resampling."""
    r_corr = rotation_aug.inv()
    t_corr = correction_translation_mm(rotation_aug, translation_aug_mm)
    return r_corr.as_rotvec().astype(np.float32), t_corr.astype(np.float32)


def voxel_shift_zyx_to_trans_mm_post_rotate(
    rotvec_rad: np.ndarray,
    shift_zyx_vox: np.ndarray,
    *,
    spacing_mm: float,
) -> np.ndarray:
    """Physical translation (ITK x,y,z) for voxel shift (z,y,x) then rot about volume center."""
    dz, dy, dx = (float(x) for x in np.asarray(shift_zyx_vox, dtype=np.float64))
    s = float(spacing_mm)
    t_shift = np.array([dx * s, dy * s, dz * s], dtype=np.float64)
    rotvec_rad = np.asarray(rotvec_rad, dtype=np.float64)
    if float(np.linalg.norm(rotvec_rad)) < 1e-8:
        return t_shift.astype(np.float32)
    r = Rotation.from_rotvec(rotvec_rad)
    return r.apply(t_shift).astype(np.float32)


def volume_center_physical_zyx(shape_zyx: tuple[int, ...], *, spacing_mm: float) -> np.ndarray:
    """Physical center (x,y,z) for [z,y,x] volume with origin 0 and isotropic spacing."""
    z, h, w = (int(shape_zyx[0]), int(shape_zyx[1]), int(shape_zyx[2]))
    s = float(spacing_mm)
    return np.array([0.5 * (w - 1) * s, 0.5 * (h - 1) * s, 0.5 * (z - 1) * s], dtype=np.float64)


def crop_center_physical_zyx(
    shape_zyx: tuple[int, ...],
    crop_z0: int,
    crop_y0: int,
    crop_x0: int,
    out_shape: tuple[int, int, int],
    *,
    spacing_mm: float,
) -> np.ndarray:
    """Physical center (x,y,z) of an axis-aligned crop inside a parent [z,y,x] volume."""
    oz, oy, ox = (int(out_shape[0]), int(out_shape[1]), int(out_shape[2]))
    s = float(spacing_mm)
    return np.array(
        [
            (float(crop_x0) + 0.5 * (ox - 1)) * s,
            (float(crop_y0) + 0.5 * (oy - 1)) * s,
            (float(crop_z0) + 0.5 * (oz - 1)) * s,
        ],
        dtype=np.float64,
    )


def compute_composed_detector_pose_transform(
    det_meta: dict,
    rotvec_pose_apply: np.ndarray,
    slab_shape_zyx: tuple[int, int, int],
    *,
    spacing_mm: float,
    out_shape: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Detector shift+rotZ composed with pose rotation in index/LPS frame (identity direction).

    Matches training/inference on vol_zyx_to_sitk volumes. Must NOT be applied directly to
  oblique patient direction matrices without conversion.
    """
    shift_zyx = np.asarray(det_meta["shift_zyx"], dtype=np.float32)
    dy, dx = float(shift_zyx[1]), float(shift_zyx[2])
    rotz_det = float(det_meta["rotz_detector_rad"])
    rotvec_det = np.array([0.0, 0.0, rotz_det], dtype=np.float32)
    trans_det_mm = voxel_shift_zyx_to_trans_mm_post_rotate(
        rotvec_det,
        np.array([0.0, dy, dx], dtype=np.float32),
        spacing_mm=spacing_mm,
    )

    slab_shape = tuple(int(x) for x in det_meta.get("slab_shape_zyx", slab_shape_zyx))
    c_slab = volume_center_physical_zyx(slab_shape, spacing_mm=spacing_mm)
    c_pose = crop_center_physical_zyx(
        slab_shape,
        int(det_meta["crop_z0"]),
        int(det_meta["crop_y0"]),
        int(det_meta["crop_x0"]),
        out_shape,
        spacing_mm=spacing_mm,
    )
    rotvec_tot, trans_tot = compose_rigid_chain_about_first_center(
        rotvec_det,
        c_slab,
        trans_det_mm,
        np.asarray(rotvec_pose_apply, dtype=np.float32),
        c_pose,
        np.zeros(3, dtype=np.float32),
    )
    affine = build_affine_4x4(rotvec_tot, trans_tot, c_slab)
    return rotvec_tot.astype(np.float32), trans_tot.astype(np.float32), affine


def rigid_offset_about_center(
    rotation: Rotation,
    center_xyz: np.ndarray,
    trans_mm: np.ndarray,
) -> np.ndarray:
    """Translation part of x' = R x + offset for rotation about center_xyz."""
    c = np.asarray(center_xyz, dtype=np.float64)
    t = np.asarray(trans_mm, dtype=np.float64)
    return c - rotation.as_matrix() @ c + t


def compose_rigid_chain_about_first_center(
    rotvec_a: np.ndarray,
    center_a: np.ndarray,
    trans_a_mm: np.ndarray,
    rotvec_b: np.ndarray,
    center_b: np.ndarray,
    trans_b_mm: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compose rigid A then B as a single rigid transform about center_a.

    A: x1 = R_a x + offset_a; B: x2 = R_b x1 + offset_b (offsets use center_a / center_b).
    """
    if trans_b_mm is None:
        trans_b_mm = np.zeros(3, dtype=np.float64)
    r_a = Rotation.from_rotvec(np.asarray(rotvec_a, dtype=np.float64))
    r_b = Rotation.from_rotvec(np.asarray(rotvec_b, dtype=np.float64))
    r_tot = r_b * r_a
    off_a = rigid_offset_about_center(r_a, center_a, trans_a_mm)
    off_b = rigid_offset_about_center(r_b, center_b, trans_b_mm)
    c_a = np.asarray(center_a, dtype=np.float64)
    trans_composed = r_b.as_matrix() @ off_a + off_b - rigid_offset_about_center(
        r_tot, c_a, np.zeros(3, dtype=np.float64)
    )
    return r_tot.as_rotvec().astype(np.float32), trans_composed.astype(np.float32)


def build_affine_4x4(
    rotvec_rad: np.ndarray,
    trans_mm: np.ndarray,
    center_xyz: np.ndarray,
) -> np.ndarray:
    """4x4 world-space affine used with resample pull via GetInverse()."""
    r = Rotation.from_rotvec(np.asarray(rotvec_rad, dtype=np.float64)).as_matrix()
    c = np.asarray(center_xyz, dtype=np.float64)
    t = np.asarray(trans_mm, dtype=np.float64)
    mat = np.eye(4, dtype=np.float64)
    mat[:3, :3] = r
    mat[:3, 3] = c - r @ c + t
    return mat


def apply_rigid_correction(
    ct: sitk.Image,
    rotvec_rad: np.ndarray,
    trans_mm: np.ndarray,
    *,
    interpolator: int = sitk.sitkLinear,
    default_value: float = -1024.0,
    expand_to_fit: bool = True,
) -> sitk.Image:
    center = np.array(image_center_physical(ct), dtype=np.float64)
    mat = build_affine_4x4(rotvec_rad, trans_mm, center)
    tf = rotation_matrix_to_transform(mat, invert=False)
    return resample_image_with_transform(
        ct,
        tf.GetInverse(),
        interpolator=interpolator,
        default_value=default_value,
        expand_to_fit=expand_to_fit,
    )


def _skew(v: torch.Tensor) -> torch.Tensor:
    vx, vy, vz = v[..., 0], v[..., 1], v[..., 2]
    o = torch.zeros_like(vx)
    return torch.stack(
        [
            torch.stack([o, -vz, vy], dim=-1),
            torch.stack([vz, o, -vx], dim=-1),
            torch.stack([-vy, vx, o], dim=-1),
        ],
        dim=-2,
    )


def rotvec_to_matrix(rotvec: torch.Tensor) -> torch.Tensor:
    """Differentiable SO(3) exponential map, rotvec shape [..., 3]."""
    theta = torch.linalg.norm(rotvec, dim=-1, keepdim=True).clamp(min=1e-8)
    k = rotvec / theta
    kx = _skew(k)
    eye = torch.eye(3, device=rotvec.device, dtype=rotvec.dtype)
    eye = eye.view(*([1] * (rotvec.dim() - 1)), 3, 3).expand(*rotvec.shape[:-1], 3, 3)
    sin_t = torch.sin(theta)[..., None]
    cos_t = torch.cos(theta)[..., None]
    return eye + sin_t * kx + (1.0 - cos_t) * (kx @ kx)


def geodesic_rot_loss(rotvec_pred: torch.Tensor, rotvec_gt: torch.Tensor) -> torch.Tensor:
    """Mean geodesic angle (rad) between predicted and GT rotations."""
    rp = rotvec_to_matrix(rotvec_pred)
    rg = rotvec_to_matrix(rotvec_gt)
    trace = (rp.transpose(-1, -2) @ rg).diagonal(dim1=-2, dim2=-1).sum(-1)
    cos_angle = ((trace - 1.0) * 0.5).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
    return torch.acos(cos_angle).mean()


def geodesic_deg_np(rotvec_pred: np.ndarray, rotvec_gt: np.ndarray) -> float:
    rp = Rotation.from_rotvec(rotvec_pred)
    rg = Rotation.from_rotvec(rotvec_gt)
    return float(np.rad2deg(np.linalg.norm((rp.inv() * rg).as_rotvec())))
