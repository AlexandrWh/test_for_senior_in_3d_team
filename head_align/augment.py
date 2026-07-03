from __future__ import annotations

import numpy as np
import SimpleITK as sitk
from scipy.spatial.transform import Rotation

from utils import image_center_physical, resample_image_with_transform, rotation_matrix_to_transform, sitk_image_to_numpy

from head_align.preprocess import BOTTOM_CROP_Z, extract_bottom_slab_z
from head_align.rigid import correction_params
from head_align.volume import center_crop_yx_to_square
from utils import apply_brain_ct_window


def sample_geodesic_deg(
    rng: np.random.Generator,
    mild_max: float = 15.0,
    strong_max: float = 45.0,
    mild_prob: float = 0.85,
) -> float:
    if rng.random() < mild_prob:
        return float(rng.uniform(0.0, mild_max))
    return float(rng.uniform(mild_max, strong_max))


def sample_signed_mm(
    rng: np.random.Generator,
    mild_max: float = 20.0,
    strong_max: float = 50.0,
    mild_prob: float = 0.85,
) -> float:
    mag = sample_geodesic_deg(rng, mild_max, strong_max, mild_prob)
    sign = -1.0 if rng.random() < 0.5 else 1.0
    return sign * mag


def random_rotation(rng: np.random.Generator, *, max_angle_deg: float = 15.0) -> Rotation:
    """Uniform geodesic rotation in [0, max_angle_deg] around a random axis."""
    angle_deg = float(rng.uniform(0.0, max_angle_deg))
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis) + 1e-8
    return Rotation.from_rotvec(axis * np.deg2rad(angle_deg))


def random_translation_mm(rng: np.random.Generator) -> np.ndarray:
    return np.array(
        [sample_signed_mm(rng, 20.0, 50.0, 0.85) for _ in range(3)],
        dtype=np.float64,
    )


def mild_intensity_aug(hu: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = hu.astype(np.float32, copy=True)
    scale = float(rng.uniform(0.92, 1.08))
    shift = float(rng.uniform(-8.0, 8.0))
    out = out * scale + shift
    noise = rng.normal(0.0, 3.0, size=out.shape).astype(np.float32)
    return out + noise


def heavy_intensity_aug(hu: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = hu.astype(np.float32, copy=True)
    scale = float(rng.uniform(0.75, 1.25))
    shift = float(rng.uniform(-25.0, 25.0))
    out = out * scale + shift
    noise = rng.normal(0.0, 8.0, size=out.shape).astype(np.float32)
    return out + noise


def rigid_misalign_transform(
    rotation: Rotation,
    center_xyz: np.ndarray,
    translation_xyz: np.ndarray,
) -> sitk.Transform:
    r = rotation.as_matrix()
    c = np.asarray(center_xyz, dtype=np.float64)
    t = np.asarray(translation_xyz, dtype=np.float64)
    mat = np.eye(4, dtype=np.float64)
    mat[:3, :3] = r
    mat[:3, 3] = c - r @ c + t
    return rotation_matrix_to_transform(mat, invert=False)


def apply_misalign(ct: sitk.Image, rotation: Rotation, trans_mm: np.ndarray) -> sitk.Image:
    center = np.array(image_center_physical(ct), dtype=np.float64)
    tf = rigid_misalign_transform(rotation, center, trans_mm)
    return resample_image_with_transform(
        ct,
        tf.GetInverse(),
        interpolator=sitk.sitkLinear,
        default_value=-1024.0,
        expand_to_fit=True,
    )


def misalign_to_windowed_volume(
    ct: sitk.Image,
    rng: np.random.Generator,
    *,
    is_positive: bool,
) -> dict[str, np.ndarray | float]:
    """
    Rigid misalign + intensity aug + brain window + square Y/X + bottom Z crop.
    Output volume is pre-detector [z,y,x] in [0,1].
    """
    if is_positive:
        r_aug = random_rotation(rng, max_angle_deg=15.0)
        t_aug = np.zeros(3, dtype=np.float64)
        aug = apply_misalign(ct, r_aug, t_aug)
        hu = mild_intensity_aug(sitk_image_to_numpy(aug).astype(np.float32), rng)
        has_head = np.float32(1.0)
    else:
        r_aug = random_rotation(rng, max_angle_deg=45.0)
        t_aug = rng.uniform(-40.0, 40.0, size=3)
        aug = apply_misalign(ct, r_aug, t_aug)
        hu = heavy_intensity_aug(sitk_image_to_numpy(aug).astype(np.float32), rng)
        has_head = np.float32(0.0)

    vol = apply_brain_ct_window(hu, output_range=(0.0, 1.0))
    vol = center_crop_yx_to_square(vol)
    vol = extract_bottom_slab_z(vol, n=BOTTOM_CROP_Z)

    rotvec_corr, t_corr = correction_params(r_aug, t_aug)
    if not is_positive:
        rotvec_corr = np.zeros(3, dtype=np.float32)
        t_corr = np.zeros(3, dtype=np.float32)

    return {
        "volume_pre": vol.astype(np.float32),
        "has_head": has_head,
        "rotvec_corr_rad": rotvec_corr.astype(np.float32),
        "trans_corr_mm": t_corr.astype(np.float32),
    }
