from __future__ import annotations

from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch

from head_align.model import FullScanClsNet, FullScanPoseNet
from head_align.preprocess import (
    BOTTOM_CROP_Z,
    OUT_SHAPE,
    SPACING_MM,
    center_crop_zyx,
    detector_align_slab_pca_zrot,
    extract_bottom_slab_z,
    pad_z_min,
    resize_trilinear,
)
from head_align.rigid import apply_rigid_correction, compute_composed_detector_pose_transform
from head_align.volume import center_crop_yx_to_square, vol_zyx_to_sitk
from utils import apply_brain_ct_window, sitk_image_to_numpy


def load_pose_model(checkpoint: Path, device: torch.device) -> FullScanPoseNet:
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    model = FullScanPoseNet(base_channels=int(ckpt.get("base_channels", 16)))
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    return model


def load_cls_model(checkpoint: Path, device: torch.device) -> FullScanClsNet:
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    model = FullScanClsNet(base_channels=int(ckpt.get("base_channels", 16)))
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    return model


def prepare_pre_detector_volume(ct: sitk.Image) -> np.ndarray:
    """4mm pipeline step before cls/detector: window + square Y/X + bottom Z crop."""
    hu = sitk_image_to_numpy(ct).astype(np.float32)
    vol = apply_brain_ct_window(hu, output_range=(0.0, 1.0))
    vol = center_crop_yx_to_square(vol)
    return extract_bottom_slab_z(vol, n=BOTTOM_CROP_Z)


def center_slice_np(vol_zyx: np.ndarray, plane: str) -> np.ndarray:
    z, y, x = (int(vol_zyx.shape[0]), int(vol_zyx.shape[1]), int(vol_zyx.shape[2]))
    if plane == "axial":
        return vol_zyx[z // 2]
    if plane == "coronal":
        return vol_zyx[:, y // 2, :]
    if plane == "sagittal":
        return vol_zyx[:, :, x // 2]
    raise ValueError(f"plane must be axial|coronal|sagittal, got {plane!r}")


def finalize_rotated_crop(vol_rot_zyx: np.ndarray, *, out_shape: tuple[int, int, int] = OUT_SHAPE) -> np.ndarray:
    """Center crop to out_shape; symmetric Z pad if still shorter than target Z."""
    oz, oy, ox = out_shape
    cropped = center_crop_zyx(vol_rot_zyx, z_side=oz, xy_side=oy)
    if cropped.shape[0] < oz:
        cropped = pad_z_min(cropped, min_z=oz)
    return cropped.astype(np.float32)


def rotvec_pred_to_apply(rotvec_pred: np.ndarray) -> np.ndarray:
    """Model predicts misalignment rotvec; apply inverse rotation (= negation)."""
    return (-np.asarray(rotvec_pred, dtype=np.float32)).copy()


def rotvec_angle_deg(rotvec: np.ndarray) -> float:
    """Geodesic angle (deg) of rotation represented by rotvec (rad)."""
    return float(np.rad2deg(np.linalg.norm(np.asarray(rotvec, dtype=np.float64))))


@torch.no_grad()
def predict_pose_volume(
    pose_model: FullScanPoseNet,
    vol_zyx: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    x = torch.from_numpy(vol_zyx.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
    rotvec, trans = pose_model(x)
    return (
        rotvec.squeeze(0).cpu().numpy().astype(np.float32),
        trans.squeeze(0).cpu().numpy().astype(np.float32),
    )


def apply_composed_detector_pose_crop(
    vol_slab: np.ndarray,
    det_meta: dict[str, object],
    rotvec_pose_apply: np.ndarray,
    *,
    spacing_mm: float = SPACING_MM,
    out_shape: tuple[int, int, int] = OUT_SHAPE,
) -> np.ndarray:
    """One resample from pre-detector slab: detector shift+rotZ composed with pose rotation."""
    rotvec_tot, trans_tot, _affine = compute_composed_detector_pose_transform(
        det_meta,
        rotvec_pose_apply,
        tuple(int(x) for x in vol_slab.shape),
        spacing_mm=spacing_mm,
        out_shape=out_shape,
    )
    ct = vol_zyx_to_sitk(vol_slab.astype(np.float32), spacing_mm=spacing_mm)
    ct_out = apply_rigid_correction(
        ct,
        rotvec_tot,
        trans_tot,
        default_value=0.0,
        expand_to_fit=False,
    )
    return finalize_rotated_crop(sitk_image_to_numpy(ct_out).astype(np.float32), out_shape=out_shape)


def single_pose_align_volume(
    pose_model: FullScanPoseNet,
    vol_zyx: np.ndarray,
    device: torch.device,
    *,
    spacing_mm: float = SPACING_MM,
    vol_slab: np.ndarray | None = None,
    det_meta: dict[str, object] | None = None,
) -> dict:
    """One pose forward pass; optional composed resample from original slab."""
    vol_in = vol_zyx.astype(np.float32, copy=False)
    rotvec_pred, _trans = predict_pose_volume(pose_model, vol_in, device)
    angle_deg = rotvec_angle_deg(rotvec_pred)
    rotvec_apply = rotvec_pred_to_apply(rotvec_pred)

    if vol_slab is not None and det_meta is not None:
        vol_aligned = apply_composed_detector_pose_crop(
            vol_slab,
            det_meta,
            rotvec_apply,
            spacing_mm=spacing_mm,
            out_shape=OUT_SHAPE,
        )
        ct_after = vol_zyx_to_sitk(vol_aligned, spacing_mm=spacing_mm)
    else:
        ct_crop = vol_zyx_to_sitk(vol_in, spacing_mm=spacing_mm)
        ct_after = apply_rigid_correction(
            ct_crop,
            rotvec_apply,
            np.zeros(3, dtype=np.float32),
            default_value=0.0,
            expand_to_fit=False,
        )
        vol_aligned = finalize_rotated_crop(sitk_image_to_numpy(ct_after).astype(np.float32))

    return {
        "ct_after": ct_after,
        "vol_aligned": vol_aligned,
        "geodesic_deg": angle_deg,
        "rotvec_pred_rad": rotvec_pred.copy(),
        "rotvec_apply_total_rad": rotvec_apply.copy(),
    }


@torch.no_grad()
def infer_case_hybrid(
    pose_model: FullScanPoseNet,
    cls_model: FullScanClsNet,
    ct: sitk.Image,
    device: torch.device,
    *,
    cls_threshold: float = 0.5,
    spacing_mm: float = SPACING_MM,
) -> dict:
    """Production pipeline @ 4mm: bottom slab -> cls -> axial PCA + crop -> pose 1-pass."""
    pre_det = prepare_pre_detector_volume(ct)
    cls_vol = resize_trilinear(pre_det, OUT_SHAPE)

    x_cls = torch.from_numpy(cls_vol).unsqueeze(0).unsqueeze(0).to(device)
    cls_prob = float(torch.sigmoid(cls_model(x_cls)).item())

    out: dict = {
        "has_head": False,
        "cls_prob": cls_prob,
        "detector_ok": False,
        "pipeline": "hybrid_pca_pose",
        "geodesic_deg": 0.0,
    }
    if cls_prob < cls_threshold:
        return out

    pose_in, _vol_axial, det_meta = detector_align_slab_pca_zrot(pre_det)
    if pose_in is None:
        return out

    aligned = single_pose_align_volume(
        pose_model,
        pose_in,
        device,
        spacing_mm=spacing_mm,
        vol_slab=pre_det,
        det_meta=det_meta,
    )

    out.update(
        {
            "has_head": True,
            "detector_ok": True,
            "ct_after": aligned["ct_after"],
            "vol_aligned": aligned["vol_aligned"],
            "vol_detector_crop": pose_in.astype(np.float32),
            "det_meta": det_meta,
            "detector_rotz_deg": float(det_meta.get("rotz_detector_deg", 0.0)),
            "shift_zyx": det_meta.get("shift_zyx"),
            "rotvec_pred_rad": aligned["rotvec_pred_rad"],
            "rotvec_corr_rad": aligned["rotvec_apply_total_rad"],
            "geodesic_deg": aligned["geodesic_deg"],
        }
    )
    return out
