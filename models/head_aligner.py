"""HeadAligner: Z-cls + PCA @ 4 mm infer, pose angles, final apply @ 1 mm."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
from scipy.spatial.transform import Rotation

from models.pose_regressor import PoseRegressor3D
from models.pre_aligner import PreAlignParams, PreAligner, crop_z_span
from paths import (
    APPLY_SPACING_MM,
    DEFAULT_POSE_REGRESSOR_CKPT,
    DEFAULT_PRE_ALIGNER_CKPT,
    POSE_INPUT_YX,
    SPACING_MM,
)
from utils import center_crop_pad_yx
from utils.angles import full_align_apply_euler_cw, pose_apply_euler_cw, prealign_apply_euler_cw
from utils.rigid import apply_rigid_volume_zyx


def align_volume(
    vol_zyx: np.ndarray,
    params: PreAlignParams,
    *,
    spacing_mm: float = SPACING_MM,
) -> tuple[np.ndarray, dict[str, object]]:
    """Z-crop + shift + PCA Rz @ spacing_mm (4 mm pose-regressor input)."""
    vol = vol_zyx.astype(np.float32, copy=False)

    if not params.has_head:
        return vol[:0].copy(), {"ok": False, "has_head": False, "reason": "no_head"}

    z_lo, z_hi = params.z_span_voxels(spacing_mm)
    cropped = crop_z_span(vol, z_lo, z_hi)
    if cropped.shape[0] == 0:
        return vol[:0].copy(), {
            "ok": False,
            "has_head": True,
            "reason": "empty_z_crop",
            "z_min": params.z_min,
            "z_max": params.z_max,
        }

    dx_vox, dy_vox = params.shift_xy_voxels(spacing_mm)
    rz_s, ry_s, rx_s = prealign_apply_euler_cw(params.rz)
    rotvec = Rotation.from_euler("ZYX", [rz_s, ry_s, rx_s]).as_rotvec()
    vol_out = apply_rigid_volume_zyx(
        cropped,
        rotvec.astype(np.float32),
        (0.0, dy_vox, dx_vox),
        spacing_mm=spacing_mm,
        expand_to_fit=False,
    )
    meta: dict[str, object] = {
        "ok": True,
        "has_head": True,
        "spacing_mm": float(spacing_mm),
        "z_min": float(params.z_min),
        "z_max": float(params.z_max),
        "dx": float(params.dx),
        "dy": float(params.dy),
        "rz_pca": float(params.rz),
        "crop_shape": list(cropped.shape),
        "out_shape": list(vol_out.shape),
    }
    return vol_out, meta


def apply_pose_volume(
    vol_zyx: np.ndarray,
    params: PreAlignParams,
    *,
    rz_aug: float,
    ry_aug: float,
    rx_aug: float,
    spacing_mm: float = SPACING_MM,
) -> tuple[np.ndarray, dict[str, object]]:
    """PreAlign + aug rotation for pose dataset generation @ spacing_mm."""
    vol = vol_zyx.astype(np.float32, copy=False)

    if not params.has_head:
        return vol[:0].copy(), {"ok": False, "has_head": False, "reason": "no_head"}

    z_lo, z_hi = params.z_span_voxels(spacing_mm)
    cropped = crop_z_span(vol, z_lo, z_hi)
    if cropped.shape[0] == 0:
        return vol[:0].copy(), {
            "ok": False,
            "has_head": True,
            "reason": "empty_z_crop",
            "z_min": params.z_min,
            "z_max": params.z_max,
        }

    rz_s, ry_s, rx_s = pose_apply_euler_cw(params.rz, rz_aug, ry_aug, rx_aug)
    rotvec = Rotation.from_euler("ZYX", [rz_s, ry_s, rx_s]).as_rotvec()
    dx_vox, dy_vox = params.shift_xy_voxels(spacing_mm)
    vol_out = apply_rigid_volume_zyx(
        cropped,
        rotvec.astype(np.float32),
        (0.0, dy_vox, dx_vox),
        spacing_mm=spacing_mm,
        expand_to_fit=False,
    )
    meta: dict[str, object] = {
        "ok": True,
        "has_head": True,
        "spacing_mm": float(spacing_mm),
        "rz_pca": float(params.rz),
        "rz_aug": float(rz_aug),
        "ry_aug": float(ry_aug),
        "rx_aug": float(rx_aug),
        "out_shape": list(vol_out.shape),
    }
    return vol_out, meta


def apply_full_align(
    vol_zyx: np.ndarray,
    params: PreAlignParams,
    *,
    rz_pose: float,
    ry_pose: float,
    rx_pose: float,
    spacing_mm: float = APPLY_SPACING_MM,
    interpolator: int | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    """Z-crop + shift + combined rotation on isotropic volume (final 1 mm output)."""
    vol = vol_zyx.astype(np.float32, copy=False)

    if not params.has_head:
        return vol[:0].copy(), {"ok": False, "has_head": False, "reason": "no_head"}

    z_lo, z_hi = params.z_span_voxels(spacing_mm)
    cropped = crop_z_span(vol, z_lo, z_hi)
    if cropped.shape[0] == 0:
        return vol[:0].copy(), {
            "ok": False,
            "has_head": True,
            "reason": "empty_z_crop",
        }

    dx_vox, dy_vox = params.shift_xy_voxels(spacing_mm)
    rz_s, ry_s, rx_s = full_align_apply_euler_cw(params.rz, rz_pose, ry_pose, rx_pose)
    rotvec = Rotation.from_euler("ZYX", [rz_s, ry_s, rx_s]).as_rotvec()
    interp = sitk.sitkLinear if interpolator is None else interpolator
    vol_out = apply_rigid_volume_zyx(
        cropped,
        rotvec.astype(np.float32),
        (0.0, dy_vox, dx_vox),
        spacing_mm=spacing_mm,
        expand_to_fit=False,
        default_value=0.0,
        interpolator=interp,
    )
    meta: dict[str, object] = {
        "ok": True,
        "has_head": True,
        "spacing_mm": float(spacing_mm),
        "apply_euler_zyx_scipy_rad": [rz_s, ry_s, rx_s],
        "crop_shape": list(cropped.shape),
        "out_shape": list(vol_out.shape),
    }
    return vol_out, meta


@dataclass
class AlignResult:
    case_id: str = ""
    has_head: bool = False
    infer_spacing_mm: float = SPACING_MM
    output_spacing_mm: float = APPLY_SPACING_MM
    z_min: float = -1.0
    z_max: float = -1.0
    dx: float = 0.0
    dy: float = 0.0
    rz_pca_rad: float = 0.0
    rz_pose_rad: float = 0.0
    ry_pose_rad: float = 0.0
    rx_pose_rad: float = 0.0
    volume_aligned_1mm: np.ndarray | None = None
    status: str = "error"
    reason: str = ""
    meta: dict = field(default_factory=dict)

    def to_json_dict(self) -> dict:
        def _deg(r: float) -> float:
            return float(math.degrees(r))

        return {
            "case_id": self.case_id,
            "has_head": self.has_head,
            "status": self.status,
            "infer_spacing_mm": self.infer_spacing_mm,
            "output_spacing_mm": self.output_spacing_mm,
            "prealign": {
                "z_min": self.z_min,
                "z_max": self.z_max,
                "dx": self.dx,
                "dy": self.dy,
                "rz_pca_rad": self.rz_pca_rad,
                "rz_pca_deg": _deg(self.rz_pca_rad),
            },
            "pose": {
                "rz_rad": self.rz_pose_rad,
                "ry_rad": self.ry_pose_rad,
                "rx_rad": self.rx_pose_rad,
                "rz_deg": _deg(self.rz_pose_rad),
                "ry_deg": _deg(self.ry_pose_rad),
                "rx_deg": _deg(self.rx_pose_rad),
            },
            **self.meta,
        }


class HeadAligner:
    """Infer @ 4 mm (cls + PCA + pose), apply final head @ 1 mm."""

    def __init__(
        self,
        pre_aligner: PreAligner,
        pose_regressor: PoseRegressor3D,
        *,
        infer_spacing_mm: float = SPACING_MM,
        output_spacing_mm: float = APPLY_SPACING_MM,
        pose_input_yx: int = POSE_INPUT_YX,
    ):
        self.pre_aligner = pre_aligner
        self.pose_regressor = pose_regressor
        self.infer_spacing_mm = float(infer_spacing_mm)
        self.output_spacing_mm = float(output_spacing_mm)
        self.pose_input_yx = int(pose_input_yx)

    @classmethod
    def from_checkpoints(
        cls,
        pre_align_ckpt: str | Path = DEFAULT_PRE_ALIGNER_CKPT,
        pose_ckpt: str | Path = DEFAULT_POSE_REGRESSOR_CKPT,
        device: torch.device | str = "cpu",
        *,
        cls_threshold: float = 0.5,
        cls_pad: int = 3,
        cls_min_head_slices: int = 10,
        infer_spacing_mm: float = SPACING_MM,
        output_spacing_mm: float = APPLY_SPACING_MM,
        pose_input_yx: int = POSE_INPUT_YX,
    ) -> HeadAligner:
        dev = torch.device(device)
        pre = PreAligner.from_checkpoint(
            pre_align_ckpt,
            dev,
            cls_threshold=cls_threshold,
            cls_pad=cls_pad,
            cls_min_head_slices=cls_min_head_slices,
        )
        pose = PoseRegressor3D.from_checkpoint(pose_ckpt, dev)
        pose.eval()
        return cls(
            pre,
            pose,
            infer_spacing_mm=infer_spacing_mm,
            output_spacing_mm=output_spacing_mm,
            pose_input_yx=pose_input_yx,
        )

    @torch.no_grad()
    def predict_pose_angles(
        self,
        vol_pre_align_4mm: np.ndarray,
        device: torch.device | None = None,
    ) -> tuple[float, float, float]:
        dev = device if device is not None else next(self.pose_regressor.parameters()).device
        vol = center_crop_pad_yx(vol_pre_align_4mm, self.pose_input_yx, pad_value=0.0)
        x = torch.from_numpy(vol).unsqueeze(0).unsqueeze(0).to(dev)
        z_len = torch.tensor([int(vol.shape[0])], dtype=torch.int64, device=dev)
        pred = self.pose_regressor(x, z_len=z_len)[0].detach().cpu().numpy()
        return float(pred[0]), float(pred[1]), float(pred[2])

    @torch.no_grad()
    def align(
        self,
        path: str | Path,
        *,
        device: torch.device | None = None,
        case_id: str = "",
    ) -> AlignResult:
        path = Path(path)
        case_id = case_id or path.name.replace(".nii.gz", "")
        dev = device if device is not None else next(self.pre_aligner.parameters()).device

        vol_4 = PreAligner.prepare_volume(path, spacing_mm=self.infer_spacing_mm)
        params = self.pre_aligner.predict_params(vol_4, device=dev)

        if not params.has_head:
            return AlignResult(case_id=case_id, has_head=False, status="no_head", reason="no_head")

        vol_pre, pre_meta = align_volume(vol_4, params, spacing_mm=self.infer_spacing_mm)
        if not pre_meta.get("ok") or vol_pre.size == 0:
            return AlignResult(
                case_id=case_id,
                has_head=True,
                status="pre_align_fail",
                reason=str(pre_meta.get("reason", "pre_align_fail")),
                z_min=params.z_min,
                z_max=params.z_max,
            )

        rz_pose, ry_pose, rx_pose = self.predict_pose_angles(vol_pre, device=dev)

        vol_1 = PreAligner.prepare_volume(path, spacing_mm=self.output_spacing_mm)
        vol_out, apply_meta = apply_full_align(
            vol_1,
            params,
            rz_pose=rz_pose,
            ry_pose=ry_pose,
            rx_pose=rx_pose,
            spacing_mm=self.output_spacing_mm,
        )
        if not apply_meta.get("ok") or vol_out.size == 0:
            return AlignResult(
                case_id=case_id,
                has_head=True,
                status="align_fail",
                reason=str(apply_meta.get("reason", "align_fail")),
                z_min=params.z_min,
                z_max=params.z_max,
                dx=params.dx,
                dy=params.dy,
                rz_pca_rad=params.rz,
                rz_pose_rad=rz_pose,
                ry_pose_rad=ry_pose,
                rx_pose_rad=rx_pose,
            )

        return AlignResult(
            case_id=case_id,
            has_head=True,
            status="ok",
            z_min=params.z_min,
            z_max=params.z_max,
            dx=params.dx,
            dy=params.dy,
            rz_pca_rad=params.rz,
            rz_pose_rad=rz_pose,
            ry_pose_rad=ry_pose,
            rx_pose_rad=rx_pose,
            volume_aligned_1mm=vol_out,
            meta=dict(apply_meta),
        )
