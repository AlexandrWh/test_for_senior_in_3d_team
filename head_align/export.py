"""Export aligned head CT (HU sitk) by applying composed rigid transform once."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import SimpleITK as sitk
import torch

from head_align.inference import (
    SPACING_MM,
    infer_case_hybrid,
    load_cls_model,
    load_pose_model,
)
from head_align.axial_detector import AXIAL_PCA_N_SLICES_UP, AXIAL_PCA_Z_STEP
from head_align.mask_utils import THR_01
from head_align.preprocess import BOTTOM_CROP_Z, OUT_SHAPE
from head_align.rigid import apply_rigid_correction, compute_composed_detector_pose_transform
from head_align.volume import extract_infer_slab_ct, vol_zyx_to_sitk
from utils import read_nifti, resample_ct_to_isotropic, sitk_image_to_numpy


def pipeline_meta(*, spacing_mm: float = SPACING_MM) -> dict[str, Any]:
    """Static pipeline descriptor for API /health and align meta."""
    return {
        "pipeline": "hybrid_pca_pose",
        "spacing_mm": float(spacing_mm),
        "bottom_crop_z": int(BOTTOM_CROP_Z),
        "out_shape_zyx": [int(x) for x in OUT_SHAPE],
        "mask_thr_window": float(THR_01 * 255.0),
        "axial_pca_n_slices_up": int(AXIAL_PCA_N_SLICES_UP),
        "axial_pca_z_step": int(AXIAL_PCA_Z_STEP),
        "export_frame": "identity_index",
        "expand_to_fit": False,
    }


@dataclass
class AlignExportResult:
    has_head: bool
    cls_prob: float
    detector_ok: bool
    geodesic_deg: float
    detector_rotz_deg: float
    rotvec_corr_rad: list[float]
    affine_4x4: list[list[float]] | None
    ct_input: sitk.Image
    ct_aligned: sitk.Image | None
    message: str = ""


def align_slab_hu_identity(
    slab_hu_zyx: np.ndarray,
    det_meta: dict[str, Any],
    rotvec_pose_apply: np.ndarray,
    *,
    spacing_mm: float = SPACING_MM,
) -> tuple[sitk.Image, np.ndarray]:
    """
    Apply detector+pose in index frame (identity direction), matching training/inference.

    Patient-specific direction from DICOM/NIfTI is NOT used here on purpose: models and
    detector rotZ are defined in axial index coordinates of the resampled slab.
    """
    slab_hu_zyx = slab_hu_zyx.astype(np.float32, copy=False)
    rotvec_tot, trans_tot, affine = compute_composed_detector_pose_transform(
        det_meta,
        rotvec_pose_apply,
        tuple(int(x) for x in slab_hu_zyx.shape),
        spacing_mm=spacing_mm,
        out_shape=OUT_SHAPE,
    )
    ct_identity = vol_zyx_to_sitk(slab_hu_zyx, spacing_mm=spacing_mm)
    ct_aligned = apply_rigid_correction(
        ct_identity,
        rotvec_tot,
        trans_tot,
        default_value=-1024.0,
        expand_to_fit=False,
    )
    return ct_aligned, affine


def align_slab_ct(
    slab_ct: sitk.Image,
    det_meta: dict[str, Any],
    rotvec_pose_apply: np.ndarray,
    *,
    spacing_mm: float = SPACING_MM,
) -> tuple[sitk.Image, np.ndarray]:
    """One HU resample on infer slab (identity frame transform)."""
    slab_hu = sitk_image_to_numpy(slab_ct).astype(np.float32)
    return align_slab_hu_identity(
        slab_hu,
        det_meta,
        rotvec_pose_apply,
        spacing_mm=spacing_mm,
    )


def align_head_from_infer(
    ct_iso: sitk.Image,
    infer: dict[str, Any],
    *,
    spacing_mm: float = SPACING_MM,
) -> tuple[sitk.Image | None, np.ndarray | None]:
    """Build aligned head slab sitk from inference dict."""
    if not infer.get("has_head") or not infer.get("detector_ok"):
        return None, None

    det_meta = infer.get("det_meta")
    rotvec_corr = infer.get("rotvec_corr_rad")
    if det_meta is None or rotvec_corr is None:
        return None, None

    slab_ct = extract_infer_slab_ct(ct_iso, bottom_n=BOTTOM_CROP_Z)
    ct_aligned, affine = align_slab_ct(
        slab_ct,
        det_meta,
        np.asarray(rotvec_corr, dtype=np.float32),
        spacing_mm=spacing_mm,
    )
    return ct_aligned, affine


def _affine_to_json(affine: np.ndarray) -> list[list[float]]:
    return np.asarray(affine, dtype=np.float64).tolist()


@torch.no_grad()
def align_head_scan(
    ct: sitk.Image,
    cls_model,
    pose_model,
    device: torch.device,
    *,
    spacing_mm: float = SPACING_MM,
    cls_threshold: float = 0.5,
) -> AlignExportResult:
    """
    Full service path: isotropic CT -> infer -> one HU resample on infer slab.

    `ct` may be native or already isotropic; if spacing differs from spacing_mm it is resampled.
    """
    spacing = tuple(float(s) for s in ct.GetSpacing())
    if not all(abs(s - float(spacing_mm)) < 1e-3 for s in spacing):
        ct_iso = resample_ct_to_isotropic(ct, target_spacing=(spacing_mm, spacing_mm, spacing_mm))
    else:
        ct_iso = ct

    infer = infer_case_hybrid(
        pose_model,
        cls_model,
        ct_iso,
        device,
        cls_threshold=float(cls_threshold),
        spacing_mm=float(spacing_mm),
    )

    base = AlignExportResult(
        has_head=bool(infer.get("has_head", False)),
        cls_prob=float(infer.get("cls_prob", 0.0)),
        detector_ok=bool(infer.get("detector_ok", False)),
        geodesic_deg=float(infer.get("geodesic_deg", 0.0)),
        detector_rotz_deg=float(infer.get("det_meta", {}).get("rotz_detector_deg", 0.0))
        if infer.get("det_meta")
        else 0.0,
        rotvec_corr_rad=np.asarray(infer.get("rotvec_corr_rad", np.zeros(3)), dtype=np.float32).tolist(),
        affine_4x4=None,
        ct_input=ct_iso,
        ct_aligned=None,
    )

    if not base.has_head:
        base.message = "no head detected (cls)"
        return base
    if not base.detector_ok:
        base.message = "detector failed"
        return base

    ct_aligned, affine = align_head_from_infer(ct_iso, infer, spacing_mm=spacing_mm)
    if ct_aligned is None or affine is None:
        base.message = "alignment export failed"
        return base

    base.ct_aligned = ct_aligned
    base.affine_4x4 = _affine_to_json(affine)
    base.message = "ok"
    return base


def align_head_from_path(
    path: str | Path,
    cls_model,
    pose_model,
    device: torch.device,
    *,
    spacing_mm: float = SPACING_MM,
    cls_threshold: float = 0.5,
) -> AlignExportResult:
    ct = read_nifti(path)
    return align_head_scan(
        ct,
        cls_model,
        pose_model,
        device,
        spacing_mm=spacing_mm,
        cls_threshold=cls_threshold,
    )


def load_service_models(
    cls_checkpoint: Path,
    pose_checkpoint: Path,
    device: torch.device | str = "auto",
) -> tuple[Any, Any, torch.device]:
    if device == "auto":
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(device)
    cls_model = load_cls_model(Path(cls_checkpoint), dev)
    pose_model = load_pose_model(Path(pose_checkpoint), dev)
    return cls_model, pose_model, dev


def write_nifti(ct: sitk.Image, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(ct, str(path), useCompression=True)
