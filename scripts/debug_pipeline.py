"""
Debug head-align pipeline step-by-step. All previews -> data/pipeline_debug/

Step 1: inferior Z slab (200mm @ 4mm).
Step 2: cls -> axial PCA shift+Z-rot.
Step 3: full pipeline — axial PCA + detector crop + pose (infer_case_hybrid).

Usage:
    python -u scripts/debug_pipeline.py --step 1 --all
    python -u scripts/debug_pipeline.py --step 3 --case-id CQ500CT48 --device cuda
    python -u scripts/run_pipeline.py --all --device cuda
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from head_align.axial_detector import volume_axial_pca_median_center
from head_align.inference import infer_case_hybrid, load_cls_model, load_pose_model
from head_align.paths import DEFAULT_CLS_CKPT, DEFAULT_POSE_CKPT
from scripts.preview_debug import (
    save_step3_final_preview,
    save_step_compare_preview,
)
from head_align.preprocess import (
    BOTTOM_CROP_Z,
    OUT_SHAPE,
    SPACING_MM,
    apply_axial_pca_align,
    extract_bottom_slab_z,
    resize_trilinear,
)
from head_align.volume import center_crop_yx_to_square, prepare_isotropic_ct
from utils import apply_brain_ct_window, sitk_image_to_numpy

DEBUG_OUT = ROOT / "data" / "pipeline_debug"


def bottom_slab_from_ct(ct) -> np.ndarray:
    hu = sitk_image_to_numpy(ct).astype(np.float32)
    vol = apply_brain_ct_window(hu, output_range=(0.0, 1.0))
    vol = center_crop_yx_to_square(vol)
    return extract_bottom_slab_z(vol, n=BOTTOM_CROP_Z)


def step1_bottom_slab(ct, *, spacing_mm: float) -> tuple:
    vol = bottom_slab_from_ct(ct)
    vol_full = apply_brain_ct_window(
        sitk_image_to_numpy(ct).astype(np.float32), output_range=(0.0, 1.0)
    )
    vol_full = center_crop_yx_to_square(vol_full)
    slab_mm = BOTTOM_CROP_Z * spacing_mm
    meta = {
        "z_before": int(vol_full.shape[0]),
        "z_after": int(vol.shape[0]),
        "n_keep": BOTTOM_CROP_Z,
        "slab_mm": slab_mm,
    }
    return vol_full, vol, meta


@torch.no_grad()
def predict_cls_prob(cls_model, vol_zyx: np.ndarray, device: torch.device) -> float:
    cls_vol = resize_trilinear(vol_zyx.astype(np.float32), OUT_SHAPE)
    x = torch.from_numpy(cls_vol).unsqueeze(0).unsqueeze(0).to(device)
    return float(torch.sigmoid(cls_model(x)).item())


def step2_cls_pca_zrot(
    vol_slab: np.ndarray,
    cls_model,
    device: torch.device,
    *,
    cls_threshold: float = 0.5,
    spacing_mm: float = SPACING_MM,
) -> tuple[np.ndarray, np.ndarray, dict]:
    cls_prob = predict_cls_prob(cls_model, vol_slab, device)
    meta: dict = {"cls_prob": cls_prob, "has_head": cls_prob >= cls_threshold}

    if not meta["has_head"]:
        return vol_slab, vol_slab.copy(), meta

    ax_det = volume_axial_pca_median_center(vol_slab)
    if ax_det is None:
        meta["pca_ok"] = False
        return vol_slab, vol_slab.copy(), meta

    vol_aligned, align_meta = apply_axial_pca_align(vol_slab, ax_det, spacing_mm=spacing_mm)
    shift = np.asarray(align_meta["shift_zyx"], dtype=np.float32)
    meta.update(
        {
            "pca_ok": True,
            "pca_n_points": int(align_meta.get("pca_n_points", 0)),
            "rotz_deg": float(align_meta["rotz_detector_deg"]),
            "e1_tilt_deg": float(align_meta["e1_tilt_deg"]),
            "shift_dy": float(shift[1]),
            "shift_dx": float(shift[2]),
            "pca_overlay": {
                "center_xy": ax_det["center"],
                "e1": ax_det["e1"],
                "e2": ax_det["e2"],
            },
        }
    )
    return vol_slab, vol_aligned, meta


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--step", type=int, default=1, choices=[1, 2, 3])
    p.add_argument("--volumes-dir", type=Path, default=ROOT / "data" / "volumes")
    p.add_argument("--out-dir", type=Path, default=DEBUG_OUT)
    p.add_argument(
        "--cls-checkpoint",
        type=Path,
        default=DEFAULT_CLS_CKPT,
    )
    p.add_argument(
        "--pose-checkpoint",
        type=Path,
        default=DEFAULT_POSE_CKPT,
    )
    p.add_argument("--cls-threshold", type=float, default=0.5)
    p.add_argument("--spacing-mm", type=float, default=SPACING_MM)
    p.add_argument("--all", action="store_true")
    p.add_argument("--case-id", nargs="*", default=None)
    p.add_argument("--device", type=str, default="auto")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)
    cls_model = None
    pose_model = None
    if args.step in (2, 3):
        cls_model = load_cls_model(args.cls_checkpoint, device)
    if args.step == 3:
        pose_model = load_pose_model(args.pose_checkpoint, device)

    if args.all:
        case_ids = sorted(p.name.replace(".nii.gz", "") for p in args.volumes_dir.glob("*.nii.gz"))
    elif args.case_id:
        case_ids = list(args.case_id)
    else:
        case_ids = ["CQ500CT6", "CQ500CT48", "CQ500CT243"]

    for case_id in tqdm(case_ids, desc=f"debug step {args.step}"):
        path = args.volumes_dir / f"{case_id}.nii.gz"
        if not path.is_file():
            continue
        ct = prepare_isotropic_ct(path, spacing_mm=float(args.spacing_mm))

        if args.step == 1:
            vol_before, vol_after, meta = step1_bottom_slab(ct, spacing_mm=float(args.spacing_mm))
            save_step_compare_preview(
                args.out_dir / f"step01_bottom_slab_{case_id}.png",
                case_id=case_id,
                step_name="bottom_slab",
                step_desc=(
                    f"шаг 1: нижний slab {meta['slab_mm']:.0f}mm "
                    f"({meta['n_keep']} срезов Z, {meta['z_before']} -> {meta['z_after']})"
                ),
                vol_before=vol_before,
                vol_after=vol_after,
                overlay_before="bottom_slab",
                overlay_kwargs={"n_keep": meta["n_keep"]},
            )
            continue

        vol_slab = bottom_slab_from_ct(ct)
        vol_before, vol_axial, meta2 = step2_cls_pca_zrot(
            vol_slab,
            cls_model,
            device,
            cls_threshold=float(args.cls_threshold),
            spacing_mm=float(args.spacing_mm),
        )

        if args.step == 2:
            if not meta2["has_head"]:
                save_step_compare_preview(
                    args.out_dir / f"step02_cls_pca_{case_id}.png",
                    case_id=case_id,
                    step_name="cls",
                    step_desc=f"шаг 2: НЕТ ГОЛОВЫ (p={meta2['cls_prob']:.2f})",
                    vol_before=vol_before,
                    vol_after=vol_axial,
                )
                continue
            if not meta2.get("pca_ok", True):
                save_step_compare_preview(
                    args.out_dir / f"step02_cls_pca_{case_id}.png",
                    case_id=case_id,
                    step_name="cls+pca",
                    step_desc=f"шаг 2: HEAD p={meta2['cls_prob']:.2f}, PCA fail",
                    vol_before=vol_before,
                    vol_after=vol_axial,
                )
                continue
            save_step_compare_preview(
                args.out_dir / f"step02_cls_pca_{case_id}.png",
                case_id=case_id,
                step_name="cls+pca",
                step_desc=(
                    f"шаг 2: HEAD p={meta2['cls_prob']:.2f} · "
                    f"axial {meta2['pca_n_points']} pts · "
                    f"shift dy={meta2['shift_dy']:+.1f} dx={meta2['shift_dx']:+.1f}px · "
                    f"e1 {meta2['e1_tilt_deg']:+.1f}° → Z {meta2['rotz_deg']:+.1f}°"
                ),
                vol_before=vol_before,
                vol_after=vol_axial,
                overlay_before="pca",
                overlay_kwargs=meta2["pca_overlay"],
            )
            continue

        # step 3 — full pipeline preview
        result = infer_case_hybrid(
            pose_model,
            cls_model,
            ct,
            device,
            cls_threshold=float(args.cls_threshold),
            spacing_mm=float(args.spacing_mm),
        )

        if not result["has_head"]:
            save_step_compare_preview(
                args.out_dir / f"step03_final_{case_id}.png",
                case_id=case_id,
                step_name="cls",
                step_desc=f"итог: НЕТ ГОЛОВЫ (p={result['cls_prob']:.2f})",
                vol_before=vol_before,
                vol_after=vol_axial,
            )
            continue

        if not result.get("detector_ok", False):
            save_step_compare_preview(
                args.out_dir / f"step03_final_{case_id}.png",
                case_id=case_id,
                step_name="detector",
                step_desc=f"итог: HEAD p={result['cls_prob']:.2f}, detector fail",
                vol_before=vol_before,
                vol_after=vol_axial,
            )
            continue

        rotvec = np.asarray(result["rotvec_pred_rad"], dtype=np.float64)
        euler = Rotation.from_rotvec(rotvec).as_euler("xyz", degrees=True)
        det = result.get("det_meta", {})
        rotz = float(det.get("rotz_detector_deg", 0.0))

        save_step3_final_preview(
            args.out_dir / f"step03_final_{case_id}.png",
            case_id=case_id,
            step_desc=(
                f"итог: PCA Z {rotz:+.1f}° · pose geo {result['geodesic_deg']:.1f}° · "
                f"euler X={euler[0]:+.1f}° Y={euler[1]:+.1f}° Z={euler[2]:+.1f}°"
            ),
            vol_slab=vol_before,
            vol_axial_aligned=vol_axial,
            vol_pose_in=result["vol_detector_crop"],
            vol_pose_out=result["vol_aligned"],
            pca_overlay=meta2["pca_overlay"],
        )

    print(f"Done -> {args.out_dir}")


if __name__ == "__main__":
    main()
