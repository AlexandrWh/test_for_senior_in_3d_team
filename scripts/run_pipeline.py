"""
Run final head-align pipeline: bottom slab -> cls -> axial PCA -> pose.

Usage:
    python -u scripts/run_pipeline.py --case-id CQ500CT48 --device cuda
    python -u scripts/run_pipeline.py --all --device cuda --save-previews
    python -u scripts/run_pipeline.py --val --save-previews --device cuda
"""

from __future__ import annotations

import argparse
import csv
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
from head_align.labels import case_ids_from_npz_split
from head_align.preprocess import (
    BOTTOM_CROP_Z,
    OUT_SHAPE,
    SPACING_MM,
    apply_axial_pca_align,
    extract_bottom_slab_z,
    resize_trilinear,
)
from scripts.preview_debug import save_step3_final_preview
from head_align.volume import center_crop_yx_to_square, prepare_isotropic_ct
from utils import apply_brain_ct_window, sitk_image_to_numpy

DEBUG_OUT = ROOT / "data" / "pipeline_debug"
VAL_OUT = ROOT / "data" / "pipeline_debug_val"
DEFAULT_VAL_VOLUMES = ROOT / "data" / "cq500_train" / "volumes"
DEFAULT_GOLDEN_VOLUMES = ROOT / "data" / "volumes"
DEFAULT_CLS_DATASET = ROOT / "data" / "head_align_cls"


def _slab_from_ct(ct) -> np.ndarray:
    hu = sitk_image_to_numpy(ct).astype(np.float32)
    vol = apply_brain_ct_window(hu, output_range=(0.0, 1.0))
    vol = center_crop_yx_to_square(vol)
    return extract_bottom_slab_z(vol, n=BOTTOM_CROP_Z)


@torch.no_grad()
def _axial_pca_overlay_meta(vol_slab: np.ndarray, cls_model, device, *, spacing_mm: float) -> tuple[np.ndarray, dict]:
    vol = vol_slab.astype(np.float32, copy=False)
    cls_vol = resize_trilinear(vol, OUT_SHAPE)
    x = torch.from_numpy(cls_vol).unsqueeze(0).unsqueeze(0).to(device)
    if float(torch.sigmoid(cls_model(x)).item()) < 0.5:
        return vol, {}
    ax_det = volume_axial_pca_median_center(vol)
    if ax_det is None:
        return vol, {}
    vol_ax, _ = apply_axial_pca_align(vol, ax_det, spacing_mm=spacing_mm)
    return vol_ax, {
        "center_xy": ax_det["center"],
        "e1": ax_det["e1"],
        "e2": ax_det["e2"],
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Final pipeline: axial PCA + pose")
    p.add_argument("--volumes-dir", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_CLS_DATASET,
        help="head_align_cls root (for --val case list)",
    )
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
    p.add_argument("--save-previews", action="store_true")
    p.add_argument("--val", action="store_true", help="Run on unique val case_ids from dataset")
    p.add_argument("--all", action="store_true")
    p.add_argument("--case-id", nargs="*", default=None)
    p.add_argument("--device", type=str, default="auto")
    return p.parse_args()


def _resolve_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    if args.val:
        volumes_dir = args.volumes_dir or DEFAULT_VAL_VOLUMES
        out_dir = args.out_dir or VAL_OUT
    else:
        volumes_dir = args.volumes_dir or DEFAULT_GOLDEN_VOLUMES
        out_dir = args.out_dir or DEBUG_OUT
    return volumes_dir, out_dir


def _resolve_case_ids(args: argparse.Namespace, volumes_dir: Path) -> list[str]:
    if args.val:
        return case_ids_from_npz_split(args.dataset_dir / "val")
    if args.all:
        return sorted(p.name.replace(".nii.gz", "") for p in volumes_dir.glob("*.nii.gz"))
    if args.case_id:
        return list(args.case_id)
    return ["CQ500CT6", "CQ500CT48", "CQ500CT243"]


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)
    cls_model = load_cls_model(args.cls_checkpoint, device)
    pose_model = load_pose_model(args.pose_checkpoint, device)

    volumes_dir, out_dir = _resolve_paths(args)
    case_ids = _resolve_case_ids(args, volumes_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    for case_id in tqdm(case_ids, desc="val pca+pose" if args.val else "pca+pose pipeline"):
        path = volumes_dir / f"{case_id}.nii.gz"
        if not path.is_file():
            continue

        ct = prepare_isotropic_ct(path, spacing_mm=float(args.spacing_mm))
        result = infer_case_hybrid(
            pose_model,
            cls_model,
            ct,
            device,
            cls_threshold=float(args.cls_threshold),
            spacing_mm=float(args.spacing_mm),
        )

        row = {
            "case_id": case_id,
            "has_head": int(result["has_head"]),
            "cls_prob": round(float(result["cls_prob"]), 4),
            "detector_ok": int(result.get("detector_ok", False)),
            "geodesic_deg": "",
            "rot_x_deg": "",
            "rot_y_deg": "",
            "rot_z_deg": "",
            "detector_rotz_deg": "",
        }

        if result["has_head"] and result.get("detector_ok"):
            rotvec = np.asarray(result["rotvec_pred_rad"], dtype=np.float64)
            euler = Rotation.from_rotvec(rotvec).as_euler("xyz", degrees=True)
            det = result.get("det_meta", {})
            row.update(
                {
                    "geodesic_deg": round(float(result["geodesic_deg"]), 4),
                    "rot_x_deg": round(float(euler[0]), 2),
                    "rot_y_deg": round(float(euler[1]), 2),
                    "rot_z_deg": round(float(euler[2]), 2),
                    "detector_rotz_deg": round(float(det.get("rotz_detector_deg", 0.0)), 2),
                }
            )

        rows.append(row)

        if args.save_previews and result["has_head"] and result.get("detector_ok"):
            vol_slab = _slab_from_ct(ct)
            vol_axial, pca_overlay = _axial_pca_overlay_meta(
                vol_slab, cls_model, device, spacing_mm=float(args.spacing_mm)
            )
            rotz = float(result.get("det_meta", {}).get("rotz_detector_deg", 0.0))
            euler = Rotation.from_rotvec(
                np.asarray(result["rotvec_pred_rad"], dtype=np.float64)
            ).as_euler("xyz", degrees=True)
            save_step3_final_preview(
                out_dir / f"step03_final_{case_id}.png",
                case_id=case_id,
                step_desc=(
                    f"итог: PCA Z {rotz:+.1f}° · pose geo {result['geodesic_deg']:.1f}° · "
                    f"euler X={euler[0]:+.1f}° Y={euler[1]:+.1f}° Z={euler[2]:+.1f}°"
                ),
                vol_slab=vol_slab,
                vol_axial_aligned=vol_axial,
                vol_pose_in=result["vol_detector_crop"],
                vol_pose_out=result["vol_aligned"],
                pca_overlay=pca_overlay,
            )

    if rows:
        csv_path = out_dir / "pipeline_results.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    n_ok = sum(1 for r in rows if r["has_head"] and r["detector_ok"])
    print(f"Done {len(rows)} cases ({n_ok} aligned) -> {out_dir}")


if __name__ == "__main__":
    main()
