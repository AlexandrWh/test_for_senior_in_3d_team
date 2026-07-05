"""
Golden eval: HeadAligner on test volumes -> aligned head NIfTI @ 1 mm.

Infer: 4 mm (Z-cls + PCA + pose). Apply: 1 mm isotropic.

Usage:
    python -u scripts/run_head_align_golden.py --device cuda
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.head_aligner import HeadAligner
from paths import (
    DEFAULT_POSE_REGRESSOR_CKPT,
    DEFAULT_PRE_ALIGNER_CKPT,
    TEST_ALIGN_DIR,
    TEST_VOLUMES,
)
from utils.rigid import save_volume_nifti


def stage(title: str, detail: str = "") -> None:
    bar = "=" * 64
    msg = f"  {title}" + (f" — {detail}" if detail else "")
    print(f"\n{bar}\n{msg}\n{bar}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="HeadAligner golden eval")
    p.add_argument("--volumes-dir", type=Path, default=TEST_VOLUMES)
    p.add_argument("--out-dir", type=Path, default=TEST_ALIGN_DIR)
    p.add_argument("--pre-align-ckpt", type=Path, default=DEFAULT_PRE_ALIGNER_CKPT)
    p.add_argument("--pose-ckpt", type=Path, default=DEFAULT_POSE_REGRESSOR_CKPT)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--cls-threshold", type=float, default=0.5)
    p.add_argument("--cls-pad", type=int, default=3)
    p.add_argument("--cls-min-head-slices", type=int, default=10)
    p.add_argument("--limit", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    )
    if not args.pre_align_ckpt.is_file():
        raise SystemExit(f"PreAlign checkpoint not found: {args.pre_align_ckpt}")
    if not args.pose_ckpt.is_file():
        raise SystemExit(f"Pose checkpoint not found: {args.pose_ckpt}")

    out_vol_dir = args.out_dir / "volumes"
    meta_dir = args.out_dir / "meta"
    out_vol_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    stage("HeadAligner golden", "infer @ 4 mm, save @ 1 mm")
    print(f"  input:  {args.volumes_dir}", flush=True)
    print(f"  output: {args.out_dir}", flush=True)
    print(f"  device: {device}", flush=True)

    aligner = HeadAligner.from_checkpoints(
        args.pre_align_ckpt,
        args.pose_ckpt,
        device,
        cls_threshold=args.cls_threshold,
        cls_pad=args.cls_pad,
        cls_min_head_slices=args.cls_min_head_slices,
    )

    paths = sorted(args.volumes_dir.glob("*.nii.gz"))
    if args.limit > 0:
        paths = paths[: args.limit]

    rows: list[dict] = []
    for path in tqdm(paths, desc="head align"):
        case_id = path.name.replace(".nii.gz", "")
        row: dict = {"case_id": case_id, "status": "error"}
        try:
            result = aligner.align(path, device=device, case_id=case_id)
            meta = result.to_json_dict()

            if result.status == "ok" and result.volume_aligned_1mm is not None:
                out_nii = out_vol_dir / f"{case_id}.nii.gz"
                save_volume_nifti(
                    result.volume_aligned_1mm,
                    out_nii,
                    spacing_mm=result.output_spacing_mm,
                )
                meta["aligned_path"] = str(out_nii)
                row.update(
                    {
                        "status": "ok",
                        "has_head": True,
                        "aligned_path": str(out_nii),
                        "z_min": round(result.z_min, 2),
                        "z_max": round(result.z_max, 2),
                        "dx": round(result.dx, 2),
                        "dy": round(result.dy, 2),
                        "rz_pca_deg": round(math.degrees(result.rz_pca_rad), 3),
                        "rz_pose_deg": round(math.degrees(result.rz_pose_rad), 3),
                        "ry_pose_deg": round(math.degrees(result.ry_pose_rad), 3),
                        "rx_pose_deg": round(math.degrees(result.rx_pose_rad), 3),
                    }
                )
            elif result.status == "no_head":
                row.update({"status": "no_head", "has_head": False})
            else:
                row.update({"status": result.status, "has_head": result.has_head})
                if result.reason:
                    row["message"] = result.reason

            (meta_dir / f"{case_id}.json").write_text(
                json.dumps(meta, indent=2), encoding="utf-8"
            )
        except Exception as exc:
            row["message"] = str(exc)
        rows.append(row)

    csv_path = args.out_dir / "results.csv"
    fieldnames = sorted({k for r in rows for k in r.keys()})
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    n_ok = sum(1 for r in rows if r.get("status") == "ok")
    n_no = sum(1 for r in rows if r.get("status") == "no_head")
    n_fail = len(rows) - n_ok - n_no
    print(
        f"\n  done: {len(rows)} cases | ok={n_ok} no_head={n_no} fail={n_fail}\n"
        f"  csv  -> {csv_path}\n"
        f"  nii  -> {out_vol_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
