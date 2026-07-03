"""Regenerate golden aligned NIfTI via export module (no HTTP)."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from head_align.export import align_head_from_path, load_service_models, write_nifti
from head_align.paths import DEFAULT_CLS_CKPT, DEFAULT_POSE_CKPT


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batch golden export (direct, no API)")
    p.add_argument("--volumes-dir", type=Path, default=ROOT / "data" / "volumes")
    p.add_argument("--out-dir", type=Path, default=ROOT / "data" / "api_golden")
    p.add_argument("--cls-checkpoint", type=Path, default=DEFAULT_CLS_CKPT)
    p.add_argument("--pose-checkpoint", type=Path, default=DEFAULT_POSE_CKPT)
    p.add_argument("--device", type=str, default="auto")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "aligned").mkdir(exist_ok=True)
    (args.out_dir / "meta").mkdir(exist_ok=True)

    cls_model, pose_model, device = load_service_models(
        args.cls_checkpoint, args.pose_checkpoint, args.device
    )

    case_ids = sorted(p.name.replace(".nii.gz", "") for p in args.volumes_dir.glob("*.nii.gz"))
    rows: list[dict] = []

    for case_id in tqdm(case_ids, desc="export golden"):
        path = args.volumes_dir / f"{case_id}.nii.gz"
        row: dict = {"case_id": case_id, "status": "error"}
        try:
            result = align_head_from_path(path, cls_model, pose_model, device)
            meta = {
                "has_head": result.has_head,
                "detector_ok": result.detector_ok,
                "cls_prob": result.cls_prob,
                "geodesic_deg": result.geodesic_deg,
                "detector_rotz_deg": result.detector_rotz_deg,
                "rotvec_corr_rad": result.rotvec_corr_rad,
                "affine_4x4": result.affine_4x4,
                "message": result.message,
                "spacing_mm": 4.0,
                "frame": "identity_index",
            }
            if result.has_head and result.detector_ok and result.ct_aligned is not None:
                write_nifti(
                    result.ct_aligned,
                    args.out_dir / "aligned" / f"{case_id}_aligned.nii.gz",
                )
                row.update(
                    {
                        "status": "ok",
                        "has_head": True,
                        "detector_ok": True,
                        "cls_prob": result.cls_prob,
                        "geodesic_deg": result.geodesic_deg,
                        "detector_rotz_deg": result.detector_rotz_deg,
                        "message": result.message,
                    }
                )
            else:
                row.update(
                    {
                        "has_head": result.has_head,
                        "detector_ok": result.detector_ok,
                        "cls_prob": result.cls_prob,
                        "message": result.message,
                    }
                )
            (args.out_dir / "meta" / f"{case_id}.json").write_text(
                json.dumps(meta, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception as exc:
            row["message"] = str(exc)
        rows.append(row)

    csv_path = args.out_dir / "results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    n_ok = sum(1 for r in rows if r.get("status") == "ok")
    print(f"Done {len(rows)} cases ({n_ok} aligned) -> {args.out_dir}")


if __name__ == "__main__":
    main()
