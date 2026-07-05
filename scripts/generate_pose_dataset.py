"""
Generate pose dataset: PreAligner on raw -> aug angles -> apply -> save volumes + labels.

Per annotated case:
  1. PreAligner on raw @ 4mm  ->  z_min, z_max, dx, dy, rz_pca
  2. Sample 30x (rz_aug, ry_aug, rx_aug) ~ U(-0.06*pi, +0.06*pi)
  3. Apply prealign + aug rotation to raw 4mm volume
  4. Labels (CW-positive rad):
       rz = rz_gt + rz_aug - rz_pca
       ry = ry_gt + ry_aug
       rx = rx_aug

Re-export guide labels after angle convention change:
  python -u scripts/export_guide_labels.py

Output:
  data/cq500_train/pose_dataset/volumes/{case}_a{idx}.nii.gz
  data/cq500_train/pose_dataset/meta/{case}_a{idx}.json
  samples.csv, manifest.json

Usage:
    python -u scripts/generate_pose_dataset.py --device cuda
    python -u scripts/generate_pose_dataset.py --limit 3 --n-aug 2
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.head_aligner import apply_pose_volume
from models.pre_aligner import PreAligner
from paths import (
    DEFAULT_PRE_ALIGNER_CKPT,
    POSE_AUG_PER_CASE,
    POSE_AUG_RANGE_PI,
    SPACING_MM,
    TRAIN_GUIDE_LABELS_JSON,
    TRAIN_GUIDES_DIR,
    TRAIN_MPR_MANIFEST,
    TRAIN_POSE_DATASET_DIR,
    TRAIN_VOLUMES,
)
from utils.guide_labels import GuideLabel, load_guide_labels
from utils.rigid import save_volume_nifti


def stage(title: str, detail: str = "") -> None:
    bar = "=" * 64
    msg = f"  {title}" + (f" - {detail}" if detail else "")
    print(f"\n{bar}\n{msg}\n{bar}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate pose volume dataset")
    p.add_argument("--out-dir", type=Path, default=TRAIN_POSE_DATASET_DIR)
    p.add_argument("--volumes-dir", type=Path, default=TRAIN_VOLUMES)
    p.add_argument("--guide-labels", type=Path, default=TRAIN_GUIDE_LABELS_JSON)
    p.add_argument("--guides-dir", type=Path, default=TRAIN_GUIDES_DIR)
    p.add_argument("--manifest", type=Path, default=TRAIN_MPR_MANIFEST)
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_PRE_ALIGNER_CKPT)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--spacing-mm", type=float, default=SPACING_MM)
    p.add_argument("--n-aug", type=int, default=POSE_AUG_PER_CASE)
    p.add_argument("--aug-range", type=float, default=POSE_AUG_RANGE_PI)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--cls-threshold", type=float, default=0.5)
    p.add_argument("--cls-pad", type=int, default=3)
    p.add_argument("--cls-min-head-slices", type=int, default=10)
    p.add_argument("--no-clean", action="store_true")
    return p.parse_args()


def _load_labels(args: argparse.Namespace) -> list[GuideLabel]:
    if args.guide_labels.is_file():
        payload = json.loads(args.guide_labels.read_text(encoding="utf-8"))
        return [
            GuideLabel(
                case_id=str(case_id),
                rot_z_rad=float(row["rot_z_rad"]),
                rot_y_rad=float(row["rot_y_rad"]),
                rot_x_rad=float(row.get("rot_x_rad", 0.0)),
                z_lo_1mm=int(row["z_lo_1mm"]),
                z_hi_1mm=int(row["z_hi_1mm"]),
                z_lo=int(row["z_lo"]),
                z_hi=int(row["z_hi"]),
                coronal_slice_index=int(row.get("coronal_slice_index", 0)),
                axial_slice_index=int(row.get("axial_slice_index", 0)),
                shape_zyx=list(row.get("shape_zyx", [])),
                source_json=str(row.get("source_json", "")),
            )
            for case_id, row in payload.get("cases", {}).items()
        ]
    labels, skipped = load_guide_labels(
        args.guides_dir,
        manifest_path=args.manifest,
        volumes_dir=args.volumes_dir,
        cls_spacing_mm=float(args.spacing_mm),
    )
    if skipped:
        print(f"  skipped {len(skipped)} cases while parsing guides", flush=True)
    return labels


def _label_payload(
    label: GuideLabel,
    aug_idx: int,
    *,
    rz_aug: float,
    ry_aug: float,
    rx_aug: float,
    params,
    spacing_mm: float,
    volume_path: str,
    shape_zyx: list[int],
) -> dict:
    rz_gt = float(label.rot_z_rad)
    ry_gt = float(label.rot_y_rad)
    rx_gt = float(label.rot_x_rad)
    rz_pca = float(params.rz)

    rz_lbl = rz_gt + rz_aug - rz_pca
    ry_lbl = ry_gt + ry_aug
    rx_lbl = rx_aug

    return {
        "case_id": label.case_id,
        "sample_id": f"{label.case_id}_a{aug_idx:03d}",
        "aug_idx": int(aug_idx),
        "spacing_mm": float(spacing_mm),
        "volume_path": volume_path,
        "shape_zyx": shape_zyx,
        "guide_gt_rad": {"rz": rz_gt, "ry": ry_gt, "rx": rx_gt},
        "prealign": {
            "z_min": float(params.z_min),
            "z_max": float(params.z_max),
            "dx": float(params.dx),
            "dy": float(params.dy),
            "rz_pca": rz_pca,
        },
        "aug_rad": {"rz": float(rz_aug), "ry": float(ry_aug), "rx": float(rx_aug)},
        "labels_rad": {"rz": rz_lbl, "ry": ry_lbl, "rx": rx_lbl},
        "labels_deg": {
            "rz": math.degrees(rz_lbl),
            "ry": math.degrees(ry_lbl),
            "rx": math.degrees(rx_lbl),
        },
    }


def main() -> None:
    args = parse_args()
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    )

    if not args.checkpoint.is_file():
        raise SystemExit(f"Checkpoint not found: {args.checkpoint}")

    labels = _load_labels(args)
    if not labels:
        raise SystemExit("No guide labels found")
    if args.limit > 0:
        labels = labels[: args.limit]

    vol_out_dir = args.out_dir / "volumes"
    meta_out_dir = args.out_dir / "meta"
    if args.out_dir.is_dir() and not args.no_clean:
        print(f"  removing existing dataset: {args.out_dir}", flush=True)
        shutil.rmtree(args.out_dir)
    vol_out_dir.mkdir(parents=True, exist_ok=True)
    meta_out_dir.mkdir(parents=True, exist_ok=True)

    stage(
        "Pose volume dataset",
        f"{len(labels)} cases x {args.n_aug} aug @ {args.spacing_mm} mm",
    )
    print(
        f"  aug: +/-{args.aug_range}*pi (+/-{math.degrees(args.aug_range * math.pi):.1f} deg)",
        flush=True,
    )
    print(f"  output: {args.out_dir}", flush=True)
    print(f"  device: {device}", flush=True)

    model = PreAligner.from_checkpoint(
        args.checkpoint,
        device,
        cls_threshold=args.cls_threshold,
        cls_pad=args.cls_pad,
        cls_min_head_slices=args.cls_min_head_slices,
    )

    rng = np.random.default_rng(args.seed)
    aug_half = float(args.aug_range) * math.pi

    sample_rows: list[dict] = []
    skip_rows: list[dict] = []
    n_saved = 0

    for label in tqdm(labels, desc="pose volumes"):
        vol_path = args.volumes_dir / f"{label.case_id}.nii.gz"
        if not vol_path.is_file():
            skip_rows.append({"case_id": label.case_id, "reason": "missing_volume"})
            continue

        try:
            vol_raw = PreAligner.prepare_volume(vol_path, spacing_mm=args.spacing_mm)
        except Exception as exc:
            skip_rows.append({"case_id": label.case_id, "reason": f"load_failed: {exc}"})
            continue

        params = model.predict_params(vol_raw, device=device)
        if not params.has_head:
            skip_rows.append({"case_id": label.case_id, "reason": "no_head"})
            print(
                f"  WARN no_head: {label.case_id} (classifier missed head on annotated case)",
                flush=True,
            )
            continue

        for aug_idx in range(int(args.n_aug)):
            rz_aug = float(rng.uniform(-aug_half, aug_half))
            ry_aug = float(rng.uniform(-aug_half, aug_half))
            rx_aug = float(rng.uniform(-aug_half, aug_half))

            vol_out, apply_meta = apply_pose_volume(
                vol_raw,
                params,
                rz_aug=rz_aug,
                ry_aug=-ry_aug,
                rx_aug=rx_aug,
                spacing_mm=float(args.spacing_mm),
            )
            if not apply_meta.get("ok") or vol_out.size == 0:
                skip_rows.append(
                    {
                        "case_id": label.case_id,
                        "aug_idx": aug_idx,
                        "reason": apply_meta.get("reason", "apply_fail"),
                    }
                )
                continue

            stem = f"{label.case_id}_a{aug_idx:03d}"
            nii_path = vol_out_dir / f"{stem}.nii.gz"
            save_volume_nifti(vol_out, nii_path, spacing_mm=float(args.spacing_mm))

            meta = _label_payload(
                label,
                aug_idx,
                rz_aug=rz_aug,
                ry_aug=ry_aug,
                rx_aug=rx_aug,
                params=params,
                spacing_mm=float(args.spacing_mm),
                volume_path=str(nii_path),
                shape_zyx=[int(x) for x in vol_out.shape],
            )
            meta["apply"] = {k: v for k, v in apply_meta.items() if k != "ok"}
            meta_path = meta_out_dir / f"{stem}.json"
            meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

            row = {
                "sample_id": stem,
                "case_id": label.case_id,
                "aug_idx": aug_idx,
                "volume_path": str(nii_path),
                "meta_path": str(meta_path),
                **{f"label_{k}_deg": meta["labels_deg"][k] for k in ("rz", "ry", "rx")},
                **{f"label_{k}_rad": meta["labels_rad"][k] for k in ("rz", "ry", "rx")},
            }
            sample_rows.append(row)
            n_saved += 1

    manifest = {
        "n_cases": len(labels),
        "n_aug_per_case": int(args.n_aug),
        "aug_range_pi": float(args.aug_range),
        "aug_half_rad": aug_half,
        "spacing_mm": float(args.spacing_mm),
        "checkpoint": str(args.checkpoint),
        "n_samples_saved": n_saved,
        "n_skipped_events": len(skip_rows),
        "label_formulas": {
            "rz": "rz_gt + rz_aug - rz_pca",
            "ry": "ry_gt + ry_aug",
            "rx": "rx_aug",
        },
    }
    (args.out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    if sample_rows:
        fieldnames = sorted({k for r in sample_rows for k in r.keys()})
        with (args.out_dir / "samples.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(sample_rows)

    if skip_rows:
        fieldnames = sorted({k for r in skip_rows for k in r.keys()})
        with (args.out_dir / "skipped.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(skip_rows)

    n_no_head = sum(1 for r in skip_rows if r.get("reason") == "no_head")
    print(
        f"\n  done: {n_saved} volumes | skipped_events={len(skip_rows)} (no_head={n_no_head})\n"
        f"  volumes -> {vol_out_dir}\n"
        f"  meta    -> {meta_out_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
