"""
Analyze manual guide-line annotations: QC + export labels for Z-slice classifier.

Expected guide orientation:
  axial, coronal  -> vertical   (brain tilt / rotation)
  sagittal        -> ignored (assumed 0° for dataset export)

Exports (under data/cq500_train/cq500_train_guides_analysis/):
  guide_labels.json   — rot_z, rot_y, rot_x=0, z_lo/z_hi @ 4mm
  guide_labels.csv
  skipped_cases.csv
  per_line_deviations.csv, summary_stats.csv

Usage:
    python -u scripts/export_guide_labels.py
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from paths import (
    SPACING_MM,
    TRAIN_GUIDES_ANALYSIS_DIR,
    TRAIN_GUIDES_DIR,
    TRAIN_MPR_MANIFEST,
    TRAIN_VOLUMES,
)
from utils.guide_labels import has_axial_coronal_segments, load_guide_labels, load_mpr_manifest

PLANES = ("axial", "coronal", "sagittal")
EXPECTED = {
    "axial": "vertical",
    "coronal": "vertical",
    "sagittal": "horizontal",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze guides + export guide_labels.json")
    p.add_argument("--guides-dir", type=Path, default=TRAIN_GUIDES_DIR)
    p.add_argument("--out-dir", type=Path, default=TRAIN_GUIDES_ANALYSIS_DIR)
    p.add_argument("--manifest", type=Path, default=TRAIN_MPR_MANIFEST)
    p.add_argument("--volumes-dir", type=Path, default=TRAIN_VOLUMES)
    p.add_argument("--cls-spacing-mm", type=float, default=SPACING_MM)
    p.add_argument("--incomplete", action="store_true", help="Include partial annotations in QC CSV")
    return p.parse_args()


def line_acute_angle_deg(p0: list[float], p1: list[float]) -> float:
    dx = float(p1[0]) - float(p0[0])
    dy = float(p1[1]) - float(p0[1])
    if dx == 0.0 and dy == 0.0:
        return float("nan")
    theta = abs(math.degrees(math.atan2(dy, dx)))
    return min(theta, 180.0 - theta)


def deviation_deg(acute_h: float, expected: str) -> float:
    if expected == "vertical":
        return abs(90.0 - acute_h)
    if expected == "horizontal":
        return acute_h
    raise ValueError(expected)


def load_qc_rows(guides_dir: Path, *, include_incomplete: bool) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(guides_dir.glob("*.json")):
        ann = json.loads(path.read_text(encoding="utf-8"))
        case_id = ann.get("case_id", path.stem)
        valid_ac = has_axial_coronal_segments(ann)
        if not valid_ac and not include_incomplete:
            continue

        for plane in PLANES:
            pl = ann.get("planes", {}).get(plane, {})
            p0, p1 = pl.get("p0"), pl.get("p1")
            if p0 is None or p1 is None:
                if include_incomplete:
                    rows.append(
                        {
                            "case_id": case_id,
                            "plane": plane,
                            "expected": EXPECTED[plane],
                            "valid_ax_cor": valid_ac,
                            "acute_angle_deg": np.nan,
                            "deviation_deg": np.nan,
                            "line_length_px": np.nan,
                        }
                    )
                continue

            dx = float(p1[0]) - float(p0[0])
            dy = float(p1[1]) - float(p0[1])
            length = math.hypot(dx, dy)
            acute = line_acute_angle_deg(p0, p1)
            dev = deviation_deg(acute, EXPECTED[plane])
            rows.append(
                {
                    "case_id": case_id,
                    "plane": plane,
                    "expected": EXPECTED[plane],
                    "valid_ax_cor": valid_ac,
                    "acute_angle_deg": acute,
                    "deviation_deg": dev,
                    "line_length_px": length,
                }
            )
    return rows


def summarize(series: pd.Series) -> dict[str, float]:
    s = series.dropna()
    if s.empty:
        return {}
    return {
        "n": int(s.count()),
        "mean": float(s.mean()),
        "median": float(s.median()),
        "std": float(s.std(ddof=0)),
        "p90": float(s.quantile(0.90)),
        "p95": float(s.quantile(0.95)),
        "max": float(s.max()),
    }


def print_stats(label: str, stats: dict[str, float]) -> None:
    if not stats:
        print(f"  {label}: no data")
        return
    print(
        f"  {label}: n={stats['n']}  "
        f"mean={stats['mean']:.2f}°  median={stats['median']:.2f}°  "
        f"std={stats['std']:.2f}°  p90={stats['p90']:.2f}°  "
        f"p95={stats['p95']:.2f}°  max={stats['max']:.2f}°"
    )


def export_labels(
    guides_dir: Path,
    out_dir: Path,
    *,
    manifest_path: Path,
    volumes_dir: Path,
    cls_spacing_mm: float,
) -> tuple[list, list]:
    labels, skipped = load_guide_labels(
        guides_dir,
        manifest_path=manifest_path,
        volumes_dir=volumes_dir,
        cls_spacing_mm=cls_spacing_mm,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    mpr_manifest = load_mpr_manifest(manifest_path)
    payload = {
        "mpr_spacing_mm": float(mpr_manifest.get("spacing_mm", 1.0)),
        "cls_spacing_mm": float(cls_spacing_mm),
        "n_cases": len(labels),
        "cases": {lb.case_id: lb.to_dict() for lb in labels},
    }
    labels_json = out_dir / "guide_labels.json"
    labels_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    if labels:
        pd.DataFrame([lb.to_dict() for lb in labels]).to_csv(
            out_dir / "guide_labels.csv", index=False
        )
    if skipped:
        pd.DataFrame(skipped).to_csv(out_dir / "skipped_cases.csv", index=False)

    return labels, skipped


def main() -> None:
    args = parse_args()
    if not args.guides_dir.is_dir():
        raise SystemExit(f"Guides dir not found: {args.guides_dir}")

    all_json = sorted(args.guides_dir.glob("*.json"))
    args.out_dir.mkdir(parents=True, exist_ok=True)

    labels, skipped = export_labels(
        args.guides_dir,
        args.out_dir,
        manifest_path=args.manifest,
        volumes_dir=args.volumes_dir,
        cls_spacing_mm=float(args.cls_spacing_mm),
    )

    rows = load_qc_rows(args.guides_dir, include_incomplete=args.incomplete)
    df = pd.DataFrame(rows)
    if not df.empty:
        per_line_csv = args.out_dir / "per_line_deviations.csv"
        df.to_csv(per_line_csv, index=False)
        valid = df.dropna(subset=["deviation_deg"])
        summary_rows: list[dict] = []

        print(f"JSON files: {len(all_json)}")
        print(f"Valid axial+coronal (dataset): {len(labels)}")
        print(f"Skipped: {len(skipped)}")
        print(f"QC lines: {len(valid)}")
        print()

        for plane in PLANES:
            sub = valid[valid["plane"] == plane]
            stats = summarize(sub["deviation_deg"])
            print(f"{plane} (expected {EXPECTED[plane]}):")
            print_stats("deviation", stats)
            if stats:
                summary_rows.append({"group": plane, **stats})
            print()

        vertical = valid[valid["expected"] == "vertical"]
        print("pooled vertical (axial + coronal):")
        print_stats("deviation", summarize(vertical["deviation_deg"]))
        print()

        if labels:
            span = pd.Series([lb.z_hi - lb.z_lo + 1 for lb in labels])
            print(
                f"Z-span @ {args.cls_spacing_mm}mm: "
                f"mean={span.mean():.1f} median={span.median():.0f} "
                f"min={span.min()} max={span.max()}"
            )
            print()

        pd.DataFrame(summary_rows).to_csv(args.out_dir / "summary_stats.csv", index=False)
        print(f"QC per-line CSV -> {per_line_csv}")

    print(f"guide_labels.json -> {args.out_dir / 'guide_labels.json'}")
    print(f"guide_labels.csv  -> {args.out_dir / 'guide_labels.csv'}")
    if skipped:
        print(f"skipped_cases.csv -> {args.out_dir / 'skipped_cases.csv'}")


if __name__ == "__main__":
    main()
