"""
Render before/after previews: raw @ 1 mm vs aligned head @ 1 mm.

Usage:
    python -u scripts/render_align_previews.py
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.pre_aligner import PreAligner
from paths import (
    APPLY_SPACING_MM,
    TEST_ALIGN_DIR,
    TEST_ALIGN_PREVIEWS,
    TEST_ALIGN_VOLUMES,
    TEST_VOLUMES,
)
from utils import center_slice_np
from utils.rigid import load_volume_nifti

PLANES = ("axial", "coronal", "sagittal")


def stage(title: str, detail: str = "") -> None:
    bar = "=" * 64
    msg = f"  {title}" + (f" — {detail}" if detail else "")
    print(f"\n{bar}\n{msg}\n{bar}", flush=True)


def _slice_img(vol_zyx: np.ndarray, plane: str) -> np.ndarray:
    return np.clip(center_slice_np(vol_zyx, plane), 0.0, 1.0)


def save_before_after_preview(
    out_path: str | Path,
    case_id: str,
    vol_before: np.ndarray,
    vol_after: np.ndarray,
    *,
    before_label: str = "до",
    after_label: str = "после",
    footer_lines: list[str] | None = None,
    dpi: int = 120,
) -> None:
    """2x3 MPR grid: row 0 = before, row 1 = after; optional monospace footer."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_footer = len(footer_lines) if footer_lines else 0
    fig_h = 6.5 + 0.22 * n_footer
    fig, axes = plt.subplots(2, 3, figsize=(10, fig_h), dpi=dpi)
    fig.suptitle(case_id, fontsize=12, fontweight="bold")

    for col, plane in enumerate(PLANES):
        ax_before = axes[0, col]
        ax_after = axes[1, col]

        ax_before.imshow(_slice_img(vol_before, plane), cmap="gray", vmin=0, vmax=1, aspect="equal")
        ax_before.set_title(plane)
        ax_before.set_ylabel(before_label, fontsize=10)
        ax_before.set_xticks([])
        ax_before.set_yticks([])

        ax_after.imshow(_slice_img(vol_after, plane), cmap="gray", vmin=0, vmax=1, aspect="equal")
        ax_after.set_ylabel(after_label, fontsize=10)
        ax_after.set_xticks([])
        ax_after.set_yticks([])

    bottom = 0.08 + 0.025 * n_footer
    if footer_lines:
        for i, line in enumerate(footer_lines):
            fig.text(
                0.5,
                0.02 + 0.028 * (len(footer_lines) - 1 - i),
                line,
                ha="center",
                va="bottom",
                fontsize=9,
                family="monospace",
            )

    fig.tight_layout(rect=(0, bottom, 1, 0.96))
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Render align golden previews")
    p.add_argument("--volumes-dir", type=Path, default=TEST_VOLUMES)
    p.add_argument("--align-dir", type=Path, default=TEST_ALIGN_DIR)
    p.add_argument("--aligned-dir", type=Path, default=TEST_ALIGN_VOLUMES)
    p.add_argument("--out-dir", type=Path, default=TEST_ALIGN_PREVIEWS)
    p.add_argument("--csv", type=Path, default=None)
    p.add_argument("--limit", type=int, default=0)
    return p.parse_args()


def _footer(meta: dict | None, row: dict) -> list[str]:
    if meta:
        pre = meta.get("prealign", {})
        pose = meta.get("pose", {})
        lines: list[str] = []
        if pre:
            lines.append(
                f"pca rz={float(pre.get('rz_pca_deg', 0)):+.2f} deg  "
                f"dx={float(pre.get('dx', 0)):+.1f} dy={float(pre.get('dy', 0)):+.1f} mm"
            )
        if pose:
            lines.append(
                f"pose rz={float(pose.get('rz_deg', 0)):+.2f}  "
                f"ry={float(pose.get('ry_deg', 0)):+.2f}  "
                f"rx={float(pose.get('rx_deg', 0)):+.2f} deg"
            )
        return lines
    if row.get("rz_pca_deg") is not None:
        return [
            f"pca rz={float(row['rz_pca_deg']):+.2f}  "
            f"pose rz={float(row.get('rz_pose_deg', 0)):+.2f}  "
            f"ry={float(row.get('ry_pose_deg', 0)):+.2f}  "
            f"rx={float(row.get('rx_pose_deg', 0)):+.2f} deg"
        ]
    return []


def main() -> None:
    args = parse_args()
    csv_path = args.csv or args.align_dir / "results.csv"
    meta_dir = args.align_dir / "meta"
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if not csv_path.is_file():
        raise SystemExit(f"Results CSV not found: {csv_path}")

    with csv_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if args.limit > 0:
        rows = rows[: args.limit]

    stage("Render align previews", f"{len(rows)} cases")
    print(f"  raw:     {args.volumes_dir}", flush=True)
    print(f"  aligned: {args.aligned_dir}", flush=True)
    print(f"  out:     {out_dir}", flush=True)

    n_ok = 0
    for row in tqdm(rows, desc="render previews"):
        case_id = row.get("case_id", "").strip()
        if not case_id or row.get("status", "").strip().lower() != "ok":
            continue

        raw_path = args.volumes_dir / f"{case_id}.nii.gz"
        aligned_path = args.aligned_dir / f"{case_id}.nii.gz"
        if not raw_path.is_file() or not aligned_path.is_file():
            continue

        meta_path = meta_dir / f"{case_id}.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else None

        vol_before = PreAligner.prepare_volume(raw_path, spacing_mm=APPLY_SPACING_MM)
        vol_after = load_volume_nifti(aligned_path)

        save_before_after_preview(
            out_dir / f"{case_id}_before_after.png",
            case_id,
            vol_before,
            vol_after,
            before_label="до (raw 1mm)",
            after_label="после (align 1mm)",
            footer_lines=_footer(meta, row),
        )
        n_ok += 1

    print(f"\n  done: {n_ok} previews -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
