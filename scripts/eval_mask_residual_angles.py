"""
Residual pose error on aligned eye/ear masks (golden eval).

Applies HeadAligner predictions from align/meta to structure masks @ 1 mm,
then measures |tilt| of L-R eye/ear segments vs horizontal (axial/coronal)
and eye-mid→ear-mid vs horizontal on sagittal.

Usage:
    python -u scripts/eval_mask_residual_angles.py
    python -u scripts/eval_mask_residual_angles.py --limit 10
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from paths import (
    APPLY_SPACING_MM,
    TEST_ALIGN_META,
    TEST_MASK_RESIDUAL_DIR,
    TEST_MASKS,
    TEST_VOLUMES,
)
from utils.mask_residual import compute_mask_residuals


def stage(title: str, detail: str = "") -> None:
    bar = "=" * 64
    msg = f"  {title}" + (f" — {detail}" if detail else "")
    print(f"\n{bar}\n{msg}\n{bar}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Residual angles on aligned eye/ear masks")
    p.add_argument("--volumes-dir", type=Path, default=TEST_VOLUMES)
    p.add_argument("--masks-dir", type=Path, default=TEST_MASKS)
    p.add_argument("--align-meta-dir", type=Path, default=TEST_ALIGN_META)
    p.add_argument("--out-dir", type=Path, default=TEST_MASK_RESIDUAL_DIR)
    p.add_argument("--spacing-mm", type=float, default=APPLY_SPACING_MM)
    p.add_argument("--limit", type=int, default=0)
    return p.parse_args()


def _stats(vals: list[float]) -> dict[str, float | int | None]:
    if not vals:
        return {"n": 0, "mean": None, "median": None, "p90": None, "max": None}
    s = sorted(vals)
    p90_i = min(len(s) - 1, int(round(0.9 * (len(s) - 1))))
    return {
        "n": len(s),
        "mean": float(statistics.mean(s)),
        "median": float(statistics.median(s)),
        "p90": float(s[p90_i]),
        "max": float(s[-1]),
    }


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir
    meta_out = out_dir / "meta"
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_out.mkdir(parents=True, exist_ok=True)

    meta_paths = sorted(args.align_meta_dir.glob("*.json"))
    if args.limit > 0:
        meta_paths = meta_paths[: args.limit]

    stage("Mask residual angles", f"{len(meta_paths)} cases @ {args.spacing_mm} mm")
    rows: list[dict] = []
    skipped: list[dict] = []

    for meta_path in tqdm(meta_paths, desc="mask residual"):
        case_id = meta_path.stem
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("status") != "ok" or not meta.get("has_head", False):
            skipped.append({"case_id": case_id, "reason": meta.get("status", "not_ok")})
            continue

        vol_path = args.volumes_dir / f"{case_id}.nii.gz"
        mask_case = args.masks_dir / case_id
        if not vol_path.is_file():
            skipped.append({"case_id": case_id, "reason": "missing_volume"})
            continue
        if not mask_case.is_dir():
            skipped.append({"case_id": case_id, "reason": "missing_masks"})
            continue

        try:
            res = compute_mask_residuals(
                vol_path,
                mask_case,
                meta,
                spacing_mm=float(args.spacing_mm),
            )
        except Exception as exc:
            skipped.append({"case_id": case_id, "reason": str(exc)})
            continue

        row = {
            "case_id": case_id,
            "rz_eyes_deg": res["rz_eyes_deg"],
            "rz_ears_deg": res["rz_ears_deg"],
            "ry_eyes_deg": res["ry_eyes_deg"],
            "ry_ears_deg": res["ry_ears_deg"],
            "rx_om_deg": res["rx_om_deg"],
            "rz_max_deg": max(float(res["rz_eyes_deg"]), float(res["rz_ears_deg"])),
            "ry_max_deg": max(float(res["ry_eyes_deg"]), float(res["ry_ears_deg"])),
        }
        rows.append(row)
        (meta_out / f"{case_id}.json").write_text(
            json.dumps({"case_id": case_id, "residual": res}, indent=2),
            encoding="utf-8",
        )

    if rows:
        fieldnames = sorted({k for r in rows for k in r.keys()})
        with (out_dir / "residual_angles.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)

    metrics = ("rz_eyes_deg", "rz_ears_deg", "ry_eyes_deg", "ry_ears_deg", "rx_om_deg", "rz_max_deg", "ry_max_deg")
    summary = {
        "n_meta": len(meta_paths),
        "n_ok": len(rows),
        "n_skipped": len(skipped),
        "spacing_mm": float(args.spacing_mm),
        "align_meta_dir": str(args.align_meta_dir),
        "stats_deg": {m: _stats([float(r[m]) for r in rows]) for m in metrics},
        "note": (
            "Residual |tilt| after HeadAligner apply on eye/ear masks. "
            "rz/ry: L-R segment vs horizontal on axial/coronal; "
            "rx: sagittal eye-mid→ear-mid vs horizontal."
        ),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if skipped:
        with (out_dir / "skipped.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["case_id", "reason"])
            w.writeheader()
            w.writerows(skipped)

    st = summary["stats_deg"]
    print(
        f"\n  done: ok={len(rows)} skipped={len(skipped)}\n"
        f"  mean°  rz_eyes={st['rz_eyes_deg']['mean']:.2f}  rz_ears={st['rz_ears_deg']['mean']:.2f}  "
        f"ry_eyes={st['ry_eyes_deg']['mean']:.2f}  ry_ears={st['ry_ears_deg']['mean']:.2f}  "
        f"rx_om={st['rx_om_deg']['mean']:.2f}\n"
        f"  median° rz_eyes={st['rz_eyes_deg']['median']:.2f}  rx_om={st['rx_om_deg']['median']:.2f}\n"
        f"  out -> {out_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
