"""
Random training sample previews for cls and pose datasets.

Usage:
    python -u scripts/export_dataset_previews.py --num 30
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def center_slices(vol: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    zz, yy, xx = vol.shape[0] // 2, vol.shape[1] // 2, vol.shape[2] // 2
    return vol[zz], vol[:, yy, :], vol[:, :, xx]


def geodesic_deg(rotvec: np.ndarray) -> float:
    return float(np.rad2deg(np.linalg.norm(rotvec)))


def save_previews(
    files: list[Path],
    out_dir: Path,
    *,
    title_prefix: str,
) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.png"):
        old.unlink()

    views = (("Axial", "Z"), ("Coronal", "Y"), ("Sagittal", "X"))
    axis_idx = {"X": 0, "Y": 1, "Z": 2}
    manifest: list[dict] = []

    for path in files:
        d = np.load(path)
        vol = d["volume"].astype(np.float32)
        axial, coronal, sagittal = center_slices(vol)
        has_head = int(d["has_head"])
        rotvec = d["rotvec_corr_rad"].astype(np.float32)
        euler = np.rad2deg(Rotation.from_rotvec(rotvec).as_euler("xyz")).astype(float)
        case_id = str(d["case_id"])
        tag = "HEAD" if has_head else "NO HEAD"
        geo = geodesic_deg(rotvec)

        fig, axes = plt.subplots(1, 3, figsize=(11, 4))
        for ax, slc, (view_name, rot_axis) in zip(axes, (axial, coronal, sagittal), views):
            ax.imshow(slc, cmap="gray", origin="upper", aspect="equal")
            ai = axis_idx[rot_axis]
            ax.set_title(f"{view_name}\nrot {rot_axis} = {float(euler[ai]):+.1f}°", fontsize=9)
            ax.axis("off")

        fig.suptitle(
            f"{title_prefix} | {path.stem} | {tag} | {case_id} | geo={geo:.1f}° | {vol.shape}",
            fontsize=10,
            y=1.02,
        )
        out_png = out_dir / f"{path.stem}.png"
        fig.savefig(out_png, dpi=120, bbox_inches="tight")
        plt.close(fig)
        manifest.append(
            {
                "sample": path.name,
                "preview": out_png.name,
                "has_head": has_head,
                "case_id": case_id,
                "geo_deg": geo,
                "shape": list(vol.shape),
            }
        )
    return manifest


def pick_random(files: list[Path], n: int, rng: np.random.Generator) -> list[Path]:
    n = min(int(n), len(files))
    if n <= 0:
        return []
    idx = rng.choice(len(files), size=n, replace=False)
    return [files[int(i)] for i in sorted(idx)]


def pick_cls_files(train_dir: Path, n: int, rng: np.random.Generator) -> list[Path]:
    all_files = sorted(train_dir.glob("*.npz"))
    pos = [p for p in all_files if float(np.load(p)["has_head"]) > 0.5]
    neg = [p for p in all_files if float(np.load(p)["has_head"]) <= 0.5]
    n_neg = max(1, int(round(n * len(neg) / max(len(all_files), 1)))) if neg else 0
    n_neg = min(n_neg, len(neg), n - 1) if pos else min(n_neg, len(neg))
    n_pos = min(n - n_neg, len(pos))
    picked = pick_random(pos, n_pos, rng) + pick_random(neg, n_neg, rng)
    rng.shuffle(picked)
    return picked


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cls-dir", type=Path, default=ROOT / "data" / "head_align_cls" / "train")
    p.add_argument("--pose-dir", type=Path, default=ROOT / "data" / "head_align_pose" / "train")
    p.add_argument("--cls-out", type=Path, default=ROOT / "data" / "head_align_cls" / "previews")
    p.add_argument("--pose-out", type=Path, default=ROOT / "data" / "head_align_pose" / "previews")
    p.add_argument("--num", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    cls_files = pick_cls_files(args.cls_dir, args.num, rng)
    pose_files = pick_random(sorted(args.pose_dir.glob("*.npz")), args.num, rng)

    cls_manifest = save_previews(cls_files, args.cls_out, title_prefix="CLS")
    pose_manifest = save_previews(pose_files, args.pose_out, title_prefix="POSE")

    cls_pos = sum(1 for m in cls_manifest if m["has_head"])
    cls_neg = len(cls_manifest) - cls_pos

    (args.cls_out / "manifest.json").write_text(json.dumps(cls_manifest, indent=2), encoding="utf-8")
    (args.pose_out / "manifest.json").write_text(json.dumps(pose_manifest, indent=2), encoding="utf-8")

    print(f"CLS previews: {len(cls_manifest)} (pos={cls_pos}, neg={cls_neg}) -> {args.cls_out}")
    print(f"POSE previews: {len(pose_manifest)} -> {args.pose_out}")


if __name__ == "__main__":
    main()
