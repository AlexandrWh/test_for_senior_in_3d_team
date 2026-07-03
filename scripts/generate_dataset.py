"""
Generate 4mm cls + pose datasets.

Cls (pre-detector, resized to 48x56x56):
  train: 800 pos + 200 neg | val: 80 pos + 20 neg

Pose (post-detector aligned 48x56x56, rotation GT):
  train: 800 pos | val: 80 pos

Usage:
    python -u scripts/generate_dataset.py --clean
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from head_align.augment import misalign_to_windowed_volume
from head_align.labels import collect_case_ids
from head_align.preprocess import OUT_SHAPE, SPACING_MM, cls_preprocess, detector_align_slab_pca_zrot
from head_align.volume import prepare_isotropic_ct


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--base-dir", type=Path, default=ROOT / "data" / "cq500_train")
    p.add_argument("--cls-dir", type=Path, default=ROOT / "data" / "head_align_cls")
    p.add_argument("--pose-dir", type=Path, default=ROOT / "data" / "head_align_pose")
    p.add_argument("--n-train-pos", type=int, default=800)
    p.add_argument("--n-val-pos", type=int, default=80)
    p.add_argument("--n-train-neg", type=int, default=200)
    p.add_argument("--n-val-neg", type=int, default=20)
    p.add_argument("--spacing-mm", type=float, default=SPACING_MM)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-detector-retries", type=int, default=8)
    p.add_argument("--clean", action="store_true")
    p.add_argument("--append", action="store_true", help="Append samples after existing indices")
    return p.parse_args()


def split_start_index(out_dir: Path) -> int:
    if not out_dir.is_dir():
        return 0
    return len(list(out_dir.glob("*.npz")))


def clean_dir(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def save_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def make_positive_pair(
    ct,
    rng: np.random.Generator,
    *,
    max_retries: int,
) -> tuple[dict | None, dict | None]:
    for _ in range(max_retries):
        local = np.random.default_rng(int(rng.integers(0, 2**31 - 1)))
        base = misalign_to_windowed_volume(ct, local, is_positive=True)
        cls_vol = cls_preprocess(base["volume_pre"], local)
        pose_vol, _vol_axial, det_meta = detector_align_slab_pca_zrot(base["volume_pre"])
        if pose_vol is None:
            continue
        cls_sample = {
            "volume": cls_vol.astype(np.float16),
            "has_head": base["has_head"],
            "rotvec_corr_rad": base["rotvec_corr_rad"],
            "trans_corr_mm": base["trans_corr_mm"],
            "pre_detector": np.int8(1),
        }
        pose_sample = {
            "volume": pose_vol.astype(np.float16),
            "has_head": base["has_head"],
            "rotvec_corr_rad": base["rotvec_corr_rad"],
            "trans_corr_mm": base["trans_corr_mm"],
            "shift_zyx": det_meta["shift_zyx"],
        }
        return cls_sample, pose_sample
    return None, None


def make_negative_cls(ct, rng: np.random.Generator) -> dict:
    base = misalign_to_windowed_volume(ct, rng, is_positive=False)
    cls_vol = cls_preprocess(base["volume_pre"], rng)
    return {
        "volume": cls_vol.astype(np.float16),
        "has_head": base["has_head"],
        "rotvec_corr_rad": base["rotvec_corr_rad"],
        "trans_corr_mm": base["trans_corr_mm"],
        "pre_detector": np.int8(1),
    }


def generate_split(
    *,
    split: str,
    n_pos: int,
    n_neg: int,
    ideal_ids: list[str],
    neg_ids: list[str],
    get_ct,
    rng: np.random.Generator,
    cls_dir: Path,
    pose_dir: Path,
    max_retries: int,
    cls_start: int = 0,
    pose_start: int = 0,
) -> dict:
    cls_out = cls_dir / split
    pose_out = pose_dir / split
    cls_out.mkdir(parents=True, exist_ok=True)
    pose_out.mkdir(parents=True, exist_ok=True)

    cls_idx = int(cls_start)
    pose_idx = int(pose_start)
    pos_fail = 0
    neg_done = 0

    for _ in tqdm(range(n_pos), desc=f"{split} pos"):
        case_id = ideal_ids[int(rng.integers(0, len(ideal_ids)))]
        cls_sample, pose_sample = make_positive_pair(
            get_ct(case_id), rng, max_retries=max_retries
        )
        if cls_sample is None:
            pos_fail += 1
            continue
        save_npz(cls_out / f"{split}_{cls_idx:05d}.npz", case_id=case_id, **cls_sample)
        save_npz(pose_out / f"{split}_{pose_idx:05d}.npz", case_id=case_id, **pose_sample)
        cls_idx += 1
        pose_idx += 1

    for _ in tqdm(range(n_neg), desc=f"{split} neg"):
        case_id = neg_ids[int(rng.integers(0, len(neg_ids)))]
        sample = make_negative_cls(get_ct(case_id), np.random.default_rng(int(rng.integers(0, 2**31 - 1))))
        save_npz(cls_out / f"{split}_{cls_idx:05d}.npz", case_id=case_id, **sample)
        cls_idx += 1
        neg_done += 1

    return {
        "n_cls_added": cls_idx - cls_start,
        "n_pose_added": pose_idx - pose_start,
        "n_cls_total": cls_idx,
        "n_pose_total": pose_idx,
        "n_pos_requested": n_pos,
        "n_neg_requested": n_neg,
        "n_neg_done": neg_done,
        "n_pos_fail": pos_fail,
    }


def main() -> None:
    args = parse_args()
    base = args.base_dir
    ideal_ids = collect_case_ids(base / "ideal_heads_only")
    neg_ids = collect_case_ids(base / "no_heads")
    vol_dir = base / "volumes"

    if args.clean and args.append:
        raise SystemExit("Use either --clean or --append, not both")

    if args.clean:
        clean_dir(args.cls_dir)
        clean_dir(args.pose_dir)
    else:
        args.cls_dir.mkdir(parents=True, exist_ok=True)
        args.pose_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    cache: dict[str, object] = {}

    def get_ct(case_id: str):
        if case_id not in cache:
            cache[case_id] = prepare_isotropic_ct(
                vol_dir / f"{case_id}.nii.gz", spacing_mm=float(args.spacing_mm)
            )
        return cache[case_id]

    stats = {}
    for split, n_pos, n_neg in (
        ("train", args.n_train_pos, args.n_train_neg),
        ("val", args.n_val_pos, args.n_val_neg),
    ):
        stats[split] = generate_split(
            split=split,
            n_pos=n_pos,
            n_neg=n_neg,
            ideal_ids=ideal_ids,
            neg_ids=neg_ids,
            get_ct=get_ct,
            rng=rng,
            cls_dir=args.cls_dir,
            pose_dir=args.pose_dir,
            max_retries=args.max_detector_retries,
            cls_start=split_start_index(args.cls_dir / split),
            pose_start=split_start_index(args.pose_dir / split),
        )

    manifest_path_cls = args.cls_dir / "manifest.json"
    if args.append and manifest_path_cls.is_file():
        meta = json.loads(manifest_path_cls.read_text(encoding="utf-8"))
    else:
        meta = {}
    meta.update(
        {
            "spacing_mm": float(args.spacing_mm),
            "out_shape_zyx": list(OUT_SHAPE),
            "pipeline": "4mm -> augment -> bottom_crop_48 -> cls_resize | detector_align_slab_pca_zrot",
            "ideal_cases": ideal_ids,
            "no_head_cases": neg_ids,
            "seed": int(args.seed),
            "append": bool(args.append),
        }
    )
    meta.update(stats)
    (args.cls_dir / "manifest.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (args.pose_dir / "manifest.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
