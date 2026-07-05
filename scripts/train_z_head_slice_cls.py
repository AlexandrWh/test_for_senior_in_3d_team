"""
Generate + train PreAligner Z-slice classifier (single entry point).

Pipeline:
  1. Generate dataset (guide labels -> positive/negative .npy)
  2. Repeat 3x: train N epochs -> in-place filter (pos prob < 0.5, neg prob > 0.5 removed)

Usage:
    python -u scripts/train_z_head_slice_cls.py --device cuda
    python -u scripts/train_z_head_slice_cls.py --skip-generate --device cuda
    python -u scripts/train_z_head_slice_cls.py --generate-only
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets.z_slice_head import (
    ZSliceNpyDataset,
    collate_slices,
    filter_z_slice_dataset_in_place,
    split_z_slice_files,
)
from models.pre_aligner import PreAligner
from paths import (
    SPACING_MM,
    TRAIN_BASE,
    TRAIN_GUIDE_LABELS_JSON,
    TRAIN_GUIDES_DIR,
    TRAIN_LOG_DIR,
    TRAIN_MPR_MANIFEST,
    TRAIN_NO_HEADS_LIST,
    TRAIN_VOLUMES,
    TRAIN_Z_SLICE_CLS_DIR,
    WEIGHTS_DIR,
)
from utils import apply_brain_ct_window, center_crop_yx_to_square, prepare_isotropic_ct, sitk_image_to_numpy
from utils.guide_labels import GuideLabel, load_guide_labels
from utils.labels import load_case_id_list

SLICE_SIZE = 56


def stage(title: str, detail: str = "") -> None:
    bar = "=" * 64
    msg = f"  {title}" + (f" — {detail}" if detail else "")
    print(f"\n{bar}\n{msg}\n{bar}", flush=True)


# --- dataset generation ---


def _center_crop_yx(arr_yx: np.ndarray, size: int) -> np.ndarray:
    h, w = arr_yx.shape
    out = np.zeros((size, size), dtype=np.float32)
    side_y = min(h, size)
    side_x = min(w, size)
    y0 = max(0, (h - side_y) // 2)
    x0 = max(0, (w - side_x) // 2)
    dy0 = (size - side_y) // 2
    dx0 = (size - side_x) // 2
    out[dy0 : dy0 + side_y, dx0 : dx0 + side_x] = arr_yx[y0 : y0 + side_y, x0 : x0 + side_x]
    return out


def _random_crop_yx(arr_yx: np.ndarray, size: int, rng: np.random.Generator) -> np.ndarray:
    h, w = arr_yx.shape
    if h >= size and w >= size:
        y0 = int(rng.integers(0, h - size + 1))
        x0 = int(rng.integers(0, w - size + 1))
        return arr_yx[y0 : y0 + size, x0 : x0 + size].astype(np.float32, copy=True)
    return _center_crop_yx(arr_yx, size)


def _is_empty_slice(hu_yx: np.ndarray, *, empty_win_frac: float, empty_air_frac: float) -> bool:
    if float((hu_yx < -500.0).mean()) >= empty_air_frac:
        return True
    win = apply_brain_ct_window(hu_yx, output_range=(0.0, 1.0))
    return float((win > 0.02).mean()) < empty_win_frac


def _prepare_volume(case_id: str, vol_dir: Path, spacing_mm: float) -> np.ndarray:
    ct = prepare_isotropic_ct(vol_dir / f"{case_id}.nii.gz", spacing_mm=spacing_mm)
    hu = sitk_image_to_numpy(ct).astype(np.float32)
    return center_crop_yx_to_square(hu)


def _save_slice(path: Path, slice_yx: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, slice_yx.astype(np.float16))


def _clip_z_span(z_lo: int, z_hi: int, n_z: int) -> tuple[int, int]:
    z_lo = max(0, int(z_lo))
    z_hi = min(int(z_hi), n_z - 1)
    if z_lo > z_hi:
        z_lo, z_hi = z_hi, z_lo
    return z_lo, z_hi


def _load_labels_from_json(path: Path) -> list[GuideLabel]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    out: list[GuideLabel] = []
    for case_id, row in payload.get("cases", {}).items():
        out.append(
            GuideLabel(
                case_id=case_id,
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
        )
    return out


def _process_labeled_cases(
    labels: list[GuideLabel],
    *,
    vol_dir: Path,
    pos_dir: Path,
    neg_dir: Path,
    spacing_mm: float,
    slice_size: int,
    empty_win_frac: float,
    empty_air_frac: float,
) -> dict:
    stats = {"pos_slices": 0, "neg_outside_span": 0, "neg_empty_in_span": 0, "cases": len(labels)}
    for label in tqdm(labels, desc="guide-labeled volumes"):
        hu = _prepare_volume(label.case_id, vol_dir, spacing_mm)
        win = apply_brain_ct_window(hu, output_range=(0.0, 1.0))
        z_lo, z_hi = _clip_z_span(label.z_lo, label.z_hi, int(hu.shape[0]))
        for z in range(int(hu.shape[0])):
            sl_hu = hu[z]
            sl_win = _center_crop_yx(win[z], slice_size)
            empty = _is_empty_slice(sl_hu, empty_win_frac=empty_win_frac, empty_air_frac=empty_air_frac)
            in_span = z_lo <= z <= z_hi
            fname = f"{label.case_id}_z{z:04d}.npy"
            if in_span and not empty:
                _save_slice(pos_dir / fname, sl_win)
                stats["pos_slices"] += 1
            else:
                _save_slice(neg_dir / fname, sl_win)
                if in_span and empty:
                    stats["neg_empty_in_span"] += 1
                else:
                    stats["neg_outside_span"] += 1
    return stats


def _process_negative_cases(
    case_ids: list[str],
    *,
    vol_dir: Path,
    neg_dir: Path,
    spacing_mm: float,
    slice_size: int,
    neg_crops_per_volume: int,
    rng: np.random.Generator,
) -> dict:
    stats = {"neg_random_crops": 0, "cases": len(case_ids)}
    for case_id in tqdm(case_ids, desc="no-head crops"):
        hu = _prepare_volume(case_id, vol_dir, spacing_mm)
        win = apply_brain_ct_window(hu, output_range=(0.0, 1.0))
        z_len = win.shape[0]
        for i in range(neg_crops_per_volume):
            z = int(rng.integers(0, z_len))
            crop = _random_crop_yx(win[z], slice_size, rng)
            fname = f"{case_id}_rand{i:05d}_z{z:04d}.npy"
            _save_slice(neg_dir / fname, crop)
            stats["neg_random_crops"] += 1
    return stats


def generate_dataset(args: argparse.Namespace) -> dict:
    out_dir = args.data_dir
    base = args.base_dir
    vol_dir = base / "volumes" if (base / "volumes").is_dir() else TRAIN_VOLUMES
    no_heads_list = args.no_heads_list

    if not args.guide_labels.is_file():
        print("  guide_labels.json not found, parsing guides dir...", flush=True)

    if out_dir.is_dir():
        print(f"  removing existing dataset: {out_dir}", flush=True)
        shutil.rmtree(out_dir)

    pos_dir = out_dir / "positive"
    neg_dir = out_dir / "negative"
    pos_dir.mkdir(parents=True, exist_ok=True)
    neg_dir.mkdir(parents=True, exist_ok=True)

    neg_ids = load_case_id_list(no_heads_list)
    if not neg_ids:
        raise SystemExit(f"No case IDs in no-heads list: {no_heads_list}")
    print(f"  no-head cases ({len(neg_ids)}): {', '.join(neg_ids)}", flush=True)
    rng = np.random.default_rng(args.seed)

    if args.guide_labels.is_file():
        labels = _load_labels_from_json(args.guide_labels)
    else:
        labels, _ = load_guide_labels(
            TRAIN_GUIDES_DIR,
            manifest_path=TRAIN_MPR_MANIFEST,
            volumes_dir=vol_dir,
            cls_spacing_mm=float(args.spacing_mm),
        )
    if not labels:
        raise SystemExit("No guide labels found; run export_guide_labels.py first")
    if args.limit_pos > 0:
        labels = labels[: args.limit_pos]

    pos_stats = _process_labeled_cases(
        labels,
        vol_dir=vol_dir,
        pos_dir=pos_dir,
        neg_dir=neg_dir,
        spacing_mm=float(args.spacing_mm),
        slice_size=int(args.slice_size),
        empty_win_frac=float(args.empty_win_frac),
        empty_air_frac=float(args.empty_air_frac),
    )
    neg_stats = _process_negative_cases(
        neg_ids,
        vol_dir=vol_dir,
        neg_dir=neg_dir,
        spacing_mm=float(args.spacing_mm),
        slice_size=int(args.slice_size),
        neg_crops_per_volume=int(args.neg_crops_per_volume),
        rng=rng,
    )
    total_neg = pos_stats["neg_outside_span"] + pos_stats["neg_empty_in_span"] + neg_stats["neg_random_crops"]
    manifest = {
        "mode": "guide_labels",
        "guide_labels_path": str(args.guide_labels),
        "spacing_mm": float(args.spacing_mm),
        "slice_size": int(args.slice_size),
        "positive_cases": [lb.case_id for lb in labels],
        "negative_no_head_cases": neg_ids,
        "total_positive": pos_stats["pos_slices"],
        "total_negative": total_neg,
        "pos_stats": pos_stats,
        "neg_stats": neg_stats,
    }

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


# --- training ---


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate + train PreAligner Z-slice classifier")
    p.add_argument("--generate-only", action="store_true", help="only build dataset, skip training")
    p.add_argument("--skip-generate", action="store_true", help="use existing dataset under --data-dir")
    p.add_argument("--base-dir", type=Path, default=TRAIN_BASE)
    p.add_argument("--data-dir", type=Path, default=TRAIN_Z_SLICE_CLS_DIR)
    p.add_argument("--guide-labels", type=Path, default=TRAIN_GUIDE_LABELS_JSON)
    p.add_argument("--spacing-mm", type=float, default=SPACING_MM)
    p.add_argument("--slice-size", type=int, default=SLICE_SIZE)
    p.add_argument("--empty-win-frac", type=float, default=0.01)
    p.add_argument("--empty-air-frac", type=float, default=0.94)
    p.add_argument("--neg-crops-per-volume", type=int, default=800)
    p.add_argument("--limit-pos", type=int, default=0, help="0 = all cases")
    p.add_argument("--no-heads-list", type=Path, default=TRAIN_NO_HEADS_LIST)
    p.add_argument("--log-dir", type=Path, default=TRAIN_LOG_DIR)
    p.add_argument("--weights-dir", type=Path, default=WEIGHTS_DIR)
    p.add_argument("--epochs-per-cycle", type=int, default=5)
    p.add_argument("--train-cycles", type=int, default=3)
    p.add_argument("--filter-threshold", type=float, default=0.5)
    p.add_argument("--filter-batch-size", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--tag", type=str, default="")
    return p.parse_args()


def train_epoch(model: PreAligner, loader, opt, device) -> float:
    model.train()
    loss_sum, n = 0.0, 0
    for batch in loader:
        sl = batch["slice"].to(device)
        y = batch["has_head"].to(device)
        opt.zero_grad(set_to_none=True)
        logit = model(sl)
        loss = F.binary_cross_entropy_with_logits(logit, y)
        loss.backward()
        opt.step()
        loss_sum += float(loss.item())
        n += 1
    return loss_sum / max(n, 1)


@torch.no_grad()
def eval_epoch(model: PreAligner, loader, device) -> dict[str, float]:
    model.eval()
    ok, total = 0, 0
    tp = fp = tn = fn = 0
    for batch in loader:
        sl = batch["slice"].to(device)
        y = batch["has_head"].to(device)
        pred = (torch.sigmoid(model(sl)) >= 0.5).float()
        ok += int((pred == y).sum().item())
        total += int(y.numel())
        tp += int(((pred == 1) & (y == 1)).sum().item())
        fp += int(((pred == 1) & (y == 0)).sum().item())
        tn += int(((pred == 0) & (y == 0)).sum().item())
        fn += int(((pred == 0) & (y == 1)).sum().item())
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    return {"acc": ok / max(total, 1), "precision": prec, "recall": rec, "tp": tp, "fp": fp, "tn": tn, "fn": fn}


def make_loaders(data_dir: Path, args):
    train_files, train_labels, val_files, val_labels = split_z_slice_files(
        data_dir, val_frac=args.val_frac, seed=args.seed
    )
    train_ds = ZSliceNpyDataset(train_files, train_labels)
    val_ds = ZSliceNpyDataset(val_files, val_labels)
    label_counts = {0.0: train_labels.count(0.0), 1.0: train_labels.count(1.0)}
    sample_weights = [1.0 / max(label_counts[l], 1) for l in train_labels]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, sampler=sampler, collate_fn=collate_slices, num_workers=0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_slices, num_workers=0,
    )
    return train_loader, val_loader, len(train_files), len(val_files), label_counts


def run_training(
    model: PreAligner,
    data_dir: Path,
    args,
    device: torch.device,
    *,
    epochs: int,
    phase: str,
    ckpt: Path,
    log: Path,
    history: list[dict],
) -> None:
    train_loader, val_loader, n_train, n_val, label_counts = make_loaders(data_dir, args)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_score = float("inf")

    with log.open("a", encoding="utf-8") as f:
        f.write(f"\n# phase={phase} data={data_dir} train={n_train} val={n_val} labels={label_counts}\n")
        f.write("epoch\ttrain_loss\tval_acc\tval_prec\tval_rec\n")
        for epoch in range(1, epochs + 1):
            tr_loss = train_epoch(model, train_loader, opt, device)
            ev = eval_epoch(model, val_loader, device)
            row = {"phase": phase, "epoch": epoch, "train_loss": tr_loss, **ev}
            history.append(row)
            f.write(f"{epoch}\t{tr_loss:.4f}\t{ev['acc']:.4f}\t{ev['precision']:.4f}\t{ev['recall']:.4f}\n")
            f.flush()
            print(
                f"[{phase}] epoch {epoch:03d} loss {tr_loss:.4f} "
                f"acc {ev['acc']:.3f} prec {ev['precision']:.3f} rec {ev['recall']:.3f}"
            )
            score = 1.0 - ev["acc"]
            if score < best_score:
                best_score = score
                model.save_checkpoint(ckpt, epoch=epoch)


def main() -> None:
    args = parse_args()
    if args.generate_only and args.skip_generate:
        raise SystemExit("Use either --generate-only or --skip-generate, not both.")

    if not args.skip_generate:
        stage("STEP 1/4", "Dataset generation")
        manifest = generate_dataset(args)
        print(
            f"  done: positive={manifest['total_positive']} negative={manifest['total_negative']}\n"
            f"  saved -> {args.data_dir}",
            flush=True,
        )
        if args.generate_only:
            return

    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    )
    args.log_dir.mkdir(parents=True, exist_ok=True)
    args.weights_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.tag}" if args.tag else ""
    ckpt = args.weights_dir / f"pre_aligner_best{suffix}.pt"
    log = args.log_dir / f"pre_aligner_train{suffix}.log"
    history: list[dict] = []
    filter_history: list[dict] = []

    stage(
        "STEP 2/4",
        f"Training on {args.data_dir} | device={device} | "
        f"{args.train_cycles} cycles x {args.epochs_per_cycle} epochs",
    )
    print(f"  checkpoint -> {ckpt}", flush=True)

    model = PreAligner().to(device)

    with log.open("w", encoding="utf-8") as f:
        f.write(f"started={datetime.now().isoformat()}\n")
        f.write(f"train_cycles={args.train_cycles} epochs_per_cycle={args.epochs_per_cycle}\n")
        f.write(f"filter_threshold={args.filter_threshold}\n")

    for cycle in range(1, args.train_cycles + 1):
        phase = f"cycle{cycle}"
        print(f"\n  --- train {phase}/{args.train_cycles} ({args.epochs_per_cycle} epochs) ---", flush=True)
        run_training(
            model, args.data_dir, args, device,
            epochs=args.epochs_per_cycle, phase=phase, ckpt=ckpt, log=log, history=history,
        )
        if cycle < args.train_cycles:
            stage("STEP 3/4", f"In-place filter after {phase} (threshold={args.filter_threshold})")
            filter_stats = filter_z_slice_dataset_in_place(
                args.data_dir, model, device,
                threshold=args.filter_threshold, batch_size=args.filter_batch_size,
            )
            filter_stats["cycle"] = cycle
            filter_history.append(filter_stats)
            print(
                f"  positive: {filter_stats['pos_in']} -> {filter_stats['pos_kept']} "
                f"(removed {filter_stats['pos_removed']})\n"
                f"  negative: {filter_stats['neg_in']} -> {filter_stats['neg_kept']} "
                f"(removed {filter_stats['neg_removed']})",
                flush=True,
            )

    meta = {
        "history": history,
        "filter_history": filter_history,
        "checkpoint": str(ckpt),
        "data_dir": str(args.data_dir),
    }
    (args.log_dir / f"pre_aligner_history{suffix}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    stage("STEP 4/4", "Pipeline complete")
    print(f"  best checkpoint -> {ckpt}", flush=True)


if __name__ == "__main__":
    main()
