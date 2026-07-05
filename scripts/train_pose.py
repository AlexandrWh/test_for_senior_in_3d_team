"""
Train 3D pose regressor on pose_dataset volumes.

Weighted L1 on (rz, ry, rx) in radians; val split 80/20 by unique case_id.

Usage:
    python -u scripts/train_pose.py --device cuda
    python -u scripts/train_pose.py --epochs 30 --batch-size 4 --device cuda
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets.pose_volume import (
    PoseVolumeDataset,
    collate_pose_volumes,
    list_pose_samples,
    split_pose_by_case,
)
from models.pose_regressor import PoseRegressor3D
from paths import (
    DEFAULT_POSE_REGRESSOR_CKPT,
    POSE_ANGLE_LOSS_WEIGHTS,
    TRAIN_LOG_DIR,
    TRAIN_POSE_DATASET_DIR,
    TRAIN_POSE_DATASET_META,
    TRAIN_POSE_DATASET_VOLUMES,
    WEIGHTS_DIR,
)


def stage(title: str, detail: str = "") -> None:
    bar = "=" * 64
    msg = f"  {title}" + (f" - {detail}" if detail else "")
    print(f"\n{bar}\n{msg}\n{bar}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train pose regressor (rz, ry, rx)")
    p.add_argument("--dataset-dir", type=Path, default=TRAIN_POSE_DATASET_DIR)
    p.add_argument("--meta-dir", type=Path, default=None)
    p.add_argument("--volumes-dir", type=Path, default=None)
    p.add_argument("--weights-dir", type=Path, default=WEIGHTS_DIR)
    p.add_argument("--log-dir", type=Path, default=TRAIN_LOG_DIR)
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_POSE_REGRESSOR_CKPT)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--base-channels", type=int, default=12)
    p.add_argument("--mlp-hidden", type=int, default=32)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--no-augment", action="store_true", help="disable train texture augmentations")
    return p.parse_args()


def weighted_l1_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    weights: tuple[float, float, float],
) -> torch.Tensor:
    w = pred.new_tensor(weights).view(1, 3)
    return (w * (pred - target).abs()).sum(dim=1).mean()


@torch.no_grad()
def angle_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    err = (pred - target).abs()
    names = ("rz", "ry", "rx")
    out: dict[str, float] = {}
    for i, name in enumerate(names):
        out[f"mae_{name}_rad"] = float(err[:, i].mean().item())
        out[f"mae_{name}_deg"] = float(math.degrees(err[:, i].mean().item()))
    out["mae_mean_rad"] = float(err.mean().item())
    out["mae_mean_deg"] = float(math.degrees(err.mean().item()))
    return out


def train_epoch(
    model: PoseRegressor3D,
    loader: DataLoader,
    opt: torch.optim.Optimizer,
    device: torch.device,
    weights: tuple[float, float, float],
) -> float:
    model.train()
    loss_sum = 0.0
    n = 0
    for batch in tqdm(loader, desc="train", leave=False):
        vol = batch["volume"].to(device)
        y = batch["angles"].to(device)
        z_len = batch["z_len"].to(device)
        opt.zero_grad(set_to_none=True)
        pred = model(vol, z_len=z_len)
        loss = weighted_l1_loss(pred, y, weights)
        loss.backward()
        opt.step()
        loss_sum += float(loss.item())
        n += 1
    return loss_sum / max(n, 1)


@torch.no_grad()
def eval_epoch(
    model: PoseRegressor3D,
    loader: DataLoader,
    device: torch.device,
    weights: tuple[float, float, float],
) -> dict[str, float]:
    model.eval()
    loss_sum = 0.0
    n = 0
    metric_acc: dict[str, float] = {}
    metric_n = 0

    for batch in loader:
        vol = batch["volume"].to(device)
        y = batch["angles"].to(device)
        z_len = batch["z_len"].to(device)
        pred = model(vol, z_len=z_len)
        loss = weighted_l1_loss(pred, y, weights)
        loss_sum += float(loss.item())
        n += 1

        m = angle_metrics(pred, y)
        for k, v in m.items():
            metric_acc[k] = metric_acc.get(k, 0.0) + v
        metric_n += 1

    out = {"val_loss": loss_sum / max(n, 1)}
    for k, v in metric_acc.items():
        out[k] = v / max(metric_n, 1)
    return out


def save_curves(history: list[dict], out_path: Path) -> None:
    if not history:
        return
    epochs = [int(r["epoch"]) for r in history]
    train_loss = [float(r["train_loss"]) for r in history]
    val_loss = [float(r["val_loss"]) for r in history]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=120)

    ax = axes[0]
    ax.plot(epochs, train_loss, label="train L1", linewidth=1.8)
    ax.plot(epochs, val_loss, label="val L1", linewidth=1.8)
    ax.set_xlabel("epoch")
    ax.set_ylabel("weighted L1 (rad)")
    ax.set_title("Loss")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1]
    for key, label in (
        ("mae_rz_deg", "rz MAE"),
        ("mae_ry_deg", "ry MAE"),
        ("mae_rx_deg", "rx MAE"),
        ("mae_mean_deg", "mean MAE"),
    ):
        if key in history[0]:
            ax.plot(epochs, [float(r[key]) for r in history], label=label, linewidth=1.6)
    ax.set_xlabel("epoch")
    ax.set_ylabel("MAE (deg)")
    ax.set_title("Val angle error")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    )

    dataset_dir = args.dataset_dir
    meta_dir = args.meta_dir or dataset_dir / "meta"
    vol_dir = args.volumes_dir or dataset_dir / "volumes"
    args.log_dir.mkdir(parents=True, exist_ok=True)
    args.weights_dir.mkdir(parents=True, exist_ok=True)

    samples = list_pose_samples(meta_dir, volumes_dir=vol_dir, limit=int(args.limit))
    if not samples:
        raise SystemExit(f"No samples in {meta_dir}")

    train_samples, val_samples, train_cases, val_cases = split_pose_by_case(
        samples, val_frac=float(args.val_frac), seed=int(args.seed)
    )
    if not train_samples or not val_samples:
        raise SystemExit(
            f"Need train and val cases (got {len(train_cases)} train / {len(val_cases)} val cases)"
        )

    split_path = args.log_dir / "pose_split.json"
    split_path.write_text(
        json.dumps(
            {
                "val_frac": float(args.val_frac),
                "seed": int(args.seed),
                "n_samples_train": len(train_samples),
                "n_samples_val": len(val_samples),
                "n_cases_train": len(train_cases),
                "n_cases_val": len(val_cases),
                "train_cases": train_cases,
                "val_cases": val_cases,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    stage(
        "Pose regressor train",
        f"{len(train_samples)} train / {len(val_samples)} val samples | "
        f"{len(train_cases)} / {len(val_cases)} cases",
    )
    print(f"  device:     {device}", flush=True)
    print(f"  loss w:     rz,ry,rx = {POSE_ANGLE_LOSS_WEIGHTS}", flush=True)
    print(f"  checkpoint: {args.checkpoint}", flush=True)
    print(f"  split:      {split_path}", flush=True)
    print(f"  texture_aug: {not args.no_augment}", flush=True)

    train_loader = DataLoader(
        PoseVolumeDataset(train_samples, augment=not args.no_augment),
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        collate_fn=collate_pose_volumes,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        PoseVolumeDataset(val_samples, augment=False),
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        collate_fn=collate_pose_volumes,
        pin_memory=device.type == "cuda",
    )

    model = PoseRegressor3D(
        base_channels=int(args.base_channels),
        mlp_hidden=int(args.mlp_hidden),
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(args.lr))

    log_path = args.log_dir / "pose_train.log"
    curves_path = args.log_dir / "pose_train_curves.png"
    history_path = args.log_dir / "pose_train_history.json"
    history: list[dict] = []
    best_val = float("inf")

    with log_path.open("w", encoding="utf-8") as f:
        f.write(f"# started {datetime.now().isoformat()}\n")
        f.write(f"# train_samples={len(train_samples)} val_samples={len(val_samples)}\n")
        f.write(f"# loss_weights={POSE_ANGLE_LOSS_WEIGHTS}\n")
        f.write(f"# texture_aug={not args.no_augment}\n")
        f.write(
            "epoch\ttrain_loss\tval_loss\t"
            "mae_rz_deg\tmae_ry_deg\tmae_rx_deg\tmae_mean_deg\n"
        )

        for epoch in range(1, int(args.epochs) + 1):
            tr_loss = train_epoch(model, train_loader, opt, device, POSE_ANGLE_LOSS_WEIGHTS)
            ev = eval_epoch(model, val_loader, device, POSE_ANGLE_LOSS_WEIGHTS)
            row = {"epoch": epoch, "train_loss": tr_loss, **ev}
            history.append(row)

            f.write(
                f"{epoch}\t{tr_loss:.6f}\t{ev['val_loss']:.6f}\t"
                f"{ev['mae_rz_deg']:.4f}\t{ev['mae_ry_deg']:.4f}\t"
                f"{ev['mae_rx_deg']:.4f}\t{ev['mae_mean_deg']:.4f}\n"
            )
            f.flush()

            print(
                f"epoch {epoch:03d}  train {tr_loss:.4f}  val {ev['val_loss']:.4f}  "
                f"MAE deg rz={ev['mae_rz_deg']:.2f} ry={ev['mae_ry_deg']:.2f} "
                f"rx={ev['mae_rx_deg']:.2f} mean={ev['mae_mean_deg']:.2f}",
                flush=True,
            )

            if ev["val_loss"] < best_val:
                best_val = ev["val_loss"]
                model.save_checkpoint(
                    args.checkpoint,
                    epoch=epoch,
                    val_loss=best_val,
                    loss_weights=POSE_ANGLE_LOSS_WEIGHTS,
                )

            save_curves(history, curves_path)

    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"\n  best val loss: {best_val:.6f}", flush=True)
    print(f"  checkpoint -> {args.checkpoint}", flush=True)
    print(f"  log        -> {log_path}", flush=True)
    print(f"  curves     -> {curves_path}", flush=True)
    print(f"  history    -> {history_path}", flush=True)


if __name__ == "__main__":
    main()
