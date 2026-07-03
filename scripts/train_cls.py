"""
Train head classifier (all samples: pos + neg).

Usage:
    python -u scripts/train_cls.py --epochs 20 --device cuda
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from head_align.dataset import NpzDataset, collate_fixed
from head_align.model import FullScanClsNet
from head_align.paths import TRAIN_LOG_DIR, WEIGHTS_DIR


def train_epoch(model, loader, opt, device):
    model.train()
    loss_sum, n = 0.0, 0
    for batch in loader:
        vol = batch["volume"].to(device)
        has_head = batch["has_head"].to(device)
        opt.zero_grad(set_to_none=True)
        logit = model(vol)
        loss = F.binary_cross_entropy_with_logits(logit, has_head)
        loss.backward()
        opt.step()
        loss_sum += float(loss.item())
        n += 1
    return loss_sum / max(n, 1)


@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    ok, total = 0, 0
    for batch in loader:
        vol = batch["volume"].to(device)
        has_head = batch["has_head"].to(device)
        logit = model(vol)
        pred = (torch.sigmoid(logit) >= 0.5).float()
        ok += int((pred == has_head).sum().item())
        total += int(has_head.numel())
    return {"cls_acc": ok / max(total, 1)}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=ROOT / "data" / "head_align_cls")
    p.add_argument("--log-dir", type=Path, default=TRAIN_LOG_DIR)
    p.add_argument("--weights-dir", type=Path, default=WEIGHTS_DIR)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--tag", type=str, default="", help="suffix for log/ckpt files, e.g. v2")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)
    train_loader = DataLoader(
        NpzDataset(args.data_dir / "train"),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fixed,
    )
    val_loader = DataLoader(
        NpzDataset(args.data_dir / "val"),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fixed,
    )
    model = FullScanClsNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    args.weights_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.tag}" if args.tag else ""
    ckpt = args.weights_dir / f"head_align_cls_best{suffix}.pt"
    log = args.log_dir / f"head_align_cls_train{suffix}.log"
    best = float("inf")
    history = []

    with log.open("w", encoding="utf-8") as f:
        f.write(f"started={datetime.now().isoformat()}\n")
        f.write("epoch\ttrain_loss\tval_cls_acc\n")
        for epoch in range(1, args.epochs + 1):
            tr_loss = train_epoch(model, train_loader, opt, device)
            ev = eval_epoch(model, val_loader, device)
            history.append({"epoch": epoch, "train_loss": tr_loss, **ev})
            f.write(f"{epoch}\t{tr_loss:.4f}\t{ev['cls_acc']:.4f}\n")
            f.flush()
            print(f"epoch {epoch:03d} loss {tr_loss:.4f} cls {ev['cls_acc']:.3f}")
            score = 1.0 - ev["cls_acc"]
            if score < best:
                best = score
                torch.save(
                    {"model_state": model.state_dict(), "epoch": epoch, "base_channels": 16},
                    ckpt,
                )

    (args.log_dir / f"head_align_cls_history{suffix}.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"Best -> {ckpt}")


if __name__ == "__main__":
    main()
