"""

Train pose model (ROTATION ONLY) on positive samples only.



Losses:

  l1    - scaled L1 on rotvec / theta_max (default legacy)

  geo   - geodesic angle loss (rad)

  combo - w_l1 * L1 + w_geo * geodesic



Usage:

    python -u scripts/train_pose.py --epochs 400 --base-channels 32 --loss geo --tag v3_c32_geo

"""



from __future__ import annotations



import argparse

import json

import sys

from datetime import datetime

from pathlib import Path



import numpy as np

import torch

import torch.nn.functional as F

from torch.utils.data import DataLoader



ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT))



from head_align.dataset import NpzDataset, collate_fixed

from head_align.model import FullScanPoseNet

from head_align.paths import TRAIN_LOG_DIR, WEIGHTS_DIR

from head_align.rigid import geodesic_deg_np, geodesic_rot_loss





def rot_loss(

    rotvec_pred: torch.Tensor,

    rotvec_gt: torch.Tensor,

    *,

    loss: str,

    theta_max: torch.Tensor,

    w_l1: float,

    w_geo: float,

) -> torch.Tensor:

    if loss == "l1":

        return w_l1 * F.l1_loss(rotvec_pred / theta_max, rotvec_gt / theta_max)

    if loss == "geo":

        return w_geo * geodesic_rot_loss(rotvec_pred, rotvec_gt)

    l1 = F.l1_loss(rotvec_pred / theta_max, rotvec_gt / theta_max)

    geo = geodesic_rot_loss(rotvec_pred, rotvec_gt)

    return w_l1 * l1 + w_geo * geo





def train_epoch(model, loader, opt, device, args, theta_max):

    model.train()

    loss_sum, n = 0.0, 0

    for batch in loader:

        vol = batch["volume"].to(device)

        rotvec_gt = batch["rotvec"].to(device)

        opt.zero_grad(set_to_none=True)

        rotvec, _trans = model(vol)

        loss = rot_loss(

            rotvec,

            rotvec_gt,

            loss=args.loss,

            theta_max=theta_max,

            w_l1=args.w_l1,

            w_geo=args.w_geo,

        )

        loss.backward()

        opt.step()

        loss_sum += float(loss.item())

        n += 1

    return loss_sum / max(n, 1)





@torch.no_grad()

def eval_epoch(model, loader, device):

    model.eval()

    geos: list[float] = []

    for batch in loader:

        vol = batch["volume"].to(device)

        rotvec_gt = batch["rotvec"].to(device)

        rotvec, _trans = model(vol)

        for rp, rg in zip(rotvec.cpu().numpy(), rotvec_gt.cpu().numpy()):

            geos.append(geodesic_deg_np(rp, rg))

    return {

        "geo_mae_deg": float(np.mean(geos)) if geos else 0.0,

    }





def parse_args():

    p = argparse.ArgumentParser()

    p.add_argument("--data-dir", type=Path, default=ROOT / "data" / "head_align_pose")

    p.add_argument("--log-dir", type=Path, default=TRAIN_LOG_DIR)

    p.add_argument("--weights-dir", type=Path, default=WEIGHTS_DIR)

    p.add_argument("--epochs", type=int, default=100)

    p.add_argument("--batch-size", type=int, default=8)

    p.add_argument("--base-channels", type=int, default=16)

    p.add_argument("--lr", type=float, default=1e-3)

    p.add_argument("--loss", choices=("l1", "geo", "combo"), default="l1")

    p.add_argument("--w-l1", type=float, default=1.0)

    p.add_argument("--w-geo", type=float, default=1.0)

    p.add_argument("--theta-max-deg", type=float, default=45.0)

    p.add_argument("--cosine-lr", action="store_true")

    p.add_argument("--num-workers", type=int, default=0)

    p.add_argument("--device", type=str, default="auto")

    p.add_argument("--tag", type=str, default="", help="suffix for log/ckpt files")

    return p.parse_args()





def main():

    args = parse_args()

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)

    train_loader = DataLoader(

        NpzDataset(args.data_dir / "train", positives_only=True),

        batch_size=args.batch_size,

        shuffle=True,

        num_workers=args.num_workers,

        collate_fn=collate_fixed,

        pin_memory=device.type == "cuda",

    )

    val_loader = DataLoader(

        NpzDataset(args.data_dir / "val", positives_only=True),

        batch_size=args.batch_size,

        shuffle=False,

        num_workers=args.num_workers,

        collate_fn=collate_fixed,

        pin_memory=device.type == "cuda",

    )

    model = FullScanPoseNet(base_channels=args.base_channels).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    sched = (

        torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr * 0.05)

        if args.cosine_lr

        else None

    )

    args.log_dir.mkdir(parents=True, exist_ok=True)

    args.weights_dir.mkdir(parents=True, exist_ok=True)

    suffix = f"_{args.tag}" if args.tag else ""

    ckpt = args.weights_dir / f"head_align_pose_best{suffix}.pt"

    log = args.log_dir / f"head_align_pose_train{suffix}.log"

    run_log = args.log_dir / f"pose_run{suffix}.log"

    best = float("inf")

    history = []

    theta_max = torch.tensor(np.deg2rad(args.theta_max_deg), device=device)



    meta = {

        "base_channels": args.base_channels,

        "loss": args.loss,

        "w_l1": args.w_l1,

        "w_geo": args.w_geo,

        "batch_size": args.batch_size,

        "lr": args.lr,

        "train_n": len(train_loader.dataset),

        "val_n": len(val_loader.dataset),

    }

    print(

        f"pose train tag={args.tag or 'default'} loss={args.loss} "

        f"base_c={args.base_channels} bs={args.batch_size} "

        f"train={meta['train_n']} val={meta['val_n']} epochs={args.epochs}"

    )



    with log.open("w", encoding="utf-8") as f, run_log.open("w", encoding="utf-8") as rf:

        header = (

            f"started={datetime.now().isoformat()}\n"

            f"meta={json.dumps(meta)}\n"

            f"theta_max_deg={args.theta_max_deg}\n"

        )

        f.write(header)

        rf.write(header)

        rf.write(f"loss scale: theta_max={args.theta_max_deg}deg loss={args.loss}\n")

        f.write("epoch\ttrain_loss\tval_geo_mae_deg\tlr\n")

        for epoch in range(1, args.epochs + 1):

            tr_loss = train_epoch(model, train_loader, opt, device, args, theta_max)

            ev = eval_epoch(model, val_loader, device)

            lr_now = float(opt.param_groups[0]["lr"])

            if sched is not None:

                sched.step()

            history.append({"epoch": epoch, "train_loss": tr_loss, "lr": lr_now, **ev})

            line = f"{epoch}\t{tr_loss:.4f}\t{ev['geo_mae_deg']:.4f}\t{lr_now:.6f}\n"

            f.write(line)

            f.flush()

            msg = f"epoch {epoch:03d} loss {tr_loss:.4f} geo {ev['geo_mae_deg']:.2f}°"

            print(msg)

            rf.write(msg + "\n")

            rf.flush()

            score = ev["geo_mae_deg"]

            if score < best:

                best = score

                torch.save(

                    {

                        "model_state": model.state_dict(),

                        "epoch": epoch,

                        "base_channels": args.base_channels,

                        "loss": args.loss,

                        "geo_mae_deg": score,

                    },

                    ckpt,

                )



    hist_path = args.log_dir / f"head_align_pose_history{suffix}.json"

    hist_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

    print(f"Best geo {best:.2f}° -> {ckpt}")





if __name__ == "__main__":

    main()


