"""
Golden eval via HeadAligner HTTP API (or offline fallback).

Usage:
    docker compose up -d
    python -u scripts/run_head_align_golden.py --service-url http://localhost:8000

    python -u scripts/run_head_align_golden.py --offline --device cuda
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import sys
import zipfile
from pathlib import Path

import httpx
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from paths import (
    DEFAULT_POSE_REGRESSOR_CKPT,
    DEFAULT_PRE_ALIGNER_CKPT,
    TEST_ALIGN_DIR,
    TEST_VOLUMES,
)
from utils.rigid import save_volume_nifti


def stage(title: str, detail: str = "") -> None:
    bar = "=" * 64
    msg = f"  {title}" + (f" — {detail}" if detail else "")
    print(f"\n{bar}\n{msg}\n{bar}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="HeadAligner golden eval")
    p.add_argument("--volumes-dir", type=Path, default=TEST_VOLUMES)
    p.add_argument("--out-dir", type=Path, default=TEST_ALIGN_DIR)
    p.add_argument(
        "--service-url",
        type=str,
        default=os.environ.get("ALIGN_SERVICE_URL", "").strip(),
        help="Head align API base URL (default: ALIGN_SERVICE_URL env)",
    )
    p.add_argument(
        "--offline",
        action="store_true",
        help="call HeadAligner in-process (no HTTP)",
    )
    p.add_argument("--timeout", type=float, default=600.0, help="HTTP timeout per case (sec)")
    p.add_argument("--pre-align-ckpt", type=Path, default=DEFAULT_PRE_ALIGNER_CKPT)
    p.add_argument("--pose-ckpt", type=Path, default=DEFAULT_POSE_REGRESSOR_CKPT)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--cls-threshold", type=float, default=0.5)
    p.add_argument("--cls-pad", type=int, default=3)
    p.add_argument("--cls-min-head-slices", type=int, default=10)
    p.add_argument("--limit", type=int, default=0)
    return p.parse_args()


def _row_from_meta(meta: dict, *, aligned_path: str | None) -> dict:
    case_id = str(meta.get("case_id", ""))
    status = str(meta.get("status", "error"))
    row: dict = {"case_id": case_id, "status": status}
    if status == "ok" and meta.get("has_head"):
        pre = meta.get("prealign", {})
        pose = meta.get("pose", {})
        row.update(
            {
                "has_head": True,
                "aligned_path": aligned_path or "",
                "z_min": round(float(pre.get("z_min", 0)), 2),
                "z_max": round(float(pre.get("z_max", 0)), 2),
                "dx": round(float(pre.get("dx", 0)), 2),
                "dy": round(float(pre.get("dy", 0)), 2),
                "rz_pca_deg": round(float(pre.get("rz_pca_deg", 0)), 3),
                "rz_pose_deg": round(float(pose.get("rz_deg", 0)), 3),
                "ry_pose_deg": round(float(pose.get("ry_deg", 0)), 3),
                "rx_pose_deg": round(float(pose.get("rx_deg", 0)), 3),
            }
        )
    elif status == "no_head":
        row.update({"has_head": False})
    else:
        row.update({"has_head": bool(meta.get("has_head", False))})
    return row


def _save_zip_response(
    zip_bytes: bytes,
    *,
    case_id: str,
    out_vol_dir: Path,
    meta_dir: Path,
) -> tuple[dict, str | None]:
    aligned_path: str | None = None
    meta: dict = {"case_id": case_id, "status": "error"}
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        if "meta.json" in zf.namelist():
            meta = json.loads(zf.read("meta.json").decode("utf-8"))
        if "aligned.nii.gz" in zf.namelist():
            out_nii = out_vol_dir / f"{case_id}.nii.gz"
            out_nii.write_bytes(zf.read("aligned.nii.gz"))
            aligned_path = str(out_nii)
            meta["aligned_path"] = aligned_path
    (meta_dir / f"{case_id}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta, aligned_path


def run_via_service(
    paths: list[Path],
    *,
    service_url: str,
    out_dir: Path,
    timeout: float,
) -> list[dict]:
    out_vol_dir = out_dir / "volumes"
    meta_dir = out_dir / "meta"
    base = service_url.rstrip("/")
    rows: list[dict] = []

    with httpx.Client(base_url=base, timeout=timeout) as client:
        health = client.get("/health")
        health.raise_for_status()
        print(f"  service health: {health.json()}", flush=True)

        for path in tqdm(paths, desc="head align (api)"):
            case_id = path.name.replace(".nii.gz", "")
            row: dict = {"case_id": case_id, "status": "error"}
            try:
                with path.open("rb") as f:
                    resp = client.post(
                        "/align",
                        files={"file": (path.name, f, "application/gzip")},
                        data={"case_id": case_id},
                    )
                resp.raise_for_status()
                meta, aligned_path = _save_zip_response(
                    resp.content,
                    case_id=case_id,
                    out_vol_dir=out_vol_dir,
                    meta_dir=meta_dir,
                )
                row = _row_from_meta(meta, aligned_path=aligned_path)
            except Exception as exc:
                row["message"] = str(exc)
            rows.append(row)
    return rows


def run_offline(
    paths: list[Path],
    *,
    args: argparse.Namespace,
    out_dir: Path,
) -> list[dict]:
    from models.head_aligner import HeadAligner

    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    )
    out_vol_dir = out_dir / "volumes"
    meta_dir = out_dir / "meta"

    aligner = HeadAligner.from_checkpoints(
        args.pre_align_ckpt,
        args.pose_ckpt,
        device,
        cls_threshold=args.cls_threshold,
        cls_pad=args.cls_pad,
        cls_min_head_slices=args.cls_min_head_slices,
    )

    rows: list[dict] = []
    for path in tqdm(paths, desc="head align (offline)"):
        case_id = path.name.replace(".nii.gz", "")
        row: dict = {"case_id": case_id, "status": "error"}
        try:
            result = aligner.align(path, device=device, case_id=case_id)
            meta = result.to_json_dict()
            aligned_path: str | None = None

            if result.status == "ok" and result.volume_aligned_1mm is not None:
                out_nii = out_vol_dir / f"{case_id}.nii.gz"
                save_volume_nifti(
                    result.volume_aligned_1mm,
                    out_nii,
                    spacing_mm=result.output_spacing_mm,
                )
                meta["aligned_path"] = str(out_nii)
                aligned_path = str(out_nii)

            (meta_dir / f"{case_id}.json").write_text(
                json.dumps(meta, indent=2), encoding="utf-8"
            )
            row = _row_from_meta(meta, aligned_path=aligned_path)
            if result.reason and row["status"] != "ok":
                row["message"] = result.reason
        except Exception as exc:
            row["message"] = str(exc)
        rows.append(row)
    return rows


def main() -> None:
    args = parse_args()
    use_service = bool(args.service_url) and not args.offline
    if not use_service and not args.offline:
        raise SystemExit(
            "Set --service-url (or ALIGN_SERVICE_URL) or pass --offline for in-process align."
        )

    out_vol_dir = args.out_dir / "volumes"
    meta_dir = args.out_dir / "meta"
    out_vol_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    mode = f"API {args.service_url}" if use_service else "offline"
    stage("HeadAligner golden", mode)
    print(f"  input:  {args.volumes_dir}", flush=True)
    print(f"  output: {args.out_dir}", flush=True)

    paths = sorted(args.volumes_dir.glob("*.nii.gz"))
    if args.limit > 0:
        paths = paths[: args.limit]

    if use_service:
        rows = run_via_service(
            paths,
            service_url=args.service_url,
            out_dir=args.out_dir,
            timeout=float(args.timeout),
        )
    else:
        if not args.pre_align_ckpt.is_file():
            raise SystemExit(f"PreAlign checkpoint not found: {args.pre_align_ckpt}")
        if not args.pose_ckpt.is_file():
            raise SystemExit(f"Pose checkpoint not found: {args.pose_ckpt}")
        print(f"  device: {args.device}", flush=True)
        rows = run_offline(paths, args=args, out_dir=args.out_dir)

    csv_path = args.out_dir / "results.csv"
    fieldnames = sorted({k for r in rows for k in r.keys()})
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    n_ok = sum(1 for r in rows if r.get("status") == "ok")
    n_no = sum(1 for r in rows if r.get("status") == "no_head")
    n_fail = len(rows) - n_ok - n_no
    print(
        f"\n  done: {len(rows)} cases | ok={n_ok} no_head={n_no} fail={n_fail}\n"
        f"  csv  -> {csv_path}\n"
        f"  nii  -> {out_vol_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
