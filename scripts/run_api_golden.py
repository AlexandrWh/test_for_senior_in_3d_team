"""Run golden volumes through /align API; save NIfTI + meta + summary CSV."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import httpx
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batch golden through head-align API")
    p.add_argument("--volumes-dir", type=Path, default=ROOT / "data" / "volumes")
    p.add_argument("--out-dir", type=Path, default=ROOT / "data" / "api_golden")
    p.add_argument("--api-url", type=str, default="http://127.0.0.1:8000/align")
    p.add_argument("--timeout", type=float, default=300.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "aligned").mkdir(exist_ok=True)
    (args.out_dir / "meta").mkdir(exist_ok=True)

    case_ids = sorted(p.name.replace(".nii.gz", "") for p in args.volumes_dir.glob("*.nii.gz"))
    rows: list[dict] = []

    with httpx.Client(timeout=args.timeout) as client:
        for case_id in tqdm(case_ids, desc="api golden"):
            path = args.volumes_dir / f"{case_id}.nii.gz"
            row: dict = {"case_id": case_id, "status": "error", "http_code": ""}
            try:
                with path.open("rb") as f:
                    resp = client.post(
                        args.api_url,
                        files={"file": (path.name, f, "application/gzip")},
                    )
                row["http_code"] = resp.status_code
                if resp.status_code == 200:
                    aligned_path = args.out_dir / "aligned" / f"{case_id}_aligned.nii.gz"
                    aligned_path.write_bytes(resp.content)
                    meta = json.loads(resp.headers.get("X-Align-Meta", "{}"))
                    (args.out_dir / "meta" / f"{case_id}.json").write_text(
                        json.dumps(meta, indent=2, ensure_ascii=False),
                        encoding="utf-8",
                    )
                    row.update(
                        {
                            "status": "ok",
                            "has_head": meta.get("has_head"),
                            "detector_ok": meta.get("detector_ok"),
                            "cls_prob": meta.get("cls_prob"),
                            "geodesic_deg": meta.get("geodesic_deg"),
                            "detector_rotz_deg": meta.get("detector_rotz_deg"),
                            "message": meta.get("message"),
                        }
                    )
                else:
                    try:
                        detail = resp.json()
                    except Exception:
                        detail = resp.text
                    row["message"] = str(detail)[:500]
                    if isinstance(detail, dict) and "detail" in detail:
                        d = detail["detail"]
                        if isinstance(d, dict):
                            row["has_head"] = d.get("has_head")
                            row["detector_ok"] = d.get("detector_ok")
                            row["cls_prob"] = d.get("cls_prob")
                            row["message"] = d.get("message", row["message"])
            except Exception as exc:
                row["message"] = str(exc)
            rows.append(row)

    csv_path = args.out_dir / "results.csv"
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    n_ok = sum(1 for r in rows if r.get("status") == "ok")
    print(f"Done {len(rows)} cases ({n_ok} aligned) -> {args.out_dir}")


if __name__ == "__main__":
    main()
