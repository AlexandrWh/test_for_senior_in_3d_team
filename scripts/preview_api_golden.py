"""Before/after central-slice previews for api_golden aligned volumes."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import SimpleITK as sitk
import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from test_align_api import save_before_after_preview
from head_align.preprocess import BOTTOM_CROP_Z
from head_align.volume import extract_infer_slab_ct, prepare_isotropic_ct
from utils import sitk_image_to_numpy


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Golden API before/after previews")
    p.add_argument("--golden-dir", type=Path, default=ROOT / "data" / "api_golden")
    p.add_argument("--volumes-dir", type=Path, default=ROOT / "data" / "volumes")
    p.add_argument("--spacing-mm", type=float, default=4.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    csv_path = args.golden_dir / "results.csv"
    out_dir = args.golden_dir / "previews"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    ok_rows = [r for r in rows if r.get("status") == "ok"]

    for row in tqdm(ok_rows, desc="golden previews"):
        case_id = row["case_id"]
        in_path = args.volumes_dir / f"{case_id}.nii.gz"
        aligned_path = args.golden_dir / "aligned" / f"{case_id}_aligned.nii.gz"
        meta_path = args.golden_dir / "meta" / f"{case_id}.json"
        preview_path = out_dir / f"{case_id}_before_after.png"

        if not in_path.is_file() or not aligned_path.is_file():
            continue

        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}

        ct_in = prepare_isotropic_ct(in_path, spacing_mm=float(args.spacing_mm))
        before_hu = sitk_image_to_numpy(
            extract_infer_slab_ct(ct_in, bottom_n=BOTTOM_CROP_Z)
        ).astype(np.float32)
        after_hu = sitk_image_to_numpy(sitk.ReadImage(str(aligned_path))).astype(np.float32)

        title = (
            f"{case_id} · cls={meta.get('cls_prob', 0):.2f} · "
            f"geo={meta.get('geodesic_deg', 0):.1f}° · "
            f"detZ={meta.get('detector_rotz_deg', 0):.1f}°"
        )
        save_before_after_preview(before_hu, after_hu, preview_path, title=title)

    n = len(list(out_dir.glob("*_before_after.png")))
    print(f"Done {n} previews -> {out_dir}")


if __name__ == "__main__":
    main()
