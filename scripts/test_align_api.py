"""Test head-align API: upload scan, save aligned NIfTI, draw before/after previews."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import httpx
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from head_align.inference import center_slice_np
from head_align.volume import extract_infer_slab_ct, prepare_isotropic_ct
from head_align.preprocess import BOTTOM_CROP_Z
from utils import apply_brain_ct_window, read_nifti, sitk_image_to_numpy

PLANES = ("axial", "coronal", "sagittal")
PLANE_LABEL = {"axial": "axial", "coronal": "coronal", "sagittal": "sagittal"}


def _slice_windowed(vol_zyx: np.ndarray, plane: str) -> np.ndarray:
    return apply_brain_ct_window(center_slice_np(vol_zyx, plane))


def save_before_after_preview(
    before_zyx: np.ndarray,
    after_zyx: np.ndarray,
    out_path: Path,
    *,
    title: str,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), dpi=120)
    fig.suptitle(title, fontsize=12)

    for col, plane in enumerate(PLANES):
        b = _slice_windowed(before_zyx, plane)
        a = _slice_windowed(after_zyx, plane)
        axes[0, col].imshow(b, cmap="gray", vmin=0.0, vmax=1.0, aspect="equal")
        axes[0, col].set_title(f"before · {PLANE_LABEL[plane]}")
        axes[0, col].axis("off")
        axes[1, col].imshow(a, cmap="gray", vmin=0.0, vmax=1.0, aspect="equal")
        axes[1, col].set_title(f"after · {PLANE_LABEL[plane]}")
        axes[1, col].axis("off")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Test /align API and save previews")
    p.add_argument("--input", type=Path, required=True, help="Input NIfTI path")
    p.add_argument("--api-url", type=str, default="http://127.0.0.1:8000/align")
    p.add_argument("--out-dir", type=Path, default=ROOT / "data" / "api_test")
    p.add_argument("--spacing-mm", type=float, default=4.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    input_path = args.input.resolve()
    aligned_path = args.out_dir / f"{input_path.stem.replace('.nii', '')}_aligned.nii.gz"
    preview_path = args.out_dir / f"{input_path.stem.replace('.nii', '')}_before_after.png"
    meta_path = args.out_dir / f"{input_path.stem.replace('.nii', '')}_meta.json"

    print(f"POST {args.api_url} <- {input_path}")
    with input_path.open("rb") as f:
        resp = httpx.post(
            args.api_url,
            files={"file": (input_path.name, f, "application/gzip")},
            timeout=600.0,
        )

    if resp.status_code != 200:
        try:
            detail = resp.json()
        except Exception:
            detail = resp.text
        raise SystemExit(f"API error {resp.status_code}: {detail}")

    aligned_path.write_bytes(resp.content)
    meta_raw = resp.headers.get("X-Align-Meta", "{}")
    meta = json.loads(meta_raw)
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"saved aligned -> {aligned_path}")
    print(f"meta -> {meta_path}")
    print(json.dumps(meta, indent=2, ensure_ascii=False))

    ct_in = prepare_isotropic_ct(input_path, spacing_mm=float(args.spacing_mm))
    before_slab = extract_infer_slab_ct(ct_in, bottom_n=BOTTOM_CROP_Z)
    before_hu = sitk_image_to_numpy(before_slab).astype(np.float32)

    ct_out = sitk.ReadImage(str(aligned_path))
    after_hu = sitk_image_to_numpy(ct_out).astype(np.float32)

    case_id = input_path.name.replace(".nii.gz", "").replace(".nii", "")
    title = (
        f"{case_id} · cls={meta.get('cls_prob', 0):.2f} · "
        f"geo={meta.get('geodesic_deg', 0):.1f}° · detZ={meta.get('detector_rotz_deg', 0):.1f}°"
    )
    save_before_after_preview(before_hu, after_hu, preview_path, title=title)
    print(f"preview -> {preview_path}")


if __name__ == "__main__":
    main()
