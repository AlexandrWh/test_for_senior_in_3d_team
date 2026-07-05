"""
Batch-render CQ500 train volumes for guide-line annotation: isotropic 1 mm, brain window, 3 mid-slice projections.

Output: per case either 3 PNGs (axial/coronal/sagittal) or one combined 1x3 PNG.

Usage:
    python -u scripts/render_guide_annotation_mpr.py
    python -u scripts/render_guide_annotation_mpr.py --combined
    python -u scripts/render_guide_annotation_mpr.py --limit 10
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils import apply_brain_ct_window, center_slice_np, prepare_isotropic_ct, sitk_image_to_numpy

from paths import TRAIN_MPR_DIR, TRAIN_VOLUMES

DEFAULT_VOLUMES = TRAIN_VOLUMES
DEFAULT_OUT = TRAIN_MPR_DIR
PLANES = ("axial", "coronal", "sagittal")


def load_volume_for_preview(path: Path, *, spacing_mm: float, windowed_input: bool) -> np.ndarray:
    """Load [z,y,x] float32 in [0,1] for MPR rendering."""
    if windowed_input:
        import SimpleITK as sitk

        vol = sitk.GetArrayFromImage(sitk.ReadImage(str(path))).astype(np.float32)
        return np.clip(vol, 0.0, 1.0)
    ct = prepare_isotropic_ct(path, spacing_mm=float(spacing_mm))
    hu = sitk_image_to_numpy(ct).astype(np.float32)
    return apply_brain_ct_window(hu, output_range=(0.0, 1.0))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Render 1mm MPR previews for CQ500 train")
    p.add_argument("--volumes-dir", type=Path, default=DEFAULT_VOLUMES)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--spacing-mm", type=float, default=1.0)
    p.add_argument(
        "--windowed-input",
        action="store_true",
        help="NIfTI already resampled + brain-windowed [0,1]; skip preprocess",
    )
    p.add_argument("--dpi", type=int, default=100)
    p.add_argument(
        "--combined",
        action="store_true",
        help="single 1x3 PNG per case ({case_id}_mpr.png) with case_id in title",
    )
    p.add_argument("--limit", type=int, default=0, help="0 = all cases")
    p.add_argument("--skip-existing", action="store_true", default=True)
    p.add_argument("--force", action="store_true", help="re-render even if PNG exists")
    return p.parse_args()


def mid_slice_indices(shape_zyx: tuple[int, int, int]) -> dict[str, int]:
    z, y, x = (int(shape_zyx[0]), int(shape_zyx[1]), int(shape_zyx[2]))
    return {"axial": z // 2, "coronal": y // 2, "sagittal": x // 2}


def render_case_pngs(
    vol_windowed: np.ndarray,
    out_dir: Path,
    case_id: str,
    *,
    dpi: int,
) -> dict:
    """Save axial/coronal/sagittal PNGs; return plane metadata."""
    meta_planes: dict[str, dict] = {}
    for plane in PLANES:
        slc = center_slice_np(vol_windowed, plane)
        fig, ax = plt.subplots(figsize=(4, 4), dpi=dpi)
        ax.imshow(slc, cmap="gray", vmin=0.0, vmax=1.0, origin="upper", aspect="equal")
        ax.axis("off")
        out_path = out_dir / f"{case_id}_{plane}.png"
        fig.savefig(out_path, bbox_inches="tight", pad_inches=0.0)
        plt.close(fig)
        meta_planes[plane] = {
            "png": out_path.name,
            "shape_yx": [int(slc.shape[0]), int(slc.shape[1])],
            "slice_index": int(mid_slice_indices(vol_windowed.shape)[plane]),
        }
    return meta_planes


def render_case_combined_png(
    vol_windowed: np.ndarray,
    out_dir: Path,
    case_id: str,
    *,
    dpi: int,
) -> dict:
    """Save one 1x3 PNG (axial | coronal | sagittal) with case_id title."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=dpi)
    fig.suptitle(case_id, fontsize=12, fontweight="bold")
    meta_planes: dict[str, dict] = {}
    mids = mid_slice_indices(vol_windowed.shape)
    for ax, plane in zip(axes, PLANES, strict=True):
        slc = center_slice_np(vol_windowed, plane)
        ax.imshow(slc, cmap="gray", vmin=0.0, vmax=1.0, origin="upper", aspect="equal")
        ax.set_title(plane)
        ax.axis("off")
        meta_planes[plane] = {
            "shape_yx": [int(slc.shape[0]), int(slc.shape[1])],
            "slice_index": int(mids[plane]),
        }
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out_path = out_dir / f"{case_id}_mpr.png"
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return {"combined_png": out_path.name, "planes": meta_planes}


def main() -> None:
    args = parse_args()
    paths = sorted(args.volumes_dir.glob("*.nii.gz"))
    if args.limit > 0:
        paths = paths[: int(args.limit)]
    if not paths:
        raise SystemExit(f"No NIfTI in {args.volumes_dir}")

    layout = (
        "1 PNG per case: {case_id}_mpr.png (1x3 axial|coronal|sagittal)"
        if args.combined
        else "3 PNG per case: {case_id}_{axial|coronal|sagittal}.png"
    )
    manifest: dict = {
        "spacing_mm": float(args.spacing_mm),
        "planes_order": list(PLANES),
        "combined": bool(args.combined),
        "layout": layout,
        "cases": {},
    }

    for path in tqdm(paths, desc="render MPR"):
        case_id = path.name.replace(".nii.gz", "")
        marker = (
            args.out_dir / f"{case_id}_mpr.png"
            if args.combined
            else args.out_dir / f"{case_id}_axial.png"
        )
        if args.skip_existing and not args.force and marker.is_file():
            continue

        vol = load_volume_for_preview(
            path,
            spacing_mm=float(args.spacing_mm),
            windowed_input=bool(args.windowed_input),
        )
        args.out_dir.mkdir(parents=True, exist_ok=True)
        if args.combined:
            planes = render_case_combined_png(vol, args.out_dir, case_id, dpi=int(args.dpi))
        else:
            planes = render_case_pngs(vol, args.out_dir, case_id, dpi=int(args.dpi))
        manifest["cases"][case_id] = {
            "source_nifti": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
            "shape_zyx": [int(x) for x in vol.shape],
            "planes": planes,
        }

    manifest_path = args.out_dir / "manifest.json"
    if manifest_path.is_file():
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        old_cases = old.get("cases", {})
        old_cases.update(manifest["cases"])
        manifest["cases"] = old_cases
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Done {len(paths)} cases -> {args.out_dir}")
    print(f"manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
