"""Parse manual guide JSONs -> rotation angles + Z-span for classifier."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from utils.angles import segment_tilt_cw_rad

Point = list[float]


@dataclass
class GuideLabel:
    case_id: str
    rot_z_rad: float
    rot_y_rad: float
    rot_x_rad: float
    z_lo_1mm: int
    z_hi_1mm: int
    z_lo: int
    z_hi: int
    coronal_slice_index: int
    axial_slice_index: int
    shape_zyx: list[int]
    source_json: str

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["rot_z_deg"] = float(math.degrees(self.rot_z_rad))
        d["rot_y_deg"] = float(math.degrees(self.rot_y_rad))
        d["rot_x_deg"] = float(math.degrees(self.rot_x_rad))
        d["z_span_len"] = int(self.z_hi - self.z_lo + 1)
        d["z_span_len_1mm"] = int(self.z_hi_1mm - self.z_lo_1mm + 1)
        return d


def normalize_segment(p0: Point, p1: Point) -> tuple[Point, Point]:
    """Sort endpoints top-to-bottom by image Y (origin=upper)."""
    if float(p0[1]) <= float(p1[1]):
        return list(p0), list(p1)
    return list(p1), list(p0)


def coronal_y_to_z_indices(y_top: float, y_bottom: float, shape_yx: tuple[int, int]) -> tuple[int, int]:
    """Map coronal PNG row (Y) to Z indices on 1 mm MPR grid."""
    h = int(shape_yx[0])
    if h <= 0:
        return 0, 0
    z_lo = int(round(float(y_top)))
    z_hi = int(round(float(y_bottom)))
    z_lo = max(0, min(z_lo, h - 1))
    z_hi = max(0, min(z_hi, h - 1))
    if z_lo > z_hi:
        z_lo, z_hi = z_hi, z_lo
    return z_lo, z_hi


def z_1mm_to_cls_indices(
    z_lo_1mm: int,
    z_hi_1mm: int,
    *,
    mpr_spacing_mm: float,
    cls_spacing_mm: float,
) -> tuple[int, int]:
    scale = float(mpr_spacing_mm) / float(cls_spacing_mm)
    z_lo = int(round(z_lo_1mm * scale))
    z_hi = int(round(z_hi_1mm * scale))
    if z_lo > z_hi:
        z_lo, z_hi = z_hi, z_lo
    return z_lo, z_hi


def has_axial_coronal_segments(ann: dict) -> bool:
    planes = ann.get("planes", {})
    for plane in ("axial", "coronal"):
        pl = planes.get(plane, {})
        if pl.get("p0") is None or pl.get("p1") is None:
            return False
    return True


def skip_reason(ann: dict) -> str:
    missing: list[str] = []
    for plane in ("axial", "coronal"):
        pl = ann.get("planes", {}).get(plane, {})
        if pl.get("p0") is None or pl.get("p1") is None:
            missing.append(plane)
    return ",".join(missing) if missing else ""


def parse_guide_annotation(
    ann: dict,
    *,
    source_json: str,
    manifest_case: dict | None,
    mpr_spacing_mm: float,
    cls_spacing_mm: float,
) -> GuideLabel | None:
    if not has_axial_coronal_segments(ann):
        return None

    case_id = str(ann.get("case_id", Path(source_json).stem))
    planes = ann["planes"]
    axial = planes["axial"]
    coronal = planes["coronal"]

    rot_z = segment_tilt_cw_rad(axial["p0"], axial["p1"])
    rot_y = segment_tilt_cw_rad(coronal["p0"], coronal["p1"])

    cor_shape = tuple(coronal.get("shape_yx") or (0, 0))
    if manifest_case and "planes" in manifest_case:
        cor_shape = tuple(manifest_case["planes"].get("coronal", {}).get("shape_yx") or cor_shape)

    top, bottom = normalize_segment(coronal["p0"], coronal["p1"])
    z_lo_1mm, z_hi_1mm = coronal_y_to_z_indices(top[1], bottom[1], cor_shape)
    z_lo, z_hi = z_1mm_to_cls_indices(
        z_lo_1mm,
        z_hi_1mm,
        mpr_spacing_mm=mpr_spacing_mm,
        cls_spacing_mm=cls_spacing_mm,
    )

    shape_zyx = list(manifest_case.get("shape_zyx", [])) if manifest_case else []
    axial_idx = int(axial.get("slice_index", 0))
    cor_idx = int(coronal.get("slice_index", 0))
    if manifest_case and "planes" in manifest_case:
        axial_idx = int(manifest_case["planes"].get("axial", {}).get("slice_index", axial_idx))
        cor_idx = int(manifest_case["planes"].get("coronal", {}).get("slice_index", cor_idx))
        if not shape_zyx:
            shape_zyx = list(manifest_case.get("shape_zyx", []))

    return GuideLabel(
        case_id=case_id,
        rot_z_rad=rot_z,
        rot_y_rad=rot_y,
        rot_x_rad=0.0,
        z_lo_1mm=z_lo_1mm,
        z_hi_1mm=z_hi_1mm,
        z_lo=z_lo,
        z_hi=z_hi,
        coronal_slice_index=cor_idx,
        axial_slice_index=axial_idx,
        shape_zyx=shape_zyx,
        source_json=source_json,
    )


def load_mpr_manifest(path: Path) -> dict:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_guide_labels(
    guides_dir: Path,
    *,
    manifest_path: Path,
    volumes_dir: Path,
    mpr_spacing_mm: float = 1.0,
    cls_spacing_mm: float = 4.0,
) -> tuple[list[GuideLabel], list[dict[str, str]]]:
    manifest = load_mpr_manifest(manifest_path)
    manifest_cases = manifest.get("cases", {})
    mpr_spacing_mm = float(manifest.get("spacing_mm", mpr_spacing_mm))

    labels: list[GuideLabel] = []
    skipped: list[dict[str, str]] = []

    for path in sorted(guides_dir.glob("*.json")):
        ann = json.loads(path.read_text(encoding="utf-8"))
        case_id = str(ann.get("case_id", path.stem))

        if not has_axial_coronal_segments(ann):
            skipped.append({"case_id": case_id, "reason": skip_reason(ann) or "missing_planes"})
            continue

        vol_path = volumes_dir / f"{case_id}.nii.gz"
        if not vol_path.is_file():
            skipped.append({"case_id": case_id, "reason": "missing_volume"})
            continue

        label = parse_guide_annotation(
            ann,
            source_json=path.name,
            manifest_case=manifest_cases.get(case_id),
            mpr_spacing_mm=mpr_spacing_mm,
            cls_spacing_mm=cls_spacing_mm,
        )
        if label is None:
            skipped.append({"case_id": case_id, "reason": "parse_failed"})
            continue
        labels.append(label)

    return labels, skipped
