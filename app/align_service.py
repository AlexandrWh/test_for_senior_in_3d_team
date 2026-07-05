"""HeadAligner wrapper for HTTP API."""

from __future__ import annotations

import io
import json
import tempfile
import zipfile
from pathlib import Path

import torch

from models.head_aligner import AlignResult, HeadAligner
from utils.rigid import volume_to_nifti_bytes


class AlignService:
    def __init__(self, aligner: HeadAligner):
        self.aligner = aligner

    @classmethod
    def from_checkpoints(
        cls,
        *,
        pre_align_ckpt: Path,
        pose_ckpt: Path,
        device: str,
        cls_threshold: float = 0.5,
        cls_pad: int = 3,
        cls_min_head_slices: int = 10,
    ) -> AlignService:
        dev = torch.device(device)
        aligner = HeadAligner.from_checkpoints(
            pre_align_ckpt,
            pose_ckpt,
            dev,
            cls_threshold=cls_threshold,
            cls_pad=cls_pad,
            cls_min_head_slices=cls_min_head_slices,
        )
        return cls(aligner)

    @property
    def device(self) -> str:
        return str(next(self.aligner.pre_aligner.parameters()).device)

    def align_file(self, src_path: Path, *, case_id: str = "") -> tuple[AlignResult, bytes]:
        """Run aligner; return result and ZIP bytes (aligned.nii.gz + meta.json)."""
        case_id = case_id or src_path.name.replace(".nii.gz", "")
        dev = next(self.aligner.pre_aligner.parameters()).device
        result = self.aligner.align(src_path, device=dev, case_id=case_id)
        return result, self._result_to_zip(result)

    def align_upload(self, data: bytes, *, filename: str, case_id: str = "") -> tuple[AlignResult, bytes]:
        suffix = ".nii.gz" if filename.endswith(".nii.gz") else Path(filename).suffix or ".nii.gz"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(data)
            tmp_path = Path(tmp.name)
        try:
            cid = case_id or Path(filename).name.replace(".nii.gz", "")
            return self.align_file(tmp_path, case_id=cid)
        finally:
            tmp_path.unlink(missing_ok=True)

    def _result_to_zip(self, result: AlignResult) -> bytes:
        meta = result.to_json_dict()
        meta["ok"] = result.status == "ok"
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("meta.json", json.dumps(meta, indent=2))
            if result.status == "ok" and result.volume_aligned_1mm is not None:
                nii = volume_to_nifti_bytes(
                    result.volume_aligned_1mm,
                    spacing_mm=result.output_spacing_mm,
                )
                zf.writestr("aligned.nii.gz", nii)
        return buf.getvalue()
