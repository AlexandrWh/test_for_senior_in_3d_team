"""Head align HTTP service."""

from __future__ import annotations

import io
import json
import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

import SimpleITK as sitk
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response, StreamingResponse

from head_align.export import align_head_scan, load_service_models, pipeline_meta
from head_align.paths import DEFAULT_CLS_CKPT, DEFAULT_POSE_CKPT
from utils import read_nifti

CLS_CKPT = Path(os.environ.get("HEAD_ALIGN_CLS_CKPT", str(DEFAULT_CLS_CKPT)))
POSE_CKPT = Path(os.environ.get("HEAD_ALIGN_POSE_CKPT", str(DEFAULT_POSE_CKPT)))
DEVICE = os.environ.get("HEAD_ALIGN_DEVICE", "auto")
SPACING_MM = float(os.environ.get("HEAD_ALIGN_SPACING_MM", "4.0"))
CLS_THRESHOLD = float(os.environ.get("HEAD_ALIGN_CLS_THRESHOLD", "0.5"))

_state: dict = {}


@asynccontextmanager
async def lifespan(_app: FastAPI):
    cls_model, pose_model, device = load_service_models(CLS_CKPT, POSE_CKPT, DEVICE)
    _state["cls_model"] = cls_model
    _state["pose_model"] = pose_model
    _state["device"] = device
    yield
    _state.clear()


app = FastAPI(title="Head Align API", version="1.1.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "device": str(_state.get("device", "not_loaded")),
        "cls_checkpoint": str(CLS_CKPT),
        "pose_checkpoint": str(POSE_CKPT),
        "pipeline": pipeline_meta(spacing_mm=SPACING_MM),
    }


@app.post("/align")
async def align(file: UploadFile = File(...)) -> Response:
    """
    Upload raw NIfTI (.nii / .nii.gz). Returns aligned head slab as NIfTI
    and JSON metadata in response header X-Align-Meta.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="empty filename")

    suffix = Path(file.filename).suffix.lower()
    if file.filename.lower().endswith(".nii.gz"):
        suffix = ".nii.gz"
    elif suffix not in {".nii", ".gz"}:
        raise HTTPException(status_code=400, detail="expected .nii or .nii.gz")

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="empty file")

    with tempfile.TemporaryDirectory() as tmp:
        in_path = Path(tmp) / f"input{suffix}"
        out_path = Path(tmp) / "aligned_head.nii.gz"
        in_path.write_bytes(data)

        try:
            ct = read_nifti(in_path)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"cannot read volume: {exc}") from exc

        result = align_head_scan(
            ct,
            _state["cls_model"],
            _state["pose_model"],
            _state["device"],
            spacing_mm=SPACING_MM,
            cls_threshold=CLS_THRESHOLD,
        )

        meta = {
            "has_head": result.has_head,
            "detector_ok": result.detector_ok,
            "cls_prob": result.cls_prob,
            "geodesic_deg": result.geodesic_deg,
            "detector_rotz_deg": result.detector_rotz_deg,
            "rotvec_corr_rad": result.rotvec_corr_rad,
            "affine_4x4": result.affine_4x4,
            "message": result.message,
            "spacing_mm": SPACING_MM,
            "frame": "identity_index",
            "pipeline": pipeline_meta(spacing_mm=SPACING_MM),
        }

        if not result.has_head or not result.detector_ok or result.ct_aligned is None:
            raise HTTPException(status_code=422, detail=meta)

        sitk.WriteImage(result.ct_aligned, str(out_path), useCompression=True)
        nifti_bytes = out_path.read_bytes()

    headers = {"X-Align-Meta": json.dumps(meta, ensure_ascii=False)}
    return Response(content=nifti_bytes, media_type="application/gzip", headers=headers)


@app.post("/align-with-meta")
async def align_with_meta(file: UploadFile = File(...)) -> StreamingResponse:
    """Multipart: aligned_head.nii.gz + meta.json (easier for some clients)."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="empty filename")

    suffix = ".nii.gz" if file.filename.lower().endswith(".nii.gz") else Path(file.filename).suffix
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="empty file")

    with tempfile.TemporaryDirectory() as tmp:
        in_path = Path(tmp) / f"input{suffix}"
        in_path.write_bytes(data)
        ct = read_nifti(in_path)

        result = align_head_scan(
            ct,
            _state["cls_model"],
            _state["pose_model"],
            _state["device"],
            spacing_mm=SPACING_MM,
            cls_threshold=CLS_THRESHOLD,
        )

        meta = {
            "has_head": result.has_head,
            "detector_ok": result.detector_ok,
            "cls_prob": result.cls_prob,
            "geodesic_deg": result.geodesic_deg,
            "detector_rotz_deg": result.detector_rotz_deg,
            "rotvec_corr_rad": result.rotvec_corr_rad,
            "affine_4x4": result.affine_4x4,
            "message": result.message,
            "spacing_mm": SPACING_MM,
            "frame": "identity_index",
            "pipeline": pipeline_meta(spacing_mm=SPACING_MM),
        }

        if not result.has_head or not result.detector_ok or result.ct_aligned is None:
            raise HTTPException(status_code=422, detail=meta)

        out_nii = Path(tmp) / "aligned.nii.gz"
        sitk.WriteImage(result.ct_aligned, str(out_nii), useCompression=True)

        boundary = "----headalignboundary"
        nii_part = out_nii.read_bytes()
        meta_part = json.dumps(meta, ensure_ascii=False, indent=2).encode("utf-8")
        body = io.BytesIO()
        body.write(f"--{boundary}\r\n".encode())
        body.write(b"Content-Disposition: form-data; name=\"aligned\"; filename=\"aligned_head.nii.gz\"\r\n")
        body.write(b"Content-Type: application/gzip\r\n\r\n")
        body.write(nii_part)
        body.write(f"\r\n--{boundary}\r\n".encode())
        body.write(b"Content-Disposition: form-data; name=\"meta\"; filename=\"meta.json\"\r\n")
        body.write(b"Content-Type: application/json\r\n\r\n")
        body.write(meta_part)
        body.write(f"\r\n--{boundary}--\r\n".encode())
        body.seek(0)

    return StreamingResponse(
        body,
        media_type=f"multipart/mixed; boundary={boundary}",
    )
