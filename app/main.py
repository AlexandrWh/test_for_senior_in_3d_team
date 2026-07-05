"""FastAPI head alignment service."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response

from app.align_service import AlignService
from app.config import (
    CLS_MIN_HEAD_SLICES,
    CLS_PAD,
    CLS_THRESHOLD,
    POSE_CKPT,
    PRE_ALIGN_CKPT,
    service_device,
)

_service: AlignService | None = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global _service
    device = service_device()
    if not PRE_ALIGN_CKPT.is_file():
        raise RuntimeError(f"PreAlign checkpoint not found: {PRE_ALIGN_CKPT}")
    if not POSE_CKPT.is_file():
        raise RuntimeError(f"Pose checkpoint not found: {POSE_CKPT}")
    _service = AlignService.from_checkpoints(
        pre_align_ckpt=PRE_ALIGN_CKPT,
        pose_ckpt=POSE_CKPT,
        device=device,
        cls_threshold=CLS_THRESHOLD,
        cls_pad=CLS_PAD,
        cls_min_head_slices=CLS_MIN_HEAD_SLICES,
    )
    yield
    _service = None


app = FastAPI(
    title="CQ500 Head Aligner",
    version="1.0.0",
    description="Upload head CT NIfTI → aligned head @ 1 mm + meta (ZIP).",
    lifespan=lifespan,
)


def _get_service() -> AlignService:
    if _service is None:
        raise HTTPException(status_code=503, detail="service not ready")
    return _service


@app.get("/health")
def health() -> dict:
    svc = _get_service()
    return {
        "status": "ok",
        "device": svc.device,
        "pre_align_ckpt": str(PRE_ALIGN_CKPT),
        "pose_ckpt": str(POSE_CKPT),
    }


@app.post("/align")
async def align(
    file: UploadFile = File(..., description="Input CT NIfTI (.nii.gz)"),
    case_id: str = Form(""),
) -> Response:
    name = file.filename or "scan.nii.gz"
    if not (name.endswith(".nii.gz") or name.endswith(".nii")):
        raise HTTPException(status_code=400, detail="expected .nii.gz or .nii")

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="empty upload")

    svc = _get_service()
    try:
        result, zip_bytes = svc.align_upload(data, filename=name, case_id=case_id.strip())
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    cid = result.case_id or Path(name).stem.replace(".nii", "")
    headers = {
        "Content-Disposition": f'attachment; filename="{cid}_align.zip"',
        "X-Align-Status": result.status,
        "X-Case-Id": cid,
    }
    return Response(content=zip_bytes, media_type="application/zip", headers=headers)
