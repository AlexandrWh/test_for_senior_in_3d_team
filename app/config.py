"""Service configuration from environment."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _path(name: str, default: Path) -> Path:
    raw = os.environ.get(name, "").strip()
    return Path(raw) if raw else default


def service_device() -> str:
    raw = os.environ.get("ALIGN_DEVICE", "auto").strip().lower()
    if raw == "auto":
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    return raw


PRE_ALIGN_CKPT = _path("PRE_ALIGN_CKPT", ROOT / "weights" / "pre_aligner_best.pt")
POSE_CKPT = _path("POSE_CKPT", ROOT / "weights" / "pose_regressor_best.pt")
CLS_THRESHOLD = float(os.environ.get("CLS_THRESHOLD", "0.5"))
CLS_PAD = int(os.environ.get("CLS_PAD", "3"))
CLS_MIN_HEAD_SLICES = int(os.environ.get("CLS_MIN_HEAD_SLICES", "10"))
