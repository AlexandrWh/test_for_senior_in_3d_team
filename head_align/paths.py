"""Default paths for checkpoints and training artifacts."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS_DIR = ROOT / "weights"
TRAIN_LOG_DIR = ROOT / "data" / "train_logs"

DEFAULT_CLS_CKPT = WEIGHTS_DIR / "head_align_cls_best_v2.pt"
DEFAULT_POSE_CKPT = WEIGHTS_DIR / "head_align_pose_best_v2.pt"
