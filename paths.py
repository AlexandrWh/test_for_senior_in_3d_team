"""Project paths and constants."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent
WEIGHTS_DIR = ROOT / "weights"
TRAIN_LOG_DIR = ROOT / "data" / "train_logs"

TRAIN_BASE = ROOT / "data" / "cq500_train"
TEST_BASE = ROOT / "data" / "cq500_test"

TRAIN_VOLUMES = TRAIN_BASE / "volumes"
TRAIN_NO_HEADS_LIST = TRAIN_BASE / "no_heads.txt"
TRAIN_MPR_DIR = TRAIN_BASE / "cq500_train_mpr_1mm"
TRAIN_MPR_MANIFEST = TRAIN_MPR_DIR / "manifest.json"
TRAIN_GUIDES_DIR = TRAIN_BASE / "cq500_train_guides"
TRAIN_GUIDES_ANALYSIS_DIR = TRAIN_BASE / "cq500_train_guides_analysis"
TRAIN_GUIDE_LABELS_JSON = TRAIN_GUIDES_ANALYSIS_DIR / "guide_labels.json"
TRAIN_Z_SLICE_CLS_DIR = TRAIN_BASE / "z_head_slice_cls"
TRAIN_POSE_DATASET_DIR = TRAIN_BASE / "pose_dataset"
TRAIN_POSE_DATASET_VOLUMES = TRAIN_POSE_DATASET_DIR / "volumes"
TRAIN_POSE_DATASET_META = TRAIN_POSE_DATASET_DIR / "meta"

POSE_AUG_RANGE_PI = 0.06
POSE_AUG_PER_CASE = 30
POSE_INPUT_YX = 72

TEST_VOLUMES = TEST_BASE / "volumes"
TEST_MASKS = TEST_BASE / "masks"
TEST_MASK_RESIDUAL_DIR = TEST_BASE / "mask_residual"
TEST_ALIGN_DIR = TEST_BASE / "align"
TEST_ALIGN_VOLUMES = TEST_ALIGN_DIR / "volumes"
TEST_ALIGN_META = TEST_ALIGN_DIR / "meta"
TEST_ALIGN_RESULTS_CSV = TEST_ALIGN_DIR / "results.csv"
TEST_ALIGN_PREVIEWS = TEST_ALIGN_DIR / "previews"

SPACING_MM = 4.0  # infer grid (classifier + PCA + pose)
APPLY_SPACING_MM = 1.0  # final aligned head output
Z_SLICE_SIZE = 56
DEFAULT_PRE_ALIGNER_CKPT = WEIGHTS_DIR / "pre_aligner_best.pt"
DEFAULT_POSE_REGRESSOR_CKPT = WEIGHTS_DIR / "pose_regressor_best.pt"
POSE_ANGLE_LOSS_WEIGHTS = (0.4, 0.4, 0.2)
