"""PreAligner: Z-span classifier + axial PCA rigid params (z_min, z_max, dx, dy, Rz)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from models.z_slice_head_cls import ZSliceHeadClsNet
from paths import SPACING_MM, Z_SLICE_SIZE
from utils import apply_brain_ct_window, center_crop_yx_to_square, prepare_isotropic_ct, sitk_image_to_numpy
from utils.angles import pca_e1_tilt_cw_rad
from utils.axial_pca import central_slice_axial_pca, shift_xy_to_center


def _voxels_to_mm_z(z: int, spacing_mm: float) -> float:
    return float(z) * float(spacing_mm)


def _mm_to_voxels_z(z_mm: float, spacing_mm: float) -> int:
    return int(round(float(z_mm) / float(spacing_mm)))


def _voxels_to_mm_xy(dx: float, dy: float, spacing_mm: float) -> tuple[float, float]:
    s = float(spacing_mm)
    return float(dx) * s, float(dy) * s


def _mm_to_voxels_xy(dx_mm: float, dy_mm: float, spacing_mm: float) -> tuple[float, float]:
    s = float(spacing_mm)
    return float(dx_mm) / s, float(dy_mm) / s


@dataclass
class PreAlignParams:
    """Spatial params in mm (isotropic spacing from PreAligner). rz in radians, CW-positive."""

    z_min: float
    z_max: float
    dx: float
    dy: float
    rz: float
    has_head: bool = True

    @classmethod
    def no_head(cls) -> PreAlignParams:
        return cls(z_min=-1.0, z_max=-1.0, dx=0.0, dy=0.0, rz=0.0, has_head=False)

    def z_span_voxels(self, spacing_mm: float) -> tuple[int, int]:
        s = float(spacing_mm)
        return _mm_to_voxels_z(self.z_min, s), _mm_to_voxels_z(self.z_max, s)

    def shift_xy_voxels(self, spacing_mm: float) -> tuple[float, float]:
        s = float(spacing_mm)
        return _mm_to_voxels_xy(self.dx, self.dy, s)


def crop_z_span(vol_zyx: np.ndarray, z_lo: int, z_hi: int) -> np.ndarray:
    z_lo = int(z_lo)
    z_hi = int(z_hi)
    if z_hi < z_lo or z_lo < 0:
        return vol_zyx[:0].astype(np.float32, copy=True)
    z_hi = min(z_hi, int(vol_zyx.shape[0]) - 1)
    return vol_zyx[z_lo : z_hi + 1].astype(np.float32, copy=True)


def center_crop_yx(arr_yx: np.ndarray, size: int = Z_SLICE_SIZE) -> np.ndarray:
    h, w = arr_yx.shape
    out = np.zeros((size, size), dtype=np.float32)
    side_y = min(h, size)
    side_x = min(w, size)
    y0 = max(0, (h - side_y) // 2)
    x0 = max(0, (w - side_x) // 2)
    dy0 = (size - side_y) // 2
    dx0 = (size - side_x) // 2
    out[dy0 : dy0 + side_y, dx0 : dx0 + side_x] = arr_yx[y0 : y0 + side_y, x0 : x0 + side_x]
    return out


class PreAligner(nn.Module):
    """
    Head pre-alignment: per-slice Z classifier + axial PCA on Z-cropped volume.

    Product: z_min, z_max, dx, dy in mm; Rz in radians.
    """

    def __init__(
        self,
        classifier: ZSliceHeadClsNet | None = None,
        *,
        base_channels: int = 32,
        spacing_mm: float = SPACING_MM,
        slice_size: int = Z_SLICE_SIZE,
        cls_threshold: float = 0.5,
        cls_pad: int = 3,
        cls_min_head_slices: int = 10,
    ):
        super().__init__()
        self.classifier = classifier if classifier is not None else ZSliceHeadClsNet(base_channels=base_channels)
        self.spacing_mm = float(spacing_mm)
        self.slice_size = int(slice_size)
        self.cls_threshold = float(cls_threshold)
        self.cls_pad = int(cls_pad)
        self.cls_min_head_slices = int(cls_min_head_slices)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Classifier forward: [B, 1, H, W] -> logit [B]."""
        return self.classifier(x)

    @staticmethod
    def prepare_volume(path: str | Path, *, spacing_mm: float = SPACING_MM) -> np.ndarray:
        ct = prepare_isotropic_ct(path, spacing_mm=spacing_mm)
        hu = sitk_image_to_numpy(ct).astype(np.float32)
        sq = center_crop_yx_to_square(hu)
        return apply_brain_ct_window(sq, output_range=(0.0, 1.0))

    def volume_to_slices(self, vol_win_zyx: np.ndarray) -> np.ndarray:
        z_len = int(vol_win_zyx.shape[0])
        return np.stack(
            [center_crop_yx(vol_win_zyx[z], self.slice_size) for z in range(z_len)],
            axis=0,
        )

    @torch.no_grad()
    def infer_z_span(self, vol_win_zyx: np.ndarray, device: torch.device | None = None) -> dict[str, object]:
        slices = self.volume_to_slices(vol_win_zyx)
        return self.infer_z_span_from_slices(slices, device=device)

    @torch.no_grad()
    def infer_z_span_from_slices(
        self,
        slices_zyx: np.ndarray,
        device: torch.device | None = None,
    ) -> dict[str, object]:
        dev = device if device is not None else next(self.parameters()).device
        return self.classifier.infer_head_z_span(
            slices_zyx,
            threshold=self.cls_threshold,
            pad=self.cls_pad,
            min_head_slices=self.cls_min_head_slices,
            device=dev,
        )

    @torch.no_grad()
    def predict_params(
        self,
        vol_win_zyx: np.ndarray,
        device: torch.device | None = None,
    ) -> PreAlignParams:
        """
        Full pipeline: cls Z-span -> crop -> central axial PCA -> dx, dy, Rz.
        """
        span = self.infer_z_span(vol_win_zyx, device=device)
        if not bool(span.get("has_head", False)):
            return PreAlignParams.no_head()

        z_lo = int(span["z_lo"])
        z_hi = int(span["z_hi"])
        cropped = crop_z_span(vol_win_zyx, z_lo, z_hi)
        if cropped.shape[0] == 0:
            return PreAlignParams.no_head()

        s = self.spacing_mm
        z_min_mm = _voxels_to_mm_z(z_lo, s)
        z_max_mm = _voxels_to_mm_z(z_hi, s)

        ax_det = central_slice_axial_pca(cropped)
        if ax_det is None:
            return PreAlignParams(
                z_min=z_min_mm, z_max=z_max_mm, dx=0.0, dy=0.0, rz=0.0, has_head=True
            )

        z_idx = int(ax_det["z_index"])
        axial = cropped[z_idx]
        dx_vox, dy_vox = shift_xy_to_center(axial.shape, ax_det["center"])
        dx_mm, dy_mm = _voxels_to_mm_xy(dx_vox, dy_vox, s)
        rz = float(pca_e1_tilt_cw_rad(ax_det["e1"]))
        return PreAlignParams(
            z_min=z_min_mm, z_max=z_max_mm, dx=dx_mm, dy=dy_mm, rz=rz, has_head=True
        )

    def predict_from_path(self, path: str | Path, device: torch.device | None = None) -> PreAlignParams:
        vol = self.prepare_volume(path, spacing_mm=self.spacing_mm)
        return self.predict_params(vol, device=device)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: str | Path,
        device: torch.device,
        **kwargs,
    ) -> PreAligner:
        checkpoint = Path(checkpoint)
        ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
        model = cls(base_channels=int(ckpt.get("base_channels", 32)), **kwargs)
        state = ckpt.get("model_state", ckpt.get("classifier_state"))
        if state is None:
            raise KeyError(f"No model weights in {checkpoint}")
        model.classifier.load_state_dict(state)
        model.to(device)
        model.eval()
        return model

    def save_checkpoint(self, path: str | Path, *, epoch: int = 0) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state": self.classifier.state_dict(),
                "epoch": epoch,
                "base_channels": 32,
                "slice_size": self.slice_size,
            },
            path,
        )
