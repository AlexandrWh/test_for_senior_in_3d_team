from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class ZSliceHeadClsNet(nn.Module):
    """Per-axial-slice head classifier: input [B, 1, 56, 56] -> logit [B]."""

    def __init__(self, base_channels: int = 32):
        super().__init__()
        c = base_channels
        self.features = nn.Sequential(
            nn.Conv2d(1, c, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
            nn.Conv2d(c, c * 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(c * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(c * 2, c * 4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(c * 4),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(c * 4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x).flatten(1)
        return self.head(feat).squeeze(-1)

    @torch.no_grad()
    def predict_probs(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(x))

    @staticmethod
    def _empty_head_span(
        probs: np.ndarray,
        mask: np.ndarray,
        head_indices: np.ndarray,
        n_head_slices: int,
    ) -> dict[str, object]:
        return {
            "probs": probs,
            "mask": mask,
            "head_indices": head_indices,
            "q05": -1,
            "q95": -1,
            "z_lo": -1,
            "z_hi": -1,
            "n_head_slices": int(n_head_slices),
            "has_head": False,
        }

    @staticmethod
    def head_z_span_from_probs(
        probs: np.ndarray,
        *,
        z_len: int,
        threshold: float = 0.5,
        q_low: float = 5.0,
        q_high: float = 95.0,
        pad: int = 3,
        min_head_slices: int = 10,
    ) -> dict[str, object]:
        probs = np.asarray(probs, dtype=np.float32).reshape(-1)
        z_len = int(z_len)
        if probs.shape[0] != z_len:
            raise ValueError(f"probs length {probs.shape[0]} != z_len {z_len}")

        mask = probs >= float(threshold)
        head_indices = np.flatnonzero(mask).astype(np.int32)
        n_head = int(head_indices.size)

        if n_head < int(min_head_slices):
            return ZSliceHeadClsNet._empty_head_span(probs, mask, head_indices, n_head)

        q05 = int(np.floor(np.percentile(head_indices, q_low)))
        q95 = int(np.ceil(np.percentile(head_indices, q_high)))
        z_lo = max(0, q05 - int(pad))
        z_hi = min(z_len - 1, q95 + int(pad))

        return {
            "probs": probs,
            "mask": mask,
            "head_indices": head_indices,
            "q05": q05,
            "q95": q95,
            "z_lo": z_lo,
            "z_hi": z_hi,
            "n_head_slices": n_head,
            "has_head": True,
        }

    @torch.no_grad()
    def infer_head_z_span(
        self,
        slices_zyx: np.ndarray,
        *,
        threshold: float = 0.5,
        q_low: float = 5.0,
        q_high: float = 95.0,
        pad: int = 3,
        min_head_slices: int = 10,
        batch_size: int = 256,
        device: torch.device | None = None,
    ) -> dict[str, object]:
        slices = np.asarray(slices_zyx, dtype=np.float32)
        if slices.ndim != 3:
            raise ValueError(f"Expected [Z,H,W], got {slices.shape}")
        z_len = int(slices.shape[0])
        dev = device if device is not None else next(self.parameters()).device

        probs_parts: list[np.ndarray] = []
        for start in range(0, z_len, batch_size):
            batch = torch.from_numpy(slices[start : start + batch_size]).unsqueeze(1).to(dev)
            probs_parts.append(self.predict_probs(batch).cpu().numpy())
        probs = np.concatenate(probs_parts, axis=0).astype(np.float32)

        out = ZSliceHeadClsNet.head_z_span_from_probs(
            probs,
            z_len=z_len,
            threshold=threshold,
            q_low=q_low,
            q_high=q_high,
            pad=pad,
            min_head_slices=min_head_slices,
        )
        out["threshold"] = float(threshold)
        out["pad"] = int(pad)
        out["min_head_slices"] = int(min_head_slices)
        return out
