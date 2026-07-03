from __future__ import annotations

import torch
import torch.nn as nn


class FullScanBackbone(nn.Module):
    """Shared 3D CNN for variable-Z full scans [B, 1, Z, 128, 128]."""

    def __init__(self, base_channels: int = 16, pool_shape: tuple[int, int, int] = (8, 4, 4)):
        super().__init__()
        c = base_channels
        self.pool_shape = pool_shape
        self.features = nn.Sequential(
            nn.Conv3d(1, c, kernel_size=3, stride=(1, 2, 2), padding=1),
            nn.BatchNorm3d(c),
            nn.ReLU(inplace=True),
            nn.Conv3d(c, c * 2, kernel_size=3, stride=(1, 2, 2), padding=1),
            nn.BatchNorm3d(c * 2),
            nn.ReLU(inplace=True),
            nn.Conv3d(c * 2, c * 4, kernel_size=3, stride=(2, 2, 2), padding=1),
            nn.BatchNorm3d(c * 4),
            nn.ReLU(inplace=True),
            nn.Conv3d(c * 4, c * 8, kernel_size=3, stride=(2, 2, 2), padding=1),
            nn.BatchNorm3d(c * 8),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool3d(pool_shape)
        self.feat_dim = c * 8 * pool_shape[0] * pool_shape[1] * pool_shape[2]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        return self.pool(feat).flatten(1)


class FullScanPoseNet(nn.Module):
    """Pose correction: rotvec (rad) + trans (mm). Train on positive samples only."""

    def __init__(self, base_channels: int = 16, pool_shape: tuple[int, int, int] = (8, 4, 4)):
        super().__init__()
        self.backbone = FullScanBackbone(base_channels, pool_shape)
        self.head_rot = nn.Linear(self.backbone.feat_dim, 3)
        self.head_trans = nn.Linear(self.backbone.feat_dim, 3)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.backbone(x)
        return self.head_rot(feat), self.head_trans(feat)


class FullScanClsNet(nn.Module):
    """Head / no-head classifier."""

    def __init__(self, base_channels: int = 16, pool_shape: tuple[int, int, int] = (8, 4, 4)):
        super().__init__()
        self.backbone = FullScanBackbone(base_channels, pool_shape)
        self.head_cls = nn.Linear(self.backbone.feat_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head_cls(self.backbone(x)).squeeze(-1)
