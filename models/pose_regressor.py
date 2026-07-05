"""3D CNN pose regressor: light stack, stride XY downsample, Z-mix, mean over Z -> MLP."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

_P111 = (1, 1, 1)
_S1 = (1, 1, 1)
_S2 = (1, 2, 2)
_SPATIAL_OUT = 18
_FEAT_PER_SLICE = _SPATIAL_OUT * _SPATIAL_OUT  # 324


def _conv3(
    cin: int,
    cout: int,
    *,
    kernel: tuple[int, int, int] = (3, 3, 3),
    stride: tuple[int, int, int] = _S1,
    padding: tuple[int, int, int] = (1, 1, 1),
) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv3d(
            cin, cout, kernel_size=kernel, stride=stride, padding=padding, bias=False
        ),
        nn.BatchNorm3d(cout),
        nn.ReLU(inplace=True),
    )


class PoseRegressor3D(nn.Module):
    """
    [B,1,Z,72,72]
      2× Conv3d 3³ s=1              -> Z×72×72
      2× Conv3d 3³ s=(1,2,2)        -> Z×18×18
      2× Conv3d (3,1,1)             -> Z×18×18  (depth mix)
      Conv3d -> 1 ch
      [B,Z,324] -> masked mean Z -> MLP(32) -> 3
    """

    SPATIAL_OUT = _SPATIAL_OUT
    FEAT_DIM = _FEAT_PER_SLICE

    def __init__(self, base_channels: int = 12, mlp_hidden: int = 32):
        super().__init__()
        c = int(base_channels)
        c2 = c * 2
        h = int(mlp_hidden)
        self.features = nn.Sequential(
            _conv3(1, c, stride=_S1, padding=_P111),
            _conv3(c, c, stride=_S1, padding=_P111),
            _conv3(c, c2, stride=_S2, padding=_P111),
            _conv3(c2, c2, stride=_S2, padding=_P111),
            _conv3(c2, c2, kernel=(3, 1, 1), stride=_S1, padding=(1, 0, 0)),
            _conv3(c2, c2, kernel=(3, 1, 1), stride=_S1, padding=(1, 0, 0)),
            _conv3(c2, 1, stride=_S1, padding=_P111),
        )
        self.head = nn.Sequential(
            nn.Linear(self.FEAT_DIM, h),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(h, 3),
        )

    def _masked_z_mean(self, feat_bzf: torch.Tensor, z_len: torch.Tensor) -> torch.Tensor:
        b, z, _f = feat_bzf.shape
        idx = torch.arange(z, device=feat_bzf.device).view(1, z, 1)
        mask = (idx < z_len.view(b, 1, 1)).to(feat_bzf.dtype)
        denom = mask.sum(dim=1).clamp(min=1.0)
        return (feat_bzf * mask).sum(dim=1) / denom

    def forward(self, x: torch.Tensor, *, z_len: torch.Tensor | None = None) -> torch.Tensor:
        feat = self.features(x)
        feat_bzf = feat.squeeze(1).flatten(2)
        if z_len is None:
            pooled = feat_bzf.mean(dim=1)
        else:
            pooled = self._masked_z_mean(feat_bzf, z_len)
        return self.head(pooled)

    def save_checkpoint(self, path: str | Path, *, epoch: int = 0, **extra) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": self.state_dict(),
                "epoch": int(epoch),
                "arch": "PoseRegressor3D",
                "base_channels": int(self.features[0][0].out_channels),
                "mlp_hidden": int(self.head[0].out_features),
                "spatial_out": int(self.SPATIAL_OUT),
                **extra,
            },
            path,
        )

    @classmethod
    def from_checkpoint(cls, path: str | Path, device: torch.device | str = "cpu") -> PoseRegressor3D:
        payload = torch.load(path, map_location=device, weights_only=False)
        model = cls(
            base_channels=int(payload.get("base_channels", 12)),
            mlp_hidden=int(payload.get("mlp_hidden", 32)),
        )
        model.load_state_dict(payload["model"])
        model.to(device)
        return model
