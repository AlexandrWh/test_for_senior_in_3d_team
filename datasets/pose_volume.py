"""Pose volume dataset: 3D NIfTI + residual angle labels (rad)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from torch.utils.data import Dataset

from paths import POSE_INPUT_YX
from utils import center_crop_pad_yx
from utils.rigid import load_volume_nifti


@dataclass(frozen=True)
class PoseSample:
    sample_id: str
    case_id: str
    meta_path: Path
    volume_path: Path
    rz_rad: float
    ry_rad: float
    rx_rad: float


def _parse_meta(path: Path, *, volumes_dir: Path) -> PoseSample | None:
    meta = json.loads(path.read_text(encoding="utf-8"))
    labels = meta.get("labels_rad", {})
    if not labels:
        return None

    sample_id = str(meta.get("sample_id", path.stem))
    case_id = str(meta.get("case_id", sample_id.rsplit("_a", 1)[0]))
    vol_path = Path(meta.get("volume_path", volumes_dir / f"{sample_id}.nii.gz"))
    if not vol_path.is_file():
        vol_path = volumes_dir / f"{sample_id}.nii.gz"
    if not vol_path.is_file():
        return None

    return PoseSample(
        sample_id=sample_id,
        case_id=case_id,
        meta_path=path,
        volume_path=vol_path,
        rz_rad=float(labels["rz"]),
        ry_rad=float(labels["ry"]),
        rx_rad=float(labels["rx"]),
    )


def list_pose_samples(
    meta_dir: Path,
    *,
    volumes_dir: Path,
    limit: int = 0,
) -> list[PoseSample]:
    paths = sorted(meta_dir.glob("*.json"))
    if limit > 0:
        paths = paths[:limit]
    out: list[PoseSample] = []
    for path in paths:
        row = _parse_meta(path, volumes_dir=volumes_dir)
        if row is not None:
            out.append(row)
    return out


def split_pose_by_case(
    samples: list[PoseSample],
    *,
    val_frac: float = 0.2,
    seed: int = 42,
) -> tuple[list[PoseSample], list[PoseSample], list[str], list[str]]:
    case_ids = sorted({s.case_id for s in samples})
    if not case_ids:
        return [], [], [], []

    rng = np.random.default_rng(seed)
    shuffled = case_ids.copy()
    rng.shuffle(shuffled)

    n_val = max(1, int(round(len(shuffled) * val_frac)))
    if len(shuffled) <= 1:
        n_val = 0
    val_cases = set(shuffled[:n_val])
    train_cases = set(shuffled[n_val:])

    train = [s for s in samples if s.case_id in train_cases]
    val = [s for s in samples if s.case_id in val_cases]
    return train, val, sorted(train_cases), sorted(val_cases)


class PoseTextureAugment:
    """Intensity/texture-only aug for windowed CT [0, 1]. No geometry — angles stay valid."""

    def __init__(
        self,
        *,
        p_brightness: float = 0.9,
        p_noise: float = 0.5,
        p_gamma: float = 0.4,
        p_blur: float = 0.25,
        p_sharpness: float = 0.25,
        contrast_range: tuple[float, float] = (0.85, 1.15),
        brightness_range: tuple[float, float] = (-0.08, 0.08),
        noise_std_range: tuple[float, float] = (0.01, 0.04),
        gamma_range: tuple[float, float] = (0.85, 1.15),
        blur_sigma_range: tuple[float, float] = (0.2, 0.7),
        sharp_amount_range: tuple[float, float] = (0.2, 0.6),
        sharp_blur_sigma_range: tuple[float, float] = (0.4, 1.0),
    ):
        self.p_brightness = float(p_brightness)
        self.p_noise = float(p_noise)
        self.p_gamma = float(p_gamma)
        self.p_blur = float(p_blur)
        self.p_sharpness = float(p_sharpness)
        self.contrast_range = contrast_range
        self.brightness_range = brightness_range
        self.noise_std_range = noise_std_range
        self.gamma_range = gamma_range
        self.blur_sigma_range = blur_sigma_range
        self.sharp_amount_range = sharp_amount_range
        self.sharp_blur_sigma_range = sharp_blur_sigma_range

    def __call__(self, vol_zyx: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        vol = vol_zyx.astype(np.float32, copy=True)

        if rng.random() < self.p_brightness:
            alpha = float(rng.uniform(*self.contrast_range))
            beta = float(rng.uniform(*self.brightness_range))
            vol = vol * alpha + beta

        if rng.random() < self.p_noise:
            std = float(rng.uniform(*self.noise_std_range))
            vol = vol + rng.normal(0.0, std, size=vol.shape).astype(np.float32)

        if rng.random() < self.p_gamma:
            gamma = float(rng.uniform(*self.gamma_range))
            vol = np.power(np.clip(vol, 0.0, 1.0), gamma)

        if rng.random() < self.p_blur:
            sigma_yx = float(rng.uniform(*self.blur_sigma_range))
            sigma_z = sigma_yx * 0.5
            vol = gaussian_filter(vol, sigma=(sigma_z, sigma_yx, sigma_yx), mode="nearest")

        if rng.random() < self.p_sharpness:
            sigma_yx = float(rng.uniform(*self.sharp_blur_sigma_range))
            sigma_z = sigma_yx * 0.5
            blurred = gaussian_filter(vol, sigma=(sigma_z, sigma_yx, sigma_yx), mode="nearest")
            amount = float(rng.uniform(*self.sharp_amount_range))
            vol = vol + amount * (vol - blurred)

        return np.clip(vol, 0.0, 1.0).astype(np.float32)


class PoseVolumeDataset(Dataset):
    def __init__(
        self,
        samples: list[PoseSample],
        *,
        input_yx: int = POSE_INPUT_YX,
        augment: bool = False,
        texture_aug: PoseTextureAugment | None = None,
    ):
        self.samples = samples
        self.input_yx = int(input_yx)
        self.augment = bool(augment)
        self.texture_aug = texture_aug if texture_aug is not None else PoseTextureAugment()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str | int]:
        row = self.samples[idx]
        vol = load_volume_nifti(row.volume_path).astype(np.float32)
        vol = center_crop_pad_yx(vol, self.input_yx, pad_value=0.0)
        if self.augment:
            vol = self.texture_aug(vol, np.random.default_rng())
        angles = np.array([row.rz_rad, row.ry_rad, row.rx_rad], dtype=np.float32)
        return {
            "volume": torch.from_numpy(vol).unsqueeze(0),
            "angles": torch.from_numpy(angles),
            "z_len": int(vol.shape[0]),
            "sample_id": row.sample_id,
            "case_id": row.case_id,
        }


def collate_pose_volumes(batch: list[dict]) -> dict[str, torch.Tensor | list[str]]:
    z_max = max(int(b["z_len"]) for b in batch)
    yx = int(batch[0]["volume"].shape[-1])

    vols = []
    for b in batch:
        v = b["volume"]
        z = int(v.shape[1])
        if z < z_max:
            pad = torch.zeros(1, z_max - z, yx, yx, dtype=v.dtype)
            v = torch.cat([v, pad], dim=1)
        vols.append(v)

    return {
        "volume": torch.stack(vols, dim=0),
        "angles": torch.stack([b["angles"] for b in batch], dim=0),
        "z_len": torch.tensor([int(b["z_len"]) for b in batch], dtype=torch.int64),
        "sample_id": [str(b["sample_id"]) for b in batch],
        "case_id": [str(b["case_id"]) for b in batch],
    }
