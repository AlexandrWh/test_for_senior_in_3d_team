from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class ZSliceNpyDataset(Dataset):
    """Single 56x56 slice .npy files from positive/negative folders."""

    def __init__(self, files: list[Path], labels: list[float]):
        self.files = files
        self.labels = labels

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        arr = np.load(self.files[idx]).astype(np.float32)
        return {
            "slice": torch.from_numpy(arr).unsqueeze(0),
            "has_head": torch.tensor(float(self.labels[idx]), dtype=torch.float32),
        }


def collate_slices(batch: list[dict]) -> dict[str, torch.Tensor]:
    return {
        "slice": torch.stack([b["slice"] for b in batch], dim=0),
        "has_head": torch.stack([b["has_head"] for b in batch], dim=0),
    }


def list_z_slice_files(data_dir: Path) -> tuple[list[Path], list[float]]:
    pos = sorted((data_dir / "positive").glob("*.npy"))
    neg = sorted((data_dir / "negative").glob("*.npy"))
    files = pos + neg
    labels = [1.0] * len(pos) + [0.0] * len(neg)
    return files, labels


def split_z_slice_files(
    data_dir: Path,
    *,
    val_frac: float = 0.1,
    seed: int = 42,
) -> tuple[list[Path], list[float], list[Path], list[float]]:
    files, labels = list_z_slice_files(data_dir)
    rng = np.random.default_rng(seed)
    idx = np.arange(len(files))
    rng.shuffle(idx)
    n_val = max(1, int(round(len(files) * val_frac)))
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]
    train_files = [files[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    val_files = [files[i] for i in val_idx]
    val_labels = [labels[i] for i in val_idx]
    return train_files, train_labels, val_files, val_labels


def _filter_slice_batch(
    paths: list[Path],
    model: torch.nn.Module,
    device: torch.device,
    *,
    is_positive: bool,
    threshold: float,
    batch_size: int,
) -> tuple[int, int]:
    """Return (removed, kept) counts for one class folder."""
    removed = kept = 0
    for start in range(0, len(paths), batch_size):
        batch_paths = paths[start : start + batch_size]
        batch = torch.from_numpy(
            np.stack([np.load(p).astype(np.float32) for p in batch_paths], axis=0)
        ).unsqueeze(1).to(device)
        probs = torch.sigmoid(model(batch)).cpu().numpy()
        for path, prob in zip(batch_paths, probs):
            p = float(prob)
            drop = p < threshold if is_positive else p > threshold
            if drop:
                path.unlink(missing_ok=True)
                removed += 1
            else:
                kept += 1
    return removed, kept


def filter_z_slice_dataset_in_place(
    data_dir: Path,
    model: torch.nn.Module,
    device: torch.device,
    *,
    threshold: float = 0.5,
    batch_size: int = 256,
) -> dict[str, int]:
    """
    In-place filter after a training phase.

    Positive (GT=1): remove slices with prob < threshold.
    Negative (GT=0): remove slices with prob > threshold.
    """
    pos_dir = data_dir / "positive"
    neg_dir = data_dir / "negative"
    pos_files = sorted(pos_dir.glob("*.npy"))
    neg_files = sorted(neg_dir.glob("*.npy"))

    model.eval()
    stats = {
        "pos_in": len(pos_files),
        "neg_in": len(neg_files),
        "pos_removed": 0,
        "pos_kept": 0,
        "neg_removed": 0,
        "neg_kept": 0,
    }
    with torch.no_grad():
        if pos_files:
            removed, kept = _filter_slice_batch(
                pos_files, model, device,
                is_positive=True, threshold=threshold, batch_size=batch_size,
            )
            stats["pos_removed"] = removed
            stats["pos_kept"] = kept

        if neg_files:
            removed, kept = _filter_slice_batch(
                neg_files, model, device,
                is_positive=False, threshold=threshold, batch_size=batch_size,
            )
            stats["neg_removed"] = removed
            stats["neg_kept"] = kept

    return stats
