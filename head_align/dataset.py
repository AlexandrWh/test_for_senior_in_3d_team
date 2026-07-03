from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def load_rotvec(d: np.lib.npyio.NpzFile) -> np.ndarray:
    if "rotvec_corr_rad" in d:
        return d["rotvec_corr_rad"].astype(np.float32)
    from scipy.spatial.transform import Rotation

    return Rotation.from_euler("xyz", d["euler_corr_rad"]).as_rotvec().astype(np.float32)


class NpzDataset(Dataset):
    def __init__(self, split_dir: Path, *, positives_only: bool = False):
        files = sorted(split_dir.glob("*.npz"))
        if positives_only:
            files = [f for f in files if float(np.load(f)["has_head"]) > 0.5]
        self.files = files

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        d = np.load(self.files[idx])
        return {
            "volume": torch.from_numpy(d["volume"].astype(np.float32)),
            "has_head": torch.tensor(float(d["has_head"]), dtype=torch.float32),
            "rotvec": torch.from_numpy(load_rotvec(d)),
            "trans": torch.from_numpy(d["trans_corr_mm"].astype(np.float32)),
            "case_id": str(d["case_id"]),
        }


def collate_fixed(batch: list[dict]) -> dict[str, torch.Tensor]:
    vols, heads, rotvecs, trans = [], [], [], []
    for b in batch:
        vols.append(b["volume"].unsqueeze(0))
        heads.append(b["has_head"])
        rotvecs.append(b["rotvec"])
        trans.append(b["trans"])
    return {
        "volume": torch.stack(vols, dim=0),
        "has_head": torch.stack(heads, dim=0),
        "rotvec": torch.stack(rotvecs, dim=0),
        "trans": torch.stack(trans, dim=0),
    }
