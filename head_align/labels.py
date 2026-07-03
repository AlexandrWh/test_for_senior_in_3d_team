from __future__ import annotations

import re
from pathlib import Path

CASE_ID_RE = re.compile(r"(CQ500CT\d+)")


def case_id_from_name(name: str) -> str | None:
    m = CASE_ID_RE.search(name)
    return m.group(1) if m else None


def collect_case_ids(folder: Path) -> list[str]:
    ids: list[str] = []
    seen: set[str] = set()
    if not folder.is_dir():
        return ids
    for path in sorted(folder.iterdir()):
        if not path.is_file():
            continue
        case_id = case_id_from_name(path.name)
        if case_id is None or case_id in seen:
            continue
        seen.add(case_id)
        ids.append(case_id)
    return ids


def case_ids_from_npz_split(split_dir: Path) -> list[str]:
    """Unique case_id values from a dataset split folder (e.g. head_align_cls/val)."""
    import numpy as np

    ids: set[str] = set()
    if not split_dir.is_dir():
        return []
    for path in sorted(split_dir.glob("*.npz")):
        with np.load(path) as d:
            if "case_id" in d:
                ids.add(str(d["case_id"]))
    return sorted(ids)
