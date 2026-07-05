from __future__ import annotations

import re
from pathlib import Path

CASE_ID_RE = re.compile(r"(CQ500CT\d+)")


def case_id_from_name(name: str) -> str | None:
    m = CASE_ID_RE.search(name)
    return m.group(1) if m else None


def load_case_id_list(path: Path) -> list[str]:
    """Read case IDs from a text file (one per line, # comments allowed)."""
    ids: list[str] = []
    seen: set[str] = set()
    if not path.is_file():
        return ids
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        case_id = case_id_from_name(line) or line
        if case_id in seen:
            continue
        seen.add(case_id)
        ids.append(case_id)
    return ids


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
