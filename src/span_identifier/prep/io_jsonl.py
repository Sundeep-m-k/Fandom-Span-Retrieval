from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Dict, Any, List


def write_jsonl(path: str | Path, rows: Iterable[Dict[str, Any]]) -> None:
    """
    Write an iterable of dicts to a JSONL file (one JSON object per line).

    Args:
        path: Output file path.
        rows: Iterable of JSON-serializable dictionaries.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")


def write_json(path: str | Path, obj: Dict[str, Any] | List[Any]) -> None:
    """
    Write a JSON object or list to a file (pretty-printed).

    Args:
        path: Output file path.
        obj: JSON-serializable object (dict or list).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")
def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    path = Path(path)
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows
