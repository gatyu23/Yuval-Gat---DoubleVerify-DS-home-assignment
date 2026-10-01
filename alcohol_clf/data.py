"""Dataset and OCR-cache I/O shared by the scripts."""
from __future__ import annotations

import json
from pathlib import Path

from .config import IMAGE_EXTS, LABELS


def list_images(directory: Path) -> list[Path]:
    """All images under `directory` (recursive), sorted by path."""
    return sorted(
        p for p in Path(directory).rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def labelled_images(data_dir: Path) -> list[tuple[str, Path]]:
    """(label, path) pairs from the <data_dir>/<label>/ folder layout."""
    return [
        (label, p)
        for label in LABELS
        for p in list_images(Path(data_dir) / label)
    ]


def load_cache(path: Path) -> dict[str, dict]:
    """OCR cache keyed by file name. The cache is append-only; last record wins."""
    records = {}
    if Path(path).exists():
        with open(path) as f:
            for line in f:
                rec = json.loads(line)
                records[rec["file_name"]] = rec
    return records
