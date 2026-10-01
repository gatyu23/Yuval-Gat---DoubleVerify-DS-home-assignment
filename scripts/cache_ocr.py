"""Run TTA OCR over the labelled dataset once and cache raw results to JSONL.

OCR is by far the slowest stage on CPU (~0.3s per pass), so everything downstream
(text cleaning, variant selection, classifier training) works off this cache.

The cache is append-only and the last record per file wins, which makes the
script resumable and lets it extend records from an older OCR version (e.g.
adding the deskew stage) without redoing the flip variants.

Usage:
    python scripts/cache_ocr.py [--data-dir images] [--out cache/ocr_raw.jsonl]
"""
import argparse
import json
import sys
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from alcohol_clf.config import CACHE_DIR, DATA_DIR  # noqa: E402
from alcohol_clf.data import labelled_images, load_cache  # noqa: E402
from alcohol_clf.ocr import OCREngine  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=DATA_DIR)
    ap.add_argument("--out", type=Path, default=CACHE_DIR / "ocr_raw.jsonl")
    args = ap.parse_args()

    items = labelled_images(args.data_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    cache = load_cache(args.out)
    todo = [x for x in items if not cache.get(x[1].name, {}).get("deskew_checked")]

    engine = OCREngine()
    with open(args.out, "a") as f:
        for label, path in tqdm(todo):
            old = cache.get(path.name)
            if old is None:
                ocr = engine.read_path(path)
            else:
                ocr = old["ocr"] + engine.read_path(path, flips=False)
            rec = {
                "file_name": path.name,
                "label": label,
                "deskew_checked": True,
                "ocr": ocr,
            }
            f.write(json.dumps(rec) + "\n")
            f.flush()


if __name__ == "__main__":
    main()
