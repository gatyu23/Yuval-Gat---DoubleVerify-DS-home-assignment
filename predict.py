"""Stand-alone inference: every image in a directory -> results.csv.

    python predict.py --input-dir path/to/images [--output results.csv]

Writes one row per image, with columns file_name,prediction, where prediction
is "alcohol" or "non_alcohol". Images are found recursively; only pixels are
used (file and folder names are ignored).

Runs on CPU. OCR takes ~2-3 s per image, because each image is read in several
orientations. The first run downloads EasyOCR's weights (~95 MB).
"""
import argparse
import csv
import logging
import sys
from pathlib import Path

from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from alcohol_clf.data import list_images, load_cache
from alcohol_clf.model import MODEL_PATH
from alcohol_clf.predictor import AlcoholPredictor

log = logging.getLogger("predict")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--input-dir", type=Path, required=True,
                    help="folder of images (searched recursively)")
    ap.add_argument("--output", type=Path, default=Path("results.csv"),
                    help="where to write the file_name,prediction CSV")
    ap.add_argument("--scores", type=Path, default=None,
                    help="optional extra CSV that also holds P(alcohol) per image")
    ap.add_argument("--model", type=Path, default=MODEL_PATH,
                    help="trained model file")
    ap.add_argument("--ocr-cache", type=Path, default=None,
                    help="OCR cache from scripts/cache_ocr.py; skips OCR for files "
                         "it contains (development only)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if not args.input_dir.is_dir():
        sys.exit(f"Input directory not found: {args.input_dir}")
    paths = list_images(args.input_dir)
    if not paths:
        sys.exit(f"No images found in {args.input_dir}")
    names = [p.name for p in paths]
    if len(set(names)) != len(names):
        log.warning("Duplicate file names in subfolders; rows will repeat names.")
    log.info("Found %d images in %s", len(paths), args.input_dir)

    cache = load_cache(args.ocr_cache) if args.ocr_cache else {}
    predictor = AlcoholPredictor(args.model, ocr_cache=cache)

    ocr_list = []
    with logging_redirect_tqdm():  # keep warnings off the progress-bar line
        for path in tqdm(paths, desc="OCR", unit="img"):
            try:
                ocr_list.append(predictor.read_text(path))
            except Exception as e:  # unreadable file -> None -> non_alcohol
                log.warning("Could not read %s (%s); predicting non_alcohol", path, e)
                ocr_list.append(None)
    results = predictor.predict_ocr(ocr_list)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file_name", "prediction"])
        w.writerows((name, label) for name, (label, _) in zip(names, results))
    if args.scores:
        args.scores.parent.mkdir(parents=True, exist_ok=True)
        with open(args.scores, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["file_name", "prob_alcohol", "prediction"])
            w.writerows((n, f"{p:.4f}", lab) for n, (lab, p) in zip(names, results))

    n_alc = sum(label == "alcohol" for label, _ in results)
    log.info("Wrote %d predictions (%d alcohol, %d non_alcohol) to %s",
             len(results), n_alc, len(results) - n_alc, args.output.resolve())


if __name__ == "__main__":
    main()
