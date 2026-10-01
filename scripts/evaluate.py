"""Score a results.csv against ground truth.

Ground truth comes either from a labelled folder layout
(<labels-dir>/alcohol/*, <labels-dir>/non_alcohol/*) or from a CSV with
columns file_name,label.

Usage:
    python scripts/evaluate.py --results results.csv --labels-dir images
    python scripts/evaluate.py --results results.csv --labels-csv truth.csv
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from alcohol_clf.data import labelled_images  # noqa: E402
from alcohol_clf.metrics import summarize  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=Path("results.csv"))
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--labels-dir", type=Path)
    src.add_argument("--labels-csv", type=Path)
    args = ap.parse_args()

    res = pd.read_csv(args.results)
    if args.labels_dir:
        truth = pd.DataFrame(
            [(p.name, label) for label, p in labelled_images(args.labels_dir)],
            columns=["file_name", "label"],
        )
    else:
        truth = pd.read_csv(args.labels_csv)

    df = truth.merge(res, on="file_name", how="left")
    missing = df.prediction.isna().sum()
    if missing:
        print(f"WARNING: {missing} labelled images have no prediction (counted as wrong).")
        df["prediction"] = df.prediction.fillna("missing")

    y = (df.label == "alcohol").astype(int)
    pred = (df.prediction == "alcohol").astype(int)
    m = summarize(y, pred)
    m.pop("threshold")
    print(json.dumps(m, indent=2))
    wrong = df[(y != pred) | (df.prediction == "missing")]
    print(f"\n{len(wrong)} misclassified:")
    print(wrong[["file_name", "label", "prediction"]].to_string(index=False))


if __name__ == "__main__":
    main()
