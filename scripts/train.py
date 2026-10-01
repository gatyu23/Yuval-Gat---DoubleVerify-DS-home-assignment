"""Train, select and evaluate the classifier from the cached OCR.

Protocol (all randomness seeded with config.SEED):
  1. Stratified 80/20 split. The 20% test set is scored exactly once, at the end.
  2. On the 80% train set, 5-fold CV compares:
       - the lexicon rule (alcohol iff a strict alcohol term is in the best
         variant; no training),
       - logistic regression on char n-grams only, lexicon features only, and
         both combined, each over a small grid of C.
     The configuration with the best out-of-fold PR-AUC wins, and the decision
     threshold is the one maximizing out-of-fold F1.
  3. The winner is refit on the train set and scored on the test set.
  4. The shipped model (models/model.joblib) is refit on all 1,200 images with
     the same configuration and threshold (--no-refit-all to skip).

Outputs in reports/: split.json, cv_results.csv, metrics.json,
test_predictions.csv (inputs for the report notebook).

Usage:
    python scripts/train.py [--cache cache/ocr_raw.jsonl]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from alcohol_clf.config import CACHE_DIR, REPORTS_DIR, SEED  # noqa: E402
from alcohol_clf.data import load_cache  # noqa: E402
from alcohol_clf.lexicon import ALCOHOL_TERMS, match_terms  # noqa: E402
from alcohol_clf.metrics import best_f1_threshold, summarize  # noqa: E402
from alcohol_clf.model import KINDS, MODEL_PATH, build_pipeline, save_model  # noqa: E402
from alcohol_clf.select import select_text  # noqa: E402

C_GRID = (0.3, 1.0, 3.0, 10.0, 30.0, 100.0)


def lexicon_rule(ocr_list) -> np.ndarray:
    return np.array([
        float(bool(match_terms(select_text(ocr), ALCOHOL_TERMS, negate=True)))
        for ocr in ocr_list
    ])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, default=CACHE_DIR / "ocr_raw.jsonl")
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--out", type=Path, default=MODEL_PATH)
    ap.add_argument("--reports-dir", type=Path, default=REPORTS_DIR)
    ap.add_argument("--no-refit-all", action="store_true")
    args = ap.parse_args()

    cache = load_cache(args.cache)
    files = sorted(cache)
    X = [cache[f]["ocr"] for f in files]
    y = np.array([int(cache[f]["label"] == "alcohol") for f in files])
    print(f"{len(files)} images: {y.sum()} alcohol, {len(y) - y.sum()} non_alcohol")

    idx_tr, idx_te = train_test_split(
        np.arange(len(files)), test_size=args.test_size, stratify=y, random_state=SEED
    )
    X_tr, y_tr = [X[i] for i in idx_tr], y[idx_tr]
    X_te, y_te = [X[i] for i in idx_te], y[idx_te]
    args.reports_dir.mkdir(parents=True, exist_ok=True)
    (args.reports_dir / "split.json").write_text(json.dumps({
        "train": [files[i] for i in idx_tr], "test": [files[i] for i in idx_te]
    }, indent=1))

    # --- model selection by cross-validation on the train set ---------------
    cv = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=SEED)
    rows = [{"model": "lexicon_rule", "C": None,
             **summarize(y_tr, lexicon_rule(X_tr))}]
    for kind in KINDS:
        for C in C_GRID:
            probs = cross_val_predict(
                build_pipeline(kind, C), X_tr, y_tr, cv=cv, method="predict_proba"
            )[:, 1]
            thr = best_f1_threshold(y_tr, probs)
            rows.append({"model": kind, "C": C, **summarize(y_tr, probs, thr)})
            print(f"  {kind:9s} C={C:<5} PR-AUC={rows[-1]['pr_auc']:.4f} "
                  f"F1={rows[-1]['f1']:.4f} @thr={thr:.3f}")
    cv_df = pd.DataFrame(rows)
    cv_df.to_csv(args.reports_dir / "cv_results.csv", index=False)

    best = cv_df[cv_df.model != "lexicon_rule"].sort_values(
        ["pr_auc", "f1"], ascending=False).iloc[0]
    kind, C = best.model, float(best.C)
    threshold = float(best.threshold)
    print(f"\nSelected: {kind}, C={C}, threshold={threshold:.3f}")

    # --- one-shot evaluation on the held-out test set ------------------------
    model = build_pipeline(kind, C).fit(X_tr, y_tr)
    probs_te = model.predict_proba(X_te)[:, 1]
    test_metrics = {
        "selected": summarize(y_te, probs_te, threshold),
        "selected_at_0.5": summarize(y_te, probs_te, 0.5),
        "lexicon_rule": summarize(y_te, lexicon_rule(X_te)),
    }
    for name, m in test_metrics.items():
        print(f"  test {name:16s} P={m['precision']:.3f} R={m['recall']:.3f} "
              f"F1={m['f1']:.3f} acc={m['accuracy']:.3f}")

    pd.DataFrame({
        "file_name": [files[i] for i in idx_te],
        "label": ["alcohol" if t else "non_alcohol" for t in y_te],
        "prob_alcohol": probs_te,
        "prediction": ["alcohol" if p >= threshold else "non_alcohol" for p in probs_te],
        "selected_text": [select_text(o) for o in X_te],
        "alcohol_hits": [
            ", ".join(sorted(match_terms(select_text(o), ALCOHOL_TERMS, negate=True)))
            for o in X_te
        ],
    }).to_csv(args.reports_dir / "test_predictions.csv", index=False)

    meta = {"kind": kind, "C": C, "threshold": threshold, "seed": SEED,
            "n_train": len(y_tr), "n_test": len(y_te)}
    (args.reports_dir / "metrics.json").write_text(json.dumps(
        {"config": meta, "test": test_metrics}, indent=2))

    # --- ship: refit on everything with the chosen config --------------------
    if args.no_refit_all:
        final, meta["trained_on"] = model, "train split"
    else:
        final, meta["trained_on"] = build_pipeline(kind, C).fit(X, y), "all images"
    save_model(final, threshold, meta, args.out)
    print(f"Saved model ({meta['trained_on']}) to {args.out}")


if __name__ == "__main__":
    main()
