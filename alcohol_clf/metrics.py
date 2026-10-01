"""Metric helpers. The positive class is 'alcohol'.

Final metrics: precision, recall, F1 (at the chosen threshold), plus the
threshold-free PR-AUC (average precision) and ROC-AUC.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score, confusion_matrix, precision_recall_curve,
    precision_recall_fscore_support, roc_auc_score,
)


def summarize(y_true, probs, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true)
    probs = np.asarray(probs, dtype=float)
    y_pred = (probs >= threshold).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out = {
        "threshold": float(threshold),
        "accuracy": float((tp + tn) / len(y_true)),
        "precision": float(p), "recall": float(r), "f1": float(f1),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }
    if len(np.unique(probs)) > 1:  # AUCs are undefined for a constant scorer
        out["pr_auc"] = float(average_precision_score(y_true, probs))
        out["roc_auc"] = float(roc_auc_score(y_true, probs))
    return out


def best_f1_threshold(y_true, probs) -> float:
    """Threshold maximizing F1, computed on out-of-fold predictions."""
    prec, rec, thr = precision_recall_curve(y_true, probs)
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-12, None)
    return float(thr[np.argmax(f1[:-1])])
