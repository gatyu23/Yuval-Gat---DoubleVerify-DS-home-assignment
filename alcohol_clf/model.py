"""Classifier definition, persistence, and prediction.

The model is a logistic regression over [char n-gram TF-IDF | lexicon
features]. Logistic regression was chosen over heavier models because:
  * the dataset is small (1,200 images) and the signal is mostly lexical;
  * its coefficients are directly inspectable (explainability requirement);
  * it trains in seconds on CPU and is deterministic.

The saved bundle holds the fitted pipeline plus the decision threshold chosen
on out-of-fold predictions, so `predict.py` needs nothing else.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler

from .config import MODELS_DIR, SEED
from .features import LexiconFeatures, SelectedText

MODEL_PATH = MODELS_DIR / "model.joblib"
KINDS = ("tfidf", "lexicon", "combined")


def build_pipeline(kind: str = "combined", C: float = 1.0) -> Pipeline:
    """kind: 'tfidf' (char n-grams only), 'lexicon' (domain features only),
    or 'combined' (both) - the first two exist for the ablation study."""
    chars = Pipeline([
        ("text", SelectedText()),
        # char_wb n-grams stay inside word boundaries. A garbled word still
        # shares most of its n-grams with the real one ("vedka" and "vodka"
        # share "dka", "dk", "ka"), which the strict lexicon cannot use.
        ("tfidf", TfidfVectorizer(
            analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True
        )),
    ])
    lex = Pipeline([("feats", LexiconFeatures()), ("scale", StandardScaler())])
    parts = {"tfidf": [("chars", chars)], "lexicon": [("lex", lex)],
             "combined": [("chars", chars), ("lex", lex)]}[kind]
    return Pipeline([
        ("features", FeatureUnion(parts)),
        ("clf", LogisticRegression(
            C=C, class_weight="balanced", max_iter=5000, random_state=SEED
        )),
    ])


def save_model(pipeline: Pipeline, threshold: float, meta: dict,
               path: Path = MODEL_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipeline, "threshold": threshold, "meta": meta}, path)


def load_model(path: Path = MODEL_PATH) -> dict:
    return joblib.load(path)


def predict_proba(bundle: dict, ocr_list: list[list[dict]]) -> np.ndarray:
    """P(alcohol) for each image's OCR output."""
    return bundle["pipeline"].predict_proba(ocr_list)[:, 1]


def to_labels(probs: np.ndarray, threshold: float) -> list[str]:
    return ["alcohol" if p >= threshold else "non_alcohol" for p in probs]
