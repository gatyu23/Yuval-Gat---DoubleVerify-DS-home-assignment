"""Feature extraction: OCR variants -> model inputs.

Every transformer here takes the raw OCR output of an image (the list of
variants produced by `OCREngine.read`), so training on the cache and inference
on new images go through exactly the same code.

Two feature families, combined in `model.py`:
  * SelectedText -> character n-gram TF-IDF. Learns sub-word evidence the
    lexicon cannot express: fragments of garbled product words ("vedk",
    "skey"), and which template words lean towards each class.
  * LexiconFeatures -> a handful of dense, interpretable domain features:
    strict lexicon hits, graded near-miss scores, and how sure the variant
    selector was about the image's orientation.
"""
from __future__ import annotations

import math

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from .lexicon import ALCOHOL_TERMS, NON_ALCOHOL_TERMS, match_terms, soft_score
from .select import rank_variants, variant_text
from .text_norm import normalize


class SelectedText(BaseEstimator, TransformerMixin):
    """OCR variants -> normalized text of the best variant."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [normalize(variant_text(rank_variants(ocr)[0][1])) if ocr else "" for ocr in X]


class LexiconFeatures(BaseEstimator, TransformerMixin):
    """OCR variants -> dense domain features (see FEATURE_NAMES)."""

    FEATURE_NAMES = (
        "alc_best",      # strongest strict alcohol match in the best variant (0-1)
        "alc_count",     # number of distinct alcohol terms in the best variant
        "alc_runner_up", # strongest alcohol match in the 2nd-ranked variant
        "alc_soft",      # best near-miss similarity to an alcohol word (0-1)
        "non_best",      # strongest strict non-alcohol match in the best variant
        "non_soft",      # best near-miss similarity to a non-alcohol word
        "has_text",      # did any variant produce text at all
        "sel_score",     # log selector score of the best variant (readability)
        "sel_margin",    # how clearly the best variant beat the runner-up (0-1)
    )

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.array([self._row(ocr) for ocr in X], dtype=float)

    def get_feature_names_out(self, input_features=None):
        return np.array(self.FEATURE_NAMES)

    @staticmethod
    def _row(ocr: list[dict]) -> list[float]:
        ranked = rank_variants(ocr) if ocr else []
        texts = [variant_text(v) for _, v in ranked] + ["", ""]
        scores = [s for s, _ in ranked] + [0.0, 0.0]
        best, runner_up = texts[0], texts[1]

        alc = match_terms(best, ALCOHOL_TERMS, negate=True)
        alc2 = match_terms(runner_up, ALCOHOL_TERMS, negate=True)
        non = match_terms(best, NON_ALCOHOL_TERMS)
        return [
            max(alc.values(), default=0.0) / 100,
            len(alc),
            max(alc2.values(), default=0.0) / 100,
            soft_score(best, ALCOHOL_TERMS, negate=True) / 100,
            max(non.values(), default=0.0) / 100,
            soft_score(best, NON_ALCOHOL_TERMS) / 100,
            float(any(t.strip() for t in texts)),
            math.log1p(scores[0]),
            (scores[0] - scores[1]) / (scores[0] + 1.0),
        ]
