"""Choose which OCR variant (orig / flipped / deskewed) to trust.

`ocr.py` reads every image in up to 8 orientations. Only one of them is the
right way up; the others produce mostly gibberish, but gibberish can still come
with confident detections. A variant is scored by two signals:

  * word mass - characters in tokens that (fuzzily) match a known word. The
    vocabulary mixes alcohol, non-alcohol and ad-template words, so it favours
    readable text without favouring either label.
  * confidence mass - sum of EasyOCR confidence x text length, a tie-breaker
    for text outside the vocabulary.
"""
from __future__ import annotations

from rapidfuzz import fuzz, process

from .lexicon import VOCAB
from .text_norm import normalize

VOCAB_WORDS = sorted({w for term in VOCAB for w in term.split() if len(w) >= 3})
WORD_CUTOFF = 80.0  # one OCR error in a 5-letter word still counts
CONF_WEIGHT = 0.5


def variant_text(variant: dict) -> str:
    return " ".join(d["text"] for d in variant["dets"])


def word_mass(text: str) -> int:
    mass = 0
    for tok in normalize(text).split():
        if len(tok) >= 3 and process.extractOne(
            tok, VOCAB_WORDS, scorer=fuzz.ratio, score_cutoff=WORD_CUTOFF
        ):
            mass += len(tok)
    return mass


def conf_mass(variant: dict) -> float:
    return sum(d["conf"] * len(d["text"].strip()) for d in variant["dets"])


def score_variant(variant: dict) -> float:
    return word_mass(variant_text(variant)) + CONF_WEIGHT * conf_mass(variant)


def rank_variants(ocr: list[dict]) -> list[tuple[float, dict]]:
    """Variants sorted best-first, each with its score."""
    scored = [(score_variant(v), v) for v in ocr]
    return sorted(scored, key=lambda sv: sv[0], reverse=True)


def select_text(ocr: list[dict]) -> str:
    """Text of the best variant ('' if OCR found nothing anywhere)."""
    if not ocr:
        return ""
    return variant_text(rank_variants(ocr)[0][1])
