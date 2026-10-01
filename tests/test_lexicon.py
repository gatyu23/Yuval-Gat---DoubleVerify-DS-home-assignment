"""Regression tests for text cleaning and lexicon matching.

Each case is a noise pattern seen in the dataset's OCR output, or a bug that
was fixed. Run with: python -m pytest tests
"""
import pytest

from alcohol_clf.lexicon import ALCOHOL_TERMS, NON_ALCOHOL_TERMS, match_terms
from alcohol_clf.text_norm import normalize, squash


def alcohol_hits(text):
    return set(match_terms(text, ALCOHOL_TERMS, negate=True))


@pytest.mark.parametrize("text, term", [
    ("Don't miss limited edition nightlife bourbon event", "bourbon"),
    ("Check out speclal] smooth budweis?r nightllfe", "budweiser"),  # OCR typo
    ("ch4mpa gne toast", "champagne"),      # leetspeak + broken word
    ("vod ka party", "vodka"),              # broken word
    ("g1n try", "gin"),                     # leetspeak in a short word
    ("noi for charnpagne", "champagne"),    # rn -> m glyph confusion
    ("smimoff", "smirnoff"),                # m -> rn glyph confusion
    ("now wlne smooth", "wine"),            # l -> i glyph confusion
    ("grab now jack danlels", "daniels"),
    ("lager night", "lager"),               # fold must not cross word boundary
    ("Moët party", "moet"),                 # accents
])
def test_alcohol_found(text, term):
    assert term in alcohol_hits(text)


@pytest.mark.parametrize("text", [
    "ey",                                   # short text must not match "whiskey"
    "water",                                # ... or "beefeater"
    "sports report",                        # "port" is token-only
    "ginger shot",                          # "gin" is token-only
    "Check out partybrew herbal tea party", # "brewery" is token-only
    "Try our NON-ALCOHOLIC beer",           # negating prefix masks next word
    "non alco holic wine",                  # ... even when OCR splits it
    "alcohol free lager",
    "virgin mojito",
    "root beer float",                      # look-alike products
    "ginger ale",
])
def test_no_alcohol(text):
    assert alcohol_hits(text) == set()


def test_negation_only_masks_the_next_word():
    assert alcohol_hits("non alcoholic beer and vodka") == {"vodka"}


def test_non_alcohol_terms():
    assert "energy drink" in match_terms("ener gy drink", NON_ALCOHOL_TERMS)


def test_normalize_keeps_numbers_and_trailing_punctuation():
    assert normalize("VODKA!! 2 for 1") == "vodka 2 for 1"
    assert squash("pre mium ch4mpa gne") == "premiumchampagne"
