"""Text cleaning for noisy / adversarial OCR output.

Three kinds of noise show up in this dataset:
  1. Adversarial obfuscation - leetspeak ("ch4mpagne", "v0dka", "$cotch").
  2. Layout noise - words broken by random spaces ("pre mium", "ch4mpa gne").
  3. OCR confusions - visually similar glyphs (0/o, 1/l/i, 5/s, rn/m).

`normalize` produces a clean token string for the ML model; `squash` produces a
space-free character stream so that fuzzy matching can see across broken words.
"""
from __future__ import annotations

import re
import unicodedata

# Digit/symbol -> letter substitutions. Covers deliberate leetspeak and the most
# common OCR glyph confusions. Applied only *inside* words that also contain
# letters, so standalone numbers ("2 for 1", "21+") survive.
LEET_MAP = {
    "0": "o",
    "1": "i",
    "3": "e",
    "4": "a",
    "5": "s",
    "7": "t",
    "8": "b",
    "9": "g",
    "@": "a",
    "$": "s",
    "!": "i",
    "|": "l",
    "€": "e",
}
_LEET_RE = re.compile("[" + re.escape("".join(LEET_MAP)) + "]")
_WORD_RE = re.compile(r"\S+")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9 ]+")
_SPACES_RE = re.compile(r"\s+")


def _strip_accents(text: str) -> str:
    # "Moët" -> "Moet", "Jägermeister" -> "Jagermeister"
    return "".join(
        c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c)
    )


def _deleet_word(word: str) -> str:
    # Edge punctuation is sentence punctuation ("VODKA!!"), not leetspeak;
    # only a mid-word "!" or a leading "$" stands in for a letter.
    word = word.rstrip("!|.,;:?").lstrip("!|")
    if not any(c.isalpha() for c in word):
        return word
    return _LEET_RE.sub(lambda m: LEET_MAP[m.group()], word)


def normalize(text: str) -> str:
    """Lowercase, strip accents, undo leetspeak, drop punctuation, collapse spaces."""
    text = _strip_accents(text).lower()
    text = " ".join(_deleet_word(w) for w in _WORD_RE.findall(text))
    text = _NON_ALNUM_RE.sub(" ", text)
    return _SPACES_RE.sub(" ", text).strip()


def squash(text: str) -> str:
    """Normalized text with all spaces removed: 'ch4mpa gne' -> 'champagne'."""
    return normalize(text).replace(" ", "")
