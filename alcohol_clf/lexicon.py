"""Domain knowledge: alcohol vocabulary, brands, and look-alike negatives.

The dataset's captions follow an ad template - "<call to action> <adjective>
<product> <context>", e.g. "Don't miss limited edition nightlife bourbon event".
Template words ("premium", "brew", "bar", "nightlife") appear in both classes,
so the label is carried by the *product* word. This module lists those product
words and provides a noise-tolerant matcher for them.

Matching rules, by term length (lengths counted without spaces):
  * <= 3 chars ("gin", "rum", "ale"): exact token only. As substrings they fire
    on "ginger", "drum", "sale".
  * 4-5 chars ("beer", "vodka"): exact substring of the space-free text, which
    survives broken words ("vod ka") but not misspellings.
  * >= 6 chars ("heineken", "chardonnay"): fuzzy partial match, tolerating OCR
    errors like "budweis?r" or "nightllfe".
Before matching alcohol terms, negated phrases ("non-alcoholic beer") and
look-alike products ("root beer", "ginger ale") are masked out so their alcohol
words don't fire.
"""
from __future__ import annotations

import re
from functools import lru_cache

from rapidfuzz import fuzz, process

from .text_norm import normalize

ALCOHOL_GENERIC = {
    # beer
    "beer", "lager", "ale", "ipa", "stout", "porter", "pilsner", "pils",
    "craft beer", "brewery", "pint", "keg", "hefeweizen",
    # wine
    "wine", "red wine", "white wine", "merlot", "chardonnay", "cabernet",
    "sauvignon", "pinot", "riesling", "shiraz", "malbec", "zinfandel", "rioja",
    "chianti", "prosecco", "champagne", "cava", "sangria", "winery", "vineyard",
    "port", "sherry", "vermouth", "sake", "soju", "cider", "mead",
    # spirits
    "vodka", "gin", "rum", "whiskey", "whisky", "bourbon", "scotch", "brandy",
    "cognac", "tequila", "mezcal", "absinthe", "schnapps", "liqueur", "liquor",
    "spirits", "distillery", "moonshine", "grappa", "ouzo", "sambuca",
    # drinks and occasions
    "cocktail", "margarita", "mojito", "martini", "daiquiri", "negroni",
    "cosmopolitan", "spritz", "mimosa", "happy hour", "booze", "alcohol",
    "alcoholic", "hangover", "tipsy",
}

ALCOHOL_BRANDS = {
    # beer
    "heineken", "budweiser", "bud light", "corona", "stella artois", "stella",
    "artois", "guinness", "coors", "miller", "carlsberg", "becks", "peroni",
    "modelo", "michelob", "pabst", "samuel adams", "blue moon", "hoegaarden",
    "leffe", "asahi", "sapporo", "kirin", "tsingtao", "amstel", "tuborg",
    "strongbow", "magners", "white claw",
    # spirits
    "smirnoff", "absolut", "grey goose", "belvedere", "ketel one", "titos",
    "bacardi", "captain morgan", "malibu", "havana club", "kraken",
    "jack daniels", "jack daniel", "daniels", "jim beam", "jameson", "johnnie walker",
    "johnnie", "chivas regal", "glenfiddich", "glenlivet", "macallan", "makers mark",
    "wild turkey", "crown royal", "jagermeister", "baileys", "kahlua",
    "cointreau", "grand marnier", "hennessy", "remy martin", "courvoisier",
    "martell", "jose cuervo", "patron", "don julio", "tanqueray",
    "bombay sapphire", "beefeater", "hendricks", "gordons", "fireball",
    "southern comfort", "disaronno", "campari", "aperol",
    # wine / champagne
    "moet", "dom perignon", "veuve clicquot", "krug", "barefoot",
    "yellow tail", "jacobs creek",
}

# Non-alcoholic products that share the ad template, plus the other "vice"
# products that appear in the negative class (cannabis). Used as evidence for
# the negative class, and to give the variant selector a label-neutral vocabulary.
NON_ALCOHOL_TERMS = {
    "tea", "herbal tea", "iced tea", "matcha", "chai", "coffee", "espresso",
    "latte", "cappuccino", "hot chocolate", "chocolate", "milk", "milkshake",
    "smoothie", "juice", "lemonade", "soda", "cola", "water", "sparkling water",
    "energy drink", "red bull", "kombucha", "mocktail", "root beer",
    "ginger beer", "ginger ale", "cream soda", "grape juice",
    "cannabis", "weed", "marijuana", "hemp", "cbd", "vape", "tobacco",
    "cigarette", "cigar",
}

# Negating prefixes: the phrase AND the product word right after it are masked
# ("non-alcoholic beer", "alcohol free wine").
NEGATING_PREFIXES = (
    "non alcoholic", "nonalcoholic", "alcohol free", "zero alcohol",
    "zero proof", "no alcohol", "without alcohol", "virgin",
)
# Look-alike products: only the phrase itself is masked.
LOOKALIKE_PHRASES = (
    "root beer", "ginger beer", "ginger ale", "butterbeer", "butter beer",
    "birch beer",
)

# Label-neutral ad-template words. Not used as class evidence; they only tell
# the variant selector that an OCR reading is real language, not gibberish.
AD_WORDS = {
    "dont", "don", "miss", "check", "out", "try", "get", "ready", "for", "grab",
    "now", "offer", "special", "premium", "imported", "smooth", "limited",
    "edition", "event", "party", "nightlife", "celebration", "reserve",
    "taste", "flavor", "flavour", "explore", "discount", "menu", "bar", "brew",
    "enjoy", "experience", "looking", "hot", "iced", "new", "best", "fresh",
    "drink", "drinks", "the", "and", "with", "your", "our", "this", "sale",
    "free", "club", "night", "weekend", "tonight", "today", "classic",
}

ALCOHOL_TERMS = ALCOHOL_GENERIC | ALCOHOL_BRANDS
VOCAB = ALCOHOL_TERMS | NON_ALCOHOL_TERMS | AD_WORDS

EXACT_TOKEN_MAX_LEN = 3
EXACT_SUBSTRING_MAX_LEN = 5
FUZZY_CUTOFF = 85.0
# Terms that are also everyday words or live inside them ("report", "for the
# sake of", "corona virus"), or that start with the shared template word "brew"
# ("partybrew herbal" fuzzy-matches "brewery"): whole-token matches only.
TOKEN_ONLY = {
    "port", "sake", "cava", "mead", "pils", "pint", "stout", "spirits", "brewery",
    "corona", "miller", "patron", "malibu", "stella", "krug", "moet",
}


def _loose(phrase: str) -> str:
    """Regex for `phrase` that tolerates OCR spacing: 'non alco holic'."""
    return r"\s?".join(re.escape(c) for c in phrase.replace(" ", ""))


_NEGATION_RE = re.compile(
    "|".join(rf"\b{_loose(p)}\b(\s+\S+)?" for p in NEGATING_PREFIXES)
    + "|"
    + "|".join(rf"\b{_loose(p)}\b" for p in LOOKALIKE_PHRASES)
)


# Glyph confusions EasyOCR makes in both directions ("charnpagne", "smimoff",
# "wlne", "danlels"). Folded on both the term and the text, so either spelling
# matches.
_GLYPH_FOLDS = (("rn", "m"), ("vv", "w"), ("l", "i"))


def _fold(s: str) -> str:
    for a, b in _GLYPH_FOLDS:
        s = s.replace(a, b)
    return s


def _views(text: str, negate: bool) -> tuple[str, str]:
    """(spaced, squashed) normalized views of `text`, negations masked if asked.

    A mask leaves a "|" token so no match can span it in the squashed view.
    The squashed view is also glyph-folded.
    """
    spaced = normalize(text)
    if negate:
        spaced = _NEGATION_RE.sub(" | ", spaced)
    # Fold word by word: folding after squashing would merge across a word
    # boundary ("lager night" -> "lage|rn|ight" -> "lagemight").
    return f" {spaced} ", "".join(_fold(tok) for tok in spaced.split())


def match_terms(
    text: str, terms: set[str], cutoff: float = FUZZY_CUTOFF, negate: bool = False
) -> dict[str, float]:
    """Return {term: score in [0, 100]} for every term found in `text`.

    `text` is raw OCR output; it is normalized here. With `negate=True`,
    negation phrases are masked first (use for alcohol terms).
    """
    spaced, squashed = _views(text, negate)
    hits = {}
    for term in terms:
        key = term.replace(" ", "")
        if len(key) <= EXACT_TOKEN_MAX_LEN or term in TOKEN_ONLY:
            if f" {term} " in spaced:
                hits[term] = 100.0
        elif len(key) <= EXACT_SUBSTRING_MAX_LEN:
            if _fold(key) in squashed:
                hits[term] = 100.0
        else:
            key = _fold(key)
            # partial_ratio aligns the SHORTER string inside the longer one, so
            # text shorter than the term ("ey") would match it ("whiskey").
            scorer = fuzz.partial_ratio if len(squashed) >= len(key) else fuzz.ratio
            score = scorer(key, squashed, score_cutoff=cutoff)
            if score:
                hits[term] = score
    return hits


def soft_score(text: str, terms: set[str], negate: bool = False) -> float:
    """Best fuzzy similarity (0-100) between any token of `text` and any word
    of `terms`, with no cutoff.

    `match_terms` is deliberately strict; this graded score keeps the near
    misses it rejects ("vedka" ~ vodka: 80, "whekey" ~ whiskey: 77) so the
    classifier can learn how much to trust them.
    """
    spaced, _ = _views(text, negate)
    words = _soft_vocab(frozenset(terms))
    best = 0.0
    for tok in spaced.split():
        if len(tok) >= SOFT_MIN_LEN:
            hit = process.extractOne(_fold(tok), words, scorer=fuzz.ratio)
            if hit:
                best = max(best, hit[1])
    return best


SOFT_MIN_LEN = 4  # shorter tokens match too many terms by chance


@lru_cache(maxsize=None)
def _soft_vocab(terms: frozenset[str]) -> list[str]:
    # Single-word terms only: splitting "root beer" would make "beer" count as
    # non-alcohol evidence.
    return sorted({_fold(t) for t in terms if " " not in t and len(t) >= SOFT_MIN_LEN})
