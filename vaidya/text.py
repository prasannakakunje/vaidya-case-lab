"""Text normalisation shared by every component.

Students type Sanskrit terms in many ways: 'vāta', 'vata', 'vaata', 'Vatha'.
Everything is folded to one plain ASCII form before matching, so the same
concept is recognised however it is spelled.
"""
from __future__ import annotations

import re
import unicodedata

from rapidfuzz import fuzz

# Common romanisation variants of Sanskrit sounds, folded to one form.
_FOLDS = [
    (r"aa", "a"), (r"ee", "i"), (r"ii", "i"), (r"oo", "u"), (r"uu", "u"),
    (r"sh", "s"), (r"ṣ", "s"), (r"ś", "s"),
    (r"th", "t"), (r"dh", "d"), (r"bh", "b"), (r"gh", "g"), (r"kh", "k"),
    (r"ph", "p"), (r"jh", "j"), (r"ch", "c"),
    (r"w", "v"), (r"ri", "r"), (r"ru", "r"),
]


def strip_diacritics(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    return "".join(c for c in text if not unicodedata.combining(c))


def words(text: str) -> list[str]:
    """Lower-case ASCII words, diacritics removed."""
    return re.findall(r"[a-z0-9]+", strip_diacritics(text).lower())


def fold(word: str) -> str:
    """Fold spelling variants: 'shleshma' and 'sleshma', 'vaata' and 'vata'."""
    w = word
    for a, b in _FOLDS:
        w = re.sub(a, b, w)
    return re.sub(r"(.)\1+", r"\1", w)  # collapse doubled letters


def folded(text: str) -> str:
    return " ".join(fold(w) for w in words(text))


def contains_phrase(text: str, phrase: str, threshold: int = 88) -> bool:
    """True if `phrase` (any spelling) appears in `text`.

    Exact substring on folded forms first; then a fuzzy match on word windows,
    so small typos ('virechna', 'langana') still count.
    """
    t, p = folded(text), folded(phrase)
    if not p:
        return False
    if f" {p} " in f" {t} " or (len(p) >= 5 and p in t):
        return True
    if len(p) >= 6 and p.replace(" ", "") in t.replace(" ", ""):
        return True  # 'udaka meha' = 'udakameha', 'kara pada daha' = 'karapada daha'
    t_words, n = t.split(), len(p.split())
    if n == 1 and len(p) >= 3 and any(w.startswith(p) for w in t_words):
        return True  # word stem: 'walk' matches 'walking', 'pregnan' matches 'pregnancy'
    if len(p) < 4:
        return False
    for i in range(len(t_words) - n + 1):
        window = " ".join(t_words[i:i + n])
        if fuzz.ratio(window, p) >= threshold:
            return True
    return False


def has_devanagari(text: str) -> bool:
    return any("ऀ" <= c <= "ॿ" for c in text)


def has_kannada(text: str) -> bool:
    return any("ಀ" <= c <= "೿" for c in text)


NEGATIONS = ("not", "never", "dont", "don't", "do not", "avoid stopping", "continue", "without stopping", "no need to stop")


def negated_before(text: str, start: int, window: int = 30) -> bool:
    """Is there a negation in the `window` characters before position `start`?"""
    before = text[max(0, start - window):start].lower()
    return any(neg in before for neg in NEGATIONS)
