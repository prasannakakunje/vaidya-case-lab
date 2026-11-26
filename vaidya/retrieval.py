"""Pramana library and retrieval.

Verses are the unit of retrieval, so every citation points to a whole,
checkable verse. Search is BM25 over diacritic-free, spelling-folded text,
with two abstention guards:
  1. a relevance floor relative to the best hit, and
  2. the question's subject words (disease, concept) must appear in the verse.
If nothing passes, the system says so instead of guessing.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from rank_bm25 import BM25Okapi

from .text import fold, words

DATA = Path(__file__).parent / "data"

# Words that say what kind of question it is, not what it is about.
GENERIC = set(fold(w) for w in (
    "what which who how why when where is are was be the a an of for in on to and or with by from about "
    "does do should can tell me explain describe according ayurveda patient treatment treat treated treating "
    "line chikitsa sutra principle management manage cause causes caused nidana hetu samprapti pathogenesis "
    "symptoms symptom signs sign rupa lakshana role important give list classical text texts reference verse shloka "
    "say says said roga rogi vyadhi disease best good use used using way ways charaka caraka sushruta susruta vagbhata samhita ashtanga hridaya madhava chakradatta cite citation citations source sources references please according briefly detail".split()))


@dataclass
class Verse:
    id: str
    text: str
    location: str
    topic: str
    sanskrit: str
    meaning: str
    keywords: list
    verified: bool = False

    @property
    def reference(self) -> str:
        return f"{self.text}, {self.location}"

    def as_markdown(self, n: int | None = None) -> str:
        tag = f"[{n}] " if n is not None else ""
        badge = "verified by faculty" if self.verified else "pending faculty verification"
        if not self.sanskrit:
            badge += "; meaning only, verse text to be added"
        lines = [f"**{tag}{self.reference}** - {self.topic} _({badge})_"]
        if self.sanskrit:
            lines.append(f"> *{self.sanskrit}*")
        lines.append(f"{self.meaning}")
        return "\n\n".join(lines)


class PramanaLibrary:
    def __init__(self, path: Path | None = None):
        raw = json.loads((path or DATA / "verses.json").read_text(encoding="utf-8"))
        self.verses = [Verse(**v) for v in raw["verses"]]
        self.by_id = {v.id: v for v in self.verses}
        self._docs = [self._doc_words(v) for v in self.verses]
        self._bm25 = BM25Okapi(self._docs)

    @staticmethod
    def _doc_words(v: Verse) -> list[str]:
        text = " ".join([v.text, v.topic, v.sanskrit, v.meaning, " ".join(v.keywords)])
        return [fold(w) for w in words(text)]

    def get(self, verse_id: str) -> Verse | None:
        return self.by_id.get(verse_id)

    def search(self, question: str, k: int = 3, rel_floor: float = 0.5, min_cover: float = 0.5) -> list[Verse]:
        q = [fold(w) for w in words(question)]
        if not q:
            return []
        subject = [w for w in q if w not in GENERIC and len(w) > 2]
        scores = self._bm25.get_scores(q)
        best = max(scores) if len(scores) else 0
        if best <= 0:
            return []
        out = []
        for score, i in sorted(zip(scores, range(len(self.verses))), reverse=True)[:k]:
            if score <= 0 or score < rel_floor * best:
                continue
            doc = set(self._docs[i])
            # The verse must cover at least half of what the question is about.
            if subject and len(doc & set(subject)) / len(set(subject)) < min_cover:
                continue
            out.append(self.verses[i])
        return out
