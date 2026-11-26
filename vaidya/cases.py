"""Case bank and the deterministic part of the Rogi (patient) agent.

The patient reveals a fact only when the student's question touches that
fact's topic. Facts marked `hidden` test whether the student takes a complete
history: the patient never volunteers them.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .text import folded, has_devanagari, has_kannada

DATA = Path(__file__).parent / "data"

# Words (English, and common romanised Hindi / Kannada) that tell us what a
# question is about. Matching is done on folded spellings.
TOPIC_WORDS: dict[str, list[str]] = {
    "complaint": ["problem", "trouble", "complaint", "brings you", "bring you", "wrong", "issue", "what happened", "takleef", "taklif", "samasya", "enu aagide", "tondare", "help you"],
    "pain": ["pain", "ache", "hurt", "sore", "swelling", "swollen", "stiff", "joint", "dard", "novu", "noovu", "shula", "headache", "burning", "burn"],
    "timing": ["when", "worse", "better", "aggravat", "relie", "trigger", "morning", "evening", "weather", "season", "winter", "cold", "warm", "time of day", "kab", "yavaga"],
    "duration": ["how long", "since", "started", "start", "duration", "begin", "began", "first notice", "kab se", "yestu dina", "how many days", "months"],
    "appetite": ["appetite", "hunger", "hungry", "taste", "eat", "bhook", "bhukh", "hasivu", "ruchi"],
    "digestion": ["digest", "gas", "bloat", "belch", "burp", "acid", "heartburn", "sour", "vomit", "nausea", "stomach", "indigestion", "pet"],
    "bowel": ["stool", "motion", "bowel", "constipat", "latrine", "toilet", "mala", "shauch", "loose", "black", "potty", "poop"],
    "urine": ["urine", "urinat", "pee", "passing water", "mutra", "peshab", "toilet"],
    "sleep": ["sleep", "nap", "nidra", "neend", "nidde", "rest", "bed"],
    "diet": ["diet", "food", "meal", "dinner", "breakfast", "lunch", "curd", "sweet", "tea", "coffee", "drink", "eat", "khana", "oota", "ahara", "habit", "snack", "spicy", "milk"],
    "lifestyle": ["exercise", "work", "job", "routine", "walk", "sit", "activity", "daily", "lifestyle", "stress", "occupation", "office", "vihara", "kaam", "kelasa", "habit"],
    "family": ["family", "father", "mother", "parent", "hereditary", "relative", "brother", "sister", "genetic"],
    "medication": ["medicine", "medication", "tablet", "drug", "taking any", "treatment", "inhaler", "pill", "dawa", "dawai", "aushadhi", "oushadha", "syrup", "allopath", "metformin", "massage", "oil"],
    "menses": ["period", "menses", "menstru", "lmp", "pregnan", "cycle", "masik", "mutthu", "missed", "conceive"],
    "weight": ["weight", "kilo", "kg", "thin", "fat", "lost", "gain", "vajan", "tooka"],
    "fever": ["fever", "temperature", "jvara", "jwara", "bukhar", "jvara", "chill"],
    "breathing": ["breath", "wheez", "cough", "phlegm", "sputum", "chest", "saans", "usiru", "kemmu", "kapha"],
    "skin": ["sweat", "odour", "odor", "smell", "skin", "itch", "palm", "sole", "mouth", "sticky"],
    "energy": ["tired", "fatigue", "weak", "energy", "lazy", "heavy", "heaviness", "dull", "body feel"],
    "past": ["before", "previous", "past", "history", "tried", "earlier", "ever", "already", "any treatment", "massage", "oil"],
    "tests": ["test", "report", "blood", "sugar", "investigation", "lab", "hba1c", "scan"],
    "personal": ["married", "marital", "husband", "wife", "children", "alcohol", "smok", "addiction"],
}

EXAMINATIONS = [
    ("nadi", "Nadi (pulse)"),
    ("jihva", "Jihva (tongue)"),
    ("mala", "Mala (stool)"),
    ("mutra", "Mutra (urine)"),
    ("agni", "Agni assessment"),
    ("vitals", "Vitals"),
    ("general", "General / anthropometry"),
    ("local", "Local and systemic examination"),
    ("labs", "Investigations"),
]

REASONING_FIELDS = [
    ("nidana", "Nidana (causes)"),
    ("purvarupa", "Purvarupa (prodrome)"),
    ("rupa", "Rupa (signs and symptoms)"),
    ("upashaya", "Upashaya / Anupashaya"),
    ("dosha", "Samprapti: Dosha"),
    ("dushya", "Samprapti: Dushya"),
    ("srotas", "Samprapti: Srotas"),
    ("agni", "Samprapti: Agni"),
    ("adhishthana", "Samprapti: Adhishthana"),
    ("vyadhi", "Vyadhi (diagnosis)"),
    ("vyavachedaka", "Vyavachedaka Nidana (differentials)"),
    ("chikitsa_sutra", "Chikitsa Sutra (line of treatment)"),
    ("plan", "Treatment plan: medicines, procedures, Anupana, Pathya-Apathya"),
]


@dataclass
class Fact:
    id: str
    topics: list[str]
    text: str
    key: bool = False
    hidden: bool = False


@dataclass
class Case:
    id: str
    title: str
    disease: str
    difficulty: str
    patient: dict
    opening: str
    facts: list[Fact]
    examinations: dict
    reference: dict
    flags: dict = field(default_factory=dict)

    @property
    def key_facts(self) -> list[Fact]:
        return [f for f in self.facts if f.key]

    def fact(self, fact_id: str) -> Fact | None:
        return next((f for f in self.facts if f.id == fact_id), None)


def load_cases(path: Path | None = None) -> dict[str, Case]:
    raw = json.loads((path or DATA / "cases.json").read_text(encoding="utf-8"))
    cases = {}
    for c in raw["cases"]:
        facts = [Fact(**f) for f in c.pop("facts")]
        cases[c["id"]] = Case(facts=facts, **c)
    return cases


def question_topics(question: str) -> set[str]:
    q = f" {folded(question)} "
    found = set()
    for topic, cue_words in TOPIC_WORDS.items():
        for w in cue_words:
            fw = folded(w)
            if fw and (f" {fw}" in q):  # word-start match: 'pain' matches 'painful'
                found.add(topic)
                break
    return found


def select_facts(case: Case, question: str, already: set[str], limit: int = 2) -> list[Fact]:
    """Facts the patient should reveal in answer to `question`.

    New facts first, then ones already told (a patient may repeat himself).
    Returns [] when the question touches nothing in the case file.
    """
    topics = question_topics(question)
    if not topics:
        return []
    hits = [f for f in case.facts if topics & set(f.topics)]
    # A hidden fact is revealed only when the question is specifically about
    # one of its topics; it never piggybacks on a broad question like 'complaint'.
    hits = [f for f in hits if not f.hidden or (topics & set(f.topics)) - {"complaint"}]
    fresh = [f for f in hits if f.id not in already]
    old = [f for f in hits if f.id in already]
    return (fresh + old)[:limit]


def reply_language(question: str) -> str:
    if has_devanagari(question):
        return "hi"
    if has_kannada(question):
        return "kn"
    return "en"
