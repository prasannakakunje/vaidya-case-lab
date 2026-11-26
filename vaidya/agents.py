"""The three agents. Each is AyurParam with a different role and a tightly
controlled context. Facts never come from the model: they come from the
faculty case file and the Pramana library. The model only supplies language,
and every reply is checked before the student sees it.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .cases import Case, Fact, reply_language, select_facts
from .llm import Engine, build_prompt
from .retrieval import PramanaLibrary, Verse
from .rules import Alert, check_plan
from .text import contains_phrase, fold, folded, words

# ---------------------------------------------------------------- shared checks

STOP = set(fold(w) for w in (
    "the and that this with from have has had for are was were you your they them their there what when which "
    "about into also been being more most such than then very will would should could may might must can its "
    "not but all any one two some other these those our out over only same each".split()))


def support_score(answer: str, context: str) -> float:
    """Share of the answer's content words that also appear in the context."""
    ans = [w for w in folded(answer).split() if len(w) > 3 and w not in STOP]
    if not ans:
        return 0.0
    ctx = set(folded(context).split())
    return sum(w in ctx for w in ans) / len(ans)


# ---------------------------------------------------------------- Rogi (patient)

DIAGNOSTIC_WORDS = ["vata", "pitta", "kapha", "dosha", "samprapti", "nidana", "diagnos", "ayurved", "vyadhi",
                    "srotas", "dushya", "as a patient", "as an ai", "language model"]
GREETINGS = ["hello", "hi", "namaste", "namaskara", "good morning", "good evening", "namaskar"]


@dataclass
class PatientReply:
    text: str
    new_facts: list[str]
    source: str  # "AyurParam" or "case file"


class RogiAgent:
    def __init__(self, engine: Engine):
        self.engine = engine

    def opening(self, case: Case) -> str:
        return case.opening

    def reply(self, case: Case, question: str, revealed: set[str], use_model: bool = True) -> PatientReply:
        q = question.strip()
        if not q:
            return PatientReply("...", [], "case file")
        facts = select_facts(case, q, revealed)
        if not facts:
            if any(folded(q).startswith(folded(g)) for g in GREETINGS) and len(words(q)) <= 4:
                return PatientReply(f"Namaskara, doctor. {case.opening}", [], "case file")
            return PatientReply("No, doctor, I have not noticed anything like that.", [], "case file")
        base = " ".join(f.text for f in facts)
        new = [f.id for f in facts if f.id not in revealed]
        lang = reply_language(q)
        if use_model and self.engine.ready:
            spoken = self._speak(case, q, facts, lang)
            if spoken:
                return PatientReply(spoken, new, "AyurParam")
        return PatientReply(base, new, "case file")

    def _speak(self, case: Case, question: str, facts: list[Fact], lang: str) -> str | None:
        p = case.patient
        lang_line = {"hi": " Answer in simple Hindi.", "kn": ""}.get(lang, "")
        context = (f"You are {p['name']}, a {p['age']}-year-old {p['occupation'].lower()} visiting an Ayurveda OPD. "
                   f"Reply to the doctor as this patient, in one or two short sentences of everyday language, "
                   f"saying only this: \"{' '.join(f.text for f in facts)}\" Do not add anything else. "
                   f"Never name a disease, dosha or diagnosis.{lang_line}")
        out = self.engine.generate(build_prompt(f"Doctor: {question}", context), max_tokens=90, temperature=0.5)
        return out if out and self._faithful(out, case, facts) else None

    @staticmethod
    def _faithful(text: str, case: Case, facts: list[Fact]) -> bool:
        if len(words(text)) < 3 or len(words(text)) > 80:
            return False
        if any(contains_phrase(text, w) for w in DIAGNOSTIC_WORDS + [case.disease]):
            return False
        source = " ".join(f.text for f in facts)
        # Every number the patient says must be in the case file (no invented durations or values).
        return all(n in source for n in re.findall(r"\d+(?:\.\d+)?", text))


# ---------------------------------------------------------------- Pariksha (examiner)

FIELD_WEIGHTS = {"dosha": 2, "vyadhi": 2, "chikitsa_sutra": 2, "plan": 2}


@dataclass
class FieldResult:
    field: str
    score: float
    matched: list[str]
    missed: list[str]
    wrong: list[str]
    reference: str
    verse_ids: list[str]
    answered: bool


@dataclass
class Assessment:
    fields: list[FieldResult]
    reasoning_score: float
    history_found: list[str]
    history_missed: list[str]
    exams_done: int
    alerts: list[Alert]
    teacher_note: str | None = None
    teacher_note_source: str = ""
    verse_ids: list[str] = field(default_factory=list)

    @property
    def blocked(self) -> bool:
        return any(a.severity == "block" for a in self.alerts)

    @property
    def total(self) -> int:
        history = len(self.history_found) / max(1, len(self.history_found) + len(self.history_missed))
        exams = min(1.0, self.exams_done / 5)
        return round(100 * (0.6 * self.reasoning_score + 0.25 * history + 0.15 * exams))


def _concept_present(text: str, aliases: list[str]) -> bool:
    return any(contains_phrase(text, a) for a in aliases)


class ParikshaAgent:
    def __init__(self, engine: Engine, library: PramanaLibrary):
        self.engine = engine
        self.library = library

    def grade(self, case: Case, answers: dict[str, str], revealed: set[str], exams_done: set[str],
              use_model: bool = True) -> Assessment:
        results, weighted, total_w = [], 0.0, 0.0
        for fname, ref in case.reference.items():
            text = answers.get(fname, "") or ""
            req, wrong = ref.get("required", []), ref.get("wrong", [])
            matched = [a[0] for a in req if _concept_present(text, a)]
            missed = [a[0] for a in req if not _concept_present(text, a)]
            wrong_found = [a[0] for a in wrong if _concept_present(text, a)]
            score = len(matched) / len(req) if req else 1.0
            score = max(0.0, score - 0.5 * len(wrong_found))
            w = FIELD_WEIGHTS.get(fname, 1)
            weighted += w * score
            total_w += w
            results.append(FieldResult(fname, score, matched, missed, wrong_found, ref["answer"],
                                       ref.get("verses", []), bool(text.strip())))

        key_ids = [f.id for f in case.key_facts]
        found = [i for i in key_ids if i in revealed]
        missed_hist = [i for i in key_ids if i not in revealed]
        plan_text = "\n".join([answers.get("chikitsa_sutra", ""), answers.get("plan", "")])
        alerts = check_plan(case, plan_text, revealed)

        verse_ids: list[str] = []
        for a in alerts:
            verse_ids += [v for v in a.verse_ids if v not in verse_ids]
        for r in sorted(results, key=lambda r: r.score):
            if r.score < 1:
                verse_ids += [v for v in r.verse_ids if v not in verse_ids]

        a = Assessment(results, weighted / total_w if total_w else 0.0, found, missed_hist, len(exams_done),
                       alerts, verse_ids=verse_ids)
        if use_model and self.engine.ready:
            a.teacher_note, a.teacher_note_source = self._teacher_note(case, a)
        return a

    def _teacher_note(self, case: Case, a: Assessment) -> tuple[str | None, str]:
        weak = [r for r in a.fields if r.score < 1][:3]
        verses = [self.library.get(v) for v in a.verse_ids[:2]]
        verses = [v for v in verses if v]
        if not weak or not verses:
            return None, ""
        context = " ".join(f"[{i + 1}] {v.reference}: {v.sanskrit} - {v.meaning}" for i, v in enumerate(verses))
        gaps = "; ".join(f"{r.field}: missed {', '.join(r.missed) or 'details'}" for r in weak)
        q = (f"A BAMS student assessed a case of {case.disease} and made these gaps: {gaps}. "
             "Using only the references above, explain in three short sentences what the student should learn. "
             "Cite the reference number like [1].")
        out = self.engine.generate(build_prompt(q, context), max_tokens=160, temperature=0.3)
        if out and support_score(out, context + " " + gaps + " " + case.disease) >= 0.45:
            return out, "AyurParam, grounded in the verses shown below"
        return None, ""


# ---------------------------------------------------------------- Pramana (explainer)

@dataclass
class PramanaAnswer:
    verses: list[Verse]
    answer: str | None
    status: str  # "answered", "verses-only", "abstained"


class PramanaAgent:
    def __init__(self, engine: Engine, library: PramanaLibrary):
        self.engine = engine
        self.library = library

    def ask(self, question: str, use_model: bool = True) -> PramanaAnswer:
        verses = self.library.search(question)
        if not verses:
            return PramanaAnswer([], None, "abstained")
        if not (use_model and self.engine.ready):
            return PramanaAnswer(verses, None, "verses-only")
        context = " ".join(f"[{i + 1}] {v.reference}: {v.sanskrit} - {v.meaning}" for i, v in enumerate(verses))
        prompt = build_prompt(f"Using only the references above, answer: {question} "
                              "Put the reference number like [1] after each point.", context)
        out = self.engine.generate(prompt, max_tokens=200, temperature=0.3)
        if not out:
            return PramanaAnswer(verses, None, "verses-only")
        # Remove citations to references that were never given to the model.
        out = re.sub(r"\[(\d+)\]", lambda m: m.group(0) if 1 <= int(m.group(1)) <= len(verses) else "", out)
        if support_score(out, context + " " + question) < 0.5:
            return PramanaAnswer(verses, None, "verses-only")
        return PramanaAnswer(verses, out, "answered")


# ---------------------------------------------------------------- plain vs grounded (demonstration)

CITATION_CLAIM = re.compile(
    r"(charaka|caraka|sushruta|susruta|vagbhata|ashtanga|madhava|chakradatta|bhavaprakasha|sharangadhara|"
    r"sutra ?sthana|nidana ?sthana|chikitsa ?sthana|sharira ?sthana|adhyaya|chapter|verse|shloka)", re.I)


@dataclass
class Comparison:
    plain: str | None
    plain_cites_texts: bool
    grounded: PramanaAnswer


def compare(engine: Engine, pramana: PramanaAgent, question: str) -> Comparison:
    """Ask AyurParam the same question twice: on its own, and through the Pramana pipeline."""
    plain = engine.generate(build_prompt(question), max_tokens=200, temperature=0.3) if engine.ready else None
    cites = bool(plain and CITATION_CLAIM.search(plain))
    return Comparison(plain, cites, pramana.ask(question, use_model=True))
