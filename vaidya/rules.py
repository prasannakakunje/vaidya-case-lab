"""Contraindication rules engine.

Deterministic: it does not depend on the language model at all. Every rule
returns the reason and the Pramana behind it, so a caught error becomes a lesson.

Severity:
  block - unsafe; the case is not passed until the plan is corrected
  warn  - questionable; shown as a teaching point
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .cases import Case
from .text import contains_phrase, folded, negated_before, strip_diacritics

SHODHANA_LEKHANA = ["virechana", "vamana", "lekhana", "lekhana basti", "tikshna", "teekshna", "shodhana", "purgation",
                    "emesis", "raktamokshana", "udvartana", "upavasa", "fasting", "navaka guggulu",
                    "triphala guggulu", "vidangadi", "kanchanar", "medohara guggulu"]
SNEHANA = ["abhyanga", "snehapana", "sneha pana", "snehana", "ghrita pana", "ghee", "oil massage", "taila abhyanga",
           "brimhana", "brumhana", "ksheera", "milk"]
NIRAMA_CUES = ["nirama", "after ama", "after pachana", "after langhana", "later", "once ama", "after deepana"]
VISHA = ["vatsanabha", "vatsanabh", "kupilu", "vishatinduka", "bhallataka", "shvasa kuthara", "swasa kuthara",
         "agnitundi", "hingula", "dhattura", "ahiphena", "parada", "rasa sindura", "mahayogaraja"]
REFERRAL = ["refer", "endoscopy", "gastroenterolog", "hospital", "emergency", "specialist", "ugi", "upper gi"]
MODERN_MEDS = ["metformin", "insulin", "inhaler", "salbutamol", "asthalin", "allopathic", "english medicine",
               "modern medicine", "tablet", "tablets", "glimepiride", "diabetes medicine", "sugar tablet"]
STOP_WORDS = ["stop", "discontinue", "withdraw", "quit", "taper off", "give up", "leave the", "avoid"]


@dataclass
class Alert:
    rule: str
    severity: str  # "block" or "warn"
    message: str
    verse_ids: list

    @property
    def icon(self) -> str:
        return "BLOCKED" if self.severity == "block" else "CAUTION"


def _mentions_any(text: str, terms: list[str]) -> list[str]:
    """Terms found in `text`, without spelling duplicates or terms contained in a longer hit."""
    hits = [t for t in terms if contains_phrase(text, t)]
    out: list[str] = []
    for t in sorted(hits, key=len, reverse=True):
        ft = folded(t)
        if not any(ft == folded(o) or ft in folded(o) for o in out):
            out.append(t)
    return out


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"[.;\n]+", text) if s.strip()]


def _stops_modern_medicine(text: str) -> bool:
    low = strip_diacritics(text).lower()
    for stop in STOP_WORDS:
        for m in re.finditer(re.escape(stop), low):
            window = low[m.start(): m.start() + 60]
            if any(med in window for med in MODERN_MEDS) and not negated_before(low, m.start()):
                return True
    return False


def check_plan(case: Case, plan_text: str, revealed: set[str]) -> list[Alert]:
    """Check a student's treatment plan (Chikitsa Sutra + plan fields) for this case."""
    alerts: list[Alert] = []
    plan = plan_text or ""
    if not plan.strip():
        return alerts

    # 1. Possible pregnancy must be ruled out before Shodhana / Lekhana / Tikshna drugs.
    if case.flags.get("pregnancy_possible"):
        strong = _mentions_any(plan, SHODHANA_LEKHANA)
        asked_lmp = "lmp" in revealed
        rules_out = any(contains_phrase(plan, p) for p in ["pregnancy test", "upt", "rule out pregnancy", "exclude pregnancy", "beta hcg", "if not pregnant"])
        if strong and not asked_lmp:
            alerts.append(Alert("garbhini_not_screened", "block",
                                f"You planned {', '.join(strong)} without asking about menses. A missed period has not been "
                                "ruled out: Garbhini must be excluded before Shodhana, Lekhana or Tikshna drugs.",
                                ["garbhini_caution"]))
        elif strong and asked_lmp and not rules_out:
            alerts.append(Alert("garbhini_suspected", "block",
                                "Her period is 7 weeks late. Confirm with a pregnancy test before any "
                                f"{', '.join(strong)}; if pregnant, avoid them altogether.", ["garbhini_caution"]))

    # 2. Sama avastha: no Snehana / Brimhana before Ama is digested.
    if case.flags.get("sama"):
        for s in _sentences(plan):
            if any(contains_phrase(s, a) for a in ["apathya", "avoid"]):
                continue  # listing things to avoid is not prescribing them
            hits = [t for t in SNEHANA if contains_phrase(s, t)]
            if hits and not any(contains_phrase(s, c) for c in NIRAMA_CUES):
                alerts.append(Alert("snehana_in_sama", "block",
                                    f"'{hits[0]}' in sama avastha increases Ama. Langhana, Ruksha Swedana and "
                                    "Deepana-Pachana come first; Snehana only after nirama lakshanas appear.",
                                    ["cd25_chikitsa"]))
                break
        if _mentions_any(plan, ["ice", "cold compress", "cold pack", "sheeta"]):
            alerts.append(Alert("sheeta_in_amavata", "warn",
                                "Cold applications aggravate Amavata; the classical upashaya is ushna and ruksha.",
                                ["mn25_nidana"]))

    # 3. Never stop prescribed modern medicines.
    if case.flags.get("on_modern_medication") and _stops_modern_medicine(plan):
        verse = "redflag_asthma" if case.flags["on_modern_medication"] == "inhaler" else "redflag_diabetes"
        alerts.append(Alert("stops_modern_medicine", "block",
                            f"Do not stop the patient's {case.flags['on_modern_medication']}. Ayurvedic treatment is "
                            "added alongside, with monitoring; changes are made only with the prescribing doctor.",
                            [verse]))

    # 4. Red-flag cases: once alarm features are known, the plan must refer.
    if case.flags.get("red_flag_case"):
        alarm_known = {"melena", "weight_loss"} & revealed
        refers = bool(_mentions_any(plan, REFERRAL))
        if not alarm_known:
            alerts.append(Alert("alarm_not_screened", "block",
                                "Alarm features were not screened. At 46 with new dyspepsia, ask about black stools, "
                                "vomiting blood, difficulty swallowing and weight loss before treating.",
                                ["redflag_gi"]))
        elif not refers:
            alerts.append(Alert("alarm_not_referred", "block",
                                "Black stools and weight loss are alarm features. Refer for upper GI endoscopy and a "
                                "blood count before treating this as simple Amlapitta.", ["redflag_gi"]))

    # 5. Visha / Upavisha dravyas need shodhana and an exact dose.
    visha = _mentions_any(plan, VISHA)
    if visha:
        has_dose = re.search(r"\d+\s*(mg|g|gm|tab|tablet|ratti|ml)", plan.lower()) is not None
        shodhita = contains_phrase(plan, "shodhita") or contains_phrase(plan, "purified")
        if not (has_dose and shodhita):
            alerts.append(Alert("visha_dravya", "warn",
                                f"{', '.join(visha)} contains a Visha or Upavisha dravya. State that it is Shodhita, "
                                "give the exact dose and duration, and supervise its use.", ["visha_dravya"]))

    # 6. Crash dieting in Sthaulya.
    if case.disease == "Sthaulya" and _mentions_any(plan, ["crash diet", "starv", "only water", "skip meals"]):
        alerts.append(Alert("crash_diet", "warn",
                            "Starvation is not the classical approach: the obese need food that is guru yet apatarpana.",
                            ["ch_su21_chikitsa"]))
    return alerts
