"""Evaluation for the technical-clinical report.

    python -m eval.run_eval

Measures, on the team's own test set:
  - retrieval hit rate @3 and precision of the first verse
  - correct abstention on questions outside the library
  - unsafe plans caught, and false alarms on safe plans
  - patient-reply fidelity filter on deliberately leaky replies
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from vaidya.agents import RogiAgent
from vaidya.cases import load_cases
from vaidya.llm import FakeEngine
from vaidya.retrieval import PramanaLibrary
from vaidya.rules import check_plan

HERE = Path(__file__).parent


def main() -> dict:
    data = json.loads((HERE / "questions.json").read_text(encoding="utf-8"))
    lib, cases = PramanaLibrary(), load_cases()

    hits = top1 = 0
    misses = []
    t0 = time.perf_counter()
    for item in data["retrieval"]:
        got = [v.id for v in lib.search(item["q"])]
        ok = any(e in got for e in item["expected"])
        hits += ok
        top1 += bool(got) and got[0] in item["expected"]
        if not ok:
            misses.append({"q": item["q"], "expected": item["expected"], "got": got})
    latency_ms = 1000 * (time.perf_counter() - t0) / len(data["retrieval"])

    declined = [q for q in data["out_of_library"] if not lib.search(q)]
    caught = [p for p in data["unsafe_plans"]
              if p["must"] in [a.rule for a in check_plan(cases[p["case"]], p["plan"], set(p["revealed"]))]]
    false_alarms = [p for p in data["safe_plans"]
                    if any(a.severity == "block" for a in check_plan(cases[p["case"]], p["plan"], set(p["revealed"])))]

    leaky = ["I have Amavata, doctor.", "My kapha dosha is high.", "This started 11 years ago.",
             "As an AI language model I cannot say.", "My vata is aggravated in the morning."]
    rejected = sum(RogiAgent(FakeEngine(t)).reply(cases["case01"], "How is your appetite?", set()).source == "case file"
                   for t in leaky)

    n = len(data["retrieval"])
    report = {
        "retrieval_questions": n,
        "retrieval_hit_rate_at_3": round(hits / n, 3),
        "retrieval_top1_precision": round(top1 / n, 3),
        "retrieval_latency_ms": round(latency_ms, 2),
        "out_of_library_questions": len(data["out_of_library"]),
        "correct_abstention_rate": round(len(declined) / len(data["out_of_library"]), 3),
        "unsafe_plans": len(data["unsafe_plans"]),
        "unsafe_plans_caught": round(len(caught) / len(data["unsafe_plans"]), 3),
        "safe_plans": len(data["safe_plans"]),
        "false_blocks_on_safe_plans": len(false_alarms),
        "leaky_patient_replies": len(leaky),
        "leaky_replies_rejected": round(rejected / len(leaky), 3),
        "retrieval_misses": misses,
        "not_declined": [q for q in data["out_of_library"] if q not in declined],
        "unsafe_not_caught": [p for p in data["unsafe_plans"] if p not in caught],
        "false_alarm_plans": false_alarms,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    (HERE / "results.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


if __name__ == "__main__":
    main()
