"""Core behaviour tests. Run with:  pytest -q"""
import pytest

from vaidya.agents import ParikshaAgent, PramanaAgent, RogiAgent, support_score
from vaidya.cases import load_cases
from vaidya.llm import Engine, FakeEngine
from vaidya.retrieval import PramanaLibrary
from vaidya.rules import check_plan
from vaidya.text import contains_phrase

CASES = load_cases()
LIB = PramanaLibrary()
NO_MODEL = Engine("none").start()


# ---------- text matching

@pytest.mark.parametrize("text,phrase", [
    ("Vāta-Kapha with āma", "vata"), ("shleshma pradhana", "sleshma"), ("Langana and swedana", "langhana"),
    ("walking 45 minutes", "walk"), ("rule out pregnancy first", "pregnan"),
])
def test_spelling_variants_match(text, phrase):
    assert contains_phrase(text, phrase)


def test_unrelated_words_do_not_match():
    assert not contains_phrase("pitta pradhana", "vata")


# ---------- patient (Rogi)

def test_hidden_fact_needs_a_specific_question():
    rogi, case = RogiAgent(NO_MODEL), CASES["case03"]
    assert "lmp" not in rogi.reply(case, "What brings you here?", set()).new_facts
    assert "lmp" in rogi.reply(case, "When was your last period?", set()).new_facts


def test_unknown_topic_is_denied_not_invented():
    r = RogiAgent(NO_MODEL).reply(CASES["case01"], "Have you travelled abroad recently?", set())
    assert r.new_facts == [] and "not noticed" in r.text


def test_patient_reply_with_diagnosis_is_discarded():
    leaky = FakeEngine("I think I have Amavata because of my kapha.")
    r = RogiAgent(leaky).reply(CASES["case01"], "How is your appetite?", set())
    assert r.source == "case file" and "amavata" not in r.text.lower()


def test_patient_reply_with_invented_number_is_discarded():
    r = RogiAgent(FakeEngine("I have had this for 9 years, doctor.")).reply(CASES["case01"], "Since when?", set())
    assert r.source == "case file"


def test_faithful_model_reply_is_used():
    r = RogiAgent(FakeEngine("No hunger at all, doctor, and food has no taste.")).reply(
        CASES["case01"], "How is your appetite?", set())
    assert r.source == "AyurParam"


# ---------- retrieval and abstention

@pytest.mark.parametrize("q,expected", [
    ("What is the chikitsa sutra of Amavata?", "cd25_chikitsa"),
    ("What causes prameha?", "ch_chi6_nidana"),
    ("Signs of amlapitta", "mn51_rupa"),
    ("Is virechana safe in pregnancy?", "garbhini_caution"),
])
def test_retrieval_finds_the_verse(q, expected):
    assert expected in [v.id for v in LIB.search(q)]


def test_request_words_do_not_cause_abstention():
    assert "cd25_chikitsa" in [v.id for v in LIB.search("What is the chikitsa sutra of Amavata? Cite the classical source.")]


def test_retrieval_abstains_outside_library():
    assert LIB.search("treatment of Ekakushtha") == []


def test_unsupported_model_answer_is_withheld():
    agent = PramanaAgent(FakeEngine("Drink coconut water and pray to the moon every evening."), LIB)
    r = agent.ask("What is the chikitsa sutra of Amavata?")
    assert r.status == "verses-only" and r.answer is None


def test_support_score():
    assert support_score("langhana and swedana", "langhana swedana tikta") == 1.0
    assert support_score("coconut water", "langhana swedana") == 0.0


# ---------- safety rules

def _rules(case_id, plan, revealed=()):
    return [a.rule for a in check_plan(CASES[case_id], plan, set(revealed))]


def test_garbhini_not_screened_is_blocked():
    assert "garbhini_not_screened" in _rules("case03", "Lekhana basti and virechana")


def test_garbhini_with_pregnancy_test_passes():
    assert _rules("case03", "Urine pregnancy test first. If not pregnant, udvartana.", ["lmp"]) == []


def test_snehana_in_sama_is_blocked_but_apathya_list_is_not():
    assert "snehana_in_sama" in _rules("case01", "Abhyanga with taila daily")
    assert "snehana_in_sama" not in _rules("case01", "Langhana. Apathya: ksheera, dadhi")
    assert "snehana_in_sama" not in _rules("case01", "Snehapana later, after nirama lakshana")


def test_stopping_metformin_is_blocked_but_continuing_is_fine():
    assert "stops_modern_medicine" in _rules("case02", "Stop metformin and give Nisha-Amalaki")
    assert "stops_modern_medicine" not in _rules("case02", "Do not stop metformin; add Nisha-Amalaki")


def test_inhaler_cannot_be_stopped():
    assert "stops_modern_medicine" in _rules("case05", "Discontinue the inhaler, give Sitopaladi")


def test_red_flags_must_be_screened_and_referred():
    assert "alarm_not_screened" in _rules("case04", "Avipattikara churna")
    assert "alarm_not_referred" in _rules("case04", "Avipattikara churna", ["melena"])
    assert _rules("case04", "Refer for endoscopy first", ["melena"]) == []


def test_visha_dravya_needs_dose_and_shodhana():
    assert "visha_dravya" in _rules("case05", "Shvasa Kuthara Rasa")
    assert "visha_dravya" not in _rules("case05", "Shodhita Shvasa Kuthara Rasa 125 mg twice daily for 7 days")


# ---------- grading (Pariksha)

def test_grading_rewards_correct_and_flags_wrong_dosha():
    case = CASES["case01"]
    agent = ParikshaAgent(NO_MODEL, LIB)
    good = agent.grade(case, {"dosha": "Vata-Kapha with ama", "vyadhi": "Amavata"}, set(), set())
    bad = agent.grade(case, {"dosha": "Pitta", "vyadhi": "Sandhivata"}, set(), set())
    gd = {r.field: r for r in good.fields}
    bd = {r.field: r for r in bad.fields}
    assert gd["dosha"].score == 1 and gd["vyadhi"].score == 1
    assert bd["dosha"].score == 0 and "pitta" in bd["dosha"].wrong


def test_all_cases_and_verse_links_are_valid():
    for case in CASES.values():
        assert case.opening and case.key_facts
        for ref in case.reference.values():
            for vid in ref.get("verses", []):
                assert LIB.get(vid), f"{case.id}: missing verse {vid}"


# ---------- plain vs grounded comparison

def test_comparison_flags_unverifiable_citations():
    from vaidya.agents import compare
    eng = FakeEngine("According to Charaka Chikitsa Sthana chapter 29, give milk and ghee.")
    c = compare(eng, PramanaAgent(eng, LIB), "What is the chikitsa sutra of Amavata?")
    assert c.plain and c.plain_cites_texts
    assert c.grounded.verses and c.grounded.verses[0].id == "cd25_chikitsa"
    assert c.grounded.answer is None  # the leaky answer is not supported by the Amavata verses


def test_words_written_with_or_without_spaces_match():
    assert contains_phrase("udaka meha", "udakameha")
    assert contains_phrase("bahumutra and ants", "bahu mutra")


def test_fatigue_is_not_a_weight_question():
    from vaidya.cases import question_topics
    assert "weight" not in question_topics("tiredness or fatigue?")


def test_hindi_and_kannada_romanised_questions_are_understood():
    from vaidya.cases import question_topics
    assert "appetite" in question_topics("kya aapko pyas zyada lagti hai?")
    assert "appetite" in question_topics("ninage hasivu hege ide?")


def test_dr_prasanna_prameha_answers_are_recognised():
    agent = ParikshaAgent(NO_MODEL, LIB)
    a = agent.grade(CASES["case02"], {"purvarupa": "burning feet, coated tongue",
                                      "rupa": "bahumutra, ants coming to urine",
                                      "vyavachedaka": "udaka meha"}, set(), set())
    f = {r.field: r for r in a.fields}
    assert f["purvarupa"].score > 0 and f["rupa"].score > 0 and f["vyavachedaka"].score == 1
