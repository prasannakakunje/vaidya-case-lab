"""vAIdya Case Lab - a clinical reasoning simulator for BAMS students, built on AyurParam.

Run locally:  python app.py
Environment:  VAIDYA_BACKEND = auto | llamacpp | transformers | none
"""
from __future__ import annotations

import logging
import os

import gradio as gr

from vaidya.agents import ParikshaAgent, PramanaAgent, RogiAgent, compare
from vaidya.cases import EXAMINATIONS, REASONING_FIELDS, load_cases
from vaidya.llm import Engine
from vaidya.retrieval import PramanaLibrary

logging.basicConfig(level=logging.INFO)

CASES = load_cases()
LIBRARY = PramanaLibrary()
ENGINE = Engine().start()
ROGI = RogiAgent(ENGINE)
PARIKSHA = ParikshaAgent(ENGINE, LIBRARY)
PRAMANA = PramanaAgent(ENGINE, LIBRARY)

CASE_CHOICES = [(f"Case {i + 1} · {c.title} ({c.difficulty})", cid) for i, (cid, c) in enumerate(CASES.items())]
FIELD_LABELS = dict(REASONING_FIELDS)
EXAM_LABELS = dict(EXAMINATIONS)

DISCLAIMER = ("For education only. All patients are simulated; do not use this tool to diagnose or treat real "
              "patients. Cases and verses reviewed by Dr. Prasanna K.G., 26 Nov 2026.")


def new_state(case_id: str | None = None) -> dict:
    return {"case": case_id, "revealed": [], "exams": [], "submitted": False}


# ---------------------------------------------------------------- rendering

def patient_card(case) -> str:
    p = case.patient
    return (f"### {p['name']}, {p['age']}, {p['sex'].lower()} · {p['occupation']}, {p['place']}\n"
            f"Difficulty: **{case.difficulty}**. Take a full history, examine, then submit your assessment.")


def history_progress(case, revealed: list[str]) -> str:
    n = sum(1 for f in case.key_facts if f.id in revealed)
    return f"**History taken:** {n} of {len(case.key_facts)} key facts elicited."


def render_exams(case, exams: list[str]) -> str:
    if not exams:
        return "_Choose what to examine, then press Examine._"
    rows = "\n".join(f"| {EXAM_LABELS[e]} | {case.examinations.get(e, 'Nothing remarkable.')} |" for e in exams)
    return "| Examination | Finding |\n| --- | --- |\n" + rows


def render_feedback(case, a) -> str:
    out = []
    if a.blocked:
        out.append(f"## Not passed: unsafe plan\n\nOverall score **{a.total} / 100**, but the plan contains "
                   "a safety error that must be corrected first.")
    else:
        out.append(f"## Score: {a.total} / 100")
    out.append(f"Reasoning {round(100 * a.reasoning_score)}% · History {len(a.history_found)} of "
               f"{len(a.history_found) + len(a.history_missed)} key facts · {a.exams_done} examinations done")

    if a.alerts:
        out.append("### Safety review")
        for al in a.alerts:
            out.append(f"> **{al.icon}.** {al.message}")

    if a.history_missed:
        missed = [case.fact(i) for i in a.history_missed]
        hidden = [f for f in missed if f and f.hidden]
        out.append(f"### History you did not elicit\n{len(missed)} key fact(s) were never asked about."
                   + (f" **{len(hidden)} of them {'changes' if len(hidden) == 1 else 'change'} the management.** Ask about every system before treating."
                      if hidden else ""))

    rows = ["### Your reasoning, field by field", "", "| Field | Score | Covered | Missed |", "| --- | --- | --- | --- |"]
    for r in a.fields:
        cov = ", ".join(r.matched) if r.matched else ("-" if r.answered else "_not answered_")
        miss = ", ".join(r.missed) or "-"
        if r.wrong:
            miss += f" · **incorrect: {', '.join(r.wrong)}**"
        rows.append(f"| {FIELD_LABELS.get(r.field, r.field)} | {round(100 * r.score)}% | {cov} | {miss} |")
    out.append("\n".join(rows))

    rows = ["### Reference reasoning (faculty)", "", "| Field | Reference |", "| --- | --- |"]
    for r in a.fields:
        rows.append(f"| {FIELD_LABELS.get(r.field, r.field)} | {r.reference} |")
    out.append("\n".join(rows))
    out.append(f"\n**Diagnosis in this case:** {case.disease}")

    if a.teacher_note:
        out.append(f"### Teacher's note\n_{a.teacher_note_source}_\n\n{a.teacher_note}")

    verses = [LIBRARY.get(v) for v in a.verse_ids]
    verses = [v for v in verses if v]
    if verses:
        out.append("### Pramana for the corrections\n_Attached by the system from the verse library, not written "
                   "by the model._")
        out += [v.as_markdown(i + 1) for i, v in enumerate(verses)]
    return "\n\n".join(out)


def engine_status() -> str:
    return f"**Language engine:** {ENGINE.status}"


# ---------------------------------------------------------------- events

def start_case(case_id):
    case = CASES[case_id]
    state = new_state(case_id)
    chat = [{"role": "assistant", "content": ROGI.opening(case)}]
    blanks = [""] * len(REASONING_FIELDS)
    return (state, patient_card(case), chat, history_progress(case, []), [], render_exams(case, []),
            "_Submit your assessment to see feedback._", *blanks)


def ask_patient(question, chat, state, use_model):
    if not state.get("case"):
        return chat, "", "_Start a case first._", state
    case = CASES[state["case"]]
    if not question.strip():
        return chat, "", history_progress(case, state["revealed"]), state
    revealed = set(state["revealed"])
    reply = ROGI.reply(case, question, revealed, use_model=use_model)
    state["revealed"] = state["revealed"] + [f for f in reply.new_facts if f not in revealed]
    chat = chat + [{"role": "user", "content": question}, {"role": "assistant", "content": reply.text}]
    return chat, "", history_progress(case, state["revealed"]), state


def examine(selected, state):
    if not state.get("case"):
        return "_Start a case first._", state
    case = CASES[state["case"]]
    state["exams"] = list(dict.fromkeys(state["exams"] + list(selected or [])))
    return render_exams(case, state["exams"]), state


def submit(state, use_model, *answers):
    if not state.get("case"):
        return "_Start a case first._", state
    case = CASES[state["case"]]
    filled = {k: v for (k, _), v in zip(REASONING_FIELDS, answers)}
    if not any(v.strip() for v in filled.values()):
        return "_Fill in at least the diagnosis and the treatment plan before submitting._", state
    a = PARIKSHA.grade(case, filled, set(state["revealed"]), set(state["exams"]), use_model=use_model)
    state["submitted"] = True
    return render_feedback(case, a), state


def ask_pramana(question, use_model):
    if not question.strip():
        return ""
    r = PRAMANA.ask(question, use_model=use_model)
    if r.status == "abstained":
        return ("**No Pramana found in the library for this question.** vAIdya does not answer without a "
                "classical source, so it will not guess.")
    out = []
    if r.answer:
        out.append(f"### Answer\n_AyurParam, using only the verses below_\n\n{r.answer}")
    else:
        out.append("### Relevant verses\n_The verses below match your question._")
    out.append("### Pramana\n_Attached by the system, not written by the model._")
    out += [v.as_markdown(i + 1) for i, v in enumerate(r.verses)]
    return "\n\n".join(out)


def render_grounded(r) -> str:
    if r.status == "abstained":
        return ("**No Pramana found in the library for this question.** vAIdya does not answer without a "
                "classical source, so it will not guess.")
    out = []
    if r.answer:
        out.append(f"{r.answer}")
    else:
        out.append("_Only the verses are shown: the model's answer was not supported by them, or the model is not loaded._")
    out.append("**Pramana** _(attached by the system, not written by the model)_")
    out += [v.as_markdown(i + 1) for i, v in enumerate(r.verses)]
    return "\n\n".join(out)


def run_compare(question):
    if not question.strip():
        return "", ""
    if not ENGINE.ready:
        note = ("_AyurParam is not loaded in this session, so there is no plain answer to compare. "
                f"Status: {ENGINE.status}_")
        return note, render_grounded(PRAMANA.ask(question, use_model=False))
    c = compare(ENGINE, PRAMANA, question)
    plain = c.plain or (f"_AyurParam returned no answer. {('Error: ' + ENGINE.last_error) if ENGINE.last_error else ''}_")
    warn = ("\n\n> **Unverifiable:** this answer names classical texts or chapters, but nothing checks that they are "
            "correct, and no verse is shown." if c.plain_cites_texts else
            "\n\n> **No source:** nothing in this answer can be traced to a verse.")
    if c.plain_looped:
        warn += "\n\n> **Looping:** the model started repeating itself; the repetition is cut off here."
    return plain + warn, render_grounded(c.grounded)


# ---------------------------------------------------------------- layout

THEME = gr.themes.Soft(primary_hue="emerald", secondary_hue="amber", neutral_hue="stone",
                       font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"])
CSS = """
#title h1 {margin-bottom: 0}
.disclaimer {font-size: 0.85rem; opacity: 0.8}
"""

ABOUT = f"""
### What this is
vAIdya Case Lab is a clinical reasoning simulator for BAMS students. A student takes a history from a virtual
patient, examines, works out the Nidana Panchaka, Samprapti and Chikitsa Sutra, and is graded field by field
against a faculty reference, with the classical verse behind every correction.

### How it stays safe
1. **The model is never the source of truth.** Patient facts come from a faculty-written case file; verses come from
   the Pramana library. AyurParam only supplies the language.
2. **Every reply is checked.** A patient reply that names a diagnosis, a dosha, or a number not in the case file is
   discarded and the case-file text is used instead. A library answer whose words are not supported by the
   retrieved verses is withheld.
3. **No claim without Pramana.** If the library has no matching verse, the system says so.
4. **Deterministic safety rules** (no language model involved) block unsafe plans: Garbhini not excluded, Snehana in
   sama avastha, stopping prescribed modern medicine, untreated alarm features, Visha dravyas without dose.

### Team vAIdya
Dr. Prasanna K.G., PhD scholar (Ayurveda), RGUHS; Professor & HoD, Kayachikitsa, Prasanna College of Ayurveda &
Hospital, Belthangady. Built on [AyurParam](https://huggingface.co/bharatgenai/AyurParam) (BharatGen, IIT Bombay).
Open source under the Apache 2.0 licence.

{DISCLAIMER}
"""

with gr.Blocks(title="vAIdya Case Lab", theme=THEME, css=CSS) as demo:
    state = gr.State(new_state())
    gr.Markdown("# vAIdya Case Lab\nA practice patient for BAMS students, with a strict examiner who always shows "
                "the classical source.", elem_id="title")
    gr.Markdown(DISCLAIMER, elem_classes="disclaimer")
    status = gr.Markdown(engine_status())
    gr.Timer(5).tick(engine_status, outputs=status)

    with gr.Row():
        case_dd = gr.Dropdown(CASE_CHOICES, value=CASE_CHOICES[0][1], label="Choose a patient", scale=3)
        start_btn = gr.Button("Start case", variant="primary", scale=1)
        use_model = gr.Checkbox(value=True, label="Use AyurParam for the patient's voice and the teacher's note",
                                scale=2)
    card = gr.Markdown("_Choose a patient and press Start case._")

    with gr.Tabs():
        with gr.Tab("1 · History"):
            chat = gr.Chatbot(type="messages", height=400, label="OPD consultation")
            with gr.Row():
                q = gr.Textbox(placeholder="Ask the patient a question, e.g. 'When is the pain worse?'",
                               show_label=False, scale=5)
                ask_btn = gr.Button("Ask", scale=1)
            progress = gr.Markdown("")
        with gr.Tab("2 · Examination"):
            exams = gr.CheckboxGroup([(label, key) for key, label in EXAMINATIONS], label="Examine")
            exam_btn = gr.Button("Examine")
            findings = gr.Markdown("_Start a case first._")
        with gr.Tab("3 · Your assessment"):
            boxes = []
            with gr.Row():
                with gr.Column():
                    for key, label in REASONING_FIELDS[:7]:
                        boxes.append(gr.Textbox(label=label, lines=2))
                with gr.Column():
                    for key, label in REASONING_FIELDS[7:]:
                        boxes.append(gr.Textbox(label=label, lines=4 if key == "plan" else 2))
            submit_btn = gr.Button("Submit for examination", variant="primary")
        with gr.Tab("4 · Feedback"):
            feedback = gr.Markdown("_Submit your assessment to see feedback._")
        with gr.Tab("Ask the Pramana library"):
            pq = gr.Textbox(label="Ask about a concept, a disease or a treatment principle",
                            placeholder="e.g. What is the chikitsa sutra of Amavata?")
            p_btn = gr.Button("Search the classics")
            p_out = gr.Markdown()
        with gr.Tab("Why Pramana matters"):
            gr.Markdown("Ask the same question two ways: to plain AyurParam on its own, and through vAIdya's "
                        "Pramana pipeline. Compare what each answer rests on.")
            cq = gr.Textbox(label="Question", value="What is the chikitsa sutra of Amavata? Cite the classical source.")
            c_btn = gr.Button("Compare", variant="primary")
            gr.Markdown("_On the free hosted version each answer takes 20 to 40 seconds._")
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Plain AyurParam\n_No retrieval, no checks_")
                    c_plain = gr.Markdown()
                with gr.Column():
                    gr.Markdown("### vAIdya\n_AyurParam + Pramana library + support check_")
                    c_ground = gr.Markdown()
        with gr.Tab("About & safety"):
            gr.Markdown(ABOUT)

    start_btn.click(start_case, case_dd, [state, card, chat, progress, exams, findings, feedback, *boxes])
    ask_btn.click(ask_patient, [q, chat, state, use_model], [chat, q, progress, state])
    q.submit(ask_patient, [q, chat, state, use_model], [chat, q, progress, state])
    exam_btn.click(examine, [exams, state], [findings, state])
    submit_btn.click(submit, [state, use_model, *boxes], [feedback, state])
    p_btn.click(ask_pramana, [pq, use_model], p_out)
    pq.submit(ask_pramana, [pq, use_model], p_out)
    c_btn.click(run_compare, cq, [c_plain, c_ground])

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch(server_name=os.getenv("GRADIO_SERVER_NAME", "0.0.0.0"),
                                                   share=os.getenv("GRADIO_SHARE") == "1")
