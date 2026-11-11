"""A question the judge did not usably answer: marked as such, sent to review when it gates a
write, and scored by nothing."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any

import pytest

from agent_memory import endpoints
from agent_memory.judge import pipeline, report
from agent_memory.judge.backends import JudgeInput
from agent_memory.judge.backends.base import answer_from_probabilities
from agent_memory.judge.backends.jev import parse_answers
from agent_memory.judge.backends.llm import answers_from_judge_json, ollama_judge
from agent_memory.judge.backends.readout import ReadoutBackend
from agent_memory.judge.backends.rules import RulesBackend
from agent_memory.judge.questions import (
    BY_ID,
    DEFAULT_THRESHOLDS,
    WRITE_GATES,
    Question,
    unanswered,
)
from agent_memory.judge.verdict import Derived, Verdict, derive_action, judge
from tests.test_judge import structured_draft, synthetic_run

# Every probability clears these, so only an unanswered gate can hold a write back.
NO_BAR = {**dict.fromkeys(WRITE_GATES, 0.0), "band": 0.0}
OMITTED = object()
ITEM = {
    "item_id": "i",
    "facts": {},
    "flags": [],
    "stratum": "T0",
    "projection_hash": "hash",
    "turn_event_id": "turn",
}
STATE = "[1] The server was running.\n[2] The database was locked."
CONFIDENT = {
    "failed": "no",
    "incomplete": "no",
    "category": "none",
    "generalizable": "1",
    "enforced": "none",
}


def reply_with(monkeypatch, spoiled: dict[str, Any]) -> list[str]:
    """The model answers every question confidently and in a draft's favour, except the
    `spoiled` ones: left out of the reply, or sent as given."""
    prompts: list[str] = []

    def post(url, body, **kwargs):
        prompt = body["messages"][-1]["content"]
        prompts.append(prompt)
        data = {}
        for qid in re.findall(r'- id "([^"]+)"', prompt):
            if qid not in spoiled:
                data[qid] = {"answer": CONFIDENT.get(qid, "yes"), "probability": 0.95}
            elif spoiled[qid] is not OMITTED:
                data[qid] = spoiled[qid]
        return {"model": "synthetic", "message": {"content": json.dumps(data)}}, 20.0

    monkeypatch.setattr(endpoints, "post_json", post)
    return prompts


def two_claims() -> dict[str, Any]:
    return structured_draft(
        claims=[
            {"text": "The server was running.", "cited": [1]},
            {"text": "The database was locked.", "cited": [2]},
        ]
    ).to_dict()


def judged(monkeypatch, spoiled, thresholds=None, draft=None) -> Mapping[str, Any]:
    """The stored row of the verdict the model's reply leads to."""
    reply_with(monkeypatch, spoiled)
    return judge(
        ollama_judge("synthetic", think=False),
        ITEM,
        STATE,
        draft or two_claims(),
        dict(thresholds or DEFAULT_THRESHOLDS),
        run_id="synthetic",
    ).row


def placeholder(qid: str) -> dict[str, Any]:
    """What a backend stores for a question its judge did not answer."""
    return answer_from_probabilities(BY_ID[qid], {}, method="llm_json_missing").to_dict()


# The judge's reply


@pytest.mark.parametrize(
    ("qid", "entry", "method"),
    [
        ("generalizable", OMITTED, "llm_json_missing"),
        ("generalizable", "1", "llm_json_missing"),
        ("generalizable", {"answer": "high", "probability": 0.9}, "llm_json_unmatched"),
        ("generalizable", {"probability": 0.9}, "llm_json_unmatched"),
        ("enforced", {"answer": None, "probability": 0.9}, "llm_json_unmatched"),
        ("enforced", {"answer": "yes", "probability": 0.9}, "llm_json_unmatched"),
        ("category", {"answer": None, "probability": 0.9}, "llm_json_unmatched"),
        ("grounded", {"answer": ["yes"], "probability": 0.9}, "llm_json_unmatched"),
        ("grounded", {"answer": "yes"}, "llm_json_no_probability"),
        ("grounded", {"answer": "yes", "probability": None}, "llm_json_no_probability"),
        ("grounded", {"answer": "yes", "probability": "high"}, "llm_json_no_probability"),
        ("grounded", {"answer": "yes", "probability": float("nan")}, "llm_json_no_probability"),
        ("grounded", {"answer": "yes", "probability": True}, "llm_json_no_probability"),
    ],
)
def test_a_reply_without_a_usable_answer_is_marked_not_guessed(qid, entry, method):
    data = {} if entry is OMITTED else {qid: entry}
    answer = answers_from_judge_json(data, [BY_ID[qid]])[qid]
    assert answer.method == method
    assert unanswered(answer.method)
    assert len(set(answer.probabilities.values())) == 1  # A placeholder, not a reading.


def test_a_usable_reply_is_read_as_before():
    data = {
        "grounded": {"answer": " Yes ", "probability": "0.8"},
        "generalizable": {"answer": 2, "probability": 0.7},
        "enforced": {"answer": "no", "probability": 1.4},
    }
    answers = answers_from_judge_json(data, [BY_ID[qid] for qid in data], method="llm_json_nothink")
    assert {a.method for a in answers.values()} == {"llm_json_nothink"}
    assert not any(unanswered(a.method) for a in answers.values())
    assert answers["grounded"].p_true == pytest.approx(0.8)
    assert answers["generalizable"].probabilities["2"] == pytest.approx(0.7)
    assert answers["enforced"].probabilities["none"] == 1.0  # "no" abbreviates "none".


# The verdict


def test_a_complete_reply_still_writes(monkeypatch):
    for thresholds in (DEFAULT_THRESHOLDS, NO_BAR):
        verdict = judged(monkeypatch, {}, thresholds)
        assert verdict["action"] == "write_lesson"
        assert not any(unanswered(a["method"]) for a in verdict["answers"].values())


def test_a_reply_that_omits_generalizable_no_longer_writes(monkeypatch):
    """The reported defect: the omitted answer was read as a uniform distribution, whose mass
    on levels 1 and 2 (0.67) cleared the default threshold and wrote the lesson."""
    verdict = judged(monkeypatch, {"generalizable": OMITTED})
    assert verdict["action"] == "review"
    assert verdict["reasons"] == ["generalizable: unanswered"]


@pytest.mark.parametrize(
    ("qid", "gate", "entry"),
    [
        ("generalizable", "generalizable", OMITTED),
        ("generalizable", "generalizable", {"answer": "high", "probability": 0.95}),
        ("generalizable", "generalizable", {"answer": "1"}),
        ("enforced", "enforced", OMITTED),
        ("enforced", "enforced", {"answer": None, "probability": 0.95}),
        ("enforced", "enforced", {"answer": "none", "probability": "certain"}),
        ("rule_supported", "rule_supported", OMITTED),
        ("rule_supported", "rule_supported", {"answer": "maybe", "probability": 0.95}),
        ("rule_supported", "rule_supported", {"answer": "yes"}),
        ("grounded.2", "grounded", OMITTED),
        ("grounded.1", "grounded", {"answer": "yes"}),
        ("faithful.1", "faithful", OMITTED),
        ("faithful.2", "faithful", {"answer": "probably", "probability": 0.95}),
    ],
)
def test_a_write_gate_the_judge_left_unanswered_goes_to_review(monkeypatch, qid, gate, entry):
    verdict = judged(monkeypatch, {qid: entry}, NO_BAR)
    assert verdict["action"] == "review"
    assert verdict["reasons"] == [f"{gate}: unanswered"]
    assert unanswered(verdict["answers"][gate]["method"])
    event = Verdict(verdict).decision_record(ITEM, "2026-01-01T00:00:00+00:00")
    assert event.payload is not None
    assert event.payload["judge"]["action"] == "review"
    assert unanswered(event.payload["judge"]["answers"][gate]["method"])
    assert Verdict(verdict).p(gate) is None


@pytest.mark.parametrize("gate", ["grounded", "faithful"])
def test_a_whole_draft_gate_left_unanswered_goes_to_review(monkeypatch, gate):
    draft = {"item_id": "i", "lesson": "Stop the server first.", "drafter": "ollama:m"}
    verdict = judged(monkeypatch, {gate: OMITTED}, NO_BAR, draft)
    assert verdict["claims"] == 0
    assert (verdict["action"], verdict["reasons"]) == ("review", [f"{gate}: unanswered"])


@pytest.mark.parametrize("qid", ["actionable", "failed", "category", "recurrence"])
def test_an_unanswered_question_that_gates_nothing_does_not_hold_a_write(monkeypatch, qid):
    verdict = judged(monkeypatch, {qid: OMITTED})
    assert unanswered(verdict["answers"][qid]["method"])
    assert verdict["action"] == "write_lesson"


def test_an_unanswered_gate_does_not_rescue_a_declined_or_undrafted_turn(monkeypatch):
    declined = {"item_id": "i", "lesson": None, "declined": True, "drafter": "ollama:m"}
    verdict = judged(monkeypatch, {"failed": OMITTED}, NO_BAR, declined)
    assert verdict["action"] == "discard"
    reply_with(monkeypatch, {})
    backend = ollama_judge("synthetic", think=False)
    undrafted = judge(backend, ITEM, STATE, None, dict(NO_BAR), run_id="synthetic").row
    assert undrafted["action"] == "discard"
    assert undrafted["reasons"] == [f"{gate}: no answer" for gate in WRITE_GATES]


def test_decide_records_the_unanswered_verdict_once_and_does_not_ask_again(tmp_path, monkeypatch):
    store = synthetic_run(tmp_path, n=2)
    for name in ("rules", "llm:ref"):
        store.verdicts_path(name).unlink()
    prompts = reply_with(monkeypatch, {"generalizable": OMITTED})
    logs: list[str] = []
    summary = pipeline.decide_items(
        store.cfg, "syn", ["llm:synthetic"], store=store, record_events=False, log=logs.append
    )
    assert summary.written == {"llm:synthetic": 2}
    assert len(prompts) == 4  # Turn and draft questions once per item: no second attempt.
    rows = store.read_jsonl(store.verdicts_path("llm:synthetic"))
    assert [(r["action"], r["reasons"]) for r in rows] == [
        ("review", ["generalizable: unanswered"])
    ] * 2
    assert all(r["answers"]["generalizable"]["method"] == "llm_json_missing" for r in rows)
    assert sum(": review " in line for line in logs) == 2
    again = pipeline.decide_items(
        store.cfg, "syn", ["llm:synthetic"], store=store, record_events=False
    )
    assert again.written == {"llm:synthetic": 0}
    assert len(prompts) == 4


def test_unanswered_is_review_and_never_asked_is_discard():
    cleared = dict.fromkeys(WRITE_GATES, 0.99)
    assert derive_action({**cleared, "enforced": None}, NO_BAR) == Derived(
        "review", ["enforced: unanswered"]
    )
    # Review outranks a gate that plainly fails, as it does for a gate inside the band.
    failing = derive_action({**cleared, "enforced": None, "grounded": 0.1})
    assert (failing.action, failing.reasons) == (
        "review",
        ["grounded=0.10 below 0.50", "enforced: unanswered"],
    )
    never_asked = {k: v for k, v in cleared.items() if k != "enforced"}
    assert derive_action(never_asked, NO_BAR) == Derived("discard", ["enforced: no answer"])
    assert derive_action({**cleared, "enforced": None}, declined=True).action == "discard"


# Every backend


def test_a_distribution_with_no_mass_on_any_option_is_marked_unanswered():
    question = BY_ID["enforced"]
    for probabilities in ({}, {"elsewhere": 1.0}, dict.fromkeys(question.options(), 0.0)):
        answer = answer_from_probabilities(question, probabilities, method="jev")
        assert answer.method == "jev_missing"
        assert set(answer.probabilities.values()) == {0.25}
    kept = answer_from_probabilities(question, {}, method="llm_json_unmatched")
    assert kept.method == "llm_json_unmatched"  # Already says why; not marked twice.
    real = answer_from_probabilities(question, {"none": 0.2}, method="jev")
    assert (real.method, real.probabilities["none"]) == ("jev", 1.0)


def test_readout_samples_that_name_no_option_are_no_answer(monkeypatch):
    def post(url, body, **kwargs):
        return {"model": "m", "message": {"content": "I cannot say."}}, 5.0

    monkeypatch.setattr(endpoints, "post_json", post)
    inp = JudgeInput(item_id="i", state="S", facts={}, flags=())
    answer = ReadoutBackend("m", samples=4).decide(inp, [BY_ID["generalizable"]])["generalizable"]
    assert answer.method == "sampling:4_missing"
    assert unanswered(answer.method)
    assert answer.latency_ms == pytest.approx(20.0)


def test_jev_entry_without_a_distribution_is_no_answer():
    response = {
        "model": "typesafe/jev",
        "answers": {
            "generalizable": {"type": "score"},
            "enforced": {"type": "choice", "probabilities": {"elsewhere": 1.0}},
            "failed": {"type": "noul"},
            "grounded": {"type": "noul", "noul": None},
        },
    }
    questions = [BY_ID[name] for name in response["answers"]]
    answers = parse_answers(response, questions, latency_ms=1.0)
    assert [answers[q.id].method for q in questions] == ["jev_missing"] * 4


def test_rules_have_no_answer_for_a_question_they_do_not_know():
    novel = Question("novel", "noul", "A question the rules were never written for.", "B")
    inp = JudgeInput(item_id="i", state="", facts={}, flags=())
    assert RulesBackend().decide(inp, [novel])["novel"].method == "rules_missing"


# The report


def draft_eval(rows):
    return report.evaluate_drafts("rules", "ollama:m", rows, DEFAULT_THRESHOLDS, WRITE_GATES)


def gate_distribution(gate: str, p: float) -> dict[str, float]:
    if gate == "generalizable":
        return {"0": 1 - p, "1": p, "2": 0.0}
    if gate == "enforced":
        return {"none": p, "harness": 1 - p, "project_docs": 0.0, "tooling": 0.0}
    return {"yes": p, "no": 1 - p}


def set_truth(gate: str, label: dict[str, Any], truth: bool) -> None:
    draft = label["drafts"]["ollama:m"]
    if gate == "failed":
        label["failed"] = truth
        label["category"] = "tool_misuse" if truth else None
    elif gate == "generalizable":
        draft["generalizable"] = 1 if truth else 0
    elif gate == "enforced":
        draft["enforced_by"] = "none" if truth else "harness"
    else:
        draft[gate] = truth


@pytest.mark.parametrize("gate", ["failed", *WRITE_GATES])
def test_unanswered_answers_are_not_used_to_fit_thresholds(tmp_path, gate):
    """Two answered rows put the best cut at 0.65. The other ten are placeholders labelled
    true: counted, their uniform probability would drag the cut below both answers."""
    store = synthetic_run(tmp_path)
    labels = store.read_jsonl(store.labels_path)
    verdicts = store.read_jsonl(store.verdicts_path("rules"))
    for i, (label, verdict) in enumerate(zip(labels, verdicts, strict=True)):
        set_truth(gate, label, i != 0)
        if i < 2:
            verdict["answers"][gate]["probabilities"] = gate_distribution(gate, 0.6 + 0.1 * i)
        else:
            verdict["answers"][gate] = placeholder(gate)
    store.write_jsonl(store.labels_path, labels)
    store.write_jsonl(store.verdicts_path("rules"), verdicts)
    report.build_report(store, freeze=True)
    assert store.read_run()["frozen"]["backends"]["rules"]["thresholds"][gate] == 0.65


def test_an_unanswered_verdict_is_counted_as_a_review_and_scores_nothing(tmp_path):
    store = synthetic_run(tmp_path)
    assert "unanswered" not in report.build_report(store)
    path = store.verdicts_path("rules")
    verdicts = store.read_jsonl(path)
    silent = verdicts[1]  # A turn whose draft the rules verdict would otherwise write.
    silent["answers"] = {qid: placeholder(qid) for qid in silent["answers"]}
    store.write_jsonl(path, verdicts)
    _, scored = report.load_scored(store)
    rows = [r for r in scored["rules"].values() if r.item["split"] == "calibration"]
    others = [r for r in rows if r.item["item_id"] != silent["item_id"]]
    assert len(rows) == len(others) + 1 == 12

    with_it, without = draft_eval(rows), draft_eval(others)
    for name in ("grounded", "faithful", "rule_supported"):
        assert getattr(with_it, name) == getattr(without, name), name
    assert with_it.generalizable_accuracy == without.generalizable_accuracy
    assert with_it.enforced_accuracy == without.enforced_accuracy
    assert with_it.actions == {**without.actions, "review": without.actions["review"] + 1}
    assert (with_it.unanswered, without.unanswered) == (1, 0)

    def turn_eval(some):
        return report.evaluate_turns("rules", some, DEFAULT_THRESHOLDS, category_on_failed=True)

    for name in ("failed", "incomplete", "category_accuracy", "paired"):
        assert getattr(turn_eval(rows), name) == getattr(turn_eval(others), name), name

    notes = [line for line in report.build_report(store).splitlines() if "unanswered" in line]
    assert len(notes) == 1
    assert "`rules` 1 of 12" in notes[0]
    assert "llm:ref" not in notes[0]


def test_a_declined_draft_is_not_counted_as_held_by_an_unanswered_gate(tmp_path):
    store = synthetic_run(tmp_path)
    path = store.verdicts_path("rules")
    verdicts = store.read_jsonl(path)
    verdicts[1]["declined"] = True
    verdicts[1]["answers"]["enforced"] = placeholder("enforced")
    store.write_jsonl(path, verdicts)
    _, scored = report.load_scored(store)
    rows = [r for r in scored["rules"].values() if r.item["split"] == "calibration"]
    assert draft_eval(rows).unanswered == 0
    assert "unanswered" not in report.build_report(store)
