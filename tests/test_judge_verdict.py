"""A verdict: built by `judge`, stored as one row, and read back through `Verdict`."""

from __future__ import annotations

import json
from typing import Any

import pytest

from agent_memory.judge.backends.base import answer_from_probabilities
from agent_memory.judge.backends.rules import RulesBackend
from agent_memory.judge.drafter import draft_hash
from agent_memory.judge.questions import (
    ALL_QUESTIONS,
    DEFAULT_THRESHOLDS,
    QUESTIONS_VERSION,
    STAGE_A,
    STAGE_B,
    WRITE_GATES,
    WRITE_GATES_V1,
    WRITE_GATES_V2,
    Question,
)
from agent_memory.judge.store import JudgeError
from agent_memory.judge.verdict import Derived, Verdict, ask_turn, derive_action, judge
from tests.test_judge import structured_draft

# Every probability clears these, so only an unanswered or unasked gate holds a write back.
NO_BAR = {**dict.fromkeys(WRITE_GATES, 0.0), "band": 0.0}
ITEM = {
    "item_id": "i",
    "facts": {},
    "flags": ["T0"],
    "stratum": "T0",
    "projection_hash": "projection-hash-0123456789",
    "turn_event_id": "turn",
}
STATE = "[1] The server was running.\n[2] SENTINEL_STATE_TEXT: the database was locked."
DRAFT = {"item_id": "i", "lesson": "Stop the server before migrating.", "drafter": "ollama:m"}
TURN_IDS = {q.id for q in STAGE_A}
ROW_KEYS = [
    "run_id",
    "item_id",
    "backend",
    "model_snapshot",
    "questions_version",
    "projection_version",
    "projection_hash",
    "drafter",
    "has_draft",
    "claims",
    "declined",
    "draft_hash",
    "stage_a_reused",
    "answers",
    "latency_ms",
    "cost_usd",
    "action",
    "reasons",
    "thresholds",
    "decided_at",
]


def distribution(question: Question, p: float | None) -> dict[str, float]:
    """`p` on the option that clears the question's gate; nothing at all for None."""
    if p is None:
        return {}
    options = question.options()
    if question.type == "score":
        return {"0": 1 - p, "1": p}
    clears = "none" if "none" in options else options[0]
    other = next(o for o in options if o != clears)
    return {clears: p, other: 1 - p}


class Scripted:
    """Answers every question with one probability of clearing, except those in `spoiled`:
    another probability, or None for a question left unanswered. Every answer is paid for."""

    name = "scripted"
    local = True

    def __init__(self, p: float = 0.9, *, fail_on: str | None = None, **spoiled: float | None):
        self.p = p
        self.fail_on = fail_on
        self.spoiled = spoiled
        self.asked: list[tuple[bool, str | None, list[str]]] = []

    def decide(self, inp, questions):
        self.asked.append((questions is STAGE_A, inp.draft, [q.id for q in questions]))
        if self.fail_on is not None and inp.draft == self.fail_on:
            raise JudgeError("synthetic model failure")
        return {
            q.id: answer_from_probabilities(
                q,
                distribution(q, self.spoiled.get(q.id.replace(".", "_"), self.p)),
                method="scripted",
                latency_ms=10.0,
                cost_usd=0.01,
                model_snapshot="scripted-1",
            )
            for q in questions
        }


def stored(probabilities: dict[str, float], *, method: str = "x", value: Any = "yes") -> dict:
    return {
        "value": value,
        "probabilities": probabilities,
        "confidence": max(probabilities.values()),
        "method": method,
    }


def yes(p: float) -> dict[str, Any]:
    return stored({"yes": p, "no": 1 - p})


# Judging


def test_judge_asks_the_turn_then_the_draft_and_returns_one_stored_row():
    backend = Scripted()
    verdict = judge(backend, ITEM, STATE, DRAFT, DEFAULT_THRESHOLDS, run_id="r")
    assert [(turn, draft) for turn, draft, _ in backend.asked] == [
        (True, None),
        (False, DRAFT["lesson"]),
    ]
    row = verdict.row
    assert list(row) == ROW_KEYS
    assert set(row["answers"]) == {q.id for q in ALL_QUESTIONS}
    assert (row["run_id"], row["item_id"], row["backend"]) == ("r", "i", "scripted")
    assert (row["model_snapshot"], row["questions_version"]) == ("scripted-1", QUESTIONS_VERSION)
    assert (row["projection_hash"], row["drafter"]) == (ITEM["projection_hash"], "ollama:m")
    assert (row["has_draft"], row["claims"], row["declined"]) == (True, 0, False)
    assert row["draft_hash"] == draft_hash(DRAFT["lesson"])
    assert row["stage_a_reused"] is False
    assert row["latency_ms"] == pytest.approx(10.0 * len(ALL_QUESTIONS))
    assert row["cost_usd"] == pytest.approx(0.01 * len(ALL_QUESTIONS))
    assert row["thresholds"] == DEFAULT_THRESHOLDS
    assert (row["action"], row["reasons"]) == ("write_lesson", ["all thresholds cleared"])
    assert verdict.recorded == Derived("write_lesson", ["all thresholds cleared"])
    assert Verdict(json.loads(json.dumps(row))) == verdict  # What is stored is the verdict.


def test_a_turn_is_asked_once_and_later_drafts_copy_its_answers_at_no_cost():
    backend = Scripted()
    turn = ask_turn(backend, ITEM, STATE)
    first = judge(backend, ITEM, STATE, DRAFT, {}, run_id="r", turn=turn)
    carried = first.turn_answers()
    assert carried is not None
    other = {"item_id": "i", "lesson": "Another lesson.", "drafter": "llamacpp:x"}
    second = judge(backend, ITEM, STATE, other, {}, run_id="r", turn=carried)
    assert sum(turn for turn, _, _ in backend.asked) == 1
    assert (first.row["stage_a_reused"], second.row["stage_a_reused"]) == (False, True)
    for qid in TURN_IDS:
        paid, copied = first.row["answers"][qid], second.row["answers"][qid]
        assert (paid["latency_ms"], paid["cost_usd"]) == (10.0, 0.01)
        assert copied == {**paid, "latency_ms": 0.0, "cost_usd": 0.0}
    assert first.latency_ms == pytest.approx(10.0 * len(ALL_QUESTIONS))
    assert second.latency_ms == pytest.approx(10.0 * len(STAGE_B))
    for verdict in (first, second):
        assert verdict.draft_latency_ms == pytest.approx(10.0 * len(STAGE_B))
        assert verdict.draft_cost_usd == pytest.approx(0.01 * len(STAGE_B))
    assert second.cost_usd == pytest.approx(second.draft_cost_usd)


def test_a_draft_whose_questions_fail_leaves_the_turn_to_be_paid_by_the_next_row():
    backend = Scripted(fail_on=DRAFT["lesson"])
    turn = ask_turn(backend, ITEM, STATE)
    with pytest.raises(JudgeError, match="synthetic model failure"):
        judge(backend, ITEM, STATE, DRAFT, {}, run_id="r", turn=turn)
    other = {"item_id": "i", "lesson": "Another lesson.", "drafter": "llamacpp:x"}
    verdict = judge(backend, ITEM, STATE, other, {}, run_id="r", turn=turn)
    assert verdict.row["stage_a_reused"] is False
    assert verdict.latency_ms == pytest.approx(10.0 * len(ALL_QUESTIONS))


def test_a_verdict_that_predates_a_turn_question_carries_no_turn_answers():
    row = judge(Scripted(), ITEM, STATE, DRAFT, {}, run_id="r").row
    row["answers"].pop("incomplete")
    assert Verdict(row).turn_answers() is None


def test_a_declined_draft_is_discarded_unasked_and_an_undrafted_turn_has_no_gates():
    backend = Scripted()
    declined_draft = {"item_id": "i", "lesson": None, "declined": True, "drafter": "ollama:m"}
    declined = judge(backend, ITEM, STATE, declined_draft, NO_BAR, run_id="r")
    assert [turn for turn, _, _ in backend.asked] == [True]  # No draft questions.
    assert set(declined.row["answers"]) == TURN_IDS
    assert declined.recorded == Derived("discard", ["drafter declined: nothing worth keeping"])
    assert (declined.declined, declined.row["has_draft"], declined.draft_hash) == (
        True,
        False,
        None,
    )
    assert declined.drafter == "ollama:m"
    assert declined.action_at(NO_BAR).action == "discard"

    undrafted = judge(backend, ITEM, STATE, None, NO_BAR, run_id="r")
    assert set(undrafted.row["answers"]) == TURN_IDS
    assert undrafted.gates() == {}
    assert undrafted.recorded == Derived("discard", [f"{gate}: no answer" for gate in WRITE_GATES])
    assert (undrafted.declined, undrafted.drafter, undrafted.draft_hash) == (False, None, None)


def test_claim_answers_are_kept_and_each_claimed_gate_is_the_weakest_claim():
    draft = structured_draft(
        claims=[
            {"text": "The server was running.", "cited": [1]},
            {"text": "The database was locked.", "cited": [2]},
        ]
    ).to_dict()
    backend = Scripted(grounded_2=0.2, faithful_1=None)
    verdict = judge(backend, ITEM, STATE, draft, NO_BAR, run_id="r")
    answers = verdict.row["answers"]
    asked = backend.asked[-1][2]
    assert asked[:4] == ["grounded.1", "faithful.1", "grounded.2", "faithful.2"]
    assert "grounded" not in asked  # The whole-draft gates are derived, not asked.
    assert "faithful" not in asked
    assert verdict.row["claims"] == 2
    assert [answers[f"grounded.{i}"]["probabilities"]["yes"] for i in (1, 2)] == [0.9, 0.2]
    assert answers["grounded"]["method"] == "min_over_claims"
    assert answers["grounded"]["probabilities"]["yes"] == 0.2
    assert answers["grounded"]["latency_ms"] == answers["grounded"]["cost_usd"] == 0
    # One claim the judge left unanswered leaves its whole gate unanswered.
    assert answers["faithful"]["method"] == "min_over_claims_missing"
    assert answers["faithful"]["latency_ms"] == answers["faithful"]["cost_usd"] == 0
    assert verdict.gates()["faithful"] is None
    assert verdict.recorded == Derived("review", ["faithful: unanswered"])


# Reading a stored row


def test_the_write_rule_reviews_before_it_discards():
    cleared = dict.fromkeys(WRITE_GATES, 0.9)
    assert derive_action(cleared) == Derived("write_lesson", ["all thresholds cleared"])
    assert derive_action({**cleared, "enforced": 0.2}) == Derived(
        "discard", ["enforced=0.20 below 0.50"]
    )
    # One gate inside the band is a review, even beside a gate that plainly fails.
    assert derive_action({**cleared, "grounded": 0.1, "faithful": 0.55}) == Derived(
        "review", ["grounded=0.10 below 0.50", "faithful=0.55 within 0.15 of 0.50"]
    )
    # Even odds on rule support point neither way, whatever the threshold.
    assert derive_action({**cleared, "rule_supported": 0.5}, NO_BAR) == Derived(
        "review", ["rule_supported=0.50: no direction of support"]
    )
    # Each question set keeps its own rule: questions_v1 gated on the turn having failed.
    v1 = {"failed": 0.9, "recurrence": 0.8, "supported": 0.9}
    assert derive_action(v1, gates=WRITE_GATES_V1).action == "write_lesson"
    assert derive_action({**v1, "recurrence": 0.2}, gates=WRITE_GATES_V1).action == "discard"
    assert derive_action({**v1, "failed": 0.55}, gates=WRITE_GATES_V1).action == "review"


def test_the_recorded_action_stays_while_other_thresholds_give_another():
    verdict = judge(Scripted(p=0.6), ITEM, STATE, DRAFT, DEFAULT_THRESHOLDS, run_id="r")
    assert verdict.gates() == dict.fromkeys(WRITE_GATES, 0.6)
    assert verdict.recorded.action == "review"  # Every gate is inside the default band.
    assert verdict.action_at(DEFAULT_THRESHOLDS) == verdict.recorded
    assert verdict.action_at({"band": 0.05}).action == "write_lesson"
    strict = {**dict.fromkeys(WRITE_GATES, 0.9), "band": 0.05}
    assert verdict.action_at(strict).action == "discard"
    assert verdict.recorded.action == verdict.row["action"] == "review"


def test_gates_are_read_by_the_question_set_the_verdict_was_judged_with():
    answers = {
        "failed": yes(0.9),
        "recurrence": yes(0.8),
        "grounded": yes(0.9),
        "faithful": yes(0.8),
        "generalizable": stored({"0": 0.2, "1": 0.5, "2": 0.3}, value=1.1),
        "enforced": stored(
            {"none": 0.7, "harness": 0.1, "project_docs": 0.1, "tooling": 0.1}, value="none"
        ),
    }
    draft_gates = {"grounded": 0.9, "faithful": 0.8, "generalizable": 0.8, "enforced": 0.7}
    v2 = Verdict({"questions_version": "questions_v2", "answers": answers})
    assert v2.gates() == draft_gates
    assert v2.action_at().action == "write_lesson"
    # The same answers under the current set lack its fifth gate, which was never asked.
    v4 = Verdict({"questions_version": QUESTIONS_VERSION, "answers": answers})
    assert v4.gates() == draft_gates
    assert v4.action_at() == Derived("discard", ["rule_supported: no answer"])
    assert v4.gates(WRITE_GATES_V2) == draft_gates
    assert v4.gates(("failed", "generalizable")) == {"failed": 0.9, "generalizable": 0.8}

    v1_answers = {"failed": yes(0.9), "recurrence": yes(0.8), "supported": yes(0.7)}
    v1 = Verdict({"questions_version": "questions_v1", "answers": v1_answers})
    assert v1.gates() == {"failed": 0.9, "recurrence": 0.8, "supported": 0.7}
    assert v1.action_at().action == "write_lesson"
    assert v1.action_at({"recurrence": 0.9}).action == "review"
    assert v1.p("grounded") == v1.p("supported") == 0.7  # One question under two names.
    assert v1.declined is False  # A questions_v1 row has no such key.


def test_an_unanswered_gate_is_none_and_a_gate_never_asked_is_absent():
    row = {
        "questions_version": QUESTIONS_VERSION,
        "answers": {
            "grounded": stored({"yes": 0.5, "no": 0.5}, method="llm_json_nothink_unmatched"),
            "faithful": yes(0.9),
            "generalizable": stored({"0": 1 / 3, "1": 1 / 3, "2": 1 / 3}, method="min_over_claims"),
        },
    }
    verdict = Verdict(row)
    assert verdict.gates() == {
        "grounded": None,
        "faithful": 0.9,
        "generalizable": pytest.approx(2 / 3),
    }
    # A placeholder is no reading: nothing scores it.
    assert verdict.p("grounded") is None
    assert verdict.value("grounded") is None
    assert verdict.confidence("grounded") == 0.0
    assert verdict.p("enforced") is None
    assert verdict.action_at(NO_BAR) == Derived(
        "review", ["grounded: unanswered", "enforced: no answer", "rule_supported: no answer"]
    )
    assert Verdict({**row, "declined": True}).action_at(NO_BAR).action == "discard"


def test_answers_are_read_as_the_text_a_label_holds():
    verdict = Verdict(
        {
            "answers": {
                "generalizable": stored({"0": 0.1, "1": 0.3, "2": 0.6}, value=1.6),
                "category": stored({"none": 0.7, "tool_misuse": 0.3}, value="none"),
                "enforced": stored({"none": 0.25, "tooling": 0.75}, value="tooling"),
                "failed": yes(0.2),
            }
        }
    )
    assert verdict.value("generalizable") == "2"  # A score is read as its nearest level.
    assert verdict.value("category") == "none"
    assert verdict.value("missing") is None
    assert (verdict.p("generalizable"), verdict.p("enforced")) == (pytest.approx(0.9), 0.25)
    assert (verdict.p("failed"), verdict.confidence("failed")) == (0.2, 0.8)
    assert verdict.confidence("missing") == 0.0


def test_a_verdict_reads_its_row_in_place_and_never_changes_it():
    row = judge(Scripted(), ITEM, STATE, DRAFT, DEFAULT_THRESHOLDS, run_id="r").row
    verdict = Verdict(row)
    assert verdict.row is row
    before = json.dumps(row)
    verdict.gates()
    verdict.action_at(NO_BAR)
    verdict.turn_answers()
    verdict.decision_record(ITEM, "2026-01-01T00:00:00+00:00")
    assert json.dumps(row) == before
    row["answers"].pop("failed")  # A later edit to the row is seen: nothing is cached.
    assert verdict.p("failed") is None
    row["answers"]["enforced"]["method"] = "llm_json_missing"
    assert verdict.gates()["enforced"] is None


# The ledger copy


def test_the_decision_record_carries_distributions_and_provenance_but_no_text():
    draft = structured_draft(claims=[{"text": "The database was locked.", "cited": [2]}]).to_dict()
    verdict = judge(RulesBackend(), ITEM, STATE, draft, {}, run_id="r")
    hashed = draft_hash(draft["lesson"])
    assert hashed is not None
    assert verdict.draft_hash == hashed
    record = verdict.decision_record(ITEM, "2026-01-01T00:00:00+00:00")
    assert (record.agent_id, record.kind, record.harness_session_id) == (
        "judge",
        "decision",
        "judge-r",
    )
    assert record.source_key == "/".join(
        (
            "rules",
            "i",
            QUESTIONS_VERSION,
            ITEM["projection_hash"][:16],
            hashed[:16],
            "ollama:m",
        )
    )
    assert record.content == f"judge rules: {verdict.recorded.action} for turn turn"
    assert record.session is not None
    assert (record.session.project, record.session.scope) == ("judge-experiment", "judge")
    assert record.session.metadata == {"run_id": "r"}
    assert record.payload is not None
    payload = record.payload["judge"]
    assert set(payload) == (set(ROW_KEYS) - {"decided_at"}) | {
        "triggers_version",
        "stratum",
        "flags",
        "turn_event_id",
    }
    for key in ("action", "reasons", "thresholds", "latency_ms", "cost_usd", "claims"):
        assert payload[key] == verdict.row[key]
    assert (payload["stratum"], payload["flags"], payload["turn_event_id"]) == (
        "T0",
        ["T0"],
        "turn",
    )
    assert set(payload["answers"]) == set(verdict.row["answers"])
    for qid, answer in payload["answers"].items():
        assert answer == {
            k: verdict.row["answers"][qid][k]
            for k in ("value", "probabilities", "confidence", "method")
        }
    text = record.to_json()
    assert "SENTINEL_STATE_TEXT" not in text
    assert "database was locked" not in text
    assert draft["rule"] not in text

    elsewhere = verdict.decision_record(
        ITEM, "2026-01-01T00:00:00+00:00", harness_session_id="admission-day", project="p"
    )
    assert elsewhere.session is not None
    assert (elsewhere.harness_session_id, elsewhere.session.project) == ("admission-day", "p")
    assert elsewhere.source_key == record.source_key

    # Keyed on every input: the same turn with no draft is another event.
    undrafted = judge(RulesBackend(), ITEM, STATE, None, {}, run_id="r")
    key = undrafted.decision_record(ITEM, "2026-01-01T00:00:00+00:00").source_key
    assert key.endswith("/nodraft/nodrafter")
    assert key != record.source_key
