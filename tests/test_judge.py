"""Judge experiment: triggers, projection, questions, backends, metrics, labeling, report."""

from __future__ import annotations

import io
import os
import stat
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from rich.console import Console

from agent_memory.adapters.droid import DroidAdapter
from agent_memory.capture import ids
from agent_memory.capture.redact import Redactor
from agent_memory.config import DEFAULT_HOME, Config
from agent_memory.judge import metrics
from agent_memory.judge.backends import JudgeInput, make_backend
from agent_memory.judge.backends import http as judge_http
from agent_memory.judge.backends.base import REMOTE_ENV, Answer, normalize
from agent_memory.judge.backends.jev import JevBackend, parse_answers, request_body
from agent_memory.judge.backends.llm import (
    LlmJudgeBackend,
    answers_from_judge_json,
    parse_judge_json,
)
from agent_memory.judge.backends.readout import ReadoutBackend, _letter_logits, first_letter
from agent_memory.judge.backends.rules import RulesBackend
from agent_memory.judge.drafter import make_drafter, parse_draft
from agent_memory.judge.labeling import label_items
from agent_memory.judge.pipeline import assign_splits
from agent_memory.judge.projection import FullEvent, build_projection
from agent_memory.judge.questions import (
    ALL_QUESTIONS,
    BY_ID,
    STAGE_A,
    STAGE_B,
    derive_action,
    thresholds_from_run,
)
from agent_memory.judge.report import build_report
from agent_memory.judge.store import JudgeError, RunStore, ensure_experiment_home
from agent_memory.judge.triggers import (
    Candidate,
    TurnEvent,
    candidates_from_turns,
    is_correction,
    sample_candidates,
    segment_turns,
    tool_error_facts,
    trigger_flags,
)
from tests.conftest import FIXTURES

JUDGE_TRANSCRIPT = FIXTURES / "droid_session_judge.jsonl"


def fixture_rows():
    """Adapter records turned into the compact rows `fetch_turn_events` would return."""
    records = DroidAdapter().from_transcript(JUDGE_TRANSCRIPT)
    sid = ids.session_id(records[0].agent_id, records[0].harness_session_id)
    rows = []
    for r in sorted(records, key=lambda r: (r.ts, r.seq)):
        if r.kind not in ("user_msg", "assistant_msg", "tool_call", "tool_result", "outcome"):
            continue
        parent = (
            ids.event_id(r.agent_id, r.harness_session_id, "tool_call", r.parent_key)
            if r.parent_key
            else None
        )
        event = TurnEvent(
            event_id=ids.event_id(r.agent_id, r.harness_session_id, r.kind, r.source_key),
            kind=r.kind,
            seq=r.seq,
            ts=r.ts.isoformat(),
            snippet=(r.content or "")[:500] if r.kind in ("user_msg", "outcome") else None,
            payload=r.payload,
            parent_event_id=parent,
        )
        rows.append((sid, "demo", "2026-10-26T09:00:00+00:00", event))
    return records, rows


def full_events(records, turn):
    by_key = {
        ids.event_id(r.agent_id, r.harness_session_id, r.kind, r.source_key): r for r in records
    }
    out = []
    for e in turn.events:
        r = by_key[e.event_id]
        out.append(FullEvent(e.event_id, r.kind, r.content or "", r.payload, e.parent_event_id))
    nxt = None
    if turn.next_user:
        r = by_key[turn.next_user.event_id]
        nxt = FullEvent(turn.next_user.event_id, r.kind, r.content or "", r.payload, None)
    return out, nxt


# Triggers


def test_segmentation_and_triggers_on_fixture():
    _, rows = fixture_rows()
    turns = segment_turns(rows)
    assert len(turns) == 4
    assert [t.outcome_reason for t in turns] == [
        "completed",
        "completed",
        "cancelled",
        "model_provider_unavailable",
    ]
    assert [trigger_flags(t) for t in turns] == [["T0"], ["T3", "T4"], ["T1"], ["T2"]]
    facts = tool_error_facts(turns[1])
    assert facts["tool_error_count"] == 2
    assert facts["repeated_tool_error"] is True
    assert facts["last_result_ok"] is False
    assert facts["tools"] == ["Execute"]
    assert turns[0].next_user is not None
    assert turns[3].next_user is None
    candidates = candidates_from_turns(turns)
    assert [c.stratum for c in candidates] == ["T0", "T3", "T1", "T2"]
    assert candidates[1].next_user_event_id == turns[2].turn_id
    assert len({c.item_id for c in candidates}) == 4


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("No, that's wrong. Revert it.", True),
        ("why did you delete the tests?", True),
        ("Thanks! Now add the docs.", False),
        ("Notice how this is fine", False),
        (None, False),
    ],
)
def test_correction_lexicon(text, expected):
    assert is_correction(text) is expected


def make_candidate(i: int, stratum: str, caller: str = "human") -> Candidate:
    return Candidate(
        item_id=f"{i:08d}-0000-0000-0000-000000000000",
        session_id="s",
        project="p" if i % 2 else "q",
        session_started_at=None,
        turn_event_id=f"t{i}",
        event_ids=[],
        next_user_event_id=None,
        outcome_reason=None,
        flags=[stratum],
        stratum=stratum,
        facts={},
        caller=caller,
    )


def test_sampling_splits_the_budget_between_human_and_agent_sessions():
    human = [make_candidate(i, "T3") for i in range(60)]
    agent = [make_candidate(1000 + i, "T3" if i % 3 else "T0", "agent") for i in range(300)]
    chosen = sample_candidates(human + agent, n=40, seed=1)
    callers = [c.caller for c in chosen]
    assert callers.count("human") == 20
    assert callers.count("agent") == 20
    assert chosen == sample_candidates(human + agent, n=40, seed=1)
    # A short human side is back-filled from agent sessions, and the reverse.
    chosen = sample_candidates(human[:5] + agent, n=40, seed=1)
    assert [c.caller for c in chosen].count("human") == 5
    assert len(chosen) == 40
    chosen = sample_candidates(human + agent[:3], n=40, seed=1)
    assert [c.caller for c in chosen].count("agent") == 3
    assert len(chosen) == 40
    # Triggered strata running dry are filled from clean turns rather than shrinking the sample.
    mostly_clean = [make_candidate(i, "T3") for i in range(4)] + [
        make_candidate(100 + i, "T0") for i in range(50)
    ]
    assert len(sample_candidates(mostly_clean, n=20, seed=1)) == 20
    splits = assign_splits(chosen, seed=1)
    assert set(splits) == {c.item_id for c in chosen}


# Provenance


@pytest.mark.parametrize(
    ("text", "caller", "expected"),
    [
        ("Please fix the failing test.", "human", "human"),
        ("<system-reminder>\nstuff\n</system-reminder>\nPlease fix the test.", "human", "human"),
        ("<system-reminder>\nonly a reminder\n</system-reminder>", "human", "harness"),
        ("<system-reminder>\ncut off by the inline limit", "human", "harness"),
        ("Request interrupted by user", "human", "harness"),
        ("Interrupted", "agent", "harness"),
        ("Script Output from toolu_01\nchain-exit=0", "human", "harness"),
        ("Background task completed. task_id: abc", "agent", "harness"),
        (
            "# Task Tool Invocation\nSubagent type: worker\n\nBuild the thing.",
            "agent",
            "parent_agent",
        ),
        ("# Follow-up Instructions\nNo, revert that.", "agent", "parent_agent"),
        ("# Follow-up Instructions\nNo, revert that.", "human", "parent_agent"),
        (
            "<system-reminder>You are a worker</system-reminder>\n## Your Assigned Feature",
            "agent",
            "parent_agent",
        ),
        ("Verify the prerequisites and report back.", "agent", "parent_agent"),
        ("", "human", "harness"),
        (None, "agent", "harness"),
    ],
)
def test_message_provenance(text, caller, expected):
    from agent_memory.judge.provenance import message_provenance

    assert message_provenance(text, caller) == expected


def test_strip_harness_blocks_and_callers():
    from agent_memory.judge.provenance import (
        caller_from_start,
        role_from_title,
        strip_harness_blocks,
    )
    from agent_memory.judge.triggers import _needs_blob, user_snippet

    body, n = strip_harness_blocks(
        "<system-reminder>a</system-reminder>\n<system-notification>b</system-notification>\n"
        "do it\n<system-reminder>unclosed"
    )
    assert (body, n) == ("do it", 3)
    assert user_snippet("<system-reminder>x</system-reminder>  hello  ") == "hello"
    assert _needs_blob("<system-reminder>cut") is True
    assert _needs_blob("") is True
    assert _needs_blob("hello") is False
    assert role_from_title("Task-coordinator: Verify prerequisites") == "task-coordinator"
    assert role_from_title("worker: Generate pages") == "worker"
    assert role_from_title("Security sweeper: scan") == "security-sweeper"
    assert role_from_title("Validate M1 protocol foundation") is None
    assert role_from_title(None) is None
    assert caller_from_start({"title": "New Session"}).kind == "human"
    agent = caller_from_start(
        {"title": "Worker: Build the parser", "callingSessionId": "p1", "callingToolUseId": "c1"}
    )
    assert (agent.kind, agent.role, agent.calling_session_id, agent.label) == (
        "agent",
        "worker",
        "p1",
        "agent:worker",
    )
    assert caller_from_start({"calling_session_id": "p2", "title": "x"}).kind == "agent"


def _rows(session: str, script: Sequence[tuple[str, Any]]):
    """Compact rows from (kind, text) pairs; tool results after a call are its results."""
    rows = []
    last_call = None
    for i, (kind, text) in enumerate(script):
        eid = f"{session}-{i}"
        payload = None
        parent = None
        if kind == "tool_call":
            payload = {"tool_name": "Execute"}
            last_call = eid
        elif kind == "tool_result":
            payload = {"is_error": text == "error"}
            parent = last_call
        snippet = text if kind in ("user_msg", "outcome") else None
        rows.append(
            (
                session,
                "p",
                None,
                TurnEvent(
                    eid, kind, i, f"2026-10-26T09:00:{i:02d}+00:00", snippet, payload, parent
                ),
            )
        )
    return rows


def test_segmentation_tags_provenance_and_splits_corrections_by_author():
    from agent_memory.judge.provenance import Caller

    script = [
        ("user_msg", "# Task Tool Invocation\nSubagent type: worker\n\nAdd the parser."),
        ("tool_call", None),
        ("tool_result", "ok"),
        ("outcome", "completed"),
        ("user_msg", "Background task completed. task_id: 1"),
        ("assistant_msg", None),
        ("outcome", "completed"),
        ("user_msg", "# Follow-up Instructions\nNo, that's wrong. Revert it."),
        ("assistant_msg", None),
        ("outcome", "completed"),
    ]
    agent_rows = _rows("worker-session", script)
    human_script = [
        (
            k,
            t.replace("# Task Tool Invocation\nSubagent type: worker\n\n", "").replace(
                "# Follow-up Instructions\n", ""
            )
            if t
            else t,
        )
        for k, t in script
    ]
    human_rows = _rows("human-session", human_script)
    callers = {"worker-session": Caller("agent", "worker", "parent-1")}
    turns = segment_turns(agent_rows + human_rows, callers)
    assert len(turns) == 6
    worker, harness_opened, follow_up = turns[:3]
    assert worker.opener_provenance == "parent_agent"
    assert harness_opened.opener_provenance == "harness"
    assert follow_up.opener_provenance == "parent_agent"
    # The harness notice is skipped: the worker turn's next message is the parent's follow-up.
    assert worker.next_user is not None
    assert worker.next_user.event_id == follow_up.turn_id
    assert worker.next_provenance == "parent_agent"
    assert trigger_flags(worker) == ["T5"]
    human = turns[3]
    assert human.opener_provenance == "human"
    assert human.next_provenance == "human"
    assert trigger_flags(human) == ["T4"]
    candidates = candidates_from_turns(turns)
    # Harness-opened turns are not candidates; the rest keep their caller.
    assert [c.stratum for c in candidates] == ["T5", "T0", "T4", "T0"]
    assert [c.caller for c in candidates] == ["agent", "agent", "human", "human"]
    assert candidates[0].caller_role == "worker"
    assert candidates[0].facts["caller"] == "agent:worker"
    assert candidates[0].next_provenance == "parent_agent"
    assert candidates[2].next_provenance == "human"


def test_projection_shows_who_is_speaking_and_strips_harness_blocks():
    from agent_memory.judge.provenance import Caller

    script = [
        ("user_msg", "<system-reminder>long reminder</system-reminder>\nAdd the parser."),
        ("tool_call", None),
        ("tool_result", "error"),
        ("outcome", "completed"),
        ("user_msg", "# Follow-up Instructions\nNo, that's wrong."),
        ("assistant_msg", None),
        ("outcome", "completed"),
    ]
    rows = _rows("w", script)
    turns = segment_turns(rows, {"w": Caller("agent", "worker", "parent-1")})
    cand = candidates_from_turns(turns)[0]
    events = [
        FullEvent(e.event_id, e.kind, e.snippet or "", e.payload, e.parent_event_id)
        for e in turns[0].events
    ]
    events[0] = FullEvent(
        events[0].event_id,
        "user_msg",
        "<system-reminder>long reminder</system-reminder>\nAdd the parser.",
        None,
        None,
    )
    nxt = FullEvent(
        turns[1].turn_id, "user_msg", "# Follow-up Instructions\nNo, that's wrong.", None, None
    )
    proj = build_projection(cand, events, nxt)
    assert "session_started_by=agent:worker" in proj.text
    assert "[1] REQUEST (parent agent): Add the parser. [1 harness block(s) removed]" in proj.text
    assert "long reminder" not in proj.text
    assert "NEXT MESSAGE (parent agent): # Follow-up Instructions" in proj.text


def test_sampling_is_seeded_and_stratified():
    pool = (
        [make_candidate(i, "T2") for i in range(5)]
        + [make_candidate(100 + i, "T1") for i in range(40)]
        + [make_candidate(200 + i, "T3") for i in range(40)]
        + [make_candidate(300 + i, "T4") for i in range(3)]
        + [make_candidate(400 + i, "T0") for i in range(100)]
    )
    chosen = sample_candidates(pool, n=40, seed=3)
    strata = [c.stratum for c in chosen]
    assert len(chosen) == 40
    assert strata.count("T2") == 5  # every T2
    assert strata.count("T0") == 10  # clean share
    assert strata.count("T4") == 3  # short stratum back-filled by the others
    assert strata.count("T1") + strata.count("T3") == 22
    assert chosen == sample_candidates(pool, n=40, seed=3)
    assert chosen != sample_candidates(pool, n=40, seed=4)
    splits = assign_splits(chosen, seed=3)
    assert set(splits.values()) == {"calibration", "held_out"}
    assert abs(sum(v == "calibration" for v in splits.values()) - 24) <= 2  # per-stratum rounding
    assert splits == assign_splits(chosen, seed=3)


# Projection


def test_projection_is_deterministic_and_bounded_and_redacted():
    records, rows = fixture_rows()
    turns = segment_turns(rows)
    cands = candidates_from_turns(turns)
    events, nxt = full_events(records, turns[1])
    first = build_projection(cands[1], events, nxt, Redactor())
    second = build_projection(cands[1], events, nxt, Redactor())
    assert first.hash == second.hash
    assert "TOOL RESULT (error)" in first.text
    assert "NEXT MESSAGE (human): No, that's wrong" in first.text
    assert "[1] REQUEST (human): " in first.text
    assert "session_started_by=human outcome=completed tool_calls=2 tool_errors=2" in first.text
    assert not any(e.event_id in first.text for e in events)
    assert [s["event_id"] for s in first.sidecar if s["kind"] == "tool_call"] == [
        e.event_id for e in events if e.kind == "tool_call"
    ]
    # A secret in a tool result never reaches the judge; a huge input is clipped.
    leaky = [
        FullEvent(
            e.event_id,
            e.kind,
            "token ghp_" + "a" * 40 if e.kind == "tool_result" else e.content,
            e.payload,
            e.parent_event_id,
        )
        for e in events
    ]
    leaky[1] = FullEvent(
        leaky[1].event_id,
        leaky[1].kind,
        leaky[1].content,
        {"tool_name": "Execute", "tool_input": {"command": "x" * 5000}},
        leaky[1].parent_event_id,
    )
    proj = build_projection(cands[1], leaky, nxt, Redactor())
    assert "ghp_" not in proj.text
    assert "[REDACTED:github_token]" in proj.text
    assert proj.redactions >= 1
    assert "x" * 300 not in proj.text


# Questions


def test_question_set_shape_and_derived_action():
    assert [q.id for q in STAGE_A] == [
        "failed",
        "user_corrected",
        "env_confirmed",
        "category",
        "recurrence",
    ]
    assert [q.id for q in STAGE_B] == ["supported", "generalizable", "actionable"]
    assert BY_ID["category"].options()[-1] == "none"
    assert BY_ID["generalizable"].options() == ["0", "1", "2"]
    assert (
        derive_action({"failed": 0.9, "recurrence": 0.8, "supported": 0.9}).action == "write_lesson"
    )
    assert derive_action({"failed": 0.9, "recurrence": 0.2, "supported": 0.9}).action == "discard"
    assert derive_action({"failed": 0.55, "recurrence": 0.9, "supported": 0.9}).action == "review"
    assert derive_action({"failed": 0.9}).action == "discard"
    run = {
        "thresholds": {"failed": 0.4},
        "frozen": {"backends": {"rules": {"thresholds": {"failed": 0.7}}}},
    }
    assert thresholds_from_run(run, "rules")["failed"] == 0.7
    assert thresholds_from_run(run, "other")["failed"] == 0.4
    assert thresholds_from_run(run)["band"] == 0.15


# Backends


def check_contract(answers: dict[str, Answer], questions) -> None:
    assert set(answers) == {q.id for q in questions}
    for q in questions:
        a = answers[q.id]
        assert set(a.probabilities) == set(q.options())
        assert abs(sum(a.probabilities.values()) - 1) < 1e-9
        assert 0 <= a.confidence <= 1
        if q.type != "score":
            assert a.value in q.options()
            assert a.probabilities[str(a.value)] == a.confidence


def test_rules_backend_contract_and_direction():
    inp = JudgeInput(
        item_id="i",
        state="TOOL CALL Execute: uv run mem migrate\nTOOL RESULT (error): locked",
        facts={"repeated_tool_error": True, "tools": ["Execute"], "last_result_ok": False},
        flags=("T3", "T4"),
        draft="Stop mem serve before you run mem migrate on this project.",
    )
    answers = RulesBackend().decide(inp, ALL_QUESTIONS)
    check_contract(answers, ALL_QUESTIONS)
    assert answers["failed"].p_true == pytest.approx(0.9)
    assert answers["user_corrected"].value == "yes"
    assert answers["category"].value == "tool_misuse"
    assert answers["recurrence"].value == "yes"
    assert answers["supported"].value == "yes"
    assert answers["actionable"].value == "yes"
    clean = RulesBackend().decide(
        JudgeInput(item_id="j", state="", facts={"last_result_ok": True}, flags=("T0",)), STAGE_A
    )
    assert clean["failed"].value == "no"
    assert clean["category"].value == "none"
    assert normalize({"a": -1, "b": 0}, ["a", "b"]) == {"a": 0.5, "b": 0.5}


def test_make_backend_specs(monkeypatch):
    assert make_backend("rules").name == "rules"
    assert make_backend("readout:gemma4:26b").name == "readout:gemma4:26b"
    assert make_backend("llm:gemma4:26b").name == "llm:gemma4:26b"
    assert make_backend("jev").name == "jev:typesafe/jev-1.14"
    assert make_backend("jev:typesafe/jev-1.12").name == "jev:typesafe/jev-1.12"
    for bad in ("readout", "llm", "nope", "??"):
        with pytest.raises(JudgeError):
            make_backend(bad)


JEV_RESPONSE = {
    "model": "typesafe/jev-1.14-20260915",
    "answers": {
        "failed": {"type": "noul", "noul": 0.91},
        "category": {
            "type": "choice",
            "choice": "tool_misuse",
            "probabilities": {
                "wrong_assumption": 0.1,
                "tool_misuse": 0.7,
                "stale_or_missing_knowledge": 0.1,
                "environment_or_provider": 0.05,
                "user_redirect": 0.03,
                "none": 0.02,
            },
            "confidence": 0.7,
        },
        "generalizable": {
            "type": "score",
            "score": 1.2,
            "probabilities": [0.1, 0.6, 0.3],
            "confidence": 0.6,
        },
    },
    "usage": {"input_tokens": 1200, "output_tokens": 3, "cost": 0.00006},
}


def test_jev_request_and_response_shapes(monkeypatch):
    questions = [BY_ID["failed"], BY_ID["category"], BY_ID["generalizable"]]
    body = request_body("typesafe/jev-1.14", "STATE", questions)
    assert body["model"] == "typesafe/jev-1.14"
    assert body["state"] == "STATE"
    assert body["questions"]["failed"]["type"] == "noul"
    assert set(body["questions"]["failed"]["criteria"]) == {"true", "false"}
    assert list(body["questions"]["category"]["criteria"]) == BY_ID["category"].options()
    assert len(body["questions"]["generalizable"]["criteria"]) == 3
    answers = parse_answers(JEV_RESPONSE, questions, latency_ms=150.0)
    assert answers["failed"].p_true == pytest.approx(0.91)
    assert answers["failed"].value == "yes"
    assert answers["category"].value == "tool_misuse"
    assert answers["category"].confidence == pytest.approx(0.7)
    assert answers["generalizable"].value == pytest.approx(1.2)
    assert answers["generalizable"].probabilities == {"0": 0.1, "1": 0.6, "2": 0.3}
    assert all(a.model_snapshot == "typesafe/jev-1.14-20260915" for a in answers.values())
    assert sum(a.cost_usd for a in answers.values()) == pytest.approx(0.00006)
    assert sum(a.latency_ms for a in answers.values()) == pytest.approx(150.0)
    with pytest.raises(JudgeError, match="lacks an answer"):
        parse_answers({"answers": {}}, questions, latency_ms=1)


def test_jev_refuses_without_opt_in_and_key(monkeypatch):
    monkeypatch.delenv(REMOTE_ENV, raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    calls = []
    monkeypatch.setattr(
        judge_http, "post_json", lambda *a, **k: calls.append(a) or (JEV_RESPONSE, 1.0)
    )
    inp = JudgeInput(item_id="i", state="S", facts={}, flags=())
    with pytest.raises(JudgeError, match=REMOTE_ENV):
        JevBackend().decide(inp, [BY_ID["failed"]])
    monkeypatch.setenv(REMOTE_ENV, "1")
    with pytest.raises(JudgeError, match="OPENROUTER_API_KEY"):
        JevBackend().decide(inp, [BY_ID["failed"]])
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    answers = JevBackend().decide(inp, [BY_ID["failed"]])
    assert answers["failed"].p_true == pytest.approx(0.91)
    assert calls
    assert calls[0][0] == "https://openrouter.ai/api/alpha/decisions"
    assert not JevBackend().local


def test_llm_judge_truncated_json_is_judge_error_and_falls_back_without_thinking(monkeypatch):
    cut = '{"failed": {"answer": "yes", "probability": 0.9}, "category": {"answer": "wro'
    with pytest.raises(JudgeError, match="unparsable JSON"):
        parse_judge_json(cut)
    with pytest.raises(JudgeError, match="no JSON"):
        parse_judge_json("")

    good = '{"failed": {"answer": "no", "probability": 0.8}}'
    calls: list[dict] = []

    def post(url, body, **k):
        calls.append(dict(body))
        if body.get("think"):
            return {"message": {"content": cut}, "done_reason": "length"}, 100.0
        return {"message": {"content": good}, "done_reason": "stop", "model": "m:1"}, 10.0

    monkeypatch.setattr(judge_http, "post_json", post)
    inp = JudgeInput(item_id="i", state="S", facts={}, flags=())
    answers = LlmJudgeBackend("m").decide(inp, [BY_ID["failed"]])
    assert [c.get("think") for c in calls] == [True, None]
    assert [c.get("format") for c in calls] == ["json", None]
    assert answers["failed"].value == "no"
    assert answers["failed"].method == "llm_json_nothink"
    assert answers["failed"].model_snapshot == "m:1"

    # Without thinking there is nothing to fall back to; the error surfaces for the caller.
    monkeypatch.setattr(
        judge_http, "post_json", lambda u, b, **k: ({"message": {"content": ""}}, 1.0)
    )
    with pytest.raises(JudgeError, match="no JSON"):
        LlmJudgeBackend("m", think=False).decide(inp, [BY_ID["failed"]])


def test_readout_logprobs_and_sampling_fallback(monkeypatch):
    entries = [
        {
            "token": "B",
            "logprob": -0.2,
            "top_logprobs": [{"token": "B", "logprob": -0.2}, {"token": " A", "logprob": -1.8}],
        }
    ]
    logits = _letter_logits(entries, 2)
    assert logits == {"B": -0.2, "A": -1.8}
    assert _letter_logits([{"token": "Z", "logprob": -0.1}], 2) == {}
    assert first_letter(" b.", 2) == "B"
    assert first_letter("maybe", 2) is None

    responses = iter(
        [
            ({"model": "m-1", "message": {"content": "B", "logprobs": entries}}, 10.0),
        ]
    )
    monkeypatch.setattr(judge_http, "post_json", lambda url, body, **k: next(responses))
    backend = ReadoutBackend("m", samples=4)
    answers = backend.decide(
        JudgeInput(item_id="i", state="S", facts={}, flags=()), [BY_ID["failed"]]
    )
    a = answers["failed"]
    assert a.method == "logprobs"
    assert a.value == "no"
    assert a.probabilities["no"] > a.probabilities["yes"]
    assert a.model_snapshot == "m-1"

    seen_bodies = []

    def no_logprobs(url, body, **k):
        seen_bodies.append(body)
        letter = "A" if body["options"]["temperature"] == 0 else "B"
        return {"model": "m-1", "message": {"content": letter}}, 5.0

    monkeypatch.setattr(judge_http, "post_json", no_logprobs)
    answers = backend.decide(
        JudgeInput(item_id="i", state="S", facts={}, flags=()), [BY_ID["failed"]]
    )
    a = answers["failed"]
    assert a.method == "sampling:4"
    assert len(seen_bodies) == 4
    assert seen_bodies[0]["logprobs"] is True
    assert a.probabilities["yes"] == pytest.approx(1.5 / 5)
    assert a.latency_ms == pytest.approx(20.0)


def test_llm_json_parsing():
    text = (
        "Sure.\n```json\n"
        '{"failed": {"answer": "YES", "probability": 0.8}, '
        '"category": {"answer": "tool_misuse", "probability": "0.6"}, '
        '"generalizable": {"answer": "level 1", "probability": 0.7}, '
        '"supported": {"answer": "maybe", "probability": 0.5}}\n```'
    )
    data = parse_judge_json(text)
    questions = [
        BY_ID["failed"],
        BY_ID["category"],
        BY_ID["generalizable"],
        BY_ID["supported"],
        BY_ID["actionable"],
    ]
    answers = answers_from_judge_json(data, questions, latency_ms=1.0, model_snapshot="m")
    check_contract(answers, questions)
    assert answers["failed"].value == "yes"
    assert answers["failed"].p_true == pytest.approx(0.8)
    assert answers["category"].confidence == pytest.approx(0.6)
    assert answers["generalizable"].value == pytest.approx(0.7 * 1 + 0.15 * 0 + 0.15 * 2)
    assert answers["supported"].method == "llm_json_unmatched"
    assert answers["actionable"].method == "llm_json_missing"
    with pytest.raises(JudgeError, match="no JSON"):
        parse_judge_json("nothing here")


def test_drafter_parsing_and_spec():
    lesson, cited = parse_draft({"lesson": "Stop  the server\nfirst.", "cited": ["[3]", 5, "x", 3]})
    assert lesson == "Stop the server first."
    assert cited == [3, 5]
    with pytest.raises(JudgeError, match="empty"):
        parse_draft({"lesson": ""})
    assert make_drafter("ollama:gemma4:26b").name == "ollama:gemma4:26b"
    with pytest.raises(JudgeError):
        make_drafter("openai:gpt")


# Metrics


def test_metrics_known_values():
    assert metrics.auroc([0.9, 0.8, 0.2, 0.1], [True, True, False, False]) == 1.0
    assert metrics.auroc([0.1, 0.2, 0.8, 0.9], [True, True, False, False]) == 0.0
    assert metrics.auroc([0.5, 0.5, 0.5], [True, False, True]) == 0.5
    assert metrics.auroc([0.5], [True]) is None
    assert metrics.brier([1.0, 0.0, 0.5], [True, False, True]) == pytest.approx(0.25 / 3)
    assert metrics.ece([0.95, 0.95, 0.05, 0.05], [True, True, False, False]) == pytest.approx(0.05)
    assert metrics.ece([0.95, 0.95], [True, False]) == pytest.approx(0.45)
    cut, acc = metrics.best_threshold([0.2, 0.4, 0.6, 0.9], [False, False, True, True])
    assert (cut, acc) == (0.5, 1.0)
    cut, acc = metrics.best_threshold([0.2, 0.6, 0.7, 0.9], [False, False, True, True])
    assert cut == pytest.approx(0.65)
    assert acc == 1.0
    assert metrics.paired_discrimination([(0.9, 0.1), (0.5, 0.5), (0.1, 0.9)]) == 0.5
    assert metrics.accuracy([], []) is None


def test_cascade_and_gate_choice():
    conf = [0.9, 0.9, 0.6, 0.6]
    cheap_ok = [True, True, False, True]
    exp_ok = [True, True, True, True]
    cheap_cost = [1.0] * 4
    exp_cost = [10.0] * 4
    full = metrics.cascade(conf, cheap_ok, exp_ok, cheap_cost, exp_cost, gate=1.01)
    assert (full.accuracy, full.cost, full.escalated_share) == (1.0, 44.0, 1.0)
    none = metrics.cascade(conf, cheap_ok, exp_ok, cheap_cost, exp_cost, gate=0.0)
    assert (none.accuracy, none.cost, none.escalated_share) == (0.75, 4.0, 0.0)
    mid = metrics.cascade(conf, cheap_ok, exp_ok, cheap_cost, exp_cost, gate=0.7)
    assert (mid.accuracy, mid.cost, mid.escalated_share) == (1.0, 24.0, 0.5)
    best = metrics.choose_gate(conf, cheap_ok, exp_ok, cheap_cost, exp_cost, target_accuracy=0.99)
    assert (best.gate, best.cost) == (0.9, 24.0)
    cheap_enough = metrics.choose_gate(conf, cheap_ok, exp_ok, cheap_cost, exp_cost, 0.7)
    assert cheap_enough.cost == 4.0


# Store, labeling, report


def test_store_refuses_default_home_and_protects_projections(tmp_path, monkeypatch):
    with pytest.raises(JudgeError, match="default home"):
        ensure_experiment_home(Config(home=DEFAULT_HOME))
    cfg = Config(home=tmp_path / "exp")
    ensure_experiment_home(cfg)
    with pytest.raises(JudgeError, match="bad run id"):
        RunStore(cfg, "../x")
    store = RunStore(cfg, "r1")
    with pytest.raises(JudgeError, match="no run"):
        store.read_run()
    store.create({"n": 2})
    with pytest.raises(JudgeError, match="already exists"):
        store.create({})
    assert store.update_run(drafter="x")["drafter"] == "x"
    path = store.write_projection("item/with/slash", "secret state")
    assert path.name == "item_with_slash.txt"
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
    assert store.read_projection("item/with/slash") == "secret state"
    child = store.child("2026-11-01")
    assert child.root == store.live_dir / "2026-11-01"
    assert store.purge_projections() == 1
    with pytest.raises(JudgeError, match="no projection"):
        store.read_projection("item/with/slash")


def _verdict(
    item_id: str, i: int, failed: bool, backend: str, p_failed: float, conf: float, latency: float
) -> dict:
    def ans(value, probs, confidence):
        return {
            "value": value,
            "probabilities": probs,
            "confidence": confidence,
            "method": "x",
            "latency_ms": latency / 3,
            "cost_usd": 0.0,
            "model_snapshot": backend,
        }

    return {
        "run_id": "syn",
        "item_id": item_id,
        "backend": backend,
        "projection_hash": f"h{i}",
        "latency_ms": latency,
        "cost_usd": 0.0,
        "answers": {
            "failed": ans(
                "yes" if p_failed >= 0.5 else "no", {"yes": p_failed, "no": 1 - p_failed}, conf
            ),
            "recurrence": ans("yes", {"yes": 0.8, "no": 0.2}, 0.8),
            "supported": ans(
                "yes" if failed else "no",
                {"yes": 0.9 if failed else 0.1, "no": 0.1 if failed else 0.9},
                0.9,
            ),
            "category": ans("tool_misuse" if failed else "none", {}, 0.7),
            "generalizable": ans(1.0 if failed else 0.0, {"0": 0.2, "1": 0.6, "2": 0.2}, 0.6),
        },
        "action": "discard",
        "reasons": [],
        "thresholds": {},
    }


def synthetic_run(tmp_path: Path, *, n: int = 20) -> RunStore:
    cfg = Config(home=tmp_path / "exp")
    store = RunStore(cfg, "syn")
    store.create({"questions_version": "questions_v1", "thresholds": {}})
    items, labels, rules, llm = [], [], [], []
    for i in range(n):
        item_id = f"{i:08d}-0000-0000-0000-000000000000"
        failed = i % 2 == 1
        stratum = "T3" if failed else "T0"
        store.write_projection(item_id, f"PROJECTION {i} with SENTINEL_STATE_TEXT")
        items.append(
            {
                "item_id": item_id,
                "project": "p" if i % 4 < 2 else "q",
                "stratum": stratum,
                "flags": [stratum],
                "facts": {"tool_count": 1},
                "turn_event_id": f"t{i}",
                "outcome_reason": "completed",
                "split": "calibration" if i < 12 else "held_out",
                "projection_hash": f"h{i}",
            }
        )
        labels.append(
            {
                "item_id": item_id,
                "failed": failed,
                "category": "tool_misuse" if failed else "none",
                "worth_remembering": failed and i % 3 == 0,
                "supported": failed,
                "generalizable": 1 if failed else 0,
            }
        )
        # rules: right except on items 3 and 8, with low confidence when wrong
        wrong = i in (3, 8)
        p = (0.3 if failed else 0.6) if wrong else (0.85 if failed else 0.15)
        rules.append(_verdict(item_id, i, failed, "rules", p, 0.55 if wrong else 0.85, 2.0))
        llm.append(_verdict(item_id, i, failed, "llm:ref", 0.95 if failed else 0.05, 0.95, 4000.0))
    store.write_jsonl(store.items_path, items)
    store.write_jsonl(store.labels_path, labels)
    store.write_jsonl(store.verdicts_path("rules"), rules)
    store.write_jsonl(store.verdicts_path("llm:ref"), llm)
    return store


def test_report_locks_held_out_until_freeze(tmp_path):
    store = synthetic_run(tmp_path)
    text = build_report(store)
    assert "## Calibration split (12 items)" in text
    assert "Locked" in text
    assert "Held-out split (8 items)" not in text
    assert "SENTINEL_STATE_TEXT" not in text
    assert "| rules | 12 |" in text
    assert "| llm:ref | 12 | 1.000 |" in text
    frozen_text = build_report(store, freeze=True)
    run = store.read_run()
    assert run["frozen"]["reference"] == "llm:ref"
    assert set(run["frozen"]["backends"]) == {"rules", "llm:ref"}
    assert run["frozen"]["backends"]["rules"]["gate"] is not None
    assert "## Held-out split (8 items)" in frozen_text
    assert "## Frozen thresholds" in frozen_text
    assert "Cascade to `llm:ref`" in frozen_text
    assert "(2) confidence AUROC ≥ 0.75: yes" in frozen_text
    assert store.report_path.read_text() == frozen_text
    with pytest.raises(JudgeError, match="already frozen"):
        build_report(store, freeze=True)


def test_labeling_hides_verdicts_and_resumes(tmp_path):
    store = synthetic_run(tmp_path, n=3)
    store.labels_path.unlink()
    items = store.read_jsonl(store.items_path)
    buffer = io.StringIO()
    console = Console(file=buffer, width=100, force_terminal=False)
    # Every item has a draft so the question sequence does not depend on the shuffle.
    store.write_jsonl(
        store.drafts_path,
        [{"item_id": i["item_id"], "lesson": "DRAFT_LESSON_TEXT"} for i in items],
    )
    scripted = iter(["y", "2", "y", "y", "1", "1", "s", "q"])
    asked: list[str] = []

    def ask(question: str, choices):
        asked.append(question)
        return next(scripted)

    summary = label_items(store, console=console, ask=ask, seed=5)
    out = buffer.getvalue()
    assert summary.labeled == 1
    assert summary.skipped == 1
    assert summary.remaining == 2
    assert "PROJECTION" in out
    assert "DRAFT_LESSON_TEXT" in out
    assert "0.85" not in out
    assert "llm:ref" not in out
    assert "probabilities" not in out
    assert "backend" not in out
    labels = store.read_jsonl(store.labels_path)
    assert len(labels) == 1
    new = labels[0]
    assert new["category"] == "tool_misuse"
    assert new["labeled_at"]
    assert (new["supported"], new["generalizable"], new["enforced_by"]) == (True, 1, "none")
    assert new["error_intended"] is None
    # Resume: the labeled item is not offered again.
    again = label_items(store, console=console, ask=lambda q, c: "q", seed=5)
    assert again.labeled == 0
    assert again.remaining == 2


def test_labeling_shows_rubric_and_cited_evidence(tmp_path):
    from agent_memory.judge.labeling import cited_entries

    projection = (
        "Coding-agent turn. project=p\n[1] USER: please run the tests\n"
        "[2] TOOL Execute: uv run pytest\n  result: 3 failed\n[3] ASSISTANT: all done\n"
        "[4] OUTCOME: completed\n[5] USER NEXT: no, the tests fail"
    )
    entries = cited_entries(projection, [2, 5, 9])
    assert [n for n, _ in entries] == [2, 5]
    assert entries[0][1] == "[2] TOOL Execute: uv run pytest\n  result: 3 failed"
    long = "[1] USER: " + "x" * 1000
    assert cited_entries(long, [1])[0][1].endswith("…")

    store = synthetic_run(tmp_path, n=1)
    store.labels_path.unlink()
    item_id = "00000000-0000-0000-0000-000000000000"
    store.write_projection(item_id, projection)
    store.write_jsonl(
        store.drafts_path,
        [
            {
                "item_id": item_id,
                "lesson": "Check the test result before claiming done.",
                "cited": [2, 5],
            }
        ],
    )
    buffer = io.StringIO()
    console = Console(file=buffer, width=100, force_terminal=False)
    scripted = iter(["y", "2", "y", "y", "2", "2"])
    label_items(store, console=console, ask=lambda q, c: next(scripted), seed=5)
    out = buffer.getvalue()
    assert "how to label" in out
    assert "enforced by" in out
    assert "cites [2] [5]" in out
    assert "cited evidence" in out
    assert "no, the tests fail" in out
    assert "[3] ASSISTANT" in out  # still in the projection panel
    label = store.read_jsonl(store.labels_path)[0]
    assert (label["supported"], label["generalizable"]) == (True, 2)
    assert label["enforced_by"] == "harness"


def test_labeling_backfills_new_questions_and_rewrites_rows(tmp_path):
    from agent_memory.judge.labeling import LABEL_KEYS, applicable_keys, missing_keys
    from agent_memory.judge.report import should_write

    assert applicable_keys(None, False) == ("failed", "category", "worth_remembering")
    assert applicable_keys("lesson", True)[-2:] == ("enforced_by", "error_intended")
    assert missing_keys({"failed": True}, None, True) == [
        "category",
        "worth_remembering",
        "error_intended",
    ]
    assert should_write({"worth_remembering": True})
    assert should_write({"worth_remembering": True, "enforced_by": "none"})
    assert not should_write({"worth_remembering": True, "enforced_by": "harness"})
    assert not should_write({"worth_remembering": False, "enforced_by": "none"})

    store = synthetic_run(tmp_path, n=4)
    ids = [f"{i:08d}-0000-0000-0000-000000000000" for i in range(4)]
    items = store.read_jsonl(store.items_path)
    items[2]["facts"]["tool_error_count"] = 1
    store.write_jsonl(store.items_path, items)
    # Labels from before the two questions existed.
    old = [
        {k: v for k, v in row.items() if k not in ("enforced_by", "error_intended")}
        for row in store.read_jsonl(store.labels_path)
    ]
    for row in old:
        row["labeled_at"] = "2026-01-01T00:00:00+00:00"
    store.write_jsonl(store.labels_path, old)
    store.write_jsonl(
        store.drafts_path,
        [
            {"item_id": ids[0], "lesson": "Lesson zero."},
            {"item_id": ids[1], "lesson": "Lesson one."},
        ],
    )
    buffer = io.StringIO()
    console = Console(file=buffer, width=100, force_terminal=False)
    scripted = iter(["1", "4", "n"])
    asked: list[str] = []

    def ask(question: str, choices):
        asked.append(question)
        return next(scripted)

    summary = label_items(store, console=console, ask=ask, seed=5)
    out = buffer.getvalue()
    assert (summary.labeled, summary.skipped, summary.remaining) == (3, 0, 0)
    assert "3 labeled item(s) come first" in out
    assert "only asking: enforced_by" in out
    assert "only asking: error_intended" in out
    assert [q.split(" (")[0] for q in asked] == [
        "lesson already enforced by",
        "lesson already enforced by",
        "was the tool error intended and used?",
    ]
    rows = {row["item_id"]: row for row in store.read_jsonl(store.labels_path)}
    assert list(rows) == ids  # rewritten in place, nothing duplicated
    assert rows[ids[0]]["enforced_by"] == "none"
    assert rows[ids[1]]["enforced_by"] == "tooling"
    assert rows[ids[2]]["error_intended"] is False
    assert "enforced_by" not in rows[ids[3]]
    assert all(rows[i]["labeled_at"] == "2026-01-01T00:00:00+00:00" for i in ids)
    assert all("relabeled_at" in rows[i] for i in ids[:3])
    assert "relabeled_at" not in rows[ids[3]]
    assert rows[ids[0]]["failed"] is False  # untouched answers survive the rewrite
    # Nothing left to backfill: a second pass asks nothing.
    again = label_items(store, console=console, ask=lambda q, c: "q", seed=5)
    assert (again.labeled, again.remaining) == (0, 0)
    # A fresh item with a draft and tool errors is asked every applicable question.
    items.append(
        {
            **items[0],
            "item_id": "ffffffff-0000-0000-0000-000000000000",
            "facts": {"tool_count": 2, "tool_error_count": 1},
        }
    )
    store.write_jsonl(store.items_path, items)
    store.write_projection("ffffffff-0000-0000-0000-000000000000", "PROJECTION fresh")
    store.write_jsonl(
        store.drafts_path,
        [{"item_id": "ffffffff-0000-0000-0000-000000000000", "lesson": "Fresh."}],
        append=True,
    )
    scripted = iter(["y", "2", "y", "y", "1", "2", "y"])
    label_items(store, console=console, ask=lambda q, c: next(scripted), seed=5)
    fresh = store.read_jsonl(store.labels_path)[-1]
    assert set(LABEL_KEYS) <= set(fresh)
    assert (fresh["enforced_by"], fresh["error_intended"]) == ("harness", True)
    report = build_report(store)
    assert "## Labels on their own" in report
    assert "| human | 3 | 1 (33%) | 1 (33%) | 0 (0%) | 1 (33%) |" in report
    assert "2 labeled items had a tool error; in 1 (50%)" in report
    assert "writes enforced" in report


def test_post_json_retries_one_timeout(monkeypatch):
    import urllib.request

    calls = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return None

        def read(self):
            return b'{"ok": true}'

    def urlopen(request, timeout):
        calls.append(timeout)
        if len(calls) == 1:
            raise TimeoutError("timed out")
        return Response()

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    parsed, _ = judge_http.post_json("http://x/api", {"a": 1}, timeout=5)
    assert parsed == {"ok": True}
    assert len(calls) == 2
    calls.clear()
    monkeypatch.setattr(
        urllib.request, "urlopen", lambda r, timeout: (_ for _ in ()).throw(TimeoutError())
    )
    with pytest.raises(JudgeError, match="timed out after 5s"):
        judge_http.post_json("http://x/api", {}, timeout=5)


def test_drafter_caps_output_and_draft_items_skips_bad_items(tmp_path, monkeypatch):
    from agent_memory.judge import pipeline
    from agent_memory.judge.drafter import OllamaDrafter

    store = synthetic_run(tmp_path, n=4)
    store.drafts_path.unlink(missing_ok=True)
    bodies = []

    def fake_post(url, body, **kwargs):
        bodies.append(body)
        if len(bodies) == 2:
            raise JudgeError("http://x: timed out after 300s")
        return {
            "model": "m",
            "message": {"content": '{"lesson": "Run tests first.", "cited": [1]}'},
        }, 12.0

    monkeypatch.setattr(judge_http, "post_json", fake_post)
    cfg = Config(home=tmp_path / "exp")
    logs: list[str] = []
    written = pipeline.draft_items(cfg, "syn", "ollama:m", store=store, log=logs.append)
    assert written == 3
    assert bodies[0]["options"]["num_predict"] == 400
    assert any("skipped" in line for line in logs)
    assert len(store.read_jsonl(store.drafts_path)) == 3
    # The skipped item is picked up by the next run.
    assert pipeline.draft_items(cfg, "syn", "ollama:m", store=store) == 1
    assert isinstance(pipeline.make_drafter("ollama:m"), OllamaDrafter)

    # On resume the first undrafted item is the one that failed before; a drafter that has
    # already produced drafts in this run skips it instead of aborting.
    drafts = store.read_jsonl(store.drafts_path)
    store.write_jsonl(store.drafts_path, drafts[:-1])
    monkeypatch.setattr(
        judge_http, "post_json", lambda *a, **k: (_ for _ in ()).throw(JudgeError("down"))
    )
    logs.clear()
    assert pipeline.draft_items(cfg, "syn", "ollama:m", store=store, log=logs.append) == 0
    assert any("skipped" in line for line in logs)

    # A failure on the very first item of a fresh run is treated as configuration and raised.
    store.drafts_path.unlink()
    with pytest.raises(JudgeError, match="down"):
        pipeline.draft_items(cfg, "syn", "ollama:m", store=store)


def test_decide_items_skips_failing_item_on_resume(tmp_path, monkeypatch):
    from agent_memory.judge import pipeline

    store = synthetic_run(tmp_path, n=3)
    store.verdicts_path("rules").unlink()
    cfg = Config(home=tmp_path / "exp")
    items = store.read_jsonl(store.items_path)
    bad = items[0]["item_id"]
    real_decide = RulesBackend.decide

    def flaky(self, inp, questions):
        if inp.item_id == bad:
            raise JudgeError("judge ran out of tokens; JSON incomplete: ''")
        return real_decide(self, inp, questions)

    monkeypatch.setattr(RulesBackend, "decide", flaky)
    # Fresh run, first item fails: configuration, raised.
    with pytest.raises(JudgeError, match="ran out of tokens"):
        pipeline.decide_items(cfg, "syn", ["rules"], store=store, record_events=False)
    # One good verdict exists, so the same failure is now skipped and reported.
    seed = items[2]
    verdict = pipeline.judge_item(
        make_backend("rules"),
        seed,
        store.read_projection(seed["item_id"]),
        None,
        thresholds_from_run(store.read_run(), "rules"),
        run_id="syn",
    )
    store.write_jsonl(store.verdicts_path("rules"), [verdict])
    logs: list[str] = []
    written = pipeline.decide_items(
        cfg, "syn", ["rules"], store=store, record_events=False, log=logs.append
    )
    assert written == {"rules": 1}
    assert any(f"item {bad} failed" in line for line in logs)
    assert any("1 item(s) failed" in line for line in logs)
    verdicts = store.read_jsonl(store.verdicts_path("rules"))
    assert len(verdicts) == 2
    assert bad not in {v["item_id"] for v in verdicts}
