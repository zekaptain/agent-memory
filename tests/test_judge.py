"""Judge experiment: triggers, projection, questions, backends, metrics, labeling, report."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import stat
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from agent_memory import endpoints, model_call
from agent_memory.adapters.droid import DroidAdapter
from agent_memory.capture import ids
from agent_memory.capture.redact import Redactor
from agent_memory.config import DEFAULT_HOME, Config
from agent_memory.judge import metrics
from agent_memory.judge.backends import JudgeInput, make_backend
from agent_memory.judge.backends.base import REMOTE_ENV, Answer, normalize
from agent_memory.judge.backends.jev import JevBackend, parse_answers, request_body
from agent_memory.judge.backends.llm import (
    ReasoningJudge,
    answers_from_judge_json,
    llamacpp_judge,
    ollama_judge,
)
from agent_memory.judge.backends.readout import ReadoutBackend, _letter_logits, first_letter
from agent_memory.judge.backends.rules import RulesBackend
from agent_memory.judge.drafter import SYSTEM, Draft, draft_hash, drafter_prompt, make_drafter
from agent_memory.judge.label_controller import LabelController
from agent_memory.judge.lesson import (
    LESSON_KINDS,
    LESSON_SCHEMA,
    MAX_TEXT_CHARS,
    Claim,
    DraftReply,
    Lesson,
    entry_numbers,
    lesson_from_row,
    parse_lesson,
    render_lesson,
)
from agent_memory.judge.pipeline import assign_splits
from agent_memory.judge.projection import FullEvent, build_projection, drafter_rendering
from agent_memory.judge.questions import (
    ALL_QUESTIONS,
    BY_ID,
    DRAFT_QUESTIONS,
    QUESTIONS_VERSION,
    TURN_QUESTIONS,
    draft_questions_for,
    thresholds_from_run,
)
from agent_memory.judge.report import build_report
from agent_memory.judge.store import (
    JudgeError,
    RunStore,
    ServerUnreachable,
    ensure_experiment_home,
)
from agent_memory.judge.triggers import (
    Candidate,
    TurnEvent,
    candidates_from_turns,
    closed_by_of,
    is_correction,
    sample_candidates,
    segment_turns,
    tool_error_facts,
    trigger_flags,
)
from agent_memory.judge.verdict import Verdict
from agent_memory.model_call import LlamaCpp, Ollama
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
    # Every fixture turn reached an outcome.
    assert [t.closed_by for t in turns] == ["outcome"] * 4
    assert all(c.closed_by == "outcome" and c.facts["closed_by"] == "outcome" for c in candidates)


def test_closed_by_distinguishes_outcome_next_message_and_session_end():
    script = [
        ("user_msg", "Do A"),
        ("tool_call", None),
        ("tool_result", "error"),
        ("user_msg", "Now do B instead"),
        ("assistant_msg", None),
        ("outcome", "completed"),
        ("user_msg", "Do C"),
        ("tool_call", None),
        ("tool_result", "error"),
    ]
    turns = segment_turns(_rows("s", script))
    assert [t.closed_by for t in turns] == ["next_message", "outcome", "session_end"]
    cands = candidates_from_turns(turns)
    assert [c.stratum for c in cands] == ["T3", "T0", "T3"]
    assert [c.closed_by for c in cands] == ["next_message", "outcome", "session_end"]
    # A clean turn only counts as a control when it completed.
    cut_off_clean = segment_turns(
        _rows(
            "c",
            [
                ("user_msg", "Do A"),
                ("tool_call", None),
                ("tool_result", "ok"),
                ("user_msg", "Next"),
            ],
        )
    )
    assert trigger_flags(cut_off_clean[0]) == []
    assert candidates_from_turns(cut_off_clean) == []
    # Items written before closed_by was recorded derive it from what they do have.
    assert closed_by_of({"outcome_reason": "completed"}) == "outcome"
    assert closed_by_of({"outcome_reason": None, "next_user_event_id": "e"}) == "next_message"
    assert closed_by_of({"outcome_reason": None, "next_user_event_id": None}) == "session_end"
    assert closed_by_of({"closed_by": "next_message", "outcome_reason": "x"}) == "next_message"
    events = [
        FullEvent(e.event_id, e.kind, e.snippet or "", e.payload, e.parent_event_id)
        for e in turns[0].events
    ]
    nxt = FullEvent(turns[1].turn_id, "user_msg", "Now do B instead", None, None)
    proj = build_projection(cands[0], events, nxt)
    assert "closed_by=next_message outcome=none" in proj.text
    assert "NEXT MESSAGE (human): Now do B instead" in proj.text
    assert "NEXT MESSAGE" not in proj.for_drafter
    assert proj.text.startswith(proj.for_drafter)
    assert proj.for_drafter.rstrip().endswith("OUTCOME: none recorded")


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
    assert (
        "session_started_by=human closed_by=outcome outcome=completed tool_calls=2 "
        "tool_errors=2" in first.text
    )
    assert not any(e.event_id in first.text for e in events)
    # The drafter sees everything but the next message; the numbering is shared.
    assert "NEXT MESSAGE" not in first.for_drafter
    assert "[1] REQUEST (human): " in first.for_drafter
    assert first.text.startswith(first.for_drafter)
    assert drafter_rendering("no entries at all") == "no entries at all"
    assert drafter_rendering("[1] A\n[2] NEXT MESSAGE: x\nmore") == "[1] A"
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


def test_question_set_shape_and_run_thresholds():
    assert [q.id for q in TURN_QUESTIONS] == [
        "failed",
        "incomplete",
        "user_corrected",
        "env_confirmed",
        "category",
        "recurrence",
    ]
    assert [q.id for q in DRAFT_QUESTIONS] == [
        "grounded",
        "faithful",
        "rule_supported",
        "generalizable",
        "actionable",
        "enforced",
    ]
    assert BY_ID["category"].options()[-1] == "none"
    assert BY_ID["generalizable"].options() == ["0", "1", "2"]
    assert BY_ID["enforced"].options() == ["none", "harness", "project_docs", "tooling"]
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
    assert answers["incomplete"].value == "no"
    assert answers["user_corrected"].value == "yes"
    assert answers["category"].value == "tool_misuse"
    assert answers["recurrence"].value == "yes"
    assert answers["grounded"].value == "yes"
    assert answers["faithful"].value == "yes"
    assert answers["faithful"].confidence < answers["grounded"].confidence
    assert answers["actionable"].value == "yes"
    assert answers["enforced"].value == "none"
    clean = RulesBackend().decide(
        JudgeInput(item_id="j", state="", facts={"last_result_ok": True}, flags=("T0",)),
        TURN_QUESTIONS,
    )
    assert clean["failed"].value == "no"
    assert clean["category"].value == "none"
    cut_off = RulesBackend().decide(
        JudgeInput(
            item_id="k",
            state="",
            facts={"closed_by": "next_message", "caller": "agent:worker"},
            flags=("T3",),
            draft="x",
        ),
        ALL_QUESTIONS,
    )
    assert cut_off["incomplete"].value == "yes"
    assert cut_off["enforced"].value == "harness"
    assert normalize({"a": -1, "b": 0}, ["a", "b"]) == {"a": 0.5, "b": 0.5}


def test_make_backend_specs(monkeypatch):
    assert make_backend("rules").name == "rules"
    assert make_backend("readout:gemma4:26b").name == "readout:gemma4:26b"
    assert make_backend("llm:gemma4:26b").name == "llm:gemma4:26b"
    assert make_backend("llamacpp:gemma-4-26b").name == "llamacpp:gemma-4-26b"
    # One judge, told which server to ask.
    ollama = make_backend("llm:gemma4:26b")
    assert isinstance(ollama, ReasoningJudge)
    assert ollama.server == Ollama("gemma4:26b")
    # Router-mode ids carry slashes and colons; the spec keeps everything after the kind.
    router = make_backend("llamacpp:ggml-org/Qwen3.8-27B-GGUF:Q8_0")
    assert isinstance(router, ReasoningJudge)
    assert router.server == LlamaCpp(
        "ggml-org/Qwen3.8-27B-GGUF:Q8_0", "http://localhost:9931", split_thinking=True
    )
    assert make_backend("jev").name == "jev:typesafe/jev-1.14"
    assert make_backend("jev:typesafe/jev-1.12").name == "jev:typesafe/jev-1.12"
    for bad in ("readout", "llm", "llamacpp", "nope", "??"):
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


@pytest.mark.parametrize(
    ("question", "entry"),
    [
        ("failed", {"type": "noul", "noul": "likely"}),
        ("failed", {"type": "noul", "noul": 0.9, "confidence": "high"}),
        ("category", {"type": "choice", "probabilities": {"none": "most"}}),
        ("category", {"type": "choice", "probabilities": 3}),
        ("generalizable", {"type": "score", "score": "two"}),
        ("generalizable", {"type": "score", "probabilities": [0.1, None]}),
    ],
)
def test_jev_answer_in_another_shape_is_a_judge_error(question, entry):
    with pytest.raises(JudgeError) as info:
        parse_answers({"answers": {question: entry}}, [BY_ID[question]], latency_ms=1)
    assert str(info.value) == f"Jev answer for '{question}' is malformed"


def test_jev_usage_in_another_shape_is_a_judge_error():
    response = {"answers": {"failed": {"type": "noul", "noul": 0.9}}, "usage": "free"}
    with pytest.raises(JudgeError, match="usage"):
        parse_answers(response, [BY_ID["failed"]], latency_ms=1)


def test_jev_refuses_without_opt_in_and_key(monkeypatch):
    monkeypatch.delenv(REMOTE_ENV, raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    calls = []
    monkeypatch.setattr(
        endpoints, "post_json", lambda *a, **k: calls.append(a) or (JEV_RESPONSE, 1.0)
    )
    inp = JudgeInput(item_id="i", state="S", facts={}, flags=())
    with pytest.raises(ServerUnreachable, match=REMOTE_ENV):
        JevBackend().decide(inp, [BY_ID["failed"]])
    monkeypatch.setenv(REMOTE_ENV, "1")
    with pytest.raises(ServerUnreachable, match="OPENROUTER_API_KEY"):
        JevBackend().decide(inp, [BY_ID["failed"]])
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    answers = JevBackend().decide(inp, [BY_ID["failed"]])
    assert answers["failed"].p_true == pytest.approx(0.91)
    assert calls
    assert calls[0][0] == "https://openrouter.ai/api/alpha/decisions"
    assert not JevBackend().local


def test_llm_judge_truncated_json_is_judge_error_and_falls_back_without_thinking(monkeypatch):
    cut = '{"failed": {"answer": "yes", "probability": 0.9}, "category": {"answer": "wro'
    good = '{"failed": {"answer": "no", "probability": 0.8}}'
    calls: list[dict] = []

    def post(url, body, **k):
        calls.append(copy.deepcopy(body))  # `options` is widened in place between calls
        if body.get("think"):
            return {"message": {"content": cut}, "done_reason": "length"}, 100.0
        return {"message": {"content": good}, "done_reason": "stop", "model": "m:1"}, 10.0

    monkeypatch.setattr(endpoints, "post_json", post)
    inp = JudgeInput(item_id="i", state="S", facts={}, flags=())
    answers = ollama_judge("m").decide(inp, [BY_ID["failed"]])
    # Cut off while thinking: once more with room, then without thinking. The last retry must
    # say `think: false` outright; Ollama treats a missing flag as "think".
    assert [c.get("think", "absent") for c in calls] == [True, True, False]
    assert [c["options"]["num_predict"] for c in calls] == [4096, 12288, 4096]
    assert [c.get("format") for c in calls] == ["json", "json", None]
    assert answers["failed"].value == "no"
    assert answers["failed"].method == "llm_json_nothink"
    assert answers["failed"].model_snapshot == "m:1"
    assert answers["failed"].latency_ms == pytest.approx(210.0)  # all three attempts billed

    # A whole budget spent thinking: the attempt without thinking still runs.
    calls.clear()

    def all_thinking(url, body, **k):
        calls.append(dict(body))
        if body.get("think"):
            return {
                "message": {"content": "", "thinking": "x" * 15000},
                "done_reason": "length",
            }, 100.0
        return {"message": {"content": good}, "done_reason": "stop"}, 10.0

    monkeypatch.setattr(endpoints, "post_json", all_thinking)
    answers = ollama_judge("m").decide(inp, [BY_ID["failed"]])
    assert [c.get("think", "absent") for c in calls] == [True, True, False]
    assert answers["failed"].method == "llm_json_nothink"

    # A model that cannot think refuses the switch. It is asked once more with the switch
    # left out, and the refused request is billed too.
    calls.clear()

    def refuses_switch(url, body, **k):
        calls.append(dict(body))
        if "think" in body:
            said = '{"error":"\\"m\\" does not support thinking"}'
            raise endpoints.HttpStatusError(
                f"{url}: HTTP 400: {said}", 400, latency_ms=50.0, detail=said
            )
        return {"message": {"content": good}, "done_reason": "stop"}, 10.0

    monkeypatch.setattr(endpoints, "post_json", refuses_switch)
    answers = ollama_judge("m").decide(inp, [BY_ID["failed"]])
    assert [c.get("think", "absent") for c in calls] == [True, "absent"]
    assert [c.get("format", "absent") for c in calls] == ["json", "absent"]
    assert answers["failed"].method == "llm_json_nothink"
    assert answers["failed"].latency_ms == pytest.approx(60.0)

    # A reply that only points at a link holds no JSON. The last attempt still says off: the
    # word "http" in the error once made it leave the switch out, and the model thought again.
    calls.clear()

    def links(url, body, **k):
        calls.append(dict(body))
        if body.get("think"):
            return {"message": {"content": "See https://example.test/docs first."}}, 5.0
        return {"message": {"content": good}, "done_reason": "stop"}, 10.0

    monkeypatch.setattr(endpoints, "post_json", links)
    answers = ollama_judge("m").decide(inp, [BY_ID["failed"]])
    assert [c.get("think", "absent") for c in calls] == [True, False]
    assert answers["failed"].method == "llm_json_nothink"

    # A missing model is not a refused switch, whatever its name says: one request.
    calls.clear()

    def missing(url, body, **k):
        calls.append(dict(body))
        said = "model 'qwen3-thinking:8b' not found"
        raise endpoints.HttpStatusError(f"{url}: HTTP 404: {said}", 404, detail=said)

    monkeypatch.setattr(endpoints, "post_json", missing)
    with pytest.raises(JudgeError, match="not found"):
        ollama_judge("qwen3-thinking:8b").decide(inp, [BY_ID["failed"]])
    assert len(calls) == 1

    # Without thinking there is nothing to fall back to; the error surfaces for the caller.
    calls.clear()
    monkeypatch.setattr(
        endpoints,
        "post_json",
        lambda u, b, **k: calls.append(dict(b)) or ({"message": {"content": ""}}, 1.0),
    )
    with pytest.raises(JudgeError, match="no JSON"):
        ollama_judge("m", think=False).decide(inp, [BY_ID["failed"]])
    assert [c.get("think", "absent") for c in calls] == [False]


def test_llamacpp_judge_speaks_openai_chat_and_falls_back_without_thinking(monkeypatch):
    cut = '{"failed": {"answer": "yes", "probability": 0.9}, "category": {"answer": "wro'
    good = '{"failed": {"answer": "no", "probability": 0.8}}'
    calls: list[tuple[str, dict]] = []

    def post(url, body, **k):
        calls.append((url, dict(body)))
        if body["chat_template_kwargs"]["enable_thinking"]:
            return {
                "model": "gemma-4-26b-it-Q4_K_M.gguf",
                "choices": [
                    {
                        "finish_reason": "length",
                        "message": {"content": cut, "reasoning_content": "Let me think..."},
                    }
                ],
            }, 100.0
        return {
            "model": "gemma-4-26b-it-Q4_K_M.gguf",
            "choices": [{"finish_reason": "stop", "message": {"content": f"Sure: {good}"}}],
        }, 10.0

    monkeypatch.setattr(endpoints, "post_json", post)
    inp = JudgeInput(item_id="i", state="S", facts={}, flags=())
    backend = llamacpp_judge("gemma-4-26b", host="http://llama.test:8080")
    assert backend.name == "llamacpp:gemma-4-26b"
    assert backend.local
    answers = backend.decide(inp, [BY_ID["failed"]])

    assert [u for u, _ in calls] == ["http://llama.test:8080/v1/chat/completions"] * 3
    first, wider, second = (b for _, b in calls)
    # OpenAI field names, not Ollama's: no `options`, `format`, or `think` keys.
    assert not {"options", "format", "think"} & set(first)
    assert first["model"] == "gemma-4-26b"
    assert first["temperature"] == 0
    assert first["max_tokens"] == 4096
    assert first["response_format"] == {"type": "json_object"}
    assert first["reasoning_format"] == "deepseek"
    assert first["chat_template_kwargs"] == {"enable_thinking": True}
    assert "reasoning_effort" not in first
    assert [m["role"] for m in first["messages"]] == ["system", "user"]
    assert "DRAFT LESSON" not in first["messages"][1]["content"]
    # Cut off while thinking: the same request once more with room, still thinking.
    assert wider["max_tokens"] == 12288
    assert wider["chat_template_kwargs"] == {"enable_thinking": True}
    assert wider["response_format"] == {"type": "json_object"}
    # The last retry turns thinking off both ways, drops the grammar, and goes back to the
    # normal budget.
    assert second["chat_template_kwargs"] == {"enable_thinking": False}
    assert second["reasoning_effort"] == "none"
    assert second["max_tokens"] == 4096
    assert "response_format" not in second
    assert answers["failed"].value == "no"
    assert answers["failed"].p_true == pytest.approx(0.2)
    assert answers["failed"].method == "llm_json_nothink"
    assert answers["failed"].model_snapshot == "gemma-4-26b-it-Q4_K_M.gguf"
    assert answers["failed"].latency_ms == pytest.approx(210.0)  # all three attempts billed

    # With thinking off from the start there is one call and nothing to fall back to.
    calls.clear()
    monkeypatch.setattr(
        endpoints,
        "post_json",
        lambda u, b, **k: calls.append((u, dict(b))) or ({"choices": [{"message": {}}]}, 1.0),
    )
    with pytest.raises(JudgeError, match="no JSON"):
        llamacpp_judge("m", think=False).decide(inp, [BY_ID["failed"]])
    assert len(calls) == 1
    assert calls[0][1]["reasoning_effort"] == "none"

    # A server error that is not about thinking or JSON is not retried.
    calls.clear()

    def loading(u, b, **k):
        calls.append((u, dict(b)))
        raise endpoints.HttpStatusError(f"{u}: HTTP 503: Loading model", 503)

    monkeypatch.setattr(endpoints, "post_json", loading)
    with pytest.raises(JudgeError, match="Loading model"):
        llamacpp_judge("m").decide(inp, [BY_ID["failed"]])
    assert len(calls) == 1

    # The response's model field is missing: the configured name stands in as the snapshot.
    monkeypatch.setattr(
        endpoints,
        "post_json",
        lambda u, b, **k: ({"choices": [{"message": {"content": good}}]}, 4.0),
    )
    answers = llamacpp_judge("alias").decide(inp, [BY_ID["failed"]])
    assert answers["failed"].model_snapshot == "alias"
    assert answers["failed"].method == "llm_json"

    # A router that does not know the id: the error names the ids it does serve, once.
    gets: list[str] = []

    def not_found(u, b, **k):
        raise endpoints.HttpStatusError(
            f'{u}: HTTP 400: {{"error":{{"message":"model \'typo\' not found"}}}}', 400
        )

    monkeypatch.setattr(endpoints, "post_json", not_found)
    monkeypatch.setattr(
        endpoints,
        "get_json",
        lambda u, **k: gets.append(u) or {"data": [{"id": "org/b:Q8_0"}, {"id": "org/a:Q4"}]},
    )
    with pytest.raises(JudgeError) as excinfo:
        llamacpp_judge("typo", host="http://r:9931").decide(inp, [BY_ID["failed"]])
    assert "model 'typo' not found" in str(excinfo.value)
    assert str(excinfo.value).endswith(". Models this server knows: org/a:Q4, org/b:Q8_0")
    assert gets == ["http://r:9931/v1/models"]
    # If the listing itself fails, the original error stands.
    monkeypatch.setattr(
        endpoints, "get_json", lambda u, **k: (_ for _ in ()).throw(endpoints.EgressError("down"))
    )
    with pytest.raises(JudgeError) as excinfo:
        llamacpp_judge("typo").decide(inp, [BY_ID["failed"]])
    assert str(excinfo.value).endswith('not found"}}')
    assert "Models this server knows" not in str(excinfo.value)


def test_judge_cut_off_while_thinking_gets_room_before_losing_thinking(monkeypatch):
    """gpt-oss-20b on exp3 item 5dedfe18: greedy reasoning looped past the 4,096-token budget
    and finished at about 6,000 tokens when given room. Its template has no thinking switch,
    so the no-think retry was the same request again and failed the same way."""
    from agent_memory.judge.backends.llm import MAX_JUDGE_TOKENS, RETRY_JUDGE_TOKENS

    good = '{"failed": {"answer": "yes", "probability": 0.95}}'
    bodies: list[dict] = []

    def gpt_oss(url, body, **k):
        bodies.append(dict(body))
        if body["max_tokens"] == MAX_JUDGE_TOKENS:
            # Budget spent in the analysis channel, nothing in the answer.
            message = {"content": "", "reasoning_content": "The next message is the same? " * 661}
            return {
                "model": "gpt-oss-20b",
                "choices": [{"finish_reason": "length", "message": message}],
            }, 100.0
        message = {"content": good, "reasoning_content": "x" * 28206}
        return {
            "model": "gpt-oss-20b",
            "choices": [{"finish_reason": "stop", "message": message}],
        }, 200.0

    monkeypatch.setattr(endpoints, "post_json", gpt_oss)
    inp = JudgeInput(item_id="i", state="S", facts={}, flags=())
    answers = llamacpp_judge("ggml-org/gpt-oss-20b-GGUF:MXFP4").decide(inp, [BY_ID["failed"]])
    assert [b["max_tokens"] for b in bodies] == [MAX_JUDGE_TOKENS, RETRY_JUDGE_TOKENS]
    # Still the thinking pass: same switch, same grammar, no reasoning_effort override.
    assert all(b["chat_template_kwargs"] == {"enable_thinking": True} for b in bodies)
    assert all(b["response_format"] == {"type": "json_object"} for b in bodies)
    assert not any("reasoning_effort" in b for b in bodies)
    assert answers["failed"].value == "yes"
    assert answers["failed"].method == "llm_json"  # reasoning was kept, not dropped
    assert answers["failed"].latency_ms == pytest.approx(300.0)
    assert answers["failed"].model_snapshot == "gpt-oss-20b"

    # Looping past the wider budget too: the no-think pass is still tried (it works for
    # templates with a switch), and when that is cut off as well the error says how far the
    # last attempt got. Three calls, no more.
    bodies.clear()

    def loops(url, body, **k):
        bodies.append(dict(body))
        message = {"content": "", "reasoning_content": "again? " * 2832}
        return {"choices": [{"finish_reason": "length", "message": message}]}, 50.0

    monkeypatch.setattr(endpoints, "post_json", loops)
    with pytest.raises(JudgeError, match=r"after 19824 chars of thinking; JSON incomplete: ''"):
        llamacpp_judge("m").decide(inp, [BY_ID["failed"]])
    assert [b["max_tokens"] for b in bodies] == [
        MAX_JUDGE_TOKENS,
        RETRY_JUDGE_TOKENS,
        MAX_JUDGE_TOKENS,
    ]
    assert [b["chat_template_kwargs"]["enable_thinking"] for b in bodies] == [True, True, False]

    # The Ollama judge widens num_predict the same way and keeps thinking on for it.
    bodies.clear()

    def ollama(url, body, **k):
        bodies.append(copy.deepcopy(body))
        if body["options"]["num_predict"] == MAX_JUDGE_TOKENS:
            return {
                "message": {"content": "", "thinking": "y" * 15000},
                "done_reason": "length",
            }, 100.0
        return {
            "message": {"content": good, "thinking": "y" * 20000},
            "done_reason": "stop",
            "model": "g",
        }, 200.0

    monkeypatch.setattr(endpoints, "post_json", ollama)
    answers = ollama_judge("g").decide(inp, [BY_ID["failed"]])
    assert [b["options"]["num_predict"] for b in bodies] == [MAX_JUDGE_TOKENS, RETRY_JUDGE_TOKENS]
    assert [b["think"] for b in bodies] == [True, True]
    assert [b["format"] for b in bodies] == ["json", "json"]
    assert answers["failed"].method == "llm_json"
    assert answers["failed"].latency_ms == pytest.approx(300.0)


def test_llamacpp_judge_end_to_end_over_http():
    """Real HTTP through `post_json` against a stand-in for llama-server, so the URL path,
    content type, and OpenAI response shape are exercised, not just the mocked seam."""
    import json
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    received: list[tuple[str, str, dict]] = []
    reply = {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "model": "stand-in",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "reasoning_content": "The tests failed and the agent said done.",
                    "content": (
                        '{"failed": {"answer": "yes", "probability": 0.95}, '
                        '"category": {"answer": "tool_misuse", "probability": 0.6}}'
                    ),
                },
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
    }

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length))
            received.append((self.path, self.headers.get("Content-Type", ""), body))
            payload = json.dumps(reply).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format: str, *args: Any) -> None:
            return None

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host = f"http://127.0.0.1:{server.server_address[1]}"
        backend = llamacpp_judge("stand-in", host=host)
        inp = JudgeInput(item_id="i", state="S", facts={}, flags=(), draft="Run tests first.")
        answers = backend.decide(inp, [BY_ID["failed"], BY_ID["category"]])
    finally:
        server.shutdown()
        server.server_close()

    assert len(received) == 1
    path, content_type, body = received[0]
    assert path == "/v1/chat/completions"
    assert content_type == "application/json"
    assert body["response_format"] == {"type": "json_object"}
    assert (
        "EVIDENCE\nS\n\nDRAFT LESSON:\nRun tests first.\n\nQUESTIONS"
        in body["messages"][1]["content"]
    )
    assert answers["failed"].value == "yes"
    assert answers["failed"].p_true == pytest.approx(0.95)
    assert answers["category"].value == "tool_misuse"
    assert answers["category"].confidence == pytest.approx(0.6)
    assert all(a.method == "llm_json" for a in answers.values())
    assert all(a.model_snapshot == "stand-in" for a in answers.values())
    assert all(a.latency_ms > 0 for a in answers.values())


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
    monkeypatch.setattr(endpoints, "post_json", lambda url, body, **k: next(responses))
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

    monkeypatch.setattr(endpoints, "post_json", no_logprobs)
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
        '"grounded": {"answer": "maybe", "probability": 0.5}}\n```'
    )
    data = model_call.json_object(text)
    questions = [
        BY_ID["failed"],
        BY_ID["category"],
        BY_ID["generalizable"],
        BY_ID["grounded"],
        BY_ID["actionable"],
    ]
    answers = answers_from_judge_json(data, questions, latency_ms=1.0, model_snapshot="m")
    check_contract(answers, questions)
    assert answers["failed"].value == "yes"
    assert answers["failed"].p_true == pytest.approx(0.8)
    assert answers["category"].confidence == pytest.approx(0.6)
    assert answers["generalizable"].value == pytest.approx(0.7 * 1 + 0.15 * 0 + 0.15 * 2)
    assert answers["grounded"].method == "llm_json_unmatched"
    assert answers["actionable"].method == "llm_json_missing"


def lesson_reply(rule="Stop the server first.", *, cited=(3, 5), claims=None):
    return {
        "lesson": {
            "rule": rule,
            "kind": "project_fact",
            "claims": claims
            if claims is not None
            else [{"text": "The database was locked.", "cited": list(cited)}],
        }
    }


def test_drafter_parsing_and_spec(monkeypatch):
    lesson = parse_lesson(lesson_reply("Stop  the server\nfirst.", cited=("[3]", 5, "x", 3)))
    assert lesson is not None
    assert lesson.rule == "Stop the server first."
    assert lesson.claims[0].cited == (3, 5)
    # A declined draft is an explicit null; a missing or empty lesson is a drafter fault.
    assert parse_lesson({"lesson": None}) is None
    with pytest.raises(JudgeError, match="empty"):
        parse_lesson(lesson_reply(""))
    with pytest.raises(JudgeError, match="lesson: Field required"):
        parse_lesson({})
    assert draft_hash(None) is None
    assert draft_hash("x") == draft_hash("x") != draft_hash("y")
    assert make_drafter("ollama:gemma4:26b").name == "ollama:gemma4:26b"
    local = make_drafter("llamacpp:ggml-org/gpt-oss-20b-GGUF:MXFP4")
    assert (local.name, local.local) == ("llamacpp:ggml-org/gpt-oss-20b-GGUF:MXFP4", True)
    assert local.server == LlamaCpp("ggml-org/gpt-oss-20b-GGUF:MXFP4", timeout=180)
    remote = make_drafter("openrouter:openai/gpt-5")
    assert (remote.name, remote.local) == ("openrouter:openai/gpt-5", False)
    for bad in ("openai:gpt", "ollama", "openrouter:"):
        with pytest.raises(JudgeError):
            make_drafter(bad)

    # The OpenAI-compatible drafter: request shape, decline, cost from usage, and the opt-in.
    bodies = []

    def fake_post(url, body, headers=None, **kwargs):
        bodies.append((url, body, headers))
        content = (
            '{"lesson": null}'
            if len(bodies) == 2
            else json.dumps(lesson_reply("Run the tests before claiming done.", cited=(2,)))
        )
        return {
            "model": "served-model",
            "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
            "usage": {"cost": 0.0012},
        }, 55.0

    monkeypatch.setattr(endpoints, "post_json", fake_post)
    draft = local.draft("i1", "[1] REQUEST: x\n[2] TOOL RESULT (ok): 3 passed")
    assert (draft.lesson, draft.cited, draft.declined) == (
        "Rule: Run the tests before claiming done.\nKind: project_fact\nClaims:\n"
        "1. The database was locked. [2]",
        [2],
        False,
    )
    # A local server charges nothing, whatever its usage block says.
    assert draft.cost_usd == 0.0
    assert draft.model_snapshot == "served-model"
    assert draft.to_dict()["declined"] is False
    url, body, headers = bodies[0]
    assert url == "http://localhost:9931/v1/chat/completions"
    assert headers == {}
    assert body["max_tokens"] == 700
    assert body["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "lesson", "schema": LESSON_SCHEMA},
    }
    assert draft.schema_enforced
    assert draft.rule == "Run the tests before claiming done."
    assert draft.to_dict()["claims"] == [{"text": "The database was locked.", "cited": [2]}]
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert body["reasoning_effort"] == "none"
    assert "closed_by" in body["messages"][0]["content"]
    assert '{"lesson": null}' in body["messages"][0]["content"]
    declined = local.draft("i2", "[1] REQUEST: y")
    assert declined.declined
    assert declined.lesson is None
    assert declined.to_dict()["declined"] is True
    monkeypatch.delenv(REMOTE_ENV, raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(JudgeError, match=REMOTE_ENV):
        remote.draft("i3", "state")
    monkeypatch.setenv(REMOTE_ENV, "1")
    with pytest.raises(JudgeError, match="OPENROUTER_API_KEY"):
        remote.draft("i3", "state")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    paid = remote.draft("i3", "state")
    assert paid.cost_usd == pytest.approx(0.0012)
    url, body, headers = bodies[-1]
    assert url == "https://openrouter.ai/api/v1/chat/completions"
    assert headers == {"Authorization": "Bearer sk-test"}
    assert body["usage"] == {"include": True}
    assert body["response_format"]["json_schema"]["strict"] is True
    assert body["response_format"]["json_schema"]["schema"] == LESSON_SCHEMA


def test_drafter_prompt_leads_with_the_decline():
    # Given the lesson template first, both local and hosted models filled it for every turn.
    # The decline comes first in both messages, and a harness refusal is named as not a lesson
    # in one sentence: a longer harness paragraph made the local model decline nearly everything.
    assert SYSTEM.index('{"lesson": null}') < SYSTEM.index("1 to 4 claims")
    assert SYSTEM.index("Most turns hold nothing worth keeping") < SYSTEM.index("Write a lesson")
    harness = "A refusal the harness itself issued"
    assert SYSTEM.count("harness") == 1
    assert SYSTEM.index(harness) < SYSTEM.index('{"lesson": null}')
    assert "plan is approved, is not a lesson" in SYSTEM
    user = drafter_prompt("[1] REQUEST: x")
    assert user.startswith("EVIDENCE\n[1] REQUEST: x\n\n")
    assert user.index('{"lesson": null}') < user.index('"rule"')
    assert user.index("tool_quirk: how a tool") > user.index('"claims"')


def test_drafter_retries_a_cut_off_draft_with_more_room(monkeypatch):
    from agent_memory.judge.drafter import MAX_DRAFT_TOKENS, RETRY_DRAFT_TOKENS

    monkeypatch.setenv(REMOTE_ENV, "1")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    full = json.dumps(
        lesson_reply("Check the script with ShellCheck before running it.", cited=(4,))
    )
    partial = full[:40]
    bodies = []

    def openai_post(url, body, headers=None, **kwargs):
        bodies.append(dict(body))
        cut = len(bodies) == 1
        return {
            "model": "served-model",
            "choices": [
                {
                    "message": {"content": partial if cut else full, "reasoning": "hmm" * 50},
                    "finish_reason": "length" if cut else "stop",
                }
            ],
            "usage": {"cost": 0.001 if cut else 0.002},
        }, 100.0

    monkeypatch.setattr(endpoints, "post_json", openai_post)
    hosted = make_drafter("openrouter:m")
    draft = hosted.draft("i1", "[4] TOOL RESULT: shellcheck ok")
    assert draft.rule == "Check the script with ShellCheck before running it."
    assert [b["max_tokens"] for b in bodies] == [MAX_DRAFT_TOKENS, RETRY_DRAFT_TOKENS]
    assert bodies[0]["seed"] == bodies[1]["seed"]
    assert draft.latency_ms == 200.0
    assert draft.cost_usd == pytest.approx(0.003)  # the cut-off attempt is billed too

    # Cut off twice: a drafter fault that names where the budget went.
    def always_cut(url, body, headers=None, **kwargs):
        return {
            "choices": [
                {
                    "message": {"content": partial, "reasoning": "x" * 300},
                    "finish_reason": "length",
                }
            ]
        }, 10.0

    monkeypatch.setattr(endpoints, "post_json", always_cut)
    with pytest.raises(JudgeError, match=r"drafter ran out of tokens after 300 chars of thinking"):
        hosted.draft("i2", "state")

    # llama-server calls thinking reasoning_content; the error counts that too.
    def llama_cut(url, body, **kwargs):
        message = {"content": partial, "reasoning_content": "x" * 120}
        return {"choices": [{"message": message, "finish_reason": "length"}]}, 10.0

    monkeypatch.setattr(endpoints, "post_json", llama_cut)
    with pytest.raises(JudgeError, match=r"drafter ran out of tokens after 120 chars of thinking"):
        make_drafter("llamacpp:m").draft("i2", "state")

    # The Ollama drafter reads done_reason and widens num_predict the same way.
    bodies.clear()

    def ollama_post(url, body, **kwargs):
        bodies.append({"num_predict": body["options"]["num_predict"]})
        cut = len(bodies) == 1
        return {
            "model": "m",
            "message": {"content": partial if cut else full},
            "done_reason": "length" if cut else "stop",
        }, 50.0

    monkeypatch.setattr(endpoints, "post_json", ollama_post)
    draft = make_drafter("ollama:m").draft("i3", "state")
    assert draft.cited == [4]
    assert [b["num_predict"] for b in bodies] == [MAX_DRAFT_TOKENS, RETRY_DRAFT_TOKENS]
    assert draft.latency_ms == 100.0
    assert draft.schema_enforced


@pytest.mark.parametrize(
    ("change", "location"),
    [
        ({"kind": "PRIVATE_BAD_KIND"}, "lesson.kind"),
        ({"claims": []}, "lesson.claims"),
        ({"claims": [{"text": "PRIVATE_TEXT", "cited": []}] * 5}, "lesson.claims"),
        ({"rule": " \n\t"}, "lesson.rule"),
        ({"claims": [{"text": " ", "cited": []}]}, "lesson.claims.0.text"),
        ({"PRIVATE_KEY": "PRIVATE_EXTRA"}, "lesson.[unexpected key]: Extra inputs"),
        (
            {"claims": [{"text": "PRIVATE_TEXT", "cited": [], "PRIVATE_KEY": 1}]},
            "lesson.claims.0.[unexpected key]: Extra inputs",
        ),
    ],
)
def test_structured_reply_rejects_invalid_content_without_echoing(change, location):
    data = lesson_reply("PRIVATE_RULE")
    data["lesson"].update(change)
    with pytest.raises(JudgeError, match="drafter reply rejected:") as error:
        parse_lesson(data)
    assert location in str(error.value)
    assert "PRIVATE_" not in str(error.value)


def test_structured_reply_coercion_bounds_declines_and_frozen_models():
    assert parse_lesson({"lesson": None}) is None
    for data in ({"lesson": None, "extra": "PRIVATE"}, {"lesson": "PRIVATE"}):
        with pytest.raises(JudgeError) as error:
            parse_lesson(data)
        assert "PRIVATE" not in str(error.value)
    lesson = parse_lesson(
        lesson_reply(
            " \tStop  the\nserver. ",
            claims=[
                {"text": " Too  much\ntext " + "x" * 400, "cited": ["[3]", 3, "x", -1, 0, 2, None]},
                {"text": "Uncited.", "cited": []},
                {"text": "Single.", "cited": "[7]"},
                {"text": "Junk.", "cited": {"junk": 1}},
            ],
        )
    )
    assert lesson is not None
    assert lesson.rule == "Stop the server."
    assert len(lesson.claims[0].text) == MAX_TEXT_CHARS
    assert lesson.claims[0].text.startswith("Too much text ")
    assert [claim.cited for claim in lesson.claims] == [(3, 2), (), (7,), ()]
    assert Lesson.model_fields["kind"].annotation is not None
    assert set(LESSON_KINDS) == set(LESSON_SCHEMA["$defs"]["Lesson"]["properties"]["kind"]["enum"])
    for model, field in (
        (lesson, "rule"),
        (lesson.claims[0], "text"),
        (DraftReply(lesson=lesson), "lesson"),
    ):
        with pytest.raises(ValueError, match="frozen"):
            setattr(model, field, None)


def test_sampler_schema_keyword_set_is_grammar_safe():
    keywords = set()

    def visit(schema):
        keywords.update(schema)
        for key, value in schema.items():
            if key in ("properties", "$defs"):
                for child in value.values():
                    visit(child)
            elif key == "items":
                visit(value)
            elif key == "anyOf":
                for child in value:
                    visit(child)

    visit(LESSON_SCHEMA)
    assert keywords == {
        "type",
        "properties",
        "required",
        "additionalProperties",
        "enum",
        "anyOf",
        "items",
        "minItems",
        "maxItems",
        "$defs",
        "$ref",
    }
    assert LESSON_SCHEMA["$defs"]["Lesson"]["properties"]["claims"]["minItems"] == 1
    assert LESSON_SCHEMA["$defs"]["Lesson"]["properties"]["claims"]["maxItems"] == 4
    for name in ("Claim", "Lesson"):
        assert LESSON_SCHEMA["$defs"][name]["additionalProperties"] is False
    assert LESSON_SCHEMA["additionalProperties"] is False


def structured_draft(item_id="i", *, claims=None, drafter="ollama:m"):
    lesson = parse_lesson(lesson_reply(claims=claims))
    assert lesson is not None
    return Draft.from_lesson(item_id, lesson, drafter=drafter, model_snapshot="m", latency_ms=12.0)


def test_render_lesson_and_hash_cover_each_structured_field():
    lesson = Lesson(
        rule="Stop `mem serve` before running migrations on the same home.",
        kind="project_fact",
        claims=[
            Claim(text='The migration failed with "database is locked".', cited=(7,)),
            Claim(text="`mem serve` was still running on that home.", cited=(2, 4)),
            Claim(text="An uncited claim.", cited=()),
        ],
    )
    expected = (
        "Rule: Stop `mem serve` before running migrations on the same home.\n"
        "Kind: project_fact\nClaims:\n"
        '1. The migration failed with "database is locked". [7]\n'
        "2. `mem serve` was still running on that home. [2, 4]\n"
        "3. An uncited claim. [no citation]"
    )
    assert render_lesson(lesson) == expected
    assert draft_hash(expected) == hashlib.sha256(expected.encode()).hexdigest()
    data = lesson.model_dump(mode="json")
    assert lesson_from_row({**data, "lesson": expected}) == lesson
    assert lesson_from_row({"lesson": "v2"}) is None
    assert lesson_from_row({"lesson": None, "claims": [], "declined": True}) is None
    assert entry_numbers("Header\n[2] hello\ncontinued\n[7] other\n") == {2, 7}
    for key, value in (
        ("rule", "Different."),
        ("kind", "process"),
        ("claims", [Claim(text="Other.", cited=(1,))]),
    ):
        changed = Lesson.model_validate({**data, key: value})
        assert draft_hash(render_lesson(changed)) != draft_hash(expected)
    changed = Lesson.model_validate(
        {**data, "claims": [{**data["claims"][0], "cited": [8]}, *data["claims"][1:]]}
    )
    assert draft_hash(render_lesson(changed)) != draft_hash(expected)


@pytest.mark.parametrize("code", [400, 422])
def test_openrouter_schema_fallback_once_and_validates_reply(monkeypatch, code):
    monkeypatch.setenv(REMOTE_ENV, "1")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    bodies = []

    def post(url, body, **kwargs):
        bodies.append(copy.deepcopy(body))
        if len(bodies) == 1:
            raise endpoints.HttpStatusError("schema unsupported", code, latency_ms=7.0)
        return {"choices": [{"message": {"content": json.dumps(lesson_reply())}}]}, 20.0

    monkeypatch.setattr(endpoints, "post_json", post)
    draft = make_drafter("openrouter:m").draft("i", "state")
    assert bodies[0]["response_format"] == {
        "type": "json_schema",
        "json_schema": {
            "name": "lesson",
            "schema": LESSON_SCHEMA,
            "strict": True,
        },
    }
    assert bodies[1]["response_format"] == {"type": "json_object"}
    assert len(bodies) == 2
    assert draft.schema_enforced is False
    assert draft.to_dict()["schema_enforced"] is False
    assert draft.to_dict()["drafter_version"] == "drafter_v3"
    assert draft.cited == [3, 5]
    assert draft.latency_ms == 27.0

    def rejected(url, body, **kwargs):
        bodies.append(copy.deepcopy(body))
        if body["response_format"]["type"] == "json_schema":
            raise endpoints.HttpStatusError("unsupported", code)
        return {"choices": [{"message": {"content": '{"lesson":"PRIVATE_INVALID"}'}}]}, 10.0

    monkeypatch.setattr(endpoints, "post_json", rejected)
    with pytest.raises(JudgeError, match="drafter reply rejected") as error:
        make_drafter("openrouter:m").draft("i", "state")
    assert "PRIVATE_INVALID" not in str(error.value)


@pytest.mark.parametrize(
    ("spec", "code"),
    [
        ("openrouter:m", 500),
        ("openrouter:m", 503),
        ("openrouter:m", 401),
        ("openrouter:m", 402),
        ("openrouter:m", 429),
        ("llamacpp:m", 400),
        ("openrouter:m", 400),
    ],
)
def test_schema_fallback_does_not_swallow_server_or_repeated_errors(monkeypatch, spec, code):
    monkeypatch.setenv(REMOTE_ENV, "1")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    calls = []

    def post(*args, **kwargs):
        calls.append(1)
        raise endpoints.HttpStatusError("rejected", code)

    monkeypatch.setattr(endpoints, "post_json", post)
    with pytest.raises(JudgeError, match="rejected"):
        make_drafter(spec).draft("i", "state")
    assert len(calls) == (2 if spec == "openrouter:m" and code == 400 else 1)


def test_claim_questions_aggregation_and_rules():
    lesson = parse_lesson(
        lesson_reply(
            claims=[
                {"text": "The migration database was locked.", "cited": [2]},
                {"text": "The migration database was locked.", "cited": [1]},
                {"text": "The migration database was locked.", "cited": []},
            ]
        )
    )
    assert lesson is not None
    questions = draft_questions_for(lesson)
    assert [q.id for q in questions] == [
        "grounded.1",
        "faithful.1",
        "grounded.2",
        "faithful.2",
        "grounded.3",
        "faithful.3",
        "rule_supported",
        "generalizable",
        "actionable",
        "enforced",
    ]
    assert (
        "Claim 3: The migration database was locked. (cites nothing)" in questions[4].instructions
    )
    assert draft_questions_for(None) is DRAFT_QUESTIONS
    assert (
        draft_questions_for(Lesson.model_construct(rule="r", kind="process", claims=[]))
        is DRAFT_QUESTIONS
    )
    assert "Every claim" not in " ".join(questions[0].option_descriptions())
    assert "This claim" in " ".join(questions[1].option_descriptions())
    inp = JudgeInput(
        item_id="i",
        state="[1] REQUEST: Read a file.\n[2] ERROR: migration database was locked.",
        facts={},
        draft=render_lesson(lesson),
        claims=tuple(lesson.claims),
    )
    assert "\nDRAFT LESSON:\nRule: " in inp.state_with_draft()
    answers = RulesBackend().decide(inp, questions)
    check_contract(answers, questions)
    assert [answers[f"grounded.{i}"].p_true for i in range(1, 4)] == [0.8, 0.2, 0.1]
    assert [answers[f"faithful.{i}"].p_true for i in range(1, 4)] == [0.65, 0.35, 0.1]
    # The whole-draft gates of a claimed draft are derived when it is judged, not asked.
    assert not {"grounded", "faithful"} & set(answers)


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
    child = store.live_day("2026-11-01")
    assert child.root == store.live_dir / "2026-11-01"
    assert store.purge_projections() == 1
    with pytest.raises(JudgeError, match="no projection"):
        store.read_projection("item/with/slash")


def _verdict(
    item_id: str,
    i: int,
    failed: bool,
    backend: str,
    p_failed: float,
    conf: float,
    latency: float,
    drafter: str | None = "ollama:m",
    *,
    v1: bool = False,
) -> dict:
    def ans(value, probs, confidence):
        return {
            "value": value,
            "probabilities": probs,
            "confidence": confidence,
            "method": "x",
            "latency_ms": latency / 5,
            "cost_usd": 0.0,
            "model_snapshot": backend,
        }

    yes_if_failed = ans(
        "yes" if failed else "no",
        {"yes": 0.9 if failed else 0.1, "no": 0.1 if failed else 0.9},
        0.9,
    )
    answers = {
        "failed": ans(
            "yes" if p_failed >= 0.5 else "no", {"yes": p_failed, "no": 1 - p_failed}, conf
        ),
        "recurrence": ans("yes", {"yes": 0.8, "no": 0.2}, 0.8),
        "category": ans("tool_misuse" if failed else "none", {}, 0.7),
        "generalizable": ans(1.0 if failed else 0.0, {"0": 0.2, "1": 0.6, "2": 0.2}, 0.6),
    }
    if v1:
        answers["supported"] = yes_if_failed
    else:
        answers["incomplete"] = ans("no", {"yes": 0.1, "no": 0.9}, 0.9)
        answers["grounded"] = yes_if_failed
        answers["faithful"] = yes_if_failed
        answers["rule_supported"] = yes_if_failed
        answers["enforced"] = ans(
            "none", {"none": 0.7, "harness": 0.1, "project_docs": 0.1, "tooling": 0.1}, 0.7
        )
    row: dict[str, Any] = {
        "run_id": "syn",
        "item_id": item_id,
        "backend": backend,
        "projection_hash": f"h{i}",
        "questions_version": "questions_v1" if v1 else QUESTIONS_VERSION,
        "latency_ms": latency,
        "cost_usd": 0.0,
        "answers": answers,
        "action": "discard",
        "reasons": [],
        "thresholds": {},
    }
    if not v1:
        row["drafter"] = drafter
        row["has_draft"] = drafter is not None
        row["declined"] = False
        row["draft_hash"] = draft_hash(f"Lesson {i}.") if drafter else None
    return row


def synthetic_run(tmp_path: Path, *, n: int = 20, v1: bool = False) -> RunStore:
    """A current run with legacy-shaped drafts and labels, exercising the draft-label fallback.
    `v1` builds an exp2-shaped run: no drafter on verdicts,
    `supported`/`generalizable`/`enforced_by` on the item."""
    cfg = Config(home=tmp_path / "exp")
    store = RunStore(cfg, "syn")
    store.create(
        {"questions_version": "questions_v1" if v1 else QUESTIONS_VERSION, "thresholds": {}}
    )
    items, labels, drafts, rules, llm = [], [], [], [], []
    for i in range(n):
        item_id = f"{i:08d}-0000-0000-0000-000000000000"
        failed = i % 2 == 1
        stratum = "T3" if failed else "T0"
        worth = failed and i % 3 == 0
        store.write_projection(item_id, f"PROJECTION {i} with SENTINEL_STATE_TEXT")
        item: dict[str, Any] = {
            "item_id": item_id,
            "project": "p" if i % 4 < 2 else "q",
            "stratum": stratum,
            "flags": [stratum],
            "facts": {"tool_count": 1},
            "turn_event_id": f"t{i}",
            "outcome_reason": "completed" if (v1 or i % 5) else None,
            "next_user_event_id": f"n{i}" if i % 2 else None,
            "split": "calibration" if i < 12 else "held_out",
            "projection_hash": f"h{i}",
        }
        if not v1:
            closed_by = closed_by_of(item)
            item["closed_by"] = closed_by
            item["facts"]["closed_by"] = closed_by
        items.append(item)
        if v1:
            labels.append(
                {
                    "item_id": item_id,
                    "failed": failed,
                    "category": "tool_misuse" if failed else "none",
                    "worth_remembering": worth,
                    "supported": failed,
                    "generalizable": 1 if failed else 0,
                    "enforced_by": "none",
                }
            )
        else:
            drafts.append(
                {
                    "item_id": item_id,
                    "lesson": f"Lesson {i}.",
                    "cited": [1],
                    "drafter": "ollama:m",
                    "model_snapshot": "m",
                    "latency_ms": 100.0,
                    "cost_usd": 0.0,
                    "declined": False,
                }
            )
            labels.append(
                {
                    "item_id": item_id,
                    "failed": failed,
                    "incomplete": False,
                    "category": "tool_misuse" if failed else None,
                    "worth_remembering": worth,
                    "lesson_kind": "process" if worth else None,
                    "error_intended": None,
                    "drafts": {
                        "ollama:m": {
                            "slot": "A",
                            "draft_hash": draft_hash(f"Lesson {i}."),
                            "declined": False,
                            "grounded": failed,
                            "faithful": failed,
                            "rule_supported": failed,
                            "store_exact": worth,
                            "generalizable": 1 if failed else 0,
                            "enforced_by": "none",
                        }
                    },
                    "preferred": None,
                    "note": None,
                    "labels_version": "labels_v4",
                    "labeled_at": "2026-01-01T00:00:00+00:00",
                }
            )
        # rules: right except on items 3 and 8, with low confidence when wrong
        wrong = i in (3, 8)
        p = (0.3 if failed else 0.6) if wrong else (0.85 if failed else 0.15)
        rules.append(_verdict(item_id, i, failed, "rules", p, 0.55 if wrong else 0.85, 2.0, v1=v1))
        llm.append(
            _verdict(item_id, i, failed, "llm:ref", 0.95 if failed else 0.05, 0.95, 4000.0, v1=v1)
        )
    store.write_jsonl(store.items_path, items)
    store.write_jsonl(store.labels_path, labels)
    if drafts:
        store.write_jsonl(store.drafts_path, drafts)
    store.write_jsonl(store.verdicts_path("rules"), rules)
    store.write_jsonl(store.verdicts_path("llm:ref"), llm)
    return store


def as_questions_version(store: RunStore, version: str) -> None:
    """Make a synthetic run one built under `version`: the run and every verdict it holds."""
    store.update_run(questions_version=version)
    for path in sorted(store.verdicts_dir.glob("*.jsonl")):
        rows = store.read_jsonl(path)
        for row in rows:
            row["questions_version"] = version
        store.write_jsonl(path, rows)


def test_report_locks_held_out_until_freeze(tmp_path):
    store = synthetic_run(tmp_path)
    text = build_report(store)
    assert "## Calibration split (12 items)" in text
    assert "Locked" in text
    assert "Held-out split (8 items)" not in text
    assert "SENTINEL_STATE_TEXT" not in text
    assert "### Stage A (one row per item)" in text
    assert "| rules | 12 |" in text
    assert "| llm:ref | 12 | 1.000 |" in text
    assert "### Stage B (one row per draft)" in text
    assert "| rules | ollama:m | 12 |" in text
    assert "### Drafters" in text
    assert "### How the turns closed (`closed_by`)" in text
    assert "| session_end |" in text
    assert "### By `closed_by`" in text
    assert "0 value(s) changed on 0 item(s)" in text
    frozen_text = build_report(store, freeze=True)
    run = store.read_run()
    assert run["frozen"]["reference"] == "llm:ref"
    assert set(run["frozen"]["backends"]) == {"rules", "llm:ref"}
    assert run["frozen"]["backends"]["rules"]["gate"] is not None
    rules_t = run["frozen"]["backends"]["rules"]["thresholds"]
    assert {"failed", "grounded", "faithful", "generalizable", "enforced"} <= set(rules_t)
    assert "## Held-out split (8 items)" in frozen_text
    assert "## Frozen thresholds" in frozen_text
    assert "grounded ≥" in frozen_text
    assert "Cascade to `llm:ref`" in frozen_text
    assert "(2) confidence AUROC ≥ 0.75: yes" in frozen_text
    assert "v4 write rule with `llm:ref` on `ollama:m` (criterion 6" in frozen_text
    assert store.report_path.read_text() == frozen_text
    with pytest.raises(JudgeError, match="already frozen"):
        build_report(store, freeze=True)


def test_report_still_reads_a_questions_v1_run(tmp_path):
    """exp1 and exp2 were built under questions_v1: one draft per item, `supported` instead of
    `grounded`, the v1 write rule. They must keep rendering."""
    store = synthetic_run(tmp_path, v1=True)
    text = build_report(store)
    assert "| rules | 12 |" in text
    assert "| llm:ref | 12 | 1.000 |" in text
    assert "| rules | — | 12 | 1.000 |" in text  # supported scored as grounded
    assert "Drafted lessons already enforced elsewhere" in text
    assert "### Drafters" not in text
    frozen = build_report(store, freeze=True)
    thresholds = store.read_run()["frozen"]["backends"]["rules"]["thresholds"]
    assert "supported" in thresholds
    assert "faithful" not in thresholds
    assert "failed ≥" in frozen
    assert "supported ≥" in frozen
    assert "criterion 6" not in frozen
    assert "| rules | — | failed 0." in frozen
    # Drafting, judging, and labeling under the new versions refuse a v1 run.
    from agent_memory.judge import pipeline

    cfg = Config(home=tmp_path / "exp")
    with pytest.raises(JudgeError, match="questions_v1"):
        pipeline.decide_items(cfg, "syn", ["rules"], store=store, record_events=False)
    with pytest.raises(JudgeError, match="questions_v1"):
        pipeline.draft_items(cfg, "syn", "ollama:m", store=store)
    with pytest.raises(JudgeError, match="questions_v1"):
        LabelController(store)


def test_report_cli_keeps_raw_stdout_clean_and_warns_about_width(tmp_path, capsys, monkeypatch):
    """`report --raw > file` must hold only the markdown; status and advice go to stderr."""
    from agent_memory.cli import main

    for var in ("AGENT_MEMORY_HOME", "FORCE_COLOR", "TTY_COMPATIBLE"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("COLUMNS", "100")
    store = synthetic_run(tmp_path)
    home = str(tmp_path / "exp")
    assert main(["--home", home, "judge", "report", "--run", "syn", "--raw"]) == 0
    out, err = capsys.readouterr()
    assert out.startswith("# Judge report: syn")
    assert out.rstrip().endswith("scored once.")
    assert "written to" not in out
    assert f"written to {store.report_path}" in err
    assert "columns" not in err  # raw output is not squeezed, so no width advice
    assert main(["--home", home, "judge", "report", "--run", "syn"]) == 0
    out, err = capsys.readouterr()
    assert "Judge report: syn" in out
    assert "written to" not in out
    assert "squeezed into 100 columns and need about 150" in err
    assert "Versions: unknown, unknown, questions_v4" in out


def test_labeling_shows_rubric_cited_evidence_and_incomplete(tmp_path):
    from agent_memory.judge.labeling import cited_entries

    projection = (
        "Coding-agent turn. project=p closed_by=next_message\n[1] USER: please run the tests\n"
        "[2] TOOL Execute: uv run pytest\n  result: 3 failed\n[3] ASSISTANT: all done\n"
        "[4] OUTCOME: none recorded\n[5] USER NEXT: no, the tests fail"
    )
    entries = cited_entries(projection, [2, 5, 9])
    assert [n for n, _ in entries] == [2, 5]
    assert entries[0][1] == "[2] TOOL Execute: uv run pytest\n  result: 3 failed"
    long = "[1] USER: " + "x" * 1000
    assert cited_entries(long, [1])[0][1].endswith("…")


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
    parsed, _ = endpoints.post_json("http://localhost/api", {"a": 1}, timeout=5)
    assert parsed == {"ok": True}
    assert len(calls) == 2
    calls.clear()
    monkeypatch.setattr(
        urllib.request, "urlopen", lambda r, timeout: (_ for _ in ()).throw(TimeoutError())
    )
    with pytest.raises(endpoints.EgressError, match="timed out after 5s"):
        endpoints.post_json("http://localhost/api", {}, timeout=5)
    # A backend that speaks to the transport itself reports the fault as a judge's.
    inp = JudgeInput(item_id="i", state="S", facts={}, flags=())
    with pytest.raises(JudgeError, match="timed out after 300s"):
        ReadoutBackend("m", host="http://localhost").decide(inp, [BY_ID["failed"]])


def test_drafter_caps_output_and_draft_items_skips_bad_items(tmp_path, monkeypatch):
    from agent_memory.judge import pipeline

    store = synthetic_run(tmp_path, n=4)
    store.drafts_path.unlink(missing_ok=True)
    first = store.read_jsonl(store.items_path)[0]["item_id"]
    store.write_projection(
        first, "[1] REQUEST: x\n[2] OUTCOME: none recorded\n[3] NEXT MESSAGE (human): unrelated"
    )
    bodies = []

    def fake_post(url, body, **kwargs):
        bodies.append(body)
        if len(bodies) == 2:
            raise endpoints.EgressError("http://x: timed out after 300s")
        return {
            "model": "m",
            "message": {"content": json.dumps(lesson_reply("Run tests first.", cited=(1,)))},
        }, 12.0

    monkeypatch.setattr(endpoints, "post_json", fake_post)
    cfg = Config(home=tmp_path / "exp")
    logs: list[str] = []
    warnings: list[str] = []
    summary = pipeline.draft_items(
        cfg, "syn", "ollama:m", store=store, log=logs.append, warn=warnings.append
    )
    assert (summary.written, summary.declined, summary.failed) == (3, 0, 1)
    assert bodies[0]["options"]["num_predict"] == 700
    assert bodies[0]["format"] == LESSON_SCHEMA
    assert bodies[0]["think"] is False
    # The drafter never sees the next message.
    assert "NEXT MESSAGE" not in bodies[0]["messages"][1]["content"]
    assert "OUTCOME: none recorded" in bodies[0]["messages"][1]["content"]
    # The failure goes to the warn channel, not into the progress lines.
    assert any("skipped (rerun to retry)" in line for line in warnings)
    assert not any("skipped" in line for line in logs)
    rows = store.read_jsonl(store.drafts_path)
    assert len(rows) == 3
    assert all(r["drafter"] == "ollama:m" and r["declined"] is False for r in rows)
    assert rows[0]["drafter_version"] == "drafter_v3"
    # The skipped item is picked up by the next run.
    assert pipeline.draft_items(cfg, "syn", "ollama:m", store=store).written == 1
    assert pipeline.make_drafter("ollama:m").server == Ollama("m")
    run = store.read_run()
    assert run["drafters"]["ollama:m"]["drafts"] == 4
    assert run["drafters"]["ollama:m"]["model_snapshot"] == "m"

    # A second drafter adds its own rows and leaves the first's alone; a decline is a row too.
    def fake_openai(url, body, headers=None, **kwargs):
        content = (
            '{"lesson": null}'
            if len(bodies) % 2
            else json.dumps(lesson_reply("Check first.", cited=(1,)))
        )
        bodies.append(body)
        return {
            "model": "x-served",
            "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
            "usage": {"cost": 0.001},
        }, 20.0

    monkeypatch.setattr(endpoints, "post_json", fake_openai)
    logs.clear()
    assert pipeline.draft_items(cfg, "syn", "llamacpp:x", store=store, log=logs.append).written == 4
    by_item = store.drafts_by_item()
    assert all(len(v) == 2 for v in by_item.values())
    assert all([d["drafter"] for d in v] == ["llamacpp:x", "ollama:m"] for v in by_item.values())
    declined = [
        d for v in by_item.values() for d in v if d["drafter"] == "llamacpp:x" and d["declined"]
    ]
    assert len(declined) == 2
    assert all(d["lesson"] is None for d in declined)
    assert sum("declined" in line for line in logs) == 2
    run = store.read_run()
    assert set(run["drafters"]) == {"ollama:m", "llamacpp:x"}
    assert run["drafters"]["llamacpp:x"]["declined"] == 2
    assert run["drafters"]["llamacpp:x"]["model_snapshot"] == "x-served"
    assert pipeline.draft_items(cfg, "syn", "llamacpp:x", store=store).written == 0

    # On resume the first undrafted item is the one that failed before; a drafter that has
    # already produced drafts in this run skips it instead of aborting.
    rows = [r for r in store.read_jsonl(store.drafts_path) if r["drafter"] == "ollama:m"]
    others = [r for r in store.read_jsonl(store.drafts_path) if r["drafter"] != "ollama:m"]
    store.write_jsonl(store.drafts_path, others + rows[:-1])
    monkeypatch.setattr(
        endpoints, "post_json", lambda *a, **k: (_ for _ in ()).throw(endpoints.EgressError("down"))
    )
    logs.clear()
    # Without a warn channel the problem still reaches the one channel there is.
    summary = pipeline.draft_items(cfg, "syn", "ollama:m", store=store, log=logs.append)
    assert (summary.written, summary.failed) == (0, 1)
    assert any("skipped" in line for line in logs)

    # A model server that cannot be reached ends a fresh run after three items: not at the
    # first, and not after all four.
    store.drafts_path.unlink()
    unanswered: list[Any] = []

    def down(url, body, **kwargs):
        unanswered.append(body)
        raise endpoints.Unreachable("down")

    monkeypatch.setattr(endpoints, "post_json", down)
    with pytest.raises(JudgeError, match="could not be reached 3 times in a row: down"):
        pipeline.draft_items(cfg, "syn", "ollama:m", store=store)
    assert len(unanswered) == 3


def test_draft_stops_after_three_failures_and_keeps_what_it_wrote(tmp_path, monkeypatch):
    from agent_memory.judge import pipeline

    store = synthetic_run(tmp_path, n=6)
    store.drafts_path.unlink()
    bodies: list[Any] = []

    def one_then_down(url, body, **kwargs):
        bodies.append(body)
        if len(bodies) > 1:
            raise endpoints.Unreachable("http://x: connection refused")
        reply = json.dumps(lesson_reply("Run tests first.", cited=(1,)))
        return {"model": "m", "message": {"content": reply}}, 12.0

    monkeypatch.setattr(endpoints, "post_json", one_then_down)
    warnings: list[str] = []
    with pytest.raises(
        JudgeError,
        match="ollama:m: stopped, the model server could not be reached 3 times in a row",
    ):
        pipeline.draft_items(store.cfg, "syn", "ollama:m", store=store, warn=warnings.append)
    assert len(bodies) == 4  # Two of the six items were never asked for.
    assert len(warnings) == 3
    assert len(store.read_jsonl(store.drafts_path)) == 1
    # The run still says what this drafter has written.
    assert store.read_run()["drafters"]["ollama:m"]["drafts"] == 1

    # A drafter whose every draft is cut off has the server's ear and nothing to show. With
    # nothing written yet it stops as well, after three items of two attempts each.
    store.drafts_path.unlink()
    bodies.clear()

    def cut_off(url, body, **kwargs):
        bodies.append(body)
        return {"model": "m", "message": {"content": "{"}, "done_reason": "length"}, 5.0

    monkeypatch.setattr(endpoints, "post_json", cut_off)
    with pytest.raises(JudgeError, match="3 failures in a row and nothing written yet"):
        pipeline.draft_items(store.cfg, "syn", "ollama:m", store=store)
    assert [b["options"]["num_predict"] for b in bodies] == [700, 2100] * 3


def _raising(fault: Exception):
    def post(*args: Any, **kwargs: Any):
        raise fault

    return post


_TURN = JudgeInput(item_id="i", state="S", facts={}, flags=())


@pytest.mark.parametrize(
    "ask",
    [
        lambda: ollama_judge("m").decide(_TURN, [BY_ID["failed"]]),
        lambda: llamacpp_judge("m").decide(_TURN, [BY_ID["failed"]]),
        lambda: ReadoutBackend("m").decide(_TURN, [BY_ID["failed"]]),
        lambda: JevBackend(api_key="k").decide(_TURN, [BY_ID["failed"]]),
        lambda: make_drafter("ollama:m").draft("i", "S"),
    ],
    ids=["llm", "llamacpp", "readout", "jev", "drafter"],
)
def test_only_a_server_that_was_not_reached_counts_against_the_server(monkeypatch, ask):
    monkeypatch.setenv(REMOTE_ENV, "1")
    monkeypatch.setattr(endpoints, "get_json", _raising(endpoints.EgressError("down")))
    monkeypatch.setattr(
        endpoints, "post_json", _raising(endpoints.Unreachable("http://x: connection refused"))
    )
    with pytest.raises(ServerUnreachable, match="connection refused"):
        ask()
    # A server that was too slow or said no was reached, and the item may be why.
    for fault in (
        endpoints.EgressError("http://x: timed out after 300s"),
        endpoints.HttpStatusError("http://x: HTTP 400: context too long", 400),
        endpoints.HttpStatusError("http://x: HTTP 500: out of memory", 500),
        endpoints.EgressError("http://x: response is not JSON"),
    ):
        monkeypatch.setattr(endpoints, "post_json", _raising(fault))
        with pytest.raises(JudgeError) as raised:
            ask()
        assert not isinstance(raised.value, ServerUnreachable), fault


def test_three_slow_items_do_not_stop_the_drafts_behind_them(tmp_path, monkeypatch):
    """A timeout may be the item's doing, a long turn. Counted against the model server,
    three such items at the head of a run's unfinished work stopped every rerun there."""
    from agent_memory.judge import pipeline

    store = synthetic_run(tmp_path, n=6)
    store.drafts_path.unlink()

    def timing_out(*slow: int):
        calls: list[Any] = []

        def post(url, body, **kwargs):
            calls.append(body)
            if len(calls) in slow:
                raise endpoints.EgressError("http://x: timed out after 300s")
            reply = json.dumps(lesson_reply("Run tests first.", cited=(1,)))
            return {"model": "m", "message": {"content": reply}}, 12.0

        return post

    def draft(**more: Any):
        return pipeline.draft_items(store.cfg, "syn", "ollama:m", store=store, **more)

    # The first command is cut short after the three slow items.
    monkeypatch.setattr(endpoints, "post_json", timing_out(2, 3, 4))
    first = draft(limit=4)
    assert (first.written, first.failed) == (1, 3)
    # The next one meets them first, skips them again, and drafts the two behind them.
    monkeypatch.setattr(endpoints, "post_json", timing_out(1, 2, 3))
    second = draft()
    assert (second.written, second.failed) == (2, 3)
    assert store.read_run()["drafters"]["ollama:m"]["drafts"] == 3


REAL_RULES_DECIDE = RulesBackend.decide


def scripted_rules(monkeypatch, store: RunStore, script: str) -> list[str]:
    """Make the rules backend treat the run's items, in order, as the script says: `ok`
    answers, `own` fails for a reason of the item's own, and `down` cannot reach the model
    server. Returns the ids of the items it is asked about, once each and in order."""
    plan = dict(zip([i["item_id"] for i in store.items()], script.split(), strict=True))
    asked: list[str] = []

    def decide(self, inp, questions):
        if inp.item_id not in asked:
            asked.append(inp.item_id)
        outcome = plan[inp.item_id]
        if outcome == "down":
            raise ServerUnreachable("http://x: connection refused")
        if outcome == "own":
            raise JudgeError("judge ran out of tokens; JSON incomplete: ''")
        return REAL_RULES_DECIDE(self, inp, questions)

    monkeypatch.setattr(RulesBackend, "decide", decide)
    return asked


def test_decide_skips_an_item_that_fails_and_says_so(tmp_path, monkeypatch):
    from agent_memory.judge import pipeline

    store = synthetic_run(tmp_path, n=3)
    store.verdicts_path("rules").unlink()
    cfg = Config(home=tmp_path / "exp")
    items = store.read_jsonl(store.items_path)
    bad = items[0]["item_id"]
    # The first item of a fresh run fails for a reason of its own. That used to end the
    # command, and every rerun after it at the same item.
    scripted_rules(monkeypatch, store, "own ok ok")
    logs: list[str] = []
    warnings: list[str] = []
    summary = pipeline.decide_items(
        cfg,
        "syn",
        ["rules"],
        store=store,
        record_events=False,
        log=logs.append,
        warn=warnings.append,
    )
    assert summary.written == {"rules": 2}
    assert summary.failed == {"rules": 1}
    assert any(f"item {bad} failed" in line for line in warnings)
    assert any("1 verdict(s) failed" in line for line in warnings)
    assert not any("failed" in line for line in logs)
    verdicts = store.read_jsonl(store.verdicts_path("rules"))
    assert [v["item_id"] for v in verdicts] == [items[1]["item_id"], items[2]["item_id"]]
    last = verdicts[-1]
    assert last["drafter"] == "ollama:m"
    assert last["draft_hash"] == draft_hash("Lesson 2.")
    assert set(last["answers"]) == {q.id for q in ALL_QUESTIONS}
    # The next command asks the skipped item again, and only that one.
    asked = scripted_rules(monkeypatch, store, "ok ok ok")
    again = pipeline.decide_items(cfg, "syn", ["rules"], store=store, record_events=False)
    assert (again.written, again.failed) == ({"rules": 1}, {"rules": 0})
    assert asked == [bad]


@pytest.mark.parametrize(
    ("script", "stops", "asked", "written"),
    [
        # The model server cannot be reached three times in a row: it is gone, whatever the
        # items are.
        ("down down down ok ok", "could not be reached 3 times in a row", 3, 0),
        ("ok down down down ok", "could not be reached 3 times in a row", 4, 1),
        # A backend that has written nothing in the run and fails three times running is
        # unfit or misconfigured, whatever the three reasons are.
        ("own own own ok ok", "3 failures in a row and nothing written yet", 3, 0),
        ("own down own ok ok", "3 failures in a row and nothing written yet", 3, 0),
        # Once it has written a verdict, an item's own failure never stops the run.
        ("ok own own own own", None, 5, 1),
        ("own own ok own own", None, 5, 1),
        # An answer of any kind shows the server is there, so the count starts again.
        ("ok down down own down down", None, 6, 1),
        ("ok down down ok down down", None, 6, 2),
    ],
)
def test_decide_stops_only_when_the_fault_is_not_the_items(
    tmp_path, monkeypatch, script, stops, asked, written
):
    from agent_memory.judge import pipeline

    store = synthetic_run(tmp_path, n=len(script.split()))
    store.verdicts_path("rules").unlink()
    seen = scripted_rules(monkeypatch, store, script)

    def decide():
        return pipeline.decide_items(store.cfg, "syn", ["rules"], store=store, record_events=False)

    if stops is None:
        summary = decide()
        assert summary.written == {"rules": written}
        assert summary.failed == {"rules": len(script.split()) - written}
    else:
        with pytest.raises(JudgeError, match=stops) as stopped:
            decide()
        assert str(stopped.value).startswith("rules: stopped, ")
    # No item after the one that stopped the run is asked, and what was written stays.
    assert len(seen) == asked
    assert len(store.read_jsonl(store.verdicts_path("rules"))) == written


def test_a_stopped_decide_still_records_the_verdicts_it_wrote(tmp_path, monkeypatch):
    from agent_memory.judge import pipeline

    store = synthetic_run(tmp_path, n=5)
    store.verdicts_path("rules").unlink()
    scripted_rules(monkeypatch, store, "ok ok down down down")
    records: list[Any] = []
    monkeypatch.setattr(pipeline, "_record", lambda cfg, batch, warn: records.extend(batch))
    monkeypatch.setattr(pipeline, "require_judge_agent", lambda cfg: None)
    with pytest.raises(JudgeError, match="rules: stopped"):
        pipeline.decide_items(store.cfg, "syn", ["rules"], store=store)
    # A decision event for each verdict on disk, though the command ended in an error.
    assert len(records) == len(store.read_jsonl(store.verdicts_path("rules"))) == 2


def test_judge_commands_exit_1_when_an_item_was_skipped(tmp_path, monkeypatch, capsys):
    from agent_memory.cli import main

    monkeypatch.delenv("AGENT_MEMORY_HOME", raising=False)
    store = synthetic_run(tmp_path, n=3)
    store.verdicts_path("rules").unlink()
    home = str(tmp_path / "exp")
    decide = ["--home", home, "judge", "decide", "--run", "syn", "--backend", "rules"]
    scripted_rules(monkeypatch, store, "ok own ok")
    assert main([*decide, "--no-events"]) == 1
    out, err = capsys.readouterr()
    # The run was not cut short: the items either side of the failed one have verdicts.
    assert len(store.read_jsonl(store.verdicts_path("rules"))) == 2
    assert "new verdicts" in out
    assert "1 verdict(s) failed" in err
    scripted_rules(monkeypatch, store, "ok ok ok")
    assert main([*decide, "--no-events"]) == 0
    assert len(store.read_jsonl(store.verdicts_path("rules"))) == 3

    store.drafts_path.unlink()
    bodies: list[Any] = []

    def second_times_out(url, body, **kwargs):
        bodies.append(body)
        if len(bodies) == 2:
            raise endpoints.EgressError("http://x: timed out after 300s")
        reply = json.dumps(lesson_reply("Run tests first.", cited=(1,)))
        return {"model": "m", "message": {"content": reply}}, 12.0

    monkeypatch.setattr(endpoints, "post_json", second_times_out)
    draft = ["--home", home, "judge", "draft", "--run", "syn", "--drafter", "ollama:m"]
    capsys.readouterr()
    assert main(draft) == 1
    assert "drafted 2 item(s) with ollama:m; 1 failed" in capsys.readouterr().out
    assert main(draft) == 0
    assert len(store.read_jsonl(store.drafts_path)) == 3


def test_live_exits_1_when_an_item_was_skipped(tmp_path, monkeypatch, capsys):
    from agent_memory.cli import main
    from agent_memory.judge import pipeline

    monkeypatch.delenv("AGENT_MEMORY_HOME", raising=False)
    store = synthetic_run(tmp_path)
    build_report(store, freeze=True)
    day = store.live_day("2026-01-02")
    day.create(
        {
            "parent_run": "syn",
            "live_day": "2026-01-02",
            "frozen": store.read_run()["frozen"],
            "questions_version": QUESTIONS_VERSION,
        }
    )
    items = store.items()[:3]
    day.write_items(items)
    for item in items:
        day.write_projection(item["item_id"], store.read_projection(item["item_id"]))
    monkeypatch.setattr(pipeline, "import_sessions", lambda *a, **k: pipeline.ImportSummary())
    monkeypatch.setattr(pipeline, "require_judge_agent", lambda cfg: None)
    monkeypatch.setattr(pipeline, "_record", lambda cfg, batch, warn: None)
    scripted_rules(monkeypatch, day, "ok own ok")
    live = ["judge", "live", "--run", "syn", "--since", "2026-01-01", "--day", "2026-01-02"]
    assert main(["--home", str(tmp_path / "exp"), *live]) == 1
    out = " ".join(capsys.readouterr().out.split())
    assert "verdicts {'rules': 2}; 1 failed, rerun to retry" in out
    assert len(day.read_jsonl(day.verdicts_path("rules"))) == 2


def test_decide_items_judges_each_draft_and_reuses_turn_answers(tmp_path, monkeypatch):
    from agent_memory.judge import pipeline

    store = synthetic_run(tmp_path, n=3)
    store.verdicts_path("rules").unlink()
    store.verdicts_path("llm:ref").unlink()
    cfg = Config(home=tmp_path / "exp")
    ids = [i["item_id"] for i in store.read_jsonl(store.items_path)]
    # ollama:m drafted everything; llamacpp:x drafted item 0, declined item 1, skipped item 2.
    store.write_jsonl(
        store.drafts_path,
        [
            {"item_id": ids[0], "lesson": "Second.", "drafter": "llamacpp:x"},
            {"item_id": ids[1], "lesson": None, "declined": True, "drafter": "llamacpp:x"},
        ],
        append=True,
    )
    turn_calls: list[str] = []
    real_decide = RulesBackend.decide

    def counting(self, inp, questions):
        if questions is TURN_QUESTIONS:
            turn_calls.append(inp.item_id)
        return real_decide(self, inp, questions)

    monkeypatch.setattr(RulesBackend, "decide", counting)
    summary = pipeline.decide_items(cfg, "syn", ["rules"], store=store, record_events=False)
    assert summary.written == {"rules": 5}
    assert sorted(turn_calls) == sorted(ids)  # once per item, not per draft
    rows = store.read_jsonl(store.verdicts_path("rules"))
    by_key = {(r["item_id"], r["drafter"]): r for r in rows}
    assert set(by_key) == {
        (ids[0], "llamacpp:x"),
        (ids[0], "ollama:m"),
        (ids[1], "llamacpp:x"),
        (ids[1], "ollama:m"),
        (ids[2], "ollama:m"),
    }
    declined = by_key[(ids[1], "llamacpp:x")]
    assert (declined["declined"], declined["has_draft"], declined["draft_hash"]) == (
        True,
        False,
        None,
    )
    assert declined["action"] == "discard"
    assert declined["reasons"] == ["drafter declined: nothing worth keeping"]
    assert set(declined["answers"]) == {q.id for q in TURN_QUESTIONS}
    full = by_key[(ids[0], "ollama:m")]
    assert set(full["answers"]) == {q.id for q in ALL_QUESTIONS}
    # The first row of an item paid for the turn questions; the second copied them at no cost.
    first, second = sorted(
        (by_key[(ids[0], "llamacpp:x")], by_key[(ids[0], "ollama:m")]),
        key=lambda r: r["stage_a_reused"],
    )
    assert (first["stage_a_reused"], second["stage_a_reused"]) == (False, True)
    assert all(
        a["latency_ms"] == 0.0
        for q, a in second["answers"].items()
        if q in {x.id for x in TURN_QUESTIONS}
    )
    assert first["answers"]["failed"] == {
        **second["answers"]["failed"],
        "latency_ms": first["answers"]["failed"]["latency_ms"],
    }
    # Ledger events are keyed per drafter too, so two drafts of one turn are two events.
    created = store.read_run()["created_at"]
    keys = {
        Verdict(r)
        .decision_record({"stratum": "T0", "flags": [], "turn_event_id": "t"}, created)
        .source_key
        for r in rows
    }
    assert len(keys) == 5
    assert all(k.endswith(("/ollama:m", "/llamacpp:x")) for k in keys)

    # Resume: a late draft on item 2 gets one new row, with turn answers copied from the stored one.
    turn_calls.clear()
    store.write_jsonl(
        store.drafts_path,
        [{"item_id": ids[2], "lesson": "Late.", "drafter": "llamacpp:x"}],
        append=True,
    )
    again = pipeline.decide_items(cfg, "syn", ["rules"], store=store, record_events=False)
    assert again.written == {"rules": 1}
    assert turn_calls == []
    rows = store.read_jsonl(store.verdicts_path("rules"))
    assert len(rows) == 6
    late = next(r for r in rows if (r["item_id"], r["drafter"]) == (ids[2], "llamacpp:x"))
    assert late["stage_a_reused"] is True
    assert late["draft_hash"] == draft_hash("Late.")
    # A changed lesson supersedes only its own row.
    drafts = store.read_jsonl(store.drafts_path)
    for d in drafts:
        if d["item_id"] == ids[0] and d["drafter"] == "ollama:m":
            d["lesson"] = "Rewritten."
    store.write_jsonl(store.drafts_path, drafts)
    again = pipeline.decide_items(cfg, "syn", ["rules"], store=store, record_events=False)
    assert again.written == {"rules": 1}
    rows = store.read_jsonl(store.verdicts_path("rules"))
    assert len(rows) == 6
    redone = next(r for r in rows if (r["item_id"], r["drafter"]) == (ids[0], "ollama:m"))
    assert redone["draft_hash"] == draft_hash("Rewritten.")
    again = pipeline.decide_items(cfg, "syn", ["rules"], store=store, record_events=False)
    assert again.written == {"rules": 0}


def test_draft_refuses_mixed_versions_before_request_or_write(tmp_path, monkeypatch):
    from agent_memory.judge import pipeline

    store = synthetic_run(tmp_path, n=1)
    rows = store.read_jsonl(store.drafts_path)
    rows[0]["drafter_version"] = "drafter_v2"
    store.write_jsonl(store.drafts_path, rows)
    before = (store.run_json.read_bytes(), store.drafts_path.read_bytes())

    def unexpected(*args):
        pytest.fail("mixed-version guard must run before constructing a drafter")

    monkeypatch.setattr(pipeline, "make_drafter", unexpected)
    with pytest.raises(
        JudgeError,
        match=(
            r"run syn holds drafter_v2 drafts; drafter_v3 arms cannot be mixed in\. "
            r"Start a new run\."
        ),
    ):
        pipeline.draft_items(store.cfg, store.run_id, "llamacpp:new", store=store)
    assert (store.run_json.read_bytes(), store.drafts_path.read_bytes()) == before


def test_questions_v2_reports_but_mutating_commands_are_read_only(tmp_path):
    from agent_memory.judge import pipeline

    store = synthetic_run(tmp_path)
    as_questions_version(store, "questions_v2")
    assert "## Claims" not in build_report(store)
    assert "questions_v2" in build_report(store, freeze=True)
    assert "criterion 6" in build_report(store)
    paths = [store.run_json, store.drafts_path, store.labels_path, store.verdicts_path("rules")]
    before = [path.read_bytes() for path in paths]
    for call in (
        lambda: pipeline.draft_items(store.cfg, store.run_id, "ollama:m", store=store),
        lambda: pipeline.decide_items(store.cfg, store.run_id, ["rules"], store=store),
        lambda: LabelController(store),
        lambda: LabelController(store, review=True),
    ):
        with pytest.raises(JudgeError, match="questions_v2"):
            call()
    assert [path.read_bytes() for path in paths] == before


def test_report_refuses_verdicts_judged_with_another_question_set(tmp_path):
    """Each question set has its own write gates, so one report cannot score two as one."""
    store = synthetic_run(tmp_path)
    path = store.verdicts_path("rules")
    rows = store.read_jsonl(path)
    rows[-1]["questions_version"] = "questions_v2"
    store.write_jsonl(path, rows)
    run_before = store.run_json.read_bytes()
    for freeze in (False, True):
        with pytest.raises(JudgeError, match=r"rules\.jsonl.*questions_v2.*questions_v4"):
            build_report(store, freeze=freeze)
    assert store.run_json.read_bytes() == run_before  # Nothing was frozen on the way out.
    assert not store.report_path.exists()
    rows[-1].pop("questions_version")
    store.write_jsonl(path, rows)
    with pytest.raises(JudgeError, match="no question set"):
        build_report(store)


def test_per_claim_labels_round_trip_partial_backfill_and_changes():
    from agent_memory.judge.labeling import (
        assemble,
        changed_values,
        missing_steps,
        order_drafts,
        raw_answers,
        steps_for,
    )

    draft = structured_draft(
        claims=[
            {"text": "First claim.", "cited": [1]},
            {"text": "Second claim.", "cited": []},
        ]
    )
    views = order_drafts("i", [draft.to_dict()], 1)
    assert len(views[0].claims) == 2
    raw = {
        "failed": "n",
        "worth_remembering": "n",
        "A.c1.grounded": "y",
        "A.c1.faithful": "y",
        "A.c2.grounded": "n",
        "A.c2.faithful": "n",
        "A.rule_supported": "n",
        "A.generalizable": "1",
        "A.enforced_by": "1",
        "A.store_exact": "n",
        "note": "",
    }
    assert steps_for(raw, views, False) == list(raw)
    row = assemble(raw, views, False)
    entry = row["drafts"]["ollama:m"]
    assert entry == {
        "slot": "A",
        "draft_hash": draft_hash(draft.lesson),
        "declined": False,
        "claims": [{"grounded": True, "faithful": True}, {"grounded": False, "faithful": False}],
        "grounded": False,
        "faithful": False,
        "rule_supported": False,
        "store_exact": False,
        "generalizable": 1,
        "enforced_by": "none",
    }
    assert raw_answers(row, views) == raw
    assert missing_steps(row, views, False) == []
    edited = assemble({**raw, "A.c1.grounded": "n"}, views, False)
    assert changed_values(row, edited) == {"drafts.ollama:m.claims.1.grounded": True}
    partial = assemble(
        {key: value for key, value in raw.items() if not key.startswith("A.c2")}, views, False
    )
    assert partial["drafts"]["ollama:m"]["grounded"] is None
    assert missing_steps(partial, views, False) == ["A.c2.grounded", "A.c2.faithful"]
    partial = assemble({"A.c1.grounded": "n"}, views, False)
    assert partial["drafts"]["ollama:m"]["grounded"] is False
    complete = assemble({**raw, "A.c2.grounded": "y", "A.c2.faithful": "y"}, views, False)
    assert complete["drafts"]["ollama:m"]["grounded"] is True
    assert complete["drafts"]["ollama:m"]["faithful"] is True


def test_claim_controller_focus_evidence_summary_navigation_and_history(tmp_path):
    from agent_memory.judge.label_controller import Action

    store = synthetic_run(tmp_path, n=1)
    store.labels_path.unlink()
    item_id = store.read_jsonl(store.items_path)[0]["item_id"]
    store.write_jsonl(
        store.drafts_path,
        [
            structured_draft(
                item_id,
                claims=[
                    {"text": "The database was locked.", "cited": [2]},
                    {"text": "The server was running.", "cited": [1, 9]},
                    {"text": "No evidence here.", "cited": []},
                ],
            ).to_dict()
        ],
    )
    store.write_projection(item_id, "[1] Server evidence.\n[2] Lock evidence.\n[3] Other evidence.")
    controller = LabelController(store)
    for value in ("n", "n"):
        controller.apply(Action("answer", value))
    assert controller.step == "A.c1.grounded"
    screen = controller.screen()
    assert screen.subject == "Draft A"
    assert screen.question == "Does this claim have evidence in the lines it cites?"
    assert screen.drafts[0].focus == "Claim 1 of 3"
    assert screen.drafts[0].lesson.count("▶") == 1
    assert "▶ 1. The database was locked." in screen.drafts[0].lesson
    assert "[2] Lock evidence." in screen.drafts[0].cited_evidence
    assert "[1]" not in screen.drafts[0].cited_evidence
    controller.apply(Action("answer", "y"))
    assert controller.screen().question == "Does this claim read those lines correctly?"
    controller.apply(Action("answer", "y"))
    assert controller.step == "A.c2.grounded"
    screen = controller.screen()
    assert screen.drafts[0].focus == "Claim 2 of 3"
    assert "[1] Server evidence." in screen.drafts[0].cited_evidence
    assert "Missing citations: [9]" in screen.drafts[0].cited_evidence
    assert "[2]" not in screen.drafts[0].cited_evidence
    controller.apply(Action("answer", "n"))
    controller.apply(Action("back"))
    assert controller.step == "A.c2.grounded"
    assert controller.screen().selected == "n"
    controller.apply(Action("answer", "n"))
    controller.apply(Action("answer", "n"))
    assert controller.screen().drafts[0].cited_evidence == "No cited evidence."
    controller.apply(Action("answer", "n"))
    controller.apply(Action("answer", "n"))
    assert controller.step == "A.rule_supported"
    screen = controller.screen()
    assert screen.drafts[0].focus == ""
    assert "▶" not in screen.drafts[0].lesson
    assert "[1] Server evidence." in screen.drafts[0].cited_evidence
    assert "[2] Lock evidence." in screen.drafts[0].cited_evidence
    controller.apply(Action("answer", "n"))
    controller.apply(Action("answer", "1"))
    controller.apply(Action("answer", "1"))
    controller.apply(Action("answer", "n"))
    controller.apply(Action("answer", "continue"))
    assert "Draft A claim 2: Grounded: No" in controller.screen().evidence
    controller.apply(Action("answer", "save"))
    saved = store.read_jsonl(store.labels_path)[0]
    assert saved["labels_version"] == "labels_v4"
    assert saved["drafts"]["ollama:m"]["grounded"] is False
    review = LabelController(store, item_ids=[item_id])
    while review.step != "A.c1.grounded":
        review.apply(Action("answer", review.screen().selected))
    review.apply(Action("answer", "n"))
    while review.step != "summary":
        review.apply(
            Action("answer", "continue" if review.step == "note" else review.screen().selected)
        )
    review.apply(Action("answer", "save"))
    saved = store.read_jsonl(store.labels_path)[0]
    assert saved["history"][0]["changed"] == {"drafts.ollama:m.claims.1.grounded": True}
    # Backfill a single missing claim answer, not the entire draft.
    saved["drafts"]["ollama:m"]["claims"][1]["faithful"] = None
    store.write_jsonl(store.labels_path, [saved])
    backfill = LabelController(store)
    assert backfill.plan == ["A.c2.faithful"]
    backfill.apply(Action("answer", "n"))
    backfill.apply(Action("answer", "save"))
    assert LabelController(store).done


def test_claim_focus_survives_a_lesson_text_that_is_not_the_rendering(tmp_path):
    from agent_memory.judge.label_controller import Action

    store = synthetic_run(tmp_path, n=1)
    store.labels_path.unlink()
    item_id = store.read_jsonl(store.items_path)[0]["item_id"]
    row = structured_draft(item_id, claims=[{"text": "Only claim.", "cited": [1]}]).to_dict()
    store.write_jsonl(store.drafts_path, [{**row, "lesson": "Edited by hand"}])
    store.write_projection(item_id, "[1] Evidence.")
    controller = LabelController(store)
    for value in ("n", "n"):
        controller.apply(Action("answer", value))
    assert controller.step == "A.c1.grounded"
    screen = controller.screen()
    assert screen.drafts[0].focus == "Claim 1 of 1"
    assert screen.drafts[0].lesson == "Edited by hand"
    assert "[1] Evidence." in screen.drafts[0].cited_evidence


def test_lesson_from_row_reports_a_bad_row_without_echoing_it():
    row = {
        "item_id": "i",
        "drafter": "ollama:m",
        "lesson": "x",
        "rule": "r",
        "kind": "PRIVATE_KIND",
        "claims": [{"text": "PRIVATE_TEXT", "cited": [], "PRIVATE_KEY": 1}],
    }
    with pytest.raises(JudgeError, match="draft row for item 'i' by 'ollama:m'") as error:
        lesson_from_row(row)
    assert "kind: Input should be" in str(error.value)
    assert "claims.0.[unexpected key]: Extra inputs" in str(error.value)
    assert "PRIVATE" not in str(error.value)


def test_claim_report_rates_metrics_kind_and_legacy_absence(tmp_path):
    from agent_memory.judge import pipeline
    from agent_memory.judge.labeling import assemble, order_drafts

    store = synthetic_run(tmp_path, n=3)
    assert "## Claims" not in build_report(store)
    items = store.read_jsonl(store.items_path)
    drafts, labels = [], []
    for i, item in enumerate(items):
        if i == 2:
            draft = Draft.from_lesson(
                item["item_id"], None, drafter="ollama:m", model_snapshot="m", latency_ms=10.0
            )
        else:
            draft = structured_draft(
                item["item_id"],
                claims=[
                    {"text": "The database was locked.", "cited": [1]},
                    {"text": "The server was running.", "cited": []},
                    {"text": "Missing evidence.", "cited": [9]},
                ],
            )
        drafts.append(draft.to_dict())
        views = order_drafts(item["item_id"], [draft.to_dict()], 1)
        label = assemble(
            {
                "failed": "n",
                "worth_remembering": "y",
                "lesson_kind": "1" if i == 0 else "3",
                "A.c1.grounded": "y",
                "A.c1.faithful": "y",
                "A.c2.grounded": "n",
                "A.c2.faithful": "n",
                "A.c3.grounded": "y",
                "A.c3.faithful": "n",
                "A.generalizable": "1",
                "A.enforced_by": "1",
                "note": "",
            },
            views,
            False,
        )
        labels.append({"item_id": item["item_id"], **label})
        store.write_projection(item["item_id"], "[1] The database was locked.")
    store.write_jsonl(store.drafts_path, drafts)
    store.write_jsonl(store.labels_path, labels)
    store.verdicts_path("rules").unlink()
    store.verdicts_path("llm:ref").unlink()
    pipeline.decide_items(store.cfg, store.run_id, ["rules"], store=store, record_events=False)
    report = build_report(store)
    assert "## Claims" in report
    assert (
        "| ollama:m | 3 | 2.00 | 6 | 0.333 | 0.500 | 6 | 0.667 | 6 | 0.333 | 2 | 0.500 |"
    ) in report
    assert "| rules | calibration | grounded | 6 | 0.667 | 1.000 | 0.500 | 0.667 |" in report
    assert "| rules | calibration | faithful | 6 | 1.000 | 1.000 | 1.000 | 1.000 |" in report
    assert "The database was locked" not in report
    # Claim decisions obey the same held-out lock as draft-level metrics.
    items[1]["split"] = "held_out"
    store.write_jsonl(store.items_path, items)
    report = build_report(store)
    assert "| rules | held_out | grounded |" not in report
    report = build_report(store, freeze=True)
    assert "| rules | held_out | grounded | 3 |" in report
    # Purging evidence does not invent missing-citation failures.
    store.purge_projections()
    report = build_report(store)
    assert "4 citation reference(s) could not be checked" in report
