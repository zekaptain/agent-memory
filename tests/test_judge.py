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
from agent_memory.judge.backends.llamacpp import LlamaCppJudgeBackend
from agent_memory.judge.backends.llm import (
    LlmJudgeBackend,
    answers_from_judge_json,
    parse_judge_json,
)
from agent_memory.judge.backends.readout import ReadoutBackend, _letter_logits, first_letter
from agent_memory.judge.backends.rules import RulesBackend
from agent_memory.judge.drafter import draft_hash, make_drafter, parse_draft
from agent_memory.judge.labeling import SAVE_PROMPT, label_items, order_drafts
from agent_memory.judge.pipeline import assign_splits
from agent_memory.judge.projection import FullEvent, build_projection, drafter_rendering
from agent_memory.judge.questions import (
    ALL_QUESTIONS,
    BY_ID,
    STAGE_A,
    STAGE_B,
    WRITE_GATES_V1,
    derive_action,
    gate_probabilities,
    thresholds_from_run,
)
from agent_memory.judge.report import build_report
from agent_memory.judge.store import JudgeError, RunStore, ensure_experiment_home
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


def test_question_set_shape_and_derived_action():
    assert [q.id for q in STAGE_A] == [
        "failed",
        "incomplete",
        "user_corrected",
        "env_confirmed",
        "category",
        "recurrence",
    ]
    assert [q.id for q in STAGE_B] == [
        "grounded",
        "faithful",
        "generalizable",
        "actionable",
        "enforced",
    ]
    assert BY_ID["category"].options()[-1] == "none"
    assert BY_ID["generalizable"].options() == ["0", "1", "2"]
    assert BY_ID["enforced"].options() == ["none", "harness", "project_docs", "tooling"]
    # The v2 write rule reads Stage B only: grounded, faithful, reusable, enforced by nothing.
    probs = {
        "failed": {"yes": 0.1, "no": 0.9},
        "grounded": {"yes": 0.9, "no": 0.1},
        "faithful": {"yes": 0.8, "no": 0.2},
        "generalizable": {"0": 0.2, "1": 0.5, "2": 0.3},
        "enforced": {"none": 0.7, "harness": 0.1, "project_docs": 0.1, "tooling": 0.1},
    }
    gates = gate_probabilities(probs)
    assert gates == {"grounded": 0.9, "faithful": 0.8, "generalizable": 0.8, "enforced": 0.7}
    assert derive_action(gates).action == "write_lesson"
    assert derive_action({**gates, "enforced": 0.2}).action == "discard"
    assert derive_action({**gates, "faithful": 0.55}).action == "review"
    assert derive_action({"grounded": 0.9}).action == "discard"
    declined = derive_action(gates, declined=True)
    assert (declined.action, declined.reasons) == (
        "discard",
        ["drafter declined: nothing worth keeping"],
    )
    # v1 runs keep their rule.
    v1 = {"failed": 0.9, "recurrence": 0.8, "supported": 0.9}
    assert derive_action(v1, gates=WRITE_GATES_V1).action == "write_lesson"
    assert derive_action({**v1, "recurrence": 0.2}, gates=WRITE_GATES_V1).action == "discard"
    assert derive_action({**v1, "failed": 0.55}, gates=WRITE_GATES_V1).action == "review"
    assert gate_probabilities({"supported": {"yes": 0.4, "no": 0.6}}, WRITE_GATES_V1) == {
        "supported": 0.4
    }
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
        JudgeInput(item_id="j", state="", facts={"last_result_ok": True}, flags=("T0",)), STAGE_A
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
    assert isinstance(make_backend("llamacpp:gemma-4-26b"), LlamaCppJudgeBackend)
    # Router-mode ids carry slashes and colons; the spec keeps everything after the kind.
    router = make_backend("llamacpp:ggml-org/Qwen3.8-27B-GGUF:Q8_0")
    assert isinstance(router, LlamaCppJudgeBackend)
    assert router.model == "ggml-org/Qwen3.8-27B-GGUF:Q8_0"
    assert router.host == "http://localhost:9931"
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
    # The retry must say `think: false` outright; Ollama treats a missing flag as "think".
    assert [c.get("think", "absent") for c in calls] == [True, False]
    assert [c.get("format") for c in calls] == ["json", None]
    assert answers["failed"].value == "no"
    assert answers["failed"].method == "llm_json_nothink"
    assert answers["failed"].model_snapshot == "m:1"

    # A whole budget spent thinking: the error names it, and the retry still runs.
    calls.clear()

    def all_thinking(url, body, **k):
        calls.append(dict(body))
        if body.get("think"):
            return {
                "message": {"content": "", "thinking": "x" * 15000},
                "done_reason": "length",
            }, 100.0
        return {"message": {"content": good}, "done_reason": "stop"}, 10.0

    monkeypatch.setattr(judge_http, "post_json", all_thinking)
    answers = LlmJudgeBackend("m").decide(inp, [BY_ID["failed"]])
    assert [c.get("think", "absent") for c in calls] == [True, False]
    assert answers["failed"].method == "llm_json_nothink"
    with pytest.raises(JudgeError, match="after 15000 chars of reasoning; JSON incomplete: ''"):
        LlmJudgeBackend("m")._ask({"think": True}, [BY_ID["failed"]], method="llm_json")

    # A server that rejects the flag outright gets the retry without it.
    calls.clear()

    def rejects_flag(url, body, **k):
        calls.append(dict(body))
        if "think" in body:
            raise JudgeError(f'{url}: HTTP 400: json: unknown field "think"')
        return {"message": {"content": good}, "done_reason": "stop"}, 10.0

    monkeypatch.setattr(judge_http, "post_json", rejects_flag)
    answers = LlmJudgeBackend("m").decide(inp, [BY_ID["failed"]])
    assert [c.get("think", "absent") for c in calls] == [True, "absent"]
    assert answers["failed"].method == "llm_json_nothink"

    # Without thinking there is nothing to fall back to; the error surfaces for the caller.
    calls.clear()
    monkeypatch.setattr(
        judge_http,
        "post_json",
        lambda u, b, **k: calls.append(dict(b)) or ({"message": {"content": ""}}, 1.0),
    )
    with pytest.raises(JudgeError, match="no JSON"):
        LlmJudgeBackend("m", think=False).decide(inp, [BY_ID["failed"]])
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

    monkeypatch.setattr(judge_http, "post_json", post)
    inp = JudgeInput(item_id="i", state="S", facts={}, flags=())
    backend = LlamaCppJudgeBackend("gemma-4-26b", host="http://llama.test:8080")
    assert backend.name == "llamacpp:gemma-4-26b"
    assert backend.local
    answers = backend.decide(inp, [BY_ID["failed"]])

    assert [u for u, _ in calls] == ["http://llama.test:8080/v1/chat/completions"] * 2
    first, second = (b for _, b in calls)
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
    # The retry turns thinking off both ways and drops the grammar.
    assert second["chat_template_kwargs"] == {"enable_thinking": False}
    assert second["reasoning_effort"] == "none"
    assert "response_format" not in second
    assert answers["failed"].value == "no"
    assert answers["failed"].p_true == pytest.approx(0.2)
    assert answers["failed"].method == "llm_json_nothink"
    assert answers["failed"].model_snapshot == "gemma-4-26b-it-Q4_K_M.gguf"
    assert answers["failed"].latency_ms == pytest.approx(10.0)

    # With thinking off from the start there is one call and nothing to fall back to.
    calls.clear()
    monkeypatch.setattr(
        judge_http,
        "post_json",
        lambda u, b, **k: calls.append((u, dict(b))) or ({"choices": [{"message": {}}]}, 1.0),
    )
    with pytest.raises(JudgeError, match="no JSON"):
        LlamaCppJudgeBackend("m", think=False).decide(inp, [BY_ID["failed"]])
    assert len(calls) == 1
    assert calls[0][1]["reasoning_effort"] == "none"

    # A server error that is not about thinking or JSON is not retried.
    calls.clear()

    def loading(u, b, **k):
        calls.append((u, dict(b)))
        raise JudgeError(f"{u}: HTTP 503: Loading model")

    monkeypatch.setattr(judge_http, "post_json", loading)
    with pytest.raises(JudgeError, match="Loading model"):
        LlamaCppJudgeBackend("m").decide(inp, [BY_ID["failed"]])
    assert len(calls) == 1

    # The response's model field is missing: the configured name stands in as the snapshot.
    monkeypatch.setattr(
        judge_http,
        "post_json",
        lambda u, b, **k: ({"choices": [{"message": {"content": good}}]}, 4.0),
    )
    answers = LlamaCppJudgeBackend("alias").decide(inp, [BY_ID["failed"]])
    assert answers["failed"].model_snapshot == "alias"
    assert answers["failed"].method == "llm_json"

    # A router that does not know the id: the error names the ids it does serve, once.
    gets: list[str] = []

    def not_found(u, b, **k):
        raise JudgeError(f'{u}: HTTP 400: {{"error":{{"message":"model \'typo\' not found"}}}}')

    monkeypatch.setattr(judge_http, "post_json", not_found)
    monkeypatch.setattr(
        judge_http,
        "get_json",
        lambda u, **k: gets.append(u) or {"data": [{"id": "org/b:Q8_0"}, {"id": "org/a:Q4"}]},
    )
    with pytest.raises(JudgeError) as excinfo:
        LlamaCppJudgeBackend("typo", host="http://r:9931").decide(inp, [BY_ID["failed"]])
    assert "model 'typo' not found" in str(excinfo.value)
    assert str(excinfo.value).endswith(". Models this server knows: org/a:Q4, org/b:Q8_0")
    assert gets == ["http://r:9931/v1/models"]
    # If the listing itself fails, the original error stands.
    monkeypatch.setattr(
        judge_http, "get_json", lambda u, **k: (_ for _ in ()).throw(JudgeError("down"))
    )
    with pytest.raises(JudgeError) as excinfo:
        LlamaCppJudgeBackend("typo").decide(inp, [BY_ID["failed"]])
    assert str(excinfo.value).endswith('not found"}}')
    assert "Models this server knows" not in str(excinfo.value)


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
        backend = LlamaCppJudgeBackend("stand-in", host=host)
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
        "EVIDENCE\nS\n\nDRAFT LESSON: Run tests first.\n\nQUESTIONS"
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
        '"grounded": {"answer": "maybe", "probability": 0.5}}\n```'
    )
    data = parse_judge_json(text)
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
    with pytest.raises(JudgeError, match="no JSON"):
        parse_judge_json("nothing here")


def test_drafter_parsing_and_spec(monkeypatch):
    from agent_memory.judge.drafter import OpenAIChatDrafter

    lesson, cited = parse_draft({"lesson": "Stop  the server\nfirst.", "cited": ["[3]", 5, "x", 3]})
    assert lesson == "Stop the server first."
    assert cited == [3, 5]
    # A declined draft is an explicit null; a missing or empty lesson is a drafter fault.
    assert parse_draft({"lesson": None, "cited": [2]}) == (None, [2])
    with pytest.raises(JudgeError, match="empty"):
        parse_draft({"lesson": ""})
    with pytest.raises(JudgeError, match="no lesson field"):
        parse_draft({"cited": [1]})
    assert draft_hash(None) is None
    assert draft_hash("x") == draft_hash("x") != draft_hash("y")
    assert make_drafter("ollama:gemma4:26b").name == "ollama:gemma4:26b"
    local = make_drafter("llamacpp:ggml-org/gpt-oss-20b-GGUF:MXFP4")
    assert isinstance(local, OpenAIChatDrafter)
    assert (local.name, local.local) == ("llamacpp:ggml-org/gpt-oss-20b-GGUF:MXFP4", True)
    assert local.url == "http://localhost:9931/v1/chat/completions"
    assert local.extra["chat_template_kwargs"] == {"enable_thinking": False}
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
            '{"lesson": null, "cited": []}'
            if len(bodies) == 2
            else ('{"lesson": "Run the tests before claiming done.", "cited": [2]}')
        )
        return {
            "model": "served-model",
            "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
            "usage": {"cost": 0.0012},
        }, 55.0

    monkeypatch.setattr(judge_http, "post_json", fake_post)
    draft = local.draft("i1", "[1] REQUEST: x\n[2] TOOL RESULT (ok): 3 passed")
    assert (draft.lesson, draft.cited, draft.declined) == (
        "Run the tests before claiming done.",
        [2],
        False,
    )
    assert draft.cost_usd == pytest.approx(0.0012)
    assert draft.model_snapshot == "served-model"
    assert draft.to_dict()["declined"] is False
    url, body, headers = bodies[0]
    assert headers == {}
    assert body["max_tokens"] == 400
    assert body["response_format"] == {"type": "json_object"}
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


def test_drafter_retries_a_cut_off_draft_with_more_room(monkeypatch):
    from agent_memory.judge.drafter import MAX_DRAFT_TOKENS, RETRY_DRAFT_TOKENS, OllamaDrafter

    full = '{"lesson": "Check the script with ShellCheck before running it.", "cited": [4]}'
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

    monkeypatch.setattr(judge_http, "post_json", openai_post)
    local = make_drafter("llamacpp:m")
    draft = local.draft("i1", "[4] TOOL RESULT: shellcheck ok")
    assert draft.lesson == "Check the script with ShellCheck before running it."
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

    monkeypatch.setattr(judge_http, "post_json", always_cut)
    with pytest.raises(JudgeError, match=r"drafter ran out of tokens after 300 chars of reasoning"):
        local.draft("i2", "state")

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

    monkeypatch.setattr(judge_http, "post_json", ollama_post)
    draft = OllamaDrafter("m").draft("i3", "state")
    assert draft.cited == [4]
    assert [b["num_predict"] for b in bodies] == [MAX_DRAFT_TOKENS, RETRY_DRAFT_TOKENS]
    assert draft.latency_ms == 100.0


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
        answers["enforced"] = ans(
            "none", {"none": 0.7, "harness": 0.1, "project_docs": 0.1, "tooling": 0.1}, 0.7
        )
    row: dict[str, Any] = {
        "run_id": "syn",
        "item_id": item_id,
        "backend": backend,
        "projection_hash": f"h{i}",
        "questions_version": "questions_v1" if v1 else "questions_v2",
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
    """A labeled run with `rules` and `llm:ref` verdicts. v2 by default: one `ollama:m` draft per
    item with per-draft labels. `v1` builds an exp2-shaped run: no drafter on verdicts,
    `supported`/`generalizable`/`enforced_by` on the item."""
    cfg = Config(home=tmp_path / "exp")
    store = RunStore(cfg, "syn")
    store.create({"questions_version": "questions_v1" if v1 else "questions_v2", "thresholds": {}})
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
                            "generalizable": 1 if failed else 0,
                            "enforced_by": "none",
                        }
                    },
                    "preferred": None,
                    "note": None,
                    "labels_version": "labels_v2",
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
    assert "v2 write rule with `llm:ref` on `ollama:m` (criterion 6" in frozen_text
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
        label_items(store, console=Console(file=io.StringIO()), ask=lambda q, c: "q")


def _scripted(answers: Sequence[str]):
    it = iter(answers)
    asked: list[str] = []

    def ask(question: str, choices):
        asked.append(question)
        return next(it)

    return ask, asked


def test_labeling_hides_verdicts_and_resumes(tmp_path):
    store = synthetic_run(tmp_path, n=3)
    store.labels_path.unlink()
    items = store.read_jsonl(store.items_path)
    buffer = io.StringIO()
    console = Console(file=buffer, width=100, force_terminal=False)
    # Two drafters on every item, so every item asks the same sequence whatever the shuffle.
    store.write_jsonl(
        store.drafts_path,
        [
            {"item_id": i["item_id"], "lesson": "DRAFT_LESSON_TEXT", "drafter": "ollama:m"}
            for i in items
        ]
        + [
            {"item_id": i["item_id"], "lesson": "OTHER_DRAFT_TEXT", "drafter": "llamacpp:x"}
            for i in items
        ],
    )
    # failed, category, worth, kind, A x4, B x4, preferred, note, save; then skip; then quit.
    ask, asked = _scripted(
        ["y", "2", "y", "3", "y", "y", "1", "1", "n", "y", "0", "1", "a", "", "y", "s", "q"]
    )
    summary = label_items(store, console=console, ask=ask, seed=5)
    out = buffer.getvalue()
    assert (summary.labeled, summary.skipped, summary.remaining) == (1, 1, 2)
    assert "PROJECTION" in out
    assert "DRAFT_LESSON_TEXT" in out
    assert "OTHER_DRAFT_TEXT" in out
    assert "draft A" in out
    assert "draft B" in out
    assert "closed by" in out
    # Each run of questions is announced, and a draft is shown again before its own questions.
    assert "about the turn itself" in out
    assert "about draft A" in out
    assert "about draft B" in out
    assert "the drafts side by side" in out
    first_item = out[: out.index("item 2/3")]
    assert first_item.count("DRAFT_LESSON_TEXT") == 2  # its own screen, then side by side
    assert (
        out.index("about the turn itself") < out.index("about draft A") < out.index("about draft B")
    )
    for hidden in ("0.85", "llm:ref", "probabilities", "backend", "ollama:m", "llamacpp:x"):
        assert hidden not in out
    assert asked[0].startswith("failed? (y/n, u unclear, i incomplete)")
    assert "lesson kind" in asked[3]
    assert asked[-3] == "save? (Enter/y save, r redo, s skip, q quit)"
    labels = store.read_jsonl(store.labels_path)
    assert len(labels) == 1
    new = labels[0]
    drafts = order_drafts(new["item_id"], store.read_drafts()[new["item_id"]], seed=5)
    a, b = drafts
    assert (new["failed"], new["incomplete"], new["category"]) == (True, False, "tool_misuse")
    assert (new["worth_remembering"], new["lesson_kind"]) == (True, "process")
    assert new["error_intended"] is None
    assert new["drafts"][a.drafter] == {
        "slot": "A",
        "draft_hash": draft_hash(
            "DRAFT_LESSON_TEXT" if a.drafter == "ollama:m" else "OTHER_DRAFT_TEXT"
        ),
        "declined": False,
        "grounded": True,
        "faithful": True,
        "generalizable": 1,
        "enforced_by": "none",
    }
    assert new["drafts"][b.drafter]["slot"] == "B"
    assert (new["drafts"][b.drafter]["grounded"], new["drafts"][b.drafter]["generalizable"]) == (
        False,
        0,
    )
    assert new["preferred"] == a.drafter
    assert new["note"] is None
    assert new["labels_version"] == "labels_v2"
    assert new["labeled_at"]
    # Resume: the labeled item is not offered again.
    again = label_items(store, console=console, ask=lambda q, c: "q", seed=5)
    assert (again.labeled, again.remaining) == (0, 2)


def test_labeling_redraws_per_section_and_pages_the_projection(tmp_path, monkeypatch):
    monkeypatch.setenv("TERM", "xterm-256color")  # Rich drops control codes on a dumb terminal
    store = synthetic_run(tmp_path, n=1)
    store.labels_path.unlink()
    item_id = "00000000-0000-0000-0000-000000000000"
    # Lesson text is untrusted: Rich markup in it must print as written, never be parsed. A
    # citation that is not in the projection is called out with what to do about it.
    store.write_jsonl(
        store.drafts_path,
        [
            {
                "item_id": item_id,
                "lesson": "DRAFT_LESSON_TEXT [bold]x [/dim]",
                "cited": [7],
                "drafter": "ollama:m",
            },
            {"item_id": item_id, "lesson": "OTHER_DRAFT_TEXT", "drafter": "llamacpp:x"},
        ],
    )
    buffer = io.StringIO()
    console = Console(file=buffer, width=100, force_terminal=False)
    # p at failed, at A.grounded, and at save pages the projection and asks again; b at
    # preferred picks draft B instead of going back.
    ask, asked = _scripted(
        ["p", "n", "n", "p", "y", "y", "1", "1", "y", "y", "1", "1", "b", "", "p", "y"]
    )
    summary = label_items(store, console=console, ask=ask, seed=5)
    out = buffer.getvalue()
    assert (summary.labeled, summary.remaining) == (1, 0)
    assert "DRAFT_LESSON_TEXT [bold]x [/dim]" in out
    slot = next(
        d.slot
        for d in order_drafts(item_id, store.read_drafts()[item_id], seed=5)
        if d.drafter == "ollama:m"
    )
    assert f"draft {slot} cites [7], which are not in the projection; judge it as" in out
    # Four screens, each with the header and the keys: the turn, draft A, draft B, side by side.
    body = out[out.index("item 1/1") :]
    assert body.count("item 1/1") == 4
    assert body.count("b back one question, p page the full projection") == 4
    assert body.count("about the turn itself") == 1
    assert body.count("about draft A") == 1
    assert body.count("about draft B") == 1
    assert body.count("the drafts side by side") == 1
    # The projection is on the turn screen, then once per p (inline when there is no terminal).
    assert body.count("SENTINEL_STATE_TEXT") == 4
    assert [q.split("?")[0] for q in asked[:2]] == ["failed", "failed"]
    assert asked[-2:] == [SAVE_PROMPT, SAVE_PROMPT]
    label = store.read_jsonl(store.labels_path)[0]
    drafts = order_drafts(item_id, store.read_drafts()[item_id], seed=5)
    assert label["preferred"] == next(d.drafter for d in drafts if d.slot == "B")
    # On a terminal every screen after the first clears the viewport.
    store.labels_path.unlink()
    buffer = io.StringIO()
    console = Console(file=buffer, width=100, force_terminal=True)
    ask, _ = _scripted(["n", "n", "y", "y", "1", "1", "y", "y", "1", "1", "a", "", "y"])
    label_items(store, console=console, ask=ask, seed=5)
    assert buffer.getvalue().count("\x1b[2J") == 3


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
    assert "Versions: unknown, unknown, questions_v2" in out


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
                "drafter": "ollama:m",
            }
        ],
    )
    buffer = io.StringIO()
    console = Console(file=buffer, width=100, force_terminal=False)
    # i = incomplete: no category; worth y, kind; one draft so no preferred; a note.
    ask, asked = _scripted(["i", "y", "1", "y", "y", "2", "2", "misread the result", "y"])
    label_items(store, console=console, ask=ask, seed=5)
    out = buffer.getvalue()
    assert "how to label" in out
    assert "enforced by" in out
    assert "cites [2] [5]" in out
    assert "cited evidence" in out
    assert "no, the tests fail" in out
    assert "[3] ASSISTANT" in out  # still in the projection panel
    assert "preferred" not in " ".join(asked)
    label = store.read_jsonl(store.labels_path)[0]
    assert (label["failed"], label["incomplete"], label["category"]) == (None, True, None)
    assert label["lesson_kind"] == "project_fact"
    entry = label["drafts"]["ollama:m"]
    assert (entry["grounded"], entry["faithful"], entry["generalizable"]) == (True, True, 2)
    assert entry["enforced_by"] == "harness"
    assert label["preferred"] is None
    assert label["note"] == "misread the result"
    assert "failed=i" in out
    assert "note=misread the result" in out


def test_labeling_backfills_relabels_and_keeps_history(tmp_path):
    from agent_memory.judge.labeling import changed_values, missing_steps, steps_for

    store = synthetic_run(tmp_path, n=4)
    ids = [f"{i:08d}-0000-0000-0000-000000000000" for i in range(4)]
    items = store.read_jsonl(store.items_path)
    items[2]["facts"]["tool_error_count"] = 1
    store.write_jsonl(store.items_path, items)
    # A second drafter arrives after labeling: a lesson on item 0, a decline on item 1.
    store.write_jsonl(
        store.drafts_path,
        [
            {"item_id": ids[0], "lesson": "Second opinion.", "drafter": "llamacpp:x"},
            {"item_id": ids[1], "lesson": None, "declined": True, "drafter": "llamacpp:x"},
        ],
        append=True,
    )
    views0 = order_drafts(ids[0], store.read_drafts()[ids[0]], seed=5)
    new_slot = next(d.slot for d in views0 if d.drafter == "llamacpp:x")
    labels = {row["item_id"]: row for row in store.read_jsonl(store.labels_path)}
    assert missing_steps(labels[ids[0]], views0, False) == [
        f"{new_slot}.grounded",
        f"{new_slot}.faithful",
        f"{new_slot}.generalizable",
        f"{new_slot}.enforced_by",
        "preferred",
    ]
    assert steps_for({"failed": "n", "worth_remembering": "n"}, [], False) == [
        "failed",
        "worth_remembering",
        "note",
    ]

    buffer = io.StringIO()
    console = Console(file=buffer, width=100, force_terminal=False)
    # item 0: the new draft's four answers and preferred; item 1: preferred only (the new draft
    # declined); item 2: error_intended only. Each closes with a save.
    ask, asked = _scripted(["y", "n", "1", "1", "same", "y", "neither", "y", "n", "y"])
    summary = label_items(store, console=console, ask=ask, seed=5)
    out = buffer.getvalue()
    assert (summary.labeled, summary.skipped, summary.remaining, summary.changed) == (3, 0, 0, 0)
    assert "3 labeled item(s) come first" in out
    assert (
        f"only asking: draft {new_slot} (grounded, faithful, generalizable, enforced by), preferred"
    ) in out
    assert "only asking: preferred" in out
    assert "only asking: error intended" in out
    rows = {row["item_id"]: row for row in store.read_jsonl(store.labels_path)}
    assert list(rows) == ids  # rewritten in place, nothing duplicated
    second = rows[ids[0]]["drafts"]["llamacpp:x"]
    assert (second["grounded"], second["faithful"], second["generalizable"]) == (True, False, 1)
    assert (second["enforced_by"], second["slot"]) == ("none", new_slot)
    assert rows[ids[0]]["preferred"] == "same"
    assert rows[ids[0]]["drafts"]["ollama:m"]["grounded"] is False  # untouched answers survive
    assert rows[ids[1]]["preferred"] == "neither"
    assert rows[ids[1]]["drafts"]["llamacpp:x"]["declined"] is True
    assert rows[ids[2]]["error_intended"] is False
    assert all(rows[i]["labeled_at"] == "2026-01-01T00:00:00+00:00" for i in ids)
    assert all("relabeled_at" in rows[i] for i in ids[:3])
    assert "relabeled_at" not in rows[ids[3]]
    assert all(not rows[i].get("history") for i in ids)  # new answers are not changes
    # Nothing left to backfill: a second pass asks nothing.
    again = label_items(store, console=console, ask=lambda q, c: "q", seed=5)
    assert (again.labeled, again.remaining) == (0, 0)

    # --item relabels in full: b steps back, r redoes from the summary, and the changed values
    # land in history. Item 3 was failed/tool_misuse/worth/process.
    ask, asked = _scripted(
        ["y", "b", "n", "n", "y", "y", "1", "1", "", "r", "n", "n", "y", "y", "1", "1", "slip", "y"]
    )
    buffer.truncate(0)
    summary = label_items(store, console=console, ask=ask, seed=5, item_ids=[ids[3]])
    out = buffer.getvalue()
    assert (summary.labeled, summary.changed) == (1, 1)
    assert "current labels" in out
    assert [q.split("?")[0] for q in asked[:4]] == [
        "failed",
        "category (number)",
        "failed",
        "worth remembering for future sessions",
    ]
    row = {r["item_id"]: r for r in store.read_jsonl(store.labels_path)}[ids[3]]
    assert (row["failed"], row["category"], row["worth_remembering"], row["lesson_kind"]) == (
        False,
        None,
        False,
        None,
    )
    assert row["note"] == "slip"
    assert len(row["history"]) == 1
    assert row["history"][0]["changed"] == {
        "failed": True,
        "category": "tool_misuse",
        "worth_remembering": True,
        "lesson_kind": "process",
    }
    # Filling in a previously unanswered key is not a change.
    assert changed_values({"note": None}, {"note": "slip"}) == {}
    assert changed_values(
        {"failed": True, "drafts": {"d": {"grounded": True}}},
        {"failed": True, "drafts": {"d": {"grounded": False}}},
    ) == {"drafts.d.grounded": True}
    with pytest.raises(JudgeError, match="no such item"):
        label_items(store, console=console, ask=ask, item_ids=["nope"])

    # --review shows current labels and asks keep or redo.
    kept = label_items(
        store, console=console, ask=_scripted(["k"])[0], seed=5, item_ids=[ids[3]], review=True
    )
    assert (kept.labeled, kept.changed) == (0, 0)
    redo, _ = _scripted(["r", "y", "2", "n", "y", "y", "1", "1", "", "y"])
    redone = label_items(store, console=console, ask=redo, seed=5, item_ids=[ids[3]], review=True)
    assert (redone.labeled, redone.changed) == (1, 1)
    row = {r["item_id"]: r for r in store.read_jsonl(store.labels_path)}[ids[3]]
    assert (row["failed"], row["category"]) == (True, "tool_misuse")
    assert len(row["history"]) == 2
    assert row["history"][1]["changed"] == {"failed": False, "note": "slip"}
    report = build_report(store)
    assert "4 value(s) changed on 1 item(s)" not in report  # 4 + 2 values moved
    assert "6 value(s) changed on 1 item(s)" in report
    assert "| llamacpp:x | 2 | 1 (50%) |" in report  # drafts, declined
    assert "1 labeled items had a tool error" in report


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
    first = store.read_jsonl(store.items_path)[0]["item_id"]
    store.write_projection(
        first, "[1] REQUEST: x\n[2] OUTCOME: none recorded\n[3] NEXT MESSAGE (human): unrelated"
    )
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
    warnings: list[str] = []
    summary = pipeline.draft_items(
        cfg, "syn", "ollama:m", store=store, log=logs.append, warn=warnings.append
    )
    assert (summary.written, summary.declined, summary.failed) == (3, 0, 1)
    assert bodies[0]["options"]["num_predict"] == 400
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
    assert rows[0]["drafter_version"] == "drafter_v2"
    # The skipped item is picked up by the next run.
    assert pipeline.draft_items(cfg, "syn", "ollama:m", store=store).written == 1
    assert isinstance(pipeline.make_drafter("ollama:m"), OllamaDrafter)
    run = store.read_run()
    assert run["drafters"]["ollama:m"]["drafts"] == 4
    assert run["drafters"]["ollama:m"]["model_snapshot"] == "m"

    # A second drafter adds its own rows and leaves the first's alone; a decline is a row too.
    def fake_openai(url, body, headers=None, **kwargs):
        content = (
            '{"lesson": null, "cited": []}'
            if len(bodies) % 2
            else ('{"lesson": "Check first.", "cited": [1]}')
        )
        bodies.append(body)
        return {
            "model": "x-served",
            "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
            "usage": {"cost": 0.001},
        }, 20.0

    monkeypatch.setattr(judge_http, "post_json", fake_openai)
    logs.clear()
    assert pipeline.draft_items(cfg, "syn", "llamacpp:x", store=store, log=logs.append).written == 4
    by_item = store.read_drafts()
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
        judge_http, "post_json", lambda *a, **k: (_ for _ in ()).throw(JudgeError("down"))
    )
    logs.clear()
    # Without a warn channel the problem still reaches the one channel there is.
    summary = pipeline.draft_items(cfg, "syn", "ollama:m", store=store, log=logs.append)
    assert (summary.written, summary.failed) == (0, 1)
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
        store.read_drafts()[seed["item_id"]][0],
        thresholds_from_run(store.read_run(), "rules"),
        run_id="syn",
    )
    assert verdict["drafter"] == "ollama:m"
    assert verdict["draft_hash"] == draft_hash("Lesson 2.")
    assert set(verdict["answers"]) == {q.id for q in ALL_QUESTIONS}
    store.write_jsonl(store.verdicts_path("rules"), [verdict])
    logs: list[str] = []
    warnings: list[str] = []
    written = pipeline.decide_items(
        cfg,
        "syn",
        ["rules"],
        store=store,
        record_events=False,
        log=logs.append,
        warn=warnings.append,
    )
    assert written == {"rules": 1}
    assert any(f"item {bad} failed" in line for line in warnings)
    assert any("1 verdict(s) failed" in line for line in warnings)
    assert not any("failed" in line for line in logs)
    verdicts = store.read_jsonl(store.verdicts_path("rules"))
    assert len(verdicts) == 2
    assert bad not in {v["item_id"] for v in verdicts}


def test_decide_items_judges_each_draft_and_reuses_stage_a(tmp_path, monkeypatch):
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
    stage_a_calls: list[str] = []
    real_decide = RulesBackend.decide

    def counting(self, inp, questions):
        if questions is STAGE_A:
            stage_a_calls.append(inp.item_id)
        return real_decide(self, inp, questions)

    monkeypatch.setattr(RulesBackend, "decide", counting)
    written = pipeline.decide_items(cfg, "syn", ["rules"], store=store, record_events=False)
    assert written == {"rules": 5}
    assert sorted(stage_a_calls) == sorted(ids)  # once per item, not per draft
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
    assert set(declined["answers"]) == {q.id for q in STAGE_A}
    full = by_key[(ids[0], "ollama:m")]
    assert set(full["answers"]) == {q.id for q in ALL_QUESTIONS}
    # The first row of an item paid for Stage A; the second copied it at no cost.
    first, second = sorted(
        (by_key[(ids[0], "llamacpp:x")], by_key[(ids[0], "ollama:m")]),
        key=lambda r: r["stage_a_reused"],
    )
    assert (first["stage_a_reused"], second["stage_a_reused"]) == (False, True)
    assert all(
        a["latency_ms"] == 0.0 for q, a in second["answers"].items() if q in {x.id for x in STAGE_A}
    )
    assert first["answers"]["failed"] == {
        **second["answers"]["failed"],
        "latency_ms": first["answers"]["failed"]["latency_ms"],
    }
    # Ledger events are keyed per drafter too, so two drafts of one turn are two events.
    created = store.read_run()["created_at"]
    keys = {
        pipeline.decision_record(
            r, {"stratum": "T0", "flags": [], "turn_event_id": "t"}, created
        ).source_key
        for r in rows
    }
    assert len(keys) == 5
    assert all(k.endswith(("/ollama:m", "/llamacpp:x")) for k in keys)

    # Resume: a late draft on item 2 gets one new row, with Stage A copied from the stored one.
    stage_a_calls.clear()
    store.write_jsonl(
        store.drafts_path,
        [{"item_id": ids[2], "lesson": "Late.", "drafter": "llamacpp:x"}],
        append=True,
    )
    assert pipeline.decide_items(cfg, "syn", ["rules"], store=store, record_events=False) == {
        "rules": 1
    }
    assert stage_a_calls == []
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
    assert pipeline.decide_items(cfg, "syn", ["rules"], store=store, record_events=False) == {
        "rules": 1
    }
    rows = store.read_jsonl(store.verdicts_path("rules"))
    assert len(rows) == 6
    redone = next(r for r in rows if (r["item_id"], r["drafter"]) == (ids[0], "ollama:m"))
    assert redone["draft_hash"] == draft_hash("Rewritten.")
    assert pipeline.decide_items(cfg, "syn", ["rules"], store=store, record_events=False) == {
        "rules": 0
    }
