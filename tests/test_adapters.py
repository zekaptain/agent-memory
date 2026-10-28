"""Harness adapters: Droid transcripts and hooks, Vibe hooks."""

from __future__ import annotations

import pytest

from agent_memory.adapters import UnsupportedTranscript, get_adapter
from agent_memory.adapters.droid import DroidAdapter
from agent_memory.adapters.vibe import VibeAdapter
from tests.conftest import FIXTURES

TRANSCRIPT = FIXTURES / "droid_session.jsonl"


def by_key(records):
    return {(r.kind, r.source_key): r for r in records}


def test_registry_accepts_cli_names_and_agent_ids():
    assert isinstance(get_adapter("droid"), DroidAdapter)
    assert isinstance(get_adapter("factory-droid"), DroidAdapter)
    assert isinstance(get_adapter("Vibe"), VibeAdapter)
    with pytest.raises(ValueError, match="unknown harness"):
        get_adapter("letta")


# Droid transcript


def test_droid_transcript_maps_every_entry_type():
    records = DroidAdapter().from_transcript(TRANSCRIPT)
    kinds = sorted((r.kind, r.source_key) for r in records)
    assert kinds == sorted(
        [
            ("session_start", "sess-0001"),
            ("user_msg", "msg-u1"),
            ("tool_call", "toolu_001"),
            ("tool_result", "toolu_001"),
            ("assistant_msg", "msg-a2"),
            ("outcome", "turn-1"),
            ("compaction", "compact-1"),
            ("user_msg", "msg-u2"),
            ("assistant_msg", "msg-a3"),
            ("capture_error", "transcript:droid_session.jsonl:11"),
        ]
    )
    r = by_key(records)
    start = r[("session_start", "sess-0001")]
    assert start.session is not None
    assert start.session.cwd == "/Users/tester/code/demo"
    assert start.session.project == "demo"
    assert start.session.harness_version == "2"
    assert start.session.model == "test-model-1"
    assert start.session.metadata == {"title": "Fixture session"}
    assert start.content == "Fixture session"
    assert start.ts.isoformat() == "2026-10-25T10:00:00+00:00"

    user = r[("user_msg", "msg-u1")]
    assert "ghp_" in (user.content or "")  # adapters do not redact; the writer does
    assert user.seq == 1

    call = r[("tool_call", "toolu_001")]
    assert call.content == 'Execute {"command": "uv run pytest"}'
    assert call.payload == {"tool_name": "Execute", "tool_input": {"command": "uv run pytest"}}
    result = r[("tool_result", "toolu_001")]
    assert result.parent_key == "toolu_001"
    assert result.effective_parent_kind == "tool_call"
    assert result.content == "3 passed in 0.12s"

    answer = r[("assistant_msg", "msg-a2")]
    assert answer.content == "All 3 tests pass."
    assert answer.payload == {
        "modelId": "test-model-1",
        "apiProvider": "test",
        "reasoningEffort": "low",
        "userMessageSource": "ui",
    }

    doc = r[("user_msg", "msg-u2")]
    assert doc.content == "Read this spec."
    assert doc.payload is not None
    assert doc.payload["documents"] == [
        {"type": "base64", "media_type": "application/pdf", "name": "spec.pdf"}
    ]
    assert "JVBERi0" not in doc.to_json()  # never the bytes

    compaction = r[("compaction", "compact-1")]
    assert compaction.content == "User asked to run tests; they passed."
    assert compaction.payload is not None
    assert compaction.payload["removedCount"] == 4

    outcome = r[("outcome", "turn-1")]
    assert (outcome.content, outcome.payload) == ("completed", {"resultKind": "text"})
    # No timestamp of its own: it follows the entry before it so it sorts inside its turn.
    assert outcome.ts.isoformat() == "2026-10-25T10:00:06.001000+00:00"
    assert outcome.seq == 4

    after = r[("assistant_msg", "msg-a3")]
    assert after.payload is not None
    assert after.payload["compactionSummaryId"] == "compact-1"

    bad = r[("capture_error", "transcript:droid_session.jsonl:11")]
    assert bad.payload == {"line": "not json at all\n"}

    assert not any("signature" in (x.to_json()) for x in records)
    assert all(x.harness_session_id == "droid_session" for x in records)


def test_droid_transcript_thinking_is_opt_in():
    without = DroidAdapter().from_transcript(TRANSCRIPT)
    assert not any(r.source_key.endswith("#thinking") for r in without)
    with_thinking = DroidAdapter().from_transcript(TRANSCRIPT, include_thinking=True)
    thinking = by_key(with_thinking)[("assistant_msg", "msg-a1#thinking")]
    assert thinking.content == "I should run pytest."
    assert thinking.payload == {"part": "thinking", "durationMs": 120}


def test_droid_transcript_since_key_skips_known_entries():
    records = DroidAdapter().from_transcript(TRANSCRIPT, since_key="msg-a2")
    keys = [(r.kind, r.source_key) for r in records]
    assert ("user_msg", "msg-u1") not in keys
    assert ("session_start", "sess-0001") not in keys
    assert ("compaction", "compact-1") in keys
    assert ("user_msg", "msg-u2") in keys
    assert records[0].session is None  # nothing re-sends the session block


def test_droid_transcript_rejects_other_files(tmp_path):
    other = tmp_path / "worker-transcripts.json"
    other.write_text("{}")
    with pytest.raises(UnsupportedTranscript):
        DroidAdapter().from_transcript(other)


# Droid hooks


def hook(event: str, **extra) -> dict:
    return {
        "session_id": "abc-123",
        "transcript_path": "/Users/tester/.factory/sessions/x/abc-123.jsonl",
        "cwd": "/Users/tester/code/demo",
        "permission_mode": "default",
        "hook_event_name": event,
        **extra,
    }


def test_droid_session_start_hook():
    records = DroidAdapter().from_hook(hook("SessionStart", source="resume", message_id="m0"))
    (r,) = records
    assert r.kind == "session_start"
    assert r.source_key == "SessionStart:resume:m0"
    assert r.harness_session_id == "abc-123"
    assert r.session is not None
    assert r.session.project == "demo"
    assert r.session.started_at == r.ts
    assert r.session.metadata == {
        "transcript_path": "/Users/tester/.factory/sessions/x/abc-123.jsonl",
        "permission_mode": "default",
    }
    assert r.payload == {
        "hook": "SessionStart",
        "source": "resume",
        "message_id": "m0",
        "permission_mode": "default",
    }


def test_droid_prompt_hook():
    (r,) = DroidAdapter().from_hook(hook("UserPromptSubmit", prompt="hi there", message_id="m1"))
    assert (r.kind, r.content, r.source_key) == ("user_msg", "hi there", "UserPromptSubmit:m1")
    assert r.payload is not None
    assert "prompt" not in r.payload


def test_droid_post_tool_hook_links_result_to_call():
    call, result = DroidAdapter().from_hook(
        hook(
            "PostToolUse",
            tool_name="Read",
            tool_input={"file_path": "/x"},
            tool_response={"content": "data"},
            tool_use_id="toolu_9",
        )
    )
    assert (call.kind, call.source_key) == ("tool_call", "toolu_9")
    assert call.content == 'Read {"file_path": "/x"}'
    assert (result.kind, result.parent_key) == ("tool_result", "toolu_9")
    assert result.content == '{"content": "data"}'


def test_droid_pre_tool_hook_is_off_by_default():
    payload = hook("PreToolUse", tool_name="Read", tool_input={}, message_id="m2")
    assert DroidAdapter().from_hook(payload) == []
    (note,) = DroidAdapter(record_pre_tool=True).from_hook(payload)
    assert (note.kind, note.source_key) == ("note", "PreToolUse:m2")


@pytest.mark.parametrize("event", ["Stop"])
def test_droid_turn_end_hooks_record_nothing(event):
    assert DroidAdapter().from_hook(hook(event, stop_hook_active=False)) == []


def test_droid_precompact_and_session_end():
    (c,) = DroidAdapter().from_hook(
        hook("PreCompact", trigger="auto", custom_instructions="keep decisions", message_id="m3")
    )
    assert (c.kind, c.content) == ("compaction", "keep decisions")
    assert c.payload is not None
    assert c.payload["trigger"] == "auto"
    (e,) = DroidAdapter().from_hook(hook("SessionEnd", reason="exit"))
    assert e.kind == "session_end"
    assert e.payload == {"hook": "SessionEnd", "reason": "exit", "permission_mode": "default"}


def test_droid_unknown_hook_becomes_a_note():
    (n,) = DroidAdapter().from_hook(hook("BrandNewEvent", detail=1))
    assert n.kind == "note"
    assert n.payload is not None
    assert n.payload["detail"] == 1


def test_droid_hook_requires_identity():
    with pytest.raises(ValueError, match="session_id"):
        DroidAdapter().from_hook({"hook_event_name": "Stop"})


# Vibe


def vibe(event: str, **extra) -> dict:
    return {
        "session_id": "v-1",
        "parent_session_id": None,
        "transcript_path": "/Users/tester/.vibe/logs/session/v-1.jsonl",
        "cwd": "/Users/tester/code/demo",
        "hook_event_name": event,
        **extra,
    }


def test_vibe_post_tool_hook():
    call, result = VibeAdapter().from_hook(
        vibe(
            "post_tool",
            tool_name="bash",
            tool_input={"command": "ls"},
            tool_call_id="tc-1",
            tool_status="success",
            tool_output={"stdout": "a\nb"},
            tool_output_text="a\nb",
        )
    )
    assert call.agent_id == "mistral-vibe"
    assert (call.kind, call.source_key, call.content) == (
        "tool_call",
        "tc-1",
        'bash {"command": "ls"}',
    )
    assert (result.kind, result.parent_key, result.content) == ("tool_result", "tc-1", "a\nb")
    assert result.payload is not None
    assert result.payload["tool_status"] == "success"
    assert call.session is not None
    assert call.session.metadata == {
        "transcript_path": "/Users/tester/.vibe/logs/session/v-1.jsonl"
    }


def test_vibe_post_agent_and_pre_tool():
    assert VibeAdapter().from_hook(vibe("post_agent")) == []
    assert VibeAdapter().from_hook(vibe("pre_tool", tool_call_id="tc-2", tool_name="bash")) == []
    (n,) = VibeAdapter(record_pre_tool=True).from_hook(
        vibe("pre_tool", tool_call_id="tc-2", tool_name="bash", tool_input={})
    )
    assert (n.kind, n.source_key) == ("note", "pre_tool:tc-2")


def test_vibe_transcripts_are_not_mapped_yet(tmp_path):
    path = tmp_path / "v-1.jsonl"
    path.write_text("{}\n")
    with pytest.raises(UnsupportedTranscript, match="Phase 2"):
        VibeAdapter().from_transcript(path)
