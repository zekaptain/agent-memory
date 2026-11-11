"""The Droid adapter: its transcripts and its hooks."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from agent_memory.adapters import UnsupportedTranscript, get_adapter
from agent_memory.adapters.droid import HOOK_EVENTS, DroidAdapter
from tests.conftest import FIXTURES

TRANSCRIPT = FIXTURES / "droid_session.jsonl"


def by_key(records):
    return {(r.kind, r.source_key): r for r in records}


def test_registry_accepts_cli_names_and_agent_ids():
    assert isinstance(get_adapter("droid"), DroidAdapter)
    assert isinstance(get_adapter("Factory-Droid"), DroidAdapter)
    # A registered agent's harness is not thereby one with an adapter.
    for name in ("vibe", "mistral-vibe", "letta"):
        with pytest.raises(ValueError, match="no adapter for harness"):
            get_adapter(name)


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


def said(entry_id: str, role: str, text: str, at: str) -> dict:
    return {
        "type": "message",
        "id": entry_id,
        "timestamp": at,
        "message": {"role": role, "content": [{"type": "text", "text": text}]},
    }


def two_turns() -> list[dict]:
    """As Droid writes them: the session's first entry has no timestamp, and an outcome has
    none either and names its turn by the id of the turn's prompt."""
    return [
        {"type": "session_start", "id": "two", "title": "Two turns", "cwd": "/x/demo"},
        said("p1", "user", "Run it.", "2026-11-11T10:00:01Z"),
        said("r1", "assistant", "Done.", "2026-11-11T10:00:02Z"),
        {"type": "agent_turn_outcome", "turnId": "p1", "reason": "completed", "resultKind": "text"},
        said("p2", "user", "Again.", "2026-11-11T10:05:01Z"),
        said("r2", "assistant", "Done again.", "2026-11-11T10:05:02Z"),
    ]


def write(path: Path, entries: list[dict]) -> Path:
    path.write_text("".join(json.dumps(entry) + "\n" for entry in entries))
    return path


def keys_of(records) -> list[tuple[str, str]]:
    return [(r.kind, r.source_key) for r in records]


def covered(records) -> dict:
    assert records[0].observation is not None
    return records[0].observation


def test_droid_session_start_without_a_time_is_placed_just_before_the_first_timed_entry(tmp_path):
    path = write(tmp_path / "two.jsonl", two_turns())
    records = DroidAdapter().from_transcript(path)
    start = records[0]
    assert start.kind == "session_start"
    assert start.ts.isoformat() == "2026-11-11T10:00:00.999000+00:00"
    assert start.session is not None
    assert start.session.started_at == start.ts
    # What the read covered is what the file says, not when it was read.
    assert covered(records)["covered_from"] == start.ts
    assert covered(records)["covered_through"].isoformat() == "2026-11-11T10:05:02+00:00"
    # The file says the same thing however often it is read.
    again = DroidAdapter().from_transcript(path)
    assert [r.to_json() for r in again] == [r.to_json() for r in records]


def test_droid_transcript_is_read_on_from_the_line_an_earlier_read_ended_on(tmp_path):
    entries = two_turns()
    path = write(tmp_path / "two.jsonl", entries[:4])
    earlier = covered(DroidAdapter().from_transcript(path))
    # The read ended on an outcome, and so on the id of its turn's prompt, two lines up.
    assert (earlier["cursor_end"], earlier["range_end"]) == ("p1", 4)
    write(path, entries)
    records = DroidAdapter().from_transcript(path, since_key="p1", since_line=4)
    assert keys_of(records) == [("user_msg", "p2"), ("assistant_msg", "r2")]
    assert [
        covered(records)[name]
        for name in ("cursor_start", "cursor_end", "range_start", "range_end")
    ] == ["p1", "r2", 5, 6]
    # By the id alone the read goes on from the prompt and brings its reply again.
    assert ("assistant_msg", "r1") in keys_of(DroidAdapter().from_transcript(path, since_key="p1"))
    # Nothing was written after the last line.
    assert DroidAdapter().from_transcript(path, since_key="r2", since_line=6) == []


@pytest.mark.parametrize(
    ("since_key", "since_line"),
    [("gone", 4), ("p1", 3), ("r2", 9)],
    ids=["no such entry", "entry on another line", "file is shorter"],
)
def test_droid_transcript_is_read_whole_when_an_earlier_read_cannot_be_placed(
    tmp_path, since_key, since_line
):
    path = write(tmp_path / "two.jsonl", two_turns())
    whole = DroidAdapter().from_transcript(path)
    records = DroidAdapter().from_transcript(path, since_key=since_key, since_line=since_line)
    assert keys_of(records) == keys_of(whole)
    assert (covered(records)["cursor_start"], covered(records)["range_start"]) == (None, 1)


def test_droid_transcript_line_cut_short_is_brought_by_the_next_read(tmp_path):
    # A file read while Droid writes to it can end part way through a line.
    entries = two_turns()
    path = tmp_path / "two.jsonl"
    whole_lines = "".join(json.dumps(entry) + "\n" for entry in entries[:3])
    path.write_text(whole_lines + json.dumps(entries[3])[:20])
    earlier = DroidAdapter().from_transcript(path)
    assert earlier[-1].kind == "capture_error"
    assert (covered(earlier)["cursor_end"], covered(earlier)["range_end"]) == ("r1", 4)
    write(path, entries)
    records = DroidAdapter().from_transcript(path, since_key="r1", since_line=4)
    assert ("outcome", "p1") in keys_of(records)
    assert ("assistant_msg", "r2") in keys_of(records)


def hook_run(entry_id: str, event: str, parent: str, at: str) -> dict:
    """The entry Droid adds to a transcript for a hook command it ran."""
    return {
        "type": "message",
        "id": entry_id,
        "parentId": parent,
        "timestamp": at,
        "message": {
            "role": "user",
            "content": [],
            "visibility": "user_only",
            "hookEventName": event,
            "hookParentId": parent,
            "hookStatus": "completed",
            "hookCommands": [{"type": "command", "command": "mem event --harness droid"}],
            "hookResults": [{"exitCode": 0, "stdout": "", "stderr": "said by the hook"}],
        },
    }


def test_droid_transcript_entry_for_a_hook_run_is_not_an_event(tmp_path):
    # With a hook registered Droid writes an entry for each run of it. The one for a Stop
    # hook falls between a reply and the outcome of its turn.
    path = tmp_path / "hooked.jsonl"
    entries = [
        {"type": "session_start", "id": "hooked", "timestamp": "2026-11-11T10:00:00Z"},
        {
            "type": "message",
            "id": "p1",
            "timestamp": "2026-11-11T10:00:01Z",
            "message": {"role": "user", "content": [{"type": "text", "text": "Run it."}]},
        },
        {
            "type": "message",
            "id": "r1",
            "timestamp": "2026-11-11T10:00:02Z",
            "message": {"role": "assistant", "content": [{"type": "text", "text": "Done."}]},
        },
        hook_run("h1", "Stop", "r1", "2026-11-11T10:00:03Z"),
        {"type": "agent_turn_outcome", "turnId": "p1", "reason": "completed", "resultKind": "text"},
        hook_run("h2", "SessionEnd", "r1", "2026-11-11T10:30:03Z"),
    ]
    path.write_text("\n".join(json.dumps(entry) for entry in entries) + "\n")
    records = DroidAdapter().from_transcript(path)
    assert [(r.kind, r.source_key) for r in records] == [
        ("session_start", "hooked"),
        ("user_msg", "p1"),
        ("assistant_msg", "r1"),
        ("outcome", "p1"),
    ]
    reply = by_key(records)[("assistant_msg", "r1")]
    assert reply.attribution is not None
    assert (reply.attribution["turn_id"], reply.attribution["turn_complete"]) == ("p1", True)
    assert "said by the hook" not in "".join(r.to_json() for r in records)


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


def test_droid_startup_hook_is_the_start_the_transcript_records():
    # A session starts once, and the transcript's first entry names that start by the
    # session's id. The hook says the same thing, so it says it under the same key.
    (r,) = DroidAdapter().from_hook(hook("SessionStart", source="startup"))
    assert (r.kind, r.source_key) == ("session_start", "abc-123")
    assert r.payload == {"hook": "SessionStart", "source": "startup", "permission_mode": "default"}
    # A start that does not say what kind it is cannot be taken for the first.
    (unsaid,) = DroidAdapter().from_hook(hook("SessionStart", message_id="m0"))
    assert unsaid.source_key == "SessionStart:startup:m0"


def test_droid_prompt_hook():
    (r,) = DroidAdapter().from_hook(hook("UserPromptSubmit", prompt="hi there", message_id="m1"))
    assert (r.kind, r.content, r.source_key) == ("user_msg", "hi there", "m1")
    assert r.payload is not None
    assert "prompt" not in r.payload


def test_droid_prompt_hook_flattens_a_prompt_that_is_not_text():
    parts = [{"type": "text", "text": "look at this"}, {"type": "image", "source": "clipboard"}]
    (r,) = DroidAdapter().from_hook(hook("UserPromptSubmit", prompt=parts, message_id="m2"))
    assert r.content == 'look at this\n{"source": "clipboard", "type": "image"}'
    (r,) = DroidAdapter().from_hook(hook("UserPromptSubmit", message_id="m3"))
    assert r.content == ""


def test_droid_post_tool_hook_records_nothing():
    # Droid names no call in this hook, so what it wrote could not be told from the
    # transcript's record of the same call. The transcript is the record of tool calls.
    payload = hook(
        "PostToolUse",
        tool_name="Read",
        tool_input={"file_path": "/x"},
        tool_response=[{"type": "text", "text": "data"}],
    )
    assert DroidAdapter().from_hook(payload) == []
    assert DroidAdapter().from_hook({**payload, "tool_use_id": "toolu_9"}) == []


def test_droid_pre_tool_hook_is_off_by_default():
    payload = hook("PreToolUse", tool_name="Read", tool_input={})
    assert DroidAdapter().from_hook(payload) == []
    (note,) = DroidAdapter(record_pre_tool=True).from_hook(payload)
    assert (note.kind, note.source_key) == ("note", f"PreToolUse:{note.ts.isoformat()}")
    assert note.content == "Read {}"


@pytest.mark.parametrize("event", ["Notification", "SubagentStop", "PreToolUse", "BrandNewEvent"])
def test_droid_hooks_one_message_can_raise_twice_are_separate_events(event, monkeypatch):
    # The message's id does not tell the first from the second, so when each ran does.
    ran_at = iter(datetime(2026, 11, 11, 12, 0, second, tzinfo=UTC) for second in (1, 2))
    monkeypatch.setattr("agent_memory.adapters.droid.now", lambda: next(ran_at))
    adapter = DroidAdapter(record_pre_tool=True)
    payload = hook(event, message_id="m7", tool_name="Read", tool_input={})
    (first,) = adapter.from_hook(payload)
    (second,) = adapter.from_hook(payload)
    assert first.source_key == f"{event}:2026-11-11T12:00:01+00:00"
    assert second.source_key == f"{event}:2026-11-11T12:00:02+00:00"
    assert first.attribution == {"message_id": "m7"}


@pytest.mark.parametrize("event", ["Stop"])
def test_droid_turn_end_hooks_record_nothing(event):
    assert DroidAdapter().from_hook(hook(event, stop_hook_active=False)) == []


def test_droid_says_at_which_hooks_its_transcript_is_read():
    # The reply is on disk when Stop runs and the last turn's outcome when the session ends.
    transcript = Path("/Users/tester/.factory/sessions/x/abc-123.jsonl")
    for event in ("Stop", "SessionEnd"):
        assert DroidAdapter().transcript_due(hook(event)) == transcript
    for event in (
        "SessionStart",
        "UserPromptSubmit",
        "PreCompact",
        "Notification",
        "SubagentStop",
        "PreToolUse",
        "PostToolUse",
    ):
        assert DroidAdapter().transcript_due(hook(event)) is None
    assert DroidAdapter().transcript_due({"session_id": "s", "hook_event_name": "Stop"}) is None


def test_droid_names_the_hook_events_capture_needs():
    # Those at which the adapter records something or reads the transcript, and no other.
    known = [
        "SessionStart",
        "SessionEnd",
        "UserPromptSubmit",
        "PreCompact",
        "SubagentStop",
        "Notification",
        "Stop",
        "PreToolUse",
        "PostToolUse",
    ]
    adapter = DroidAdapter()
    needed = [
        event
        for event in known
        if adapter.from_hook(hook(event)) or adapter.transcript_due(hook(event))
    ]
    assert list(HOOK_EVENTS) == list(adapter.hook_events) == needed
    assert needed == known[:7]
    # Asked to record a note before each tool call, it needs that hook event too.
    noting = DroidAdapter(record_pre_tool=True)
    assert list(noting.hook_events) == [
        event
        for event in known
        if noting.from_hook(hook(event)) or noting.transcript_due(hook(event))
    ]
    assert list(noting.hook_events) == known[:8]


def test_droid_precompact_and_session_end():
    (c,) = DroidAdapter().from_hook(
        hook("PreCompact", trigger="auto", custom_instructions="keep decisions", message_id="m3")
    )
    # The hook says a compaction was asked for and names the message it follows. The
    # compaction itself is the transcript's to record.
    assert (c.kind, c.content, c.source_key) == ("note", "keep decisions", "PreCompact:m3")
    assert c.payload is not None
    assert c.payload["trigger"] == "auto"
    (e,) = DroidAdapter().from_hook(hook("SessionEnd", reason="exit"))
    assert e.kind == "session_end"
    assert e.payload == {"hook": "SessionEnd", "reason": "exit", "permission_mode": "default"}


def test_droid_session_end_is_placed_by_the_last_message():
    # What Droid sends when it ends a session nobody has used for half an hour. The id is
    # that of the session's last message, which the hook for a resume names too.
    payload = hook(
        "SessionEnd", reason="other", message_id="m9", message_count=4, session_duration_ms=9.5
    )
    (e,) = DroidAdapter().from_hook(payload)
    assert (e.kind, e.source_key, e.content) == ("session_end", "SessionEnd:m9", None)
    assert e.payload == {
        "hook": "SessionEnd",
        "reason": "other",
        "message_id": "m9",
        "message_count": 4,
        "session_duration_ms": 9.5,
        "permission_mode": "default",
    }
    # Ended again with nothing said since, the session has ended nowhere new.
    (again,) = DroidAdapter().from_hook(payload)
    assert again.source_key == e.source_key


def test_droid_precompact_instructions_that_are_not_text_become_text():
    (as_object,) = DroidAdapter().from_hook(
        hook("PreCompact", custom_instructions={"keep": "decisions"}, message_id="m4")
    )
    assert as_object.content == '{"keep": "decisions"}'
    (as_parts,) = DroidAdapter().from_hook(
        hook(
            "PreCompact",
            custom_instructions=[{"type": "text", "text": "keep decisions"}],
            message_id="m5",
        )
    )
    assert as_parts.content == "keep decisions"
    (empty,) = DroidAdapter().from_hook(hook("PreCompact", custom_instructions="", message_id="m6"))
    assert empty.content is None


def test_droid_notification_hook_keeps_what_it_said():
    (n,) = DroidAdapter().from_hook(
        hook(
            "Notification",
            message="Droid needs your input",
            notification_type="elicitation_dialog",
        )
    )
    assert (n.kind, n.content) == ("note", "Droid needs your input")
    assert n.payload is not None
    assert n.payload["notification_type"] == "elicitation_dialog"
    assert n.attribution == {}


def test_droid_unknown_hook_becomes_a_note():
    (n,) = DroidAdapter().from_hook(hook("BrandNewEvent", detail=1))
    assert n.kind == "note"
    assert n.payload is not None
    assert n.payload["detail"] == 1


def test_droid_hook_requires_identity():
    with pytest.raises(ValueError, match="session_id"):
        DroidAdapter().from_hook({"hook_event_name": "Stop"})
