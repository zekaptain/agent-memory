"""Capture building blocks: records, ids, redaction, blobs."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import pytest

from agent_memory.capture import blobs, ids
from agent_memory.capture.record import Record, RecordError, SessionInfo, parse_ts
from agent_memory.capture.redact import RULES, Redactor, redactor_for
from tests.conftest import FIXTURES

NOW = datetime(2026, 10, 25, 14, 3, 11, 412000, tzinfo=UTC)


def rec(**over: Any) -> Record:
    base: dict[str, Any] = {
        "agent_id": "factory-droid",
        "harness_session_id": "sess-1",
        "kind": "user_msg",
        "ts": NOW,
        "source_key": "msg-1",
        "content": "hello",
    }
    return Record(**{**base, **over})


# ids


def test_ids_are_stable_and_distinct():
    assert ids.session_id("factory-droid", "s") == ids.session_id("factory-droid", "s")
    assert ids.session_id("factory-droid", "s") != ids.session_id("mistral-vibe", "s")
    e = ids.event_id("factory-droid", "s", "tool_call", "k")
    assert e == ids.event_id("factory-droid", "s", "tool_call", "k")
    assert e != ids.event_id("factory-droid", "s", "tool_result", "k")
    assert e != ids.event_id("factory-droid", "s2", "tool_call", "k")
    assert ids.memory_id("a", "s", "k") != ids.action_id("a", "s", "k")


# records


def test_record_round_trips_through_json():
    original = rec(
        payload={"n": 1, "nested": {"ok": True}},
        parent_key="p",
        parent_kind="tool_call",
        seq=7,
        session=SessionInfo(cwd="/x/demo", project="demo", started_at=NOW, metadata={"t": 1}),
        memory={"content": "c", "type": "fact"},
    )
    line = original.to_json()
    assert "\n" not in line
    assert json.loads(line)["v"] == 1
    assert Record.from_json(line) == original


def test_tool_result_implies_parent_kind():
    r = rec(kind="tool_result", parent_key="toolu_1")
    assert r.effective_parent_kind == "tool_call"


@pytest.mark.parametrize(
    ("over", "message"),
    [
        ({"kind": "nope"}, "unknown event kind"),
        ({"source_key": ""}, "required"),
        ({"ts": datetime(2026, 1, 1)}, "timezone"),
        ({"kind": "note", "parent_key": "x"}, "parent_kind is required"),
    ],
)
def test_record_validation(over, message):
    with pytest.raises(RecordError, match=message):
        rec(**over)


@pytest.mark.parametrize(
    "line", ["not json", "[1, 2]", '{"v": 2, "agent_id": "a"}', '{"agent_id": "a"}']
)
def test_bad_json_records_are_rejected(line):
    with pytest.raises(RecordError):
        Record.from_json(line)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("2026-10-25T14:03:11.412Z", NOW),
        ("2026-10-25T10:03:11.412-04:00", NOW),
        (NOW.timestamp(), NOW),
        (int(NOW.timestamp() * 1000), NOW),
        (datetime(2026, 10, 25, 14, 3, 11, 412000), NOW),
    ],
)
def test_parse_ts(value, expected):
    assert parse_ts(value) == expected


# redaction


def _cases():
    return json.loads((FIXTURES / "redact_cases.json").read_text())


@pytest.mark.parametrize("case", _cases(), ids=lambda c: f"{c['rule']}:{c['text'][:20]}")
def test_redaction_rules(case):
    out, count = Redactor().text(case["text"])
    assert out == case["expect"]
    assert count == case["count"]


def test_every_rule_has_a_fixture_case():
    covered = {c["rule"] for c in _cases()}
    assert {r.name for r in RULES} <= covered


def test_server_token_is_always_redacted():
    token = "Zq8vL2mN4pR7sT1uW3xY5aB6cD9eF0gH"
    r = redactor_for(token)
    out, n = r.text(f"attach with {token} please")
    assert out == "attach with [REDACTED:mem_token] please"
    assert n == 1
    assert redactor_for(None).text(token)[1] == 0


def test_redact_value_walks_structures_and_counts():
    r = Redactor()
    out, n = r.value(
        {
            "a": "sk-ant-api03-abcdefghijklmnopqrstuvwxyz0123456789",
            "b": [1, {"c": "fine"}],
            "d": None,
        }
    )
    assert out == {"a": "[REDACTED:anthropic_key]", "b": [1, {"c": "fine"}], "d": None}
    assert n == 1


# blobs


def test_small_content_stays_inline(tmp_path):
    stored = blobs.store(tmp_path, "hello")
    assert (stored.content, stored.truncated, stored.blob_ref) == ("hello", False, None)
    assert stored.content_hash == blobs.content_hash("hello")
    assert stored.content_bytes == 5
    assert blobs.store(tmp_path, None).content_hash is None


def test_large_content_spills_to_a_blob_once(tmp_path):
    text = "é" * (blobs.BLOB_THRESHOLD + 10)  # 2 bytes per char; exceeds the threshold
    first = blobs.store(tmp_path, text)
    assert first.truncated
    assert first.blob_ref is not None
    blob = tmp_path / first.blob_ref
    assert blob.read_text() == text
    assert (blob.stat().st_mode & 0o777) == 0o600
    assert first.content is not None
    assert len(first.content.encode()) <= blobs.INLINE_BYTES
    assert first.content_bytes == len(text.encode())
    assert first.content_hash == blobs.content_hash(text)
    mtime = blob.stat().st_mtime_ns
    second = blobs.store(tmp_path, text)
    assert second == first
    assert blob.stat().st_mtime_ns == mtime
    assert not list(tmp_path.rglob("*.tmp"))
