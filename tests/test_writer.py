"""The writer against a live server: idempotence, ordering, redaction, blobs, sessions."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from agent_memory.capture import blobs, ids
from agent_memory.capture.record import Record, SessionInfo
from agent_memory.capture.redact import Redactor
from agent_memory.capture.writer import Writer, WriterError
from agent_memory.db import sql_literal

NOW = datetime(2026, 10, 25, 12, 0, tzinfo=UTC)


@pytest.fixture
def writer(live_cfg, live_client) -> Writer:
    return Writer(live_client, live_cfg.blob_dir, Redactor())


def rec(hsid: str, kind: str, key: str, **over: Any) -> Record:
    base: dict[str, Any] = {
        "agent_id": "factory-droid",
        "harness_session_id": hsid,
        "kind": kind,
        "ts": NOW,
        "source_key": key,
        "content": f"{kind} {key}",
    }
    return Record(**{**base, **over})


def rows(client, sql):
    return client.query(sql)[1]


def test_replay_is_a_no_op(writer, live_client):
    hsid = "w-replay"
    batch = [
        rec(
            hsid,
            "session_start",
            "s",
            session=SessionInfo(cwd="/x/demo", model="m1", started_at=NOW - timedelta(minutes=1)),
        ),
        rec(hsid, "user_msg", "u1", seq=1),
        rec(hsid, "tool_call", "t1", seq=2),
        rec(hsid, "tool_result", "t1", seq=3, parent_key="t1"),
    ]
    first = writer.write(batch)
    assert (first.sessions, first.events, first.skipped) == (1, 4, 0)
    second = writer.write(batch)
    assert (second.sessions, second.events, second.skipped) == (0, 0, 4)
    sid = ids.session_id("factory-droid", hsid)
    assert rows(
        live_client, f"SELECT count(*) FROM events WHERE session_id = {sql_literal(sid)}"
    ) == [(4,)]
    session = rows(
        live_client,
        "SELECT agent_id, harness_session_id, model, project, cwd, capture_completeness, "
        f"started_at::VARCHAR FROM sessions WHERE session_id = {sql_literal(sid)}",
    )
    assert session[0][:6] == ("factory-droid", hsid, "m1", "demo", "/x/demo", "partial")
    assert session[0][6].startswith("2026-10-25")


def test_results_are_linked_to_calls_even_when_listed_first(writer, live_client):
    hsid = "w-order"
    batch = [
        rec(hsid, "tool_result", "t1", seq=2, parent_key="t1"),
        rec(hsid, "tool_call", "t1", seq=1),
    ]
    assert writer.write(batch).events == 2
    sid = ids.session_id("factory-droid", hsid)
    parent = ids.event_id("factory-droid", hsid, "tool_call", "t1")
    got = rows(
        live_client,
        "SELECT parent_event_id::VARCHAR FROM events "
        f"WHERE session_id = {sql_literal(sid)} AND kind = 'tool_result'",
    )
    assert got == [(parent,)]


def test_result_whose_call_was_never_seen_is_kept_unlinked(writer, live_client):
    hsid = "w-orphan"
    result = writer.write([rec(hsid, "tool_result", "ghost", parent_key="ghost")])
    assert result.events == 1
    assert result.unresolved_parents == ["ghost"]
    sid = ids.session_id("factory-droid", hsid)
    got = rows(
        live_client,
        "SELECT parent_event_id, payload::JSON::VARCHAR FROM events "
        f"WHERE session_id = {sql_literal(sid)}",
    )
    assert got[0][0] is None
    assert '"unresolved_parent":"ghost"' in got[0][1].replace(" ", "")


def test_parent_already_in_ledger_is_linked_on_a_later_write(writer, live_client):
    hsid = "w-later"
    writer.write([rec(hsid, "tool_call", "t1")])
    writer.write([rec(hsid, "tool_result", "t1", parent_key="t1")])
    sid = ids.session_id("factory-droid", hsid)
    got = rows(
        live_client,
        f"SELECT count(*) FROM events WHERE session_id = {sql_literal(sid)} "
        "AND parent_event_id IS NOT NULL",
    )
    assert got == [(1,)]


def test_content_is_redacted_and_counted(writer, live_client):
    hsid = "w-redact"
    writer.write(
        [
            rec(
                hsid,
                "user_msg",
                "u1",
                content="key ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghij0123 and more",
                payload={"env": {"API_KEY": "API_KEY=abcdefgh12345678"}},
            )
        ]
    )
    sid = ids.session_id("factory-droid", hsid)
    got = rows(
        live_client,
        "SELECT content, redactions, payload::JSON::VARCHAR FROM events "
        f"WHERE session_id = {sql_literal(sid)}",
    )
    content, redactions, payload = got[0]
    assert content == "key [REDACTED:github_token] and more"
    assert redactions == 2
    assert "abcdefgh12345678" not in payload
    assert "[REDACTED:generic_assignment]" in payload


def test_large_content_goes_to_a_blob(writer, live_client, live_cfg):
    hsid = "w-blob"
    big = "x" * (blobs.BLOB_THRESHOLD + 1)
    writer.write([rec(hsid, "tool_result", "big", content=big)])
    sid = ids.session_id("factory-droid", hsid)
    (content, truncated, blob_ref, content_hash, content_bytes, tokens) = rows(
        live_client,
        "SELECT length(content), truncated, blob_ref, content_hash, content_bytes, token_estimate "
        f"FROM events WHERE session_id = {sql_literal(sid)}",
    )[0]
    assert (content, truncated) == (blobs.INLINE_BYTES, True)
    assert (live_cfg.blob_dir / blob_ref).read_text() == big
    assert content_hash == blobs.content_hash(big)
    assert content_bytes == len(big)
    assert tokens == len(big) // 4


def test_session_row_merges_info_and_defaults_from_tier(writer, live_client):
    hsid = "w-merge"
    writer.write(
        [
            rec(hsid, "user_msg", "u1", session=SessionInfo(cwd="/p/alpha")),
            rec(
                hsid,
                "session_start",
                "s",
                ts=NOW - timedelta(seconds=5),
                session=SessionInfo(model="m2", harness_version="9"),
            ),
        ]
    )
    sid = ids.session_id("factory-droid", hsid)
    got = rows(
        live_client,
        "SELECT model, harness_version, project, "
        "epoch(started_at) = epoch(TIMESTAMPTZ '2026-10-25 11:59:55+00') FROM sessions "
        f"WHERE session_id = {sql_literal(sid)}",
    )
    assert got[0] == ("m2", "9", "alpha", True)


def test_partial_tier_sessions_are_marked(writer, live_client):
    result = writer.write(
        [
            rec("w-amp", "user_msg", "u1", agent_id="amp"),
            rec("w-junie", "note", "n", agent_id="jetbrains-junie"),
        ]
    )
    assert result.sessions == 2
    got = rows(
        live_client,
        "SELECT agent_id, capture_completeness FROM sessions "
        "WHERE harness_session_id IN ('w-amp', 'w-junie') ORDER BY agent_id",
    )
    assert got == [("amp", "partial"), ("jetbrains-junie", "partial")]


def test_unknown_agent_is_rejected_before_anything_is_written(writer, live_client):
    before = rows(live_client, "SELECT count(*) FROM sessions")[0][0]
    with pytest.raises(WriterError, match="unknown agent"):
        writer.write([rec("w-bad", "note", "n", agent_id="letta")])
    assert rows(live_client, "SELECT count(*) FROM sessions")[0][0] == before


def test_losing_a_race_retries_and_skips(writer, live_client, monkeypatch):
    hsid = "w-race"
    batch = [rec(hsid, "user_msg", "u1"), rec(hsid, "user_msg", "u2")]
    writer.write(batch)
    # Pretend the pre-read missed the rows another capturer just wrote.
    real = writer._existing
    calls = {"n": 0}

    def flaky(table, column, values):
        if table != "events":
            return real(table, column, values)
        calls["n"] += 1
        return set() if calls["n"] == 1 else real(table, column, values)

    monkeypatch.setattr(writer, "_existing", flaky)
    result = writer.write(batch)
    assert (result.events, result.skipped) == (0, 2)
    assert calls["n"] == 2


def test_queued_memory_and_action_records_create_rows(writer, live_client):
    hsid = "w-queued"
    memory_rec = rec(
        hsid,
        "memory_write",
        "mcp:1",
        memory={
            "content": "The deploy script lives in ops/deploy.sh",
            "type": "fact",
            "scope": "demo",
        },
    )
    writer.write([memory_rec])
    mid = ids.memory_id("factory-droid", hsid, "mcp:1")
    got = rows(
        live_client,
        f"SELECT content, scope, trust_level FROM memories WHERE memory_id = {sql_literal(mid)}",
    )
    assert got == [("The deploy script lives in ops/deploy.sh", "demo", "agent")]
    sources = rows(
        live_client,
        f"SELECT relation FROM memory_sources WHERE memory_id = {sql_literal(mid)}",
    )
    assert sources == [("derived_from",)]
    action_rec = rec(
        hsid,
        "memory_write",
        "mcp:2",
        action={"memory_id": mid, "action": "retire", "reason": "moved"},
    )
    result = writer.write([action_rec, memory_rec])
    assert (result.actions, result.memories, result.skipped) == (1, 0, 1)
    assert rows(
        live_client, f"SELECT status FROM memory_status WHERE memory_id = {sql_literal(mid)}"
    ) == [("retired",)]
    again = writer.write([action_rec])
    assert (again.actions, again.skipped) == (0, 1)
