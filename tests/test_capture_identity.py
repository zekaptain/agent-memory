"""Identity regressions through hooks, transcript import, spool replay, and MCP."""

from __future__ import annotations

import io
import json
import sys
from dataclasses import replace

import duckdb
import pytest

from agent_memory.adapters.droid import DroidAdapter
from agent_memory.capture import identity, ids
from agent_memory.capture.pipeline import deliver
from agent_memory.cli import main
from agent_memory.db import sql_literal
from agent_memory.mcp_server import MemoryTools
from tests.conftest import FIXTURES
from tests.support import new_id


def transcript(tmp_path, retrieval=None, placeholder=None):
    path = tmp_path / f"{tmp_path.name}.jsonl"
    text = (FIXTURES / "droid_identity.jsonl").read_text()
    path.write_text(
        text.replace("@RETRIEVAL_ID@", retrieval or new_id()).replace(
            "@PLACEHOLDER_ID@", placeholder or new_id()
        )
    )
    return path


def run(cfg, *args):
    return main(["--home", str(cfg.home), "--port", str(cfg.port), *args])


def test_prompt_hook_import_and_spool_replay_are_one_event(
    live_cfg, live_client, tmp_path, monkeypatch
):
    path = transcript(tmp_path)
    payload = {
        "session_id": path.stem,
        "hook_event_name": "UserPromptSubmit",
        "message_id": "prompt-1",
        "prompt": "What do we know?",
    }
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(payload)))
    assert run(live_cfg, "event", "--harness", "droid") == 0
    records = DroidAdapter().from_hook(payload)
    assert run(live_cfg, "ingest", "--harness", "droid", "--transcript", str(path)) == 0
    assert deliver(live_cfg, records, spool_only=True).spooled
    assert run(live_cfg, "ingest", "--spool") == 0
    sid = ids.session_id("factory-droid", path.stem)
    assert live_client.query(
        f"SELECT count(*) FROM events WHERE session_id = {sql_literal(sid)} AND kind = 'user_msg'"
    )[1] == [(1,)]


def droid_hook(path, event, **fields):
    """A hook payload with the fields Droid sends on every event."""
    return {
        "session_id": path.stem,
        "transcript_path": str(path),
        "cwd": "/synthetic/identity",
        "permission_mode": "auto-high",
        "hook_event_name": event,
        **fields,
    }


def test_a_tool_call_is_one_event_when_a_hook_and_the_transcript_report_it(
    live_cfg, live_client, tmp_path, monkeypatch
):
    path = transcript(tmp_path)
    # What Droid sends after a tool call. Nothing in it names the call.
    payload = droid_hook(
        path,
        "PostToolUse",
        tool_name="agent-memory__recall",
        tool_input={"query": "synthetic"},
        tool_response=[{"type": "text", "text": "synthetic"}],
    )
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(payload)))
    assert run(live_cfg, "event", "--harness", "droid") == 0
    assert run(live_cfg, "ingest", "--harness", "droid", "--transcript", str(path)) == 0
    # A replay of what the import wrote finds the same events.
    assert deliver(live_cfg, DroidAdapter().from_transcript(path), spool_only=True).spooled
    assert run(live_cfg, "ingest", "--spool") == 0
    sid = ids.session_id("factory-droid", path.stem)
    assert live_client.query(
        f"SELECT kind, count(*) FROM events WHERE session_id = {sql_literal(sid)} "
        "AND kind IN ('tool_call', 'tool_result') GROUP BY kind ORDER BY kind"
    )[1] == [("tool_call", 2), ("tool_result", 2)]
    call = ids.event_id("factory-droid", path.stem, "tool_call", "recall-call")
    result = ids.event_id("factory-droid", path.stem, "tool_result", "recall-call")
    assert live_client.query(
        "SELECT kind, parent_event_id::VARCHAR FROM events "
        f"WHERE event_id IN ({sql_literal(call)}, {sql_literal(result)}) ORDER BY kind"
    )[1] == [("tool_call", None), ("tool_result", call)]


def test_a_compaction_is_one_event_when_a_hook_announced_it(
    live_cfg, live_client, tmp_path, monkeypatch
):
    path = tmp_path / f"{tmp_path.name}.jsonl"
    path.write_bytes((FIXTURES / "droid_session.jsonl").read_bytes())
    # Droid's hook names the message the compaction follows, not the compaction.
    payload = droid_hook(
        path, "PreCompact", message_id="msg-a2", trigger="auto", estimated_tokens=900
    )
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(payload)))
    assert run(live_cfg, "event", "--harness", "droid") == 0
    run(live_cfg, "ingest", "--harness", "droid", "--transcript", str(path))
    sid = ids.session_id("factory-droid", path.stem)
    assert live_client.query(
        f"SELECT compactions FROM session_summary WHERE session_id = {sql_literal(sid)}"
    )[1] == [(1,)]
    assert live_client.query(
        "SELECT kind, json_extract_string(payload::JSON, '$.trigger') FROM events "
        f"WHERE session_id = {sql_literal(sid)} "
        "AND json_extract_string(payload::JSON, '$.hook') = 'PreCompact'"
    )[1] == [("note", "auto")]


@pytest.mark.parametrize("import_first", [False, True])
def test_own_mcp_session_binds_captured_reply_and_cleans_audit(
    live_cfg, live_client, tmp_path, monkeypatch, capsys, import_first
):
    monkeypatch.delenv("AGENT_MEMORY_SESSION", raising=False)
    tools = MemoryTools(live_cfg, "factory-droid")
    try:
        kept = tools.remember("Synthetic identity fact", "fact")
        recalled = tools.recall("synthetic")
        placeholder = new_id()
        original = tools._event

        def event(client, kind, content, payload):
            row = original(client, kind, content, payload)
            if kind == "assistant_msg":
                row["event_id"] = placeholder
            return row

        monkeypatch.setattr(tools, "_event", event)
        path = transcript(tmp_path, recalled["retrieval_id"], placeholder)
        if import_first:
            assert run(live_cfg, "ingest", "--harness", "droid", "--transcript", str(path)) == 0
        cited = tools.cite(recalled["retrieval_id"], [kept["memory_id"]])
        assert not cited["degraded"], cited
        assert cited["answer_event_id"] == placeholder
        if import_first:
            assert live_client.query(
                "SELECT reason_code FROM effective_answer_bindings "
                f"WHERE placeholder_event_id = {sql_literal(placeholder)}"
            )[1] == [("unique_captured_turn",)]
        assert run(live_cfg, "ingest", "--harness", "droid", "--transcript", str(path)) == 0
        answer = ids.event_id("factory-droid", path.stem, "assistant_msg", "reply")
        rows = live_client.query(
            "SELECT answer_event_id::VARCHAR, reason_code, evidence::JSON::VARCHAR "
            "FROM effective_answer_bindings "
            f"WHERE placeholder_event_id = {sql_literal(placeholder)}"
        )[1]
        assert len(rows) == 1
        assert rows[0][:2] == (answer, "unique_captured_turn")
        assert json.loads(rows[0][2])[0]["placeholder_session_id"] == tools.session_id
        assert live_client.query(
            "SELECT DISTINCT captured_in_context, declared_used FROM why_answer "
            f"WHERE answer_event_id = {sql_literal(answer)} AND via = 'retrieval' "
            f"AND retrieval_id = {sql_literal(recalled['retrieval_id'])}"
        )[1] == [(True, True)]
        capsys.readouterr()
        assert run(live_cfg, "audit", "check", "--json") in (0, 1)
        report = json.loads(capsys.readouterr().out)
        assert not any(
            f["code"] in {"cite_placeholder_unresolved", "answer_binding_invalid"}
            and f["entity_id"] == placeholder
            for f in report["findings"]
        )
        assert not any(
            f["code"] == "retrieval_missing_attribution"
            and f["entity_id"] == recalled["retrieval_id"]
            for f in report["findings"]
        )
    finally:
        tools.close()


def test_unkeyed_hook_prompt_is_marked():
    record = DroidAdapter().from_hook(
        {"session_id": "unkeyed", "hook_event_name": "UserPromptSubmit", "prompt": "hi"}
    )[0]
    assert record.source_key == f"prompt:unkeyed:{record.ts.isoformat()}"
    assert record.payload is not None
    assert record.payload["dedupe"] == "unkeyed"


def test_identity_builders_preserve_existing_keys_and_hashes():
    import uuid
    from datetime import UTC, datetime

    from agent_memory.capture.observations import fingerprint

    ts = datetime(2026, 11, 9, tzinfo=UTC)
    assert identity.prompt_key("message-1", ts) == "message-1"
    assert identity.reply_key("message-1") == "message-1"
    assert identity.thinking_key("message-1") == "message-1#thinking"
    assert identity.tool_call_key("tool-1") == "tool-1"
    assert identity.tool_result_key("tool-1") == "tool-1"
    assert identity.session_marker_key("SessionStart", "resume", "m1") == "SessionStart:resume:m1"
    assert identity.session_marker_key("SessionEnd", None, "m1") == "SessionEnd:m1"
    assert identity.note_key("Notification", "m1") == "Notification:m1"
    for previous in (None, "prior-binding"):
        expected = str(
            uuid.uuid5(
                uuid.NAMESPACE_URL,
                "agent-memory/answer-binding/" + fingerprint(["placeholder", previous]),
            )
        )
        assert identity.binding_id("placeholder", previous) == expected


@pytest.mark.parametrize("value", [None, "", "harness-session"])
def test_mcp_session_identity_and_metadata(live_cfg, live_client, monkeypatch, value):
    if value is None:
        monkeypatch.delenv("AGENT_MEMORY_SESSION", raising=False)
    else:
        monkeypatch.setenv("AGENT_MEMORY_SESSION", value)
    tools = MemoryTools(live_cfg, "factory-droid")
    try:
        assert tools.own_session is (not bool(value))
        if value:
            assert tools.harness_session_id == value
        else:
            assert tools.harness_session_id.startswith("mcp-")
        tools.status()
        metadata = live_client.query(
            "SELECT metadata::JSON::VARCHAR FROM sessions "
            f"WHERE session_id = {sql_literal(tools.session_id)}"
        )[1]
        assert json.loads(metadata[0][0]) == {
            "via": "mem-mcp",
            "own_session": not bool(value),
            "session_link": "env" if value else "own",
        }
    finally:
        tools.close()


def test_explicit_future_reply_binds_across_mcp_session(
    live_cfg, live_client, tmp_path, monkeypatch
):
    monkeypatch.delenv("AGENT_MEMORY_SESSION", raising=False)
    tools = MemoryTools(live_cfg, "factory-droid")
    try:
        recalled = tools.recall("synthetic")
        path = transcript(tmp_path, recalled["retrieval_id"])
        actual = ids.event_id("factory-droid", path.stem, "assistant_msg", "reply")
        cited = tools.cite(recalled["retrieval_id"], answer_event_id=actual)
        assert not cited["degraded"], cited
        placeholder = cited["placeholder_event_id"]
        assert run(live_cfg, "ingest", "--harness", "droid", "--transcript", str(path)) == 0
        row = live_client.query(
            "SELECT answer_event_id::VARCHAR, reason_code, evidence::JSON::VARCHAR "
            "FROM effective_answer_bindings "
            f"WHERE placeholder_event_id = {sql_literal(placeholder)}"
        )[1][0]
        assert row[:2] == (actual, "explicit_answer_id")
        assert json.loads(row[2])["placeholder_session_id"] == tools.session_id
    finally:
        tools.close()


def test_transcript_side_events_cannot_cross_session_boundary(
    live_cfg, live_client, tmp_path, monkeypatch
):
    monkeypatch.delenv("AGENT_MEMORY_SESSION", raising=False)
    tools = MemoryTools(live_cfg, "factory-droid")
    try:
        recalled = tools.recall("synthetic")
        placeholder = tools.cite(recalled["retrieval_id"])["answer_event_id"]
        foreign = tmp_path / "foreign-reply.jsonl"
        foreign.write_text(
            json.dumps(
                {
                    "type": "message",
                    "id": "foreign",
                    "timestamp": "2026-11-09T10:00:06Z",
                    "message": {"role": "assistant", "content": "Unrelated reply"},
                }
            )
            + "\n"
        )
        assert deliver(live_cfg, DroidAdapter().from_transcript(foreign), deadline=10).written
        foreign_id = ids.event_id("factory-droid", foreign.stem, "assistant_msg", "foreign")
        path = transcript(tmp_path, recalled["retrieval_id"], placeholder)
        records = DroidAdapter().from_transcript(path)
        observation = records[0].observation
        assert observation is not None
        evidence = observation["attribution_evidence"]
        assert len(evidence) == 1
        records[0] = replace(
            records[0],
            observation={
                **observation,
                "attribution_evidence": [{**evidence[0], "reply_event_ids": [foreign_id]}],
            },
        )
        assert deliver(live_cfg, records, deadline=10).written
        assert live_client.query(
            "SELECT answer_event_id, reason_code FROM effective_answer_bindings "
            f"WHERE placeholder_event_id = {sql_literal(placeholder)}"
        )[1] == [(None, "reply_not_imported")]
    finally:
        tools.close()


def test_forward_binding_migration_preserves_history_and_guards(tmp_path, monkeypatch):
    from datetime import UTC, datetime

    from agent_memory import db
    from agent_memory.config import Config
    from agent_memory.smoke import free_port
    from tests.support import insert_rows

    cfg = Config(home=tmp_path / "legacy", port=free_port())
    migrations = db.available_migrations()
    with monkeypatch.context() as patch:
        patch.setattr(db, "available_migrations", lambda: [m for m in migrations if m.version <= 4])
        assert run(cfg, "init") == 0
    first, second = new_id(), new_id()
    placeholder, later_placeholder, unused_placeholder, local_reply, remote_reply, note = [
        new_id() for _ in range(6)
    ]
    now = datetime.now(UTC)
    with db.open_direct(cfg.db_path) as con:
        insert_rows(
            con,
            {
                "sessions": [
                    {
                        "session_id": sid,
                        "agent_id": "human",
                        "capture_completeness": "partial",
                        "started_at": now,
                    }
                    for sid in (first, second)
                ],
                "events": [
                    {
                        "event_id": eid,
                        "session_id": sid,
                        "kind": kind,
                        "seq": seq,
                        "ts": now,
                        "payload": {"placeholder": True} if is_placeholder else {},
                    }
                    for seq, (eid, sid, kind, is_placeholder) in enumerate(
                        [
                            (placeholder, first, "assistant_msg", True),
                            (later_placeholder, first, "assistant_msg", True),
                            (unused_placeholder, first, "assistant_msg", True),
                            (local_reply, first, "assistant_msg", False),
                            (remote_reply, second, "assistant_msg", False),
                            (note, second, "note", False),
                        ]
                    )
                ],
                "answer_bindings": [
                    {
                        "binding_id": identity.binding_id(placeholder),
                        "placeholder_event_id": placeholder,
                        "root_placeholder_id": placeholder,
                        "answer_event_id": local_reply,
                        "evidence": {"synthetic": True},
                        "resolution_method": "synthetic",
                        "reason_code": "synthetic",
                    }
                ],
            },
        )
        before = con.execute("SELECT t::VARCHAR FROM events t ORDER BY event_id").fetchall()
        old_binding = con.execute("SELECT t::VARCHAR FROM answer_bindings t").fetchall()
    assert run(cfg, "init") == 0
    assert run(cfg, "init") == 0
    with db.open_direct(cfg.db_path) as con:
        assert con.execute("SELECT t::VARCHAR FROM events t ORDER BY event_id").fetchall() == before
        assert con.execute("SELECT t::VARCHAR FROM answer_bindings t").fetchall() == old_binding

        def bind(source, target):
            insert_rows(
                con,
                {
                    "answer_bindings": [
                        {
                            "binding_id": identity.binding_id(source),
                            "placeholder_event_id": source,
                            "root_placeholder_id": source,
                            "answer_event_id": target,
                            "evidence": {"synthetic": True},
                            "resolution_method": "synthetic",
                            "reason_code": "synthetic",
                        }
                    ]
                },
            )

        bind(later_placeholder, remote_reply)
        with pytest.raises(duckdb.Error, match="invalid answer binding"):
            bind(local_reply, remote_reply)
        with pytest.raises(duckdb.Error, match="invalid answer binding"):
            bind(unused_placeholder, note)
        with pytest.raises(duckdb.Error, match="append-only"):
            con.execute("UPDATE answer_bindings SET reason_code = 'changed'")
        with pytest.raises(duckdb.Error, match="append-only"):
            con.execute("DELETE FROM answer_bindings")
        assert con.execute("SELECT count(*) FROM answer_bindings").fetchone() == (2,)
