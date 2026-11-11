"""One shape for a session and an event, whoever writes them: the capture writer, the MCP
tools, `mem graph`, and admission."""

from __future__ import annotations

import json

import pytest

from agent_memory.capture import blobs
from agent_memory.capture.pipeline import deliver, make_redactor, replay_spool
from agent_memory.capture.record import Record, SessionInfo, now
from agent_memory.capture.session import LedgerSession, Observed, UnknownAgent
from agent_memory.db import MemoryClient, MemoryServer, sql_literal
from agent_memory.mcp_server import MemoryTools


def q(client, sql):
    return client.query(sql)[1]


def session_shape(client, session_id):
    [row] = q(
        client,
        "SELECT agent_id, scope, project, cwd, capture_completeness, metadata::JSON::VARCHAR "
        f"FROM sessions WHERE session_id = {sql_literal(session_id)}",
    )
    return (*row[:5], json.loads(row[5]) if row[5] else None)


def observations(client, session_id):
    return [
        (mode, adapter, json.loads(exclusions))
        for mode, adapter, exclusions in q(
            client,
            "SELECT source_mode, adapter_version, exclusions::JSON::VARCHAR "
            f"FROM capture_observations WHERE session_id = {sql_literal(session_id)} "
            "ORDER BY recorded_at",
        )
    ]


def write_events(client, session_id):
    return q(
        client,
        "SELECT role, length(content), content_hash, content_bytes, truncated, blob_ref, "
        f"token_estimate FROM events WHERE session_id = {sql_literal(session_id)} "
        "AND kind = 'memory_write' ORDER BY ts",
    )


def test_a_session_another_writer_opens_first_does_not_fail_the_write(
    live_cfg, live_client, monkeypatch
):
    """A hook of the same harness session can commit the session row between the tool's
    read and its insert. The write used to fail with a duplicate key and was not queued."""
    tools = MemoryTools(live_cfg, "factory-droid", harness_session_id="raced-open", project="demo")
    original = MemoryClient.query
    raced: list[str] = []

    def racing(client, sql):
        found = original(client, sql)
        if "FROM sessions" in sql and not raced:
            raced.append(sql)
            hook = Record(
                agent_id="factory-droid",
                harness_session_id="raced-open",
                kind="user_msg",
                ts=now(),
                source_key="prompt-1",
                content="hello",
            )
            assert deliver(live_cfg, [hook], deadline=10).written is not None
        return found

    monkeypatch.setattr(MemoryClient, "query", racing)
    try:
        kept = tools.remember("Written while a hook opened the session", "observation")
    finally:
        tools.close()
    assert raced
    assert kept["queued"] is False
    assert q(
        live_client,
        f"SELECT count(*) FROM sessions WHERE session_id = {sql_literal(tools.session_id)}",
    ) == [(1,)]
    assert q(
        live_client,
        f"SELECT count(*) FROM memories WHERE memory_id = {sql_literal(kept['memory_id'])}",
    ) == [(1,)]


def test_a_session_first_written_from_the_spool_matches_one_opened_online(down_cfg):
    """A queued write that reached the ledger before its session did used to leave a session
    with no project, no working directory, no metadata, and a generic observation."""
    queued_tools = MemoryTools(down_cfg, "amp", harness_session_id="spool-first", project="demo")
    online_tools = MemoryTools(down_cfg, "amp", harness_session_id="online-first", project="demo")
    try:
        assert queued_tools.remember("Queued before the session existed", "observation")["queued"]
        server = MemoryServer(down_cfg, export_interval_hours=0)
        server.start()
        try:
            assert replay_spool(down_cfg).failed_lines == 0
            written = online_tools.remember("Written while the server was up", "observation")
            assert written["queued"] is False
            with MemoryClient.from_config(down_cfg) as client:
                online = session_shape(client, online_tools.session_id)
                assert online[2] == "demo"
                assert online[5] == {"via": "mem-mcp", "own_session": False, "session_link": "env"}
                assert session_shape(client, queued_tools.session_id) == online
                expected = [("mcp", "mem_mcp_v1", ["non_mcp_activity_unobserved"])]
                assert observations(client, online_tools.session_id) == expected
                assert observations(client, queued_tools.session_id) == expected
        finally:
            server.stop()
    finally:
        queued_tools.close()
        online_tools.close()


def test_an_event_is_stored_the_same_online_and_replayed(down_cfg):
    """Online, a tool's event kept its whole content inline with no byte count, blob, or
    token estimate; the same write replayed from the spool got all three."""
    big = ("A lesson that is far too long to keep inline. " * 800).strip()
    assert len(big.encode()) > blobs.BLOB_THRESHOLD
    queued_tools = MemoryTools(down_cfg, "amp", harness_session_id="big-queued", project="demo")
    online_tools = MemoryTools(down_cfg, "amp", harness_session_id="big-online", project="demo")
    try:
        assert queued_tools.remember(big, "observation", scope="replayed")["queued"]
        server = MemoryServer(down_cfg, export_interval_hours=0)
        server.start()
        try:
            assert replay_spool(down_cfg).failed_lines == 0
            assert online_tools.remember(big, "observation", scope="online")["queued"] is False
            with MemoryClient.from_config(down_cfg) as client:
                [replayed] = write_events(client, queued_tools.session_id)
                [online] = write_events(client, online_tools.session_id)
                assert q(client, "SELECT DISTINCT content FROM memories") == [(big,)]
        finally:
            server.stop()
    finally:
        queued_tools.close()
        online_tools.close()
    digest = blobs.content_hash(big)
    assert replayed == (
        "memory",
        blobs.INLINE_BYTES,
        digest,
        len(big),
        True,
        f"{digest[:2]}/{digest}.txt",
        len(big) // 4,
    )
    assert online == replayed
    assert (down_cfg.blob_dir / online[5]).read_text() == big


def test_a_recall_event_counts_its_content_like_any_other_event(live_cfg, live_client):
    tools = MemoryTools(live_cfg, "jetbrains-junie", project="demo")
    try:
        got = tools.recall("which package manager")
    finally:
        tools.close()
    assert q(
        live_client,
        "SELECT role, content, content_bytes, truncated, token_estimate FROM events "
        f"WHERE event_id = {sql_literal(got['request_event_id'])}",
    ) == [("memory", "which package manager", 21, False, 5)]


# The session module itself


def test_opening_a_session_twice_writes_one_row_and_one_observation(live_cfg, live_client):
    session = LedgerSession(
        "factory-droid",
        "opened-twice",
        SessionInfo(project="demo", metadata={"origin": "https://bot:hunter2secret@example.test"}),
        Observed("a test", "record", "test_v1"),
    )
    redactor = make_redactor(live_cfg)
    session.open(live_client, redactor)
    session.open(live_client, redactor)
    assert session_shape(live_client, session.session_id) == (
        "factory-droid",
        "global",
        "demo",
        None,
        "partial",
        {"via": "a test", "origin": "https://[REDACTED:url_credentials]@example.test"},
    )
    assert observations(live_client, session.session_id) == [
        ("record", "test_v1", ["no_complete_capture_evidence"])
    ]


def test_an_unregistered_agent_opens_no_session(live_cfg, live_client):
    session = LedgerSession(
        "nobody-registered", "x", SessionInfo(), Observed("a test", "record", "test_v1")
    )
    with pytest.raises(UnknownAgent, match="unknown agent 'nobody-registered'"):
        session.open(live_client, make_redactor(live_cfg))
    assert q(live_client, "SELECT count(*) FROM sessions WHERE agent_id = 'nobody-registered'") == [
        (0,)
    ]
