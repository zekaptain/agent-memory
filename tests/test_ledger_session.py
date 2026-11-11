"""One shape for a session and an event, whoever writes them: the capture writer, the MCP
tools, `mem graph`, and admission."""

from __future__ import annotations

import json

import pytest

from agent_memory.capture.pipeline import make_redactor
from agent_memory.capture.record import SessionInfo
from agent_memory.capture.session import LedgerSession, Observed, UnknownAgent
from agent_memory.db import sql_literal


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
