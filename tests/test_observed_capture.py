"""Coverage must come from captured evidence, not a harness's advertised tier."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import duckdb
import pytest

from agent_memory.adapters.droid import DroidAdapter
from agent_memory.capture import ids
from agent_memory.capture.pipeline import deliver
from agent_memory.capture.record import Record
from agent_memory.db import AgentMemoryError, sql_literal
from agent_memory.mcp_server import MemoryTools


def summary(client, sid):
    columns, rows = client.query(
        f"SELECT * FROM session_summary WHERE session_id = {sql_literal(sid)}"
    )
    return dict(zip(columns, rows[0], strict=True))


def transcript(path, *, model="fixture-model"):
    rows = [
        {"type": "session_start", "id": "start", "timestamp": "2026-01-01T00:00:00Z"},
        {
            "type": "message",
            "id": "reply",
            "timestamp": "2026-01-01T00:01:00Z",
            "message": {"role": "assistant", "modelId": model, "content": "Synthetic reply."},
        },
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")


def test_mcp_harness_session_does_not_prove_capture(live_cfg, live_client):
    tools = MemoryTools(live_cfg, "factory-droid", harness_session_id="coverage-mcp")
    try:
        tools.status()
        got = summary(live_client, tools.session_id)
        assert got["capture_completeness"] == "partial"
        assert got["recorded_capture_completeness"] == "partial"
    finally:
        tools.close()


def test_one_hook_is_partial(live_cfg, live_client):
    records = DroidAdapter().from_hook(
        {"session_id": "coverage-hook", "hook_event_name": "SessionStart", "message_id": "h1"}
    )
    result = deliver(live_cfg, records, deadline=10)
    assert result.written
    got = summary(live_client, ids.session_id("factory-droid", "coverage-hook"))
    assert got["capture_completeness"] == "partial"
    assert got["coverage_source_mode"] == "hook"


def test_import_enriches_immutable_session_and_has_boundary(live_cfg, live_client, tmp_path):
    path = tmp_path / "coverage-enrich.jsonl"
    tools = MemoryTools(live_cfg, "factory-droid", harness_session_id=path.stem)
    try:
        tools.status()
        _, before = live_client.query(
            f"SELECT * FROM sessions WHERE session_id = {sql_literal(tools.session_id)}"
        )
        transcript(path)
        records = DroidAdapter().from_transcript(path)
        assert deliver(live_cfg, records, deadline=10).written
        assert deliver(live_cfg, records, deadline=10).written
        _, after = live_client.query(
            f"SELECT * FROM sessions WHERE session_id = {sql_literal(tools.session_id)}"
        )
        assert before == after
        got = summary(live_client, tools.session_id)
        assert got["capture_completeness"] == "transcript_import"
        assert got["recorded_capture_completeness"] == "partial"
        assert got["model"] == "fixture-model"
        assert got["coverage_range_end"] == 2
        assert got["coverage_cursor_end"] == "reply"
        assert "thinking_omitted" in str(got["coverage_exclusions"])
        _, observations = live_client.query(
            "SELECT count(*) FROM capture_observations "
            f"WHERE session_id = {sql_literal(tools.session_id)} "
            "AND source_mode = 'transcript_import'"
        )
        assert observations == [(1,)]
        later = Record("factory-droid", path.stem, "note", datetime.now(UTC), "later-activity")
        assert deliver(live_cfg, [later], deadline=10).written
        assert summary(live_client, tools.session_id)["coverage_gap"]
    finally:
        tools.close()


def test_observation_redaction_and_append_only(live_cfg, live_client):
    rec = Record(
        "factory-droid",
        "coverage-redaction",
        "note",
        datetime.now(UTC),
        "safe-id",
        observation={
            "source_mode": "record",
            "adapter_version": "test",
            "source_fingerprint": "sha256:synthetic",
            "cursor_end": "password=synthetic-password",
            "exclusions": ["token=synthetic-secret"],
            "session_metadata": {"password": "synthetic-password"},
        },
    )
    assert deliver(live_cfg, [rec], deadline=10).written
    _, rows = live_client.query(
        "SELECT observation_id::VARCHAR, cursor_end, exclusions::JSON::VARCHAR, "
        "session_metadata::JSON::VARCHAR FROM capture_observations "
        f"WHERE session_id = {sql_literal(ids.session_id(rec.agent_id, rec.harness_session_id))}"
    )
    assert len(rows) == 1
    assert "synthetic-password" not in str(rows)
    assert "synthetic-secret" not in str(rows)
    # The server policy rejects writes other than INSERT.
    with pytest.raises(AgentMemoryError, match=r"authorization|Authorization"):
        live_client.query("DELETE FROM capture_observations")


def test_legacy_full_is_not_evidence(con):
    sid = str(con.execute("SELECT uuidv7()").fetchone()[0])
    con.execute(
        "INSERT INTO sessions (session_id, agent_id, capture_completeness, started_at) "
        "VALUES (?, 'factory-droid', 'full', now())",
        [sid],
    )
    assert con.execute(
        "SELECT capture_completeness, recorded_capture_completeness FROM session_summary "
        "WHERE session_id = ?",
        [sid],
    ).fetchone() == ("unverified_legacy_assertion", "full")


def test_explicit_supported_capture_does_not_cover_later_events(con):
    from tests.support import insert_rows, new_id

    sid, oid = new_id(), new_id()
    now = datetime.now(UTC)
    insert_rows(
        con,
        {
            "sessions": [
                {
                    "session_id": sid,
                    "agent_id": "factory-droid",
                    "capture_completeness": "partial",
                    "started_at": now,
                }
            ],
            "events": [
                {"event_id": new_id(), "session_id": sid, "kind": "note", "seq": 1, "ts": now}
            ],
            "capture_observations": [
                {
                    "observation_id": oid,
                    "session_id": sid,
                    "source_mode": "supported_capture",
                    "adapter_version": "synthetic",
                    "policy_version": "test",
                    "source_fingerprint": "synthetic",
                    "covered_from": now,
                    "covered_through": now,
                    "parse_errors": [],
                    "exclusions": [],
                    "complete_supported": True,
                }
            ],
        },
    )
    assert (
        con.execute(
            "SELECT capture_completeness FROM session_summary WHERE session_id = ?", [sid]
        ).fetchone()[0]
        == "full"
    )
    insert_rows(
        con,
        {
            "events": [
                {
                    "event_id": new_id(),
                    "session_id": sid,
                    "kind": "note",
                    "seq": 2,
                    "ts": now + timedelta(seconds=1),
                }
            ]
        },
    )
    assert con.execute(
        "SELECT capture_completeness, coverage_gap FROM session_summary WHERE session_id = ?",
        [sid],
    ).fetchone() == ("partial", True)
    with pytest.raises(duckdb.Error, match="append-only"):
        con.execute("DELETE FROM capture_observations WHERE observation_id = ?", [oid])


def test_malformed_import_is_idempotent_and_exposes_errors(live_cfg, live_client, tmp_path):
    path = tmp_path / "malformed-observation.jsonl"
    path.write_text("not valid json\n")
    for _ in range(2):
        assert deliver(live_cfg, DroidAdapter().from_transcript(path), deadline=10).written
    sid = ids.session_id("factory-droid", path.stem)
    got = summary(live_client, sid)
    assert got["coverage_gap"]
    assert "invalid_json_object" in str(got["coverage_parse_errors"])
    _, rows = live_client.query(
        f"SELECT count(*) FROM capture_observations WHERE session_id = {sql_literal(sid)}"
    )
    assert rows == [(1,)]


def test_since_key_must_identify_a_real_boundary(tmp_path):
    path = tmp_path / "missing-cursor.jsonl"
    transcript(path)
    with pytest.raises(ValueError, match="since_key was not found"):
        DroidAdapter().from_transcript(path, since_key="not-a-message")


def test_generated_cursor_is_redacted_without_changing_event_identity(live_cfg, live_client):
    key = "password=synthetic-cursor-secret"
    rec = Record("factory-droid", "redacted-cursor", "note", datetime.now(UTC), key)
    assert deliver(live_cfg, [rec], deadline=10).written
    sid = ids.session_id(rec.agent_id, rec.harness_session_id)
    _, rows = live_client.query(
        "SELECT cursor_start, cursor_end FROM capture_observations "
        f"WHERE session_id = {sql_literal(sid)}"
    )
    assert rows == [("password=[REDACTED:generic_assignment]",) * 2]
    eid = ids.event_id(rec.agent_id, rec.harness_session_id, rec.kind, rec.source_key)
    _, events = live_client.query(
        f"SELECT event_id::VARCHAR FROM events WHERE session_id = {sql_literal(sid)}"
    )
    assert events == [(eid,)]
