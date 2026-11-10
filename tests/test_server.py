"""End to end over Quack: a real server on a free port, real clients."""

from __future__ import annotations

import json
import subprocess
import sys
import threading
from datetime import UTC, datetime
from pathlib import Path

import duckdb
import pytest

from agent_memory import db
from agent_memory.config import Config
from agent_memory.db import (
    ALL_TABLES,
    AgentMemoryError,
    MemoryClient,
    MemoryServer,
    Migration,
    export_snapshot,
    initialize_home,
    open_direct,
    prune_snapshots,
    sql_literal,
)
from agent_memory.smoke import free_port, run_checks


@pytest.fixture(scope="module")
def served(tmp_path_factory):
    cfg = Config(home=tmp_path_factory.mktemp("served") / "home", port=free_port())
    initialize_home(cfg)
    server = MemoryServer(cfg, export_interval_hours=0)
    server.start()
    yield cfg, server
    server.stop()


@pytest.fixture
def client(served):
    cfg, _ = served
    with MemoryClient.from_config(cfg) as c:
        yield c


@pytest.fixture
def session_id(client):
    sid = client.new_ids()[0]
    client.insert(
        {
            "sessions": [
                {
                    "session_id": sid,
                    "agent_id": "oh-my-pi",
                    "capture_completeness": "full",
                    "started_at": datetime.now(UTC),
                }
            ]
        }
    )
    return sid


def event(session_id: str, seq: int, **extra) -> dict:
    return {
        "event_id": extra.pop("event_id", None) or _uuid(),
        "session_id": session_id,
        "seq": seq,
        "kind": "note",
        "ts": datetime.now(UTC),
        **extra,
    }


def _uuid() -> str:
    import uuid

    return str(uuid.uuid4())


def test_smoke_checks_pass(client, tmp_path):
    failed = [c for c in run_checks(client, write=True, export_dir=tmp_path) if not c.ok]
    assert failed == []


def test_variant_payload_round_trips(client, session_id):
    payload = {"hook": "PostToolUse", "args": [1, "two", None], "nested": {"ok": True, "n": 1.5}}
    eid = _uuid()
    client.insert({"events": [event(session_id, 1, event_id=eid, payload=payload)]})
    stored = client.query(
        f"SELECT payload::JSON::VARCHAR FROM events WHERE event_id = {sql_literal(eid)}"
    )[1][0][0]
    assert json.loads(stored) == payload


def test_omitted_columns_get_server_defaults(client, session_id):
    a, b = _uuid(), _uuid()
    client.insert(
        {
            "events": [
                event(session_id, 1, event_id=a, truncated=True),
                event(session_id, 2, event_id=b),
            ]
        }
    )
    rows = client.query(
        "SELECT event_id::VARCHAR, truncated, redactions, recorded_at IS NOT NULL FROM events "
        f"WHERE session_id = {sql_literal(session_id)} ORDER BY seq"
    )[1]
    assert rows == [(a, True, 0, True), (b, False, 0, True)]


def test_concurrent_writers_all_land(served, session_id):
    cfg, _ = served
    writers, per_writer = 4, 10
    errors: list[BaseException] = []

    def write(n: int) -> None:
        try:
            with MemoryClient.from_config(cfg) as c:
                for i in range(per_writer):
                    c.insert({"events": [event(session_id, n * 100 + i)]})
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=write, args=(n,)) for n in range(writers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
    assert errors == []
    with MemoryClient.from_config(cfg) as c:
        count = c.query(
            f"SELECT count(*) FROM events WHERE session_id = {sql_literal(session_id)}"
        )[1][0][0]
    assert count == writers * per_writer


def test_client_refuses_vocabulary_tables(client):
    with pytest.raises(AgentMemoryError, match="not writable"):
        client.insert({"agents": [{"agent_id": "letta-code"}]})


def test_client_refuses_unknown_columns(client, session_id):
    with pytest.raises(AgentMemoryError, match="no column"):
        client.insert({"events": [event(session_id, 1, bogus="x")]})


def test_policy_rejection_is_explained(client):
    with pytest.raises(AgentMemoryError, match="authorization policy"):
        client.query("DELETE FROM events WHERE false")


def test_server_settings_are_locked(served):
    _, server = served
    with pytest.raises(duckdb.Error, match="lock"):
        server.con.execute("SET enable_external_access = true")


def test_wrong_token_is_rejected(served):
    cfg, _ = served
    with pytest.raises(AgentMemoryError, match="Cannot connect"):
        MemoryClient(cfg.uri, "not-the-token-" * 3)


def test_direct_open_while_serving_reports_the_lock(served):
    cfg, _ = served
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "agent_memory.cli",
            "--home",
            str(cfg.home),
            "sql",
            "--direct",
            "SELECT 1",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 1
    assert "locked by another process" in proc.stderr


def test_server_export_writes_every_table(served):
    cfg, server = served
    path = server.export()
    assert sorted(p.stem for p in path.iterdir()) == sorted(ALL_TABLES)
    assert (cfg.export_dir / "latest").resolve() == path.resolve()


def test_client_reports_when_server_is_down(cfg):
    with pytest.raises(AgentMemoryError, match="Is `mem serve` running"):
        MemoryClient.from_config(cfg)


def test_server_refuses_pending_migrations(cfg, monkeypatch):
    shipped = db.available_migrations()
    extra = Migration(max(m.version for m in shipped) + 1, "later", "SELECT 1;\n")
    monkeypatch.setattr(db, "available_migrations", lambda: [*shipped, extra])
    with pytest.raises(AgentMemoryError, match="Pending migrations"):
        MemoryServer(cfg).start()


# The attached-catalog path (no CONNECT) must be authorized by the server too;
# WRITABLE_TABLES in the client is a convenience, not the boundary.
ATTACHED_PATH_ATTEMPTS = [
    "INSERT INTO mem.main.agents SELECT * FROM mem.main.agents WHERE false",
    "INSERT INTO mem.main.schema_migrations VALUES (99, 'x', 'y', now())",
    "INSERT INTO mem.main._append_only_guard VALUES ('x')",
    "UPDATE mem.main.agents SET notes = 'x' WHERE agent_id = 'human'",
    "DELETE FROM mem.main.agents WHERE agent_id = 'human'",
    "DROP TABLE mem.main.context_links",
    "CREATE TABLE mem.main.evil (x INTEGER)",
    "CREATE OR REPLACE MACRO mem.main.mem_authorize(a, b) AS true",
    "CREATE OR REPLACE VIEW mem.main.active_memories AS SELECT * FROM mem.main.memories",
    "ALTER TABLE mem.main.events ADD COLUMN evil INTEGER",
]


@pytest.mark.parametrize("sql", ATTACHED_PATH_ATTEMPTS)
def test_attached_catalog_path_is_authorized_by_the_server(client, sql):
    with pytest.raises(duckdb.Error, match="Authorization failed"):
        client.con.execute(sql)


@pytest.mark.parametrize(
    "uri",
    ["quack:localhost:1", "quack:local host:1", "quack:localhost:notaport", None],
)
def test_connection_errors_never_include_the_token(served, uri):
    cfg, _ = served
    token = "SECRET-TOKEN-THAT-MUST-NOT-LEAK-0123456789"
    with pytest.raises(AgentMemoryError) as info:
        MemoryClient(uri or cfg.uri, token)
    assert token not in str(info.value)
    assert token not in str(info.value.__cause__)


def test_serve_forever_survives_export_failures(cfg):
    server = MemoryServer(cfg, export_interval_hours=0.02 / 3600)
    calls: list[str] = []
    errors: list[Exception] = []
    exports: list[Path] = []
    stop = threading.Event()

    def flaky_export() -> Path:
        calls.append("export")
        if len(calls) == 1:
            raise OSError("disk full")
        stop.set()
        return Path("/snapshot")

    server.export = flaky_export  # type: ignore[method-assign]
    server.serve_forever(stop, on_export=exports.append, on_error=errors.append, poll_seconds=0.005)
    assert [type(e).__name__ for e in errors] == ["OSError"]
    assert exports == [Path("/snapshot")]


def test_export_retention(cfg, tmp_path):
    out = tmp_path / "exports"
    out.mkdir()
    con = open_direct(cfg.db_path, read_only=True)
    try:
        paths = [export_snapshot(con, out, keep=2) for _ in range(3)]
    finally:
        con.close()
    remaining = sorted(p.name for p in out.iterdir() if p.name != "latest")
    assert remaining == sorted(p.name for p in paths[1:])
    assert (out / "latest").resolve() == paths[-1].resolve()
    assert prune_snapshots(out, 0) == sorted(paths[1:])
    assert (out / "latest").is_symlink()
