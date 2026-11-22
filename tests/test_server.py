"""End to end over Quack: a real server on a free port, real clients."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import duckdb
import pytest

from agent_memory import db
from agent_memory.config import Config
from agent_memory.db import (
    ALL_TABLES,
    MEMORY_LIMIT,
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


def test_a_server_keeps_its_two_newest_snapshots(served):
    """It kept 14, which were three quarters of what a home took on disk, and each holds
    all that the ones before it do."""
    cfg, server = served
    paths = [server.export() for _ in range(3)]
    kept = sorted(p.name for p in cfg.export_dir.iterdir() if p.name != "latest")
    assert kept == [p.name for p in paths[1:]]
    assert (cfg.export_dir / "latest").resolve() == paths[-1].resolve()


class _Between:
    """A connection under which the ledger grows right after `sessions` has been copied."""

    def __init__(self, con: duckdb.DuckDBPyConnection, write: Callable[[], object], times: int):
        self.con, self.write, self.left = con, write, times

    def __getattr__(self, name: str):
        return getattr(self.con, name)

    def execute(self, sql: str, *args):
        out = self.con.execute(sql, *args)
        if self.left and sql.startswith("COPY") and "sessions)" in sql:
            self.left -= 1
            self.write()
        return out


def written_to_during_export(
    con: duckdb.DuckDBPyConnection, cfg: Config, *, times: int = 1
) -> tuple[duckdb.DuckDBPyConnection, list[str]]:
    """The connection, and the sessions a second client adds (each with one event) while an
    export through it runs."""
    added: list[str] = []

    def write() -> None:
        with MemoryClient.from_config(cfg) as other:
            sid = other.new_ids()[0]
            session = {
                "session_id": sid,
                "agent_id": "human",
                "capture_completeness": "full",
                "started_at": datetime.now(UTC),
            }
            other.insert([("sessions", [session]), ("events", [event(sid, 1)])])
        added.append(sid)

    return cast(duckdb.DuckDBPyConnection, _Between(con, write, times)), added


def exported(path: Path) -> tuple[set[str], int]:
    """The sessions in an export, and how many of its events belong to none of them."""
    sessions, events = (sql_literal(str(path / f"{t}.parquet")) for t in ("sessions", "events"))
    con = duckdb.connect()
    try:
        ids = con.execute(f"SELECT session_id::VARCHAR FROM read_parquet({sessions})").fetchall()
        orphans = con.execute(
            f"SELECT count(*) FROM read_parquet({events}) WHERE session_id NOT IN "
            f"(SELECT session_id FROM read_parquet({sessions}))"
        ).fetchone()
    finally:
        con.close()
    assert orphans is not None
    return {row[0] for row in ids}, orphans[0]


def test_the_servers_export_holds_one_moment_of_the_ledger(served):
    """Each table was copied by its own statement. A session and its event written while the
    export ran left the event in the files and its session out of them."""
    cfg, server = served
    con, added = written_to_during_export(server.con, cfg)
    sessions, orphans = exported(export_snapshot(con, cfg.export_dir.resolve()))
    assert len(added) == 1
    assert orphans == 0
    assert added[0] not in sessions


def test_a_clients_export_is_made_again_when_the_ledger_grew_under_it(
    served, client, tmp_path, monkeypatch
):
    """A client cannot hold the tables still: each of its statements sees the server as it
    is then. It finds out that a table grew and copies them again."""
    cfg, _ = served
    con, added = written_to_during_export(client.con, cfg)
    monkeypatch.setattr(client, "con", con)
    sessions, orphans = exported(client.export(tmp_path))
    assert orphans == 0
    assert added[0] in sessions


def test_a_clients_export_publishes_nothing_while_the_ledger_keeps_growing(
    served, client, tmp_path, monkeypatch
):
    cfg, _ = served
    previous = client.export(tmp_path)
    con, added = written_to_during_export(client.con, cfg, times=db.EXPORT_ATTEMPTS)
    monkeypatch.setattr(client, "con", con)
    with pytest.raises(AgentMemoryError, match="written to during each of 3 attempts"):
        client.export(tmp_path)
    assert len(added) == db.EXPORT_ATTEMPTS
    assert (tmp_path / "latest").resolve() == previous.resolve()
    assert not list(tmp_path.glob(".tmp-*"))
    monkeypatch.undo()
    assert exported(client.export(tmp_path))[1] == 0


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


def test_a_server_holds_no_more_memory_than_it_is_told(cfg):
    """DuckDB's own limit is 80% of the machine's memory, and it keeps what it has read until
    it holds that much."""
    server = MemoryServer(cfg, export_interval_hours=0, memory_limit="256MiB")
    server.start()
    try:
        assert server.con is not None
        limit = "SELECT current_setting('memory_limit')"
        assert server.con.execute(limit).fetchall() == [("256.0 MiB",)]
        with pytest.raises(duckdb.Error, match="locked"):
            server.con.execute("SET memory_limit = '64GiB'")
        with MemoryClient.from_config(cfg) as client:
            with pytest.raises(AgentMemoryError, match="authorization policy"):
                client.query("SET memory_limit = '64GiB'")
            assert client.query(limit)[1] == [("256.0 MiB",)]
    finally:
        server.stop()


def test_a_server_has_a_memory_limit_when_given_none(served):
    _, server = served
    con = duckdb.connect()
    try:
        con.execute(f"SET memory_limit = {sql_literal(MEMORY_LIMIT)}")
        expected = con.execute("SELECT current_setting('memory_limit')").fetchall()
    finally:
        con.close()
    assert server.con.execute("SELECT current_setting('memory_limit')").fetchall() == expected
    assert expected != duckdb.connect().execute("SELECT current_setting('memory_limit')").fetchall()


def test_a_server_does_not_start_with_a_limit_it_cannot_read(cfg):
    server = MemoryServer(cfg, export_interval_hours=0, memory_limit="lots")
    with pytest.raises(AgentMemoryError, match="'lots' is not a memory limit"):
        server.start()
    assert server.con is None
    with pytest.raises(AgentMemoryError, match="Is `mem serve` running"):
        MemoryClient.from_config(cfg)
    open_direct(cfg.db_path).close()  # the file is not left locked


def test_a_server_writes_a_snapshot_of_more_events_than_it_may_hold(cfg):
    """Events went to Parquet in the order of the table, for which DuckDB holds every event
    it has read and not yet written: 15 GB and more for a million of them. A server held to
    less ran out of memory at each snapshot."""
    sid = _uuid()
    con = open_direct(cfg.db_path)
    try:
        con.execute(
            "INSERT INTO sessions (session_id, agent_id, capture_completeness, started_at) "
            "VALUES (?, 'human', 'full', now())",
            [sid],
        )
        con.execute(
            "INSERT INTO events (event_id, session_id, seq, kind, content, ts) "
            "SELECT uuidv7(), ?, i, 'note', repeat(md5(i::VARCHAR), 125), now() "
            "FROM range(30000) AS t(i)",
            [sid],
        )
    finally:
        con.close()
    server = MemoryServer(cfg, export_interval_hours=0, memory_limit="100MB")
    server.start()
    try:
        path = server.export()
    finally:
        server.stop()
    events = sql_literal(str(path / "events.parquet"))
    reader = duckdb.connect()
    try:
        copied = reader.execute(
            "SELECT count(*), count(DISTINCT event_id), sum(length(content)) "
            f"FROM read_parquet({events})"
        ).fetchall()
    finally:
        reader.close()
    assert copied == [(30_000, 30_000, 30_000 * 4_000)]


def test_a_server_removes_the_temporary_files_one_before_it_left(cfg):
    """What a query cannot hold in memory DuckDB writes to a directory beside the database,
    and removes when it closes. A server that was killed left the files, 4 GB of them from
    one killed during a snapshot, and they stayed until a later server wrote temporary
    files of its own, which it may never need to."""
    con = open_direct(cfg.db_path)
    try:
        directory = Path(con.execute("SELECT current_setting('temp_directory')").fetchall()[0][0])
    finally:
        con.close()
    directory.mkdir(exist_ok=True)
    left = directory / f"duckdb_temp_{os.getpid() + 1}_0_storage_DEFAULT-0.tmp"
    in_use = directory / f"duckdb_temp_{os.getpid()}_0_owner"
    not_duckdbs = directory / "notes.txt"
    for path in (left, in_use, not_duckdbs):
        path.write_bytes(b"x" * 4096)
    server = MemoryServer(cfg, export_interval_hours=0)
    server.start()
    try:
        assert not left.exists()
        # DuckDB names a temporary file by its process, and this process's may be in use.
        assert in_use.exists()
        assert not_duckdbs.exists()
        assert server.removed_temporary_bytes == 4096
    finally:
        server.stop()


def test_serve_forever_survives_export_failures(cfg):
    server = MemoryServer(cfg, export_interval_hours=0.02 / 3600)
    calls: list[str] = []
    errors: list[tuple[str, str]] = []
    exports: list[Path] = []
    stop = threading.Event()

    def flaky_export() -> Path:
        calls.append("export")
        if len(calls) == 1:
            raise OSError("disk full")
        stop.set()
        return Path("/snapshot")

    server.export = flaky_export  # type: ignore[method-assign]
    server.serve_forever(
        stop,
        on_export=exports.append,
        on_error=lambda step, exc: errors.append((step, str(exc))),
        poll_seconds=0.005,
    )
    assert errors == [("export", "disk full")]
    assert exports == [Path("/snapshot")]


def test_serve_forever_names_a_failed_maintenance_pass(served, monkeypatch):
    _, server = served
    errors: list[tuple[str, str]] = []
    stop = threading.Event()

    def busy(connection, **_) -> int:
        stop.set()
        raise duckdb.Error("ledger busy")

    monkeypatch.setattr("agent_memory.lifecycle.maintain_corrections", busy)
    server.serve_forever(
        stop, on_error=lambda step, exc: errors.append((step, str(exc))), poll_seconds=0.005
    )
    assert errors == [("correction maintenance", "ledger busy")]


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
