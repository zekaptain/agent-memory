"""End-to-end checks of the server, its policy, and the client."""

from __future__ import annotations

import socket
import tempfile
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb

from .config import Config
from .db import AgentMemoryError, MemoryClient, MemoryServer, initialize_home, sql_literal

HOSTILE_CONTENT = "it's; DROP TABLE events; -- 'quoted' \"double\" $$dollar$$ é 🦆\nsecond line"

# Every probe is a no-op if it were ever allowed to run, so a broken policy
# shows up as a failed check rather than as damage to a live database.
#
# The DuckDB client splits a multi-statement batch and sends each statement
# separately, so the server authorizes them one at a time. The batch probe
# checks that a denied statement cannot ride along behind an allowed one.
DENIED_PROBES = (
    ("UPDATE", "UPDATE events SET content = content WHERE false"),
    ("DELETE", "DELETE FROM events WHERE false"),
    ("DDL", "DROP TABLE IF EXISTS mem_smoke_does_not_exist"),
    ("DDL after an allowed statement", "SELECT 1; DROP TABLE IF EXISTS mem_smoke_does_not_exist"),
    ("comments", "SELECT 1 -- comment"),
    ("vocabulary inserts", "INSERT INTO agents SELECT * FROM agents WHERE false"),
    ("server control functions", "SELECT * FROM quack_active_connections() WHERE false"),
)
FILE_PROBE = "SELECT count(*) FROM read_text('/etc/hosts')"
LATENCY_SAMPLES = 20


@dataclass(frozen=True)
class Check:
    name: str
    ok: bool
    detail: str


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def synthetic_chain(
    ids: Sequence[str], now: datetime, content: str = HOSTILE_CONTENT
) -> dict[str, list[dict[str, Any]]]:
    """A complete provenance chain: user statement -> memory -> recall -> reply.

    Needs 7 ids: session, 4 events (user_msg, memory_recall, assistant_msg,
    session_end), memory, retrieval.
    """
    session_id, e_user, e_recall, e_answer, e_end, memory_id, retrieval_id = ids
    return {
        "sessions": [
            {
                "session_id": session_id,
                "agent_id": "factory-droid",
                "capture_completeness": "full",
                "scope": "global",
                "started_at": now,
                "metadata": {"source": "synthetic"},
            }
        ],
        "events": [
            {
                "event_id": e_user,
                "session_id": session_id,
                "seq": 1,
                "kind": "user_msg",
                "role": "user",
                "content": content,
                "ts": now,
                "payload": {"hook": "UserPromptSubmit"},
            },
            {
                "event_id": e_recall,
                "session_id": session_id,
                "seq": 2,
                "kind": "memory_recall",
                "role": "memory",
                "content": "recall: duck habitat",
                "ts": now,
                "payload": None,
            },
            {
                "event_id": e_answer,
                "session_id": session_id,
                "seq": 3,
                "kind": "assistant_msg",
                "role": "assistant",
                "content": "Ducks prefer ponds.",
                "ts": now,
                "payload": None,
            },
            {
                "event_id": e_end,
                "session_id": session_id,
                "seq": 4,
                "kind": "session_end",
                "role": "harness",
                "content": None,
                "ts": now,
                "payload": None,
            },
        ],
        "memories": [
            {
                "memory_id": memory_id,
                "scope": "global",
                "type": "fact",
                "content": "Ducks prefer ponds",
                "content_hash": "synthetic",
                "trust_level": "user",
                "created_by_session": session_id,
            }
        ],
        "memory_sources": [{"memory_id": memory_id, "event_id": e_user, "relation": "user_stated"}],
        "retrievals": [
            {
                "retrieval_id": retrieval_id,
                "session_id": session_id,
                "request_event_id": e_recall,
                "query_text": "duck habitat",
                "scopes": ["global"],
                "strategy": "bm25",
                "candidates": 1,
                "returned": 1,
                "params": {"k": 5},
            }
        ],
        "retrieval_items": [
            {
                "retrieval_id": retrieval_id,
                "rank": 1,
                "memory_id": memory_id,
                "score": 0.9,
                "injected": True,
                "status_at_recall": "active",
            }
        ],
        "context_links": [
            {"answer_event_id": e_answer, "link_type": "retrieval", "retrieval_id": retrieval_id},
            {"answer_event_id": e_answer, "link_type": "cited_memory", "memory_id": memory_id},
        ],
    }


def run_checks(client: MemoryClient, *, write: bool, export_dir: Path | None) -> list[Check]:
    checks: list[Check] = []

    def check(name: str, fn: Callable[[], str | None]) -> None:
        try:
            checks.append(Check(name, True, fn() or ""))
        except Exception as exc:
            checks.append(Check(name, False, f"{type(exc).__name__}: {exc}"))

    check("connect and read", lambda: f"{_scalar(client, 'SELECT count(*) FROM agents')} agents")
    for label, sql in DENIED_PROBES:
        check(f"policy rejects {label}", lambda sql=sql: _expect_rejected(client, sql, "policy"))
    check(
        "server blocks file access",
        lambda: _expect_rejected(client, FILE_PROBE, "file system operations are disabled"),
    )
    if write:
        _write_checks(client, check)
    if export_dir is not None:
        check("client export to Parquet", lambda: _export_check(client, export_dir))
    return checks


def _write_checks(
    client: MemoryClient, check: Callable[[str, Callable[[], str | None]], None]
) -> None:
    ids = client.new_ids(7)
    session_id, e_user, _, e_answer = ids[:4]
    now = datetime.now(UTC)

    def write_chain() -> str:
        client.insert(synthetic_chain(ids, now))
        return "8 tables, one transaction"

    def why_answer() -> str:
        rows = client.query(
            "SELECT via, cited, memory_content, source_kind FROM why_answer "
            f"WHERE answer_event_id = {sql_literal(e_answer)} ORDER BY via"
        )[1]
        expected = [
            ("cited_memory", True, "Ducks prefer ponds", "user_msg"),
            ("retrieval", True, "Ducks prefer ponds", "user_msg"),
        ]
        if rows != expected:
            raise AssertionError(f"unexpected rows: {rows}")
        return "answer -> retrieval -> memory -> user_msg"

    def round_trip() -> str:
        stored = _scalar(
            client, f"SELECT content FROM events WHERE event_id = {sql_literal(e_user)}"
        )
        if stored != HOSTILE_CONTENT:
            raise AssertionError(f"stored {stored!r}")
        return "quotes, comment markers, unicode, newlines intact"

    def atomic() -> str:
        bad_session, bad_event = client.new_ids(2)
        try:
            client.insert(
                {
                    "sessions": [
                        {
                            "session_id": bad_session,
                            "agent_id": "amp",
                            "capture_completeness": "partial",
                            "started_at": now,
                        }
                    ],
                    "events": [
                        {
                            "event_id": bad_event,
                            "session_id": bad_session,
                            "seq": 1,
                            "kind": "not_a_kind",
                            "ts": now,
                        }
                    ],
                }
            )
        except (AgentMemoryError, duckdb.Error):
            pass
        else:
            raise AssertionError("insert with an unknown event kind was accepted")
        count = _scalar(
            client, f"SELECT count(*) FROM sessions WHERE session_id = {sql_literal(bad_session)}"
        )
        if count != 0:
            raise AssertionError("session row survived a failed transaction")
        return "foreign-key failure rolled back every table"

    def latency() -> str:
        started = time.perf_counter()
        for seq in range(LATENCY_SAMPLES):
            client.insert(
                {
                    "events": [
                        {
                            "event_id": client.new_ids()[0],
                            "session_id": session_id,
                            "seq": 100 + seq,
                            "kind": "note",
                            "content": f"latency {seq}",
                            "ts": datetime.now(UTC),
                        }
                    ]
                }
            )
        avg_ms = (time.perf_counter() - started) / LATENCY_SAMPLES * 1000
        return f"{avg_ms:.1f} ms per single-row insert (avg of {LATENCY_SAMPLES})"

    check("write a provenance chain", write_chain)
    check("why_answer returns the chain", why_answer)
    check("content round-trips exactly", round_trip)
    check("multi-table insert is atomic", atomic)
    check("insert latency", latency)


def _export_check(client: MemoryClient, export_dir: Path) -> str:
    path = client.export(export_dir)
    files = sorted(p.name for p in path.iterdir())
    row = client.con.execute(
        "SELECT count(*) FROM read_parquet(?)", [str(path / "events.parquet")]
    ).fetchone()
    exported = row[0] if row else None
    live = _scalar(client, "SELECT count(*) FROM events")
    if exported != live:
        raise AssertionError(f"exported {exported} events, server has {live}")
    return f"{len(files)} tables -> {path}"


def _expect_rejected(client: MemoryClient, sql: str, needle: str) -> str:
    try:
        client.query(sql)
    except (AgentMemoryError, duckdb.Error) as exc:
        if needle.lower() not in str(exc).lower():
            raise AssertionError(f"rejected for an unexpected reason: {exc}") from exc
        return "rejected"
    raise AssertionError("statement was allowed")


def _scalar(client: MemoryClient, sql: str) -> Any:
    return client.query(sql)[1][0][0]


def run_ephemeral() -> list[Check]:
    """Build a throwaway home, serve it on a free port, and run every check."""
    with tempfile.TemporaryDirectory(prefix="agent-memory-smoke-") as tmp:
        cfg = Config(home=Path(tmp), port=free_port())
        initialize_home(cfg)
        server = MemoryServer(cfg, export_interval_hours=0)
        server.start()
        try:
            with MemoryClient.from_config(cfg) as client:
                checks = run_checks(client, write=True, export_dir=cfg.home / "client-export")
            try:
                path = server.export()
                checks.append(Check("server export to Parquet", True, path.name))
            except Exception as exc:
                checks.append(
                    Check("server export to Parquet", False, f"{type(exc).__name__}: {exc}")
                )
        finally:
            server.stop()
    return checks


def run_live(cfg: Config) -> list[Check]:
    """Read-only checks against the configured server. Writes nothing."""
    with MemoryClient.from_config(cfg) as client:
        return run_checks(client, write=False, export_dir=None)
