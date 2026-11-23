"""Direct database access, migrations, the Quack server, and the Quack client."""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import secrets
import shutil
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib import resources
from pathlib import Path
from typing import Any

import duckdb

from .config import Config

# Every table, in foreign-key dependency order.
ALL_TABLES = (
    "schema_migrations",
    "agents",
    "event_kinds",
    "memory_types",
    "memory_action_types",
    "sessions",
    "events",
    "memories",
    "memory_sources",
    "memory_actions",
    "memory_deduplications",
    "memory_replacements",
    "memory_correction_work",
    "memory_correction_closures",
    "retrievals",
    "retrieval_items",
    "context_links",
    "capture_observations",
    "answer_bindings",
)

# Tables Quack clients may insert into. Must match the list in mem_authorize.
WRITABLE_TABLES = frozenset(
    {
        "sessions",
        "events",
        "memories",
        "memory_sources",
        "memory_actions",
        "memory_deduplications",
        "memory_replacements",
        "memory_correction_closures",
        "retrievals",
        "retrieval_items",
        "context_links",
        "capture_observations",
        "answer_bindings",
    }
)

CLIENT_ALIAS = "mem"
_IDENTIFIER = re.compile(r"[a-z_][a-z0-9_]*")
_MIGRATION_FILE = re.compile(r"(\d{4})_([a-z0-9_]+)\.sql")


class AgentMemoryError(Exception):
    """An operation on the memory database failed in an expected way."""


def sql_literal(value: str) -> str:
    """Quote a trusted value (path, URI, token) as a SQL string literal."""
    return "'" + value.replace("'", "''") + "'"


def new_token() -> str:
    return secrets.token_urlsafe(32)


# Direct access (the server, migrations, admin)


def install_extensions(con: duckdb.DuckDBPyConnection) -> None:
    """Download fts and quack into the local extension cache. Needs network once."""
    con.execute("INSTALL fts")
    con.execute("INSTALL quack")


def _connect(path: Path, *, read_only: bool = False) -> duckdb.DuckDBPyConnection:
    try:
        return duckdb.connect(str(path), read_only=read_only)
    except duckdb.IOException as exc:
        if "lock" in str(exc).lower():
            raise AgentMemoryError(
                f"{path} is locked by another process, probably `mem serve`. "
                "Stop the server for direct access, or go through Quack instead."
            ) from exc
        raise


def open_direct(path: Path, *, read_only: bool = False) -> duckdb.DuckDBPyConnection:
    if read_only and not path.exists():
        raise AgentMemoryError(f"No database at {path}. Run `mem init` first.")
    con = _connect(path, read_only=read_only)
    con.execute("LOAD fts")
    return con


@dataclass(frozen=True)
class InitResult:
    created_token: bool
    applied: list[Migration]


def initialize_home(cfg: Config) -> InitResult:
    """Create the data directory, token, and database. Safe to re-run."""
    for directory in (cfg.home, cfg.export_dir, cfg.spool_dir, cfg.blob_dir):
        directory.mkdir(parents=True, exist_ok=True)
    os.chmod(cfg.home, 0o700)
    created_token = False
    if not cfg.token_path.exists():
        fd = os.open(cfg.token_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as fh:
            fh.write(new_token() + "\n")
        created_token = True
    con = _connect(cfg.db_path)
    try:
        try:
            install_extensions(con)
        except duckdb.Error as exc:
            raise AgentMemoryError(
                f"Could not install the fts and quack extensions (first run needs network): {exc}"
            ) from exc
        con.execute("LOAD fts")
        applied = migrate(con)
    finally:
        con.close()
    return InitResult(created_token=created_token, applied=applied)


@dataclass(frozen=True)
class Migration:
    version: int
    name: str
    sql: str

    @property
    def checksum(self) -> str:
        return hashlib.sha256(self.sql.encode()).hexdigest()

    @property
    def label(self) -> str:
        return f"{self.version:04d}_{self.name}"


_BOOTSTRAP = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    version     INTEGER PRIMARY KEY,
    name        VARCHAR NOT NULL,
    checksum    VARCHAR NOT NULL,
    applied_at  TIMESTAMPTZ NOT NULL DEFAULT now()
)
"""


def available_migrations() -> list[Migration]:
    found = []
    for entry in resources.files("agent_memory.migrations").iterdir():
        match = _MIGRATION_FILE.fullmatch(entry.name)
        if match:
            found.append(Migration(int(match[1]), match[2], entry.read_text()))
    return sorted(found, key=lambda m: m.version)


def applied_migrations(con: duckdb.DuckDBPyConnection) -> dict[int, str]:
    exists = con.execute(
        "SELECT count(*) FROM duckdb_tables() "
        "WHERE database_name = current_database() AND schema_name = 'main' "
        "AND table_name = 'schema_migrations'"
    ).fetchone()
    if not exists or not exists[0]:
        return {}
    return dict(con.execute("SELECT version, checksum FROM schema_migrations").fetchall())


def pending_migrations(con: duckdb.DuckDBPyConnection) -> list[Migration]:
    available = available_migrations()
    applied = applied_migrations(con)
    unknown = sorted(set(applied) - {m.version for m in available})
    if unknown:
        raise AgentMemoryError(
            f"The database has migrations this version does not know about: {unknown}. "
            "Upgrade agent-memory."
        )
    pending = []
    for migration in available:
        if migration.version not in applied:
            pending.append(migration)
        elif applied[migration.version] != migration.checksum:
            raise AgentMemoryError(
                f"Migration {migration.label} changed after it was applied. "
                "Applied migrations are immutable; add a new migration instead."
            )
    return pending


def migrate(con: duckdb.DuckDBPyConnection) -> list[Migration]:
    con.execute(_BOOTSTRAP)
    applied = []
    for migration in pending_migrations(con):
        con.execute("BEGIN TRANSACTION")
        try:
            if migration.label == "0002_memory_lifecycle":
                _migrate_lifecycle(con, migration.sql)
            else:
                con.execute(migration.sql)
            con.execute(
                "INSERT INTO schema_migrations (version, name, checksum) VALUES (?, ?, ?)",
                [migration.version, migration.name, migration.checksum],
            )
            con.execute("COMMIT")
        except Exception:
            _rollback_quietly(con)
            raise
        applied.append(migration)
    return applied


def _migrate_lifecycle(con: duckdb.DuckDBPyConnection, sql: str) -> None:
    """Preflight old rows before installing any new semantics, in the migration transaction."""
    from .audit import legacy_preflight
    from .lifecycle import stable_id

    preflight = legacy_preflight(con)
    if not preflight.ok:
        blockers = "; ".join(f"{f.code}:{f.entity_id}" for f in preflight.blockers)
        raise AgentMemoryError(
            "Lifecycle migration blocked by ambiguous legacy history. "
            f"Run `mem audit check --direct` and resolve explicit review evidence: {blockers}"
        )
    marker = "CREATE TRIGGER memories_valid_interval"
    schema, guards = sql.split(marker, 1)
    con.execute(schema)
    # Only reconstruction runs before the runtime guards. There is deliberately
    # no client-accessible method string that bypasses acceptance validation.
    for acceptance in preflight.acceptances:
        proposal = stable_id(acceptance.replacement_id, "legacy-proposal")
        evidence = {
            "memory_id": acceptance.replacement_id,
            "approval_action_id": acceptance.evidence_action_id,
            "method": acceptance.method,
            "preflight": "legacy_preflight_v1",
        }
        con.execute(
            "INSERT INTO memory_actions "
            "(action_id, memory_id, action, actor, related_memory_id, created_at, metadata) "
            "VALUES (?, ?, 'propose_correction', 'migration:0002', ?, ?, ?::JSON::VARIANT)",
            [
                proposal,
                acceptance.replacement_id,
                acceptance.predecessor_id,
                acceptance.accepted_at,
                json.dumps({"legacy_evidence": evidence}),
            ],
        )
        con.execute(
            "INSERT INTO memory_replacements "
            "(predecessor_id, successor_id, proposal_id, evidence_action_id, accepted_at, method) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            [
                acceptance.predecessor_id,
                acceptance.replacement_id,
                proposal,
                acceptance.evidence_action_id or proposal,
                acceptance.accepted_at,
                "legacy_" + acceptance.method,
            ],
        )
    con.execute(marker + guards)


def _rollback_quietly(con: duckdb.DuckDBPyConnection) -> None:
    # A failed statement may already have aborted the transaction.
    with contextlib.suppress(duckdb.Error):
        con.execute("ROLLBACK")


# Export


# A snapshot's directory is named for the moment it was taken, in UTC.
_SNAPSHOT_DIR = re.compile(r"\d{8}T\d{6}\d{6}Z")
_SNAPSHOT_STAMP = "%Y%m%dT%H%M%S%fZ"

# How many times a client copies the tables before it gives up on a ledger that keeps growing.
EXPORT_ATTEMPTS = 3
# How long the server waits after an export that failed before it tries again, unless the
# interval between exports is shorter. A full disk stays full for a while, and every try
# reads every table.
EXPORT_RETRY_SECONDS = 600.0
# How many snapshots the server keeps. The tables only grow, so the newest snapshot holds all
# that an older one does. The one before it stays for a reader that opened it as the latest,
# and in case the newest cannot be read.
EXPORT_KEEP = 2


def export_snapshot(con: duckdb.DuckDBPyConnection, export_dir: Path, keep: int = 0) -> Path:
    """Write every table to Parquet under export_dir/<timestamp>/ and repoint export_dir/latest.

    For a connection to the database file. The copies share one transaction, so the files
    hold the ledger as it was at one moment, whatever clients write meanwhile. With
    keep > 0, snapshots older than the newest `keep` are removed afterwards.
    """

    def copy(staging: Path) -> None:
        con.execute("BEGIN TRANSACTION")
        try:
            _copy_tables(con, staging)
        finally:
            _rollback_quietly(con)

    return _publish_snapshot(export_dir, keep, copy)


# How `events` is written to Parquet. A table is written in the order of its rows, and for
# that DuckDB holds what it has read and not yet written, 122,880 rows at a time on every
# thread: 15 to 35 GB for a million events, and out of memory under any limit a server is
# given. So events are written as they are read, in row groups of 8 MB. Nothing asks for
# the order they were written in: each has its `seq`, no view breaks a tie between two by
# `rowid`, and one cannot be inserted in the statement that inserts its parent, whatever
# the order. The other tables keep theirs, which views do break ties by.
_EVENTS_COPY = "FORMAT parquet, PRESERVE_ORDER false, ROW_GROUP_SIZE_BYTES '8MB'"


def _copy_tables(con: duckdb.DuckDBPyConnection, staging: Path, prefix: str = "") -> dict[str, int]:
    """Copy every table into the directory. Returns how many rows each file holds."""
    copied = {}
    for table in ALL_TABLES:
        target = sql_literal(str(staging / f"{table}.parquet"))
        options = _EVENTS_COPY if table == "events" else "FORMAT parquet"
        row = con.execute(
            f"COPY (SELECT * FROM {prefix}{table}) TO {target} ({options})"
        ).fetchone()
        copied[table] = int(row[0]) if row else 0
    return copied


def _publish_snapshot(export_dir: Path, keep: int, copy: Callable[[Path], None]) -> Path:
    """Have `copy` fill a staging directory, then publish it as the newest snapshot. A copy
    that fails publishes nothing and leaves only the earlier snapshots behind."""
    stamp = datetime.now(UTC).strftime(_SNAPSHOT_STAMP)
    staging = export_dir / f".tmp-{stamp}"
    final = export_dir / stamp
    staging.mkdir(parents=True)
    link = export_dir / f".latest-{stamp}"
    try:
        copy(staging)
        staging.rename(final)
        link.symlink_to(final.name)
        os.replace(link, export_dir / "latest")
    finally:
        with contextlib.suppress(OSError):
            shutil.rmtree(staging)
        with contextlib.suppress(OSError):
            link.unlink(missing_ok=True)
    if keep > 0:
        prune_snapshots(export_dir, keep)
    return final


def _snapshots(export_dir: Path) -> list[Path]:
    """The snapshot directories, oldest first."""
    if not export_dir.is_dir():
        return []
    return sorted(p for p in export_dir.iterdir() if p.is_dir() and _SNAPSHOT_DIR.fullmatch(p.name))


def newest_snapshot_at(export_dir: Path) -> datetime | None:
    """When the newest snapshot was taken, which its name says. None when there is none."""
    for snapshot in reversed(_snapshots(export_dir)):
        with contextlib.suppress(ValueError):
            return datetime.strptime(snapshot.name, _SNAPSHOT_STAMP).replace(tzinfo=UTC)
    return None


def prune_snapshots(export_dir: Path, keep: int) -> list[Path]:
    """Delete all but the newest `keep` snapshot directories. Returns what was removed."""
    snapshots = _snapshots(export_dir)
    removed = []
    for old in snapshots[: max(0, len(snapshots) - keep)]:
        shutil.rmtree(old)
        removed.append(old)
    return removed


# Server

# The most memory DuckDB may hold of the database while it is served: a share of the
# machine's, `10%`, or a size, `2GB`. DuckDB's own limit is 80%, and what it reads it keeps
# until it has that much, so a server left alone takes more as its ledger grows. Past the
# limit it lets go of what it read, and a query that needs more writes to
# `memory.duckdb.tmp` beside the database, or fails when it cannot. The default is a share
# because what a query needs grows with the threads it runs on, one for each core, and a
# machine with more cores has more memory: on 18 threads and 10 million events, the judge's
# read of every turn failed with 1 GB and completed with 2. The process holds more than
# the limit, because DuckDB does not count all the process takes: about 1 GiB more at
# rest, and several times the limit while a client takes a result of millions of rows.
MEMORY_LIMIT = "10%"


def _remove_stale_temporary_files(con: duckdb.DuckDBPyConnection) -> int:
    """Delete the temporary files an earlier process left. Returns how many bytes they held.

    DuckDB removes its temporary files when it closes, so a server that was killed leaves
    them. DuckDB removes those the next time it writes temporary files of its own, which a
    server may never need to. The connection holds the database file, so no other process
    is using the directory. DuckDB names a file by its process, and this process's are left
    alone.
    """
    row = con.execute("SELECT current_setting('temp_directory')").fetchone()
    directory = Path(row[0]) if row and row[0] else None
    if directory is None or not directory.is_dir():
        return 0
    own = f"duckdb_temp_{os.getpid()}_"
    removed = 0
    for path in directory.iterdir():
        if path.name.startswith("duckdb_temp_") and not path.name.startswith(own):
            with contextlib.suppress(OSError):
                size = path.stat().st_size
                path.unlink()
                removed += size
    return removed


class MemoryServer:
    """Owns the database file and serves it to local clients over Quack."""

    def __init__(
        self,
        cfg: Config,
        *,
        export_interval_hours: float = 24.0,
        export_keep: int = EXPORT_KEEP,
        memory_limit: str = MEMORY_LIMIT,
    ) -> None:
        self.cfg = cfg
        self.export_interval_hours = export_interval_hours
        self.export_keep = export_keep
        self.memory_limit = memory_limit
        self.removed_temporary_bytes = 0
        self.con: duckdb.DuckDBPyConnection | None = None

    def start(self) -> None:
        token = self.cfg.read_token()
        self.cfg.export_dir.mkdir(parents=True, exist_ok=True)
        con = open_direct(self.cfg.db_path)
        try:
            self.removed_temporary_bytes = _remove_stale_temporary_files(con)
            try:
                con.execute(f"SET memory_limit = {sql_literal(self.memory_limit)}")
            except duckdb.Error as exc:
                raise AgentMemoryError(
                    f"{self.memory_limit!r} is not a memory limit. Give a size such as 2GB, "
                    "or a share of the machine's memory such as 10%."
                ) from exc
            con.execute("LOAD quack")
            pending = pending_migrations(con)
            if pending:
                names = ", ".join(m.label for m in pending)
                raise AgentMemoryError(f"Pending migrations: {names}. Run `mem migrate` first.")
            from .lifecycle import maintain_corrections

            maintain_corrections(con)
            con.execute("SET GLOBAL quack_authorization_function = 'mem_authorize'")
            con.execute("CALL quack_serve(?, token = ?)", [self.cfg.uri, token])
            # Lock down after the listener is up. The server process itself may
            # only touch the export directory, and no one can change settings.
            export_root = sql_literal(str(self.cfg.export_dir.resolve()) + "/")
            con.execute(f"SET allowed_directories = [{export_root}]")
            con.execute("SET autoinstall_known_extensions = false")
            con.execute("SET autoload_known_extensions = false")
            con.execute("SET enable_external_access = false")
            con.execute("SET lock_configuration = true")
        except Exception:
            con.close()
            raise
        self.con = con

    def export(self) -> Path:
        if self.con is None:
            raise AgentMemoryError("Server is not running.")
        return export_snapshot(self.con, self.cfg.export_dir.resolve(), keep=self.export_keep)

    def serve_forever(
        self,
        stop: threading.Event,
        on_export: Callable[[Path], None] | None = None,
        on_error: Callable[[str, Exception], None] | None = None,
        poll_seconds: float = 1.0,
        clock: Callable[[], float] = time.time,
    ) -> None:
        """Block until `stop` is set, applying due corrections and exporting on schedule.

        An export is due when the newest snapshot is as old as the interval: at the first
        turn on a home that has none or an old one, so a server that is restarted every
        day still exports. `clock` is the wall clock. A monotonic clock stops while the
        machine sleeps, and counted by it a laptop's server exported days apart.

        A failed step (a full disk, a busy ledger) is reported through on_error with the
        step's name and tried again: maintenance at its next turn, an export after
        EXPORT_RETRY_SECONDS or the interval, whichever is shorter. It must not take the
        server down.
        """
        interval = self.export_interval_hours * 3600
        export_at = self._first_export_at(interval, clock()) if interval > 0 else None
        from .lifecycle import maintain_corrections

        while not stop.wait(poll_seconds):
            if self.con is not None:
                try:
                    maintain_corrections(self.con)
                except Exception as exc:
                    if on_error:
                        on_error("correction maintenance", exc)
            if export_at is not None and clock() >= export_at:
                try:
                    path = self.export()
                except Exception as exc:
                    export_at = clock() + min(interval, EXPORT_RETRY_SECONDS)
                    if on_error:
                        on_error("export", exc)
                    continue
                export_at = clock() + interval
                if on_export:
                    on_export(path)

    def _first_export_at(self, interval: float, now: float) -> float:
        """When a server that starts now owes its first export. Never later than one
        interval from now, so a snapshot named while the clock was wrong puts nothing off."""
        newest = newest_snapshot_at(self.cfg.export_dir)
        if newest is None:
            return now
        return min(newest.timestamp() + interval, now + interval)

    def stop(self) -> None:
        if self.con is None:
            return
        try:
            self.con.execute("CALL quack_stop(?)", [self.cfg.uri])
            self.con.execute("CHECKPOINT")
        finally:
            self.con.close()
            self.con = None


# Client


class MemoryClient:
    """Talks to `mem serve` over Quack.

    Quack in DuckDB 2.0.0 does not forward bound parameters, so values are
    never placed in SQL text. Rows are staged in a local temp table with
    ordinary parameters and then streamed with INSERT INTO mem.t SELECT ...,
    which Quack sends as data rather than SQL.
    """

    def __init__(self, uri: str, token: str) -> None:
        self.con = duckdb.connect(":memory:")
        self.con.execute("LOAD quack")
        try:
            self.con.execute(
                f"ATTACH {sql_literal(uri)} AS {CLIENT_ALIAS} (TOKEN {sql_literal(token)})"
            )
        except duckdb.Error as exc:
            self.con.close()
            raise AgentMemoryError(
                f"Cannot connect to {uri} ({exc}). Is `mem serve` running?"
            ) from exc
        self._column_types: dict[str, dict[str, str]] = {}
        self._stage_counter = 0

    @classmethod
    def from_config(cls, cfg: Config) -> MemoryClient:
        return cls(cfg.uri, cfg.read_token())

    def close(self) -> None:
        self.con.close()

    def __enter__(self) -> MemoryClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def new_ids(self, n: int = 1) -> list[str]:
        rows = self.con.execute("SELECT uuidv7()::VARCHAR FROM range(?)", [n]).fetchall()
        return [row[0] for row in rows]

    def column_types(self, table: str) -> dict[str, str]:
        if table not in self._column_types:
            rows = self.con.execute(
                "SELECT column_name, data_type FROM duckdb_columns() "
                "WHERE database_name = ? AND schema_name = 'main' AND table_name = ? "
                "ORDER BY column_index",
                [CLIENT_ALIAS, table],
            ).fetchall()
            if not rows:
                raise AgentMemoryError(f"Unknown table {table!r} on the server.")
            self._column_types[table] = dict(rows)
        return self._column_types[table]

    def insert(
        self,
        rows_by_table: Mapping[str, Sequence[Mapping[str, Any]]]
        | Sequence[tuple[str, Sequence[Mapping[str, Any]]]],
    ) -> None:
        """Insert rows into ledger tables in one transaction, in the given order.

        A sequence of (table, rows) pairs may name a table more than once; each pair
        becomes its own statement. Rows that reference other rows of the same table
        (events.parent_event_id) must arrive in a later pair, because DuckDB checks the
        foreign key against the index before the statement's own rows are indexed.
        """
        pairs = rows_by_table.items() if isinstance(rows_by_table, Mapping) else rows_by_table
        staged: list[tuple[str, str, list[str]]] = []
        try:
            for table, rows in pairs:
                # Rows that omit different columns are staged separately so the
                # server-side defaults still apply to whatever each row omits.
                groups: dict[tuple[str, ...], list[Mapping[str, Any]]] = {}
                for row in rows:
                    groups.setdefault(tuple(row), []).append(row)
                for group in groups.values():
                    staged.append(self._stage(table, group))
            if not staged:
                return
            self.con.execute("BEGIN TRANSACTION")
            try:
                for table, stage, cols in staged:
                    collist = ", ".join(f'"{c}"' for c in cols)
                    self.con.execute(
                        f"INSERT INTO {CLIENT_ALIAS}.main.{table} ({collist}) "
                        f"SELECT {collist} FROM {stage}"
                    )
                self.con.execute("COMMIT")
            except Exception:
                _rollback_quietly(self.con)
                raise
        finally:
            for _, stage, _ in staged:
                self.con.execute(f"DROP TABLE IF EXISTS {stage}")

    def _stage(self, table: str, rows: Sequence[Mapping[str, Any]]) -> tuple[str, str, list[str]]:
        if table not in WRITABLE_TABLES:
            raise AgentMemoryError(f"{table!r} is not writable by clients.")
        types = self.column_types(table)
        cols = list(rows[0])
        for col in cols:
            if not _IDENTIFIER.fullmatch(col) or col not in types:
                raise AgentMemoryError(f"{table} has no column {col!r}.")
        self._stage_counter += 1
        stage = f"stage_{table}_{self._stage_counter}"
        col_defs = ", ".join(f'"{c}" {types[c]}' for c in cols)
        self.con.execute(f"CREATE TEMP TABLE {stage} ({col_defs})")
        try:
            placeholders = ", ".join(
                "?::JSON::VARIANT" if types[c] == "VARIANT" else "?" for c in cols
            )
            values = [[_encode(row[c], types[c]) for c in cols] for row in rows]
            self.con.executemany(f"INSERT INTO {stage} VALUES ({placeholders})", values)
        except BaseException:
            with contextlib.suppress(duckdb.Error):
                self.con.execute(f"DROP TABLE IF EXISTS {stage}")
            raise
        return table, stage, cols

    def query(self, sql: str) -> tuple[list[str], list[tuple]]:
        """Run literal SQL on the server. No bound parameters; see the class docstring."""
        self.con.execute(f"CONNECT {CLIENT_ALIAS}")
        try:
            cursor = self.con.execute(sql)
            columns = [d[0] for d in cursor.description] if cursor.description else []
            rows = cursor.fetchall() if columns else []
        except duckdb.Error as exc:
            if "Authorization failed" in str(exc):
                raise AgentMemoryError(
                    "The server's authorization policy rejected this statement. Clients may run "
                    "SELECT-style queries and INSERT into ledger tables; comments, multiple "
                    "statements, DDL, UPDATE/DELETE, and file access are not allowed."
                ) from exc
            raise
        finally:
            self.con.execute("DISCONNECT")
        return columns, rows

    def export(self, export_dir: Path, keep: int = 0) -> Path:
        """Write every table to Parquet, as export_snapshot does for the database file.

        Each statement a client sends sees the server as it is then, so no transaction here
        holds the tables still. But the tables only grow: files with as many rows as one
        later statement counts are the tables as they were at that statement. The copies are
        made again while a table has grown, and nothing is published if they never agree.
        """
        prefix = f"{CLIENT_ALIAS}.main."

        def copy(staging: Path) -> None:
            for _ in range(EXPORT_ATTEMPTS):
                if _copy_tables(self.con, staging, prefix) == self._row_counts():
                    return
            raise AgentMemoryError(
                f"The ledger was written to during each of {EXPORT_ATTEMPTS} attempts to export "
                "it, so no snapshot was published. Export again when no agent is writing."
            )

        return _publish_snapshot(export_dir, keep, copy)

    def _row_counts(self) -> dict[str, int]:
        """How many rows every table holds, counted by one statement and so at one moment."""
        counts = ", ".join(f"(SELECT count(*) FROM {table})" for table in ALL_TABLES)
        _, rows = self.query(f"SELECT {counts}")
        return dict(zip(ALL_TABLES, (int(n) for n in rows[0]), strict=True))


def _encode(value: Any, column_type: str) -> Any:
    if value is None:
        return None
    if column_type == "VARIANT":
        return json.dumps(value)
    return value
