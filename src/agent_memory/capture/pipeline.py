"""Deliver records to the ledger, or to the spool when the server is unreachable; replay later."""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import threading
from collections.abc import Sequence
from dataclasses import dataclass, field
from graphlib import CycleError, TopologicalSorter
from pathlib import Path

import duckdb

from ..config import Config, ConfigError
from ..db import AgentMemoryError, MemoryClient
from ..lifecycle import is_validation_error, validate_interval
from . import ids
from .record import Record, RecordError, now
from .redact import Redactor, redactor_for
from .spool import Spool, SpoolFile
from .writer import Writer, WriterError, WriteResult

DEFAULT_DEADLINE = 2.0


def deadline_from_env() -> float:
    raw = os.environ.get("AGENT_MEMORY_DEADLINE")
    try:
        return float(raw) if raw else DEFAULT_DEADLINE
    except ValueError:
        return DEFAULT_DEADLINE


@dataclass
class Delivery:
    written: WriteResult | None = None
    spooled: list[Path] = field(default_factory=list)
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.written is not None or bool(self.spooled)


def make_redactor(cfg: Config) -> Redactor:
    try:
        token = cfg.read_token()
    except ConfigError:
        token = None
    return redactor_for(token)


def deliver(
    cfg: Config,
    records: Sequence[Record],
    *,
    spool_only: bool = False,
    deadline: float | None = None,
    keep_raw: bool = False,
) -> Delivery:
    """Write records now if the server answers within the deadline, else append to the spool.

    Spooled records are redacted again at replay, so the spool holds redacted text too.
    """
    if not records:
        return Delivery(written=WriteResult())
    spool = Spool(cfg.spool_dir)
    redactor = make_redactor(cfg)
    try:
        for rec in records:
            if rec.memory:
                validate_interval(rec.memory)
    except ValueError as exc:
        return Delivery(error=redactor.text(str(exc))[0])
    if not spool_only:
        outcome: dict[str, object] = {}

        def attempt() -> None:
            try:
                with MemoryClient.from_config(cfg) as client:
                    writer = Writer(client, cfg.blob_dir, redactor, keep_raw=keep_raw)
                    outcome["written"] = writer.write(records)
            except Exception as exc:
                outcome["terminal"] = is_validation_error(exc) or isinstance(exc, WriterError)
                outcome["error"] = f"{type(exc).__name__}: {exc}"

        worker = threading.Thread(target=attempt, daemon=True)
        worker.start()
        worker.join(deadline if deadline is not None else deadline_from_env())
        if worker.is_alive():
            outcome.setdefault("error", "timed out waiting for the server")
        written = outcome.get("written")
        if isinstance(written, WriteResult):
            return Delivery(written=written)
        error = redactor.text(str(outcome.get("error") or "server unavailable"))[0]
        if outcome.get("terminal"):
            return Delivery(error=error)
    else:
        error = None
    paths = spool.append(redactor.record(rec, keep_raw=keep_raw) for rec in records)
    return Delivery(spooled=paths, error=error)


@dataclass
class ReplayResult:
    files: int = 0
    inserted: int = 0
    skipped: int = 0
    failed_lines: int = 0
    failed_paths: list[Path] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def replay_spool(cfg: Config, *, keep_raw: bool = False) -> ReplayResult:
    """Replay every pending spool file in order. A bad line becomes a capture_error event and
    is kept under spool/failed/; the rest of the file still lands."""
    spool = Spool(cfg.spool_dir)
    result = ReplayResult()
    with spool.replay_lock() as acquired:
        if not acquired:
            return result
        pending = spool.pending()
        if not pending:
            return result
        redactor = make_redactor(cfg)
        with MemoryClient.from_config(cfg) as client:
            writer = Writer(client, cfg.blob_dir, redactor, keep_raw=keep_raw)
            _replay_files(spool, writer, [spool.claim(file) for file in pending], result)
    return result


def _replay_files(
    spool: Spool, writer: Writer, files: list[SpoolFile], result: ReplayResult
) -> None:
    """Resolve record dependencies across the claimed snapshot, not file mtimes."""
    records: list[Record] = []
    errors: list[Record] = []
    bad_by_file: dict[Path, list[str]] = {file.path: [] for file in files}
    record_files: dict[int, tuple[SpoolFile, int]] = {}
    for file in files:
        sanitized: list[str] = []
        for number, line in spool.lines(file.path):
            try:
                record = writer.redactor.record(Record.from_json(line), keep_raw=writer.keep_raw)
                records.append(record)
                record_files[id(record)] = (file, number)
                sanitized.append(record.to_json())
            except RecordError as exc:
                try:
                    rejected = json.loads(line)
                except json.JSONDecodeError:
                    rejected = {
                        "unparseable_line_sha256": hashlib.sha256(line.encode()).hexdigest()
                    }
                line = json.dumps(writer.redactor.value(rejected)[0], ensure_ascii=False)
                sanitized.append(line)
                bad_by_file[file.path].append(line)
                errors.append(_capture_error(file, number, line, str(exc)))
        spool.rewrite(file, sanitized)
    try:
        written = writer.write(records + errors)
    except (AgentMemoryError, duckdb.Error) as exc:
        if not (is_validation_error(exc) or isinstance(exc, WriterError)):
            raise
        # Insert as a whole failed: try record by record so one bad row cannot hold the rest.
        written, bad_records = _write_individually(
            writer, records, files[0], errors, record_files=record_files
        )
        for rec in bad_records:
            file, _ = record_files[id(rec)]
            bad_by_file[file.path].append(rec.to_json())
        if bad_records:
            result.errors.append(f"claimed replay: {type(exc).__name__}: {exc}")
    result.files += len(files)
    result.inserted += written.inserted
    result.skipped += written.skipped
    for file in files:
        bad = bad_by_file[file.path]
        result.failed_lines += len(bad)
        failed = spool.write_failed(file, bad)
        if failed:
            result.failed_paths.append(failed)
        spool.mark_done(file)


def _write_individually(
    writer: Writer,
    records: list[Record],
    file: SpoolFile,
    errors: list[Record],
    *,
    record_files: dict[int, tuple[SpoolFile, int]] | None = None,
) -> tuple[WriteResult, list[Record]]:
    total = WriteResult()
    bad: list[Record] = []
    by_key = {
        (r.agent_id, r.harness_session_id, r.kind, r.source_key): i for i, r in enumerate(records)
    }
    memories = {
        ids.memory_id(r.agent_id, r.harness_session_id, r.source_key): i
        for i, r in enumerate(records)
        if r.memory
    }
    events = {
        ids.event_id(r.agent_id, r.harness_session_id, r.kind, r.source_key): i
        for i, r in enumerate(records)
    }
    dependencies: dict[int, set[int]] = {}
    for index, rec in enumerate(records):
        parent = None
        if rec.parent_key and (kind := rec.effective_parent_kind):
            parent = by_key.get((rec.agent_id, rec.harness_session_id, kind, rec.parent_key))
        dependencies[index] = {parent} if parent is not None else set()
        references = []
        if rec.action:
            references.extend([rec.action.get("memory_id"), rec.action.get("related_memory_id")])
        if rec.memory:
            references.append(rec.memory.get("supersedes"))
            for source in rec.memory.get("source_event_ids") or []:
                if (source_index := events.get(source)) is not None and source_index != index:
                    dependencies[index].add(source_index)
        for memory_id in references:
            if isinstance(memory_id, str) and (memory := memories.get(memory_id)) is not None:
                dependencies[index].add(memory)
    graph = TopologicalSorter(dependencies)
    with contextlib.suppress(CycleError):
        graph.prepare()
    failed: set[int] = set()
    visited: set[int] = set()

    def reject(index: int, reason: str) -> None:
        failed.add(index)
        bad.append(records[index])
        source, number = (record_files or {}).get(id(records[index]), (file, index + 1))
        errors.append(_capture_error(source, number, records[index].to_json(), reason))

    while ready := graph.get_ready():
        for index in ready:
            visited.add(index)
            if dependencies[index] & failed:
                reject(index, "a referenced record could not be replayed")
                continue
            try:
                _accumulate(total, writer.write([records[index]]))
            except (AgentMemoryError, duckdb.Error) as exc:
                if not (is_validation_error(exc) or isinstance(exc, WriterError)):
                    raise
                reject(index, str(exc))
        graph.done(*ready)
    for index in sorted(set(dependencies) - visited):
        reject(index, "cyclic record references")
    if errors:
        with contextlib.suppress(AgentMemoryError, duckdb.Error):
            _accumulate(total, writer.write(errors))
    return total, bad


def _accumulate(total: WriteResult, part: WriteResult) -> None:
    total.sessions += part.sessions
    total.events += part.events
    total.memories += part.memories
    total.actions += part.actions
    total.skipped += part.skipped
    total.unresolved_parents.extend(part.unresolved_parents)


def _capture_error(file: SpoolFile, number: int, line: str, error: str) -> Record:
    return Record(
        agent_id=file.agent_id,
        harness_session_id=file.harness_session_id,
        kind="capture_error",
        ts=now(),
        source_key=hashlib.sha256(line.encode()).hexdigest(),
        content=error[:4096],
        payload={"file": file.path.name, "line": number, "raw": line[:4096]},
    )
