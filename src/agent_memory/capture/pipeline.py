"""Deliver records to the ledger, or to the spool when the server is unreachable; replay later."""

from __future__ import annotations

import contextlib
import hashlib
import os
import threading
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

import duckdb

from ..config import Config, ConfigError
from ..db import AgentMemoryError, MemoryClient
from .record import Record, RecordError, now
from .redact import Redactor, redactor_for
from .spool import Spool, SpoolFile
from .writer import Writer, WriteResult

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
    if not spool_only:
        outcome: dict[str, object] = {}

        def attempt() -> None:
            try:
                with MemoryClient.from_config(cfg) as client:
                    writer = Writer(client, cfg.blob_dir, redactor, keep_raw=keep_raw)
                    outcome["written"] = writer.write(records)
            except Exception as exc:  # anything here means "spool it", never a crash
                outcome["error"] = f"{type(exc).__name__}: {exc}"

        worker = threading.Thread(target=attempt, daemon=True)
        worker.start()
        worker.join(deadline if deadline is not None else deadline_from_env())
        if worker.is_alive():
            outcome.setdefault("error", "timed out waiting for the server")
        written = outcome.get("written")
        if isinstance(written, WriteResult):
            return Delivery(written=written)
        error = str(outcome.get("error") or "server unavailable")
    else:
        error = None
    paths = spool.append(_pre_redact(records, redactor))
    return Delivery(spooled=paths, error=error)


def _pre_redact(records: Sequence[Record], redactor: Redactor) -> list[Record]:
    out = []
    for rec in records:
        content = rec.content
        if content is not None:
            content, _ = redactor.text(content)
        payload = rec.payload
        if payload is not None:
            payload, _ = redactor.value(payload)
        out.append(replace(rec, content=content, payload=payload))
    return out


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
    pending = spool.pending()
    if not pending:
        return result
    redactor = make_redactor(cfg)
    with MemoryClient.from_config(cfg) as client:
        writer = Writer(client, cfg.blob_dir, redactor, keep_raw=keep_raw)
        for file in pending:
            _replay_file(spool, writer, file, result)
    return result


def _replay_file(spool: Spool, writer: Writer, file: SpoolFile, result: ReplayResult) -> None:
    records: list[Record] = []
    bad: list[str] = []
    errors: list[Record] = []
    for number, line in spool.lines(file.path):
        try:
            records.append(Record.from_json(line))
        except RecordError as exc:
            bad.append(line)
            errors.append(_capture_error(file, number, line, str(exc)))
    try:
        written = writer.write(records + errors)
    except (AgentMemoryError, duckdb.Error) as exc:
        # Insert as a whole failed: try record by record so one bad row cannot hold the rest.
        written, bad_records = _write_individually(writer, records, file, errors)
        bad.extend(r.to_json() for r in bad_records)
        if bad_records:
            result.errors.append(f"{file.path.name}: {type(exc).__name__}: {exc}")
    result.files += 1
    result.inserted += written.inserted
    result.skipped += written.skipped
    result.failed_lines += len(bad)
    failed = spool.write_failed(file, bad)
    if failed:
        result.failed_paths.append(failed)
    spool.mark_done(file)


def _write_individually(
    writer: Writer, records: list[Record], file: SpoolFile, errors: list[Record]
) -> tuple[WriteResult, list[Record]]:
    total = WriteResult()
    bad: list[Record] = []
    for index, rec in enumerate(records):
        try:
            part = writer.write([rec])
        except (AgentMemoryError, duckdb.Error) as exc:
            bad.append(rec)
            errors.append(_capture_error(file, index + 1, rec.to_json(), str(exc)))
            continue
        _accumulate(total, part)
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
