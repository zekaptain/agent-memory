"""Deliver records to the ledger, or to the spool when the server is unreachable; replay later."""

from __future__ import annotations

import hashlib
import json
import os
import threading
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..config import Config, ConfigError
from ..db import MemoryClient
from .record import Record, RecordError, now
from .redact import Redactor, redactor_for
from .spool import Spool, SpoolFile
from .writer import Rejection, Writer, WriteResult

DEFAULT_DEADLINE = 2.0


def deadline_from_env() -> float:
    raw = os.environ.get("AGENT_MEMORY_DEADLINE")
    try:
        return float(raw) if raw else DEFAULT_DEADLINE
    except ValueError:
        return DEFAULT_DEADLINE


@dataclass
class Delivery:
    """Where the records handed to `deliver` went. Each one is in the ledger, in the spool,
    or, when the ledger rejected it, in a file under spool/failed/."""

    written: WriteResult | None = None  # None when the ledger was not reached
    spooled: list[Path] = field(default_factory=list)
    failed_paths: list[Path] = field(default_factory=list)
    error: str | None = None  # why records went to the spool


def make_redactor(cfg: Config) -> Redactor:
    try:
        token = cfg.read_token()
    except ConfigError:
        token = None
    return redactor_for(token)


@dataclass
class _Attempt:
    """What the writing thread got done before the deadline."""

    written: WriteResult | None = None
    unreported: list[Record] = field(default_factory=list)
    error: str | None = None


def deliver(
    cfg: Config,
    records: Sequence[Record],
    *,
    spool_only: bool = False,
    deadline: float | None = None,
) -> Delivery:
    """Write records now if the server answers within the deadline, else append to the spool.

    Nothing handed over is dropped. A record the ledger rejects is kept under spool/failed/
    and noted in the ledger as a capture error, while the rest of the batch is written.
    Spooled records are redacted again at replay, so the spool holds redacted text too.
    Raises only when the records could not be kept anywhere.
    """
    if not records:
        return Delivery(written=WriteResult())
    spool = Spool(cfg.spool_dir)
    redactor = make_redactor(cfg)
    error = None
    if not spool_only:
        attempt = _Attempt()

        def write() -> None:
            try:
                with MemoryClient.from_config(cfg) as client:
                    writer = Writer(client, cfg.blob_dir, redactor)
                    written = writer.write(records)
                    attempt.unreported = [
                        _capture_error(r.record, r.record.to_json(), r.reason)
                        for r in written.rejected
                    ]
                    attempt.written = written
                    if attempt.unreported:
                        writer.write(attempt.unreported)
                        attempt.unreported = []
            except Exception as exc:
                attempt.error = f"{type(exc).__name__}: {exc}"

        worker = threading.Thread(target=write, daemon=True)
        worker.start()
        worker.join(deadline if deadline is not None else deadline_from_env())
        if worker.is_alive():
            attempt.error = attempt.error or "timed out waiting for the server"
        error = redactor.text(attempt.error)[0] if attempt.error else None
        if attempt.written is not None:
            return Delivery(
                written=attempt.written,
                # A capture error the ledger did not take in time is replayed like any record.
                spooled=spool.append(redactor.record(rec) for rec in attempt.unreported),
                failed_paths=_keep_rejected(spool, attempt.written.rejected),
                error=error,
            )
        error = error or "server unavailable"
    paths = spool.append(redactor.record(rec) for rec in records)
    return Delivery(spooled=paths, error=error)


def _keep_rejected(spool: Spool, rejected: Sequence[Rejection]) -> list[Path]:
    lines: dict[tuple[str, str], list[str]] = {}
    for rejection in rejected:
        rec = rejection.record
        lines.setdefault((rec.agent_id, rec.harness_session_id), []).append(rec.to_json())
    return [path for key, kept in lines.items() if (path := spool.write_failed(*key, kept))]


@dataclass
class ReplayResult:
    files: int = 0
    inserted: int = 0
    skipped: int = 0
    failed_lines: int = 0
    failed_paths: list[Path] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def replay_spool(cfg: Config) -> ReplayResult:
    """Replay every pending spool file in order. A line that cannot be read, or a record the
    ledger rejects, becomes a capture_error event and is kept under spool/failed/; the rest
    still lands. When the ledger is unavailable this raises and the files stay pending."""
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
            writer = Writer(client, cfg.blob_dir, redactor)
            _replay_files(spool, writer, [spool.claim(file) for file in pending], result)
    return result


def _replay_files(
    spool: Spool, writer: Writer, files: list[SpoolFile], result: ReplayResult
) -> None:
    """Write the claimed snapshot as one batch, so that a record can refer to one in another
    file whatever the files' mtimes."""
    records: list[Record] = []
    origin: list[tuple[SpoolFile, int]] = []
    errors: list[Record] = []
    bad_by_file: dict[Path, list[str]] = {file.path: [] for file in files}

    def fail(
        file: SpoolFile, number: int, line: str, reason: str, session: Record | SpoolFile
    ) -> None:
        bad_by_file[file.path].append(line)
        errors.append(_capture_error(session, line, reason, file=file.path.name, line=number))
        result.errors.append(f"{file.path.name} line {number}: {reason}")

    for file in files:
        sanitized: list[str] = []
        for number, line in spool.lines(file.path):
            try:
                record = writer.redactor.record(Record.from_json(line))
            except RecordError as exc:
                try:
                    unread = json.loads(line)
                except json.JSONDecodeError:
                    unread = {"unparseable_line_sha256": hashlib.sha256(line.encode()).hexdigest()}
                line = json.dumps(writer.redactor.value(unread)[0], ensure_ascii=False)
                sanitized.append(line)
                fail(file, number, line, writer.redactor.text(str(exc))[0], file)
                continue
            records.append(record)
            origin.append((file, number))
            sanitized.append(record.to_json())
        spool.rewrite(file, sanitized)
    written = writer.write(records)
    for rejection in written.rejected:
        file, number = origin[rejection.index]
        fail(file, number, rejection.record.to_json(), rejection.reason, rejection.record)
    result.errors.extend(written.warnings)
    if errors:
        reported = writer.write(errors)
        written.add(reported)
        result.errors.extend(f"capture error not recorded: {r.reason}" for r in reported.rejected)
    result.files += len(files)
    result.inserted += written.inserted
    result.skipped += written.skipped
    for file in files:
        bad = bad_by_file[file.path]
        result.failed_lines += len(bad)
        kept = spool.write_failed(file.agent_id, file.harness_session_id, bad)
        if kept and kept not in result.failed_paths:
            result.failed_paths.append(kept)
        spool.mark_done(file)


def _capture_error(session: Record | SpoolFile, raw: str, reason: str, **source: Any) -> Record:
    """The note the ledger keeps about text it did not take, in the session the text belongs
    to. It is keyed by that text, so failing on the same text again adds nothing."""
    return Record(
        agent_id=session.agent_id,
        harness_session_id=session.harness_session_id,
        kind="capture_error",
        ts=now(),
        source_key=hashlib.sha256(raw.encode()).hexdigest(),
        content=reason[:4096],
        payload={**source, "raw": raw[:4096]},
    )
