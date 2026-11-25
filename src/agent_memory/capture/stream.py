"""Capture records taken from a harness extension, a line at a time, while its input stays
open."""

from __future__ import annotations

import json
import os
import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, replace

from ..config import Config
from .observations import covering
from .pipeline import (
    capture_error,
    deadline_from_env,
    deliver,
    kept_of_line,
    make_redactor,
    replay_spool,
)
from .record import Record, RecordError
from .spool import Spool, SpoolFile

# How long a pass waits, from the first line it takes, for the lines that follow, so that
# what an extension writes in one burst is one write to the ledger.
LINGER = 0.5
# The most lines one pass takes.
PASS_LINES = 500
# Where a line that is not a record is kept when it names no session and none was read
# before it.
NO_SESSION = ("unknown", "records")

type Line = tuple[int, str]


@dataclass(frozen=True)
class _End:
    """The input ended at this moment on the monotonic clock."""

    at: float


@dataclass
class Taken:
    """What became of the lines read from one input."""

    records: int = 0
    unread: int = 0  # lines that are not capture records
    rejected: int = 0  # records the ledger refused
    lost: int = 0  # lines that could not be kept anywhere


class RecordStream:
    """Delivers the capture records read from one input, in the order they were read.

    The lines are taken in passes: a pass waits for a line, takes those that follow it
    within `LINGER`, and delivers them as `deliver` does for a hook, with a hook's deadline.
    Once a pass has left records in the spool, the passes after it add theirs to the spool
    and replay it, so that a record never reaches the ledger before one read ahead of it.

    When the input ends, what is still held is delivered in what is left of one deadline,
    counted from the end of the input, and spooled after that.
    """

    def __init__(
        self,
        cfg: Config,
        *,
        spool_only: bool = False,
        report: Callable[[str], None] = lambda message: None,
    ) -> None:
        self.cfg = cfg
        self.spool_only = spool_only
        self.report = report
        self.taken = Taken()
        self._deadline = deadline_from_env()
        self._spool = Spool(cfg.spool_dir)
        self._redactor = make_redactor(cfg)
        # A signal handler may add to this queue while the loop below is taking from it,
        # which only a SimpleQueue allows.
        self._lines: queue.SimpleQueue[Line | _End | None] = queue.SimpleQueue()
        self._stopped = False
        self._last: tuple[str, str] | None = None
        # The sessions, as the spool names them, whose records this stream left there.
        self._waiting: set[tuple[str, str]] = set()
        self._replay: threading.Thread | None = None

    def run(self, fd: int) -> Taken:
        """Take lines from the file descriptor until it ends or `stop` is called."""
        threading.Thread(target=self._read, args=(fd,), daemon=True).start()
        while True:
            lines, ended = self._gather()
            if self._stopped:
                self._pass(lines + self._queued(), 0.0)
                return self.taken
            if ended is not None:
                self._pass(lines, ended + self._deadline - time.monotonic())
                return self.taken
            self._pass(lines, self._deadline)

    def stop(self) -> None:
        """Spool what is held and end `run`, without waiting for the ledger. Safe to call
        from a signal handler."""
        self._stopped = True
        self._lines.put(None)

    def _read(self, fd: int) -> None:
        # Read with os.read and not through a buffered file object: Python aborts at exit
        # when a daemon thread holds such an object's lock, as one blocked in its read does.
        parts: list[bytes] = []
        number = 0
        try:
            while chunk := os.read(fd, 1 << 16):
                *ends, rest = chunk.split(b"\n")
                for end in ends:
                    number += 1
                    self._take(number, b"".join([*parts, end]))
                    parts = []
                parts.append(rest)
        except OSError:
            pass  # an input that can no longer be read has ended
        finally:
            # A harness that was killed leaves a last line without its end.
            self._take(number + 1, b"".join(parts))
            self._lines.put(_End(time.monotonic()))

    def _take(self, number: int, raw: bytes) -> None:
        if line := raw.decode("utf-8", errors="replace").strip():
            self._lines.put((number, line))

    def _gather(self) -> tuple[list[Line], float | None]:
        """The lines of the next pass, and when the input ended if it has."""
        lines: list[Line] = []
        until: float | None = None
        while len(lines) < PASS_LINES:
            try:
                item = self._lines.get(
                    timeout=None if until is None else max(0.0, until - time.monotonic())
                )
            except queue.Empty:
                break
            if item is None:
                break
            if isinstance(item, _End):
                return lines, item.at
            lines.append(item)
            until = until or time.monotonic() + LINGER
        return lines, None

    def _queued(self) -> list[Line]:
        lines: list[Line] = []
        while True:
            try:
                item = self._lines.get_nowait()
            except queue.Empty:
                return lines
            if isinstance(item, tuple):
                lines.append(item)

    def _pass(self, lines: list[Line], wait: float) -> None:
        """Deliver these lines, giving the ledger `wait` seconds; none when `wait` is not
        positive."""
        behind = bool(self._waiting)
        try:
            records = _observed(self._records(lines))
            delivery = deliver(
                self.cfg,
                records,
                spool_only=self.spool_only or behind or wait <= 0,
                deadline=wait,
            )
        except Exception as exc:
            self.taken.lost += len(lines)
            self.report(f"{len(lines)} line(s) could not be kept: {type(exc).__name__}: {exc}")
            return
        if delivery.written is not None and (rejected := delivery.written.rejected):
            self.taken.rejected += len(rejected)
            kept = ", ".join(str(path) for path in delivery.failed_paths)
            reasons = "; ".join(dict.fromkeys(r.reason for r in rejected))
            self.report(f"{len(rejected)} record(s) rejected, kept in {kept}: {reasons}")
        if self.spool_only:
            return
        if delivery.spooled and not behind and wait > 0:
            self.report(f"ledger not reached ({delivery.error}); records wait in the spool")
        self._waiting.update((path.parent.name, path.stem) for path in delivery.spooled)
        if behind and wait > 0:
            self._catch_up(wait)

    def _catch_up(self, wait: float) -> None:
        """Replay the spool, and see whether what this stream left there is in the ledger."""
        if self._replay is None or not self._replay.is_alive():
            self._replay = threading.Thread(target=self._replay_spool, daemon=True)
            self._replay.start()
        self._replay.join(wait)
        # A file is pending until a replay has written it, a claimed file included.
        self._waiting &= {(f.agent_id, f.harness_session_id) for f in self._spool.pending()}
        if not self._waiting:
            self.report("the spool was replayed; records go to the ledger again")

    def _replay_spool(self) -> None:
        try:
            result = replay_spool(self.cfg)
        except Exception:
            return  # the ledger is still out of reach, and the next pass asks again
        for error in result.errors:
            self.report(error)

    def _records(self, lines: list[Line]) -> list[Record]:
        records: list[Record] = []
        for number, line in lines:
            try:
                record = Record.from_json(line)
            except RecordError as exc:
                self.taken.unread += 1
                if error := self._unread(number, line, str(exc)):
                    records.append(error)
                continue
            self.taken.records += 1
            self._last = (record.agent_id, record.harness_session_id)
            records.append(record)
        return records

    def _unread(self, number: int, line: str, reason: str) -> Record | None:
        """Keep a line that is not a capture record under spool/failed/, and return the
        capture error for the session it belongs to: the one it names, or else the one of
        the record read before it. Returns None when there is neither."""
        kept = kept_of_line(self._redactor, line)
        reason = self._redactor.text(reason)[0]
        session = _named(line) or self._last
        agent_id, harness_session_id = session or NO_SESSION
        what = f"line {number} is not a capture record ({reason})"
        try:
            path = self._spool.write_failed(agent_id, harness_session_id, [kept])
            self.report(f"{what}; kept in {path}")
        except OSError as exc:
            self.taken.lost += 1
            self.report(f"{what} and could not be kept: {exc}")
        if session is None:
            return None
        place = SpoolFile(self._spool.failed_dir, agent_id, harness_session_id)
        return capture_error(place, kept, reason, input_line=number)


def _named(line: str) -> tuple[str, str] | None:
    """The agent and the session a line names, when it is an object that names both."""
    try:
        data = json.loads(line)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    agent_id, harness_session_id = data.get("agent_id"), data.get("harness_session_id")
    if isinstance(agent_id, str) and agent_id and isinstance(harness_session_id, str):
        return (agent_id, harness_session_id) if harness_session_id else None
    return None


def _observed(records: list[Record]) -> list[Record]:
    """Give each session one capture observation of what this pass read of it, on the first
    of its records, unless one of them brings an observation of its own. Without it every
    record would be observed by itself, a row for each event."""
    by_session: dict[tuple[str, str], list[int]] = {}
    for index, rec in enumerate(records):
        by_session.setdefault((rec.agent_id, rec.harness_session_id), []).append(index)
    out = list(records)
    for indexes in by_session.values():
        read = [records[index] for index in indexes]
        if all(rec.observation is None for rec in read):
            out[indexes[0]] = replace(read[0], observation=covering(read))
    return out
