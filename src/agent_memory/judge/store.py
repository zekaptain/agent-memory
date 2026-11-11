"""The files of a judge run, and the guard against the production home.

A run is a directory of private files: `run.json`, one JSON object per line in `items.jsonl`,
`drafts.jsonl`, `labels.jsonl`, and `verdicts/<backend>.jsonl`, the projections, and
`report.md`. Rows are plain dicts that are read and written back unchanged, so a run built
under any question set stays readable.

How those files are kept whole is decided here and nowhere else:

- A file that is replaced (run.json, items, labels, a projection, the report) is written
  beside itself, flushed to disk, and moved into place. A reader or a crash sees the old
  file or the new one, never part of one.
- A file that grows (drafts, verdicts, the imports log) gets one row at a time. A killed
  command can leave its last row cut off: reading leaves that row out, and the next append
  removes it, since the command that resumes writes the row again.
- run.json is changed by commands that run side by side, so each change reads the file,
  merges, and writes it back under a lock.
"""

from __future__ import annotations

import fcntl
import json
import os
import re
import shutil
import tempfile
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..config import DEFAULT_HOME, Config
from ..db import AgentMemoryError

_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")
_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


class JudgeError(AgentMemoryError):
    """A judge command cannot proceed for an expected reason."""


def ensure_experiment_home(cfg: Config) -> None:
    """Refuse to run against the production ledger. The experiment imports and writes freely."""
    if cfg.home.expanduser().resolve() == DEFAULT_HOME.expanduser().resolve():
        raise JudgeError(
            "mem judge refuses to run against the default home. Point AGENT_MEMORY_HOME or "
            "--home at a separate experiment directory (for example ~/.agent-memory-judge)."
        )


def safe_name(value: str) -> str:
    return _UNSAFE.sub("_", value)


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def imports_log(cfg: Config) -> Path:
    """Bookkeeping of imported transcripts: path, mtime, and counts, one line per import."""
    return cfg.home / "judge" / "imports.jsonl"


# Files


def private_dir(path: Path) -> None:
    """Make `path` a directory only this user can enter, creating it if it is missing."""
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    path.chmod(0o700)


def atomic_write(path: Path, body: str) -> None:
    """Replace `path` with `body` in one step. The new file is on disk before it takes the
    old one's place, so a failure at any point leaves the old file as it was."""
    private_dir(path.parent)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    tmp = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            os.fchmod(stream.fileno(), 0o600)
            stream.write(body)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def read_json(path: Path) -> dict[str, Any]:
    """The JSON object in `path`. An error names the file and never quotes it."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, UnicodeError):
        raise JudgeError(f"{path}: invalid JSON") from None
    if not isinstance(data, dict):
        raise JudgeError(f"{path}: expected a JSON object")
    return data


def _lock_path(path: Path) -> Path:
    return path.with_name(path.name + ".lock")


@contextmanager
def lock(path: Path) -> Iterator[None]:
    """Hold the lock file `path` until the block ends, waiting for another holder first."""
    private_dir(path.parent)
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        os.close(fd)


def _mend_tail(path: Path) -> None:
    """Make the next append start on a line of its own. A killed run can leave the last row
    without its line ending: a whole row gets one, and a fragment is cut off, since
    `read_jsonl` never returned it and the run that resumes writes that row again."""
    try:
        fh = path.open("r+b")
    except FileNotFoundError:
        return
    with fh:
        end = fh.seek(0, os.SEEK_END)
        if not end:
            return
        fh.seek(end - 1)
        if fh.read(1) == b"\n":
            return
        start = end
        while start:
            step = min(start, 1 << 16)
            fh.seek(start - step)
            found = fh.read(step).rfind(b"\n")
            if found >= 0:
                start = start - step + found + 1
                break
            start -= step
        fh.seek(start)
        tail = fh.read()
        if not tail.strip():
            return
        try:
            json.loads(tail.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            fh.truncate(start)
        else:
            fh.write(b"\n")


class RunStore:
    """One run: its items, drafts, verdicts, labels, projections, and report, kept under
    `<home>/judge/<run_id>/`. A live day is a run of its own under the run's `live/`."""

    def __init__(self, cfg: Config, run_id: str, *, root: Path | None = None) -> None:
        if not _RUN_ID.fullmatch(run_id):
            raise JudgeError(f"bad run id {run_id!r}: letters, digits, dot, dash, underscore")
        self.cfg = cfg
        self.run_id = run_id
        self.root = root if root is not None else cfg.home / "judge" / run_id

    def live_day(self, name: str) -> RunStore:
        """The live day `name`, which shares this run's id and frozen thresholds."""
        return RunStore(self.cfg, self.run_id, root=self.live_dir / safe_name(name))

    def live_days(self) -> list[RunStore]:
        """Every live day of this run, in name order."""
        if not self.live_dir.exists():
            return []
        return [self.live_day(p.name) for p in sorted(self.live_dir.iterdir()) if p.is_dir()]

    # Where the files are. Commands read and write them through the methods further down.

    @property
    def run_json(self) -> Path:
        return self.root / "run.json"

    @property
    def items_path(self) -> Path:
        return self.root / "items.jsonl"

    @property
    def drafts_path(self) -> Path:
        return self.root / "drafts.jsonl"

    @property
    def labels_path(self) -> Path:
        return self.root / "labels.jsonl"

    @property
    def report_path(self) -> Path:
        return self.root / "report.md"

    @property
    def verdicts_dir(self) -> Path:
        return self.root / "verdicts"

    @property
    def projections_dir(self) -> Path:
        return self.root / "projections"

    @property
    def live_dir(self) -> Path:
        return self.root / "live"

    def verdicts_path(self, backend_name: str) -> Path:
        return self.verdicts_dir / f"{safe_name(backend_name)}.jsonl"

    def _projection_path(self, item_id: str) -> Path:
        return self.projections_dir / f"{safe_name(item_id)}.txt"

    # The run

    def exists(self) -> bool:
        return self.run_json.exists()

    def create(self, params: Mapping[str, Any]) -> None:
        if self.exists():
            raise JudgeError(f"run {self.run_id!r} already exists at {self.root}")
        for directory in (self.root, self.verdicts_dir, self.projections_dir):
            private_dir(directory)
        self._write_run({"run_id": self.run_id, "created_at": utc_now_iso(), **params})

    def read_run(self) -> dict[str, Any]:
        if not self.exists():
            raise JudgeError(f"no run {self.run_id!r} at {self.root}; run `mem judge candidates`")
        return read_json(self.run_json)

    def update_run(self, **fields: Any) -> dict[str, Any]:
        """Set `fields` on the run and return it. Commands running side by side each keep
        the fields the others set."""
        with lock(_lock_path(self.run_json)):
            data = self.read_run()
            data.update(fields)
            self._write_run(data)
        return data

    def _write_run(self, data: Mapping[str, Any]) -> None:
        atomic_write(self.run_json, json.dumps(data, indent=2, sort_keys=True, default=str))

    # Rows

    @staticmethod
    def read_jsonl(path: Path) -> list[dict[str, Any]]:
        """Every row of `path`, or none when there is no file. A last line that was cut off
        is left out."""
        if not path.exists():
            return []
        rows = []
        with path.open("rb") as fh:
            for number, line in enumerate(fh, start=1):
                if line.strip():
                    try:
                        row = json.loads(line.decode("utf-8"))
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        # Only an unterminated final line can be a torn append.
                        if not line.endswith(b"\n"):
                            break
                        raise JudgeError(f"{path}: invalid JSON on line {number}") from None
                    if not isinstance(row, dict):
                        raise JudgeError(f"{path}: expected a JSON object on line {number}")
                    rows.append(row)
        return rows

    @staticmethod
    def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]], *, append: bool = False) -> int:
        """Replace the rows of `path`, or with `append` add to them. Returns how many rows
        were written."""
        lines = [
            json.dumps(row, ensure_ascii=False, sort_keys=True, default=str) + "\n" for row in rows
        ]
        if not append:
            atomic_write(path, "".join(lines))
            return len(lines)
        private_dir(path.parent)
        _mend_tail(path)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            os.fchmod(stream.fileno(), 0o600)
            stream.write("".join(lines))
            stream.flush()
            os.fsync(stream.fileno())
        return len(lines)

    # Items

    def has_items(self) -> bool:
        return self.items_path.exists()

    def items(self) -> list[dict[str, Any]]:
        return self.read_jsonl(self.items_path)

    def write_items(self, rows: Iterable[Mapping[str, Any]]) -> int:
        return self.write_jsonl(self.items_path, rows)

    # Drafts

    def drafts(self) -> list[dict[str, Any]]:
        """Every draft row, in the order the drafts were written."""
        return self.read_jsonl(self.drafts_path)

    def drafts_by_item(self) -> dict[str, list[dict[str, Any]]]:
        """Draft rows by item, each item's drafts in a stable order by drafter name."""
        by_item: dict[str, list[dict[str, Any]]] = {}
        for row in self.drafts():
            by_item.setdefault(row["item_id"], []).append(row)
        for rows in by_item.values():
            rows.sort(key=lambda r: str(r.get("drafter") or ""))
        return by_item

    def add_draft(self, row: Mapping[str, Any]) -> None:
        self.write_jsonl(self.drafts_path, [row], append=True)

    # Verdicts, one file per backend

    def verdicts_by_file(self) -> dict[str, list[dict[str, Any]]]:
        """The rows of every verdict file in the run, by the file's name without its ending
        and in name order. Each row names its own backend."""
        if not self.verdicts_dir.exists():
            return {}
        return {
            path.stem: self.read_jsonl(path) for path in sorted(self.verdicts_dir.glob("*.jsonl"))
        }

    def verdicts(self, backend: str) -> list[dict[str, Any]]:
        """The verdicts of one backend, in the order they were written."""
        return self.read_jsonl(self.verdicts_path(backend))

    def add_verdict(self, backend: str, row: Mapping[str, Any]) -> None:
        self.write_jsonl(self.verdicts_path(backend), [row], append=True)

    def replace_verdicts(self, backend: str, rows: Iterable[Mapping[str, Any]]) -> None:
        self.write_jsonl(self.verdicts_path(backend), rows)

    # Labels

    def labels(self) -> dict[str, dict[str, Any]]:
        """The label of every labeled item by item id, in the order the items were first
        labeled."""
        labels: dict[str, dict[str, Any]] = {}
        for row in self.read_jsonl(self.labels_path):
            if not isinstance(row.get("item_id"), str):
                raise JudgeError(f"{self.labels_path}: a label names no item")
            labels[row["item_id"]] = row
        return labels

    # Report

    def write_report(self, text: str) -> None:
        atomic_write(self.report_path, text)

    # Projections: the one place state text lives on disk. Purgeable.

    def write_projection(self, item_id: str, text: str) -> Path:
        path = self._projection_path(item_id)
        atomic_write(path, text)
        return path

    def read_projection(self, item_id: str) -> str:
        path = self._projection_path(item_id)
        if not path.exists():
            raise JudgeError(f"no projection for item {item_id}; run `mem judge candidates`")
        return path.read_text(encoding="utf-8")

    def purge_projections(self) -> int:
        if not self.projections_dir.exists():
            return 0
        count = sum(1 for _ in self.projections_dir.iterdir())
        shutil.rmtree(self.projections_dir)
        return count
