"""Run directory layout for judge experiments, and the guard against the production home."""

from __future__ import annotations

import json
import os
import re
import shutil
from collections.abc import Iterable, Mapping
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


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def imports_log(cfg: Config) -> Path:
    """Bookkeeping of imported transcripts: path, mtime, and counts, one line per import."""
    return cfg.home / "judge" / "imports.jsonl"


class RunStore:
    """One experiment run: `<home>/judge/<run_id>/` with JSONL files and 0600 projections."""

    def __init__(self, cfg: Config, run_id: str, *, root: Path | None = None) -> None:
        if not _RUN_ID.fullmatch(run_id):
            raise JudgeError(f"bad run id {run_id!r}: letters, digits, dot, dash, underscore")
        self.cfg = cfg
        self.run_id = run_id
        self.root = root if root is not None else cfg.home / "judge" / run_id

    def child(self, name: str) -> RunStore:
        """A sub-run under live/, sharing the parent's run id and frozen thresholds."""
        return RunStore(self.cfg, self.run_id, root=self.live_dir / safe_name(name))

    # Paths

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

    # Run metadata

    def exists(self) -> bool:
        return self.run_json.exists()

    def create(self, params: Mapping[str, Any]) -> None:
        if self.exists():
            raise JudgeError(f"run {self.run_id!r} already exists at {self.root}")
        self.root.mkdir(parents=True, exist_ok=True)
        os.chmod(self.root, 0o700)
        self.verdicts_dir.mkdir(exist_ok=True)
        self.projections_dir.mkdir(exist_ok=True)
        os.chmod(self.projections_dir, 0o700)
        self._write_run({"run_id": self.run_id, "created_at": utc_now_iso(), **params})

    def read_run(self) -> dict[str, Any]:
        if not self.exists():
            raise JudgeError(f"no run {self.run_id!r} at {self.root}; run `mem judge candidates`")
        return json.loads(self.run_json.read_text(encoding="utf-8"))

    def update_run(self, **fields: Any) -> dict[str, Any]:
        data = self.read_run()
        data.update(fields)
        self._write_run(data)
        return data

    def _write_run(self, data: Mapping[str, Any]) -> None:
        tmp = self.run_json.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, indent=2, sort_keys=True, default=str), encoding="utf-8")
        os.replace(tmp, self.run_json)

    # JSONL

    @staticmethod
    def read_jsonl(path: Path) -> list[dict[str, Any]]:
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
        path.parent.mkdir(parents=True, exist_ok=True)
        n = 0
        target = path if append else path.with_name(path.name + ".tmp")
        if append:
            _mend_tail(path)
            stream = path.open("a", encoding="utf-8")
        else:
            fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            os.fchmod(fd, 0o600)
            stream = os.fdopen(fd, "w", encoding="utf-8")
        with stream as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True, default=str) + "\n")
                n += 1
        if not append:
            os.replace(target, path)
        return n

    def read_drafts(self) -> dict[str, list[dict[str, Any]]]:
        """Draft rows by item, each item's drafts in a stable order by drafter name."""
        by_item: dict[str, list[dict[str, Any]]] = {}
        for row in self.read_jsonl(self.drafts_path):
            by_item.setdefault(row["item_id"], []).append(row)
        for rows in by_item.values():
            rows.sort(key=lambda r: str(r.get("drafter") or ""))
        return by_item

    # Projections: the one place state text lives on disk. Mode 0600, purgeable.

    def write_projection(self, item_id: str, text: str) -> Path:
        self.projections_dir.mkdir(parents=True, exist_ok=True)
        path = self.projections_dir / f"{safe_name(item_id)}.txt"
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        return path

    def read_projection(self, item_id: str) -> str:
        path = self.projections_dir / f"{safe_name(item_id)}.txt"
        if not path.exists():
            raise JudgeError(f"no projection for item {item_id}; run `mem judge candidates`")
        return path.read_text(encoding="utf-8")

    def purge_projections(self) -> int:
        if not self.projections_dir.exists():
            return 0
        count = sum(1 for _ in self.projections_dir.iterdir())
        shutil.rmtree(self.projections_dir)
        return count
