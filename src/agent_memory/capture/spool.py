"""JSONL spool: where capture records wait when the server is unreachable."""

from __future__ import annotations

import fcntl
import os
import re
import shutil
import tempfile
import time
import uuid
from collections.abc import Generator, Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from .record import Record

_SAFE = re.compile(r"[^A-Za-z0-9._-]+")
_CLAIM_SUFFIX = re.compile(r"\.claim-[0-9a-f]{32}$")


def _safe(name: str) -> str:
    return _SAFE.sub("_", name)[:200] or "_"


@contextmanager
def _lock(path: Path, *, blocking: bool = True) -> Generator[bool]:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT, 0o600)
    with os.fdopen(fd, "w") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        except BlockingIOError:
            yield False
            return
        yield True


@dataclass(frozen=True)
class SpoolFile:
    path: Path
    agent_id: str
    harness_session_id: str


@dataclass(frozen=True)
class SpoolStatus:
    agent_id: str
    files: int
    lines: int
    oldest: float | None  # mtime of the oldest pending file


class Spool:
    def __init__(self, root: Path) -> None:
        self.root = root

    @property
    def done_dir(self) -> Path:
        return self.root / "done"

    @property
    def failed_dir(self) -> Path:
        return self.root / "failed"

    def path_for(self, agent_id: str, harness_session_id: str) -> Path:
        return self.root / _safe(agent_id) / f"{_safe(harness_session_id)}.jsonl"

    def append(self, records: Iterable[Record]) -> list[Path]:
        """Append under a stable session lock shared with replay's claim operation."""
        by_file: dict[Path, list[str]] = {}
        for rec in records:
            by_file.setdefault(self.path_for(rec.agent_id, rec.harness_session_id), []).append(
                rec.to_json()
            )
        for path, lines in by_file.items():
            with _lock(path.with_suffix(".lock")):
                fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
                with os.fdopen(fd, "w", encoding="utf-8") as fh:
                    fh.write("".join(line + "\n" for line in lines))
        return list(by_file)

    def pending(self) -> list[SpoolFile]:
        if not self.root.exists():
            return []
        found = []
        recovering = self.root / "replaying"
        directories = sorted(recovering.iterdir()) if recovering.exists() else []
        directories.extend(sorted(p for p in self.root.iterdir() if p.is_dir()))
        for agent_dir in directories:
            if agent_dir.name in ("done", "failed", "replaying"):
                continue
            for path in sorted(agent_dir.glob("*.jsonl"), key=lambda p: p.stat().st_mtime):
                hsid = (
                    _CLAIM_SUFFIX.sub("", path.stem)
                    if agent_dir.parent == recovering
                    else path.stem
                )
                found.append(SpoolFile(path, agent_dir.name, hsid))
        return found

    @contextmanager
    def replay_lock(self) -> Generator[bool]:
        """Only one replay owns claimed files; hooks never acquire this long-lived lock."""
        with _lock(self.root / ".replay.lock", blocking=False) as acquired:
            yield acquired

    def claim(self, file: SpoolFile) -> SpoolFile:
        if file.path.parent.parent == self.root / "replaying":
            return file
        target = self.root / "replaying" / file.agent_id / file.path.name
        target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        with _lock(file.path.with_suffix(".lock")):
            if target.exists():
                target = target.with_name(f"{file.path.stem}.claim-{uuid.uuid4().hex}.jsonl")
            file.path.rename(target)
        return SpoolFile(target, file.agent_id, file.harness_session_id)

    @staticmethod
    def rewrite(file: SpoolFile, lines: Iterable[str]) -> None:
        """Publish a redacted snapshot of a claimed file before archiving it."""
        fd, name = tempfile.mkstemp(prefix=".redacted-", dir=file.path.parent)
        temporary = Path(name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                stream.writelines(line + "\n" for line in lines)
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(file.path)
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def lines(path: Path) -> Iterator[tuple[int, str]]:
        with path.open(encoding="utf-8") as fh:
            for number, line in enumerate(fh, start=1):
                if line.strip():
                    yield number, line.rstrip("\n")

    def mark_done(self, file: SpoolFile) -> Path:
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        target = self.done_dir / file.agent_id / f"{file.path.stem}.{stamp}.{time.time_ns()}.jsonl"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(file.path, target)
        return target

    def write_failed(self, file: SpoolFile, lines: Iterable[str]) -> Path | None:
        lines = list(lines)
        if not lines:
            return None
        target = self.failed_dir / file.agent_id / file.path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write("".join(line + "\n" for line in lines))
        return target

    def purge_done(self, older_than_days: float) -> list[Path]:
        cutoff = time.time() - older_than_days * 86400
        removed = []
        if self.done_dir.exists():
            for path in self.done_dir.rglob("*.jsonl"):
                if path.stat().st_mtime < cutoff:
                    path.unlink()
                    removed.append(path)
        return removed

    def status(self) -> list[SpoolStatus]:
        per_agent: dict[str, list[SpoolFile]] = {}
        for file in self.pending():
            per_agent.setdefault(file.agent_id, []).append(file)
        out = []
        for agent, files in sorted(per_agent.items()):
            lines = sum(1 for f in files for _ in self.lines(f.path))
            oldest = min(f.path.stat().st_mtime for f in files)
            out.append(SpoolStatus(agent, len(files), lines, oldest))
        return out
