"""Run files, write-once manifests, and experiment-home guard for the usefulness experiment.

Reuses the atomic-write, locking, private-directory, and JSON-lines primitives
from `judge/store.py` while keeping its own error type, home guard message, and
write-once semantics for manifests and registrations.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ..config import DEFAULT_HOME, Config
from ..db import AgentMemoryError
from ..judge.store import (
    JudgeError,
    RunStore,
    atomic_write,
    lock,
    private_dir,
    read_json,
    sole_writer,
    utc_now_iso,
)

_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")


class UsefulnessError(AgentMemoryError):
    """A usefulness-experiment operation cannot proceed for an expected reason."""


def ensure_experiment_home(home: Config | Path) -> Path:
    """Refuse to run the usefulness experiment against the default production home."""
    path = home.home if isinstance(home, Config) else home
    resolved = path.expanduser().resolve()
    if resolved == DEFAULT_HOME.expanduser().resolve():
        raise UsefulnessError(
            "the usefulness experiment refuses to use the default home. Point "
            "AGENT_MEMORY_HOME or --home at a separate experiment directory."
        )
    return resolved


def sha256_bytes(data: bytes) -> str:
    """Lowercase hex SHA-256 digest of ``data``."""
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    """Lowercase hex SHA-256 digest of the file at ``path``."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    """Lowercase hex SHA-256 of canonical UTF-8 JSON for ``value``."""
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return sha256_bytes(encoded)


def _lock_path(path: Path) -> Path:
    return path.with_name(path.name + ".lock")


def write_once_json(path: Path, data: Mapping[str, Any]) -> str:
    """Write ``data`` as canonical JSON to ``path`` once, refusing if it already exists.

    Returns the SHA-256 hex digest of the bytes written to disk.
    """
    private_dir(path.parent)
    with lock(_lock_path(path)):
        if path.exists():
            raise UsefulnessError(f"{path} already exists and cannot be overwritten")
        body = json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        atomic_write(path, body)
        return sha256_bytes(body.encode("utf-8"))


def read_json_file(path: Path) -> dict[str, Any]:
    """Read a JSON object from ``path``, translating store errors to ``UsefulnessError``."""
    if not path.exists():
        raise UsefulnessError(f"missing required file: {path}")
    try:
        return read_json(path)
    except JudgeError as exc:
        raise UsefulnessError(str(exc)) from None


def append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    """Append one JSON object to an append-only JSON-lines file under a writer lock."""
    private_dir(path.parent)
    try:
        with sole_writer(path):
            RunStore.write_jsonl(path, [row], append=True)
    except JudgeError as exc:
        raise UsefulnessError(str(exc)) from None


def read_jsonl(path: Path, *, strict: bool = False) -> list[dict[str, Any]]:
    """Read rows from a JSON-lines file, returning [] if the file does not exist."""
    try:
        return RunStore.read_jsonl(path, strict=strict)
    except JudgeError as exc:
        raise UsefulnessError(str(exc)) from None


class UsefulnessRunStore:
    """Files of one usefulness run under ``<experiment_home>/runs/<run_id>/``.

    Write-once files (task manifest, registration) cannot be overwritten through
    this store. Deviations are append-only and never rewrite earlier rows.
    """

    def __init__(self, home: Config | Path, run_id: str) -> None:
        if not _RUN_ID.fullmatch(run_id):
            raise UsefulnessError(
                f"bad run id {run_id!r}: letters, digits, dot, dash, underscore only"
            )
        self.home = ensure_experiment_home(home)
        self.run_id = run_id
        self.root = self.home / "runs" / run_id

    @property
    def tasks_dir(self) -> Path:
        return self.home / "tasks"

    @property
    def memory_set_dir(self) -> Path:
        return self.home / "memory-set"

    @property
    def task_manifest_path(self) -> Path:
        return self.root / "task-manifest.json"

    @property
    def registration_path(self) -> Path:
        return self.root / "registration.json"

    @property
    def deviations_path(self) -> Path:
        return self.root / "deviations.jsonl"

    @property
    def trials_dir(self) -> Path:
        return self.root / "trials"

    @property
    def grades_path(self) -> Path:
        return self.root / "grades.jsonl"

    @property
    def report_path(self) -> Path:
        return self.root / "report.md"

    def write_task_manifest(self, manifest: Mapping[str, Any]) -> str:
        """Write ``task-manifest.json`` once and return its SHA-256."""
        return write_once_json(self.task_manifest_path, manifest)

    def read_task_manifest(self) -> dict[str, Any]:
        """Read ``task-manifest.json``."""
        return read_json_file(self.task_manifest_path)

    def task_manifest_sha256(self) -> str:
        """SHA-256 of the written ``task-manifest.json``."""
        if not self.task_manifest_path.exists():
            raise UsefulnessError(f"no task manifest at {self.task_manifest_path}")
        return sha256_file(self.task_manifest_path)

    def write_registration(self, registration: Mapping[str, Any]) -> str:
        """Write ``registration.json`` once at the freeze and return its SHA-256."""
        return write_once_json(self.registration_path, registration)

    def read_registration(self) -> dict[str, Any]:
        """Read ``registration.json``."""
        return read_json_file(self.registration_path)

    def registration_sha256(self) -> str:
        """SHA-256 of the written ``registration.json``."""
        if not self.registration_path.exists():
            raise UsefulnessError(f"no registration at {self.registration_path}")
        return sha256_file(self.registration_path)

    def append_deviation(
        self,
        reason: str,
        *,
        date: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Append a dated deviation record; earlier deviations are never rewritten."""
        if not isinstance(reason, str) or not reason.strip():
            raise UsefulnessError("deviation reason must be a non-empty string")
        row: dict[str, Any] = {
            "recorded_at": date if date is not None else utc_now_iso(),
            "reason": reason.strip(),
        }
        if metadata is not None:
            row["metadata"] = dict(metadata)
        append_jsonl(self.deviations_path, row)
        return row

    def deviations(self) -> list[dict[str, Any]]:
        """Every recorded deviation, in append order."""
        return read_jsonl(self.deviations_path)
