"""Private, atomic admission files and effective human-example history."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Iterable, Mapping
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from ..capture.redact import Redactor
from ..config import Config
from ..judge.store import RunStore, safe_name
from . import AdmissionError


def private_dir(path: Path) -> None:
    """Create a private leaf, refusing symlinks rather than changing their targets."""
    if path.is_symlink():
        raise AdmissionError("admission directories must not be symbolic links")
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    path.chmod(0o700)


def redactor_for_config(cfg: Config) -> Redactor:
    return Redactor([cfg.read_token()] if cfg.token_path.exists() else [])


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    if path.is_symlink():
        raise AdmissionError("admission files must not be symbolic links")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, UnicodeError):
        raise AdmissionError(f"invalid admission JSON: {path.name}") from None
    if not isinstance(data, dict):
        raise AdmissionError(f"expected an admission object: {path.name}")
    return data


def _atomic_text(path: Path, body: str) -> None:
    private_dir(path.parent)
    if path.is_symlink():
        raise AdmissionError("admission files must not be symbolic links")
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


def write_json(path: Path, data: Mapping[str, Any]) -> None:
    clean, _ = Redactor().value(dict(data))
    _atomic_text(path, json.dumps(clean, indent=2, sort_keys=True, allow_nan=False) + "\n")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]], *, append: bool = False) -> int:
    """Write private redacted JSONL; reject torn existing input rather than silently losing it."""
    values = list(rows)
    clean, _ = Redactor().value(values)
    body = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n" for row in clean
    )
    if append and path.exists():
        read_jsonl(path)  # A corrupt history must not be disguised by a new append.
        previous = path.read_text(encoding="utf-8")
        body = previous + ("\n" if previous and not previous.endswith("\n") else "") + body
    _atomic_text(path, body)
    return len(values)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    if path.is_symlink():
        raise AdmissionError("admission files must not be symbolic links")
    rows = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError
            rows.append(row)
    except (ValueError, UnicodeError):
        raise AdmissionError(f"invalid admission JSONL: {path.name}") from None
    return rows


def memory_split(memory_id: str) -> str:
    """A memory keeps its 60/40 split across human corrections and reimports."""
    bucket = int(hashlib.sha256(memory_id.encode()).hexdigest(), 16) % 10
    return "calibration" if bucket < 6 else "held_out"


def latest_examples(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """One effective sample per memory (latest action) or experiment source."""
    latest: dict[str, dict[str, Any]] = {}
    orders: dict[str, tuple[datetime, int, int]] = {}
    actions: dict[tuple[str, str], int] = {}
    for original in rows:
        if not isinstance(original, Mapping):
            raise AdmissionError("admission examples must be objects")
        row = dict(original)
        identity = row.get("id") or row.get("source")
        if not identity:
            raise AdmissionError("admission example is missing its source identity")
        source = json.dumps(identity, sort_keys=True)
        row.setdefault("id", hashlib.sha256(source.encode()).hexdigest())
        memory_id = row.get("memory_id")
        if memory_id:
            key = f"memory:{memory_id}"
            row["split"] = memory_split(str(memory_id))
            row["group_id"] = key
            # Retry order is not human-action order. Old pending reviews replayed after
            # a correction cannot resurrect their old label or gain duplicate influence.
            order = actions.setdefault((key, source), len(actions))
            stamp = row.get("reviewed_at")
            try:
                when = datetime.fromisoformat(stamp) if stamp else datetime.min.replace(tzinfo=UTC)
                when = when.replace(tzinfo=UTC) if when.tzinfo is None else when.astimezone(UTC)
                revision = row.get("revision", 0)
                if type(revision) is not int:
                    raise ValueError
            except (ValueError, TypeError):
                raise AdmissionError("invalid human review timestamp or revision") from None
            current_order = (when, revision, order)
            if key in orders and current_order < orders[key]:
                continue
            orders[key] = current_order
        else:
            key = source
        latest[key] = row
    return [latest[key] for key in sorted(latest)]


class _DayStore(RunStore):
    """RunStore mechanics, without weakening the experiment command's home guard."""

    def _write_run(self, data: Mapping[str, Any]) -> None:
        clean, _ = redactor_for_config(self.cfg).value(dict(data))
        write_json(self.run_json, clean)

    def write_projection(self, item_id: str, text: str) -> Path:
        private_dir(self.projections_dir)
        path = self.projections_dir / f"{safe_name(item_id)}.txt"
        clean, _ = redactor_for_config(self.cfg).text(text)
        _atomic_text(path, clean)
        return path

    @staticmethod
    def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]], *, append: bool = False) -> int:
        return write_jsonl(path, rows, append=append)

    def create(self, params: Mapping[str, Any]) -> None:
        super().create(params)
        private_dir(self.verdicts_dir)
        private_dir(self.projections_dir)


class AdmissionStore:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.root = cfg.home / "admission"
        private_dir(cfg.home)
        private_dir(self.root)

    @property
    def state_path(self) -> Path:
        return self.root / "state.json"

    @property
    def approver_path(self) -> Path:
        return self.root / "approver.json"

    @property
    def examples_path(self) -> Path:
        return self.root / "examples.jsonl"

    def day(self, day: str) -> RunStore:
        try:
            if date.fromisoformat(day).isoformat() != day:
                raise ValueError
        except ValueError:
            raise AdmissionError("admission day must be YYYY-MM-DD") from None
        private_dir(self.root / "days")
        return _DayStore(self.cfg, day, root=self.root / "days" / day)

    def read_state(self) -> dict[str, Any]:
        data = {"mode": "shadow", "watermark": None, "approver_version": None}
        data.update(_read_json(self.state_path))
        if data["mode"] not in ("shadow", "auto"):
            raise AdmissionError("invalid admission mode")
        return data

    def write_state(self, **fields: Any) -> dict[str, Any]:
        state = self.read_state()
        state.update(fields)
        if state["mode"] not in ("shadow", "auto"):
            raise AdmissionError("invalid admission mode")
        clean, _ = redactor_for_config(self.cfg).value(state)
        write_json(self.state_path, clean)
        return clean

    def read_approver(self) -> dict[str, Any]:
        return _read_json(self.approver_path)

    def write_approver(self, data: Mapping[str, Any]) -> None:
        clean, _ = redactor_for_config(self.cfg).value(dict(data))
        write_json(self.approver_path, clean)

    def read_examples(self) -> list[dict[str, Any]]:
        return latest_examples(read_jsonl(self.examples_path))
