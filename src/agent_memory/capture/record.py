"""The capture record: the one intermediate every adapter produces and the writer consumes."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any

RECORD_VERSION = 1

# Mirrors the event_kinds seed in migration 0001.
EVENT_KINDS = frozenset(
    {
        "session_start",
        "session_end",
        "user_msg",
        "assistant_msg",
        "system_context",
        "tool_call",
        "tool_result",
        "memory_recall",
        "memory_write",
        "compaction",
        "decision",
        "outcome",
        "capture_error",
        "note",
    }
)

# The kind a parent_key points at, when the child kind implies it.
PARENT_KIND = {"tool_result": "tool_call"}

# Shapes a serialized record must have; JSON written by anything else can hold any value.
_OBJECT_FIELDS = ("payload", "memory", "action", "raw", "observation", "attribution")
_TEXT_FIELDS = ("role", "content", "parent_key", "parent_kind")


class RecordError(ValueError):
    """A capture record is malformed."""


@dataclass(frozen=True)
class SessionInfo:
    cwd: str | None = None
    model: str | None = None
    harness_version: str | None = None
    project: str | None = None
    started_at: datetime | None = None
    scope: str | None = None
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class Record:
    agent_id: str
    harness_session_id: str
    kind: str
    ts: datetime
    source_key: str
    role: str | None = None
    content: str | None = None
    payload: dict[str, Any] | None = None
    parent_key: str | None = None
    parent_kind: str | None = None
    seq: int = 0
    session: SessionInfo | None = None
    # Set by mem-mcp when the server was down: the memory or review action to
    # create at replay, so a queued write becomes a real memory, not just an event.
    memory: dict[str, Any] | None = None
    action: dict[str, Any] | None = None
    raw: dict[str, Any] | None = field(default=None, compare=False)
    redactions: int = 0
    observation: dict[str, Any] | None = None
    attribution: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.kind not in EVENT_KINDS:
            raise RecordError(f"unknown event kind {self.kind!r}")
        if not self.agent_id or not self.harness_session_id or not self.source_key:
            raise RecordError("agent_id, harness_session_id, and source_key are required")
        if self.ts.tzinfo is None:
            raise RecordError("ts must be timezone-aware")
        if self.parent_key and not self.effective_parent_kind:
            raise RecordError(f"parent_kind is required for a parented {self.kind}")

    @property
    def effective_parent_kind(self) -> str | None:
        return self.parent_kind or PARENT_KIND.get(self.kind)

    def to_json(self) -> str:
        data = asdict(self)
        data["v"] = RECORD_VERSION
        data["ts"] = self.ts.isoformat()
        if self.session and self.session.started_at:
            data["session"]["started_at"] = self.session.started_at.isoformat()
        return json.dumps(data, ensure_ascii=False, default=str)

    @classmethod
    def from_json(cls, line: str) -> Record:
        try:
            data = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RecordError(f"not JSON: {exc}") from exc
        if not isinstance(data, dict):
            raise RecordError("record must be an object")
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Record:
        version = data.get("v", RECORD_VERSION)
        if version != RECORD_VERSION:
            raise RecordError(f"unsupported record version {version!r}")
        for name in _OBJECT_FIELDS:
            if not isinstance(data.get(name), dict | None):
                raise RecordError(f"{name} must be an object")
        for name in _TEXT_FIELDS:
            if not isinstance(data.get(name), str | None):
                raise RecordError(f"{name} must be a string")
        try:
            session = data.get("session")
            info = None
            if session is not None:
                if not isinstance(session, dict):
                    raise RecordError("session must be an object")
                if not isinstance(session.get("metadata"), dict | None):
                    raise RecordError("session.metadata must be an object")
                info = SessionInfo(
                    cwd=session.get("cwd"),
                    model=session.get("model"),
                    harness_version=session.get("harness_version"),
                    project=session.get("project"),
                    started_at=parse_ts(session["started_at"])
                    if session.get("started_at")
                    else None,
                    scope=session.get("scope"),
                    metadata=session.get("metadata"),
                )
            return cls(
                agent_id=str(data["agent_id"]),
                harness_session_id=str(data["harness_session_id"]),
                kind=str(data["kind"]),
                ts=parse_ts(data["ts"]),
                source_key=str(data["source_key"]),
                role=data.get("role"),
                content=data.get("content"),
                payload=data.get("payload"),
                parent_key=data.get("parent_key"),
                parent_kind=data.get("parent_kind"),
                seq=int(data.get("seq") or 0),
                session=info,
                memory=data.get("memory"),
                action=data.get("action"),
                raw=data.get("raw"),
                redactions=int(data.get("redactions") or 0),
                observation=data.get("observation"),
                attribution=data.get("attribution"),
            )
        except (KeyError, TypeError, ValueError) as exc:
            if isinstance(exc, RecordError):
                raise
            raise RecordError(f"bad record: {exc}") from exc


def parse_ts(value: Any) -> datetime:
    """Parse an ISO-8601 timestamp (Z or offset) or epoch seconds/milliseconds into UTC."""
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    if isinstance(value, int | float):
        seconds = value / 1000 if value > 1e11 else value
        return datetime.fromtimestamp(seconds, tz=UTC)
    if not isinstance(value, str):
        raise RecordError(f"bad timestamp {value!r}")
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise RecordError(f"bad timestamp {value!r}") from exc
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def now() -> datetime:
    return datetime.now(UTC)
