"""Who is speaking (`provenance_v1`): the caller of a session and the author of a user message.

A Droid transcript's `user` role carries three different authors. A person types in a
top-level session. A parent agent writes the opening task and any follow-ups of a spawned
session (workers, reviewers, explorers). The harness injects system reminders, async
completion notices, and interruption markers. Triggers that read "the user's next message"
mean something different in each case, so every user message is tagged before use.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from ..config import Config
from ..db import MemoryClient
from .store import RunStore, imports_log

PROVENANCE_VERSION = "provenance_v1"

CallerKind = Literal["human", "agent"]
Provenance = Literal["human", "parent_agent", "harness"]

_BLOCK = re.compile(r"<system-(reminder|notification)>.*?</system-\1>", re.DOTALL)
# A block the inline 4 KiB cut off before its closing tag.
_UNCLOSED = re.compile(r"<system-(?:reminder|notification)>.*\Z", re.DOTALL)
_ROLE = re.compile(r"^\s*([A-Za-z][\w -]{0,40}?)\s*:")

# Openings of what remains after the harness blocks are removed. Lowercased prefix match.
PARENT_OPENINGS: tuple[str, ...] = (
    "# task tool invocation",
    "# follow-up instructions",
    "## your assigned feature",
)
HARNESS_OPENINGS: tuple[str, ...] = (
    "request interrupted by user",
    "request cancelled by user",
    "the user interrupted the agent",
    "background task completed",
    "background execute commands",
    "script output from",
    "[script completed",
    "script toolu_",
    "mcp status:",
    'skill "',
    "[resume the mission]",
    "you've reached your",
)
HARNESS_EXACT: frozenset[str] = frozenset({"interrupted"})


def strip_harness_blocks(text: str | None) -> tuple[str, int]:
    """The message without its `<system-reminder>`/`<system-notification>` blocks, and how
    many were removed."""
    if not text:
        return "", 0
    body, n = _BLOCK.subn("", text)
    body, m = _UNCLOSED.subn("", body)
    return body.strip(), n + m


def message_provenance(text: str | None, caller_kind: CallerKind) -> Provenance:
    body, _ = strip_harness_blocks(text)
    if not body:
        return "harness"
    head = body[:80].lower()
    if head.startswith(PARENT_OPENINGS):
        return "parent_agent"
    if head.startswith(HARNESS_OPENINGS) or head in HARNESS_EXACT:
        return "harness"
    return "parent_agent" if caller_kind == "agent" else "human"


@dataclass(frozen=True)
class Caller:
    """Who started the session: a person, or an agent in another session."""

    kind: CallerKind = "human"
    role: str | None = None  # spawned droid's role from its title prefix, e.g. "worker"
    calling_session_id: str | None = None

    @property
    def label(self) -> str:
        return f"agent:{self.role}" if self.kind == "agent" and self.role else self.kind

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


HUMAN = Caller()


def role_from_title(title: str | None) -> str | None:
    match = _ROLE.match(title or "")
    if not match:
        return None
    return re.sub(r"[\s_]+", "-", match.group(1).strip().lower())


def caller_from_start(entry: Mapping[str, Any]) -> Caller:
    """From a transcript's `session_start` entry, or the session metadata built from one."""
    calling = entry.get("callingSessionId") or entry.get("calling_session_id")
    if not calling:
        return HUMAN
    return Caller(
        kind="agent", role=role_from_title(entry.get("title")), calling_session_id=str(calling)
    )


SESSIONS_SQL = (
    "SELECT session_id::VARCHAR, harness_session_id, CAST(metadata AS JSON)::VARCHAR FROM sessions"
)
# What a session's start hooks said of the session that called it.
STARTS_SQL = (
    "SELECT session_id::VARCHAR, max(caller) FROM ("
    "SELECT session_id, json_extract_string(CAST(payload AS JSON), '$.calling_session_id') "
    "AS caller FROM events WHERE kind = 'session_start') WHERE caller IS NOT NULL "
    "GROUP BY session_id"
)


def session_callers(cfg: Config, client: MemoryClient) -> dict[str, Caller]:
    """Caller per ledger session id, from the session's row. A row that names no caller is
    completed from elsewhere. Where the import log points at the session's transcript, the
    caller and the title come from its first line, as for a session imported before the
    adapter read them there. Failing that, the caller is what the session's start hooks
    said, which their events hold: a start hook wrote the row before the adapter kept the
    caller there. A hook names no title, so that caller has no role."""
    transcripts = {
        Path(row["path"]).stem: Path(row["path"])
        for row in RunStore.read_jsonl(imports_log(cfg))
        if row.get("path")
    }
    _, rows = client.query(SESSIONS_SQL)
    _, starts = client.query(STARTS_SQL)
    said_at_start = {str(sid): str(caller) for sid, caller in starts}
    out: dict[str, Caller] = {}
    for sid, hsid, raw in rows:
        meta: dict[str, Any] = {}
        if raw:
            try:
                parsed = json.loads(raw)
                meta = parsed if isinstance(parsed, dict) else {}
            except ValueError:
                meta = {}
        if "calling_session_id" not in meta and hsid in transcripts:
            meta = {**meta, **_start_entry(transcripts[str(hsid)])}
        if not meta.get("calling_session_id") and str(sid) in said_at_start:
            meta = {**meta, "calling_session_id": said_at_start[str(sid)]}
        out[str(sid)] = caller_from_start(meta)
    return out


def _start_entry(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as fh:
            first = json.loads(fh.readline())
    except (OSError, ValueError):
        return {}
    if not isinstance(first, dict) or first.get("type") != "session_start":
        return {}
    return {
        "calling_session_id": first.get("callingSessionId"),
        "title": first.get("title"),
    }
