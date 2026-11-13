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
from .store import JudgeError, RunStore, imports_log

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


# The migration from which `session_summary` gives a session's title and its caller.
SUMMARY_MIGRATION = 9
CALLERS_SQL = (
    "SELECT session_id::VARCHAR, harness_session_id, title, calling_session_id FROM session_summary"
)


def require_summary(cfg: Config, client: MemoryClient) -> None:
    """Fail where the database served has not had the migration that puts a session's title
    and caller in `session_summary`. A server started before that migration existed goes on
    serving the older view, so a command asks this before it imports or writes anything."""
    _, applied = client.query(
        f"SELECT 1 FROM schema_migrations WHERE version = {SUMMARY_MIGRATION}"
    )
    if not applied:
        raise JudgeError(
            f"the database served at {cfg.uri} lacks migration {SUMMARY_MIGRATION:04d}: stop "
            f"the server, run mem --home {cfg.home} migrate, and start it again"
        )


def session_callers(cfg: Config, client: MemoryClient) -> dict[str, Caller]:
    """Caller per ledger session id, from the title and the calling session that
    `session_summary` gives: what a transcript import said of the session, else a hook, else
    its row, else its start events. A session the summary names no caller for is read from
    the first line of its transcript where the import log points at one, as for a session
    imported before the adapter kept the caller."""
    require_summary(cfg, client)
    transcripts = {
        Path(row["path"]).stem: Path(row["path"])
        for row in RunStore.read_jsonl(imports_log(cfg))
        if row.get("path")
    }
    _, rows = client.query(CALLERS_SQL)
    out: dict[str, Caller] = {}
    for sid, hsid, title, calling in rows:
        start = {"title": title, "calling_session_id": calling}
        if not calling and hsid in transcripts:
            start.update(_start_entry(transcripts[str(hsid)]))
        out[str(sid)] = caller_from_start(start)
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
