"""The ledger session: one shape for a session row and an event row, whoever writes them.

The capture writer, the MCP tools, `mem graph`, and admission all put sessions and events
into the ledger, and each composes its own transaction. They take the rows from here, so a
session is the same row whether it is opened online or from the spool, and every event's
content is stored by one rule: hashed, measured, and moved to a blob when it is large.

Text handed to `session_row` and `event_row` is already redacted. `LedgerSession.open` takes
the redactor and applies it itself.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import duckdb

from ..db import AgentMemoryError, MemoryClient, sql_literal
from . import blobs, ids
from .observations import fingerprint, pending_observations
from .record import Record, SessionInfo, now
from .redact import Redactor

ROLE_FOR_KIND = {
    "user_msg": "user",
    "assistant_msg": "assistant",
    "tool_call": "assistant",
    "tool_result": "tool",
    "memory_recall": "memory",
    "memory_write": "memory",
    "session_start": "harness",
    "session_end": "harness",
    "compaction": "harness",
    "system_context": "harness",
    "capture_error": "harness",
}

# How often a write is repeated after another writer committed the same rows first.
RACE_RETRIES = 3


class UnknownAgent(AgentMemoryError):
    """The session names an agent the ledger has not registered."""


@dataclass(frozen=True)
class Observed:
    """How a session's activity reaches the ledger when no harness adapter reports it.
    `via` is kept in the session's metadata; the rest becomes its capture observation."""

    via: str
    source_mode: str
    adapter_version: str
    exclusions: tuple[str, ...] = ("no_complete_capture_evidence",)


def session_row(
    agent_id: str, harness_session_id: str, info: SessionInfo, started_at: datetime
) -> dict[str, Any]:
    """The row for a session the ledger does not hold yet. `started_at` stands in when
    `info` does not say when the session began."""
    cwd = info.cwd
    return {
        "session_id": ids.session_id(agent_id, harness_session_id),
        "agent_id": agent_id,
        "harness_session_id": harness_session_id,
        "harness_version": info.harness_version,
        "model": info.model,
        "scope": info.scope or "global",
        "project": info.project or (Path(cwd).name if cwd else None),
        "cwd": cwd,
        "capture_completeness": "partial",
        "started_at": info.started_at or started_at,
        "metadata": info.metadata,
    }


def event_row(
    blob_dir: Path,
    session_id: str,
    event_id: str,
    kind: str,
    content: str | None,
    payload: dict[str, Any] | None,
    *,
    ts: datetime,
    role: str | None = None,
    seq: int = 0,
    redactions: int = 0,
) -> dict[str, Any]:
    """The row for one event. Content above the blob threshold is written to `blob_dir`;
    the row keeps its first bytes, its hash, and its size."""
    stored = blobs.store(blob_dir, content)
    return {
        "event_id": event_id,
        "session_id": session_id,
        "seq": seq,
        "kind": kind,
        "role": role or ROLE_FOR_KIND.get(kind),
        "content": stored.content,
        "content_hash": stored.content_hash,
        "content_bytes": stored.content_bytes,
        "truncated": stored.truncated,
        "blob_ref": stored.blob_ref,
        "payload": payload,
        "parent_event_id": None,
        "token_estimate": stored.content_bytes // 4 if stored.content_bytes else None,
        "redactions": redactions,
        "ts": ts,
    }


def retrying[T](action: Callable[[], T]) -> T:
    """Repeat a write that lost a race. Another writer can commit the same session or event
    between our pre-read and our commit; reading again finds its rows and skips them."""
    retries = RACE_RETRIES
    while True:
        try:
            return action()
        except duckdb.Error as exc:
            # A conflict found at commit is a TransactionException; one found at insert is a
            # ConstraintException that names a duplicate key.
            lost_race = isinstance(exc, duckdb.TransactionException) or (
                isinstance(exc, duckdb.ConstraintException) and "Duplicate key" in str(exc)
            )
            if not lost_race or not retries:
                raise
            retries -= 1


class LedgerSession:
    """A session as the MCP tools, `mem graph`, and admission keep it: opened once, then
    given events in the caller's own transactions."""

    def __init__(
        self, agent_id: str, harness_session_id: str, info: SessionInfo, observed: Observed
    ) -> None:
        self.agent_id = agent_id
        self.harness_session_id = harness_session_id
        self.session_id = ids.session_id(agent_id, harness_session_id)
        self.info = replace(info, metadata={"via": observed.via, **(info.metadata or {})})
        self.observed = observed

    def record(self, kind: str, source_key: str, **fields: Any) -> Record:
        """A capture record in this session. It carries what the session row and the
        observation are made from, so a record that reaches the ledger from the spool before
        `open` has run opens the same session."""
        return Record(
            agent_id=self.agent_id,
            harness_session_id=self.harness_session_id,
            kind=kind,
            ts=now(),
            source_key=source_key,
            session=self.info,
            observation={
                "source_mode": self.observed.source_mode,
                "adapter_version": self.observed.adapter_version,
                "source_fingerprint": fingerprint([self.observed.source_mode, self.session_id]),
                "exclusions": list(self.observed.exclusions),
            },
            **fields,
        )

    def open(self, client: MemoryClient, redactor: Redactor) -> None:
        """Write the session row and its observation unless the ledger already has them.

        Another writer may open the same session at the same moment: the first commit wins
        and the other finds the row. Raises UnknownAgent, having written nothing, when the
        agent is not registered.
        """
        opening = redactor.record(self.record("note", "session-open"))
        info = opening.session or self.info

        def attempt() -> None:
            rows: list[tuple[str, list[dict[str, Any]]]] = []
            _, found = client.query(
                f"SELECT 1 FROM sessions WHERE session_id = {sql_literal(self.session_id)}"
            )
            if not found:
                _, registered = client.query(
                    f"SELECT 1 FROM agents WHERE agent_id = {sql_literal(self.agent_id)}"
                )
                if not registered:
                    raise UnknownAgent(f"unknown agent {self.agent_id!r}")
                row = session_row(self.agent_id, self.harness_session_id, info, opening.ts)
                rows.append(("sessions", [row]))
            observations = pending_observations(client, [opening])
            if observations:
                rows.append(("capture_observations", observations))
            if rows:
                client.insert(rows)

        retrying(attempt)
