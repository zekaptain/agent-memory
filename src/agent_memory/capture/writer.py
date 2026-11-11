"""Turns capture records into ledger rows and writes them idempotently."""

from __future__ import annotations

import contextlib
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from graphlib import CycleError, TopologicalSorter
from pathlib import Path
from typing import Any

import duckdb

from ..db import AgentMemoryError, MemoryClient, sql_literal
from ..lifecycle import LifecycleError, LifecyclePlan, is_validation_error
from . import blobs, ids
from .attribution import reconcile_answers
from .observations import pending_observations
from .record import Record, SessionInfo, parse_ts
from .redact import Redactor
from .session import event_row, retrying, session_row

_UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")


class WriterError(AgentMemoryError):
    """A record could not be turned into rows (a bad reference, a memory without content)."""


@dataclass(frozen=True)
class Rejection:
    """A capture record the ledger refused, with its position in the batch given to `write`."""

    index: int
    record: Record
    reason: str


@dataclass
class WriteResult:
    sessions: int = 0
    events: int = 0
    memories: int = 0
    actions: int = 0
    skipped: int = 0
    unresolved_parents: list[str] = field(default_factory=list)
    rejected: list[Rejection] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def inserted(self) -> int:
        return self.sessions + self.events + self.memories + self.actions

    def add(self, part: WriteResult) -> None:
        self.sessions += part.sessions
        self.events += part.events
        self.memories += part.memories
        self.actions += part.actions
        self.skipped += part.skipped
        self.unresolved_parents.extend(part.unresolved_parents)


class Writer:
    """Pre-reads what the ledger already has, builds the missing rows, and inserts them in one
    transaction. Deterministic ids make a replay a no-op, and a concurrent capturer that
    commits first only costs another read. A record the ledger refuses is set aside so that
    the rest of its batch still lands."""

    def __init__(
        self,
        client: MemoryClient,
        blob_dir: Path,
        redactor: Redactor,
    ) -> None:
        self.client = client
        self.blob_dir = blob_dir
        self.redactor = redactor
        self._tiers: dict[str, str] | None = None

    # Lookups

    def agent_tiers(self) -> dict[str, str]:
        if self._tiers is None:
            _, rows = self.client.query("SELECT agent_id, capture_tier FROM agents")
            self._tiers = {str(a): str(t) for a, t in rows}
        return self._tiers

    def _existing(self, table: str, column: str, values: Iterable[str]) -> set[str]:
        wanted = sorted(set(values))
        if not wanted:
            return set()
        for value in wanted:
            if not _UUID.fullmatch(value):
                raise WriterError(f"{column} {value!r} is not a UUID")
        found: set[str] = set()
        # Keep each statement a sane size; sessions can have thousands of events.
        for start in range(0, len(wanted), 500):
            chunk = ", ".join(sql_literal(v) for v in wanted[start : start + 500])
            _, rows = self.client.query(
                f"SELECT {column}::VARCHAR FROM {table} WHERE {column} IN ({chunk})"
            )
            found.update(str(r[0]) for r in rows)
        return found

    # Writing

    def write(self, records: Sequence[Record]) -> WriteResult:
        """Write every record the ledger accepts and report the rest as rejected.

        A record the ledger refuses is rejected with the reason, along with every record
        that refers to it; the others are still written. Raises only when the ledger is
        unavailable (AgentMemoryError, duckdb.Error, or OSError). Deterministic ids make it
        safe to hand over the same batch again later.
        """
        result = WriteResult()
        if not records:
            return result
        batch = [self.redactor.record(rec) for rec in records]
        groups: dict[tuple[str, str], list[Record]] = {}
        for rec in batch:
            groups.setdefault((rec.agent_id, rec.harness_session_id), []).append(rec)
        dependencies, unreadable = _dependencies(batch)
        order, cyclic = _dependency_order(dependencies)
        rejected = {index: self._reason(exc) for index, exc in unreadable.items()}
        rejected.update(dict.fromkeys(cyclic, "cyclic record references"))
        tiers = self.agent_tiers()
        for index, rec in enumerate(batch):
            if rec.agent_id not in tiers:
                rejected[index] = f"unknown agent {rec.agent_id!r}"

        def attempt(chunk: list[int]) -> None:
            for index in chunk:
                if dependencies[index] & rejected.keys():
                    rejected[index] = "a referenced record was rejected"
            chunk = [index for index in chunk if index not in rejected]
            if not chunk:
                return
            part = [batch[index] for index in sorted(chunk)]
            try:
                result.add(retrying(lambda: self._write_once(part, groups, batch)))
            except Exception as exc:
                if _ledger_unavailable(exc):
                    raise
                if len(chunk) == 1:
                    rejected[chunk[0]] = self._reason(exc)
                    return
                # No error names the record that caused it, so halve until one is left.
                # References precede their referrers, so the first half never needs the second.
                half = len(chunk) // 2
                attempt(chunk[:half])
                attempt(chunk[half:])

        attempt(order)
        result.rejected = [
            Rejection(index, batch[index], rejected[index]) for index in sorted(rejected)
        ]
        # Evidence is already durable. A later delivery can finish this derived binding
        # step, even when the original capture committed before its connection failed.
        sessions = [ids.session_id(*key) for key in groups]
        try:
            retrying(lambda: reconcile_answers(self.client, sessions))
        except Exception as exc:
            if _ledger_unavailable(exc):
                raise
            result.warnings.append(f"answer binding failed: {self._reason(exc)}")
        return result

    def _reason(self, exc: Exception) -> str:
        expected = isinstance(exc, (AgentMemoryError, duckdb.Error))
        return self.redactor.text(str(exc) if expected else f"{type(exc).__name__}: {exc}")[0]

    def _write_once(
        self,
        records: Sequence[Record],
        groups: Mapping[tuple[str, str], Sequence[Record]],
        batch: Sequence[Record],
    ) -> WriteResult:
        """Write `records` in one transaction. `groups` and `batch` describe the whole batch
        they come from, so a session row and its coverage do not depend on how it was split."""
        result = WriteResult()
        rows: list[tuple[str, list[dict[str, Any]]]] = []

        # Sessions
        session_of = {
            (rec.agent_id, rec.harness_session_id): ids.session_id(
                rec.agent_id, rec.harness_session_id
            )
            for rec in records
        }
        have_sessions = self._existing("sessions", "session_id", session_of.values())
        session_rows = [
            _session_row(key[0], groups[key])
            for key, sid in session_of.items()
            if sid not in have_sessions
        ]
        if session_rows:
            rows.append(("sessions", session_rows))
            result.sessions = len(session_rows)

        # Events
        event_ids = {
            id(rec): ids.event_id(rec.agent_id, rec.harness_session_id, rec.kind, rec.source_key)
            for rec in records
        }
        parent_ids = {
            id(rec): ids.event_id(
                rec.agent_id,
                rec.harness_session_id,
                rec.effective_parent_kind or "",
                rec.parent_key,
            )
            for rec in records
            if rec.parent_key
        }
        have_events = self._existing(
            "events", "event_id", [*event_ids.values(), *parent_ids.values()]
        )
        batch_ids = {eid for eid in event_ids.values() if eid not in have_events}
        event_rows: dict[str, dict[str, Any]] = {}
        for rec in records:
            eid = event_ids[id(rec)]
            if eid in have_events or eid in event_rows:
                result.skipped += 1
                continue
            parent_id = None
            unresolved = None
            if rec.parent_key:
                candidate = parent_ids[id(rec)]
                if candidate in have_events or candidate in batch_ids:
                    parent_id = candidate
                else:
                    unresolved = rec.parent_key
                    result.unresolved_parents.append(rec.parent_key)
            row = self._event_row(eid, session_of[(rec.agent_id, rec.harness_session_id)], rec)
            row["parent_event_id"] = parent_id
            if unresolved:
                row["payload"] = {
                    **(row["payload"] or {}),
                    "unresolved_parent": self.redactor.text(unresolved)[0],
                }
            event_rows[eid] = row
        if event_rows:
            depth_of: dict[str, int] = {}
            for eid in event_rows:
                chain: list[str] = []
                seen: set[str] = set()
                current = eid
                while current in event_rows and current not in depth_of:
                    if current in seen:
                        raise WriterError("cyclic event parent references")
                    seen.add(current)
                    chain.append(current)
                    current = event_rows[current]["parent_event_id"]
                depth = depth_of.get(current, -1)
                for ancestor in reversed(chain):
                    depth += 1
                    depth_of[ancestor] = depth
            # One statement per depth, parents first (see MemoryClient.insert).
            layers: dict[int, list[dict[str, Any]]] = {}
            for row in sorted(event_rows.values(), key=lambda r: (r["seq"], r["ts"])):
                layers.setdefault(depth_of[row["event_id"]], []).append(row)
            for depth in sorted(layers):
                rows.append(("events", layers[depth]))
            result.events = len(event_rows)

        # Online writes and queued replay share exactly the same dedupe,
        # proposal, sources and review semantics. Returned queued IDs persist.
        plan = LifecyclePlan(self.client)
        memory_recs = {
            ids.memory_id(r.agent_id, r.harness_session_id, r.source_key): r
            for r in records
            if r.memory
        }
        dependencies = {
            mid: {str(rec.memory["supersedes"])} & memory_recs.keys()
            if rec.memory and rec.memory.get("supersedes")
            else set()
            for mid, rec in memory_recs.items()
        }
        try:
            ordered = list(TopologicalSorter(dependencies).static_order())
        except CycleError as exc:
            raise LifecycleError("cyclic queued correction references") from exc
        for mid in ordered:
            rec = memory_recs[mid]
            sid = session_of[(rec.agent_id, rec.harness_session_id)]
            spec = rec.memory or {}
            wanted = spec.get("source_event_ids") or []
            known = self._existing("events", "event_id", wanted) | (set(wanted) & batch_ids)
            plan.remember(
                self._memory_row(mid, sid, rec),
                event_id=event_ids[id(rec)],
                actor=rec.agent_id,
                preserve_identity=True,
                sources=[
                    {
                        "event_id": event_ids[id(rec)],
                        "relation": "user_stated"
                        if spec.get("trust_level") == "user"
                        else "derived_from",
                    },
                    *({"event_id": eid, "relation": "derived_from"} for eid in sorted(known)),
                ],
            )
        for rec in records:
            if not rec.action:
                continue
            sid = session_of[(rec.agent_id, rec.harness_session_id)]
            plan.review(
                {
                    "action_id": ids.action_id(
                        rec.agent_id, rec.harness_session_id, rec.source_key
                    ),
                    "memory_id": rec.action["memory_id"],
                    "action": rec.action["action"],
                    "actor": rec.action.get("actor") or rec.agent_id,
                    "session_id": sid,
                    "event_id": event_ids[id(rec)],
                    "related_memory_id": rec.action.get("related_memory_id"),
                    "reason": rec.action.get("reason"),
                }
            )
        rows.extend(plan.rows)
        result.memories = sum(len(rs) for table, rs in plan.rows if table == "memories")
        result.actions = sum(len(rs) for table, rs in plan.rows if table == "memory_actions")
        # Default observation cursors are derived from stable capture identities,
        # which the record redactor intentionally does not rewrite.
        observations = [
            self.redactor.value(row)[0]
            for row in pending_observations(self.client, records, batch=batch)
        ]
        if observations:
            rows.append(("capture_observations", observations))

        if rows:
            self.client.insert(rows)
        return result

    # Row builders

    def _event_row(self, eid: str, sid: str, rec: Record) -> dict[str, Any]:
        payload = rec.payload
        if rec.attribution:
            payload = {**(payload or {}), "capture_attribution": rec.attribution}
        if rec.raw is not None:
            payload = {**(payload or {}), "raw": rec.raw}
        return event_row(
            self.blob_dir,
            sid,
            eid,
            rec.kind,
            rec.content,
            payload,
            ts=rec.ts,
            role=rec.role,
            seq=rec.seq,
            redactions=rec.redactions,
        )

    def _memory_row(self, mid: str, sid: str, rec: Record) -> dict[str, Any]:
        spec = rec.memory or {}
        content = str(spec.get("content", ""))
        if not content:
            raise WriterError(f"memory in record {rec.source_key} has no content")
        return {
            "memory_id": mid,
            "scope": spec.get("scope") or "global",
            "type": spec.get("type") or "fact",
            "key": spec.get("key"),
            "content": content,
            "content_hash": blobs.content_hash(content),
            "confidence": float(spec.get("confidence", 0.7)),
            "importance": float(spec.get("importance", 0.5)),
            "trust_level": spec.get("trust_level") or "agent",
            "supersedes": spec.get("supersedes"),
            "valid_from": _ts(spec.get("valid_from")),
            "valid_to": _ts(spec.get("valid_to")),
            "expires_at": _ts(spec.get("expires_at")),
            "created_by_session": sid,
            "metadata": spec.get("metadata"),
        }


def _session_row(agent_id: str, recs: Sequence[Record]) -> dict[str, Any]:
    """The session row from everything the batch's records say about their session."""
    info = SessionInfo()
    for rec in sorted(recs, key=lambda r: (r.kind != "session_start", r.ts)):
        if rec.session:
            info = _merge(info, rec.session)
    return session_row(agent_id, recs[0].harness_session_id, info, min(r.ts for r in recs))


def _ledger_unavailable(exc: Exception) -> bool:
    """True when no record caused the failure, so handing the batch over again later can work."""
    if isinstance(exc, (WriterError, LifecycleError)):
        return False
    if isinstance(exc, duckdb.Error):
        # A commit conflict that outlasts its retries is contention, not a bad record.
        return isinstance(exc, duckdb.TransactionException) or not is_validation_error(exc)
    return isinstance(exc, (AgentMemoryError, OSError))


def _dependencies(
    batch: Sequence[Record],
) -> tuple[dict[int, set[int]], dict[int, Exception]]:
    """For each position in the batch, the positions of the records it refers to: its parent
    event, the events a memory names as sources, and the memories a correction or review
    names. Also returns the positions whose references could not be read, with the error."""
    events = {
        ids.event_id(rec.agent_id, rec.harness_session_id, rec.kind, rec.source_key): index
        for index, rec in enumerate(batch)
    }
    memories = {
        ids.memory_id(rec.agent_id, rec.harness_session_id, rec.source_key): index
        for index, rec in enumerate(batch)
        if rec.memory
    }

    def references(rec: Record) -> set[int]:
        found: set[int | None] = set()
        if rec.parent_key and (kind := rec.effective_parent_kind):
            found.add(
                events.get(ids.event_id(rec.agent_id, rec.harness_session_id, kind, rec.parent_key))
            )
        named: list[Any] = []
        if rec.action:
            named += [rec.action.get("memory_id"), rec.action.get("related_memory_id")]
        if rec.memory:
            named.append(rec.memory.get("supersedes"))
            found.update(events.get(source) for source in rec.memory.get("source_event_ids") or [])
        found.update(memories.get(name) for name in named if isinstance(name, str))
        return {index for index in found if index is not None}

    dependencies: dict[int, set[int]] = {}
    unreadable: dict[int, Exception] = {}
    for index, rec in enumerate(batch):
        try:
            # A record never waits for itself; writing it reports a self-reference.
            dependencies[index] = references(rec) - {index}
        except Exception as exc:
            dependencies[index] = set()
            unreadable[index] = exc
    return dependencies, unreadable


def _dependency_order(dependencies: Mapping[int, set[int]]) -> tuple[list[int], list[int]]:
    """Positions ordered so that every reference comes before its referrer, followed by the
    positions that a reference cycle keeps out of that order."""
    sorter = TopologicalSorter(dependencies)
    # After a CycleError the sorter still hands out every position that no cycle blocks.
    with contextlib.suppress(CycleError):
        sorter.prepare()
    order: list[int] = []
    while ready := sorter.get_ready():
        order.extend(ready)
        sorter.done(*ready)
    return order, sorted(dependencies.keys() - set(order))


def _merge(base: SessionInfo, extra: SessionInfo) -> SessionInfo:
    return SessionInfo(
        cwd=base.cwd or extra.cwd,
        model=base.model or extra.model,
        harness_version=base.harness_version or extra.harness_version,
        project=base.project or extra.project,
        started_at=base.started_at or extra.started_at,
        scope=base.scope or extra.scope,
        metadata={**(extra.metadata or {}), **(base.metadata or {})}
        if (base.metadata or extra.metadata)
        else None,
    )


def _ts(value: Any) -> datetime | None:
    return None if value is None else parse_ts(value)
