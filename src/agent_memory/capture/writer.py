"""Turns capture records into ledger rows and writes them idempotently."""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import duckdb

from ..db import AgentMemoryError, MemoryClient, sql_literal
from . import blobs, ids
from .record import Record, SessionInfo, parse_ts
from .redact import Redactor

_UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
_ROLE_FOR_KIND = {
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


class WriterError(AgentMemoryError):
    """Records could not be turned into rows (unknown agent, bad reference)."""


@dataclass
class WriteResult:
    sessions: int = 0
    events: int = 0
    memories: int = 0
    actions: int = 0
    skipped: int = 0
    unresolved_parents: list[str] = field(default_factory=list)

    @property
    def inserted(self) -> int:
        return self.sessions + self.events + self.memories + self.actions


class Writer:
    """Pre-reads what the ledger already has, builds the missing rows, inserts them in one
    transaction. Deterministic ids make a replay a no-op; a concurrent capturer that wins
    the race only costs one retry."""

    def __init__(
        self,
        client: MemoryClient,
        blob_dir: Path,
        redactor: Redactor,
        *,
        keep_raw: bool = False,
    ) -> None:
        self.client = client
        self.blob_dir = blob_dir
        self.redactor = redactor
        self.keep_raw = keep_raw
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

    def _existing_events_for_sessions(self, session_ids: Iterable[str]) -> set[str]:
        wanted = sorted(set(session_ids))
        if not wanted:
            return set()
        chunk = ", ".join(sql_literal(v) for v in wanted)
        _, rows = self.client.query(
            f"SELECT event_id::VARCHAR FROM events WHERE session_id IN ({chunk})"
        )
        return {str(r[0]) for r in rows}

    # Writing

    def write(self, records: Sequence[Record]) -> WriteResult:
        if not records:
            return WriteResult()
        tiers = self.agent_tiers()
        for rec in records:
            if rec.agent_id not in tiers:
                raise WriterError(f"unknown agent {rec.agent_id!r}")
        try:
            return self._write_once(records, tiers)
        except duckdb.ConstraintException as exc:
            if "Duplicate key" not in str(exc):
                raise
            # Another capturer inserted the same rows between our read and write.
            return self._write_once(records, tiers)

    def _write_once(self, records: Sequence[Record], tiers: dict[str, str]) -> WriteResult:
        result = WriteResult()
        rows: list[tuple[str, list[dict[str, Any]]]] = []

        # Sessions
        groups: dict[tuple[str, str], list[Record]] = {}
        for rec in records:
            groups.setdefault((rec.agent_id, rec.harness_session_id), []).append(rec)
        session_of = {key: ids.session_id(*key) for key in groups}
        have_sessions = self._existing("sessions", "session_id", session_of.values())
        session_rows = []
        for key, recs in groups.items():
            sid = session_of[key]
            if sid in have_sessions:
                continue
            session_rows.append(self._session_row(sid, key[0], recs, tiers[key[0]]))
        if session_rows:
            rows.append(("sessions", session_rows))
            result.sessions = len(session_rows)

        # Events
        event_ids = {
            id(rec): ids.event_id(rec.agent_id, rec.harness_session_id, rec.kind, rec.source_key)
            for rec in records
        }
        have_events = self._existing_events_for_sessions(session_of.values())
        batch_ids = {eid for eid in event_ids.values() if eid not in have_events}
        depth_of: dict[str, int] = {}
        event_rows: dict[str, dict[str, Any]] = {}
        for rec in records:
            eid = event_ids[id(rec)]
            if eid in have_events or eid in event_rows:
                result.skipped += 1
                continue
            parent_id = None
            unresolved = None
            if rec.parent_key:
                candidate = ids.event_id(
                    rec.agent_id,
                    rec.harness_session_id,
                    rec.effective_parent_kind or "",
                    rec.parent_key,
                )
                if candidate in have_events or candidate in batch_ids:
                    parent_id = candidate
                else:
                    unresolved = rec.parent_key
                    result.unresolved_parents.append(rec.parent_key)
            depth = 0
            if parent_id in batch_ids:
                depth = depth_of.get(parent_id, 0) + 1
            depth_of[eid] = depth
            row = self._event_row(eid, session_of[(rec.agent_id, rec.harness_session_id)], rec)
            row["parent_event_id"] = parent_id
            if unresolved:
                row["payload"] = {**(row["payload"] or {}), "unresolved_parent": unresolved}
            event_rows[eid] = row
        if event_rows:
            # One statement per depth, parents first (see MemoryClient.insert).
            layers: dict[int, list[dict[str, Any]]] = {}
            for row in sorted(event_rows.values(), key=lambda r: (r["seq"], r["ts"])):
                layers.setdefault(depth_of[row["event_id"]], []).append(row)
            for depth in sorted(layers):
                rows.append(("events", layers[depth]))
            result.events = len(event_rows)

        # Memories and review actions queued by mem-mcp during an outage
        memory_recs = [r for r in records if r.memory]
        action_recs = [r for r in records if r.action]
        if memory_recs:
            mids = {
                id(r): ids.memory_id(r.agent_id, r.harness_session_id, r.source_key)
                for r in memory_recs
            }
            have = self._existing("memories", "memory_id", mids.values())
            mem_rows, src_rows = [], []
            for rec in memory_recs:
                mid = mids[id(rec)]
                if mid in have:
                    continue
                sid = session_of[(rec.agent_id, rec.harness_session_id)]
                mem_rows.append(self._memory_row(mid, sid, rec))
                trust = (rec.memory or {}).get("trust_level")
                relation = "user_stated" if trust == "user" else "derived_from"
                src_rows.append(
                    {"memory_id": mid, "event_id": event_ids[id(rec)], "relation": relation}
                )
            if mem_rows:
                rows.append(("memories", mem_rows))
                rows.append(("memory_sources", src_rows))
                result.memories = len(mem_rows)
        if action_recs:
            aids = {
                id(r): ids.action_id(r.agent_id, r.harness_session_id, r.source_key)
                for r in action_recs
            }
            have = self._existing("memory_actions", "action_id", aids.values())
            act_rows = []
            for rec in action_recs:
                aid = aids[id(rec)]
                if aid in have or not rec.action:
                    continue
                sid = session_of[(rec.agent_id, rec.harness_session_id)]
                act_rows.append(
                    {
                        "action_id": aid,
                        "memory_id": rec.action["memory_id"],
                        "action": rec.action["action"],
                        "actor": rec.action.get("actor") or rec.agent_id,
                        "session_id": sid,
                        "related_memory_id": rec.action.get("related_memory_id"),
                        "reason": rec.action.get("reason"),
                    }
                )
            if act_rows:
                rows.append(("memory_actions", act_rows))
                result.actions = len(act_rows)

        if rows:
            self.client.insert(rows)
        return result

    # Row builders

    def _session_row(
        self, sid: str, agent_id: str, recs: Sequence[Record], tier: str
    ) -> dict[str, Any]:
        info = SessionInfo()
        for rec in sorted(recs, key=lambda r: (r.kind != "session_start", r.ts)):
            if rec.session:
                info = _merge(info, rec.session)
        started = info.started_at or min(r.ts for r in recs)
        cwd = info.cwd
        project = info.project or (Path(cwd).name if cwd else None)
        metadata, _ = self.redactor.value(info.metadata) if info.metadata else (None, 0)
        return {
            "session_id": sid,
            "agent_id": agent_id,
            "harness_session_id": recs[0].harness_session_id,
            "harness_version": info.harness_version,
            "model": info.model,
            "scope": info.scope or "global",
            "project": project,
            "cwd": cwd,
            "capture_completeness": "full" if tier == "A" else "partial",
            "started_at": started,
            "metadata": metadata,
        }

    def _event_row(self, eid: str, sid: str, rec: Record) -> dict[str, Any]:
        redactions = 0
        content = rec.content
        if content is not None:
            content, n = self.redactor.text(content)
            redactions += n
        payload: dict[str, Any] | None = None
        if rec.payload is not None:
            payload, n = self.redactor.value(rec.payload)
            redactions += n
        if self.keep_raw and rec.raw is not None:
            raw, n = self.redactor.value(rec.raw)
            redactions += n
            payload = {**(payload or {}), "raw": raw}
        stored = blobs.store(self.blob_dir, content)
        return {
            "event_id": eid,
            "session_id": sid,
            "seq": rec.seq,
            "kind": rec.kind,
            "role": rec.role or _ROLE_FOR_KIND.get(rec.kind),
            "content": stored.content,
            "content_hash": stored.content_hash,
            "content_bytes": stored.content_bytes,
            "truncated": stored.truncated,
            "blob_ref": stored.blob_ref,
            "payload": payload,
            "parent_event_id": None,
            "token_estimate": (stored.content_bytes or 0) // 4 if stored.content_bytes else None,
            "redactions": redactions,
            "ts": rec.ts,
        }

    def _memory_row(self, mid: str, sid: str, rec: Record) -> dict[str, Any]:
        spec = rec.memory or {}
        content, _ = self.redactor.text(str(spec.get("content", "")))
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
            "expires_at": _ts(spec.get("expires_at")),
            "created_by_session": sid,
            "metadata": spec.get("metadata"),
        }


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
