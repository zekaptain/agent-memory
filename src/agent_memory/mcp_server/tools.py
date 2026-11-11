"""The memory tools behind mem-mcp. Plain methods, so they can be tested without a transport."""

from __future__ import annotations

import contextlib
import os
import re
import uuid
from collections.abc import Sequence
from dataclasses import replace
from datetime import UTC, datetime
from typing import Any

import duckdb
from mcp.server.mcpserver.exceptions import ToolError as McpToolError

from ..capture import blobs, identity, ids
from ..capture.attribution import reconcile_answers
from ..capture.pipeline import deliver, make_redactor
from ..capture.record import SessionInfo, parse_ts
from ..capture.redact import Redactor
from ..capture.session import LedgerSession, Observed, UnknownAgent, event_row
from ..capture.spool import Spool
from ..config import Config, ConfigError
from ..db import AgentMemoryError, MemoryClient, sql_literal
from ..graph import (
    MAX_HOPS,
    EntityMerge,
    Relation,
    RelationListing,
    RelationWalk,
    apply_merge,
    normalize_entity,
)
from ..lifecycle import (
    REVIEW_ACTIONS,
    LifecycleError,
    LifecyclePlan,
    check_asker,
    is_validation_error,
    maintain_corrections,
    validate_interval,
)
from . import ranking
from .recall import Recalled, Strategy, run_recall

_UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
TRUST_LEVELS = ("user", "agent", "tool")
DEFAULT_LIMIT = 8
MAX_LIMIT = 50
DEFAULT_GRAPH_LIMIT = 25
MAX_GRAPH_LIMIT = 100
MCP = Observed("mem-mcp", "mcp", "mem_mcp_v1", ("non_mcp_activity_unobserved",))


class ToolError(McpToolError, ValueError):
    """Bad arguments from the agent. Subclassing the MCP error keeps the message in the
    tool result; any other exception is reported to the agent without detail."""


class MemoryTools:
    def __init__(
        self,
        cfg: Config,
        agent_id: str,
        *,
        harness_session_id: str | None = None,
        project: str | None = None,
        observed: Observed = MCP,
    ) -> None:
        self.cfg = cfg
        self.agent_id = agent_id
        self.harness_session_id, self.session_link = identity.mcp_harness_session(
            agent_id,
            {"AGENT_MEMORY_SESSION": harness_session_id}
            if harness_session_id is not None
            else os.environ,
        )
        self.own_session = self.session_link == "own"
        self.project = project or os.environ.get("AGENT_MEMORY_PROJECT") or _cwd_name()
        # How an MCP process found its harness session. A command names its own.
        link = (
            {"own_session": self.own_session, "session_link": self.session_link}
            if observed is MCP
            else None
        )
        self.session = LedgerSession(
            agent_id,
            self.harness_session_id,
            SessionInfo(cwd=_cwd(), project=self.project, metadata=link),
            observed,
        )
        self.session_id = self.session.session_id
        self._client: MemoryClient | None = None
        self._session_ready = False
        self._redactor: Redactor | None = None
        self._vocab: dict[str, set[str]] = {}

    # Connection and session

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None
        self._session_ready = False

    def _client_or_raise(self) -> MemoryClient:
        if self._client is None:
            self._client = MemoryClient.from_config(self.cfg)
            self._vocab = {}
        if not self._session_ready:
            try:
                self.session.open(self._client, self.redactor())
            except UnknownAgent as exc:
                raise ToolError(
                    f"{exc}; set AGENT_MEMORY_AGENT, or --agent at the CLI, to a registered agent"
                ) from None
            self._session_ready = True
            # Also covers CLI graph readers, which obtain a client directly.
            maintain_corrections(self._client)
        return self._client

    def _reset(self) -> None:
        if self._client is not None:
            with contextlib.suppress(duckdb.Error):
                self._client.close()
        self._client = None
        self._session_ready = False

    def _vocabulary(self, client: MemoryClient, table: str, column: str) -> set[str]:
        if table not in self._vocab:
            _, rows = client.query(f"SELECT {column} FROM {table}")
            self._vocab[table] = {str(r[0]) for r in rows}
        return self._vocab[table]

    def redactor(self) -> Redactor:
        if self._redactor is None:
            self._redactor = make_redactor(self.cfg)
        return self._redactor

    def _event(self, client: MemoryClient, kind: str, content: str | None, payload: Any) -> dict:
        content, content_count = self.redactor().value(content)
        payload, payload_count = self.redactor().value(payload)
        return event_row(
            self.cfg.blob_dir,
            self.session_id,
            client.new_ids()[0],
            kind,
            content,
            payload,
            ts=datetime.now(UTC),
            redactions=content_count + payload_count,
        )

    def _scopes(self, scopes: Sequence[str] | None) -> list[str]:
        values = (
            [s for s in scopes if s]
            if scopes
            else (["global", self.project] if self.project else ["global"])
        )
        return [self.redactor().text(value)[0] for value in values]

    # Tools

    def recall(
        self,
        query: str,
        scopes: Sequence[str] | None = None,
        limit: int = DEFAULT_LIMIT,
        types: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        limit = max(1, min(limit, MAX_LIMIT))
        scopes = self._scopes(scopes)
        asked = query
        query = self.redactor().text(query or "")[0]
        types = self.redactor().value(list(types) if types is not None else None)[0]
        got = self._recall(ranking.for_query(query, scopes, types, limit), asked)
        if got.reason is not None:
            return {"degraded": True, "reason": got.reason, "retrieval_id": None, "items": []}
        return {
            "degraded": False,
            "retrieval_id": got.retrieval_id,
            "request_event_id": got.request_event_id,
            "strategy": got.strategy,
            "candidates": got.candidates,
            "items": got.items,
            "hint": "Pass retrieval_id (and the memory_ids you relied on) to cite when you answer.",
        }

    def _recall(self, strategy: Strategy, asked: str | None) -> Recalled:
        """Recall in this session. An unreachable ledger gives a result with a reason
        rather than an error, so the agent can carry on without memory."""
        try:
            client = self._client_or_raise()
            return run_recall(
                client,
                strategy,
                lambda request: self._event(client, "memory_recall", asked, request),
            )
        except (AgentMemoryError, ConfigError, duckdb.Error) as exc:
            self._reset()
            return Recalled(strategy.name, reason=f"{type(exc).__name__}: {exc}")

    def remember(
        self,
        content: str,
        memory_type: str,
        scope: str | None = None,
        key: str | None = None,
        confidence: float = 0.7,
        importance: float = 0.5,
        trust_level: str = "agent",
        supersedes: str | None = None,
        source_event_ids: Sequence[str] | None = None,
        valid_from: str | None = None,
        expires_at: str | None = None,
        valid_to: str | None = None,
    ) -> dict[str, Any]:
        content = (content or "").strip()
        if not content:
            raise ToolError("content is required")
        if trust_level not in TRUST_LEVELS:
            raise ToolError(f"trust_level must be one of {', '.join(TRUST_LEVELS)}")
        if not 0 <= confidence <= 1 or not 0 <= importance <= 1:
            raise ToolError("confidence and importance must be between 0 and 1")
        if supersedes is not None:
            supersedes = _uuid(supersedes, "supersedes")
        sources = [_uuid(e, "source_event_ids") for e in (source_event_ids or [])]
        scope = self.redactor().text(scope or self.project or "global")[0]
        key = self.redactor().value(key)[0]
        redacted, _ = self.redactor().text(content)
        spec = {
            "content": redacted,
            "type": memory_type,
            "scope": scope,
            "key": key,
            "confidence": confidence,
            "importance": importance,
            "trust_level": trust_level,
            "supersedes": supersedes,
            "valid_from": valid_from,
            "expires_at": expires_at,
            "valid_to": valid_to,
            "metadata": None,
        }
        return self._write_memory(spec, sources)

    def _write_memory(
        self,
        spec: dict[str, Any],
        sources: Sequence[str],
    ) -> dict[str, Any]:
        """Use the same lifecycle plan as queued replay and entity merges."""
        spec = {**spec, "source_event_ids": list(sources)}
        try:
            validate_interval(spec)
        except LifecycleError as exc:
            raise ToolError(str(exc)) from None
        content = spec["content"]
        try:
            client = self._client_or_raise()
            if spec["type"] not in self._vocabulary(client, "memory_types", "type"):
                raise ToolError(
                    "type must be one of "
                    + ", ".join(sorted(self._vocabulary(client, "memory_types", "type")))
                )
            known_sources = self._existing_events(client, sources)
            event = self._event(client, "memory_write", content, spec)
            row = {
                "memory_id": client.new_ids()[0],
                **{
                    k: spec[k]
                    for k in (
                        "scope",
                        "type",
                        "key",
                        "content",
                        "confidence",
                        "importance",
                        "trust_level",
                        "supersedes",
                        "metadata",
                    )
                },
                "content_hash": blobs.content_hash(content),
                "valid_from": _ts(spec.get("valid_from")),
                "valid_to": _ts(spec.get("valid_to")),
                "expires_at": _ts(spec.get("expires_at")),
                "created_by_session": self.session_id,
            }
            plan = LifecyclePlan(client)
            mid, duplicate = plan.remember(
                row,
                event_id=event["event_id"],
                actor=self.agent_id,
                sources=[
                    {
                        "event_id": event["event_id"],
                        "relation": "user_stated"
                        if spec["trust_level"] == "user"
                        else "derived_from",
                    },
                    *({"event_id": eid, "relation": "derived_from"} for eid in known_sources),
                ],
            )
            client.insert([("events", [event]), *plan.rows])
            correction = {}
            if spec.get("supersedes"):
                _, proposals = client.query(
                    "SELECT correction_status, pending_reason FROM memory_proposal_status "
                    f"WHERE successor_id = {sql_literal(mid)} "
                    f"AND predecessor_id = {sql_literal(plan.resolve(spec['supersedes']))}"
                )
                if proposals:
                    correction = {
                        "correction_status": proposals[0][0],
                        "correction_pending_reason": proposals[0][1],
                    }
            return {
                "memory_id": mid,
                "deduplicated": duplicate,
                "queued": False,
                "status": self._status_of(client, mid),
                "pending_reason": self._pending_reason(client, mid),
                "ignored_source_event_ids": sorted(set(sources) - set(known_sources)),
                **correction,
            }
        except (AgentMemoryError, ConfigError, duckdb.Error) as exc:
            if is_validation_error(exc):
                raise ToolError(f"memory write conflict or validation failure: {exc}") from None
            self._reset()
            return self._queue(
                "memory_write",
                content,
                memory=spec,
                reason=f"{type(exc).__name__}: {exc}",
            )

    def cite(
        self,
        retrieval_id: str | None = None,
        memory_ids: Sequence[str] | None = None,
        context_event_ids: Sequence[str] | None = None,
        answer_event_id: str | None = None,
    ) -> dict[str, Any]:
        links: list[dict[str, Any]] = []
        if retrieval_id:
            links.append(
                {"link_type": "retrieval", "retrieval_id": _uuid(retrieval_id, "retrieval_id")}
            )
        for mid in memory_ids or []:
            links.append({"link_type": "cited_memory", "memory_id": _uuid(mid, "memory_ids")})
        for eid in context_event_ids or []:
            links.append(
                {"link_type": "context_event", "context_event_id": _uuid(eid, "context_event_ids")}
            )
        if not links:
            raise ToolError("cite needs a retrieval_id, memory_ids, or context_event_ids")
        try:
            client = self._client_or_raise()
            rows: dict[str, list[dict[str, Any]]] = {}
            explicit = _uuid(answer_event_id, "answer_event_id") if answer_event_id else None
            missing_explicit = explicit is not None and not self._existing_events(
                client, [explicit]
            )
            if explicit is None or missing_explicit:
                placeholder = self._event(
                    client,
                    "assistant_msg",
                    None,
                    {
                        "placeholder": True,
                        "retrieval_id": retrieval_id,
                        **({"explicit_answer_event_id": explicit} if explicit else {}),
                    },
                )
                answer_event_id = placeholder["event_id"]
                rows["events"] = [placeholder]
            else:
                answer_event_id = explicit
            rows["context_links"] = [
                {
                    "answer_event_id": answer_event_id,
                    "link_type": link["link_type"],
                    "retrieval_id": link.get("retrieval_id"),
                    "memory_id": link.get("memory_id"),
                    "context_event_id": link.get("context_event_id"),
                }
                for link in links
            ]
            client.insert(rows)
            reconcile_answers(
                client, [self.session_id], placeholder_event_ids=[str(answer_event_id)]
            )
        except (AgentMemoryError, ConfigError, duckdb.Error) as exc:
            self._reset()
            return {"degraded": True, "reason": f"{type(exc).__name__}: {exc}", "links": 0}
        return {
            "degraded": False,
            "answer_event_id": explicit or answer_event_id,
            "links": len(links),
            **({"placeholder_event_id": answer_event_id} if missing_explicit else {}),
        }

    def review(
        self,
        action: str,
        memory_id: str,
        reason: str,
        related_memory_id: str | None = None,
    ) -> dict[str, Any]:
        if action not in REVIEW_ACTIONS:
            raise ToolError("action must be one of " + ", ".join(sorted(REVIEW_ACTIONS)))
        try:
            # Refused before anything is written or queued.
            check_asker(self.agent_id, action)
        except LifecycleError as exc:
            raise ToolError(str(exc)) from None
        memory_id = _uuid(memory_id, "memory_id")
        if related_memory_id is not None:
            related_memory_id = _uuid(related_memory_id, "related_memory_id")
        if not (reason or "").strip():
            raise ToolError("reason is required")
        original_reason = reason
        reason = self.redactor().text(reason)[0]
        spec = {
            "memory_id": memory_id,
            "action": action,
            "actor": self.agent_id,
            "related_memory_id": related_memory_id,
            "reason": reason,
        }
        try:
            client = self._client_or_raise()
            event = self._event(client, "memory_write", original_reason, {"review": spec})
            plan = LifecyclePlan(client)
            resolved = plan.review(
                {
                    "action_id": client.new_ids()[0],
                    "session_id": self.session_id,
                    "event_id": event["event_id"],
                    **spec,
                }
            )
            client.insert([("events", [event]), *plan.rows])
            return {
                "memory_id": memory_id,
                "action": action,
                "queued": False,
                "resolved_memory_id": resolved,
                "status": self._status_of(client, resolved),
                "pending_reason": self._pending_reason(client, resolved),
            }
        except (AgentMemoryError, ConfigError, duckdb.Error) as exc:
            if is_validation_error(exc):
                raise ToolError(f"review conflict or validation failure: {exc}") from None
            self._reset()
            return self._queue(
                "memory_write", reason, action=spec, reason=f"{type(exc).__name__}: {exc}"
            )

    def status(self) -> dict[str, Any]:
        spool = Spool(self.cfg.spool_dir)
        info: dict[str, Any] = {
            "agent_id": self.agent_id,
            "session_id": self.session_id,
            "harness_session_id": self.harness_session_id,
            "project": self.project,
            "server": self.cfg.uri,
            "spool_pending_lines": sum(s.lines for s in spool.status()),
            "spool_rejected_lines": sum(s.lines for s in spool.rejected()),
        }
        try:
            client = self._client_or_raise()
            _, rows = client.query(
                "SELECT kind, count(*) FROM events "
                f"WHERE session_id = {sql_literal(self.session_id)} GROUP BY kind ORDER BY kind"
            )
            info["reachable"] = True
            info["events_this_session"] = {str(k): int(n) for k, n in rows}
        except (AgentMemoryError, ConfigError, duckdb.Error) as exc:
            self._reset()
            info["reachable"] = False
            info["reason"] = f"{type(exc).__name__}: {exc}"
        return info

    # Graph: relations stored as fact memories, traversed with a recursive CTE

    def relate(
        self,
        subject: str,
        predicate: str,
        object: str,
        subject_type: str | None = None,
        object_type: str | None = None,
        content: str | None = None,
        scope: str | None = None,
        confidence: float = 0.7,
        importance: float = 0.5,
        trust_level: str = "agent",
        source_event_ids: Sequence[str] | None = None,
        valid_from: str | None = None,
        expires_at: str | None = None,
        valid_to: str | None = None,
    ) -> dict[str, Any]:
        redact = self.redactor()
        try:
            relation = Relation(
                redact.text(subject or "")[0],
                redact.text(predicate or "")[0],
                redact.text(object or "")[0],
                redact.value(subject_type)[0],
                redact.value(object_type)[0],
            )
        except ValueError as exc:
            raise ToolError(str(exc)) from None
        if trust_level not in TRUST_LEVELS:
            raise ToolError(f"trust_level must be one of {', '.join(TRUST_LEVELS)}")
        if not 0 <= confidence <= 1 or not 0 <= importance <= 1:
            raise ToolError("confidence and importance must be between 0 and 1")
        sources = [_uuid(e, "source_event_ids") for e in (source_event_ids or [])]
        scope = redact.text(scope or self.project or "global")[0]
        redacted, _ = redact.text((content or "").strip() or relation.sentence)
        spec = {
            "content": redacted,
            "type": "fact",
            "scope": scope,
            "key": relation.key,
            "confidence": confidence,
            "importance": importance,
            "trust_level": trust_level,
            "supersedes": None,
            "valid_from": valid_from,
            "expires_at": expires_at,
            "valid_to": valid_to,
            "metadata": relation.metadata(),
        }
        result = self._write_memory(spec, sources)
        result["key"] = relation.key
        result["triple"] = relation.triple
        return result

    def graph(
        self,
        entity: str,
        hops: int = 1,
        scopes: Sequence[str] | None = None,
        limit: int = DEFAULT_GRAPH_LIMIT,
        types: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        asked = entity
        entity = self.redactor().text(entity or "")[0]
        types = self.redactor().value(list(types) if types is not None else None)[0]
        entity_norm = normalize_entity(entity or "")
        if not entity_norm:
            raise ToolError("entity is required")
        hops = max(1, min(int(hops), MAX_HOPS))
        limit = max(1, min(int(limit), MAX_GRAPH_LIMIT))
        walk = RelationWalk(entity, entity_norm, hops, self._scopes(scopes), types, limit)
        got = self._recall(walk, asked)
        if got.reason is not None:
            return {
                "degraded": True,
                "reason": got.reason,
                "retrieval_id": None,
                "items": [],
                "triples": [],
            }
        names: dict[str, str] = {}
        for item in got.items:
            names.setdefault(normalize_entity(item["subject"]), item["subject"])
            names.setdefault(normalize_entity(item["object"]), item["object"])
        return {
            "degraded": False,
            "retrieval_id": got.retrieval_id,
            "request_event_id": got.request_event_id,
            "strategy": got.strategy,
            "entity": entity,
            "entity_norm": entity_norm,
            "hops": hops,
            "candidates": got.candidates,
            "items": got.items,
            "triples": [item["triple"] for item in got.items],
            "entities": sorted(names.values(), key=str.casefold),
            "hint": "Answer from these edges and pass retrieval_id (and the memory_ids you "
            "relied on) to cite.",
        }

    def relations(self, scopes: Sequence[str] | None = None) -> dict[str, Any]:
        """Every active relation in the scopes, oldest first. Not offered to agents as a
        tool: `mem graph` reads the whole graph through it, so that read is recorded like
        any other recall."""
        got = self._recall(RelationListing(self._scopes(scopes)), None)
        if got.reason is not None:
            return {"degraded": True, "reason": got.reason, "retrieval_id": None, "items": []}
        return {
            "degraded": False,
            "retrieval_id": got.retrieval_id,
            "request_event_id": got.request_event_id,
            "strategy": got.strategy,
            "candidates": got.candidates,
            "items": got.items,
        }

    def merge_entity(
        self,
        alias: str,
        canonical: str,
        reason: str,
        scopes: Sequence[str] | None = None,
        canonical_type: str | None = None,
        actor: str | None = None,
        retrieval_id: str | None = None,
    ) -> dict[str, Any]:
        """Entity resolution as append-only history: what `graph.apply_merge` does, in this
        session. `retrieval_id` names the recall the merge was proposed from."""
        redact = self.redactor()
        try:
            merge = EntityMerge(
                alias=redact.text(alias or "")[0],
                canonical=redact.text(canonical or "")[0],
                reason=redact.text(reason or "")[0],
                actor=redact.text(actor or self.agent_id)[0],
                scopes=self._scopes(scopes),
                canonical_type=redact.value(canonical_type)[0],
            )
        except ValueError as exc:
            raise ToolError(str(exc)) from None
        if retrieval_id is not None:
            merge = replace(merge, retrieval_id=_uuid(retrieval_id, "retrieval_id"))
        try:
            client = self._client_or_raise()
            done = apply_merge(
                client,
                merge,
                lambda content, payload: self._event(client, "memory_write", content, payload),
                redact,
            )
            return {
                "degraded": False,
                "alias": merge.alias,
                "canonical": merge.canonical,
                **done,
            }
        except (AgentMemoryError, ConfigError, duckdb.Error) as exc:
            if is_validation_error(exc):
                raise ToolError(f"merge conflict or validation failure: {exc}") from None
            self._reset()
            return {
                "degraded": True,
                "reason": f"{type(exc).__name__}: {exc}",
                "alias": merge.alias,
                "canonical": merge.canonical,
                "superseded": [],
                "retired": [],
            }

    # Helpers

    def _status_of(self, client: MemoryClient, memory_id: str) -> str | None:
        _, rows = client.query(
            f"SELECT status FROM memory_status WHERE memory_id = {sql_literal(memory_id)}"
        )
        return str(rows[0][0]) if rows else None

    def _pending_reason(self, client: MemoryClient, memory_id: str) -> str | None:
        _, rows = client.query(
            f"SELECT pending_reason FROM memory_status WHERE memory_id = {sql_literal(memory_id)}"
        )
        return rows[0][0] if rows else None

    def _existing_events(self, client: MemoryClient, event_ids: Sequence[str]) -> list[str]:
        if not event_ids:
            return []
        wanted = ", ".join(sql_literal(e) for e in sorted(set(event_ids)))
        _, rows = client.query(f"SELECT event_id::VARCHAR FROM events WHERE event_id IN ({wanted})")
        return [str(r[0]) for r in rows]

    def _queue(
        self,
        kind: str,
        content: str | None,
        *,
        reason: str,
        memory: dict[str, Any] | None = None,
        action: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        source_key = f"mcp:{uuid.uuid4()}"
        record = self.session.record(
            kind,
            source_key,
            role="memory",
            content=content,
            payload={"queued": True, "memory": memory, "review": action},
            memory=memory,
            action=action,
        )
        delivery = deliver(self.cfg, [record], spool_only=True)
        out: dict[str, Any] = {
            "queued": True,
            "reason": reason,
            "spool": [str(p) for p in delivery.spooled],
        }
        if memory is not None:
            out["memory_id"] = ids.memory_id(self.agent_id, self.harness_session_id, source_key)
            out["deduplicated"] = False
        if action is not None:
            out["memory_id"] = action["memory_id"]
            out["action"] = action["action"]
        return out


def _uuid(value: str, name: str) -> str:
    value = value.strip().lower()
    if not _UUID.fullmatch(value):
        raise ToolError(f"{name} must be a UUID, got {value!r}")
    return value


def _ts(value: str | None) -> datetime | None:
    if value is None:
        return None
    try:
        return parse_ts(value)
    except ValueError as exc:
        raise ToolError(f"bad timestamp {value!r}") from exc


def _cwd() -> str | None:
    try:
        return os.getcwd()
    except OSError:
        return None


def _cwd_name() -> str | None:
    return os.path.basename(_cwd() or "") or None
