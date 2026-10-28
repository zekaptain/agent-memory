"""The memory tools behind mem-mcp. Plain methods, so they can be tested without a transport."""

from __future__ import annotations

import contextlib
import json
import os
import re
import time
import uuid
from collections import defaultdict
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

import duckdb
from mcp.server.mcpserver.exceptions import ToolError as McpToolError

from ..capture import blobs, ids
from ..capture.pipeline import deliver, make_redactor
from ..capture.record import Record, parse_ts
from ..capture.redact import Redactor
from ..capture.spool import Spool
from ..config import Config, ConfigError
from ..db import AgentMemoryError, MemoryClient, sql_literal
from ..graph import GRAPH_STRATEGY, MAX_HOPS, Relation, hop_sql, normalize_entity, relations_sql
from ..graph.sql import ROW as GRAPH_ROW
from ..graph.sql import active_keys_sql, sources_sql
from . import ranking

_UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
TRUST_LEVELS = ("user", "agent", "tool")
DEFAULT_LIMIT = 8
MAX_LIMIT = 50
DEFAULT_GRAPH_LIMIT = 25
MAX_GRAPH_LIMIT = 100


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
    ) -> None:
        self.cfg = cfg
        self.agent_id = agent_id
        self.own_session = harness_session_id is None
        self.harness_session_id = harness_session_id or (
            f"mcp-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{os.getpid()}"
        )
        self.session_id = ids.session_id(agent_id, self.harness_session_id)
        self.project = project or os.environ.get("AGENT_MEMORY_PROJECT") or _cwd_name()
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
            self._ensure_session(self._client)
            self._session_ready = True
        return self._client

    def _reset(self) -> None:
        if self._client is not None:
            with contextlib.suppress(duckdb.Error):
                self._client.close()
        self._client = None
        self._session_ready = False

    def _ensure_session(self, client: MemoryClient) -> None:
        _, rows = client.query(
            f"SELECT 1 FROM sessions WHERE session_id = {sql_literal(self.session_id)}"
        )
        if rows:
            return
        _, tier = client.query(
            f"SELECT capture_tier FROM agents WHERE agent_id = {sql_literal(self.agent_id)}"
        )
        if not tier:
            raise ToolError(f"unknown agent {self.agent_id!r}; set AGENT_MEMORY_AGENT")
        completeness = "partial" if self.own_session or tier[0][0] != "A" else "full"
        client.insert(
            {
                "sessions": [
                    {
                        "session_id": self.session_id,
                        "agent_id": self.agent_id,
                        "harness_session_id": self.harness_session_id,
                        "scope": "global",
                        "project": self.project,
                        "cwd": os.getcwd(),
                        "capture_completeness": completeness,
                        "started_at": datetime.now(UTC),
                        "metadata": {"via": "mem-mcp", "own_session": self.own_session},
                    }
                ]
            }
        )

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
        return {
            "event_id": client.new_ids()[0],
            "session_id": self.session_id,
            "seq": 0,
            "kind": kind,
            "role": "memory",
            "content": content,
            "content_hash": blobs.content_hash(content) if content else None,
            "payload": payload,
            "ts": datetime.now(UTC),
        }

    def _scopes(self, scopes: Sequence[str] | None) -> list[str]:
        if scopes:
            return [s for s in scopes if s]
        return ["global", self.project] if self.project else ["global"]

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
        normalized = ranking.normalize_query(query or "")
        strategy = ranking.FTS_STRATEGY if normalized else ranking.RECENT_STRATEGY
        try:
            client = self._client_or_raise()
            started = time.perf_counter()
            sql = (
                ranking.fts_sql(normalized, scopes, types)
                if normalized
                else ranking.recent_sql(scopes, types, limit)
            )
            _, rows = client.query(sql)
            latency_ms = (time.perf_counter() - started) * 1000
            items = _rank(rows, strategy, limit)
            event = self._event(
                client,
                "memory_recall",
                query,
                {"scopes": scopes, "types": list(types) if types else None, "limit": limit},
            )
            retrieval_id = client.new_ids()[0]
            client.insert(
                {
                    "events": [event],
                    "retrievals": [
                        {
                            "retrieval_id": retrieval_id,
                            "session_id": self.session_id,
                            "request_event_id": event["event_id"],
                            "query_text": query,
                            "scopes": scopes,
                            "strategy": strategy,
                            "sql_used": sql,
                            "params": {"normalized": normalized, "types": types, "limit": limit},
                            "candidates": len(rows),
                            "returned": len(items),
                            "token_estimate": sum(len(i["content"]) // 4 for i in items),
                            "latency_ms": latency_ms,
                        }
                    ],
                    "retrieval_items": [
                        {
                            "retrieval_id": retrieval_id,
                            "rank": rank,
                            "memory_id": item["memory_id"],
                            "score": item["score"],
                            "score_components": item["score_components"],
                            "injected": True,
                            "status_at_recall": item["status"],
                        }
                        for rank, item in enumerate(items, start=1)
                    ],
                }
            )
        except (AgentMemoryError, ConfigError, duckdb.Error) as exc:
            self._reset()
            return {
                "degraded": True,
                "reason": f"{type(exc).__name__}: {exc}",
                "retrieval_id": None,
                "items": [],
            }
        return {
            "degraded": False,
            "retrieval_id": retrieval_id,
            "request_event_id": event["event_id"],
            "strategy": strategy,
            "candidates": len(rows),
            "items": [{k: v for k, v in item.items() if k not in ("status",)} for item in items],
            "hint": "Pass retrieval_id (and the memory_ids you relied on) to cite when you answer.",
        }

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
        scope = scope or self.project or "global"
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
            "metadata": None,
        }
        return self._write_memory(
            spec,
            sources,
            dedupe_sql=(
                "SELECT memory_id::VARCHAR FROM active_memories "
                f"WHERE scope = {sql_literal(scope)} "
                f"AND content_hash = {sql_literal(blobs.content_hash(redacted))} "
                "ORDER BY created_at DESC LIMIT 1"
            ),
            confirm_reason="remember() with identical content",
        )

    def _write_memory(
        self,
        spec: dict[str, Any],
        sources: Sequence[str],
        *,
        dedupe_sql: str,
        confirm_reason: str,
    ) -> dict[str, Any]:
        """Store one memory from a spec the spool can replay, or confirm the active memory
        dedupe_sql finds. When the server is down the spec is queued instead."""
        content = spec["content"]
        try:
            client = self._client_or_raise()
            if spec["type"] not in self._vocabulary(client, "memory_types", "type"):
                raise ToolError(
                    "type must be one of "
                    + ", ".join(sorted(self._vocabulary(client, "memory_types", "type")))
                )
            _, dup = client.query(dedupe_sql)
            if dup:
                existing = str(dup[0][0])
                event = self._event(
                    client, "memory_write", content, {"deduplicated": existing, **spec}
                )
                client.insert(
                    {
                        "events": [event],
                        "memory_actions": [
                            {
                                "action_id": client.new_ids()[0],
                                "memory_id": existing,
                                "action": "confirm",
                                "actor": self.agent_id,
                                "session_id": self.session_id,
                                "reason": confirm_reason,
                            }
                        ],
                    }
                )
                return {
                    "memory_id": existing,
                    "deduplicated": True,
                    "queued": False,
                    "status": self._status_of(client, existing),
                }
            known_sources = self._existing_events(client, sources)
            event = self._event(client, "memory_write", content, spec)
            memory_id = client.new_ids()[0]
            relation = "user_stated" if spec["trust_level"] == "user" else "derived_from"
            client.insert(
                {
                    "events": [event],
                    "memories": [
                        {
                            "memory_id": memory_id,
                            "scope": spec["scope"],
                            "type": spec["type"],
                            "key": spec["key"],
                            "content": content,
                            "content_hash": blobs.content_hash(content),
                            "confidence": spec["confidence"],
                            "importance": spec["importance"],
                            "trust_level": spec["trust_level"],
                            "supersedes": spec["supersedes"],
                            "valid_from": _ts(spec["valid_from"]),
                            "expires_at": _ts(spec["expires_at"]),
                            "created_by_session": self.session_id,
                            "metadata": spec["metadata"],
                        }
                    ],
                    "memory_sources": [
                        {
                            "memory_id": memory_id,
                            "event_id": event["event_id"],
                            "relation": relation,
                        },
                        *(
                            {"memory_id": memory_id, "event_id": e, "relation": "derived_from"}
                            for e in known_sources
                        ),
                    ],
                }
            )
            return {
                "memory_id": memory_id,
                "deduplicated": False,
                "queued": False,
                "status": self._status_of(client, memory_id),
                "ignored_source_event_ids": sorted(set(sources) - set(known_sources)),
            }
        except (AgentMemoryError, ConfigError, duckdb.Error) as exc:
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
            if answer_event_id is None:
                placeholder = self._event(
                    client,
                    "assistant_msg",
                    None,
                    {"placeholder": True, "retrieval_id": retrieval_id},
                )
                placeholder["role"] = "assistant"
                answer_event_id = placeholder["event_id"]
                rows["events"] = [placeholder]
            else:
                answer_event_id = _uuid(answer_event_id, "answer_event_id")
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
        except (AgentMemoryError, ConfigError, duckdb.Error) as exc:
            self._reset()
            return {"degraded": True, "reason": f"{type(exc).__name__}: {exc}", "links": 0}
        return {"degraded": False, "answer_event_id": answer_event_id, "links": len(links)}

    def review(
        self,
        action: str,
        memory_id: str,
        reason: str,
        related_memory_id: str | None = None,
    ) -> dict[str, Any]:
        memory_id = _uuid(memory_id, "memory_id")
        if related_memory_id is not None:
            related_memory_id = _uuid(related_memory_id, "related_memory_id")
        if not (reason or "").strip():
            raise ToolError("reason is required")
        spec = {
            "memory_id": memory_id,
            "action": action,
            "actor": self.agent_id,
            "related_memory_id": related_memory_id,
            "reason": reason,
        }
        try:
            client = self._client_or_raise()
            allowed = self._vocabulary(client, "memory_action_types", "action")
            if action not in allowed:
                raise ToolError("action must be one of " + ", ".join(sorted(allowed)))
            event = self._event(client, "memory_write", reason, {"review": spec})
            client.insert(
                {
                    "events": [event],
                    "memory_actions": [
                        {"action_id": client.new_ids()[0], "session_id": self.session_id, **spec}
                    ],
                }
            )
            return {
                "memory_id": memory_id,
                "action": action,
                "queued": False,
                "status": self._status_of(client, memory_id),
            }
        except (AgentMemoryError, ConfigError, duckdb.Error) as exc:
            self._reset()
            return self._queue(
                "memory_write", reason, action=spec, reason=f"{type(exc).__name__}: {exc}"
            )

    def status(self) -> dict[str, Any]:
        spool = Spool(self.cfg.spool_dir).status()
        info: dict[str, Any] = {
            "agent_id": self.agent_id,
            "session_id": self.session_id,
            "harness_session_id": self.harness_session_id,
            "project": self.project,
            "server": self.cfg.uri,
            "spool_pending_lines": sum(s.lines for s in spool),
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
    ) -> dict[str, Any]:
        redact = self.redactor()
        try:
            relation = Relation(
                redact.text(subject or "")[0],
                redact.text(predicate or "")[0],
                redact.text(object or "")[0],
                subject_type,
                object_type,
            )
        except ValueError as exc:
            raise ToolError(str(exc)) from None
        if trust_level not in TRUST_LEVELS:
            raise ToolError(f"trust_level must be one of {', '.join(TRUST_LEVELS)}")
        if not 0 <= confidence <= 1 or not 0 <= importance <= 1:
            raise ToolError("confidence and importance must be between 0 and 1")
        sources = [_uuid(e, "source_event_ids") for e in (source_event_ids or [])]
        scope = scope or self.project or "global"
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
            "metadata": relation.metadata(),
        }
        result = self._write_memory(
            spec,
            sources,
            dedupe_sql=(
                "SELECT memory_id::VARCHAR FROM active_memories "
                f"WHERE scope = {sql_literal(scope)} AND key = {sql_literal(relation.key)} "
                "ORDER BY created_at DESC LIMIT 1"
            ),
            confirm_reason="relate() with the same subject, predicate, and object",
        )
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
        entity_norm = normalize_entity(entity or "")
        if not entity_norm:
            raise ToolError("entity is required")
        hops = max(1, min(int(hops), MAX_HOPS))
        limit = max(1, min(int(limit), MAX_GRAPH_LIMIT))
        scopes = self._scopes(scopes)
        try:
            client = self._client_or_raise()
            started = time.perf_counter()
            sql = hop_sql(entity_norm, hops, scopes, types)
            _, rows = client.query(sql)
            latency_ms = (time.perf_counter() - started) * 1000
            items = _graph_items(rows)[:limit]
            event = self._event(
                client,
                "memory_recall",
                entity,
                {
                    "graph": {"entity_norm": entity_norm, "hops": hops},
                    "scopes": scopes,
                    "types": list(types) if types else None,
                    "limit": limit,
                },
            )
            retrieval_id = client.new_ids()[0]
            client.insert(
                {
                    "events": [event],
                    "retrievals": [
                        {
                            "retrieval_id": retrieval_id,
                            "session_id": self.session_id,
                            "request_event_id": event["event_id"],
                            "query_text": entity,
                            "scopes": scopes,
                            "strategy": GRAPH_STRATEGY,
                            "sql_used": sql,
                            "params": {
                                "entity_norm": entity_norm,
                                "hops": hops,
                                "types": types,
                                "limit": limit,
                            },
                            "candidates": len(rows),
                            "returned": len(items),
                            "token_estimate": sum(len(i["content"]) // 4 for i in items),
                            "latency_ms": latency_ms,
                        }
                    ],
                    "retrieval_items": [
                        {
                            "retrieval_id": retrieval_id,
                            "rank": rank,
                            "memory_id": item["memory_id"],
                            "score": item["score"],
                            "score_components": item["score_components"],
                            "injected": True,
                            "status_at_recall": item["status"],
                        }
                        for rank, item in enumerate(items, start=1)
                    ],
                }
            )
        except (AgentMemoryError, ConfigError, duckdb.Error) as exc:
            self._reset()
            return {
                "degraded": True,
                "reason": f"{type(exc).__name__}: {exc}",
                "retrieval_id": None,
                "items": [],
                "triples": [],
            }
        names: dict[str, str] = {}
        for item in items:
            names.setdefault(normalize_entity(item["subject"]), item["subject"])
            names.setdefault(normalize_entity(item["object"]), item["object"])
        return {
            "degraded": False,
            "retrieval_id": retrieval_id,
            "request_event_id": event["event_id"],
            "strategy": GRAPH_STRATEGY,
            "entity": entity,
            "entity_norm": entity_norm,
            "hops": hops,
            "candidates": len(rows),
            "items": [{k: v for k, v in item.items() if k != "status"} for item in items],
            "triples": [item["triple"] for item in items],
            "entities": sorted(names.values(), key=str.casefold),
            "hint": "Answer from these edges and pass retrieval_id (and the memory_ids you "
            "relied on) to cite.",
        }

    def merge_entity(
        self,
        alias: str,
        canonical: str,
        reason: str,
        scopes: Sequence[str] | None = None,
        canonical_type: str | None = None,
        actor: str | None = None,
    ) -> dict[str, Any]:
        """Entity resolution as append-only history: every active relation that mentions
        `alias` is superseded by the same relation with `canonical` in its place, or retired
        when that relation already exists. Nothing is rewritten; the alias stays in the
        new rows' metadata and the old rows stay readable."""
        alias_norm = normalize_entity(alias or "")
        canonical_norm = normalize_entity(canonical or "")
        if not alias_norm or not canonical_norm:
            raise ToolError("alias and canonical are required")
        if alias_norm == canonical_norm:
            raise ToolError("alias and canonical normalize to the same entity")
        if not (reason or "").strip():
            raise ToolError("reason is required")
        scopes = self._scopes(scopes)
        actor = actor or self.agent_id
        try:
            client = self._client_or_raise()
            _, rows = client.query(relations_sql(scopes, [alias_norm]))
            old: list[tuple[dict[str, Any], Relation, Relation]] = []
            for row in rows:
                record = dict(zip(GRAPH_ROW, row, strict=True))
                relation = Relation.from_metadata(json.loads(record["metadata"]))
                if relation is not None:
                    old.append(
                        (record, relation, relation.renamed(alias_norm, canonical, canonical_type))
                    )
            summary: dict[str, Any] = {
                "degraded": False,
                "alias": alias,
                "canonical": canonical,
                "superseded": [],
                "retired": [],
            }
            if not old:
                return summary
            _, existing_rows = client.query(
                active_keys_sql(scopes, sorted({new.key for _, _, new in old}))
            )
            existing = {(str(k), str(s)): str(mid) for k, s, mid in existing_rows}
            _, source_rows = client.query(sources_sql([rec["memory_id"] for rec, _, _ in old]))
            sources: dict[str, list[tuple[str, str]]] = defaultdict(list)
            for mid, eid, rel in source_rows:
                sources[str(mid)].append((str(eid), str(rel)))
            event = self._event(
                client,
                "memory_write",
                f"merge entity {alias} into {canonical}",
                {
                    "merge": {
                        "alias": alias,
                        "alias_norm": alias_norm,
                        "canonical": canonical,
                        "canonical_norm": canonical_norm,
                        "reason": reason,
                        "actor": actor,
                        "scopes": scopes,
                    }
                },
            )
            memories: list[dict[str, Any]] = []
            memory_sources: list[dict[str, Any]] = []
            actions: list[dict[str, Any]] = []
            created_here: dict[tuple[str, str], str] = {}
            for record, relation, renamed in old:
                mid = record["memory_id"]
                slot = (renamed.key, record["scope"])
                target = existing.get(slot) or created_here.get(slot)
                if target and target != mid:
                    actions.append(
                        {
                            "action_id": client.new_ids()[0],
                            "memory_id": mid,
                            "action": "retire",
                            "actor": actor,
                            "session_id": self.session_id,
                            "related_memory_id": target,
                            "reason": f"merged {alias} into {canonical}; same relation as "
                            f"{target}: {reason}",
                        }
                    )
                    summary["retired"].append({"memory_id": mid, "into": target})
                    continue
                new_id = client.new_ids()[0]
                created_here[slot] = new_id
                content = _rename_in_text(record["content"], relation, renamed)
                memories.append(
                    {
                        "memory_id": new_id,
                        "scope": record["scope"],
                        "type": record["type"],
                        "key": renamed.key,
                        "content": content,
                        "content_hash": blobs.content_hash(content),
                        "confidence": float(record["confidence"]),
                        "importance": float(record["importance"]),
                        "trust_level": record["trust_level"],
                        "supersedes": mid,
                        "valid_from": _ts(record["valid_from"]),
                        "expires_at": _ts(record["expires_at"]),
                        "created_by_session": self.session_id,
                        "metadata": renamed.metadata(),
                    }
                )
                memory_sources.append(
                    {"memory_id": new_id, "event_id": event["event_id"], "relation": "derived_from"}
                )
                memory_sources.extend(
                    {"memory_id": new_id, "event_id": eid, "relation": rel}
                    for eid, rel in sources[mid]
                    if eid != event["event_id"]
                )
                # A superseding row has no review history. Without this, an approved
                # tool-derived relation would fall back to quarantine after the merge.
                if record["status"] == "active" and record["trust_level"] == "tool":
                    actions.append(
                        {
                            "action_id": client.new_ids()[0],
                            "memory_id": new_id,
                            "action": "approve",
                            "actor": actor,
                            "session_id": self.session_id,
                            "related_memory_id": mid,
                            "reason": f"approval carried over from {mid} at merge",
                        }
                    )
                summary["superseded"].append({"memory_id": mid, "by": new_id, "key": renamed.key})
            client.insert(
                [
                    ("events", [event]),
                    ("memories", memories),
                    ("memory_sources", memory_sources),
                    ("memory_actions", actions),
                ]
            )
            summary["event_id"] = event["event_id"]
            return summary
        except (AgentMemoryError, ConfigError, duckdb.Error) as exc:
            self._reset()
            return {
                "degraded": True,
                "reason": f"{type(exc).__name__}: {exc}",
                "alias": alias,
                "canonical": canonical,
                "superseded": [],
                "retired": [],
            }
    # Helpers

    def _status_of(self, client: MemoryClient, memory_id: str) -> str | None:
        _, rows = client.query(
            f"SELECT status FROM memory_status WHERE memory_id = {sql_literal(memory_id)}"
        )
        return str(rows[0][0]) if rows else None

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
        record = Record(
            agent_id=self.agent_id,
            harness_session_id=self.harness_session_id,
            kind=kind,
            ts=datetime.now(UTC),
            source_key=source_key,
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


def _rank(rows: Sequence[tuple], strategy: str, limit: int) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for row in rows:
        (
            memory_id,
            mtype,
            key,
            content,
            scope,
            confidence,
            importance,
            trust,
            created,
            expires,
            status,
            bm25,
        ) = row
        if strategy == ranking.FTS_STRATEGY:
            components = ranking.score(
                float(bm25 or 0), float(importance), float(confidence), scope
            )
            total = components.pop("score")
        else:
            components = {"importance": float(importance)}
            total = float(importance)
        items.append(
            {
                "memory_id": str(memory_id),
                "type": mtype,
                "key": key,
                "content": content,
                "scope": scope,
                "confidence": float(confidence),
                "importance": float(importance),
                "trust_level": trust,
                "created_at": created,
                "expires_at": expires,
                "score": total,
                "score_components": components,
                "status": status,
            }
        )
    items.sort(key=lambda i: (-float(i["score"]), str(i["created_at"] or "")))
    return items[:limit]


def _graph_items(rows: Sequence[tuple]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for row in rows:
        record = dict(zip(GRAPH_ROW, row, strict=True))
        relation = Relation.from_metadata(json.loads(record["metadata"]))
        if relation is None:
            continue
        depth = int(record["depth"])
        items.append(
            {
                "memory_id": str(record["memory_id"]),
                "subject": relation.subject,
                "predicate": relation.predicate,
                "object": relation.object,
                "subject_type": relation.subject_type,
                "object_type": relation.object_type,
                "aliases": list(relation.aliases),
                "triple": relation.triple,
                "key": record["key"],
                "content": record["content"],
                "scope": record["scope"],
                "confidence": float(record["confidence"]),
                "importance": float(record["importance"]),
                "trust_level": record["trust_level"],
                "created_at": record["created_at"],
                "depth": depth,
                "score": 1.0 / (1 + depth),
                "score_components": {"depth": float(depth)},
                "status": record["status"],
            }
        )
    return items


def _rename_in_text(content: str, old: Relation, new: Relation) -> str:
    """Carry a merge into the memory's prose when the old name appears in it; otherwise
    the new relation's own sentence is the content."""
    text = content
    changed = False
    for before, after in ((old.subject, new.subject), (old.object, new.object)):
        if before == after:
            continue
        pattern = re.compile(re.escape(before), re.IGNORECASE)
        if pattern.search(text):
            text = pattern.sub(after, text)
            changed = True
    return text if changed else new.sentence


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


def _cwd_name() -> str | None:
    try:
        return os.path.basename(os.getcwd()) or None
    except OSError:
        return None
