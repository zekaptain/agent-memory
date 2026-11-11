"""The two recall strategies that answer a query, with their SQL and scoring. Every string
the agent supplies goes through sql_literal; the server's policy strips literals before
classifying the statement, so query text cannot change what the statement does."""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from ..db import sql_literal

FTS_STRATEGY = "fts_bm25_v1"
RECENT_STRATEGY = "recent_v1"
CANDIDATE_CAP = 50

_COLUMNS = (
    "m.memory_id::VARCHAR, m.type, m.key, m.content, m.scope, m.confidence, m.importance, "
    "m.trust_level, m.created_at::VARCHAR, m.expires_at::VARCHAR, ms.status"
)
_NON_WORD = re.compile(r"[^\w\s]+", re.UNICODE)


def normalize_query(query: str) -> str:
    """Lowercase, drop punctuation (the FTS tokenizer keeps curly quotes attached to words)."""
    return " ".join(_NON_WORD.sub(" ", query.lower()).split())


def _in_list(values: Sequence[str]) -> str:
    return ", ".join(sql_literal(v) for v in values)


def fts_sql(normalized: str, scopes: Sequence[str], types: Sequence[str] | None) -> str:
    type_filter = f" AND m.type IN ({_in_list(types)})" if types else ""
    return (
        "WITH hits AS (SELECT memory_id, fts_main_memories.match_bm25(memory_id, "
        f"{sql_literal(normalized)}) AS bm25 FROM memories) "
        f"SELECT {_COLUMNS}, h.bm25 FROM active_memories m "
        "JOIN hits h USING (memory_id) JOIN memory_status ms USING (memory_id) "
        f"WHERE h.bm25 IS NOT NULL AND m.scope IN ({_in_list(scopes)}){type_filter} "
        f"ORDER BY h.bm25 DESC, m.created_at DESC LIMIT {CANDIDATE_CAP}"
    )


def recent_sql(scopes: Sequence[str], types: Sequence[str] | None, limit: int) -> str:
    type_filter = f" AND m.type IN ({_in_list(types)})" if types else ""
    return (
        f"SELECT {_COLUMNS}, NULL::DOUBLE AS bm25 FROM active_memories m "
        "JOIN memory_status ms USING (memory_id) "
        f"WHERE m.scope IN ({_in_list(scopes)}){type_filter} "
        f"ORDER BY m.importance DESC, m.created_at DESC LIMIT {limit}"
    )


def score(bm25: float, importance: float, confidence: float, scope: str) -> dict[str, float]:
    scope_boost = 1.0 if scope == "global" else 1.1
    return {
        "bm25": bm25,
        "importance": 0.6 + 0.4 * importance,
        "confidence": 0.5 + 0.5 * confidence,
        "scope_boost": scope_boost,
        "score": bm25 * (0.6 + 0.4 * importance) * (0.5 + 0.5 * confidence) * scope_boost,
    }


@dataclass(frozen=True)
class TextMatch:
    """Recall by the query's words: bm25 weighted by importance, confidence and scope."""

    query_text: str
    normalized: str
    scopes: Sequence[str]
    types: Sequence[str] | None
    limit: int

    @property
    def name(self) -> str:
        return FTS_STRATEGY

    @property
    def sql(self) -> str:
        return fts_sql(self.normalized, self.scopes, self.types)

    @property
    def params(self) -> dict[str, Any]:
        return {"normalized": self.normalized, "types": self.types, "limit": self.limit}

    @property
    def request(self) -> dict[str, Any]:
        return _request(self.scopes, self.types, self.limit)

    def items(self, rows: Sequence[tuple]) -> list[dict[str, Any]]:
        return _items(rows, self.limit, _weighted)


@dataclass(frozen=True)
class Recent:
    """Recall without words to match: the most important memories, newest first."""

    query_text: str
    scopes: Sequence[str]
    types: Sequence[str] | None
    limit: int

    @property
    def name(self) -> str:
        return RECENT_STRATEGY

    @property
    def sql(self) -> str:
        return recent_sql(self.scopes, self.types, self.limit)

    @property
    def params(self) -> dict[str, Any]:
        return {"normalized": "", "types": self.types, "limit": self.limit}

    @property
    def request(self) -> dict[str, Any]:
        return _request(self.scopes, self.types, self.limit)

    def items(self, rows: Sequence[tuple]) -> list[dict[str, Any]]:
        return _items(rows, self.limit, _by_importance)


def for_query(
    query: str, scopes: Sequence[str], types: Sequence[str] | None, limit: int
) -> TextMatch | Recent:
    """The strategy for a query: its words when it has any, otherwise what is recent."""
    normalized = normalize_query(query)
    if normalized:
        return TextMatch(query, normalized, scopes, types, limit)
    return Recent(query, scopes, types, limit)


def _request(scopes: Sequence[str], types: Sequence[str] | None, limit: int) -> dict[str, Any]:
    return {"scopes": scopes, "types": list(types) if types else None, "limit": limit}


def _weighted(
    bm25: float | None, importance: float, confidence: float, scope: str
) -> tuple[float, dict[str, float]]:
    components = score(float(bm25 or 0), importance, confidence, scope)
    return components.pop("score"), components


def _by_importance(
    bm25: float | None, importance: float, confidence: float, scope: str
) -> tuple[float, dict[str, float]]:
    return importance, {"importance": importance}


def _items(
    rows: Sequence[tuple],
    limit: int,
    scored: Callable[[float | None, float, float, str], tuple[float, dict[str, float]]],
) -> list[dict[str, Any]]:
    """Rows in the order of _COLUMNS plus bm25, as items ranked by score."""
    items: list[dict[str, Any]] = []
    for (
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
    ) in rows:
        total, components = scored(bm25, float(importance), float(confidence), scope)
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
    # Equal scores go newest first, the order the candidate SQL already uses.
    items.sort(key=lambda i: str(i["created_at"] or ""), reverse=True)
    items.sort(key=lambda i: -float(i["score"]))
    return items[:limit]
