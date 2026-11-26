"""The two recall strategies that answer a query, with their SQL and scoring. Every string
the agent supplies goes through sql_literal; the server's policy strips literals before
classifying the statement, so query text cannot change what the statement does."""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ..db import sql_literal

FTS_STRATEGY = "fts_bm25_v2"
# What a recall by words was called while the index dropped every digit, and still is where
# the database served has not had the migration that builds the index again.
FTS_LETTERS_STRATEGY = "fts_bm25_v1"
DIGITS_MIGRATION = 10
RECENT_STRATEGY = "recent_v1"
SESSION_START_STRATEGY = "session_start_v1"
CANDIDATE_CAP = 50
SESSION_START_DEFAULT_MAX_ITEMS = 12
SESSION_START_MIN_ITEMS = 1
SESSION_START_MAX_ITEMS = CANDIDATE_CAP
SESSION_START_DEFAULT_MAX_CHARS = 2400
SESSION_START_MIN_CHARS = 200
SESSION_START_MAX_CHARS = 16000
SESSION_START_HEADER = "# Active memories"

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


def session_start_sql(scopes: Sequence[str], types: Sequence[str] | None) -> str:
    type_filter = f" AND m.type IN ({_in_list(types)})" if types else ""
    return (
        f"SELECT {_COLUMNS}, NULL::DOUBLE AS bm25 FROM active_memories m "
        "JOIN memory_status ms USING (memory_id) "
        f"WHERE m.scope IN ({_in_list(scopes)}){type_filter} "
        "ORDER BY CASE WHEN m.scope = 'global' THEN 1 ELSE 0 END ASC, "
        "m.importance DESC, m.created_at DESC, m.memory_id ASC "
        f"LIMIT {CANDIDATE_CAP}"
    )


def clamp_session_max_items(value: int) -> int:
    return max(SESSION_START_MIN_ITEMS, min(value, SESSION_START_MAX_ITEMS))


def clamp_session_max_chars(value: int) -> int:
    return max(SESSION_START_MIN_CHARS, min(value, SESSION_START_MAX_CHARS))


def format_session_line(item: Mapping[str, Any]) -> str:
    single_line = " ".join(str(item["content"]).split())
    return f"- [{item['memory_id']}] ({item['scope']}/{item['type']}) {single_line}"


def format_session_context(
    items: Sequence[Mapping[str, Any]],
    *,
    max_items: int = SESSION_START_DEFAULT_MAX_ITEMS,
    max_chars: int = SESSION_START_DEFAULT_MAX_CHARS,
) -> tuple[str, list[dict[str, Any]]]:
    """Render a deterministic bounded block of active memories and tag overflow items."""
    clamped_items = clamp_session_max_items(max_items)
    clamped_chars = clamp_session_max_chars(max_chars)
    lines: list[str] = []
    used_chars = len(SESSION_START_HEADER)
    shown_count = 0
    tagged: list[dict[str, Any]] = []
    for item in items:
        line = format_session_line(item)
        added_chars = 1 + len(line)
        fits = shown_count < clamped_items and (used_chars + added_chars) <= clamped_chars
        if fits:
            lines.append(line)
            used_chars += added_chars
            shown_count += 1
        tagged.append({**item, "injected": fits})
    if not lines:
        return "", tagged
    return "\n".join([SESSION_START_HEADER, *lines]), tagged


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
    index_keeps_digits: bool = True

    @property
    def name(self) -> str:
        return FTS_STRATEGY if self.index_keeps_digits else FTS_LETTERS_STRATEGY

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


@dataclass(frozen=True)
class SessionStart:
    """Deterministic active-only session-start recall bounded by item count and char budget."""

    scopes: Sequence[str]
    types: Sequence[str] | None = None
    max_items: int = SESSION_START_DEFAULT_MAX_ITEMS
    max_chars: int = SESSION_START_DEFAULT_MAX_CHARS

    @property
    def name(self) -> str:
        return SESSION_START_STRATEGY

    @property
    def query_text(self) -> None:
        return None

    @property
    def sql(self) -> str:
        return session_start_sql(self.scopes, self.types)

    @property
    def params(self) -> dict[str, Any]:
        return {
            "types": list(self.types) if self.types else None,
            "max_items": clamp_session_max_items(self.max_items),
            "max_chars": clamp_session_max_chars(self.max_chars),
        }

    @property
    def request(self) -> dict[str, Any]:
        return {
            "scopes": list(self.scopes),
            "types": list(self.types) if self.types else None,
            "max_items": clamp_session_max_items(self.max_items),
            "max_chars": clamp_session_max_chars(self.max_chars),
        }

    def items(self, rows: Sequence[tuple]) -> list[dict[str, Any]]:
        candidates = [_row_to_item(row, _session_score) for row in rows]
        _, tagged = format_session_context(
            candidates, max_items=self.max_items, max_chars=self.max_chars
        )
        return tagged


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


def _session_score(
    bm25: float | None, importance: float, confidence: float, scope: str
) -> tuple[float, dict[str, float]]:
    return importance, {
        "importance": importance,
        "project_scope": 1.0 if scope != "global" else 0.0,
    }


def _row_to_item(
    row: tuple,
    scored: Callable[[float | None, float, float, str], tuple[float, dict[str, float]]],
) -> dict[str, Any]:
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
    total, components = scored(bm25, float(importance), float(confidence), scope)
    return {
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


def _items(
    rows: Sequence[tuple],
    limit: int,
    scored: Callable[[float | None, float, float, str], tuple[float, dict[str, float]]],
) -> list[dict[str, Any]]:
    """Rows in the order of _COLUMNS plus bm25, as items ranked by score."""
    items = [_row_to_item(row, scored) for row in rows]
    # Equal scores go newest first, the order the candidate SQL already uses.
    items.sort(key=lambda i: str(i["created_at"] or ""), reverse=True)
    items.sort(key=lambda i: -float(i["score"]))
    return items[:limit]
