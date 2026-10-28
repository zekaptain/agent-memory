"""SQL builders for recall. Every string the agent supplies goes through sql_literal; the
server's policy strips literals before classifying the statement, so query text cannot
change what the statement does."""

from __future__ import annotations

import re
from collections.abc import Sequence

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
