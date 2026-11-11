"""SQL builders for the graph. Entity names reach SQL only through sql_literal; the server's
policy strips literals before classifying a statement."""

from __future__ import annotations

from collections.abc import Sequence

from ..db import sql_literal

GRAPH_STRATEGY = "graph_hop_v1"
MAX_HOPS = 3
EDGE_CAP = 200

# Row shape every graph statement returns, in order. The first eleven match
# ranking._COLUMNS; metadata is the JSON the relation is rebuilt from.
# Read rows as dict(zip(ROW, row)).
ROW = (
    "memory_id",
    "type",
    "key",
    "content",
    "scope",
    "confidence",
    "importance",
    "trust_level",
    "created_at",
    "expires_at",
    "status",
    "metadata",
    "valid_from",
    "valid_to",
    "depth",
)

_PROJECTION = (
    "m.memory_id::VARCHAR AS memory_id, m.type, m.key, m.content, m.scope, m.confidence, "
    "m.importance, m.trust_level, m.created_at::VARCHAR AS created_at, "
    "m.expires_at::VARCHAR AS expires_at, ms.status, "
    "json_merge_patch(m.metadata::JSON, json_object('graph', "
    "json_merge_patch(json_extract(m.metadata::JSON, '$.graph'), "
    "json_object('aliases', coalesce((SELECT list(DISTINCT a.alias ORDER BY a.alias) "
    "FROM memory_aliases a WHERE a.memory_id = m.memory_id), [])))))::VARCHAR AS metadata, "
    "m.valid_from::VARCHAR AS valid_from, m.valid_to::VARCHAR AS valid_to, "
    "json_extract_string(m.metadata::JSON, '$.graph.subject_norm') AS s_norm, "
    "json_extract_string(m.metadata::JSON, '$.graph.object_norm') AS o_norm"
)
_SELECT = ", ".join(f"r.{c}" for c in ROW[:-1])


def _in_list(values: Sequence[str]) -> str:
    return ", ".join(sql_literal(v) for v in values)


def _relations_cte(scopes: Sequence[str], types: Sequence[str] | None) -> str:
    type_filter = f" AND m.type IN ({_in_list(types)})" if types else ""
    return (
        f"rel AS (SELECT {_PROJECTION} "
        "FROM active_memories m JOIN memory_status ms USING (memory_id) "
        f"WHERE m.scope IN ({_in_list(scopes)}) "
        f"AND json_extract(m.metadata::JSON, '$.graph') IS NOT NULL{type_filter})"
    )


def hop_sql(
    entity_norm: str, hops: int, scopes: Sequence[str], types: Sequence[str] | None = None
) -> str:
    """Every active relation within `hops` of the entity, with the depth of its nearer end.

    The walk is breadth-first over undirected edges; a node reached at several depths
    keeps the smallest. Edges whose nearer endpoint sits at depth `hops` lead out of the
    neighbourhood and are not returned."""
    hops = max(1, min(int(hops), MAX_HOPS))
    return (
        f"WITH RECURSIVE {_relations_cte(scopes, types)}, "
        "walk(node, depth) AS ("
        f"SELECT {sql_literal(entity_norm)}, 0 "
        "UNION "
        "SELECT CASE WHEN r.s_norm = w.node THEN r.o_norm ELSE r.s_norm END, w.depth + 1 "
        "FROM walk w JOIN rel r ON (r.s_norm = w.node OR r.o_norm = w.node) "
        f"WHERE w.depth < {hops}), "
        "nodes AS (SELECT node, min(depth) AS depth FROM walk GROUP BY node) "
        f"SELECT {_SELECT}, least(ns.depth, no.depth) AS depth "
        "FROM rel r JOIN nodes ns ON ns.node = r.s_norm JOIN nodes no ON no.node = r.o_norm "
        f"WHERE least(ns.depth, no.depth) < {hops} "
        f"ORDER BY depth, r.created_at DESC LIMIT {EDGE_CAP}"
    )


def relations_sql(
    scopes: Sequence[str],
    entity_norms: Sequence[str] | None = None,
    types: Sequence[str] | None = None,
) -> str:
    """Active relations in the scopes, optionally only those touching the given entities."""
    where = ""
    if entity_norms:
        where = (
            f" WHERE r.s_norm IN ({_in_list(entity_norms)}) "
            f"OR r.o_norm IN ({_in_list(entity_norms)})"
        )
    return (
        f"WITH {_relations_cte(scopes, types)} "
        f"SELECT {_SELECT}, 0 AS depth FROM rel r{where} ORDER BY r.created_at"
    )


def sources_sql(memory_ids: Sequence[str]) -> str:
    return (
        "SELECT memory_id::VARCHAR, event_id::VARCHAR, relation FROM memory_sources "
        f"WHERE memory_id IN ({_in_list(memory_ids)})"
    )
