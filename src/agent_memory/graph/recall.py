"""The graph's recall strategy: relations chosen by their distance from an entity."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .model import Relation
from .sql import GRAPH_STRATEGY, ROW, hop_sql


@dataclass(frozen=True)
class RelationWalk:
    """Recall by walking relations outward from an entity; nearer relations rank higher."""

    query_text: str
    entity_norm: str
    hops: int
    scopes: Sequence[str]
    types: Sequence[str] | None
    limit: int

    @property
    def name(self) -> str:
        return GRAPH_STRATEGY

    @property
    def sql(self) -> str:
        return hop_sql(self.entity_norm, self.hops, self.scopes, self.types)

    @property
    def params(self) -> dict[str, Any]:
        return {
            "entity_norm": self.entity_norm,
            "hops": self.hops,
            "types": self.types,
            "limit": self.limit,
        }

    @property
    def request(self) -> dict[str, Any]:
        return {
            "graph": {"entity_norm": self.entity_norm, "hops": self.hops},
            "scopes": self.scopes,
            "types": list(self.types) if self.types else None,
            "limit": self.limit,
        }

    def items(self, rows: Sequence[tuple]) -> list[dict[str, Any]]:
        return relation_items(rows)[: self.limit]


def relation_items(rows: Sequence[tuple]) -> list[dict[str, Any]]:
    """Rows of a graph statement (see sql.ROW) as items, in the order the statement gave
    them. A row whose metadata holds no relation is skipped."""
    items: list[dict[str, Any]] = []
    for row in rows:
        record = dict(zip(ROW, row, strict=True))
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
