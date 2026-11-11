"""The graph's recall strategies: relations chosen by their distance from an entity, or
every relation in the scopes."""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any

from .model import Relation
from .sql import GRAPH_STRATEGY, LISTING_STRATEGY, ROW, hop_sql, relations_sql


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
        items = []
        for record, item in _relations(rows):
            depth = int(record["depth"])
            items.append(
                {
                    **item,
                    "depth": depth,
                    "score": 1.0 / (1 + depth),
                    "score_components": {"depth": float(depth)},
                    "status": record["status"],
                }
            )
        return items[: self.limit]


@dataclass(frozen=True)
class RelationListing:
    """Recall of every relation in the scopes, oldest first: what entity resolution and
    `mem graph entities` read."""

    scopes: Sequence[str]

    @property
    def name(self) -> str:
        return LISTING_STRATEGY

    @property
    def query_text(self) -> None:
        return None

    @property
    def sql(self) -> str:
        return relations_sql(self.scopes)

    @property
    def params(self) -> dict[str, Any]:
        return {"relations": "all"}

    @property
    def request(self) -> dict[str, Any]:
        return {"graph": {"relations": "all"}, "scopes": self.scopes}

    def items(self, rows: Sequence[tuple]) -> list[dict[str, Any]]:
        # Nothing is ranked or left out, so no relation has a score.
        return [
            {**item, "score": None, "score_components": None, "status": record["status"]}
            for record, item in _relations(rows)
        ]


def relations_of(items: Sequence[dict[str, Any]]) -> list[Relation]:
    """The relations that the items of a graph recall describe."""
    return [
        Relation(
            item["subject"],
            item["predicate"],
            item["object"],
            item["subject_type"],
            item["object_type"],
            tuple(item["aliases"]),
        )
        for item in items
    ]


def _relations(rows: Sequence[tuple]) -> Iterator[tuple[dict[str, Any], dict[str, Any]]]:
    """Each row of a graph statement (see sql.ROW) that holds a relation, in the order the
    statement gave them: the row as a record, and what every graph item says about it."""
    for row in rows:
        record = dict(zip(ROW, row, strict=True))
        relation = Relation.from_metadata(json.loads(record["metadata"]))
        if relation is None:
            continue
        yield (
            record,
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
            },
        )
