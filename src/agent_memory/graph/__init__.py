"""Knowledge-graph layer over the memory ledger.

A relation (subject, predicate, object) is stored as an ordinary `fact` memory, so it gets
provenance, status derivation, and `why_answer` for free. The graph is reconstructed from
`metadata.graph` at query time with a recursive CTE; there are no graph tables.
"""

from .merge import EntityMerge, apply_merge
from .model import GRAPH_VERSION, Relation, normalize_entity
from .recall import RelationListing, RelationWalk
from .sql import GRAPH_STRATEGY, MAX_HOPS, hop_sql, relations_sql

__all__ = [
    "GRAPH_STRATEGY",
    "GRAPH_VERSION",
    "MAX_HOPS",
    "EntityMerge",
    "Relation",
    "RelationListing",
    "RelationWalk",
    "apply_merge",
    "hop_sql",
    "normalize_entity",
    "relations_sql",
]
