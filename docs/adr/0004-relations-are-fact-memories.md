# Relations are fact memories, not graph tables

The relation graph needed storage, and dedicated `entities` and
`relations` tables were the obvious shape. Instead a relation is an
ordinary `fact` memory whose key is the normalized triple
`subject|predicate|object`, with display names, entity types, and merged
aliases in `metadata.graph`. An entity merge supersedes or retires
relations and never rewrites a row. This gives the graph the same
provenance, status derivation, and `why_answer` coverage as every other
memory without a schema change. The cost is that a walk is a recursive
CTE over `active_memories` and that an entity is resolved from the names
relations use.

## Consequences

- An entity exists only as a name in relations. There is no entity row
  to attach attributes to.
- Dedicated tables are deferred until use shows that multi-hop recall
  earns them. See [roadmap.md](../roadmap.md#phases).
