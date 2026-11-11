# Every recall goes through one recorded path

The ledger must be able to say what an agent was shown, and it can only
do so if no read of memories on an agent's behalf skips the record. All
such reads therefore run through `run_recall`, which writes the
`memory_recall` event, the retrieval, and one item per memory returned,
misses included. The ways of choosing memories (a text match, the most
recent, a walk of relations, the listing of every relation) are
strategies passed to it. This includes `mem graph entities` and `mem
graph resolve` at the CLI, which read relations to show them to a person
or a model. A new way of choosing memories is added as a strategy, never
as a query beside the path.

## Consequences

- A CLI command run by a person opens a ledger session for the `human`
  agent and records a retrieval. `mem audit check` expects no citation
  for a recall made in such a session.
- Reads that are not made on an agent's behalf stay unrecorded by
  design: `mem sql`, `mem export`, and `mem audit check`.
