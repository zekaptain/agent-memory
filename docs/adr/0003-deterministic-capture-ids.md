# Capture ids are deterministic and the writer reads before it inserts

The same event reaches the ledger more than once: Droid hooks and
transcript import overlap, hooks are retried, transcripts are reimported,
and the spool is replayed. The id of every captured row is therefore a
`uuid5` over the agent, the harness session, the kind, and a source key,
so a second delivery of the same event produces the same id. `INSERT OR
IGNORE` and `ON CONFLICT` are not available, because DuckDB runs them as
`MERGE`, which fails on tables with triggers, and an anti-join through
the attached Quack catalog fails too. So the writer looks up the incoming
ids in batches, drops the ones the ledger already holds, and inserts the
rest. A lost race is a duplicate-key error, which triggers one re-read
and retry.

## Consequences

- The key rules in `capture/identity.py` are part of the stored data.
  Changing how a source key is built makes every later reimport a
  duplicate of history, so such a change needs a plan for the rows
  already written.
- An event whose source gives it no stable key falls back to the wall
  clock and is not idempotent.
