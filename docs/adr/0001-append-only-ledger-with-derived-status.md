# Every ledger table is append-only and status is derived

The ledger exists to answer what an agent was shown and what it used, so
its history must not be rewritable. Every table rejects `UPDATE` and
`DELETE` through triggers, and anything that changes over time, such as a
memory's status, is derived in views (`memory_status`, `active_memories`)
from appended rows: a correction is a superseding memory or a review
action, never an edit. The first spec had a stored `status` column kept
current by triggers. We dropped it because a DuckDB 2.0 trigger cannot
modify its own table, and because an in-place update would have left
`events` as the only table whose history could be trusted.

## Consequences

- There is no `audit_log` table. The tables are the audit log, and
  `memory_history` is the chronological view.
- Content cannot be removed after the fact without an admin migration
  that drops the guards, so redaction has to happen before the write.
- Status is computed on every read. Because nothing is overwritten,
  `why_answer` can report a memory's status at recall and its status now.
