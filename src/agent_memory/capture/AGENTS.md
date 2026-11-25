# Working on capture

This guide applies to changes under `capture/` and to `mem event`,
`mem records`, `mem ingest`, and the spool in `cli.py`, on top of the
root `AGENTS.md` and `src/agent_memory/AGENTS.md`. What a hook may
record, and `mem hooks`, are in `src/agent_memory/adapters/AGENTS.md`.
The spec is `docs/phase-1-spec.md`.

- Capture IDs are deterministic so transcript reimports, overlapping hooks,
  retries, and spool replay do not duplicate events. A server outage should
  queue redacted records rather than break the agent's hook. Delivery never
  drops a record: one the ledger rejects is kept under `spool/failed/` with
  a `capture_error` event while the rest of its batch is written. `mem
  event` runs as a hook, so it prints nothing on stdout and never exits 2.
  Add regression coverage for replay, partial failures, and parent/result
  linkage.
- `mem records` is kept running by a harness extension, which waits for
  no answer and may read neither output stream. It prints nothing on
  stdout and writes to stderr only when something changes or goes wrong.
  A line it cannot take must not stop the lines after it, and a record
  must not reach the ledger ahead of one read before it: an event is
  not rewritten when its parent arrives.
- A hook has two seconds, and binding runs after every write. What runs
  at every write must not take longer as the ledger grows: look events
  up by id, and do not filter the ledger on `payload`, which no index
  covers.
