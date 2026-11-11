# Plan history

Text taken from `PLAN.md` when it was split into
[architecture.md](../architecture.md) and [roadmap.md](../roadmap.md) on
2026-11-11. It is kept for the record and is not the current contract.

## Changes from the approved spec

- **Status is derived, not stored.** The spec had triggers on `memories` and
  `memory_actions` updating `memories.status`. DuckDB 2.0 triggers cannot
  modify their own table, and an update would break append-only. Status now
  comes from `memory_status`, and every table is append-only, not only
  `events`.
- **No `audit_log` table.** With every table append-only, the tables are the
  audit log. `memory_history` gives the chronological view.
- **Foreign keys are on.** The spec dropped them because of trigger-driven
  updates. Without those updates they work and catch real bugs (unknown event
  kinds, dangling session IDs).
- **`AFTER` guards, not `BEFORE`.** `BEFORE` triggers cannot see transition
  tables.
- **Staged inserts instead of bound parameters**, because Quack does not
  forward them.
- **`mem sql --direct`** replaces the spec's `--file` flag, with `--write` for
  read-write admin work.
- **`why_answer` reports status at recall and status now**, so an audit can
  tell when a memory was retired after the answer that used it.

## Phases as they were described when done

**Phase 0: storage layer (done).** Migration 0001, the `mem` CLI (`init`,
`migrate`, `serve`, `sql`, `export`, `smoke`), the Quack server with
authorization and lockdown, Parquet export, a launchd template, and tests for
append-only, constraints, status derivation, `why_answer`, the policy, and
end-to-end Quack behavior including concurrent writers.

**Phase 1a: capture path (done).** Spec in `docs/phase-1-spec.md`. One
normalised *capture record* (JSONL) feeds a shared pipeline: redaction, blob
spill over 16 KiB, deterministic ids (`uuid5` over agent, harness session,
kind, and source key), pre-read plus insert in one transaction, or the spool
when the server is down. `mem event --harness droid|vibe` takes one hook
payload on stdin; `mem ingest --transcript` imports a Droid session file
(every entry type mapped; thinking opt-in); `mem ingest --spool` replays, and
`mem serve` drains the spool at startup. A record the ledger rejects, or a
spool line it cannot read, becomes a `capture_error` event and is kept under
`spool/failed/` while the rest of its batch is written. A real 1000-event
transcript imports in about 0.8 s and re-imports in 50 ms. The Vibe
transcript adapter is a stub until a real file exists (Phase 2).

**Phase 1b: in-session tools (done).** `mem-mcp` (stdio, `mcp>=2`) with
`recall`, `remember`, `cite`, `review`, and `status` (`relate` and `graph`
came later with the relation graph). Every call writes a
`memory_recall` or `memory_write` event; `recall` logs `retrievals` and
`retrieval_items` (strategy `fts_bm25_v1`, or `recent_v1` for an empty query)
with `status_at_recall`, and returns a `retrieval_id` for `cite`. `remember`
dedupes on `(scope, content_hash)` with a `confirm` action. When the server
is down, `recall` answers `degraded`, and `remember` and `review` queue a
spool record that replay turns into the real memory or action.
`AGENT_MEMORY_AGENT` names the harness; `AGENT_MEMORY_SESSION` shares the
ledger session with hooks when the harness can pass it, otherwise the server
opens its own `partial` session.

Phase 0 numbers on this machine: about 4 to 5 ms per single-row insert over
Quack, measured by `mem smoke`.

## Droid hook and transcript overlap, B6 live check (2026-11-09)

From "Risks and open questions".

- Droid hooks and transcript import already overlap on prompts and tools.
  The hook's `UserPromptSubmit:` prefix, not a demonstrated difference between
  `message_id` and transcript entry `id`, caused duplicate prompts.
  `src/agent_memory/capture/identity.py` now owns the key rules and removes
  that prefix for keyed prompts. Synthetic overlap and spool replay tests
  pass. B6 live check (2026-11-09): for one synthetic prompt captured by a
  temporary local hook, `message_id` equaled the matching transcript entry
  `id`. No key-rule change was needed. Raw hook payloads and transcripts
  stay outside the repository; the live ledger was not used.

## Phase 0 audit (2026-10-25)

Probed a live server for policy bypasses, reviewed the code and schema, and
checked the Droid hooks documentation. Held up: the policy on both client
paths, token privacy in settings and error messages, file-access blocking,
and Unicode and literal tricks. Fixed afterwards: export failures no longer
stop the server, snapshots are pruned, extension autoload is off, replay
identities were added (`context_links.link_id`, unique harness session), and
`status_at_recall` is constrained. Tests pin each of these. The reassessment
above (CLI for capture, transcript import for Droid and Vibe, spool-based
extensions for omp and Amp, Phase 1 split) came out of the same pass, after
checking each harness's hook documentation.
