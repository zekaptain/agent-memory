# Phase 1 spec: original design text

This text was moved out of [phase-1-spec.md](../phase-1-spec.md) on
2026-11-11. It holds the parts of the original Phase 1 design that later
changes replaced, the test plan that was carried out, and the questions put
to review. It is kept for the record and is not the current contract; the
spec describes what the code does today.

## Idempotent insert

Rows are staged in temp tables as today, then inserted with an anti-join:

```sql
INSERT INTO mem.main.events (cols...)
SELECT cols... FROM stage_events_1 s
WHERE NOT EXISTS (SELECT 1 FROM mem.main.events e WHERE e.event_id = s.event_id)
```

`sessions` the same way on `session_id`. Everything for one input goes in one
`BEGIN ... COMMIT` so a partial write is impossible. Rows are inserted in
dependency order (sessions, then events with no parent, then events with a
parent) because foreign keys are checked at insert time.

First implementation step: a probe that confirms the anti-join runs correctly
through the attached Quack catalog (it should push the filter server-side).
If it does not, the fallback is one `SELECT event_id FROM mem.main.events
WHERE session_id = ?` before the insert and client-side filtering, with a
single retry on a primary-key violation.

## Spool

- `mem ingest --spool` walks the spool, replays each file in line order
  inside one transaction per file, then moves the file to
  `spool/done/<agent_id>/<harness_session_id>.<ts>.jsonl`. `--purge-done
  <days>` deletes old done files; nothing is deleted by default.

## Server

Every tool call writes an event in the caller's session first
(`memory_recall` or `memory_write`), then does its work, in one transaction.
The event is the hook that `why_answer` and `context_links` attach to.

**`recall(query: str, scopes: list[str] = ["global", <project>], limit: int = 8, types: list[str] | None = None)`**
1. Insert `events(kind='memory_recall', content=query)`.
2. Run the ranking query server-side through `CONNECT` (FTS lives on the
   server):
   ```sql
   WITH hits AS (
     SELECT memory_id, fts_main_memories.match_bm25(memory_id, <query literal>) AS bm25
     FROM memories)
   SELECT m.*, h.bm25,
          h.bm25 * (0.6 + 0.4 * m.importance) * (0.5 + 0.5 * m.confidence)
            * CASE WHEN m.scope = 'global' THEN 1.0 ELSE 1.1 END AS score
   FROM active_memories m JOIN hits h USING (memory_id)
   WHERE h.bm25 IS NOT NULL AND m.scope IN (<scope literals>)
   ORDER BY score DESC LIMIT <limit>
   ```
   The query text is the one place user content enters SQL text. It goes
   through `sql_literal` and the server's policy strips literals before the
   keyword check, so a crafted query cannot change the statement class.
   Tests cover quotes, semicolons, comments, and `UPDATE` inside the query.
3. Insert `retrievals` (strategy `fts_bm25_v1`, `sql_used`, `params`,
   `candidates`, `returned`, `latency_ms`, `request_event_id`) and one
   `retrieval_items` row per hit with `score_components`
   (`{"bm25", "importance", "confidence", "scope_boost"}`), `injected = true`,
   and `status_at_recall` from `memory_status`.
4. Return `retrieval_id` and the items (`memory_id`, `type`, `key`, `content`,
   `confidence`, `created_at`, `score`). The agent is told to pass
   `retrieval_id` to `cite`.

## Changes to migration 0001

Still unapplied, so edited in place rather than a 0002:
- `events.payload` keeps the raw hook or transcript entry under
  `payload.raw` only when `--keep-raw` is set; the default keeps the mapped
  fields only (size).
- Add `memory_recall`/`memory_write` to the `why_answer` chain if not already
  included: `why_answer` should also list the `memory_recall` event's
  `content` (the query) so an answer shows what was asked of memory.
- Index: `CREATE INDEX events_session_ts ON events(session_id, ts)` and
  `memories_scope_hash ON memories(scope, content_hash)` for the dedupe check.

## Tests

All with the real Quack server fixture from Phase 0.

1a:
- ids: stable across runs, differ by agent/session/kind/key.
- redact: fixture-driven positives and negatives per rule; count matches.
- blobs: threshold, idempotent write, `content_hash` of the full text.
- writer: idempotent replay (insert twice, row counts unchanged); dependency
  order with parented events in one batch; a rejected record and its
  dependents set aside while the rest lands; two writers racing on a new
  session.
- spool: append during simulated outage (server stopped), replay after
  restart, done/failed layout, `capture_error` for a bad line, redaction on
  replay.
- Droid adapter: a sanitised fixture transcript covering every row of the
  mapping table, including thinking on/off, document parts, compaction, and
  outcome; `--since-key` skips earlier entries.
- hook adapter: one fixture payload per hook event; unknown event becomes
  `note`.
- CLI: `mem event` and `mem ingest` exit codes and stderr with server up and
  down.

1b:
- Each tool through an in-process MCP client (`mcp` 2.x client session over
  stdio) against the Quack fixture: recall writes `events`, `retrievals`,
  `retrieval_items` with correct `status_at_recall`; quarantined and retired
  memories never returned; `remember` dedupe path; `cite` with and without
  `answer_event_id`; `review` flips status; `why_answer` returns the full
  chain for a recall-cite pair.
- Adversarial recall queries (`'; UPDATE ...`, `--`, `/* */`, unicode
  quotes) are stored verbatim in `retrievals.query_text` and never change the
  statement class.
- Degraded mode: server stopped mid-session, `recall` returns `degraded`,
  `remember` queues to the spool, replay creates the memory.

Checks before calling Phase 1 done: ruff, pyrefly, pytest, `mem smoke`, and a
manual run: `mem ingest --harness droid --transcript` on this session's own
transcript, then `mem sql "SELECT * FROM session_summary"`.

## Open questions for review

1. Thinking blocks default off. They are the most direct "why" signal but
   also the most sensitive. Flip the default?
2. Blob threshold 16 KiB / inline 4 KiB. Tool results from `Execute` often
   exceed this; is the inline prefix enough for audits, given the blob is
   always one path away?
3. `recall` scoring weights are a first guess; Phase 4 will tune them with
   real data. Fine to ship as `fts_bm25_v1` and keep the strategy name in
   every retrieval row so later changes are distinguishable?
4. `AGENT_MEMORY_AGENT` is self-reported by the MCP config. Per-harness
   tokens (Phase 3) are the real fix; acceptable for now?
