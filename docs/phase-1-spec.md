# Phase 1 spec: capture path (1a) and in-session tools (1b)

Status: approved and implemented, 2026-10-25 (see "As built" at the end).
Earlier design examples below are historical. Migration 0001 is immutable;
later contract changes use new migrations and the hardening spec.

## Goal

Real sessions land in the ledger (1a), and agents can read and write memory
from inside a session with full provenance (1b). Both go through the Phase 0
`MemoryClient` over Quack; nothing opens the database file directly.

Non-goals for Phase 1: installing hooks into any harness (Phase 2), audit
commands (`mem why`, Phase 3), dedupe and quality (Phase 4).

## 1a. Capture path

### Pipeline

```
adapter -> capture records -> writer (redact, blob, ids) -> ledger | spool
```

One normalised intermediate, the *capture record*, so every harness shares
the same redaction, blob handling, id derivation, insert, and spool code.

- `mem event --harness <id>`: reads one hook payload (JSON) from stdin and
  converts it into capture records. Used by shell-hook harnesses (Droid,
  Vibe).
- `mem ingest --harness <id> --transcript <path>`: converts a transcript
  file into capture records. Used at turn end by Droid and Vibe.
- `mem ingest --spool`: reads capture records already written to `spool/`
  (by `mem event` during an outage, or by in-process extensions for omp and
  Amp) and writes them to the ledger.

Both commands exit 0 whenever the record was preserved somewhere (ledger or
spool). They exit non-zero only when the input itself is unreadable. Hooks
must never block or fail a harness because the memory server is down.

### Capture record (schema v1)

JSON object, one per line in the spool:

```json
{
  "v": 1,
  "agent_id": "factory-droid",
  "harness_session_id": "68e0fcdd-...",
  "kind": "tool_call",
  "ts": "2026-10-25T14:03:11.412Z",
  "role": "assistant",
  "content": "Execute {\"command\": \"uv run pytest\"}",
  "payload": {"tool_name": "Execute", "tool_input": {"command": "uv run pytest"}},
  "source_key": "toolu_01ABC",
  "parent_key": null,
  "seq": 412,
  "session": {"cwd": "/Users/.../agent-memory", "model": "claude-...", "harness_version": null, "project": "agent-memory", "started_at": "2026-10-25T13:59:02Z", "metadata": {"title": "..."}}
}
```

Rules:
- `kind` is one of the seeded `event_kinds`. `session_start` records carry a
  `session` block; other records may carry one too, but only the first write
  for a `(agent_id, harness_session_id)` creates the `sessions` row.
- `source_key` identifies the thing in the harness (message id, tool-use id,
  compaction id, or `"<hook_event>:<message_id>"` for hook-only events).
  `parent_key` is the `source_key` of the record this one answers (a
  `tool_result` points at its `tool_call`).
- `seq` is the harness's own counter when it has one (Droid transcripts have
  `seq`), otherwise 0. Audits order by `(ts, recorded_at)` as already noted
  in `PLAN.md`.
- `payload` is stored as `VARIANT`. Adapters put structured fields here and a
  readable rendering in `content`.

### Deterministic ids

Namespace `AGENT_MEMORY_NS = uuid5(NAMESPACE_URL, "agent-memory")`.

- `session_id = uuid5(NS, f"{agent_id}/{harness_session_id}")`
- `event_id   = uuid5(NS, f"{agent_id}/{harness_session_id}/{kind}/{source_key}")`
- `parent_event_id` is derived the same way from `parent_key`, with the parent
  kind implied by the child (`tool_result` -> `tool_call`).

Replaying the same record produces the same ids, so replay is a no-op.

`agent_memory.capture.identity` chooses the source keys and MCP session
identity; `capture.ids` hashes those choices. Both Droid capture paths and
Vibe tool hooks use the shared builders. A keyed Droid prompt uses the bare
message ID. Without one, its hook uses `prompt:unkeyed:<ts>` and marks
`payload.dedupe` as `"unkeyed"`; it makes no hook/transcript dedupe claim.
Replies keep their entry IDs, thinking appends `#thinking`, and tool calls
and results keep their tool-use IDs. Session markers and notes retain their
existing keys.

### Idempotent insert

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

### Redaction

Applied to `content`, to every string inside `payload`, and to
`session.metadata`, before the row is staged or spooled. Replacement text is
`[REDACTED:<rule>]`; `events.redactions` holds the count.

Rules (module `agent_memory.capture.redact`, each a compiled regex with a name):
`aws_access_key`, `aws_secret` (when adjacent to an access key),
`github_token` (`ghp_`, `gho_`, `ghu_`, `ghs_`, `ghr_`, `github_pat_`),
`openai_key` (`sk-`), `anthropic_key` (`sk-ant-`), `slack_token` (`xox[abpr]-`),
`google_api_key` (`AIza`), `jwt`, `private_key_block`
(`-----BEGIN ... PRIVATE KEY-----` to `END`), `bearer_header`,
`url_credentials` (`scheme://user:pass@`), `generic_assignment`
(`(api[_-]?key|secret|token|password|passwd)\s*[:=]\s*['"]?\S{8,}`), and
`mem_token` (the exact value in `~/.agent-memory/token`, always).

Redaction also runs on spool records at replay, so a rule added later still
applies to records written before it. Tests include a fixtures file of
positive and negative samples per rule.

### Large content and blobs

If `content` exceeds 16 KiB after redaction, the full text is written to
`blobs/<sha256[:2]>/<sha256>.txt` (mode 0600, write-then-rename), the
`content` column keeps the first 4 KiB, `truncated = true`, `blob_ref` holds
the path relative to `blobs/`. `content_hash` and `content_bytes` always
describe the full redacted text. Blob writes are content-addressed, so replay
is idempotent here too.

### Spool

Layout: `spool/<agent_id>/<harness_session_id>.jsonl`, appended with
`O_APPEND` so concurrent hooks for one session interleave by line.

- `mem event` tries the server first with a short deadline (default 2 s,
  `AGENT_MEMORY_DEADLINE`). On any connection or write failure it appends the
  records to the spool and exits 0, printing one line to stderr.
- `mem ingest --spool` walks the spool, replays each file in line order
  inside one transaction per file, then moves the file to
  `spool/done/<agent_id>/<harness_session_id>.<ts>.jsonl`. `--purge-done
  <days>` deletes old done files; nothing is deleted by default.
- A line that fails to parse or to insert is recorded as a `capture_error`
  event (content: the error, payload: the raw line truncated to 4 KiB,
  source_key: `sha256(line)`), the line is kept in
  `spool/failed/<agent_id>/<file>`, and replay continues. This is the only
  place `capture_error` is written.
- `mem ingest --spool` is also what omp and Amp extensions call at turn end
  (Phase 2); in Phase 1 the spool is tested with hand-written records.
- `mem serve` runs `mem ingest --spool` logic once at startup, so an outage
  drains itself as soon as the server is back.

### Transcript adapters

Each adapter is a module in `agent_memory/adapters/` with two functions:
`from_hook(payload: dict) -> list[Record]` and
`from_transcript(path: Path, since_key: str | None) -> list[Record]`.

**Droid** (`~/.factory/sessions/<cwd-slug>/<uuid>.jsonl`, structure verified
locally on 2026-10-25). Entry `type` values and their mapping:

| transcript entry | record kind | source_key | content |
|---|---|---|---|
| `session_start` | `session_start` | entry `id` | title; `session` block from `cwd`, `version`, `id` |
| `message` role user, `text` parts | `user_msg` | entry `id` | joined text |
| `message` role user, `document` parts | `user_msg` | entry `id` | `[document]` plus text parts; payload keeps source type and name, never bytes |
| `message` role user, `tool_result` parts | `tool_result` (one per part) | part `tool_use_id` | result content; `parent_key` = `tool_use_id` |
| `message` role assistant, `text` parts | `assistant_msg` | entry `id` | joined text; payload `modelId`, `apiProvider`, `reasoningEffort`, `compactionSummaryId` |
| `message` role assistant, `thinking` parts | `assistant_msg` | `<entry id>#thinking` | thinking text; payload `{"part": "thinking", "durationMs": ...}`; off unless `--include-thinking` |
| `message` role assistant, `tool_use` parts | `tool_call` (one per part) | part `id` | `name` + JSON input |
| `compaction_state` | `compaction` | entry `id` | `summaryText`; payload `summaryTokens`, `removedCount`, `anchorMessage` |
| `agent_turn_outcome` | `outcome` | `turnId` | `reason`; payload `resultKind` |
| `todo_state` | skipped | | |

`modelId` on the first assistant message fills `sessions.model`.
`session.project` is the basename of `cwd`. Signatures, encrypted reasoning
content, and `openai*` fields are dropped. Droid hooks already record
`user_msg`, `tool_call`, and `tool_result`, as well as session boundaries,
compaction, and notes. `capture.identity` gives hook and transcript prompts
the same bare message ID and tools the same tool-use ID. The former
`UserPromptSubmit:` prefix caused duplicate prompts even when the harness
IDs matched. Live comparison of hook `message_id` and transcript entry `id`
is recorded in [PLAN.md](../PLAN.md), separately from the synthetic tests.

**Vibe**: `~/.vibe/logs/session/` exists locally but holds no transcript yet.
The adapter ships as a stub that raises `UnsupportedTranscript` with the path,
and gets its mapping in Phase 2 once a real file is available. `from_hook`
handles `pre_tool`/`post_tool`/`post_agent` as documented (session row
created from `session_id`, `cwd`; `tool_call`/`tool_result` keyed on
`tool_call_id`).

**omp, Amp**: no transcript adapter; their extensions write capture records
directly (Phase 2). `mem ingest --spool` is the whole integration.

**Junie**: none.

### Hook adapter (`mem event`)

Common stdin fields across Droid and Vibe (`session_id`, `cwd`,
`hook_event_name`, `transcript_path`) map to the `session` block.
`capture.identity` owns the keys: prompt and tool IDs are shared with
transcript import, while hook-only notes use the event name plus ID or
timestamp. `SessionStart` also includes its source (startup/resume/clear/compact).
Mapping:

| hook event | kind | notes |
|---|---|---|
| SessionStart | `session_start` | payload `source` (startup/resume/clear/compact); the `sessions` row is created once |
| SessionEnd | `session_end` | payload `reason` |
| UserPromptSubmit | `user_msg` | content = prompt |
| PreToolUse / pre_tool | skipped by default | `--record-pre-tool` for debugging |
| PostToolUse / post_tool | `tool_call` + `tool_result` | result parented to call |
| PreCompact | `compaction` | payload `trigger`, `custom_instructions` |
| Stop / post_agent | none | the Phase 2 hook script runs `mem ingest --transcript` after this |
| SubagentStop, Notification | `note` | payload = stdin minus transcript text |

Any hook event not in the table becomes a `note` with the raw payload, so new
harness events are never silently lost.

### CLI shapes

```
mem event  --harness droid|vibe [--spool-only] [--keep-raw]
mem ingest --harness droid --transcript PATH [--since-key KEY] [--include-thinking]
mem ingest --spool [--purge-done DAYS]
mem spool  status            # files, lines, oldest record per harness
```

`--since-key` lets a hook pass the last message id it knows about so
re-ingesting a long transcript each turn stays cheap; without it the anti-join
makes re-ingest correct but slower.

## 1b. In-session tools (`mem-mcp`)

### Server

`mem-mcp` console script, `python -m agent_memory.mcp_server`, stdio transport,
built on `mcp>=2,<3` (`MCPServer` from `mcp.server.mcpserver`; verified 2.3.0
installs and imports). One `MemoryClient` per server process, opened lazily
on the first tool call and reopened on connection error. Environment:
`AGENT_MEMORY_HOME`, `AGENT_MEMORY_PORT`, and `AGENT_MEMORY_AGENT` (required;
the harness id, since MCP gives no reliable caller identity) plus
`AGENT_MEMORY_SESSION` (harness session id when the harness can pass it;
otherwise the server derives one from its own start time and marks the
session `partial`).

Every tool call writes an event in the caller's session first
(`memory_recall` or `memory_write`), then does its work, in one transaction.
The event is the hook that `why_answer` and `context_links` attach to.

Tools (names and schemas; all return structured content plus a short text
summary):

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

Empty-query recall (`query=""`) returns the newest active memories in scope
by `importance`, strategy `recent_v1`, for session-start context.

**`remember(content: str, type: str, scope: str = <project>, key: str | None, confidence: float = 0.7, importance: float = 0.5, trust_level: "user"|"agent"|"tool" = "agent", supersedes: str | None, source_event_ids: list[str] = [], valid_from, expires_at)`**
1. Insert `events(kind='memory_write')` with the request as payload.
2. Insert `memories` (`memory_id = uuidv7()`, `content_hash = sha256`,
   `created_by_session`). Content is redacted with the same rules as capture.
3. Insert `memory_sources`: the `memory_write` event as `derived_from` (or
   `user_stated` when `trust_level='user'`), plus any `source_event_ids` the
   caller names.
4. Return `memory_id` and the status after insert (`quarantined` for
   `trust_level='tool'`, from `memory_status`).

If an active memory with the same `(scope, content_hash)` already exists, no
new row is written; the tool inserts `memory_actions(action='confirm')` and
returns the existing id with `deduplicated = true`.

**`cite(retrieval_id: str | None, memory_ids: list[str] = [], context_event_ids: list[str] = [], answer_event_id: str | None)`**
Writes `context_links` for the answer. When `answer_event_id` is omitted the
server inserts an `events(kind='assistant_msg', content=NULL,
payload={"placeholder": true})` row and returns its id so the transcript
ingest can later link the real reply. A captured, server-qualified `cite`
tool result identifies that placeholder by UUID; a complete captured turn
must have one unique reply. The placeholder can live in an MCP-owned
session, while calls, results, and replies must belong to the observed
transcript session. Binding evidence records `placeholder_session_id`.
Explicit answer IDs take priority. Missing or ambiguous evidence remains
unresolved; there is no nearest-message fallback. This also gives Tier C
harnesses partial `why_answer` coverage without a captured reply.

**`review(action: "approve"|"retire"|"quarantine"|"confirm"|"flag_contradiction", memory_id: str, reason: str, related_memory_id: str | None)`**
Inserts `memory_actions` with `actor = AGENT_MEMORY_AGENT` and the session id,
and an `events(kind='memory_write')` row. Returns the resulting status.

**`status()`** (read-only): server reachable, `session_id`, counts for this
session, spool depth. Lets a harness surface outages to the user.

`relate` and `graph` were added after Phase 1 with the relation graph; see
`README.md` and the "Relation graph" paragraph in `PLAN.md`.

Error handling: when the server is unreachable, `recall` returns an empty
result with `degraded: true` and the reason; `remember`/`review` write a
capture record to the spool (`kind='memory_write'`, payload = the request)
and return `queued: true`. Nothing in the agent loop raises.

### Tool descriptions

The MCP tool descriptions are the agent-facing contract and are part of the
spec. `remember` says: store durable facts, decisions, preferences, and
procedures; not transient task state; one statement per call; name the
`supersedes` id when correcting. `recall` says: call before answering
questions about the user, the project's history, or past decisions; pass
`retrieval_id` to `cite`. `review` says: use `retire` when a memory is wrong,
`flag_contradiction` when two conflict.

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

## Package layout

```
src/agent_memory/
  capture/
    record.py      # Record dataclass, validation, (de)serialisation
    identity.py    # source keys, binding identities, MCP session selection
    ids.py         # uuid5 derivation
    redact.py      # rules and Redactor; cases in tests/fixtures/redact_cases.json
    blobs.py       # write_blob(), threshold constants
    writer.py      # records -> staged rows -> idempotent insert, or spool
    spool.py       # append, iterate, mark done/failed, status
  adapters/
    __init__.py    # registry: {"droid": DroidAdapter, "vibe": VibeAdapter}
    base.py        # Adapter protocol
    droid.py
    vibe.py
  mcp_server/
    __init__.py    # MCPServer construction, console script entry
    tools.py       # recall/remember/cite/review/status
    ranking.py     # SQL builders for fts_bm25_v1 and recent_v1
  cli.py           # + event, ingest, spool
```

`pyproject.toml`: add `mcp>=2,<3`; scripts `mem` (existing) and `mem-mcp`.

## Tests

All with the real Quack server fixture from Phase 0.

1a:
- ids: stable across runs, differ by agent/session/kind/key.
- redact: fixture-driven positives and negatives per rule; count matches.
- blobs: threshold, idempotent write, `content_hash` of the full text.
- writer: idempotent replay (insert twice, row counts unchanged); dependency
  order with parented events in one batch; partial-failure rollback.
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

## As built (2026-10-25)

Capture identity hardening (2026-11-09):

- Shared key builders remove only the Droid prompt-hook prefix. Unkeyed
  prompts are marked, not guessed. Existing transcript, tool, thinking,
  session-marker, and note keys remain stable.
- MCP sessions use `AGENT_MEMORY_SESSION` when available (`session_link =
  "env"`), otherwise a unique own session (`session_link = "own"`).
  Session metadata retains `own_session`; MCP-only coverage stays partial.
- Migration `0006_cross_session_answer_bindings` replaces only the binding
  validation trigger's session-equality rule. Event kinds, placeholder
  checks, foreign keys, linear corrections, and append-only guards remain.
  Migration `0005_judge_agent_policy` separately registers the judge agent
  and removes correction-work writes from client authorization.
- Reconciliation fetches placeholders by UUID across sessions. Transcript
  evidence keeps its session boundary. Both citation/import arrival orders,
  explicit future reply IDs, repeated imports, and spool replay are covered
  through real MCP, capture, CLI, and Quack paths.

Implemented as specified except for these points, each forced by a probe:

- **Idempotent insert.** The anti-join `INSERT ... WHERE NOT EXISTS` through
  the attached catalog is not supported by DuckDB 2.0.0 ("multiple streaming
  scans + insert"). The writer uses the spec's fallback: it reads the
  session's existing ids first, filters, inserts, and retries once on a
  duplicate key.
- **Parent ordering.** A self-referencing foreign key is checked before the
  statement's own rows are indexed, so parents and children cannot share one
  `INSERT`. `MemoryClient.insert` now takes ordered `(table, rows)` pairs and
  the writer sends one statement per depth in one transaction.
- **Recall SQL.** `match_bm25` cannot be aliased and referenced; the ranking
  query uses a CTE. Queries are normalised (lowercase, punctuation removed)
  before FTS because the tokenizer keeps curly quotes attached to words. The
  bm25 ranking runs server-side and the final score is applied in Python over
  at most 50 candidates, so `retrievals.candidates` is that candidate count.
- **Transactions.** Recall's ranking read and its provenance write are two
  statements (one `CONNECT` query, then one transaction with the event, the
  retrieval, and its items). The write is atomic; the read is not in it.
- **Spool records for queued writes.** `Record` gained optional `memory` and
  `action` fields so a `remember` or `review` queued during an outage becomes
  the real row at replay, with deterministic `memory_id` / `action_id`.
- **Package names.** The MCP package is `agent_memory.mcp_server` (not `mcp`)
  to avoid shadowing the SDK, run as `mem-mcp` or
  `python -m agent_memory.mcp_server`. `ToolError` subclasses the SDK's so the
  message reaches the agent; other exceptions are reported without detail.
- **Tool arguments.** `remember` takes `type` in the MCP schema (the method is
  `memory_type` internally). `status()` was added as a fifth tool.
- **Migration 0001** gained two indexes only; `why_answer` already exposed
  the recall query as `query_text`.
- **Droid hooks** record `UserPromptSubmit` and `PostToolUse` too (not only
  session boundaries); Phase 2 decides which hooks to install once the
  `message_id` question is settled.

Verified: ruff, pyrefly, 293 tests (110 new), `mem smoke` 16/16, and an import
of this project's own 1000-event Droid transcript (0.8 s, 0 orphan results,
replay 50 ms).

### Audit corrections (2026-11-08)

- Record redaction now covers content, payload, retained raw input, session
  fields, queued memories, and queued actions, including nested dictionary
  keys. Identity fields used to derive capture IDs stay unchanged. Raw input
  is dropped unless `keep_raw` is enabled. An optional `redactions` field
  carries the removal count through the v1 spool format; older records
  default to zero. Reapplying a rule to its own marker does not count as
  another removal.
- Structured credential fields such as `{"password": "..."}` use the
  existing assignment rules as well as ordinary value redaction. Specific
  secret rules still take precedence. Schema-invalid JSON is decoded and
  structurally redacted, including escaped strings. Undecodable lines are
  retained as SHA-256 digests with diagnostics, not raw text that may hide
  escaped credentials.
- MCP recall and graph queries are redacted before normalization and SQL
  generation. Review reasons, memory keys/scopes, and merge inputs use the
  same redactor before constructing their persisted copies.
- Appenders and replay share a short per-session file lock. Replay moves
  each pending file into `spool/replaying/<agent>/` under that lock, then
  releases it before contacting the ledger. New appends go into a new
  pending file. One replay process owns claimed files at a time; an
  interrupted claim remains discoverable on the next replay. Claimed
  records and malformed lines are redacted before entering `done/` or
  `failed/`. Existing archives are not rewritten.
- The writer looks up incoming event IDs and referenced parent IDs in
  batches of at most 500, rather than reading the whole session. It resolves
  all parent depths before inserting and rejects cycles. Queued memories
  and actions deduplicate within a batch as events already did. The
  duplicate-key retry and separate parent-before-child statements remain.
- Record-by-record recovery also orders parent and queued-memory
  dependencies. A failed parent or cycle keeps its dependent records in
  `failed/` rather than permanently inserting them without their links.
  Unrelated valid records still replay.
- Failed local staging removes its temporary table. Failed exports remove
  their own partial directory and unpublished link, leaving the previous
  `latest` snapshot intact.

### Contract hardening

`docs/logic-hardening-spec.md` defines the acceptance policy and rollout
boundaries. New migrations, not edits to 0001, separate proposed corrections
from accepted replacements, preserve queued IDs through deduplication links,
record capture observations, and bind placeholders to actual replies.

Observed coverage does not inherit the harness tier. New MCP sessions and
isolated hooks are partial. A transcript import records the source
fingerprint, adapter/policy version, cursor and line boundaries, parse
errors, exclusions, and session metadata. The immutable session row remains
available alongside effective metadata. Omitted thinking and attachment
content are explicitly excluded. A previous import does not cover later
activity. A transcript's `user` role still does not establish human authorship.

The Droid adapter retains message/turn/channel identifiers and uses a
completed captured turn to reconcile a cite result's placeholder with a
unique reply. Explicit final-channel evidence can disambiguate a reply;
timestamps and "the next assistant message" cannot. Incremental imports keep
the full source boundary evidence needed for reconciliation. Repeated
imports do not duplicate observations or bindings. New ambiguity appends a
binding correction rather than rewriting the original link.

Captured tool identity must identify the configured `agent-memory` MCP
server, not an unrelated tool with a `cite` suffix. Override the default
name with `AGENT_MEMORY_MCP_SERVER_NAME` in the capture environment when
needed. Missing server identity leaves attribution unresolved. An explicit
answer ID supplied before import is retained on a placeholder and bound
when that exact reply arrives.

Replay resolves dependencies across the claimed spool snapshot rather than
relying on file modification times. This preserves cross-session queued
corrections and reviews, including interleaved files whose records form an
acyclic dependency graph. Interrupted and newly appended files for the same
session get distinct claim paths. As before, failed validation is terminal,
while a transport interruption leaves claimed files available for retry.

`why_answer` preserves raw link and placeholder identities, exposes resolved
attribution, and retains zero-result FTS and graph searches. A miss has
`returned = 0`; an unavailable recall remains degraded. Original retrieval
status snapshots are never recalculated. Returned, captured in context when
observed, and declared used are distinct evidence, not causal attribution
to a model's reasoning.
