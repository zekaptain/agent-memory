# Phase 1 spec: capture path (1a) and in-session tools (1b)

Status: approved and implemented, 2026-10-25 (see "As built" at the end).
The parts of the original design that later changes replaced, the test
plan, and the review questions are kept in
[archive/phase-1-original-design.md](archive/phase-1-original-design.md).
Migration 0001 is immutable; later contract changes use new migrations and
the hardening spec.

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

Both commands exit 0 whenever every record was kept somewhere: in the
ledger, in the spool, or under `spool/failed/` for a record the ledger
rejected. Hooks must never block or fail a harness because the memory server
is down or a record is bad.

`mem event` runs as a hook, and a harness acts on a hook's exit code and
stdout. It prints nothing on stdout (a harness can add it to the agent's
context) and reports problems on stderr. It exits 1 when the input is
unreadable or nothing could be kept, and never 2, which a harness reads as
an instruction to block the agent; usage errors exit 1 as well.
`mem ingest` prints a summary on stdout and exits 2 on a usage error.

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
  in [architecture.md](architecture.md).
- `payload` is stored as `VARIANT`. Adapters put structured fields here and a
  readable rendering in `content`.
- `raw` is the whole hook payload, present only when the hook ran as
  `mem event --keep-raw`. A record that has one keeps it, redacted, under
  `payload.raw`. No later step is told to keep it or drop it, so a record
  that waits in the spool keeps it too.

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

The writer first reads which of the incoming session and event ids the
ledger already holds, in batches of at most 500, and inserts only the rest
in one transaction. A write that loses a race with another capturer is read
again and retried up to three times. See "As built" for the probe that
forced the pre-read and for parent ordering, and "Capture delivery
(2026-11-10)" for the retry. The anti-join designed first is in
[archive/phase-1-original-design.md](archive/phase-1-original-design.md#idempotent-insert).

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
  `AGENT_MEMORY_DEADLINE`). When the ledger is unavailable or too slow it
  appends the records to the spool and exits 0, printing one line to stderr.
- `mem ingest --spool` claims every pending file by moving it to
  `spool/replaying/<agent_id>/`, writes the claimed records as one batch so
  that a record can refer to one in another file, then moves each file to
  `spool/done/<agent_id>/<harness_session_id>.<ts>.<ns>.jsonl`. `--purge-done
  <days>` deletes old done files; nothing is deleted by default.
- A line that fails to parse, or a record the ledger rejects, is recorded as
  a `capture_error` event (content: the reason, payload: the raw line
  truncated to 4 KiB, source_key: `sha256(line)`), the line is kept in
  `spool/failed/<agent_id>/<harness_session_id>.jsonl`, and the rest is
  written. Records that refer to a rejected record are rejected with it.
  The same happens when a record is rejected on the direct path, without
  passing through the spool. A record rejected again, by a retried hook or
  a reimported transcript, is kept there once.
- `mem spool retry` moves every kept line back to the spool. Nothing
  retries a rejected record on its own.
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
is recorded in [archive/plan-history.md](archive/plan-history.md),
separately from the synthetic tests.

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
mem event  --harness droid|vibe [--spool-only] [--keep-raw] [--record-pre-tool]
mem ingest --harness droid --transcript PATH [--since-key KEY] [--include-thinking]
           [--spool-only]
mem ingest --spool [--purge-done DAYS]
mem spool  status            # files, lines, oldest record per agent: pending and rejected
mem spool  retry             # return rejected records to the spool
```

`--since-key` lets a hook pass the last message id it knows about so
re-ingesting a long transcript each turn stays cheap; without it the
writer's check for ids the ledger already holds makes re-ingest correct but
slower.

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

`recall` writes a `memory_recall` event in the caller's session, and
`remember` and `review` write a `memory_write` event. A recall reads first
and then writes its event, its retrieval, and the retrieval's items in one
transaction (see "Transactions" under "As built").
The event is the hook that `why_answer` and `context_links` attach to.

Tools (names and schemas; all return structured content plus a short text
summary):

**`recall(query: str, scopes: list[str] = ["global", <project>], limit: int = 8, types: list[str] | None = None)`**
1. Redact the query, then normalise it for FTS (lowercase, punctuation
   removed).
2. Read at most 50 candidates on the server (FTS lives there), ordered by
   bm25 and then newest first, from `active_memories` in the requested
   scopes. The query text is the one place user content enters SQL text. It
   goes through `sql_literal` and the server's policy strips literals before
   the keyword check, so a crafted query cannot change the statement class.
   Tests cover quotes, semicolons, comments, and `UPDATE` inside the query.
3. Score the candidates in Python as `bm25 * (0.6 + 0.4 * importance) *
   (0.5 + 0.5 * confidence) * scope_boost`, where `scope_boost` is 1.0 for
   `global` and 1.1 for any other scope. Keep the best `limit`; equal
   scores go newest first.
4. In one transaction insert `events(kind='memory_recall')`, `retrievals`
   (strategy `fts_bm25_v1`, `sql_used`, `params`, `candidates`, `returned`,
   `latency_ms`, `request_event_id`) and one `retrieval_items` row per hit
   with `score_components` (`{"bm25", "importance", "confidence",
   "scope_boost"}`), `injected = true`, and `status_at_recall` from
   `memory_status`.
5. Return `retrieval_id` and the items. The agent is told to pass
   `retrieval_id` to `cite`.

The statement itself is in `mcp_server/ranking.py`; see "Recall SQL" under
"As built" and "One recall path and its strategies (2026-11-11)". The
single SQL statement designed first is in
[archive/phase-1-original-design.md](archive/phase-1-original-design.md#server).

Empty-query recall (`query=""`) returns the newest active memories in scope
by `importance`, strategy `recent_v1`, for session-start context.

**`remember(content: str, type: str, scope: str = <project>, key: str | None, confidence: float = 0.7, importance: float = 0.5, trust_level: "user"|"agent"|"tool" = "agent", supersedes: str | None, source_event_ids: list[str] = [], valid_from, expires_at, valid_to)`**
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
`approve` is refused: only the `human` agent approves, and `mem-mcp` does
not run as it. See "Only a human approves a memory" under "As built".

**`status()`** (read-only): server reachable, `session_id`, counts for this
session, spool depth (`spool_pending_lines`), and the number of rejected
records kept under `spool/failed/` (`spool_rejected_lines`). Lets a harness
surface outages and rejected records to the user.

`relate` and `graph` were added after Phase 1 with the relation graph; see
`README.md` and the "Relation graph" paragraph in
[architecture.md](architecture.md).

Error handling: when the server is unreachable, `recall` returns an empty
result with `degraded: true` and the reason; `remember`/`review` write a
capture record to the spool (`kind='memory_write'`, payload = the request)
and return `queued: true`. Nothing in the agent loop raises. A request
the tool refuses, such as an agent's `approve`, is a tool error and is
never queued.

### Tool descriptions

The MCP tool descriptions are the agent-facing contract and are part of the
spec. `remember` says: store durable facts, decisions, preferences, and
procedures; not transient task state; one statement per call; name the
`supersedes` id when correcting. `recall` says: call before answering
questions about the user, the project's history, or past decisions; pass
`retrieval_id` to `cite`. `review` says: use `retire` when a memory is wrong,
`flag_contradiction` when two conflict; only the user approves a
quarantined memory, with `mem review`.

## Migration 0001

Migration 0001 is applied and immutable. The edits planned for it while it
was still unapplied are in
[archive/phase-1-original-design.md](archive/phase-1-original-design.md#changes-to-migration-0001);
"As built" says which of them it gained. Every later schema change is a new
migration.

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
    session.py     # the session and event rows of every writer (added later)
    spool.py       # append, iterate, mark done/failed, status
    pipeline.py    # deliver to the ledger or the spool; replay the spool
    observations.py  # observed capture boundaries (added with hardening)
    attribution.py   # answer attribution from captured tool results
  adapters/
    __init__.py    # registry: {"droid": DroidAdapter, "vibe": VibeAdapter}
    base.py        # Adapter protocol
    droid.py
    vibe.py
  mcp_server/
    __init__.py    # MCPServer construction, console script entry
    __main__.py    # python -m agent_memory.mcp_server
    tools.py       # recall/remember/cite/review/status
    recall.py      # run a strategy and record the retrieval (added later)
    ranking.py     # SQL builders for fts_bm25_v1 and recent_v1
  graph/           # added after Phase 1 with the relation graph
    model.py       # relations as fact memories with normalized keys
    sql.py         # SQL builders for the graph
    recall.py      # graph_hop_v1 and graph_relations_v1 strategies
    merge.py       # entity merges as append-only history
    resolve.py     # entity resolution with a local model
    cli.py         # mem graph
  cli.py           # + event, ingest, spool
```

`pyproject.toml`: add `mcp>=2,<3`; scripts `mem` (existing) and `mem-mcp`.

## Tests and review questions

The Phase 1 test plan and the four questions put to review are in
[archive/phase-1-original-design.md](archive/phase-1-original-design.md#tests).
The tests are `tests/test_capture.py`, `test_writer.py`, `test_spool.py`,
`test_adapters.py`, `test_cli_capture.py`, `test_mcp.py`, and
`test_ledger_session.py`.

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
  is dropped unless `keep_raw` is enabled. (A record that carries raw input
  now keeps it; see "A kept hook payload stays with its record" below.) An
  optional `redactions` field carries the removal count through the v1
  spool format; older records default to zero. Reapplying a rule to its
  own marker does not count as another removal.
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
- Recovery from a rejected record also orders parent and queued-memory
  dependencies. A failed parent or cycle keeps its dependent records in
  `failed/` rather than permanently inserting them without their links.
  Unrelated valid records still replay. (The writer does this itself since
  the capture delivery change below.)
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
session get distinct claim paths. As before, a record that fails validation
is rejected, while a transport interruption leaves claimed files pending.

`why_answer` preserves raw link and placeholder identities, exposes resolved
attribution, and retains zero-result FTS and graph searches. A miss has
`returned = 0`; an unavailable recall remains degraded. Original retrieval
status snapshots are never recalculated. Returned, captured in context when
observed, and declared used are distinct evidence, not causal attribution
to a model's reasoning.

### Capture delivery (2026-11-10)

Two defects prompted this. Two hooks that started one session at the same
time could lose a record: the second commit failed with a conflict, which
the direct path treated as final and dropped. And `mem event` exited 2 on
unreadable input and printed a success line on stdout; a harness blocks the
agent on exit 2 and can add a hook's stdout to the agent's context.

- `Writer.write` returns what it wrote together with the records the ledger
  rejected, each with its position in the batch and the reason. It raises
  only when the ledger is unavailable. A failed answer-binding pass is
  returned as a warning; the write it follows has already committed.
- A write that loses a race is repeated. Another capturer can commit the
  same session or event between the pre-read and the commit, which DuckDB
  reports as a `TransactionException` at commit or as a duplicate-key
  `ConstraintException` at insert. The writer reads again and retries up to
  three times. A conflict that outlasts the retries counts as an
  unavailable ledger, so the batch is spooled.
- The writer isolates a record at fault. It orders the batch so that every
  reference (parent event, memory source, superseded or reviewed memory)
  comes before its referrer, halves a failing batch until the record at
  fault is alone, rejects that record and every record that refers to it,
  and writes the rest. An exception nobody expected is handled the same
  way when one record causes it, with the exception type in the reason.
  The session row and the transcript observation are still derived from
  the whole batch. Replay no longer has its own record-by-record recovery.
- `deliver` keeps the deadline and the choice between ledger and spool. It
  files rejected records under `spool/failed/` and writes one
  `capture_error` event per rejected record; if the ledger does not take
  that event in time, the event is spooled. `deliver` raises when it could
  keep nothing, and it no longer checks memory intervals before writing:
  an invalid interval is rejected like any other bad record. MCP `remember`
  still validates the interval before queueing.
- `Record.from_dict` checks that `payload`, `memory`, `action`, `raw`,
  `observation`, and `attribution` are objects and that `role`, `content`,
  `parent_key`, and `parent_kind` are strings. A spool line with the wrong
  shape is now a parse error. Before, it raised inside the writer and left
  the claimed files in `spool/replaying/` on every replay.
- `mem event` follows the hook contract in the pipeline section above.
  `mem spool status` lists rejected records, and `mem spool retry` returns
  them to the spool.
- `mem judge import` writes the records of a transcript that the ledger
  accepts, reports the file as failed with the first reason, and leaves it
  out of the imports log so that the next import reads it again.

Verified: ruff, pyrefly, 803 tests, `mem smoke` 16/16, and 25 rounds of
three concurrent `mem event` processes against a running `mem serve` with no
record lost, spooled, or rejected. The same 1000-record batch takes 440 ms
to write (414 ms before) and 68 ms to write again (63 ms before); one hook
record takes 16 ms, as before.

Follow-ups the same day:

- The Droid adapter passed a `PreCompact` hook's `custom_instructions`
  through as content without turning it into text, so a value that was not
  text was rejected. It now goes through `text_of` like other hook text.
- The MCP `status` tool reports rejected records as `spool_rejected_lines`
  beside `spool_pending_lines`, so an agent can see them without the CLI.

### Recall order and merged prose (2026-11-11)

- Recall listed memories with equal scores oldest first, although the
  candidate SQL orders them newest first and empty-query recall promises
  the newest memories. Equal scores now go newest first under both
  strategies.
- An entity merge carries the canonical name into a relation's prose. It
  replaced the alias wherever those letters appeared, inside other words
  and inside longer names, and a backslash in the canonical name failed
  the tool call. The alias is now replaced only where it stands alone as
  a name, and the canonical name is inserted as written.
- A relation between the alias and the canonical name, in either
  direction, stopped the merge with an internal error, because renamed it
  would connect an entity to itself. The merge now retires that relation
  and lists it under `retired` with `into` empty. The merge event still
  records both names.

### One recall path and its strategies (2026-11-11)

`recall` and `graph` each repeated the read, the ranking and the write of
the event, the retrieval and its items, and two `mem graph` commands read
relations with no retrieval at all.

- `run_recall` in `mcp_server/recall.py` is the one place that reads
  memories on an agent's behalf. It runs a strategy, then writes the
  `memory_recall` event, the retrieval and one item per memory returned in
  one transaction. `MemoryTools` turns an unreachable ledger into a
  degraded result, as before.
- A strategy supplies its name, its SQL, the parameters and request payload
  to store, and how rows become items. There are four: `TextMatch`
  (`fts_bm25_v1`) and `Recent` (`recent_v1`) in `mcp_server/ranking.py`,
  and `RelationWalk` (`graph_hop_v1`) and `RelationListing`
  (`graph_relations_v1`) in `graph/recall.py`. A new way of choosing
  memories is a new strategy, never a read beside `run_recall`.
- What `recall` and `graph` return and store is unchanged.
- `mem graph entities` and `mem graph resolve` now read through
  `MemoryTools.relations`, which records a `graph_relations_v1` retrieval
  with one item per relation read, oldest first and without a score,
  because a listing does not rank. A listing of an empty scope is recorded
  as a miss. `relations` is not an MCP tool.
- Both commands print the retrieval's id. Each merge that
  `mem graph resolve --apply` records carries that id as
  `merge.retrieval_id` in its event, so a merge can be traced to what was
  read for its proposal. The model itself is given the entity names and up
  to three sentences per entity drawn from those relations, narrowed by
  `--entity-type` when it is given; the retrieval lists every relation
  read, not only the ones that reached the prompt.

### One ledger session for every writer (2026-11-11)

The capture writer, the MCP tools, and admission each built session and
event rows by hand, and the three had drifted apart. `capture/session.py`
now holds the one shape of both rows, and a `LedgerSession` that the MCP
tools, `mem graph`, and admission open once and then give events in
their own transactions.

- A tool write failed when a hook of the same harness session committed
  the session row between the tool's read and its insert. The duplicate
  key was neither retried nor queued. Opening a session is now repeated
  after a lost race, as a capture write is, and finds the row.
- A queued tool write that reached the ledger before its session did
  left a session with no project, no working directory, and no metadata,
  and a generic observation (adapter `<agent>_records_v1`). A queued
  write now carries what the session row and the observation are made
  from, so the session is the same whichever arrives first. Lines queued
  by an earlier version do not carry it and replay as they did.
- Online, a tool's event kept its whole content inline with no byte
  count, blob, or token estimate; the same write replayed from the spool
  got all three. Every event's content now follows the rule in "Large
  content and blobs" above. A `memory_recall` event gains its byte count
  and token estimate, and the event of an empty query stores the hash of
  the empty text where it stored none.
- `mem graph` runs through the MCP tools' code, and its sessions said so:
  `via: mem-mcp`, observed in source mode `mcp`. A command's session now
  says `via: mem graph` and is observed in mode `record` with adapter
  version `mem_cli_v1`. `own_session` and `session_link` are kept for MCP
  sessions only. Sessions already recorded keep what they say.
- The session row and its observation go in one insert, where the
  observation used to follow in a second.
- An entity merge is built and applied in `graph/merge.py` (`EntityMerge`,
  `apply_merge`). `merge_entity` on the tools supplies the session and
  turns a failure into a tool error or a degraded result, as before.
- What `recall`, `remember`, `review`, `cite`, `relate`, and `graph`
  return is unchanged.

### Entity resolution keeps what worked (2026-11-11)

`mem graph resolve` asks a local model to cluster one block of entities
at a time: up to sixty names of one entity type (`--block-size`). It
asked once per block, with 4,096 tokens and nothing said about thinking.
A thinking model that spent the budget before the clusters closed raised
an error, and the error discarded the merges already proposed from every
other block.

- Each block is now one model call through `model_call.py`, with the
  reasoning judge's three attempts: the request as before; once more with
  12,288 tokens when the reply is cut off; and a last time with thinking
  off and the grammar dropped when there is still no JSON object. See
  "Model calls, as built" in
  [judge-experiment-spec.md](judge-experiment-spec.md).
- A block that still fails does not cost the others. The command prints
  the merges from the blocks that worked, records them when `--apply` is
  given, then names each failed block (how many entities, their type,
  the first and last name, and the reason) and exits 1. It prints
  "proposes no merges" only when no block failed.
- A model server that cannot be reached is asked once. The blocks after
  it are not tried, and the output says how many.
- `propose` returns a `Proposal` (merges, failed blocks, and the count of
  blocks not tried) where it returned a list of merges. `parse_clusters`
  takes the JSON object the model call found, not the reply's text.

Regressions are in `tests/test_graph.py` and `tests/test_model_call.py`.

### A kept hook payload stays with its record (2026-11-11)

`mem event --keep-raw` keeps the whole hook payload under `payload.raw`.
Whether to keep it was an argument handed to each step, so a record that
went through the spool lost its payload unless the replay was given
`--keep-raw` again, and the replay at server start never was.
`mem ingest --transcript --keep-raw` did nothing, because no transcript
adapter sets a raw payload.

- A record that carries `raw` keeps it wherever it goes: written at once,
  spooled, replayed by `mem ingest --spool`, or replayed when `mem serve`
  starts. `deliver`, `replay_spool`, the writer, and the redactor no
  longer take a `keep_raw` argument.
- `raw` is redacted before it reaches the spool and again at replay, like
  every other field.
- `mem ingest` no longer has `--keep-raw`, with `--transcript` or with
  `--spool`. A script that passes it gets a usage error and exit 2.

Regressions are in `tests/test_cli_capture.py` and
`tests/test_audit_persistence.py`.

### Only a human approves a memory (2026-11-11)

Any agent could approve any quarantined memory through the `review` tool,
including one it had just written from a tool's output. Quarantine
therefore held nothing back from an agent that wanted a memory recalled.
Outside admission, no command showed a human what was quarantined.

- `review` refuses `approve` unless the session's agent is `human`. The
  refusal is a tool error that tells the agent to ask the user to run
  `mem review`. Nothing is written and nothing is queued. `quarantine`,
  `retire`, `confirm`, and `flag_contradiction` stay open to every agent.
- A queued approval is checked again at replay. One from any other agent
  is rejected: it becomes a `capture_error` event and is kept under
  `spool/failed/`, like any record the ledger refuses.
- The actor of a queued review is the agent of the record's session. The
  record's own `actor` field is ignored, so a spool line cannot name
  another actor.
- `mem-mcp` exits at start when `AGENT_MEMORY_AGENT=human`.
- `mem review` is the human's route (`review.py`). `list` prints what
  waits, oldest first: quarantined memories, and active ones that a
  tool's output asked to put in another memory's place. It is a human's
  read of memories that no agent may be shown, so it is not a recall and
  records nothing. `approve` takes one waiting memory, shows it, and asks
  for a yes typed at a terminal. It needs the server, so an approval is
  never queued unseen. `retire` withdraws any memory and is queued when
  the server is down. Both need `--reason`. Both write as agent `human`
  in a session of their own (`via: mem review`, source mode `record`,
  adapter version `mem_cli_v1`), through the same code as the tool.
- The check is made where an approval is asked for: the tool and the
  replay (`lifecycle.check_asker`). `LifecyclePlan.review` does not make
  it, so the approval that an entity merge carries over and admission's
  approvals are written as before.

The rule binds agents that use the tools. Every client holds the same
token, so a process that inserts into the ledger directly can still
record an approval under any actor, and a pseudo-terminal satisfies the
terminal check. Both limits are listed in
[roadmap.md](roadmap.md#known-defects). An approval the tool refuses
leaves no row in the ledger; one rejected at replay leaves its
`capture_error` event.

Regressions are in `tests/test_review.py`, `tests/test_mcp.py`, and
`tests/test_lifecycle.py`.
