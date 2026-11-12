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
  converts it into capture records. Used by shell-hook harnesses (Droid).
  At a hook event where the harness has written to its transcript what
  no hook carries (for Droid, `Stop` and `SessionEnd`), it also converts
  what the transcript has gained since its last import.
- `mem ingest --harness <id> --transcript <path>`: converts a transcript
  file into capture records. For a session that no hook captured.
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
identity; `capture.ids` hashes those choices. Both Droid capture paths use
the shared builders. A keyed Droid prompt uses the bare
message ID. Without one, its hook uses `prompt:unkeyed:<ts>` and marks
`payload.dedupe` as `"unkeyed"`; it makes no hook/transcript dedupe claim.
Replies keep their entry IDs, thinking appends `#thinking`, and tool calls
and results keep their tool-use IDs. The start of a new session is keyed
on the session ID by its hook and by the transcript alike. Any other
session marker, and the note a `PreCompact` hook leaves, are keyed on the
message ID when the payload has one. Any other hook-only note is keyed on
the time its hook ran.

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
  The write of a transcript's records, from `mem event` or from
  `mem ingest --transcript`, may take up to 300 s instead: giving up on
  it would put the whole transcript in the spool.
- A record that already waits in a session's spool file is not appended
  to it again.
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

Each adapter is a module in `agent_memory/adapters/` with three
functions: `from_hook(payload: dict) -> list[Record]`,
`transcript_due(payload: dict) -> Path | None` (the transcript to read
when this hook runs, if there is one), and `from_transcript(path: Path,
*, since_key, since_line, include_thinking) -> list[Record]`.

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
content, and `openai*` fields are dropped. Droid hooks record `user_msg`,
session boundaries, and notes. Tool calls, their results, and compactions
come from the transcript alone; see "What a Droid hook carries" under "As
built". `capture.identity` gives hook and transcript prompts the same bare
message ID. The former `UserPromptSubmit:` prefix caused duplicate prompts
even when the harness IDs matched. Live comparison of hook `message_id`
and transcript entry `id` is recorded in
[archive/plan-history.md](archive/plan-history.md), separately from the
synthetic tests.

**Vibe**: no adapter. One written from Vibe's documentation was removed;
see "The Vibe adapter is removed" under "As built".

**omp, Amp**: no transcript adapter; their extensions write capture records
directly (Phase 2). `mem ingest --spool` is the whole integration.

**Junie**: none.

### Hook adapter (`mem event`)

The stdin fields every Droid hook sends (`session_id`, `cwd`,
`hook_event_name`, `transcript_path`) map to the `session` block.
`capture.identity` owns the keys. Two are shared with transcript import:
a prompt's message ID, and the session ID for the start of a new session
(`SessionStart` with `source` `startup`). A hook-only event uses the
event name plus the message ID when Droid sends one that places the event
in the session (`SessionStart` with any other source, `SessionEnd`,
`PreCompact`), and otherwise the time the hook ran. Such a `SessionStart`
key also includes its source. `resume` is the only such source recorded.
Mapping:

| hook event | kind | notes |
|---|---|---|
| SessionStart | `session_start` | payload `source`: a `startup`, which `/clear` raises too, is the event the transcript's `session_start` entry records, and a `resume` is the hook's alone; the `sessions` row is created once |
| SessionEnd | `session_end` | payload `reason` and the `message_id` of the session's last message. Droid ends a session left idle, so a session can end more than once. The transcript is imported at this hook event too |
| UserPromptSubmit | `user_msg` | content = prompt |
| PreToolUse | skipped by default | `--record-pre-tool` records a note, for debugging |
| PostToolUse | none | the payload names no call; the transcript records the call and its result |
| PreCompact | `note` | content = `custom_instructions`; payload `trigger` and the anchor `message_id`. The `compaction` event comes from the transcript |
| Stop | none | the transcript is imported at this hook event; see "A Droid transcript is imported by `mem event`" under "As built" |
| SubagentStop | none | the payload carries a subagent's finished task, its name and result, and no id; the transcript of the session that started the task records both |
| Notification | `note` | content = `message`; payload = stdin minus transcript text |

Any hook event not in the table becomes a `note` with the raw payload, so new
harness events are never silently lost.

### CLI shapes

```
mem event  --harness droid [--spool-only] [--keep-raw] [--record-pre-tool]
mem ingest --harness droid --transcript PATH [--since-key KEY] [--include-thinking]
           [--spool-only]
mem ingest --spool [--purge-done DAYS]
mem spool  status            # files, lines, oldest record per agent: pending and rejected
mem spool  retry             # return rejected records to the spool
mem hooks  install   [--file PATH] [--mem PATH] [--keep-raw] [--record-pre-tool]
mem hooks  status    [--file PATH] [--mem PATH]
mem hooks  uninstall [--file PATH]
```

`--since-key` leaves out every entry up to and including the one with
that id. A hook does not pass it: `mem event` reads on from where the
ledger says the last import ended. Without either, the writer's check
for ids the ledger already holds makes a second import correct but
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
Writes `context_links` for the reply. When `answer_event_id` is omitted the
server inserts an `events(kind='assistant_msg', content=NULL,
payload={"placeholder": true})` row and returns its id so the transcript
import can later link the real reply. A captured, server-qualified `cite`
tool result identifies that placeholder by UUID; a complete captured turn
must have one unique reply. The placeholder can live in an MCP-owned
session, while calls, results, and replies must belong to the observed
transcript session. Binding evidence records `placeholder_session_id`.
An explicit `answer_event_id` takes priority. Missing or ambiguous
evidence remains unresolved; there is no nearest-message fallback. This
also gives Tier C harnesses partial `why_answer` coverage without a
captured reply.

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
    attribution.py   # attribution of citations, from captured tool results
  adapters/
    __init__.py    # registry: {"droid": DroidAdapter}
    base.py        # Adapter protocol
    droid.py
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
  `message_id` question is settled. (`PostToolUse` no longer records
  anything; see "What a Droid hook carries" below.)

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
  generation. Review action reasons, memory keys/scopes, and merge inputs
  use the same redactor before constructing their persisted copies.
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

The Droid adapter retains each message's id and the turn its outcome
names, and uses a completed captured turn to reconcile a cite result's
placeholder with a unique reply. Timestamps and "the next assistant
message" cannot stand in for the turn. A channel named on a message
could once choose among replies; Droid names none, and that reading was
removed (see "What an event keeps of its message and turn" below).
Incremental imports keep the full source boundary evidence needed for
reconciliation. Repeated imports do not duplicate observations or
bindings. New ambiguity appends a binding correction rather than
rewriting the original link.

Captured tool identity must identify the configured `agent-memory` MCP
server, not an unrelated tool with a `cite` suffix. Override the default
name with `AGENT_MEMORY_MCP_SERVER_NAME` in the capture environment when
needed. Missing server identity leaves attribution unresolved. An explicit
`answer_event_id` supplied before import is retained on a placeholder and
bound when that exact reply arrives.

Replay resolves dependencies across the claimed spool snapshot rather than
relying on file modification times. This preserves cross-session queued
corrections and review actions, including interleaved files whose records
form an acyclic dependency graph. Interrupted and newly appended files for
the same session get distinct claim paths. As before, a record that fails
validation is rejected, while a transport interruption leaves claimed
files pending.

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
  only when the ledger is unavailable. A failed binding pass is
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
- The actor of a queued review action is the agent of the record's
  session. The record's own `actor` field is ignored, so a spool line
  cannot name another actor.
- `mem-mcp` exits at start when `AGENT_MEMORY_AGENT=human`.
- `mem review` is the human's route (`review.py`). `list` prints the
  memories awaiting approval, oldest first: quarantined memories, and
  active ones that a tool's output asked to put in another memory's
  place. It is a human's read of memories that no agent may be shown, so
  it is not a recall and records nothing. `approve` takes one memory
  awaiting approval, shows it, and asks for a yes typed at a terminal.
  It needs the server, so an approval is never queued unseen. `retire`
  withdraws any memory and is queued when the server is down. Both need
  `--reason`. Both write as agent `human` in a session of their own
  (`via: mem review`, source mode `record`, adapter version
  `mem_cli_v1`), through the same code as the tool.
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

### What a Droid hook carries (2026-11-11)

The hook adapter was written from Factory's documentation, which lists an
optional `message_id` on every hook. A probe recorded the shape of 650
hook payloads from one Droid session in the Factory app, where tools are
called through Script, and compared them with that session's transcript.

| hook | payloads | ids in the payload | what the transcript holds |
|---|---|---|---|
| `PreToolUse` | 554 | none | The call, with its id, written before the hook runs. |
| `PostToolUse` | 93 | none | The call, with its id. The result is written after the hook runs. |
| `PreCompact` | 2 | `message_id` | That id is the anchor message of the `compaction_state` entry that follows, not the entry's own id. |
| `Notification` | 1 | none | Nothing. |

- `PreToolUse` ran for every call, the 460 made inside a Script included.
  `PostToolUse` ran only for top-level calls. The transcript holds every
  call, and a nested one names the call it ran inside.
- Hooks and transcript agree on order: the n-th top-level `PostToolUse`
  is the n-th top-level call in the transcript, for all 93.
- Content does not identify a call. Of the session's 5,198 calls, 524
  have the same name and input as another.
- Parsing that transcript (36 MB, 10,671 records) takes 0.17 s. Writing
  it takes much longer: see "A Droid transcript is imported by
  `mem event`" below for what the import of a session that long takes.

A `PostToolUse` hook was recorded as a `tool_call` and a `tool_result`
keyed on the time the hook ran, because the `tool_use_id` the adapter
looked for is never sent. The transcript import then recorded the same
call under its id. Every top-level call was in the ledger twice, and no
nested call came from a hook at all. `PreCompact` was recorded as a
`compaction` keyed on the anchor message and the transcript's
`compaction_state` as another, so `session_summary` counted two
compactions for one.

- A Droid `PostToolUse` hook records nothing. The transcript is the
  record of tool calls and results, and they reach the ledger when the
  `Stop` hook imports it.
- `PreCompact` is a `note`: a compaction was asked for, by this
  `trigger`, with these instructions, after this message. The
  `compaction` event is the transcript's.
- `Notification`, `SubagentStop`, `PreToolUse` (with
  `--record-pre-tool`), and events the adapter does not know are keyed
  on the time the hook ran, whether or not the payload has a
  `message_id`. One message can raise several of them, and keyed on its
  id the second would have been dropped as a duplicate. The `message_id`
  stays in the event's attribution. (`SubagentStop` has since stopped
  recording anything; see "What Droid's `SubagentStop` hook carries"
  below.)
- Vibe was left unchanged: its `post_tool` payload is documented to
  carry `tool_call_id`. Its adapter has since been removed; see "The
  Vibe adapter is removed" below.

No `UserPromptSubmit`, `Stop`, `SessionStart`, `SessionEnd`, or
`SubagentStop` ran in that session while the probe was recording. The
next two sections cover all of them but `SubagentStop`, which has a
section of its own further down.

Regressions are in `tests/test_adapters.py`,
`tests/test_capture_identity.py`, and `tests/test_cli_capture.py`.

### What Droid's turn and session hooks carry (2026-11-11)

The probe went on to record hooks from four more sessions in the Factory
app, two of them new and two that the app reopened. With the first
session that made 1,167 payloads when these counts were taken. Each row
now also holds the ids and types of the last three transcript entries on
disk when the hook ran.

| hook | payloads | ids in the payload | the transcript when the hook runs |
|---|---|---|---|
| `UserPromptSubmit` | 4 | `message_id` | The prompt is not there yet. It is written afterwards, under that id. |
| `Stop` | 3 | `message_id` | That id is the last entry: the reply (2), or a tool result when the turn ends on a tool call (1). The turn's `agent_turn_outcome` is written afterwards. |
| `SessionStart`, `source` `startup` | 2 | none | The `session_start` entry is there, and its id is the session's id. |
| `SessionStart`, `source` `resume` | 2 | `message_id` | That id is the session's last reply. |

- The first entry of every local transcript (2,379 files) is a
  `session_start` whose id is the session's id, which is also the name
  of the file.
- The app raised a `SessionStart` with `source` `resume` for each of two
  idle sessions it reopened. No `SessionStart` followed any of the three
  compactions.
- The later payloads agree with the table in the section above. None of
  978 `PreToolUse` and 171 `PostToolUse` payloads carries an id, and in
  each of three sessions the n-th `PostToolUse` is the n-th top-level
  call in the transcript, for all 171.

The `SessionStart` hook of a new session was keyed on the time it ran,
and the transcript's `session_start` entry on the session's id, so one
start was two `session_start` events.

- `SessionStart` with `source` `startup` is keyed on the session's id.
  The hook and the import write one event, whichever comes first. A
  `SessionStart` that names no source is keyed as before.
- A resume stays keyed on its `message_id`. A session reopened twice
  with nothing said in between is one `session_start` marker, not two.
- `Stop` still records nothing. An import at `Stop` brings the reply,
  which is on disk by then. It does not bring the turn's outcome, which
  Droid writes after the hook has run. The outcome, and with it the
  binding of a citation to the reply, arrives with the next import: at
  the next `Stop`, or at `SessionEnd` for the last turn of a session.
  See [roadmap.md](roadmap.md#known-defects) for a session that reaches
  neither.

`SessionEnd` ran later the same day and has the next section.
`SubagentStop` did not run while the probe was recording, and neither
did a `SessionStart` after `/clear`. Both ran later; see "What Droid's
`SubagentStop` hook carries" and "What `/clear` starts in Droid" below.

Regressions are in `tests/test_adapters.py` and
`tests/test_capture_identity.py`.

### When a Droid session ends, and what its transcript says of hooks (2026-11-11)

The probe recorded three `SessionEnd` payloads, one from each of three
sessions, which brought it to 1,513 payloads.

| hook | payloads | ids in the payload | the transcript when the hook runs |
|---|---|---|---|
| `SessionEnd` | 3 | `message_id` | That id is the session's last message: the reply (2), or a tool result when the last turn ended on a tool call (1). The outcome of the last turn is there. |

- Each ran 30 minutes after its session was last used, with `reason`
  `other`: twice after the app reopened a session and nothing was said
  in it, once after a spec handoff took the work to a new session. The
  payload also has `message_count` and `session_duration_ms`.
- An end is therefore not final. Droid ends a session that sits idle,
  and a resume starts it again.
- The `SessionStart` of the session the handoff created names the one it
  came from as `previous_session_id`, which stays in the event's payload.
- The adapter needed no change. `SessionEnd` is a `session_end` keyed on
  its `message_id`. A session that ends twice with nothing said in
  between has one `session_end`, at the time of the first, as a session
  resumed twice has one marker. `session_summary.ended_at` is the latest
  `session_end`, and events can follow it.
- By `SessionEnd` the outcome of the last turn is on disk, so an import
  then brings what the import at the last `Stop` could not.

With a hook registered, Droid also writes an entry to the transcript for
every hook command it runs: a `message` with role `user`, visibility
`user_only`, and no content. It holds the event's name, the command, its
exit code, what it printed on stdout and stderr, and when it started and
ended. Each of the 1,513 payloads has one.

- The entry names what the payload does not. For `PreToolUse` and
  `PostToolUse` it has the id of the tool call (`hookToolCallId`), and
  for all 1,485 that id is a call in the same transcript. For an event
  whose payload has a `message_id`, the entry's `hookParentId` is that
  id.
- The import records nothing for these entries. The one for a `Stop`
  hook falls between a reply and the outcome of its turn and does not
  separate the two.
- What a hook command prints is kept in the session's transcript, so a
  hook that has nothing wrong to report should print nothing.
- A hook costs a process and a transcript entry each time it runs.
  `mem event` takes about 0.3 s, which a `PreToolUse` or `PostToolUse`
  hook that records nothing would spend on every tool call.

Regressions are in `tests/test_adapters.py`.

### The Vibe adapter is removed (2026-11-11)

The Vibe adapter is gone: `adapters/vibe.py`, the names `vibe` and
`mistral-vibe` in the registry, and its tests. `mem event --harness
vibe` exits 1 and `mem ingest --harness vibe` exits 2, and each names
the harness an adapter exists for.

It had been written from Vibe's hook documentation and never run against
a payload Vibe sent. Droid's hook mapping was written the same way, and
the first real payloads showed it recording a tool call, a compaction,
and the start of a session twice. Nothing suggested the Vibe adapter
would do better. It could not read a transcript, and no session had been
captured through it.

- `mistral-vibe` stays a registered agent, as migration 0001 made it, so
  an agent in Vibe can still use the `mem-mcp` tools.
- An adapter for any harness starts from payloads and transcripts
  recorded from that harness, as the three sections above did for Droid.
- Droid is now the one harness with an adapter. The `Adapter` protocol
  and the registry stay, though a seam with one adapter is a
  hypothetical one: `--harness` is part of every hook command, and a
  second adapter is the next candidate in
  [roadmap.md](roadmap.md#phases).

Regressions are in `tests/test_adapters.py` and
`tests/test_cli_capture.py`.

### A Droid transcript is imported by `mem event` (2026-11-11)

Capture needed a second command beside the hook: `mem ingest
--transcript`, to be run from a `Stop` hook script that nobody had
written. That command prints its summary on stdout and exits 2 on a
usage error, neither of which a hook may do. It also gave the ledger the
2 s a hook gives it, so the import of a long session gave up while its
write was under way and put the whole transcript in the spool, at every
turn.

- `mem event` imports the transcript itself at the hook events where
  Droid has written to it what no hook carries: `Stop`, when the reply
  and the tool calls of the turn are on disk, and `SessionEnd`, when the
  outcome of the last turn is. The adapter says which hook events those
  are (`transcript_due`), so the command knows nothing of Droid. One
  hook command now does all of capture.
- The read goes on from where the last import of the session ended. The
  ledger says where that was: the `cursor_end` and `range_end` of the
  session's latest `transcript_import` observation.
- An earlier read is found by the line it ended on, not by the id alone.
  A read that ends on an outcome ends on the id of its turn, which is
  also the id of the turn's prompt, up to thousands of lines above.
- If the file does not hold that id at that line, the whole file is
  read. Ids are deterministic, so that gives the ledger nothing twice.
- When the ledger does not answer within the hook's 2 s, the import of
  the session that waits in the spool says where to read on from, and
  what is read goes to the spool. With no import waiting there, the
  whole transcript is read into the spool, once for an outage. A ledger
  that did not take the hook's own record is not asked a second time.
- The write of an import may take up to 300 s, from `mem event` and from
  `mem ingest --transcript`. A hook's own record keeps the 2 s.
- A record that already waits in a session's spool file is not appended
  to it again. Two imports of a 41.9 MB transcript while the server was
  down had put 37 MB in the spool twice.
- Droid gives the first entry of a transcript, its `session_start`, no
  timestamp (60 of 60 transcripts checked). The adapter gave that entry
  the time of the read, so an import said its session started when it
  was imported, and two reads of one file gave two different first
  records. The entry now takes the time of the first entry that has one,
  less a millisecond, as an `agent_turn_outcome` takes the time of the
  entry before it plus one.
- A transcript that cannot be read does not cost the hook its own
  record. That record is delivered first, and `mem event` then exits 1
  with one line on stderr.
- The hooks import without thinking blocks. `mem ingest
  --include-thinking` on the same transcript adds them afterwards.

Measured as the hook command, each run a new process, on a real
transcript of 41.9 MB and 13,515 lines:

| hook | server up | server down |
|---|---|---|
| the session's first `Stop` | 8.75 s | 4.63 s, and 37 MB in the spool |
| `Stop` with nothing new | 0.55 s | 0.57 s |
| `Stop` with 300 more lines | 0.83 s | 0.72 s |
| `SessionEnd` | 0.70 s | not measured |
| `Notification` | 0.47 s | not measured |

- When the server came back, the replay inserted 185 records and found
  11,817 already present. The ledger then also holds an observation that
  covers the whole file, from the read made during the outage.
- Local transcripts have a median size of 0.4 MB and a 99th percentile
  of 6.1 MB. The largest is 79 MB. Its first import was not run; at the
  rate above it would take about 17 s of Droid's 60 s hook timeout.
- A transcript of 999 events (2.9 MB) takes `mem ingest --transcript`
  1.0 s the first time and 0.65 s when the ledger already holds it.

Regressions are in `tests/test_cli_capture.py` and
`tests/test_adapters.py`.

### `mem hooks` writes capture into Droid's hooks file (2026-11-11)

Capture reached the ledger only through hooks a person wrote by hand,
and the pattern the README gave for them could not work as written. A
hook runs with Droid's environment and working directory, so a bare
`mem` with the default home is not the `mem`, the home, or the port the
person set up.

- `mem hooks install` writes one command at each hook event where the
  Droid adapter records something or reads the transcript:
  `SessionStart`, `SessionEnd`, `UserPromptSubmit`, `PreCompact`,
  `Notification`, and `Stop`. The adapter names them
  (`DroidAdapter.hook_events`), so the installer cannot disagree with
  it. `--record-pre-tool` adds `PreToolUse`. `SubagentStop` was among
  them until the adapter stopped recording anything there.
- The command is `mem event --harness droid`, written with the absolute
  path of `mem`, `--home`, and `--port`, and with `--keep-raw` or
  `--record-pre-tool` when asked for. The `mem` is the one installed
  beside the running Python, or the one `--mem` names.
- A hook is mem's only when its whole command is `mem ... event ...`,
  with no shell operator in it. `install` replaces those, leaves one at
  each hook event above, and takes them out of any other. Every other
  hook stays as it was, in its place.
- A hook that runs `mem event` inside a longer command (`... || true`,
  `cd ... && ...`) is someone else's. It is left alone and reported on
  stderr, because capture then runs twice at that hook event.
- The file is written in one step (a temporary file, then a rename),
  through a link to it, with the mode it had or 0600 if it is new, and
  not at all when nothing would change. Its hooks and their order are
  kept. Its line layout is not: the file is written as indented JSON.
- `install` refuses, and changes nothing, when the file is not JSON or
  not in the shape Droid documents, and when its directory does not
  exist. It also refuses to create `hooks.json` while the
  `settings.json` beside it holds hooks or the earlier
  `hooks/hooks.json` exists. Droid reads those only while there is no
  `hooks.json`, so a new one would switch them off.
- `mem hooks status` prints, for each hook event, whether mem's hook is
  installed, missing, different, or repeated. It exits 1 unless the
  command for the home and port in use runs once at each hook event
  capture needs and at no other. `mem hooks uninstall` takes mem's hooks
  out and leaves the rest.
- `install` warns when the home holds no ledger yet, where a hook keeps
  what it records in the spool, and when `AGENT_MEMORY_HOST` is set,
  which the hook command does not carry.
- The installer is Droid's. The `Adapter` protocol names no hook events,
  and nothing here is written for a second harness.

The installed commands were then run as Droid runs them: taken from the
hooks file, through a shell, with `/` as the working directory. The
hooks file was a copy of a real one that held nine other hooks, the
transcript a real one of 0.94 MB, and the home a throwaway. The payloads
were written by hand in the shapes the probe recorded, and the
`SubagentStop` one from Droid's documentation. No session in a running
Droid has been captured through installed hooks yet.

- With the server up, every hook exited 0, printed nothing, and took
  0.38 to 0.61 s. The ledger held 219 events and one `transcript_import`
  observation, of lines 1 to 338.
- With the server down, a `Notification` and the first `Stop` went to
  the spool, and each printed one line on stderr, which Droid keeps in
  the transcript's entry for the hook. A second `Stop` had nothing to
  add and printed nothing. The replay inserted one record and found 214
  already present.
- `uninstall` left the file with the content and the key order it had
  before `install`, and the nine other hooks in it.
- `mem audit check` found no error in that ledger and gave the session
  three warnings: `capture_coverage_gap`, `capture_exclusions`, and
  `capture_transcript_partial`. See
  [roadmap.md](roadmap.md#known-defects).

Droid's documentation says a session reads its hooks when it starts. A
session that was running when `hooks.json` changed ran the new hooks ten
seconds later (the Factory app, seen once). `install` and `uninstall`
therefore say that a running session may use the change at once or keep
the hooks it started with.

Regressions are in `tests/test_hooks.py` and `tests/test_adapters.py`.

### What an event keeps of its message and turn (2026-11-11)

The Droid adapter gives each record an attribution: the transcript
message it was made from and, once an outcome names it, its turn. An
import reads the attributions of the whole transcript to find the reply
a `cite` was made for. The writer also stores each record's attribution
with its event, in the payload under `capture_attribution`. Nothing
reads the stored copy. It is kept for a person who asks why a reply
was, or was not, bound to a citation. No test fixed what it held: with
the write taken out, all 1075 tests of that day passed. Tests fix it
now.

An event made from a transcript message holds these fields:

| Field | What it says |
|-------|--------------|
| `message_id` | The id of the transcript entry. A tool call and a tool result hold the id of the message that carried them. |
| `has_tool_calls` | Whether the message held a tool call. |
| `turn_id` | The turn the message is in. Absent until an outcome has named the turn. |
| `turn_complete` | Whether that turn ended in a reply. Present with `turn_id` only. |

An event made from any other entry holds none, with one exception: a
compaction between a prompt and the outcome of its turn holds the two
turn fields. A hook's event holds the `message_id` its payload carried,
or nothing.

What Droid sends, counted on 2026-11-11:

- None of the 33,433 message entries in the 60 most recent local
  transcripts names a turn or a channel. All 207 outcomes in them name
  their turn, 145 of them with the `resultKind` `text` that
  `turn_complete` reads.
- None of the 3,044 hook payloads the probe recorded carries `turn_id`,
  `channel`, or `final_message_id`. `UserPromptSubmit`, `Stop`,
  `PreCompact`, `SessionEnd`, and a resumed `SessionStart` carry
  `message_id`.

For Droid, then, only an outcome names a turn, and a hook payload names
a message or nothing. The adapter reads no more than that. Until later
the same day it also read a turn or a channel named on a message, and
`turn_id`, `channel`, and `final_message_id` in a hook payload. That
reading had met no real input and was removed, and with it the choice
of a reply by its channel.

The stored copy is what the adapter knew when the event was first
written, and no event is rewritten. Droid writes a turn's outcome after
its `Stop` hook has run, so the import at `Stop` writes the turn's
messages before anything has named their turn. An event captured
through the hooks therefore names no turn, also once the outcome is in
the ledger. A prompt the `UserPromptSubmit` hook wrote keeps only the
hook's `message_id`. Read an event without a `turn_id` as "not known
when this was written". The turn is stored only when the import that
first writes a message also reads the outcome of its turn, as `mem
ingest --transcript` of a finished session does.

An event written before the removal can hold more: a null `turn_id`
with a false `turn_complete` where the turn was not known, a `channel`
that is always null, and a `turn_boundary` of `agent_turn_outcome`
wherever a turn is named. That false `turn_complete` says nothing of
the turn.

Binding a reply to a citation is not affected. Each import works out
its evidence from the whole transcript, where the outcome has named the
turn, and stores it with the import's observation
(`attribution_evidence`). The removal left the rest of an import as it
was: the 60 transcripts give the same records and the same observation
before and after it, and synthetic turns that cite give the same
evidence. The adapter version (`droid_transcript_v2`) and the
attribution method (`captured_cite_turn_v1`) therefore kept their
names, and a second import of a transcript adds no observation.

Regressions are in `tests/test_cli_capture.py` and
`tests/test_adapters.py`.

### What Droid's `SubagentStop` hook carries (2026-11-11)

The probe recorded three `SubagentStop` payloads from two sessions in
the Factory app, which brought it to 3,691 payloads when these counts
were taken. Each ran in the session that had started a subagent's task
in the background, when that task finished.

| hook | payloads | ids in the payload | what the transcript holds |
|---|---|---|---|
| `SubagentStop` | 3 | none | A `message` with the id `task-completion:<id>` and role `user`, written 0.05 to 13 s after the hook ran. It holds the task's id, kind, and name, and its result. |

- The payload has `task_name` and `task_result`, with `session_id`,
  `transcript_path`, `cwd`, `permission_mode`, and `stop_hook_active`.
  The session and the transcript are those of the session that started
  the task. Nothing names the task or a message.
- `task_name` is as long as the `description` the task was started
  with, in all three. `task_result` was 3,540 to 12,379 characters.
- The probe keeps the length of a text and not the text, so that the
  transcript message holds the result whole is read from the lengths.
  The message is longer than `task_result` by 483 characters for a
  `worker` with a name of 30, by 485 for one with a name of 31, and by
  489 for an `explorer` with a name of 32: a header that holds the name
  twice and the kind once.
- The message is written when the session next takes input. One hook
  ran in the middle of a turn, and the message followed 13 s later,
  after the turn's next tool call had returned.

The adapter, written from Droid's documentation, recorded a `note`
keyed on the time the hook ran, with `task_name` and the whole
`task_result` in its payload. The import records the transcript message
as a `user_msg`. The result would have been in the ledger twice, with
nothing to tie the note to the message.

- A Droid `SubagentStop` hook records nothing. The transcript is the
  record of a subagent's finished task, as it is of a tool call.
- `mem hooks install` writes no hook at `SubagentStop` and takes out one
  an earlier install wrote there. `mem hooks status` reports one there
  as not needed.
- None of the 3,691 payloads carries `turn_id`, `channel`, or
  `final_message_id`, the three from `SubagentStop` included.

A `SessionStart` after `/clear` was recorded later and has the next
section.

Regressions are in `tests/test_adapters.py` and `tests/test_hooks.py`.

### What `/clear` starts in Droid (2026-11-11)

`/clear` was typed twice in the Factory app, each time followed by one
message. The probe recorded the second. It had reached its size limit
and stopped writing before the first, which is known from its transcript
alone. The second brought the probe to 3,719 payloads when these counts
were taken.

| hook | payloads | ids in the payload | the transcript when the hook runs |
|---|---|---|---|
| `SessionStart` after `/clear`, `source` `startup` | 1 | none | The `session_start` entry is there and nothing else, and its id is the session's id. |

- Each `/clear` started a session with a new id and its own transcript.
  The `session_start` entry names no other session: it has no `parent`
  and no `forkedAtMessageId`.
- The payload names the new session. It has the fields every recorded
  `startup` payload has (`session_id`, `transcript_path`, `cwd`,
  `hook_event_name`, `permission_mode`, `source`, `CLAUDE_ENV_FILE`) and
  no others: no `message_id` and no `previous_session_id`.
- No `SessionEnd` ran in the four minutes after it, for any session.
- No payload says `clear`. The 15 `SessionStart` payloads are 9
  `startup` and 6 `resume`, and the 10 `SessionEnd` payloads all give
  `other` as their `reason`.

The adapter needed no change. A start after `/clear` is a new session's
start, so the hook and the import write one event, keyed on the
session's id. The ledger holds nothing that ties the new session to the
one `/clear` left, because neither the payload nor the transcript does.

Every one of the 3,719 payloads came from the Factory app: the
environment of each hook held the app's `FACTORY_DESKTOP_*` variables.
Droid in a terminal has not been recorded, and its `/clear` may say
something else; see [roadmap.md](roadmap.md#known-defects).

The regression is
`test_a_session_start_is_one_event_when_a_hook_and_the_transcript_report_it`
in `tests/test_capture_identity.py`.
