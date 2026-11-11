# agent-memory roadmap

What is built, what is planned, what is known to be wrong, and what is
still undecided. The design of what is built is in
[architecture.md](architecture.md). Earlier plans and audits are kept in
[archive/plan-history.md](archive/plan-history.md).

## Status

As of 2026-11-11.

| Work | State | Where it is specified |
|------|-------|-----------------------|
| Phase 0: storage layer | Done | [architecture.md](architecture.md) |
| Phase 0.5: trial of ai-memory | Not started | below |
| Phase 1a: capture path | Done | [phase-1-spec.md](phase-1-spec.md) |
| Phase 1b: in-session tools | Done | [phase-1-spec.md](phase-1-spec.md) |
| Phase 2: harness adapters | Droid hooks and transcript import and Vibe hooks are built. The Vibe transcript adapter, omp, and Amp are not. Hook installation is manual. | below |
| Phase 3: audit tooling | Not started. `mem audit check` exists, but as a preflight for migrations, not as the commands below. | below |
| Phase 4: memory quality | Partly built: the relation graph's first step and lesson admission (migration 0007). | below, [admission-spec.md](admission-spec.md) |
| Phase 5: Markdown projection | Not started | below |
| Lifecycle, coverage, and attribution hardening | Done (migrations 0002 to 0006). No live home has been migrated past 0001. | [logic-hardening-spec.md](logic-hardening-spec.md) |
| Judge experiment | Built; four runs stored | [judge-experiment-spec.md](judge-experiment-spec.md), [run logs](archive/judge-experiments.md) |
| Memory usefulness study | Design only | [memory-utility-evaluation-spec.md](memory-utility-evaluation-spec.md) |
| Parametric consolidation | Research proposal | [parametric-consolidation-spec.md](parametric-consolidation-spec.md) |

## Next approved work

The next approved work should verify Droid capture end to end, then build
and run the local usefulness experiment. Retrieval tuning and wider graph
use follow measured failures. Parametric consolidation remains research
until usefulness and withdrawal/backflow can be measured.

## Phases

The phases that are done are described in
[architecture.md](architecture.md); their original text is in
[archive/plan-history.md](archive/plan-history.md).

**Phase 0.5: trial akitaonrails/ai-memory.** Run it against Droid and omp for a
week of normal work, in parallel with Phase 1 rather than before it. Compare
its capture and recall against this schema, then decide what to adopt
(extraction prompts, dedupe) and what to skip.

**Phase 2: harness adapters.** Two adapter shapes, chosen by what each
harness exposes (see "Verified harness behavior" below):

- *Shell hooks + transcript* (Droid, Vibe). Hooks pipe their stdin JSON to
  `mem event`, which records prompts, tool calls and results, compaction, and
  session boundaries as they happen. Neither harness hands the assistant's
  reply to a hook, so a turn-end hook (`Stop` for Droid, `post_agent` for
  Vibe) runs `mem ingest --transcript <path>` and the transcript's new entries
  are merged by message id. Vibe has no session or prompt hooks at all, so its
  session and user rows also come from the transcript.
- *In-process extension + spool* (omp, Amp). A TypeScript module subscribes to
  the lifecycle events, appends JSONL to `spool/`, and runs `mem ingest` at
  turn end and shutdown. Writing a file keeps the agent loop unblocked and
  survives a server outage. omp delivers everything including compaction;
  Amp delivers every message per turn but no compaction or session-end
  signal, so Amp sessions stay `partial` until that gap is closed or shown
  not to matter.

Junie through MCP alone, marked partial. Redaction and `capture_error`
events live in `mem event` and `mem ingest`, so all adapters get them.

Out of scope: Letta Code. Deferred: Delta (Zed).

**Phase 3: audit tooling.** `mem why <event>`, `mem session <id>`, and
`mem memory <id>` built on the views. An optional Znote notebook over the
Parquet export for browsing.

**Phase 4: memory quality.** Dedupe by `content_hash` and similarity,
contradiction detection (`flag_contradiction`), confirmation and decay of
`confidence`. The quarantined lesson review queue now exists as
`mem admit review`, with seeded discard sampling and feedback into the
fitted approver. `mem admit run` drafts from captured production turns;
auto-approval requires held-out precision of at least 0.8 over 20 scored
writes. See [admission-spec.md](admission-spec.md). The
relation graph's first step is in: `relate`, `graph`, and `mem graph resolve`
(see the "Relation graph" paragraph in
[architecture.md](architecture.md#data-model)). Still open: offline
extraction of triples from `events` with a local model, and whether
multi-hop recall earns dedicated tables.

**Phase 5: Markdown projection.** Render active memories to an IWE-compatible
Markdown wiki (one note per key or topic, with links to sources), in the
spirit of the LLM wiki. The database stays the source of truth; the wiki is a
read-only view.

## Verified harness behavior

Checked against the vendor documentation on 2026-10-25.

| Harness | Events | Payload | Gaps for capture |
|---------|--------|---------|------------------|
| Droid (`~/.factory/hooks.json`) | `SessionStart` (`source`: startup, resume, clear, compact), `UserPromptSubmit`, `PreToolUse`, `PostToolUse`, `PreCompact`, `Stop`, `SubagentStop`, `Notification`, `SessionEnd` | JSON on stdin: `session_id`, `transcript_path`, `cwd`, `permission_mode`, `hook_event_name`, `message_id`; tool hooks add `tool_name`, `tool_input`, `tool_response`. Default timeout 60 s. | No hook carries the assistant reply; read the transcript. `SessionStart` repeats per resume and compaction. |
| Vibe (`~/.vibe/hooks.toml`) | `pre_tool`, `post_tool`, `post_agent` | JSON on stdin: `session_id`, `parent_session_id`, `transcript_path`, `cwd`, `hook_event_name`; tool hooks add `tool_name`, `tool_call_id`, `tool_input`, `tool_status`, `tool_output`, `tool_output_text`, `tool_error`. Default timeout 60 s. | No prompt, session, or compaction hooks; `post_agent` carries no text. Project-level hooks load only in trusted folders. Plugin-declared hooks do not run yet. |
| omp (`~/.omp/agent/hooks/{pre,post}/*.ts`) | `session_start`, `session_switch`, `session_compact`, `session_shutdown`, `agent_start`, `agent_end`, `turn_start`, `turn_end`, `message_start`, `message_end`, `tool_call`, `tool_result`, `tool_execution_*`, `tool_approval_*`, `auto_compaction_*`, `session_stop` | Typed in-process payloads with full message text, tool inputs and results, compaction entries, `session_file`. | None material. Skip the high-volume `message_update` and `tool_execution_update`. |
| Amp (`.amp/plugins/*.ts`) | `session.start`, `agent.start`, `tool.call`, `tool.result`, `agent.end` | In-process: thread id, prompt and message id on `agent.start`; `toolUseID`, `tool`, `input`, `status`, `output` on tool events; every `ThreadMessage` of the turn on `agent.end`. | No compaction or session-end event; `onDispose` runs on unload for about 3 s. |
| Junie | none | MCP tools only | Everything except what the agent chooses to log. |

## Risks and open questions

- Quack shipped in 2.0.0. Its protocol and authorization hook may change;
  `mem smoke` and the server tests are the early warning.
- History can only be corrected by appending. A wrong memory is superseded or
  retired, never edited. Content redaction after the fact needs an explicit,
  rare admin migration.
- The FTS index is maintained on every insert. Watch insert latency as
  `memories` grows; switch to periodic rebuilds if it degrades.
- Tier B and C capture is lossy by design. Audit queries should filter or
  label by `capture_completeness`.
- One token means `agent_id` is self-reported. Quack's authentication hook
  takes three arguments and accepts a Python UDF, so per-harness tokens are
  possible later without protocol changes, and the server already sees each
  connection and its query in `quack_active_connections()`. All harnesses run
  as the same OS user, so this guards against bugs rather than malice; it is
  scheduled with the audit tooling in Phase 3.
- The server caches results for clients (`quack_result_ttl` 3600 s,
  `quack_cache_max_rows` 100 000). Watch memory if many clients poll.
- The writer's event lookup is bounded by the incoming batch and its parent
  references, not session history. Full transcript reimports still parse
  their input; `--since-key` avoids rebuilding records for earlier entries.
- The `generic_assignment` redaction rule is deliberately broad (`token =
  ...`, `password: ...`) and does hit code and config text in tool calls. The
  count lands in `events.redactions`; loosen the rule only with new fixture
  cases.

## Known defects

Verified against the code in the audit of 2026-11 and still open.

- Admission in auto mode can approve the canonical memory of a duplicate
  even when a person quarantined that memory (`admission/ledger.py`,
  `lifecycle.py`). Auto mode is not in use.
- One admission item that fails stops the items and the days after it
  (`admission/run.py`).
- A model server that accepts a connection and never answers is not told
  apart from a slow item. `mem judge draft` and `decide` then wait out
  the timeout twice for every remaining item and skip each one
  (`endpoints.py`, `judge/pipeline.py`).
- `mem judge report --freeze` does not check that the calibration labels
  are complete.
- The Parquet export copies tables one after another while clients write,
  so a snapshot can be inconsistent across tables (`db.py`).
- A caller chooses the `trust_level` of what it remembers, and any agent
  can approve any quarantined memory. Outside admission, no command lets a
  person review quarantined memories (`mcp_server/tools.py`).
- `mem ingest --transcript --keep-raw` has no effect.
- A Droid hook that carries no message or tool id is keyed on the wall
  clock, so a retry of that hook becomes a second event
  (`adapters/droid.py`).
- The listing retrieval of `mem graph resolve` records every relation
  even when `--entity-type` shows the model a subset.

Reported in the same audit and not yet checked against the code: the
hand-maintained `ALL_TABLES` list; lockdown happening after the listener
starts; the experiment-home guard excluding only the default path; the
approver not pinning question, projection, and drafter versions;
`--similarity-floor 0` and `--min-writes 1` weakening the auto-approval
guards; turns still in flight being admitted and then frozen; the Go
helper's stderr being discarded; the drafter's name leaking through
blinding; an egress error body in exception text; no CI configuration.

## Proposed directions

Proposed in the audit of 2026-11. None of these is approved.

- Finish capture: a hook installer, the Vibe transcript adapter, and the
  end-to-end check of Droid capture.
- Build `mem why` and `mem session` (Phase 3) as recorded reads, each a
  new strategy for the recall path.
- Run the memory usefulness experiment.
- Make review depend on the actor: per-harness tokens, and a route for a
  person to review quarantined memories.
- Hybrid recall, once the usefulness experiment shows where text match
  fails.
- A restore that is verified, blobs included.
- A labeler's identity on each label, and a machine-readable report.
- Harden admission before auto mode is used.
- Add CI.

Modules that would gain from deepening, in the order the audit ranked
them: one session driver for labeling and admission review; a ledger
connection with two adapters; auto-approval in one place. The two ranked
above these were deepened on 2026-11-11: the ledger session
(`capture/session.py`) and the model call (`model_call.py`).

## Open decisions

- May an agent approve its own quarantined memory, or assert
  `trust_level = 'user'`?
- Should timeouts in a row stop a judge command? A timeout can be the
  item's doing, so today only an unreachable model server stops one.
- Are the `readout` and `jev` backends and the `actionable` question still
  wanted? May the unreachable v4 fallbacks and `Question.stage` go?
- Should `--freeze` require complete calibration labels? May a run be
  relabeled after a freeze?
- Do Phases 0.5, 3, and 5 stay on this roadmap?
- `--keep-raw` for transcripts: implement or remove?
- Is the Parquet export a backup or a browsing aid?
- What happens to `example-memory-db-schema.sql` and the prototype branch?
