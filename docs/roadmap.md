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
| Phase 2: harness adapters | Droid hooks and transcript import are built and checked against real payloads. Vibe, omp, and Amp have no adapter; one for Vibe, written from its documentation, was removed on 2026-11-11. `mem hooks install` writes Droid's hooks; no session in a running Droid has been captured through them yet. | below |
| Phase 3: audit tooling | Not started. `mem audit check` exists, but as a preflight for migrations, not as the commands below. | below |
| Phase 4: memory quality | Partly built: the relation graph's first step and lesson admission (migration 0007). | below, [admission-spec.md](admission-spec.md) |
| Phase 5: Markdown projection | Not started | below |
| Lifecycle, coverage, and attribution hardening | Done (migrations 0002 to 0006). No live home has been migrated past 0001. | [logic-hardening-spec.md](logic-hardening-spec.md) |
| Judge experiment | Built; four runs stored | [judge-experiment-spec.md](judge-experiment-spec.md), [run logs](archive/judge-experiments.md) |
| Memory usefulness study | Design only | [memory-utility-evaluation-spec.md](memory-utility-evaluation-spec.md) |
| Parametric consolidation | Research proposal | [parametric-consolidation-spec.md](parametric-consolidation-spec.md) |

## Next approved work

The next approved work should finish verifying Droid capture end to end,
then build and run the local usefulness experiment. Droid's hooks were
checked against real payloads on 2026-11-11, all but `SubagentStop`.
`mem hooks install` and the hooks it writes were run against a copy of a
real hooks file and a real transcript. No session in a running Droid has
yet been captured through installed hooks. Retrieval tuning and wider
graph use follow measured failures.
Parametric consolidation remains research until usefulness and
withdrawal/backflow can be measured.

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
  `mem event`, which records what the transcript lacks or holds only
  later: prompts, notifications, and session boundaries for Droid, and tool
  calls and results for Vibe. Neither harness hands the assistant's reply
  to a hook, and a Droid hook does not name the tool call it reports. The
  transcript is therefore imported at the turn-end hook event (`Stop` for
  Droid, `post_agent` for Vibe), which brings in replies, Droid's tool
  calls and compactions, and merges entries a hook already recorded by
  their id. For Droid, `mem event` does that import itself, at `Stop` and
  at `SessionEnd`, and `mem hooks install` writes the hooks. Vibe has no
  session or prompt hooks at all, so its session and user rows also come
  from the transcript. Only the Droid half is built.
  What is said of Vibe here and in the table below comes from its
  documentation, not from a payload it sent.
- *In-process extension + spool* (omp, Amp). A TypeScript module subscribes to
  the lifecycle events, appends JSONL to `spool/`, and runs `mem ingest` at
  turn end and shutdown. Writing a file keeps the agent loop unblocked and
  survives a server outage. omp delivers everything including compaction;
  Amp delivers every message per turn but no compaction or session-end
  signal, so Amp sessions stay `partial` until that gap is closed or shown
  not to matter.

Junie through MCP alone, marked partial. Redaction and `capture_error`
events live in `mem event` and `mem ingest`, so all adapters get them.

*One harness with an adapter.* Decided on 2026-11-11: Droid is the one
harness with an adapter until another harness is in daily use. A second
adapter is written from payloads and transcripts recorded from that
harness, as Droid's was corrected from its own.

The next candidate is the runtime of OpenRouter's Ori, which was looked
at as a way to capture several harnesses at once (Ori 0.15.6, its help
and its local stores; no session was run). Ori uses "harness" for two
things:

- A *launcher* (`ori droid`, `ori pi`, and 15 more) starts the harness's
  own CLI with OpenRouter models and credentials and passes its
  arguments through. Nothing in Ori's help or stores shows it keeping a
  record of a session started this way: the session stays the
  harness's, with its own hooks and transcript. A Droid session started
  with `ori droid` should be captured as any Droid session is. Not
  checked: that Droid's hooks run unchanged there.
- A *runtime harness* is what Ori's own agent (`ori code`) runs a turn
  with. Ori records every session its runtime runs in one format,
  whichever registered harness ran the turn, under `.ori/logs/sessions`
  and as a JSONL event stream (`ori code --output jsonl`). One is
  registered by default, `ori` itself, and "external launcher names are
  not automatically runtime harnesses."

So an Ori adapter would capture the work done in `ori code` and nothing
else, and it would make an OpenRouter login a condition of that capture.
Reading its real records takes a billable model call, which has not
been made.

Out of scope: Letta Code. Deferred: Delta (Zed).

**Phase 3: audit tooling.** `mem why <event>`, `mem session <id>`, and
`mem memory <id>` built on the views. An optional Znote notebook over the
Parquet export for browsing.

**Phase 4: memory quality.** Dedupe by `content_hash` and similarity,
contradiction detection (`flag_contradiction`), confirmation and decay of
`confidence`. The quarantined lesson review queue now exists as
`mem admit review`, with seeded discard sampling and feedback into the
fitted approver. `mem review` lists every memory that waits and lets a
human approve or retire one. `mem admit run` drafts from captured
production turns; auto-approval requires held-out precision of at least
0.8 over 20 scored writes. See [admission-spec.md](admission-spec.md). The
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

Checked against the vendor documentation on 2026-10-25. Droid's hooks
were checked against real payloads on 2026-11-11, all but
`SubagentStop`. `PreToolUse`, `PostToolUse`, and `Notification` carry no
`message_id` and no id for a tool call. `PreCompact` carries the
`message_id` of the compaction's anchor message. `PostToolUse` does not
run for a call made inside a Script. `Stop` runs after the reply is
written and before the turn's outcome is. The `SessionStart` of a new
session carries no `message_id`; a resume carries that of the session's
last reply. `SessionEnd` carries that of the session's last message and
runs 30 minutes after a session was last used, so a session can end
more than once. Droid writes an entry to the transcript for every hook
command it runs. A session that was running when `hooks.json` changed
ran the new hooks ten seconds later, where Droid's documentation says a
session reads its hooks when it starts (seen once, in the Factory app).
See "What a Droid hook carries", "What Droid's turn and session hooks
carry", "When a Droid session ends, and what its transcript says of
hooks", and "`mem hooks` writes capture into Droid's hooks file" in
[phase-1-spec.md](phase-1-spec.md).

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
  scheduled with the audit tooling in Phase 3. Until then the rule that only
  a human approves a memory is kept by the tools, not by the server.
- The server caches results for clients (`quack_result_ttl` 3600 s,
  `quack_cache_max_rows` 100 000). Watch memory if many clients poll.
- The writer's event lookup is bounded by the incoming batch and its parent
  references, not session history. Every import parses the whole
  transcript. A hook, and `mem ingest --since-key`, hand the writer only
  the entries after an earlier import.
- The `generic_assignment` redaction rule is deliberately broad (`token =
  ...`, `password: ...`) and does hit code and config text in tool calls. The
  count lands in `events.redactions`; loosen the rule only with new fixture
  cases.

## Known defects

Verified against the code in the audit of 2026-11 and still open.

- Admission in auto mode can approve the canonical memory of a duplicate
  even when a human quarantined that memory (`admission/ledger.py`,
  `lifecycle.py`). Auto mode is not in use.
- One admission item that fails stops the items and the days after it
  (`admission/run.py`).
- A model server that accepts a connection and never answers is not told
  apart from a slow item. `mem judge draft` and `decide` then wait out
  the timeout twice for every remaining item and skip each one
  (`endpoints.py`, `judge/pipeline.py`).
- `mem judge report --freeze` does not check that every backend has
  judged every calibration item. A backend that skipped items gets its
  thresholds from the ones it judged, and nothing says so
  (`judge/report.py`).
- A caller chooses the `trust_level` of what it remembers and nothing
  checks it. A memory an agent writes with `user` trust is recalled
  without review (`mcp_server/tools.py`).
- Only a human approves a memory, but the tools keep that rule and the
  server does not. Every client holds the same token, so a process that
  inserts into the ledger directly can record an approval as `human`.
  `mem review approve` asks for a terminal, which a pseudo-terminal
  satisfies. `mem admit review` makes no such check: when its input is
  piped it falls back to the `--plain` prompts, reads the answers, and
  records a Yes as an approval by `human` (`lifecycle.py`, `review.py`,
  `admission/review.py`).
- A memory that admission wrote and a human approves or retires with
  `mem review` gives the fitted approver no example. Only
  `mem admit review` records one (`review.py`).
- Droid's `SubagentStop` payload, and a `SessionStart` after `/clear`,
  have not been checked against real ones (`adapters/droid.py`).
- No session in a running Droid has been captured through installed
  hooks. `mem hooks install` and the hooks it writes were run against a
  copy of a real hooks file, with payloads written by hand (`hooks.py`).
- A Droid turn that never reaches `Stop` (a crash, a killed session) is
  in the ledger only after the session's next `Stop` or `SessionEnd`, or
  after someone imports its transcript. Droid writes a turn's outcome
  after its `Stop` hook has run, so the outcome of a session's last turn
  arrives at `SessionEnd`, 30 minutes after the session was last used.
  Until then, and for good if `SessionEnd` never runs, the judge reads
  that turn as cut off when the session ended (`closed_by` is
  `session_end`), not as completed. Not checked: whether quitting Droid
  raises `SessionEnd`.
- The first import of a long session runs inside its `Stop` hook and
  takes seconds: 8.75 s for a transcript of 41.9 MB, and at that rate
  about 17 s for the largest local one, 79 MB. Droid's hook timeout is
  60 s. Not checked: what Droid shows while a hook runs that long, and
  what a hook killed at the timeout leaves behind.
- A whole-file import of a transcript the ledger already holds redacts
  every record before the writer finds it present: 4 s for 11,217
  records. A hook avoids that by reading on from the last import.
  `mem ingest --transcript` without `--since-key` does not, and neither
  does a hook whose transcript no longer matches its last import
  (`capture/writer.py`).
- After an outage the first `Stop` reads the whole transcript into the
  spool, once: 4.6 s and 37 MB for a transcript of 41.9 MB. The spool
  does not know what the ledger already holds (`capture/pipeline.py`).
- `mem audit check` warns about every Droid session captured through
  the hooks. Each import lists what it leaves out, so
  `capture_exclusions` and `capture_transcript_partial` always apply,
  and a hook's event later than the last transcript entry, such as the
  `session_end`, raises `capture_coverage_gap`. No Droid session can
  read as fully covered (`audit.py`, `adapters/droid.py`).
- `session_summary.started_at` is the time of the first record the
  ledger received for a session. When that record is a hook that is not
  the session's start (a prompt, or the resume of a session begun before
  its hooks were installed), it is later than the start the transcript
  records, which the view shows only inside `effective_metadata`
  (migration 0003). Every session that is running when the hooks are
  installed will show it: one whose first record was a prompt hook at
  16:19 has a transcript that starts at 14:12.
- A hook with no id of its own is keyed on the time it ran, so a hook
  command that runs twice records each notification twice.
  `mem hooks install` leaves one hook of its own at each hook event. A
  hook that runs `mem event` inside a longer command is only reported
  (`adapters/droid.py`, `hooks.py`).
- `mem event` takes 0.3 s to start, and about half of that loads the
  `mcp` library, which capture does not use: the CLI imports it with the
  graph and review commands (`cli.py`, `mcp_server/tools.py`). A hook
  runs at every prompt, notification, and turn end.
- A hook that spools prints one line on stderr, and Droid keeps what a
  hook prints in the transcript. During an outage each hook adds that
  line to the session's transcript (`cli.py`).
- The command a hook runs names the home and the port. It does not
  carry `AGENT_MEMORY_HOST` or `AGENT_MEMORY_DEADLINE`, which a hook
  sees only if Droid's environment has them (`hooks.py`).
- The hooks import a transcript without its thinking blocks, and
  `mem hooks install` has no flag to ask for them. `mem ingest
  --include-thinking` adds them by hand.
- The transcript import drops which Script call a nested tool call ran
  inside (`script_execution.outer_tool_use_id`), so the ledger cannot
  tell a nested call from a top-level one (`adapters/droid.py`).

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

- Finish capture: a session captured end to end through hooks installed
  in a running Droid, with `SubagentStop` and a `SessionStart` after
  `/clear` checked against real payloads. The installer and the import
  at `Stop` and `SessionEnd` were built on 2026-11-11.
- Keep `mem event` from loading what capture does not use, which would
  about halve a hook's start-up.
- An adapter for Ori's runtime, as the second harness (see "One harness
  with an adapter" under Phase 2).
- Build `mem why` and `mem session` (Phase 3) as recorded reads, each a
  new strategy for the recall path.
- Run the memory usefulness experiment.
- Per-harness tokens, so that the server and not only the tools keeps
  the rule that only a human approves a memory.
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

- Should `mem admit review` and `mem judge label` take their answers
  from a terminal only, as `mem review approve` does? Both read piped
  answers today.
- Should a refused approval leave a row in the ledger? Today one the
  tool refuses leaves none.
- Should `session_summary.started_at` take the start the transcript
  records? That needs a new view, in migration 0008.
- Should a Droid session captured from start to end be able to read as
  fully covered, and what would have to be true of it?
- Should timeouts in a row stop a judge command? A timeout can be the
  item's doing, so today only an unreachable model server stops one.
- Are the `readout` and `jev` backends and the `actionable` question still
  wanted? May the unreachable v4 fallbacks and `Question.stage` go?
- May a run be relabeled after a freeze? Should `--freeze` also ask for
  every backend's verdicts on the calibration items?
- Do Phases 0.5, 3, and 5 stay on this roadmap?
- Is the Parquet export a backup or a browsing aid?
- What happens to `example-memory-db-schema.sql` and the prototype branch?
