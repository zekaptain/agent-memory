# agent-memory roadmap

What is built, what is planned, what is known to be wrong, and what is
still undecided. The design of what is built is in
[architecture.md](architecture.md). Earlier plans and audits are kept in
[archive/plan-history.md](archive/plan-history.md).

## Status

As of 2026-11-14.

| Work | State | Where it is specified |
|------|-------|-----------------------|
| Phase 0: storage layer | Done | [architecture.md](architecture.md) |
| Phase 0.5: trial of ai-memory | Not started | below |
| Phase 1a: capture path | Done | [phase-1-spec.md](phase-1-spec.md) |
| Phase 1b: in-session tools | Done | [phase-1-spec.md](phase-1-spec.md) |
| Phase 2: harness adapters | Droid hooks and transcript import are built and checked against real payloads. Vibe, omp, and Amp have no adapter; one for Vibe, written from its documentation, was removed on 2026-11-11. `mem hooks install` writes Droid's hooks, and sessions in a running Droid have been captured through them since 2026-11-12. | below |
| Phase 3: audit tooling | Not started. `mem audit check` exists, but as a preflight for migrations, not as the commands below. | below |
| Phase 4: memory quality | Partly built: the relation graph's first step and lesson admission (migration 0007). | below, [admission-spec.md](admission-spec.md) |
| Phase 5: Markdown projection | Not started | below |
| Lifecycle, coverage, and attribution hardening | Done (migrations 0002 to 0006, 0008 for a session's start, and 0009 for its title, caller, and harness version). The production home is at 0008 and the experiment home at 0001. The migrations through 0008 ran on copies of both homes on 2026-11-11, and 0009 on a copy of the experiment home on 2026-11-13. | [logic-hardening-spec.md](logic-hardening-spec.md) |
| Judge experiment | Built; four runs stored | [judge-experiment-spec.md](judge-experiment-spec.md), [run logs](archive/judge-experiments.md) |
| Memory usefulness study | Design only | [memory-utility-evaluation-spec.md](memory-utility-evaluation-spec.md) |
| Parametric consolidation | Research proposal | [parametric-consolidation-spec.md](parametric-consolidation-spec.md) |

## Next approved work

The next approved work should finish verifying Droid capture end to end,
then build and run the local usefulness experiment. Droid's hooks were
checked against real payloads on 2026-11-11, all of them recorded in the
Factory app. On 2026-11-12 the first 120 sessions were captured through
installed hooks, in the app, in a terminal, and with no interface, and
each agreed with its transcript. Still to see: `/clear` and the end of a
session in a terminal, and what Droid shows during a long first import.
Retrieval tuning and wider graph use follow measured failures.
Parametric consolidation remains research until usefulness and
withdrawal/backflow can be measured.

Also agreed on 2026-11-13, from the measurements under "Growth" below,
and in this order: give the server a memory limit and remove old files
from `spool/done/` on a schedule; then decide whether old events ever
leave the ledger (see "Open decisions"). Two steps agreed before these
are done: the binding check after a write no longer reads every event,
and the server keeps two Parquet snapshots where it kept 14.

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
`confidence`. Admission review of the memories that lessons became now
exists as `mem admit review`, with a seeded sample of the discarded
lessons and feedback into the fitted approver. `mem review` lists every
memory awaiting approval and lets a human approve or retire one.
`mem admit run` drafts from captured production turns; auto mode
requires held-out precision of at least 0.8 over 20 scored writes. See
[admission-spec.md](admission-spec.md). The
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
were checked against real payloads on 2026-11-11, all of them recorded
in the Factory app. `PreToolUse`, `PostToolUse`, and
`Notification` carry no `message_id` and no id for a tool call.
`SubagentStop` carries the name and the result of a subagent's finished
task and no id, and the session that started the task gains a transcript
message with both. `PreCompact` carries the `message_id` of the
compaction's anchor message. `PostToolUse` does not
run for a call made inside a Script. `Stop` runs after the reply is
written and before the turn's outcome is. The `SessionStart` of a new
session carries no `message_id`; a resume carries that of the session's
last reply. `/clear` starts a new session, with a new id and its own
transcript, and its `SessionStart` says `startup`. `SessionEnd` carries
the `message_id` of the session's last message and runs 30 minutes
after a session was last used, so a session can end more than once.
Droid writes an entry to the transcript for every hook
command it runs. A session that was running when `hooks.json` changed
ran the new hooks ten seconds later, where Droid's documentation says a
session reads its hooks when it starts (seen once, in the Factory app).
Sessions captured through the installed hooks on 2026-11-12 added a
terminal and sessions with no interface. The `SessionStart` of a session
a Task call started names the calling session, and most such sessions
raise a second `SessionStart` within a second, a `resume` with no
`message_id`. A session with no interface raises `SessionEnd` as it
finishes. Droid can raise `UserPromptSubmit` and `Stop` many times for
one message, and raises `UserPromptSubmit` for text it wrote itself. A
transcript message says who saw it (`visibility`), and a call a Script
made names the Script's call. What Droid tells the user alone is in the
transcript as a message with the role of user, and a tool result holds
the text the agent was given beside the one the user was shown, where
the two differ.
See "What a Droid hook carries", "What Droid's turn and session hooks
carry", "When a Droid session ends, and what its transcript says of
hooks", "`mem hooks` writes capture into Droid's hooks file", "What
Droid's `SubagentStop` hook carries", "What `/clear` starts in Droid",
"What the first sessions captured through the hooks showed", and "A
notice only the user saw, and the text of a tool result" in
[phase-1-spec.md](phase-1-spec.md).

| Harness | Events | Payload | Gaps for capture |
|---------|--------|---------|------------------|
| Droid (`~/.factory/hooks.json`) | `SessionStart` (`source`: startup, resume, clear, compact), `UserPromptSubmit`, `PreToolUse`, `PostToolUse`, `PreCompact`, `Stop`, `SubagentStop`, `Notification`, `SessionEnd` | JSON on stdin: `session_id`, `transcript_path`, `cwd`, `permission_mode`, `hook_event_name`, `message_id`; tool hooks add `tool_name`, `tool_input`, `tool_response`; `SubagentStop` adds `task_name`, `task_result`. Default timeout 60 s. | No hook carries the assistant reply; read the transcript. `SessionStart` repeats per resume and compaction. |
| Vibe (`~/.vibe/hooks.toml`) | `pre_tool`, `post_tool`, `post_agent` | JSON on stdin: `session_id`, `parent_session_id`, `transcript_path`, `cwd`, `hook_event_name`; tool hooks add `tool_name`, `tool_call_id`, `tool_input`, `tool_status`, `tool_output`, `tool_output_text`, `tool_error`. Default timeout 60 s. | No prompt, session, or compaction hooks; `post_agent` carries no text. Project-level hooks load only in trusted folders. Plugin-declared hooks do not run yet. |
| omp (`~/.omp/agent/hooks/{pre,post}/*.ts`) | `session_start`, `session_switch`, `session_compact`, `session_shutdown`, `agent_start`, `agent_end`, `turn_start`, `turn_end`, `message_start`, `message_end`, `tool_call`, `tool_result`, `tool_execution_*`, `tool_approval_*`, `auto_compaction_*`, `session_stop` | Typed in-process payloads with full message text, tool inputs and results, compaction entries, `session_file`. | None material. Skip the high-volume `message_update` and `tool_execution_update`. |
| Amp (`.amp/plugins/*.ts`) | `session.start`, `agent.start`, `tool.call`, `tool.result`, `agent.end` | In-process: thread id, prompt and message id on `agent.start`; `toolUseID`, `tool`, `input`, `status`, `output` on tool events; every `ThreadMessage` of the turn on `agent.end`. | No compaction or session-end event; `onDispose` runs on unload for about 3 s. |
| Junie | none | MCP tools only | Everything except what the agent chooses to log. |

## Growth

Measured on 2026-11-13 and 14 on a machine with 18 cores and 48 GB, on
copies of the experiment home: 330,779 events of 2,335 sessions over
194 days, and that ledger grown to 1, 3, and 10 million events by
adding its sessions again under new ids. No ledger that large has come
from real use.

**Disk.** A home takes about 7.7 KB for each event: 3.65 KB in the
database, 1.9 KB in blobs, and 1.07 KB in each of the two Parquet
snapshots the server keeps. It kept 14 until 2026-11-14, which made
20.5 KB, 73% of it snapshots. The experiment home gained 52,000 events
a month over six months, and 251,000 in its busiest. The production
home, captured through hooks, is gaining about 205,000 a month. At
52,000 a month a home reaches 5 GB in a year and 24 GB in five; at
250,000 a month, 23 GB and 115 GB. With 14 snapshots those were 13 and
64 GB, and 62 and 308 GB.

**Hooks.** A hook has 2 s before it spools. Each one ran the binding
check, which read the payload of every event (see "Which placeholders a
write reads" in [logic-hardening-spec.md](logic-hardening-spec.md)).
Before that change and after it, on the same ledgers with no citation
in them:

| Events | Prompt hook | Eight hooks at once, the slowest | `Stop` with one new turn |
|--------|-------------|----------------------------------|--------------------------|
| 330,000 | 1.02 s, then 0.40 s | 1.79 s, then 0.68 s | 1.09 s, then 0.47 s |
| 1 million | 1.34 s, then 0.40 s | 2.70 s, then 0.64 s | 1.40 s, then 0.46 s |
| 3 million | 1.94 s, then 0.43 s | 2.82 s, then 0.75 s | 2.16 s, then 0.50 s |
| 10 million | 2.60 s, then 0.39 s | 3.42 s, then 0.64 s | 4 to 17 s over four runs, then 0.46 s |

Before, all eight hooks at once missed the deadline from 1 million
events on, and so did every hook alone at 10 million. After, none did.
Most of what is left is the start of the `mem` process (see "Known
defects").

The check now costs by the placeholders that wait for a reply: 0.1 to
0.2 ms each, at every write. At 10 million events, 750 waiting made a
prompt hook 0.56 s. 7,500 made it 1.95 s, and all eight at once missed
the deadline again. No ledger in use holds a citation yet. The typed
table under "Proposed directions" took 1.5 ms as a prototype, whatever
waited, and 0.3 s to fill for 20,000 cited events.

**What else grows with the ledger.**

- Memory. After 20 hooks on the ledger of 10 million events, with 7,500
  placeholders waiting, the server held 2.3 GB, and 5.9 GB with the
  earlier check. After an audit and the judge's reads as well, with the
  earlier check, it held 0.9 GB at 330,000 events, 2.8 GB at 1 million,
  8.2 GB at 3 million, and 27 GB at 10 million.
- A snapshot: 381 MB in 3.4 s at 330,000 events, and 11.2 GB in 128 s
  at 10 million.
- `mem audit check`: 1.3 s at 330,000 events, and 15 s at 10 million.
- Reading every row of `session_summary`: 1.1 s at 1 million events,
  and 6.3 s at 10 million.
- `cite`. To find the sessions that saw its placeholder it reads the
  attribution evidence of every capture observation: 0.3 s for 420,000
  observations, and 0.6 s on a busy server (`capture/attribution.py`).

Recall did not slow: 24 to 35 ms at every size.

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
- `mem judge report --freeze` does not check that every judge has
  judged every calibration item. A judge that skipped items gets its
  thresholds from the ones it judged, and nothing says so
  (`judge/report.py`).
- A caller chooses the `trust_level` of what it remembers and nothing
  checks it. A memory an agent writes with `user` trust is recalled
  without an approval (`mcp_server/tools.py`).
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
- `/clear` has been recorded in the Factory app only. There it starts a
  new session whose `SessionStart` says `startup`, which the hook and
  the transcript's first entry record as one event. The one terminal
  session captured so far started and resumed as a session in the app
  does, and nobody typed `/clear` in it. A `SessionStart` that named a
  new session and gave `clear` as its source, which Droid's
  documentation lists, would be recorded twice (`adapters/droid.py`).
- A Droid turn that never reaches `Stop` (a crash, a killed session) is
  in the ledger only after the session's next `Stop` or `SessionEnd`, or
  after someone imports its transcript. Droid writes a turn's outcome
  after its `Stop` hook has run, so the outcome of a session's last turn
  arrives at `SessionEnd`, 30 minutes after the session was last used.
  Until then, and for good if `SessionEnd` never runs, the judge reads
  that turn as cut off when the session ended (`closed_by` is
  `session_end`), not as completed. A session with no interface raises
  `SessionEnd` as it finishes. Not checked: whether quitting Droid in a
  terminal or closing the app does. Two of the 13 sessions a Task call
  started had written 214 and 295 transcript lines and reached neither
  hook within the hour.
- The first import of a long session runs inside its `Stop` hook and
  takes seconds: 8.75 s for a transcript of 41.9 MB and 13.2 s for one
  of 61.0 MB, and at that rate about 17 s for the largest local one,
  79 MB. Each later `Stop` reads the whole file again to find its place,
  1.0 s at 61 MB. Droid's hook timeout is 60 s. Not checked: what Droid
  shows while a hook runs that long, and what a hook killed at the
  timeout leaves behind.
- Droid can raise `UserPromptSubmit` and `Stop` for one message every
  few seconds: 302 runs of each in half an hour of one terminal session.
  The ledger gains nothing twice, and each run is a process that takes
  0.4 to 0.5 s (see the start-up cost below).
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
  `capture_exclusions` and `capture_transcript_partial` always apply.
  `capture_coverage_gap` always applies too, for three reasons. The
  first import begins at the transcript's first entry, which is dated
  after the `SessionStart` hook's event. `session_summary` shows one
  observation of a session, the import that reaches furthest, and each
  import after the first begins where the one before it ended. A hook's
  event later than the last transcript entry, such as the `session_end`,
  lies past what that import covers. No Droid session can read as fully
  covered (`audit.py`, `adapters/droid.py`, the `session_summary` view).
  The live ledger showed it on 2026-11-12: no error, and 259 warnings
  for 93 sessions.
- The judge reads some harness text as a person's. It tells the author
  of a user message by how the text opens, and it does not know
  "Background Execute command finished." or "Background task error.".
  In 69 captured transcripts it took 15 of 295 texts that Droid gave
  the agent alone for a person's or a calling agent's. Droid marks such
  a text (`visibility`), and an import keeps the mark since 2026-11-12,
  but nothing reads it, and a prompt the `UserPromptSubmit` hook wrote
  first has none (`judge/provenance.py`, `adapters/droid.py`).
- A notice imported before 2026-11-13 is a `user_msg`, where a later
  import writes a `note`. The judge starts a turn at it, and does not
  know how most of them open ("The model provider declined this request
  again.", "Request blocked by model provider safety checks."): it took
  21 of 29 for a person's or a calling agent's. The ledger of the
  production home holds 40. No event is rewritten, and a whole-file
  import of such a transcript adds the `note` beside the `user_msg`
  (`adapters/droid.py`, `judge/triggers.py`, `judge/provenance.py`).
- A tool result imported before 2026-11-13 holds what Droid showed the
  user, where Droid stored a second text for the agent: for an edit, a
  record of the change and not the one line the agent got. A result
  keeps its id, so reading its transcript again changes nothing
  (`adapters/droid.py`).
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
- A tool call imported before 2026-11-12 does not say which Script call
  it ran inside, so the ledger cannot tell such a nested call from a
  top-level one. A later import keeps it as `outer_tool_use_id` in the
  call's payload, and nothing reads it yet (`adapters/droid.py`).
- `tests/fixtures/droid_session.jsonl` gives two messages a `visibility`
  of `visible` and `hidden`, values Droid does not write.
- An event captured through Droid's hooks names no turn in the
  attribution stored with it (`capture_attribution`). The import at
  `Stop` writes a turn's messages before Droid has written the outcome
  that names the turn, and no event is rewritten. Nothing reads the
  stored copy, and binding a reply to a citation uses the import's own
  evidence, so no result is wrong. Since 2026-11-11 such an event says
  nothing of its turn. Before, it held a null turn marked not complete
  (`capture/writer.py`, `adapters/droid.py`). See "What an event keeps
  of its message and turn" in [phase-1-spec.md](phase-1-spec.md).
- A source distribution built from a working tree holds every file there
  that `.gitignore` does not name, tracked or not: `uv build` packed an
  untracked `.ori/state.sqlite`. The wheel holds the package alone, and
  nothing in the repository publishes a source distribution
  (`pyproject.toml`).
- The server sets no memory limit, so DuckDB's default applies: 80% of
  the machine's memory (`db.py`). See "Growth" for what a server holds.
- The server's first automatic export comes a full interval after it
  starts, 24 hours by default, and the interval is counted on a clock
  that stops while the machine sleeps: on the development machine it was
  4.3 hours behind the wall clock 78 hours after boot. A server that is
  restarted every day never exports, and one on a laptop exports less
  often than daily. With two snapshots kept, the newest is the copy to
  fall back on, and it can be days old (`db.py`).
- Nothing removes what a home gathers beside the database. A replayed
  spool file stays under `spool/done/` until someone runs `mem ingest
  --spool --purge-done DAYS`, and launchd appends to `serve.log` and
  `admit.log` without end (`capture/spool.py`, `scripts/`).
- A hook whose write is still going at its deadline puts its records in
  the spool, and the server may commit the same write a moment later.
  Nothing is stored twice, because replay finds the records present,
  but the file waits until the server next starts or someone runs
  `mem ingest --spool` (`capture/pipeline.py`).
- A tool call's input is stored twice: in the text of the event, and as
  `tool_input` in its payload, which has no size limit. The largest
  payload measured was 713 KB (`adapters/droid.py`).

Reported in the same audit and not yet checked against the code: the
hand-maintained `ALL_TABLES` list; lockdown happening after the listener
starts; the experiment-home guard excluding only the default path; the
approver not pinning question, projection, and drafter versions;
`--similarity-floor 0` and `--min-writes 1` weakening the auto-approval
guards; turns still in flight being admitted and then frozen; the Go
helper's stderr being discarded; the drafter's name leaking through
blinding; an egress error body in exception text.

## Proposed directions

Proposed in the audit of 2026-11. None of these is approved.

- Finish capture. Sessions were captured end to end through installed
  hooks on 2026-11-12, one of them in a terminal. Left to see: `/clear`
  and the end of a session in a terminal, and a long first import in a
  running Droid.
- Give the judge a new provenance version that reads who saw a message
  from the event, and that knows a notice an earlier import wrote as a
  `user_msg`.
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
- A typed table of the placeholders that wait for a reply, which `cite`
  would write (a migration). The binding check after a write would then
  never read `payload`. Proposed on 2026-11-13; see "Growth".

Modules that would gain from deepening, in the order the audit ranked
them: a ledger connection with two adapters; auto-approval in one place.
The three ranked above these were deepened on 2026-11-11: the ledger
session (`capture/session.py`), the model call (`model_call.py`), and
the loop that shows a controller's screens (`drive` in
`judge/label_frontend.py`), which labeling and admission review now
share. The two keep their own controllers. They ask different questions
and save differently, and they check a note by different rules: labeling
refuses a tab and admission review takes one. Each controller still
answers back, skip, redo, and quit in about 25 lines that are nearly the
other's.

The same audit proposed removing the following. Each was looked at on
2026-11-11 and stays:

- The judge's handling of a draft without claims, and the check that
  refuses to draft into a run whose drafts another drafter version
  wrote. A lesson drafted now always has claims, but the drafts stored
  in exp1 to exp3 have none, and their reports must still reproduce.
- `build_server`, which declares each MCP tool beside the `MemoryTools`
  method it calls. The declaration is what a harness is told, so it
  stays written out. A test fixes every tool's inputs, and each input
  is handed on by name.
- The attribution stored with an event. It holds only what the adapter
  knows when the event is written, and what Droid sends: the reading of
  fields no recorded transcript or hook payload carries was removed. It
  is described in [phase-1-spec.md](phase-1-spec.md) and fixed by tests.
- The key builders in `capture/identity.py` that return what they are
  given. They are the one place the rules for a key are stated
  ([ADR 0003](adr/0003-deterministic-capture-ids.md)).
- Ten type-checker warnings, each about a `str()` or an `int()` around
  a value from a harness payload, a stored row, or a tool call. The
  annotation says more than that data can promise, so the conversion
  stays.
- The reading and the migrating of a ledger that has applied migration
  0001 only (`legacy_preflight` in `audit.py`, `_migrate_lifecycle` in
  `db.py`). The experiment home is still at 0001, and its next
  `mem migrate` runs it.
- Migrations 0002 to 0006 as five files. One file in their place would
  change nothing at run time. `mem migrate` refuses a home whose applied
  migrations have no file or another checksum, so any ledger that has
  applied one of the five would be shut out. The upgrade of a ledger at
  0001 is keyed on the name of 0002, and tests stop at the versions
  between.

## Open decisions

- Should `mem admit review` and `mem judge label` take their answers
  from a terminal only, as `mem review approve` does? Both read piped
  answers today.
- Should a refused approval leave a row in the ledger? Today one the
  tool refuses leaves none.
- Should a Droid session captured from start to end be able to read as
  fully covered, and what would have to be true of it?
- Should timeouts in a row stop a judge command? A timeout can be the
  item's doing, so today only an unreachable model server stops one.
- Are the `readout` and `jev` judges and the `actionable` question still
  wanted?
- May a run be relabeled after a freeze? Should `--freeze` also ask for
  every judge's verdicts on the calibration items?
- Do Phases 0.5, 3, and 5 stay on this roadmap?
- Do old events ever leave the ledger for an archive? The ledger only
  grows. Moving events out would be a new rule beside "append-only",
  and would need a record under `docs/adr/`.
- What happens to `example-memory-db-schema.sql` and the prototype branch?
