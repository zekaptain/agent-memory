# agent-memory roadmap

What is built, what is planned, what is known to be wrong, and what is
still undecided. The design of what is built is in
[architecture.md](architecture.md). Earlier plans and audits are kept in
[archive/plan-history.md](archive/plan-history.md).

## Status

As of 2026-11-23.

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
| Lifecycle, coverage, and attribution hardening | Done (migrations 0002 to 0006, 0008 for a session's start, and 0009 for its title, caller, and harness version). The production home has applied every migration through 0010 since 2026-11-22, and the experiment home is at 0001. The migrations through 0008 ran on copies of both homes on 2026-11-11, and 0009 on a copy of the experiment home on 2026-11-13. | [logic-hardening-spec.md](logic-hardening-spec.md) |
| Judge experiment | Built; four runs stored | [judge-experiment-spec.md](judge-experiment-spec.md), [run logs](archive/judge-experiments.md) |
| Usefulness experiment | Designed, and its build planned in stages; nothing built | [memory-utility-evaluation-spec.md](memory-utility-evaluation-spec.md) |
| Parametric consolidation | Research proposal | [parametric-consolidation-spec.md](parametric-consolidation-spec.md) |

## Next approved work

The next approved work is to build and run the local usefulness
experiment. Droid capture has been verified end to end. Its hooks were
checked against real payloads on 2026-11-11, all of them recorded in the
Factory app. On 2026-11-12 the first 120 sessions were captured through
installed hooks, in the app, in a terminal, and with no interface, and
each agreed with its transcript. On 2026-11-23 a person at a running
Droid saw what was left: `/clear` and the end of a session in a
terminal, and what Droid shows during a long first import. What that
found wrong is under "Known defects". The largest is that no hook runs
to its end when Droid exits in a terminal, so nothing records the end
of such a session.
The build of the experiment is planned in stages under "Building it" in
[memory-utility-evaluation-spec.md](memory-utility-evaluation-spec.md).
Retrieval tuning and wider graph use follow measured failures.
Parametric consolidation remains research until usefulness and
withdrawal/backflow can be measured.

Also agreed on 2026-11-13, from the measurements under "Growth" below:
decide whether old events ever leave the ledger. Decided on 2026-11-23:
they stay (see "Events stay" under "Growth", and
[ADR 0010](adr/0010-events-stay-and-a-human-may-blank-text.md)).
The steps agreed before it are done. The binding check after a write no
longer reads every event, the server keeps two Parquet snapshots where
it kept 14, it has a memory limit, and a replay deletes the files
replayed more than 30 days ago.

On 2026-11-22 the approved work was put into issues (see
[agents/issue-tracker.md](agents/issue-tracker.md)), and some of what
was proposed that day was approved (see "Proposed directions"). In
order:

1. MEM-14: the three checks of capture that were still to see. A person
   at a running Droid made them on 2026-11-23.
2. MEM-1, then MEM-15: the design of the usefulness experiment gains
   what the peer benchmarks leave out, and its build is planned in
   stages, with the issues of each. Building starts when that plan is
   accepted. A read of a home in use, a model that is not on this
   machine, and every run then each need the user's go-ahead, as the
   design says.
3. MEM-2, then MEM-16: measure what matching text over events costs,
   and with that number decide whether old events leave the ledger.
   Decided on 2026-11-23: they stay.

Approved beside that line, to be taken up as it allows: MEM-13 (a
memory with invisible characters or text that reads as an instruction
starts quarantined, which closes a known defect) and MEM-9 (admission
records whether a rule's exact values are in the lines it cites). MEM-6
(what Droid does with the stdout of a `SessionStart` hook, also for a
person) was checked with MEM-14 on 2026-11-23: Droid gives it to the
agent. MEM-5 is approved once the user agrees to a read of the
production home.

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
transcript. Its `SessionStart` said `startup` in the app on 2026-11-11.
In a terminal on 2026-11-23 it said `clear` and named the session that
was left, which raised `SessionEnd` twice. `SessionEnd` carries
the `message_id` of the session's last message and runs 30 minutes
after a session was last used in the app, so a session can end more
than once.
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
A person at a terminal and at the app on 2026-11-23, with Droid 1.17.2
and 1.18.0, added the end of a session in a terminal and the stdout of
a `SessionStart` hook. When Droid exits in a terminal it raises
`SessionEnd`, and every hook command fails within a third of a second.
Plain text that a `SessionStart` hook prints on exit 0 is given to the
agent: at a new session, after `/clear`, at a resume with the next
prompt, and after a compaction. It is in the transcript as a message
that only the agent saw, and 62 KB of it arrived whole. Droid runs an
event from `settings.json` only when `hooks.json` in the same directory
does not define it.
See "What a Droid hook carries", "What Droid's turn and session hooks
carry", "When a Droid session ends, and what its transcript says of
hooks", "`mem hooks` writes capture into Droid's hooks file", "What
Droid's `SubagentStop` hook carries", "What `/clear` starts in Droid",
"What the first sessions captured through the hooks showed", "A notice
only the user saw, and the text of a tool result", and "What a terminal
and a `SessionStart` hook's stdout showed" in
[phase-1-spec.md](phase-1-spec.md).

omp was checked against real payloads on 2026-11-24: version 19.1.1, in
22 runs with no interface and a local model, recorded by an extension.
An extension is handed every prompt, reply, tool call, tool result, and
compaction in omp's own process. No message carries an id. A tool call
does, and so does a transcript entry, which the extension reads from
the session and not from the payload. `session_start` is raised once in
a process and says nothing of a resume. A new session, a fork, and a
branch in a running omp raise `session_switch` or `session_branch`.
`session_stop` closes a prompt that ran to its reply. It is not raised
after a signal or a time limit, nor in a spawned session, which runs in
its caller's process with an id and a transcript of its own. omp waits
for a function at `tool_call`, and fails the call when the function
takes 30 s. It tells an MCP server nothing of the session that calls
it. Not checked: omp in a terminal, an automatic compaction, and a
model that is not on this machine. See "What omp hands an extension" in
[phase-1-spec.md](phase-1-spec.md).

| Harness | Events | Payload | Gaps for capture |
|---------|--------|---------|------------------|
| Droid (`~/.factory/hooks.json`) | `SessionStart` (`source`: startup, resume, clear, compact), `UserPromptSubmit`, `PreToolUse`, `PostToolUse`, `PreCompact`, `Stop`, `SubagentStop`, `Notification`, `SessionEnd` | JSON on stdin: `session_id`, `transcript_path`, `cwd`, `permission_mode`, `hook_event_name`, `message_id`; tool hooks add `tool_name`, `tool_input`, `tool_response`; `SubagentStop` adds `task_name`, `task_result`; a `SessionStart` after `/clear` in a terminal adds `previous_session_id`. Default timeout 60 s. | No hook carries the assistant reply; read the transcript. `SessionStart` repeats per resume and compaction. No hook command runs to its end when Droid exits in a terminal. |
| Vibe (`~/.vibe/hooks.toml`) | `pre_tool`, `post_tool`, `post_agent` | JSON on stdin: `session_id`, `parent_session_id`, `transcript_path`, `cwd`, `hook_event_name`; tool hooks add `tool_name`, `tool_call_id`, `tool_input`, `tool_status`, `tool_output`, `tool_output_text`, `tool_error`. Default timeout 60 s. | No prompt, session, or compaction hooks; `post_agent` carries no text. Project-level hooks load only in trusted folders. Plugin-declared hooks do not run yet. |
| omp (an extension: a TypeScript module given with `-e`, or found in `~/.omp/agent/extensions/` or `.omp/extensions/` by omp's source) | Raised on 2026-11-24: `session_start`, `session_before_switch`, `session_switch`, `session_before_branch`, `session_branch`, `session_before_compact`, `session.compacting`, `session_compact`, `session_stop`, `session_shutdown`, `input`, `before_agent_start`, `agent_start`, `agent_end`, `context`, `turn_start`, `turn_end`, `message_start`, `message_update`, `message_end`, `assistant_message`, `before_provider_request`, `after_provider_response`, `tool_call`, `tool_result`, `tool_execution_start`, `tool_execution_update`, `tool_execution_end`, `before_subagent_spawn`. Registered and not raised: 23 more, among them `session_clear`, `auto_compaction_*`, and `tool_approval_*`. | In process, as values: whole messages, tool inputs and results, the compaction's entry; the session's id, file, and entries from the function's context. A function has 30 s, and 2 s at `session_shutdown`. | No message carries an id. `session_start` once in a process, and none after a switch or a branch. No `session_stop` after a signal or a time limit, or in a spawned session. A slow function at `tool_call` fails the call. An MCP server is told nothing of the session. Not checked in a terminal. |
| Amp (`.amp/plugins/*.ts`) | `session.start`, `agent.start`, `tool.call`, `tool.result`, `agent.end` | In-process: thread id, prompt and message id on `agent.start`; `toolUseID`, `tool`, `input`, `status`, `output` on tool events; every `ThreadMessage` of the turn on `agent.end`. | No compaction or session-end event; `onDispose` runs on unload for about 3 s. |
| Junie | none | MCP tools only | Everything except what the agent chooses to log. |

## Growth

Measured on 2026-11-13 and 14 on a machine with 18 cores and 48 GB, on
copies of the experiment home: 330,779 events of 2,335 sessions over
194 days, and that ledger grown to 1, 3, and 10 million events by
adding its sessions again under new ids. No ledger that large has come
from real use.

**Events stay.** Decided on 2026-11-23 from what follows, for MEM-16: no
event leaves the ledger for an archive, and no text is taken out of it
to save space
([ADR 0010](adr/0010-events-stay-and-a-human-may-blank-text.md)).

- What a hook or an agent waits for does not slow as the ledger grows.
  At 10 million events a hook takes the 0.4 s it takes at 330,000, a
  recall takes 24 to 35 ms, and a match of text over the newest 600,000
  events, three months of the production home's capture, takes 0.1 to
  0.3 s with no index (see "Matching text over events" below). The one
  cost of a hook that grows with history is the placeholders that wait
  for a reply, and the typed table under "Proposed directions" answers
  that.
- What reads the whole ledger takes seconds at 10 million events and
  completes under the server's memory limit: a match of text over every
  event 1.6 to 2.5 s, a snapshot 9 to 12 s, `mem audit check` 15 s.
- So an archive would save disk and nothing else. It would cost the
  join from a memory, a recall, or a reply to its events, because
  events that had left would have to be looked for in a second place.

What bounds a home instead:

- The server keeps two snapshots, whatever the ledger's age, and a
  replay deletes the spool files replayed more than 30 days ago.
- The server's memory has a limit, and what waits on the ledger takes
  the same at every size measured.
- Nothing bounds the database and the blobs. They grow by about 5.6 KB
  with each event, and the two snapshots by 2.1 KB more. At the
  production home's 205,000 events a month that is 19 GB a year, if an
  event there takes what one takes in the experiment home. The most
  measured here, 10 million events, is four years of it.

If disk is what runs short, three things make an event smaller without
touching a row. They were measured on 2026-11-23 on the experiment
home's 330,000 events, and none of them is approved work:

- Blobs are kept as plain text: 12,329 files holding 0.62 GB, which
  zlib takes to 0.26 GB. Kept compressed, the 1.9 KB of blobs for each
  event would be about 0.8 KB.
- The second snapshot is 1.07 KB for each event. `--export-keep 1`
  drops it, and with it the copy that is read if the newest cannot be.
- A tool call's input is stored twice (see "Known defects"). The
  payloads of tool calls, which hold the second copy, are 141 MB of the
  770 MB of text the events hold in the database, before DuckDB
  compresses it.

The decision is looked at again when a home in use nears 10 million
events, or when something a hook or an agent waits for is measured to
slow with the ledger's size. Text that must not be kept is a separate
matter: a human may blank it, and its row stays, as the same record
has it.

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
  8.2 GB at 3 million, and 27 GB at 10 million. The server had no
  memory limit then; see "Under a memory limit" below.
- A snapshot: 381 MB in 3.4 s at 330,000 events, and 11.2 GB in 128 s
  at 10 million. Events were copied in the order of the table then; see
  the same section.
- `mem audit check`: 1.3 s at 330,000 events, and 15 s at 10 million.
- Reading every row of `session_summary`: 1.1 s at 1 million events,
  and 6.3 s at 10 million.
- `cite`. To find the sessions that saw its placeholder it reads the
  attribution evidence of every capture observation: 0.3 s for 420,000
  observations, and 0.6 s on a busy server (`capture/attribution.py`).

Recall did not slow: 24 to 35 ms at every size.

**Under a memory limit.** Measured on 2026-11-22 on the same machine
and the ledgers of 1 and 10 million events, once the server had a
memory limit (10% of the machine's memory unless told otherwise) and
copied events to Parquet as it reads them. Memory is in GiB here, as
DuckDB prints a limit: 10% of that machine is 4.7 GiB, and `2GB` is 1.8
GiB. What a process held is its physical footprint, which counts the
memory macOS has compressed: the resident size of a server that held 26
GiB fell to 2 GiB.

What the `mem serve` process held at 10 million events:

| Limit | After 20 hooks | Reading all of `session_summary`, a content search, and `mem audit check`: the most, and idle afterwards | Its own snapshot, in a server just started | While `mem export` read through it |
|-------|----------------|-------------------|-------------------|-------------------|
| DuckDB's own, 80% (38.3 GiB) | 0.15 GiB | 27.9 GiB, 26.2 GiB | 28.0 GiB in 11.9 s | 42.6 GiB |
| 10% (4.7 GiB) | 0.16 GiB | 7.0 GiB, 5.7 GiB | 5.6 GiB in 11.6 s | 20.2 GiB |
| `2GB` (1.8 GiB) | 0.15 GiB | 3.7 GiB, 2.9 GiB | 3.3 GiB in 10.5 s | 9.2 GiB |

At 1 million events and 10% it held 3.1 GiB after the same reads, which
is less than the limit, and 7.7 GiB while `mem export` read through it.

- The process holds more than the limit: about 1 GiB more at rest, and
  far more while a client takes a large result. DuckDB counts the pages
  of the database it keeps and what a query works with, and not a
  result on its way to a client; `mem export` asks for every row of
  every table (see "Known defects").
- Hooks took the same under every limit from `512MB` up, at both sizes:
  a prompt hook 0.38 to 0.40 s, the slowest of eight at once 0.64 to
  0.71 s, and `Stop` with one new turn 0.44 to 0.51 s. None missed its
  deadline.
- Reads at 10 million events took what they did without a limit, but
  for two. `mem audit check` took 18 to 21 s with no limit, 15 to 16 s
  with 10%, 29 s with `2GB`, and 35 s with `1GB`. A content search took
  4.0 s with no limit and 2.0 to 2.3 s with one.
- What fails for want of memory at 10 million events. With `1GB`: the
  judge's read of every turn, and a window function over the content of
  a million events. With `512MB`, also reading all of
  `session_summary`, the judge's read of callers, and the snapshot.
  With `2GB` all of them complete. At 1 million events the window
  function still needs `2GB`, and everything else measured there
  completes with `512MB`; the snapshot fails with `256MB`. The judge's
  read of every turn was measured at 10 million events only.
- A snapshot. Copied in the order of the table, the events of the
  ledger of 1 million took 15 to 35 GiB of memory, and the copy ran out
  of memory under every limit tried, `512MB` to `8GB`. Copied as they
  are read, in row groups of 8 MB, they take about 1 s under every
  limit from `512MB` up, and those of 10 million take 9 to 12 s under
  every limit from `1GB` up. That snapshot is 10.1 GB on disk, where it
  was 11.2 GB.
- `mem export` through the server took 4 to 6 s at 1 million events and
  60 to 106 s at 10 million, and its own process held 7.0 and 8.5 GiB.
  `mem audit check` held 1.3 and 8.5 GiB in its own.

**Matching text over events.** Measured on 2026-11-22 and 23 for MEM-2,
on the same machine, on copies of 330,000, 1 million, and 10 million
events made as the ones above and brought to migration 0010. The server
had its default memory limit, 10% (4.7 GiB), and 18 threads. User
messages and replies are 8% of the events (26,000, 80,000, and 798,000)
and hold 9% of the text. A scratch probe made every index and table on
the copies, and nothing was added to the schema. A hook ran as Droid
runs it, as a new `mem event` process, and a query went through the
server three times, the median given. The terms were two single words,
in 0.5% and 0.8% of the events, the two together, in 0.2%, and a word
that is nowhere. Sizes are in GiB, as above.

With no index, `content ILIKE` for a term that is found, returning the
20 newest matches or counting them:

| Events | The whole ledger | User messages and replies | Those among the newest 600,000 events | Those among the newest 200,000 |
|--------|------------------|---------------------------|---------------------------------------|--------------------------------|
| 330,000 | 0.23 to 0.26 s | 0.23 to 0.27 s | the whole ledger | 0.14 to 0.19 s |
| 1 million | 0.27 to 0.30 s | 0.27 to 0.30 s | 0.12 to 0.29 s | 0.11 to 0.17 s |
| 10 million | 1.6 to 2.5 s | 1.6 to 1.7 s | 0.13 to 0.31 s | 0.10 to 0.18 s |

- Only a span of time shortens the read. Keeping to user messages and
  replies saves nothing through 1 million events and little at 10
  million. A span costs at 10 million events what it costs in a small
  ledger, because DuckDB skips the row groups whose times lie outside
  it: there the newest 200,000 events lay in 5 of 82. That needs events
  to sit in the table in about the order of their times, as capture
  writes them. The newest 200,000 events are a month of the production
  home's capture, and the newest 600,000 are three.
- A word that is nowhere took 0.2 s over the whole ledger through 1
  million events and 1.3 s at 10 million, and 0.11 to 0.23 s in a span.
- No query needed more than the limit. DuckDB counted 0.7 GiB at
  330,000 events and 2.0 GiB at 1 million. At 10 million it stayed at
  the limit and wrote nothing to its temporary directory, and the
  process held 5.3 to 7.1 GiB.

With an index over the text of every event that every write maintains,
which is how the index over memories is kept. A cell with two times
gives the time without the index, then with it:

| Events | Build | Size | One query | One event written, in process | Prompt hook | `Stop` with one new turn | Eight hooks at once, the slowest |
|--------|-------|------|-----------|-------------------------------|-------------|--------------------------|----------------------------------|
| 330,000 | 14.5 s, 2.6 GiB | 0.17 GiB | 0.02 to 0.03 s | 0.05 s, then 0.11 s | 0.38 s, then 0.45 s | 0.44 s, then 0.57 to 0.60 s | 0.64 s, then 1.5 to 1.6 s, and 7 of 16 spooled |
| 1 million | 17.7 s, at the limit | 0.50 GiB | 0.05 s | 0.06 s, then 0.19 to 0.20 s | 0.39 s, then 0.54 to 0.55 s | 0.46 s, then 0.75 to 0.78 s | 0.64 s, then 2.8 to 2.9 s, and 12 and 13 of 16 missed the deadline |
| 10 million | 136 s, at the limit | 4.9 GiB | 0.30 to 0.34 s | 0.06 s, then 1.2 to 1.3 s | 0.40 s, then 1.65 to 1.68 s | 0.47 s, then 2.9 to 3.0 s | 0.64 s, then 3.1 to 3.2 s, and all 16 missed it |

- Without an index, hooks took what they took before at every size,
  and none missed its deadline.
- What the index adds to one write grows with the ledger: 0.06 s, 0.14
  s, and 1.2 s.
- Writes at once conflict. Each changes rows of the index that every
  other write changes, and DuckDB lets only one of them commit
  ("Conflict on update!"). A hook's write is tried again three times
  and then goes to the spool, as 7 of 16 did at 330,000 events. From 1
  million events the hooks pass their 2 s first. After the eight at
  once, five prompt hooks alone spooled too at 1 and at 10 million
  events, and the server could not checkpoint as it stopped ("there are
  other write transactions active"). Every hook still exited 0.
- A build that needs more than the limit writes what does not fit to
  the temporary directory, which was allowed 150 GB, and completes: 1.0
  GiB at 1 million events and 42 GiB at 10 million.
- A query through the index finds whole words by their stems and ranks
  them, where `ILIKE` finds the text anywhere. For these terms the two
  counted within 2% of each other.
- Text of many different words changed none of these times.

With an index over user messages and replies alone, in a table of their
own beside `events`, which no write maintains:

| Events | Copied and built with no server | Built in the server while hooks ran | The copy, and the index | One query |
|--------|---------------------------------|-------------------------------------|-------------------------|-----------|
| 330,000 | 3.2 s, 0.8 GiB | not measured | 0.06 and 0.02 GiB | 0.01 to 0.02 s |
| 1 million | 9.4 s, 2.6 GiB | 11.7 s, 2.7 GiB | 0.18 and 0.04 GiB | 0.01 to 0.02 s |
| 10 million | 21 s, at the limit, 0.6 GiB to the temporary directory | 27.9 s, at the limit, 0.8 GiB to the temporary directory | 1.82 and 0.39 GiB | 0.03 to 0.05 s |

- No hook missed its deadline while the server built the index. At 1
  million events a prompt hook took 0.43 s where it took 0.38 s before
  the build, and the slowest of 48 released eight at once took 0.75 s.
  At 10 million it took 0.54 s, and the slowest of 80 took 1.08 s.
- The index holds the messages that were there when its build began: 69
  and 114 arrived during the two builds.
- Maintained by each write instead, the index made the insert of one
  message take 16, 21, and 105 ms, where it takes 0.2 to 0.5 ms into a
  table with no index. Of 40 inserts by eight writers at once, 34, 33,
  and 32 failed with the same conflict, and none failed without the
  index.

What the numbers favor, for a recall of events by text (MEM-3):

- No index to begin with. A match over the user messages and replies
  of a span of time costs 0.1 to 0.3 s at every size. It adds nothing
  to a write and nothing to the schema, and it sees an event as soon as
  it is written. Over the whole ledger it costs 0.3 s through 1 million
  events, which the production home reaches in about five months at
  its pace, and 1.6 to 2.5 s at 10 million.
- When that is too slow, or when matches are to be ranked: an index
  over messages alone, in a table of its own, which the server builds
  on a schedule, aside from every write, the way it writes a snapshot.
  It answers in 0.01 to 0.05 s. It knows nothing written since its
  build began, so a match that must be current also reads the events
  since then with no index.
- Never an index that the write of a hook maintains. It makes hooks
  that run at once conflict at every size measured, and from 1 million
  events they miss their deadline.

## Risks and open questions

- Quack shipped in 2.0.0. Its protocol and authorization hook may change;
  `mem smoke` and the server tests are the early warning.
- History can only be corrected by appending. A wrong memory is superseded or
  retired, never edited. Text that passed redaction and must not be kept
  cannot be taken out yet. A human may blank it
  ([ADR 0010](adr/0010-events-stay-and-a-human-may-blank-text.md)), but
  nothing does, and it cannot be done by hand. On a throwaway home on
  2026-11-23, with the guard against updates dropped, DuckDB changed an
  event's text but refused to change its payload once another row referred
  to the event. The text it changed was still in the database file after a
  checkpoint, until later writes took the space (seen with compression off,
  where text can be searched for). Until a blanking is built (see "Proposed
  directions"), what is left is to rotate a secret or to delete the home.
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
- A session that `/clear` started in a terminal holds two
  `session_start` events, the hook's and the transcript's, when a `mem`
  from before 2026-11-23 captured it: the adapter keyed a
  `SessionStart` whose source is `clear` on the time it ran. The two
  sessions of that day's check hold both. `/clear` in the app was last
  recorded on 2026-11-11, where it said `startup`. What it says there
  now is not checked; either is recorded once (`adapters/droid.py`).
- A Droid turn that never reaches `Stop` (a crash, a killed session) is
  in the ledger only after the session's next `Stop` or `SessionEnd`, or
  after someone imports its transcript. Droid writes a turn's outcome
  after its `Stop` hook has run, so the outcome of a session's last turn
  arrives at `SessionEnd`, 30 minutes after the session was last used.
  Until then, and for good if `SessionEnd` never runs, the judge reads
  that turn as cut off when the session ended (`closed_by` is
  `session_end`), not as completed. A session with no interface raises
  `SessionEnd` as it finishes. Quitting Droid in a terminal raises it
  and runs no hook to its end (see the next entry). Not checked:
  closing the app. Two of the 13 sessions a Task call started had
  written 214 and 295 transcript lines and reached neither hook within
  the hour.
- When Droid exits in a terminal, by `/quit` or by Ctrl+C pressed twice,
  every command of its `SessionEnd` hook fails: exit code 1 within a
  third of a second and no output, seven times of seven on 2026-11-23,
  with Droid 1.17.2 and 1.18.0. `mem event` records nothing then. Such
  a session has no `session_end`, and the outcome of its last turn
  stays unread, as in the entry above, until the session is resumed and
  reaches a `Stop`, or someone imports its transcript. Why the commands
  fail is not known: a script that only appends a line to a file failed
  the same way. A `SessionEnd` that Droid raises while it goes on
  running, after `/clear` or a change of session, runs as it should.
- Hooks that another tool declares in Droid's `settings.json` do not
  run at the events where `hooks.json` holds mem's, and `mem hooks`
  says nothing of it. Droid runs an event from `settings.json` only
  when `hooks.json` in the same directory does not define it. In one
  session a command declared in `settings.json` at six events ran at
  each of 407 `PostToolUse` hooks, which `hooks.json` does not define,
  and at none of 59 hooks of the other five. `mem hooks install`
  refuses to create `hooks.json` while `settings.json` holds hooks, on
  the older rule that Droid then reads none of them. It does not look
  there once `hooks.json` exists, and `mem hooks status` never does.
  The README states the older rule (`hooks.py`).
- No session records Droid's version. `sessions.harness_version` is
  empty for all 166 Droid sessions of the production home. No hook
  payload carries the version, and the `version` in a transcript's
  first entry, which an import takes for it, is the format's: 2 for a
  session of 1.17.2 and for one of 1.18.0 (`adapters/droid.py`).
- The first import of a long session runs inside its `Stop` hook and
  takes seconds: 8.75 s for a transcript of 41.9 MB, 13.2 s for one of
  61.0 MB, and 13.6 s for one of 79.3 MB in a running Droid on
  2026-11-23, where the person could type meanwhile and Droid showed no
  error. Each later `Stop` reads the whole file again to find its
  place, 1.0 s at 61 MB. Droid's hook timeout is 60 s. Not checked:
  what a hook killed at the timeout leaves behind.
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
- A server killed while it writes a snapshot leaves what it had written
  in `export/.tmp-<timestamp>/`, and nothing removes it: up to the size
  of a snapshot each time. A signal waits for the snapshot to end, and
  launchd kills a server that has not stopped 30 s after it asked
  (`ExitTimeOut` in the plist). A killed `mem export` leaves the same
  (`db.py`).
- The server's memory limit does not bound a result on its way to a
  client. While `mem export` copied 10 million events through a server
  limited to 4.7 GiB, the server process held 20 GiB, and 9 GiB when
  limited to 1.8 GiB. DuckDB counts the pages of the database it keeps
  and what a query works with, and the rest of what the process held
  then is outside the limit. The server's own snapshot adds about 1 GiB
  to the limit, so what reaches this is a client that asks for millions
  of rows: `mem export`, or a large `mem sql` (`db.py`). See "Growth".
- `mem export` through a running server holds 7.0 GiB in its own
  process at 1 million events and 8.5 GiB at 10 million, and `mem audit
  check` 1.3 and 8.5 GiB. The server's limit is not theirs: each client
  has a DuckDB of its own with DuckDB's default limit (`db.py`,
  `audit.py`).
- Nothing removes the logs a home gathers beside the database: launchd
  appends to `serve.log` and `admit.log` without end (`scripts/`).
- A hook whose write is still going at its deadline puts its records in
  the spool, and the server may commit the same write a moment later.
  Nothing is stored twice, because replay finds the records present,
  but the file waits until the server next starts or someone runs
  `mem ingest --spool` (`capture/pipeline.py`). A server that replayed
  now and then while it serves would end the wait, and would purge
  `spool/done/` as often, which today happens only when it starts.
- A tool call's input is stored twice: in the text of the event, and as
  `tool_input` in its payload, which has no size limit. The largest
  payload measured was 713 KB (`adapters/droid.py`).
- A recall reads a number with a point in it as two words: `3.12` is
  `3` and `12`, so a query of `3.12` ranks a memory about 3.12 first
  and also returns one about 3.13. A word that holds a digit is one
  word, so `python` does not find `python3`
  (`migrations/0010_recall_digits.sql`, `mcp_server/ranking.py`). See
  "Recall by a number" in [phase-1-spec.md](phase-1-spec.md).
- Memories written in the same moment conflict, and all but one are
  refused. The index over the text of memories is maintained by every
  write, and two writes in flight change the same rows of it. On a
  throwaway home, of 2, 4, and 8 `remember` calls released together,
  one was written each time, in ten rounds of each, and every other got
  "memory write conflict or validation failure" as a tool error. With
  the index dropped all were written. A refused write is not queued, so
  the memory is lost unless the agent calls again. One write takes
  about 30 ms, so this takes two writers inside the same 30 ms. Not
  checked: the other writers of memories, which are admission, a replay
  of queued writes, `relate`, and an entity merge
  (`mcp_server/tools.py`, `lifecycle.py`,
  `migrations/0010_recall_digits.sql`). See "Matching text over events"
  under "Growth".

Reported in the same audit and not yet checked against the code: the
hand-maintained `ALL_TABLES` list; lockdown happening after the listener
starts; the experiment-home guard excluding only the default path; the
approver not pinning question, projection, and drafter versions;
`--similarity-floor 0` and `--min-writes 1` weakening the auto-approval
guards; turns still in flight being admitted and then frozen; the Go
helper's stderr being discarded; the drafter's name leaking through
blinding; an egress error body in exception text.

## Proposed directions

Proposed in the audit of 2026-11. None of these is approved but the two
that "Next approved work" names: finishing capture, and the usefulness
experiment.

- Finish capture. Done. Sessions were captured end to end through
  installed hooks on 2026-11-12, one of them in a terminal, and
  `/clear` and the end of a session in a terminal and a long first
  import in a running Droid were seen on 2026-11-23.
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

Proposed on 2026-11-22, from a reading of Mastra's memory, the memory
of Hermes Agent, and Anthropic's "The components of agent memory". The
evidence behind them is vendor documentation and a vendor's research
post, with one outside probe of Mastra made in one run; none of it is
peer reviewed. Each entry is an issue in Linear (see
[agents/issue-tracker.md](agents/issue-tracker.md)), named here by its
identifier. They were triaged the same day, and each entry ends with
what was decided: approved (see "Next approved work"), waiting for what
it names, or out of the tracker.

- MEM-1. The design of the usefulness experiment names a stratum of
  tasks that hinge on an exact value or on a tool's output, counts a
  confident wrong reply and "no recall made" on their own, and names
  the strength of the model and the wording of the MCP instructions as
  factors. A change to the design only. Approved.
- MEM-2. Measure what matching text over events costs, with an index
  and without one, at the sizes under "Growth" and under the server's
  memory limit. Approved, and measured on 2026-11-23: see "Matching
  text over events" under "Growth".
- MEM-3. An agent can recall events by text, as a new strategy for the
  recall path. Waiting for the usefulness experiment to show where
  recall of memories fails. What MEM-2 measured favors a match with no
  index, bounded to a span of time.
- MEM-4. An agent can read on from a recalled event: the events around
  it, or a session's first and last messages. Waiting with MEM-3.
- MEM-5. Report how many captured sessions make a recall at all, by
  model and by caller. It reads the production home, and is approved
  once the user agrees to that read.
- MEM-6. Check in a running Droid what the stdout of a `SessionStart`
  hook does: whether it reaches the agent, and at which kinds of start.
  Approved, and checked on 2026-11-23: Droid gives it to the agent at a
  new session, after `/clear`, at a resume, and after a compaction, as
  a message that only the agent sees and that an import records. See
  "What a terminal and a `SessionStart` hook's stdout showed" in
  [phase-1-spec.md](phase-1-spec.md).
- MEM-7. `mem` prints a bounded block of active memories for a project
  and records the retrieval under a strategy of its own. Waiting for
  the number MEM-5 reports: if most sessions already make a recall,
  this is dropped.
- MEM-8. A Droid session is given that block once when it starts, by a
  second hook command that a user opts in to; `mem event` still prints
  nothing. Waiting for MEM-7, and it needs a record under `docs/adr/`.
  MEM-6 found that such a block reaches the agent, that the hook runs
  again at every resume and compaction, and that an import records the
  block as text of the session.
- MEM-9. Admission records, without a model, whether the exact values
  in a rule are in the lines its claims cite. It changes no action.
  Approved.
- MEM-10. A dry run reads active memories against their source events
  with a local model, and writes no memory and no review action.
  Waiting for a count: how many active memories in the production home
  name source events.
- MEM-11. That re-check records a confirmation, or a flag for a human.
  It never approves, retires, or accepts. Waiting for what the dry run
  of MEM-10 finds. Decided at triage: a memory the re-check finds
  unsupported stays active until a human acts, because nothing yet
  says how often the re-check is right.
- MEM-12. `mem review` lists the active memories that recalls return
  and no reply cites. It says nothing until a ledger in use holds
  citations, so it is out of the tracker until one does: canceled
  there, not refused.
- MEM-13. A memory with invisible characters or text that reads as an
  instruction starts quarantined, whatever trust level its writer
  claims. Approved.

The same reading adds to one entry of the audit's list above. Hermes
Agent's documentation warns that two agents sharing one memory compound
each other's entries, and Mastra deprecated the scope that shared
observations across threads. This ledger is shared by design and its
`agent_id` is self-reported, which is a further reason for per-harness
tokens.

Proposed on 2026-11-23, with the decision that events stay
([ADR 0010](adr/0010-events-stay-and-a-human-may-blank-text.md)):

- A human can blank the text of an event or a memory. A blanking writes
  the database anew without the text, as a restore writes it from a
  snapshot, so it waits for the verified restore in the audit's list
  above, and is not an issue until that is built. Its design has to say
  which text a blanking reaches beside an event's content, payload, and
  blob and a memory's content (a session's title and directory, a
  recall's query), and what it shows a human of the rows that refer to
  a blanked one.

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
- What happens to `example-memory-db-schema.sql` and the prototype branch?
- What would Phase 5 be for? Anthropic reports that a wiki beside a
  search tool met 61% of the graded criteria of 25 questions, against
  53% for the search tool alone, and in a second test 87% against 73%
  on the questions that need several documents. Those pages combined
  sources, and one wiki grew to 1.77 times its corpus. A read-only
  rendering of active memories combines nothing: it would serve a
  person browsing, not recall. This bears on whether Phase 5 stays
  (above).
