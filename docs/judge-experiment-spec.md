# Judge experiment spec: decision models as failure and lesson triage

Status: approved and implemented, 2026-10-26 (see "As built" at the end).
Builds on Phase 0 and Phase 1 (`PLAN.md`, `docs/phase-1-spec.md`, migration
0001). Not part of the Phase 4 roadmap yet; this is the bounded experiment
that decides whether a decision model earns a place there. The experiment
runs against a separate home and never touches the production ledger.

## Goal

Answer one question with this project's own sessions: over a state
projection built from ledger events, can a calibrated decision model
(a) decide whether a turn failed in a way worth a lesson and (b) decide
whether a drafted lesson is supported by the events it cites, within 3
points of a reasoning LLM judge, at a fraction of its cost and latency, with
a confidence signal that separates its own errors, and what write and discard
error rates result at a threshold frozen before a live test?

Grounding: the paper's §6.5 ("failure-derived lessons require a judge of
failure, and self-judgement is noisy") and §13.3 (trust-scoring calibration
has a cost and no reported false-quarantine rate). External: JEV-as-a-Judge
(arXiv 2609.26550, within ~3 points of a frontier judge where the verdict is
readable from the text, confidence detects its own errors at AUROC 0.76 to
0.84), the false-success study (2606.09863, 45 to 76% of agent failures are
false success and LLM judges reach at most 0.65 AUROC on them), and the
trajectory-judge study (2609.00038, outcome-only judges have no paired
discrimination against clean parents).

The role under test is a cheap, calibrated *triage* layer sitting between
deterministic ground truth (outcome, tool errors, the user's next message)
and an expensive reasoning judge or a human. It is not a sole judge of
failure.

## Non-goals

- Judging code correctness. Tests and exit codes do that; here they are
  inputs.
- Writing `lesson` memories into any ledger. Lessons stay in run files.
- Hooks, Phase 2 adapters, or comparisons on published benchmarks.
- Any schema change. Verdicts are `decision` events; no migration 0002.

## Data

- Source: local Droid transcripts under `~/.factory/sessions`
  (`<project slug>/<session id>.jsonl`), imported with the existing
  `DroidAdapter` into a separate experiment home, `~/.agent-memory-judge`
  (`AGENT_MEMORY_HOME` or `--home`). Redaction, blobs, FTS, and `why_answer`
  all apply unchanged. `mem judge` refuses to run against the default home.
- Unit of judgment: a *turn*, a `user_msg` and everything up to the next
  `outcome` event in the same session, ordered by `(ts, seq, recorded_at)`.
- Ground truth already in the ledger (verified against real transcripts):
  `outcome.content` (`completed` 2457, `cancelled` 669, `error` 5, provider
  errors 8 across the local corpus), `tool_result.payload.is_error` (3,529
  true), and the next `user_msg`.
- `agent_turn_outcome` entries in Droid transcripts carry neither
  `timestamp` nor `seq`, and `resultKind` is `text` or `null`, never
  `success`. The adapter now gives such entries the previous entry's time
  plus one millisecond and its `seq`, so a sort by `(ts, seq)` keeps them
  inside their turn. The fixture was corrected and a second fixture,
  `droid_session_judge.jsonl`, covers a clean turn, a repeated tool error
  followed by a user correction, a cancelled turn, and a provider error.

## Pipeline

```
transcripts -> mem judge import -> experiment ledger
            -> candidates (provenance_v1, triggers_v2, seeded sample) -> projection_v2
            -> draft (drafter_v1)
            -> decide (rules | readout | jev | llm) -> verdicts + decision events
            -> label (blinded)  -> report (calibrate, freeze, held-out)
            -> live (new sessions, frozen thresholds)
```

### Who is speaking (`provenance_v1`)

A Droid transcript's `user` role has three authors, and the pilot run
(exp1, below) showed the triggers cannot tell them apart on their own. A
person types in a top-level session. A parent agent writes the opening task
and every follow-up of a session it spawned (Task workers, reviewers,
explorers, Mission orchestration). The harness injects system reminders,
async completion notices, and interruption markers. In this corpus 1,741 of
2,251 sessions were spawned by another session, and 6,076 of 9,699 turns
were opened by harness text.

Two tags, both deterministic:

- **Caller** of a session: `human`, or `agent` with the spawned droid's role
  from its title prefix (`worker`, `task-coordinator`, `security-sweeper`,
  …). Read from `callingSessionId` on the transcript's `session_start`
  entry; the adapter now records it in session metadata, and sessions
  imported before that are resolved from the transcript the import log
  points at.
- **Provenance** of each user message: `harness` when, after removing
  `<system-reminder>` and `<system-notification>` blocks, nothing is left or
  what is left is a harness notice ("Request interrupted by user", "Script
  Output from …", "Background task completed", …); `parent_agent` when the
  session's caller is an agent or the text is a Task prompt or
  `# Follow-up Instructions`; `human` otherwise. User messages longer than
  4 KiB are stored truncated with the full text in a blob, and the person's
  request often sits after the reminder blocks, so truncated messages are
  completed from the blob before tagging or projecting.

Harness-opened turns are segmented (so the agent's reaction to a notice is
not glued to its neighbour) but are not candidates, and a harness message
never counts as a turn's "next message": that is the next message a person
or a parent agent wrote.

### Candidates (`triggers_v2`)

Deterministic flags from ledger facts only; the agent's own claims are never
read:

- T1 `outcome = 'cancelled'`.
- T2 `outcome IN ('error', 'model_provider_unavailable', 'model_rate_limited',
  'model_authentication_failed')`.
- T3 at least one `tool_result` with `payload.is_error = true`.
- T4 the next message is from a person and matches a correction lexicon
  (regexes in `triggers.py`, recorded by version).
- T5 the next message matches the lexicon and is from a parent agent: a
  reviewer's opinion rather than the requester's.
- T0 clean control: `completed` with none of T1 to T5.

A turn enters the pool only if a person or a parent agent opened it and it
did work (an assistant message or a tool call). Stratum is the first matching
flag in priority order T2, T1, T3, T4, T5, T0. Each item records its caller
and the provenance of its opener and next message.

The sample is seeded and stratified twice. `human_share` (default 50%) of
`n` comes from sessions a person started and the rest from sessions an
agent started; a short side is back-filled from the other. Within each side:
every T2, the rest of the triggered budget split round-robin across T1, T3,
T4, T5, and `clean_share` (default 25%) from T0, the "clean parents" for
paired discrimination; when the triggered strata run dry the remainder comes
from T0 so the sample keeps its size. Each item is assigned to the
calibration or held-out split (60/40, stratified by stratum and caller) at
sampling time. The report scores each caller's items separately as well as
together.

Per turn the pipeline also records facts the rules backend and the report
share: tool count, error count, which tools errored, whether the same tool
errored twice, whether a later result succeeded, and whether the last result
was clean.

### State projection (`projection_v2`)

Deterministic, versioned, hashed. One header line (project, who started the
session, outcome, tool and error counts), then numbered evidence lines: the
request (≤1,500 chars, labelled `human` or `parent agent`, harness blocks
removed with a note saying how many), each tool call (name and input, ≤200)
with its result (`ok` ≤120 chars, `error` ≤300), the final assistant text
(≤1,500), the recorded outcome, and the next message (≤1,000, labelled with
its author) or a note that the session ended. At most 12 tool calls are shown (first and last
six); the whole text is capped at 24,000 characters. The redactor runs again
over the finished text. Event ids never appear in the text; a sidecar maps
line numbers to events so a cited lesson can be traced without the state
leaving the machine. Projections are written 0600 under the run directory
and `mem judge purge` deletes them.

### Drafter (`drafter_v1`)

A local Ollama model writes a lesson of at most two sentences and lists the
evidence line numbers it relied on, for every item including clean controls,
so Stage B has "unsupported lesson from a clean turn" negatives. Output is
JSON; the lesson is normalised to one line and capped at 400 characters.
Drafting is local only.

### Questions (`questions_v1`)

Stage A, on the projection alone:

| id | type | asks |
|---|---|---|
| `failed` | noul | the agent did not accomplish the request, judged by tool results, outcome, and the next user message |
| `user_corrected` | noul | the next user message corrects or redirects the work |
| `env_confirmed` | noul | a tool result independently confirms the claimed completion |
| `category` | choice | wrong_assumption, tool_misuse, stale_or_missing_knowledge, environment_or_provider, user_redirect, none |
| `recurrence` | noul | the cause is project-specific and likely to recur |

Stage B, on the projection plus the draft:

| id | type | asks |
|---|---|---|
| `supported` | noul | every claim in the draft traces to cited evidence |
| `generalizable` | score 0..2 | one-off, project-reusable, general practice |
| `actionable` | noul | the draft says what to do differently |

Derived action: `write_lesson` when `failed`, `recurrence`, and `supported`
all clear their thresholds; `review` when any of the three lies inside the
uncertainty band (default 0.15) around its threshold; else `discard`.
Defaults are 0.5 everywhere until `--freeze` replaces them per backend.

### Backends

One interface, four implementations:

```python
class DecisionBackend(Protocol):
    name: str  # "rules", "readout:gemma4:26b", "llm:gemma4:26b", "jev:typesafe/jev-1.14"
    local: bool

    def decide(self, inp: JudgeInput, questions: Sequence[Question]) -> dict[str, Answer]: ...


# Answer: value, probabilities over options, confidence (= max probability unless the
# service reports its own), method, latency_ms, cost_usd, model_snapshot
```

- `rules`: trigger flags and the shared facts (a repeated same-tool error
  means recurrence; a draft that names a tool or several tokens from the
  turn counts as supported). The floor.
- `readout:<model>`: LLM-as-Jev over Ollama's native `/api/chat`. One
  question per call, options as letters, `num_predict` 4, thinking off, and
  the distribution read from `top_logprobs` on the first letter token. If
  the server returns no logprobs, eight samples at temperature 1 with
  add-half smoothing, and the answer records `sampling:8` instead of
  `logprobs`.
- `llm:<model>`: the reasoning baseline. Same rubric, all questions in one
  prompt, JSON output, thinking on, temperature 0, self-reported probability
  as confidence.
- `jev`: the OpenRouter Decisions API (`POST /api/alpha/decisions`, model
  pinned to `typesafe/jev-1.14`), standard-library HTTP, key from
  `OPENROUTER_API_KEY`. It sends redacted state off the machine, so it runs
  only with `AGENT_MEMORY_JUDGE_ALLOW_REMOTE=1`. State is never logged.

Stage A is asked on the projection alone; Stage B on the projection with the
draft appended, so the draft cannot colour the failure verdict.

Every verdict is written to `verdicts/<backend>.jsonl` and appended to the
experiment ledger as a `decision` event in a `human` session
(`harness_session_id = judge-<run>`, project `judge-experiment`, scope
`judge`). The payload carries run id, item id, backend, model snapshot,
question and projection versions, projection hash, trigger flags, each
answer's value, distribution, confidence and method, latency, cost, the
derived action and its reasons, and the thresholds in force. The event id
is derived from the backend, item, question version, projection hash, and
draft hash, so re-running a backend on unchanged inputs is a no-op in the
ledger, while a verdict made before a draft existed is superseded in the
verdict file and appended to the ledger as a second event once the item has
a draft. No state text enters the ledger.

## Labeling (`mem judge label`)

Interactive review in the terminal, rendered with Rich. The reviewer sees the
projection and the draft, never a verdict, in a seeded random order. Per item:
`failed` (y/n/u), `category`, `worth_remembering` (y/n); when a draft
exists `supported` (y/n), `generalizable` (0/1/2), and `enforced_by`
(none/harness/project_docs/tooling); when the turn had a tool error,
`error_intended` (y/n). `s` skips, `q` stops; labels append to
`labels.jsonl` and the next session resumes from there. When a question is
added after labeling has started, already-labeled items that it applies to
come back first and ask only that question; their row is rewritten in place
with the merged answers and a `relabeled_at` stamp.

The session opens with a rubric, and the draft panel names the projection
lines the drafter cited and shows them again as excerpts, so `supported` can
be answered against the lines it points at. The rubric, in short:

- `failed`: the agent did not do what the user asked in this turn. Decide
  from tool results, the recorded outcome, and the user's next message,
  never from the agent's own summary. `u` when the evidence does not say
  (for example a cancelled turn with no agent output after the cut).
- `category`: the main cause, or `none`. A clean turn is `none` even
  when the agent's work was imperfect in ways the user did not object to.
- `worth_remembering`: a lesson from this turn would help a future session
  on this project or any project. One-off slips, outages, and user
  redirects are `n`. A `none` turn is almost always `n`; the drafter
  writes a lesson for every item, including the clean ones, and those
  drafts are the negatives Stage B has to reject.
- `supported`: every claim in the draft can be traced to something in the
  projection, in particular the cited lines. Evidence that the agent did X
  supports "the agent did X"; it does not support "X caused the failure" or
  "always do X" unless the projection shows that too. Vague, true-anywhere
  advice ("verify your work") is `supported` only if the turn actually
  shows the check being skipped or done; it will usually fail
  `worth_remembering` instead.
- `generalizable`: 0 if it only applies to this request, 1 if to future
  sessions on this project (a convention, a tool quirk, a layout fact), 2
  if to any project.
- `enforced_by`: is the lesson already enforced somewhere the agent meets
  anyway? `harness` when the Mission orchestration prompts, the system
  prompt, or subagent constraints already demand it ("report back to the
  coordinator", "do not touch files outside your feature"); `project_docs`
  when AGENTS.md or the repo's guidelines already say it; `tooling` when
  lint, type checks, CI, or a hook would catch the slip; `none` when memory
  would be the only place. A true, supported, general lesson can still be
  redundant, and writing it would only add noise to recall. Judge the
  lesson, not the turn.
- `error_intended`: a failed tool call is sometimes the point: probing
  whether a file exists, a test expected to fail before the fix, a grep
  with no match. `y` when the agent expected or used the error. Such a turn
  is usually `failed=n`; this label exists so the report can say how often
  the tool-error trigger (T3) fires on turns that were fine.

Labels are per item and independent of each other: a draft can be
`supported` and still not `worth_remembering`, and a turn can have
`failed=y` with a draft that is not `supported` because it names the
wrong cause.

## Metrics and procedure (`mem judge report`)

Verdicts are joined to labels by item and to the current projection by
hash. On the calibration split, per backend: accuracy, AUROC, Brier, and
ECE (10 bins) for `failed` and `supported`; the AUROC of the backend's
confidence as a detector of its own errors; `category` and `generalizable`
accuracy; paired discrimination (share of same-project (failed, clean)
label pairs where the failed turn received the higher `failed`
probability); mean latency and total cost per item; and the write decision
against the label: a write is correct when the item is `worth_remembering`
and `enforced_by` is `none` (or absent); the report gives precision of
`write_lesson`, miss rate among items that should be written, and the count
of writes that landed on an already-enforced lesson. Before the splits, a
"labels on their own" section reports `enforced_by` by who started the
session and, over labeled turns with a tool error, the share where the error
was intended and the share labeled failed (the precision a bare T3 trigger
would have).

The reference judge is the `llm:` backend with the most verdicts. For every
other backend the report fits a cascade: accept the cheap verdict when its
confidence reaches a gate `q`, otherwise pay for the reference; `q` is the
cheapest gate whose cascade accuracy is within 1 point of the reference.

`--freeze` chooses, on calibration items only, each backend's
accuracy-maximising cut on `failed` and `supported` and its cascade gate,
writes them to `run.json`, and refuses to run twice. The held-out split is
scored only after a freeze; before that the report says so and stops.

### Live week

`mem judge live --run <id> --since DATE` imports sessions modified since the
date, segments every turn of sessions started since then that is not already
an item, projects, optionally drafts, and decides with the frozen thresholds
into `judge/<run>/live/<day>/`. At the end of the week, label every
`write_lesson` and a random 30 `discard`s (`mem judge label --live <day>`)
and report precision and miss rate.

## Success criteria (stated before running)

1. A decision backend within 3 points of the reference on `failed` and
   `supported` on the held-out split, at ≤10% of its latency (and cost,
   for a hosted model): adopt it as the Phase 4 triage layer.
2. Confidence AUROC on its own errors ≥ 0.75: the cascade is viable.
3. The cascade at the frozen gate: accuracy ≥ reference − 1 point at ≤50%
   of its cost.
4. `rules` within 3 points of the reference on Stage A: no model for
   Stage A; models only for Stage B.

If (1) fails for both model backends, the decision-model role in Phase 4 is
limited to write-time admission screening and this spec closes with the
numbers.

## CLI

All under `mem judge`, all refusing the default home:

| command | does |
|---|---|
| `import --sessions-dir PATH [--limit N] [--project SLUG] [--since DATE]` | write transcripts to the experiment ledger; unchanged files are skipped |
| `candidates --run ID [--n 200] [--seed 1] [--clean-share 0.25]` | segment, trigger, sample, project; creates the run |
| `draft --run ID [--drafter ollama:gemma4:26b] [--limit N]` | draft lessons; resumable |
| `decide --run ID --backend SPEC [--backend SPEC ...] [--no-events]` | run backends; resumable per backend |
| `label --run ID [--limit N] [--seed 1]` | blinded review |
| `report --run ID [--freeze] [--raw]` | score, optionally freeze, write `report.md` |
| `live --run ID --since DATE [--backend SPEC ...] [--drafter SPEC] [--day D]` | one live-week step |
| `status --run ID` | items per stratum, drafts, verdicts, labels, frozen or not |
| `purge --run ID` | delete projection text; verdicts and labels stay |

`--live DAY` on `draft`, `decide`, `label`, `report`, and `status` targets a
live-week sub-run.

## Layout

`src/agent_memory/judge/`: `store.py` (run directory, home guard),
`provenance.py` (session callers, message provenance), `triggers.py`
(segmentation, triggers, sampling), `projection.py`,
`questions.py`, `drafter.py`, `backends/{base,http,rules,readout,llm,jev}.py`,
`pipeline.py` (import, candidates, draft, decide, live, purge),
`labeling.py`, `metrics.py`, `report.py`, `cli.py` (`judge_app`, registered
on `mem` next to `spool`). Run directory:
`<home>/judge/<run>/{run.json, items.jsonl, projections/ (0600),
drafts.jsonl, verdicts/<backend>.jsonl, labels.jsonl, report.md, live/<day>/}`.
Import bookkeeping in `<home>/judge/imports.jsonl`. New dependency: `rich`
(already installed by typer; declared because it is imported directly).
HTTP is standard library.

## Tests

`tests/test_judge.py` (no server): segmentation and every trigger on the
judge fixture; message provenance and caller parsing; segmentation across
harness and parent-agent messages (T4 vs T5, harness turns excluded, next
message skips harness notices); seeded, stratified sampling with the
human/agent quota and splits; projection determinism, caps, sidecar,
redaction, and harness-block stripping; question set and derived action;
rules backend contract; backend spec parsing; Jev request shape and response
parsing against a recorded response, plus refusal without opt-in and key;
readout logprob normalisation and the sampling fallback; LLM JSON parsing;
drafter parsing; metrics on arrays with known values; cascade and gate
choice; store guard, run id check, 0600 projections, purge; report locks
held-out until freeze and refuses a second freeze; labeling hides verdicts
and resumes; backfill asks only the questions a labeled item is missing and
rewrites its row; `should_write` and the labels-only report section.

`tests/test_judge_pipeline.py` (running server): import is idempotent and
skips unchanged files; candidates, projections, and decide with `rules`;
decision events land with the required keys and without state text;
`mem judge status` and `purge`; the CLI refuses the default home and the
`jev` backend refuses without the opt-in variable.

## Security

Hosted calls are opt-in, over projections that were redacted at import and
again at projection time; the server token is a redaction rule. Projections
are 0600 under the experiment home and `purge` removes them. Verdict files
and ledger events exclude state. The experiment home is never the production
home.

## As built

Implemented 2026-10-26 as specified, with these deviations:

- HTTP uses `urllib` rather than `httpx`; `httpx` is not a declared
  dependency of the project.
- The readout backend uses Ollama's native `/api/chat` with
  `logprobs`/`top_logprobs`; the OpenAI-compatible endpoint does not return
  logprobs.
- Thresholds and cascade gates are frozen per backend, not globally, because
  each backend's probabilities are calibrated differently.
- Labeling renders with Rich instead of plain `input()`.
- The Droid adapter change for timestamp-less `agent_turn_outcome` entries
  was required for segmentation on real data and is covered in
  `tests/test_adapters.py`.
- `provenance_v1`, `triggers_v2`, and `projection_v2` were added after the
  pilot run below showed who the "user" is in this corpus. The adapter also
  records `calling_session_id` and `calling_tool_use_id` in session metadata.

### Run `exp1` (2026-10-26): pilot, superseded by `exp2`

exp1 was built with `triggers_v1`, which read every `user_msg` as the
person's words. Labeling it was abandoned once the sample was inspected; it
is kept as the pilot, and its probe findings below still hold.

- 1,741 of 2,251 sessions were spawned by another session (one Mission
  orchestrator spawned 1,076). 145 of exp1's 200 items came from those.
- T4 "user corrections" were 82% misfires: of 39, 7 were the person, 17
  were harness `<system-reminder>` text, 14 were `# Follow-up Instructions`
  from a parent agent, 1 was a background-task notice.
- Many turns were opened by harness text ("Script Output from …", a
  reminder alone), so "did the agent do what the user asked" had no user in
  it; with `triggers_v2` 6,076 of 9,699 turns are harness-opened and leave
  the pool.
- 2,525 user messages are stored truncated at 4 KiB with the rest in a
  blob; the person's request often follows the reminder blocks, so both
  segmentation and `projection_v1` sometimes saw only the reminder.

Data notes and probes from exp1:

- Import: 2,231 transcripts with content (20 held only a `session_start`),
  300,640 rows, no failures. Outcome counts match the pre-import survey
  exactly (completed 2457, cancelled 669, error 5, provider errors 8).
- Segmentation: 9,699 turns, 2,973 with an outcome (166 outcome events are
  consecutive duplicates and attach to no turn). Every T2 turn (13) and 523
  of the 564 cancelled turns are `user_msg -> outcome` with no agent output
  at all: the user stopped the agent before it answered, or the provider
  never answered. There is nothing to judge in them, so the "did work" rule
  removes them; **the T2 stratum is empty in this corpus** and T1 holds 41.
  Pool: T1 41, T3 1,467, T4 39, T0 1,897. Sample: T1 41, T3 70, T4 39,
  T0 50 (120 calibration / 80 held-out, 11 projects). Projections run 741 to
  8,506 characters (median 5,871); one redaction fired.
- `rules` over all 200 before any drafts existed: 131 discard, 69 review,
  no `write_lesson`. 200 `decision` events landed in one `judge-exp1`
  session.
- Probes on `gemma4:26b`: drafter 35 to 45 s per item; `readout` 3.3 s for
  all eight questions (the evidence prefix is shared across questions, so
  Ollama's prompt cache absorbs it); `llm` with thinking 97 s. Readout
  confidences on the probe item were all ≥ 0.99 at temperature 0, which is
  the over-concentration the literature warns about for instruction-tuned
  readouts; ECE on the labeled set will say whether it matters.
- The first full `draft` pass stalled on item 7: gemma4 is a thinking
  model and Ollama thinks by default, so the drafter spent its whole budget
  (and more than 300 s) thinking. With `think: false` and a 400-token cap
  a draft takes about 1 s; the drafter probe times above were thinking time.
  `draft` now skips a failing item and continues, HTTP retries one timeout,
  and the `llm` judge is capped at 4,096 tokens including its thinking.
- That shared cap bit the `llm` pass too: on 4 of the first 180 items the
  model thought until the budget ran out and returned empty or cut-off JSON.
  Empty content was skipped as intended; cut-off JSON escaped as a raw
  `JSONDecodeError` and ended the pass at item 180. Both now raise
  `JudgeError`, and when Ollama reports `done_reason: length` (or the JSON
  is unusable) the backend asks the same question once more with thinking
  off and tags those answers `llm_json_nothink`, so the report can count
  how often the baseline had to drop its reasoning. The retry also drops
  `format: json`: grammar-constrained output with no thinking channel is
  where Gemma-family models emit whitespace until the budget is gone, and
  the object is extracted from free text anyway. Budget exhaustion is not
  deterministic; the same item that failed in the run answered in 29 s
  (2,151 tokens) when retried by hand.
- Resuming exposed a second rule: `draft` and `decide` treated a failure
  on the first item as misconfiguration and raised, but on resume the first
  undecided item is exactly the one that failed before. Both now fail fast
  only when the run has produced nothing for that backend yet; otherwise
  the item is skipped, counted in a closing line, and retried next run.
- The ten drafts made with thinking on were discarded and all 200 items
  were redrafted with one drafter configuration: 200/200 in about 7 min,
  no failures. `rules` was then run again; every verdict was superseded
  because each item now had a draft (122 discard, 69 review, 9
  `write_lesson`), and the ledger holds both generations: 402 `decision`
  events, one session.
- Projected cost of the remaining passes on this machine: readout ≈ 15 min,
  llm ≈ 5.5 h. Verdicts made before an item had a draft are superseded
  automatically on the next `decide`.
- The `llm` pass reached 158 of 200 before exp1 was retired.

### Run `exp2` (2026-10-26): `--n 200 --seed 2`

- Pool after `triggers_v2`: 1,648 turns (human 247, agent 1,401); T1 8,
  T3 580, T4 8, T5 19, T0 1,033; T2 still empty.
- Sample: 100 human-caller items (T1 6, T3 61, T4 8, T0 25) and 100
  agent-caller items (T1 2, T3 54, T5 19, T0 25); agent roles
  task-coordinator 34, worker 24, security-sweeper 17, untitled 16, others
  9. Calibration 120 / held-out 80, balanced by caller. 21 projections had
  harness blocks removed; none was harness-only. Median projection 5,180
  chars, max 8,585; 8 redactions.
- Every T4 and T5 is in the sample (the strata are small), so the report's
  "user corrected" slice is the whole population, not an estimate.
- Two labels were added at 103/200 labeled, after reading the first drafts
  on Mission-spawned sessions: many were true, supported, and general, yet
  restated what the Mission prompts already enforce ("report progress to the
  coordinator"), and several T3 turns had tool errors the agent meant to
  cause. `enforced_by` and `error_intended` (see Labeling) were backfilled
  onto the 103 existing rows before labeling continued. The five scored
  labels were not changed; the new labels only alter what counts as a
  correct write and add the labels-only section.

Results from labeling, the frozen thresholds, the held-out split, and the
live week are appended below when available.
