# Judge experiment spec: decision models as failure and lesson triage

Status: approved and implemented, 2026-10-26; exp3 changes implemented
2026-11-07; structured drafts and per-claim Stage B implemented 2026-11-08
(see "exp4 design: structured drafts and per-claim Stage B" below).
Builds on Phase 0 and Phase 1 ([roadmap.md](roadmap.md),
`docs/phase-1-spec.md`, migration
0001). Not part of the Phase 4 roadmap yet; this is the bounded experiment
that decides whether a decision model earns a place there. The experiment
runs against a separate home and never touches the production ledger.

For setup commands and the step-by-step workflow, see
[Running judge experiments](judge-how-to.md).

This spec describes what new runs do. Definitions that only stored runs
still use (`drafter_v1`, `questions_v1`, the first labeling screens) are in
[archive/judge-earlier-versions.md](archive/judge-earlier-versions.md).
The logs of the runs exp1 to exp4, with the designs decided at the time,
the success criteria, and the test plan, are in
[archive/judge-experiments.md](archive/judge-experiments.md).

Production lesson admission is a separate subsystem, specified in
[admission-spec.md](admission-spec.md). It reads frozen v4 human labels
and composes guard-free judge helpers. Experiment guards and commands
remain unchanged; no module under `judge/` imports admission. A verdict's
ledger copy (`Verdict.decision_record`) accepts an optional harness session
and project for admission events, with its historical defaults preserved.

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
- Any change to the ledger's tables. Verdicts are `decision` events. The
  one migration the judge needed, `0005_judge_agent_policy`, registers the
  `judge` agent that records them (see "As built").

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
            -> candidates (provenance_v1, triggers_v3, seeded sample) -> projection_v3
            -> draft (drafter_v3)
            -> decide (rules | readout | jev | llm | llamacpp) -> verdicts + decision events
            -> label (blinded)  -> report (calibrate, freeze, held-out)
            -> live (new sessions, frozen thresholds)
```

### Who is speaking (`provenance_v1`)

A Droid transcript's `user` role has three authors, and the pilot run
(exp1, in [archive/judge-experiments.md](archive/judge-experiments.md))
showed the triggers cannot tell them apart on their own. A
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

### Candidates (`triggers_v3`)

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

Each turn also records `closed_by`: `outcome`, `next_message`, or
`session_end`. T0 controls must be `completed`; triggered strata keep turns
closed any way. `triggers_v2`, which sampled exp2, did not record
`closed_by`; the change is described under "Run `exp3`: changes from
`exp2`" in
[archive/judge-experiments.md](archive/judge-experiments.md#run-exp3-changes-from-exp2).

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

### State projection (`projection_v3`)

Deterministic, versioned, hashed. One header line (project, who started the
session, how the turn closed, outcome, tool and error counts), then
numbered evidence lines: the
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

The drafter is given the `for_drafter` rendering, the same text without the
next-message line, so a lesson cannot fuse the request with an unrelated
follow-up. The sidecar is shared, so cited line numbers mean the same in
both renderings. `projection_v2`, which exp2 used, had no `closed_by` in
its header and no `for_drafter` rendering.

### Drafter (`drafter_v3`)

A drafter writes a structured lesson (a rule, a kind, and one to four cited
claims) or declines, for every item including clean controls. It is
specified under "exp4 design: structured drafts and per-claim Stage B"
below. The drafter kinds are `ollama:<model>` and `llamacpp:<model>`, which
stay local, and `openrouter:<model>`, which sends projections off the
machine and needs `AGENT_MEMORY_JUDGE_ALLOW_REMOTE=1`. `drafter_v1` is in
[archive/judge-earlier-versions.md](archive/judge-earlier-versions.md#drafter-drafter_v1),
and `drafter_v2` under "Run `exp3`: changes from `exp2`" in
[archive/judge-experiments.md](archive/judge-experiments.md#run-exp3-changes-from-exp2).

### Questions (`questions_v4`)

The turn questions (Stage A in reports and stored rows) are asked once per
turn, on the projection alone:

| id | type | asks |
|---|---|---|
| `failed` | noul | the agent did not accomplish the request, judged by tool results, outcome, and the next user message |
| `incomplete` | noul | the turn ended before the agent finished (cut off by the next message or the session end), so success or failure cannot be read from it |
| `user_corrected` | noul | the next user message corrects or redirects the work |
| `env_confirmed` | noul | a tool result independently confirms the claimed completion |
| `category` | choice | wrong_assumption, tool_misuse, stale_or_missing_knowledge, environment_or_provider, user_redirect, none |
| `recurrence` | noul | the cause is project-specific and likely to recur |

The draft questions (Stage B) are asked once per draft, on the projection
plus the draft:

| id | type | asks |
|---|---|---|
| `grounded.i`, `faithful.i` | noul | per claim: the lines claim i cites support it, and it reads them correctly. The draft-level `grounded` and `faithful` are the minimum over the claims |
| `rule_supported` | noul | the supported claims and cited evidence justify this exact rule, including its scope and conditions |
| `generalizable` | score 0..2 | one-off, project-reusable, general practice |
| `actionable` | noul | the draft says what to do differently |
| `enforced` | choice | none, harness, project_docs, tooling |

Derived action: `write_lesson` when `grounded`, `faithful`, `generalizable`
(levels 1 and 2), `enforced = none`, and `rule_supported` all clear their
thresholds; `review` when any of them lies inside the uncertainty band
(default 0.15) around its threshold or was left unanswered; else `discard`,
and always `discard` for a declined draft. `failed` and `recurrence`
describe the turn and do not gate a write. Defaults are 0.5 everywhere
until `--freeze` replaces them per backend. The details are under "exp4
design", "Exact-rule admission", and "Unanswered gates" below.

The code still applies `questions_v1` to `questions_v3` when it reports the
runs built with them. `questions_v1` is in
[archive/judge-earlier-versions.md](archive/judge-earlier-versions.md#questions-questions_v1),
and `questions_v2` under "Run `exp3`: changes from `exp2`" in
[archive/judge-experiments.md](archive/judge-experiments.md#run-exp3-changes-from-exp2).
`questions_v3` is `questions_v4` without `rule_supported`.

### Backends

One interface, five implementations:

```python
class DecisionBackend(Protocol):
    # "rules", "readout:gemma4:26b", "llm:gemma4:26b", "llamacpp:gemma-4-26b",
    # "jev:typesafe/jev-1.14"
    name: str
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
  as confidence. Served by Ollama's native `/api/chat` at `OLLAMA_HOST`
  (default `http://localhost:11434`). The judge makes one model call of at
  most three attempts, and what happened chooses the next one. A reply
  cut off by the 4,096-token budget is asked for once more, still
  thinking, with three times the budget (12,288 tokens; recorded as
  `llm_json` like any thinking answer). If that is cut off too, or a
  reply holds no JSON object, the same question is asked a last time with
  thinking off and the grammar dropped, recorded as `llm_json_nothink`. A
  model that cannot think makes Ollama refuse the thinking switch (HTTP
  400, "does not support thinking"); it gets that last attempt with the
  switch left out. Any other fault fails the item at once, because
  another attempt would not change it: a server that cannot be reached, a
  timeout, any other HTTP rejection. Every attempt's latency is billed to
  the answer, a refused request included.
- `llamacpp:<model>`: the same reasoning judge asking llama.cpp's
  `llama-server` over its OpenAI-compatible `/v1/chat/completions` at
  `LLAMACPP_HOST` (default `http://localhost:9931`, llama-server's own
  default port). Identical prompt, rubric, JSON parsing, and attempts;
  the request uses `max_tokens`,
  `response_format: {"type": "json_object"}`, `reasoning_format: deepseek`
  (thoughts go to `reasoning_content`, not `content`), and
  `chat_template_kwargs: {"enable_thinking": ...}`. The attempt without
  thinking adds `reasoning_effort: none` and leaves `response_format`
  out. Templates without a switch (gpt-oss renders `reasoning_effort`
  verbatim as `Reasoning: <value>`) ignore both, so for them only the
  wider budget can rescue a cut-off item.
  `<model>` is an id from the server's `/v1/models`: in router mode (the
  `llama` app's default) it selects and, on first use, loads that model,
  e.g. `llamacpp:ggml-org/Qwen3.8-27B-GGUF:Q8_0`; in single-model mode the
  server answers with whatever it loaded. An unknown id fails with the list
  of ids the server knows. The response's `model` is recorded as the
  snapshot. A separate backend name, so one run can hold both servers'
  verdicts and the report compares them; only an `llm:` backend is ever
  the reference judge.
- `jev`: the OpenRouter Decisions API (`POST /api/alpha/decisions`, model
  pinned to `typesafe/jev-1.14`), standard-library HTTP, key from
  `OPENROUTER_API_KEY`. It sends redacted state off the machine, so it runs
  only with `AGENT_MEMORY_JUDGE_ALLOW_REMOTE=1`. State is never logged.

Stage A is asked on the projection alone; Stage B on the projection with the
draft appended, so the draft cannot colour the failure verdict.

Every verdict is written to `verdicts/<backend>.jsonl` and appended to the
experiment ledger as a `decision` event in a session of agent `judge`
(`harness_session_id = judge-<run>`, project `judge-experiment`, scope
`judge`). The payload carries run id, item id, backend, model snapshot,
question and projection versions, projection hash, trigger flags, each
answer's value, distribution, confidence and method, latency, cost, the
derived action and its reasons, and the thresholds in force. The event id
is derived from the backend, item, question version, projection hash, draft
hash, and drafter, so re-running a backend on unchanged inputs is a no-op in
the ledger, while a verdict made before a draft existed is superseded in the
verdict file and appended to the ledger as a second event once the item has
a draft. No state text enters the ledger.

## Labeling (`mem judge label`)

Blinded human review in the terminal. The reviewer sees the projection and
the drafts, never a verdict or a drafter's name, in a seeded random order.
The interface is described under "Charmbracelet labeling interface" below;
`--plain` gives numbered prompts without it. New runs use `labels_v4`: the
per-claim labels are under "exp4 design" and `rule_supported` and
`store_exact` under "Exact-rule admission" below. The reviewer's guide is
`GUIDE` in `judge/label_questions.py`.

The first labeling screens, rendered with Rich, and their rubric are in
[archive/judge-earlier-versions.md](archive/judge-earlier-versions.md#labeling-mem-judge-label).
The `labels_v2` set that the later sections build on is under "Run `exp3`:
changes from `exp2`" in
[archive/judge-experiments.md](archive/judge-experiments.md#run-exp3-changes-from-exp2).

## Metrics and procedure (`mem judge report`)

Verdicts are joined to labels by item and to the current projection by
hash. On the calibration split, per backend: accuracy, AUROC, Brier, and
ECE (10 bins) for `failed` and `supported`; the AUROC of the backend's
confidence as a detector of its own errors; `category` and `generalizable`
accuracy; paired discrimination (share of same-project (failed, clean)
label pairs where the failed turn received the higher `failed`
probability); mean latency and total cost per item; and the write decision
against the label. For `questions_v4` runs a write is correct when the
reviewer answered yes to `store_exact` for that draft (see "Exact-rule
admission" below); for earlier question sets, when the item is
`worth_remembering` and `enforced_by` is `none` (or absent). The report
gives precision of
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
writes them to `run.json`, and refuses to run twice. It also refuses while
calibration items are not fully labeled, unless `--partial` is given (see
"A freeze asks for the calibration labels" under As built). The held-out
split is scored only after a freeze; before that the report says so and
stops.

### Live week

`mem judge live --run <id> --since DATE` imports sessions modified since the
date, segments every turn of sessions started since then that is not already
an item, projects, optionally drafts, and decides with the frozen thresholds
into `judge/<run>/live/<day>/`. At the end of the week, label every
`write_lesson` and a random 30 `discard`s (`mem judge label --live <day>`)
and report precision and miss rate.

## Success criteria

Criteria 1 to 4 were stated before the first run, and 5 and 6 were added
for exp3. They are in
[archive/judge-experiments.md](archive/judge-experiments.md#success-criteria-stated-before-running),
with each run's result against them. `mem judge report` still prints a
"Success criteria" section computed from them.

## CLI

All under `mem judge`, all refusing the default home:

| command | does |
|---|---|
| `import --sessions-dir PATH [--limit N] [--project SLUG] [--since DATE]` | write transcripts to the experiment ledger; unchanged files are skipped |
| `candidates --run ID [--n 200] [--seed 1] [--clean-share 0.25] [--human-share 0.5]` | segment, trigger, sample, project; creates the run |
| `draft --run ID [--drafter SPEC ...] [--limit N]` | draft lessons, one arm per `--drafter` (default `ollama:gemma4:26b`); resumable per drafter |
| `decide --run ID --backend SPEC [--backend SPEC ...] [--limit N] [--no-events]` | run backends; resumable per backend |
| `label --run ID [--limit N] [--seed 1] [--item ID ...] [--review] [--plain]` | blinded review; `--item` relabels in full, `--review` steps through labeled items, `--plain` uses numbered prompts without the Go helper |
| `report --run ID [--freeze [--partial]] [--raw]` | score, optionally freeze, write `report.md`; `--partial` freezes although calibration items are not fully labeled |
| `live --run ID --since DATE [--sessions-dir PATH] [--backend SPEC ...] [--drafter SPEC ...] [--day D]` | one live-week step |
| `status --run ID` | items per stratum, drafts, verdicts, labels, frozen or not |
| `purge --run ID` | delete projection text; verdicts and labels stay |

`--live DAY` on `draft`, `decide`, `label`, `report`, and `status` targets a
live-week sub-run.

`draft`, `decide`, and `live` skip an item that fails and exit 1 when
they skipped any; see "A failed item is skipped" under As built.

## Layout

`src/agent_memory/judge/`: `store.py` (the run's files, the rules that
keep them whole, home guard),
`provenance.py` (session callers, message provenance), `triggers.py`
(segmentation, triggers, sampling), `projection.py`,
`questions.py`, `lesson.py` (the structured lesson and its sampler schema),
`drafter.py`, `backends/{base,rules,readout,llm,jev}.py`,
`verdict.py` (asking the questions, the write rule, reading stored
verdicts), `pipeline.py` (import, candidates, draft, decide, live, purge),
`labeling.py` (label definitions), `label_controller.py` (the labeling
workflow), `label_questions.py` (display wording and the guide),
`label_frontend.py` (the transport to the Go helper, and the plain
interface), `label_build.py` (building and caching the Go helper),
`label_ui/` (the Go helper),
`metrics.py`, `report.py`, `cli.py` (`judge_app`, registered
on `mem` next to `spool`). The drafter and the reasoning judge make
their model calls through `src/agent_memory/model_call.py`, which sits
outside this package because `mem graph resolve` uses it too. Run
directory:
`<home>/judge/<run>/{run.json, items.jsonl, projections/,
drafts.jsonl, verdicts/<backend>.jsonl, labels.jsonl, report.md, live/<day>/}`,
directories 0700 and files 0600. A file that commands write can have an
empty `<file>.lock` beside it.
Import bookkeeping in `<home>/judge/imports.jsonl`. New dependency: `rich`
(already installed by typer; declared because it is imported directly).
HTTP is standard library.

## Tests

`tests/test_judge.py`, `test_judge_pipeline.py`, `test_judge_verdict.py`,
`test_judge_unanswered.py`, `test_judge_core_regressions.py`,
`test_judge_store.py`, and `test_label_ui.py`; `test_model_call.py` for
the model calls the judge and the drafter make; the Go helper's own tests
sit beside its sources in `judge/label_ui/`. The test plan written for the
first build is in
[archive/judge-experiments.md](archive/judge-experiments.md#tests).

## Security

Hosted calls are opt-in, over projections that were redacted at import and
again at projection time; the server token is a redaction rule. Projections
are 0600 under the experiment home and `purge` removes them. Verdict files
and ledger events exclude state. The experiment home is never the production
home.

## As built

Small-wins hardening (2026-11-09):

- Migration `0005_judge_agent_policy` registers agent `judge` for verdict
  events, rather than attributing them to `human`. Decide and live commands
  check for that row before doing work and request `mem --home ... init`
  when it is missing. Experiment commands still never admit lessons.
- JSONL rewrites, including human labels and superseded verdicts, write a
  sibling `.tmp` file before `os.replace`. A torn, unterminated final line
  is ignored on read; other corrupt lines report the file and line number,
  never their contents. Appends retain their existing behavior.
- `agent_memory.endpoints` owns model hosts and HTTP egress. Literal
  loopback hosts need no opt-in; every other host, including a remote
  `OLLAMA_HOST`, requires `AGENT_MEMORY_JUDGE_ALLOW_REMOTE=1`. The judge
  HTTP module remains a compatibility seam for callers and tests. Graph
  resolution no longer imports the judge package.

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
  pilot run (exp1, in [archive/judge-experiments.md](archive/judge-experiments.md))
  showed who the "user" is in this corpus. The adapter also
  records `calling_session_id` and `calling_tool_use_id` in session metadata.
- A fifth backend, `llamacpp:<model>` (see Backends), was added 2026-11-06
  so a second reasoning judge could run from llama-server on a different
  model; it shares the `llm` prompt, parser, and retry. The same day the
  `llm` retry was changed to send `think: false` explicitly (see the exp2
  notes in [archive/judge-experiments.md](archive/judge-experiments.md)).
- The exp3 changes (2026-11-07) are listed under "exp3 changes, built
  2026-11-07" below.

### exp3 changes, built 2026-11-07

The exp3 design ("Run `exp3`: changes from `exp2`" in
[archive/judge-experiments.md](archive/judge-experiments.md#run-exp3-changes-from-exp2))
is implemented as written, with these details and deviations:

- Versions: `triggers_v3`, `projection_v3`, `drafter_v2`, `questions_v2`,
  `labels_v2`. A run records its `questions_version`; `draft`, `decide`,
  and `label` refuse a run built under an older one, so exp1 and exp2 are
  read-only. `report` still renders them: `supported` is scored as
  `grounded` under the frozen `supported` cut, the v1 write rule applies,
  and the Stage B `faithful` and `enforced` columns read `n/a`. exp2's
  report re-rendered under the new code reproduces every frozen number
  (checked 2026-11-07; the `report.md` written at freeze time is kept as
  is, the re-render sits next to it).
- `closed_by` is derived for items that predate it (from `closed_by`, then
  `outcome_reason`, then `next_user_event_id`), so the closed-by slices
  and label notes appear for exp2 as well.
- Verdict rows are per (item, drafter) and carry `drafter`, `draft_hash`,
  `declined`, and `stage_a_reused`. Stage A answers copied from an earlier
  row of the same item have their latency and cost zeroed, so per-draft
  latency is Stage B alone. A legacy draft row with no `drafter` is keyed
  `(item, None)` and matches its own verdict on resume.
- A declined draft is stored as `lesson: null, declined: true` with no
  draft hash, gets no Stage B, and is always `discard`.
- `rules` answers the new questions from facts alone: `incomplete` from
  `closed_by`; `grounded` as `supported` was; `faithful` from the same
  overlap at lower confidence; `enforced` is `harness` when the session
  was started by an agent, else `none`, at 0.6. That 0.6 sits inside the
  default band, so the rules floor sends every surviving draft to
  `review` rather than writing on its own.
- Drafters: `ollama:<model>` (think off, temperature 0.2, seed 7, 400
  tokens), `llamacpp:<model>` through the router's OpenAI-compatible
  endpoint with thinking disabled, `openrouter:<model>` with
  `usage.include` so the gateway's cost lands on each draft. The
  `openrouter` kind needs `OPENROUTER_API_KEY` and
  `AGENT_MEMORY_JUDGE_ALLOW_REMOTE=1`. `run.json` keeps one entry per
  drafter (version, model snapshot, drafts, declined, drafted at). A
  draft cut off at the 400-token budget is asked once more with 1,200
  tokens (same seed; on OpenAI-compatible endpoints reasoning tokens
  share `max_tokens` with the answer), with both attempts' latency and
  cost recorded; the first exp3 openrouter pass hit this on one item.
- Judge token budget (2026-11-07): the exp3 `llamacpp:…gpt-oss-20b` pass
  failed Stage A on one item (`5dedfe18`), which lost both drafter rows:
  `judge ran out of tokens after 19824 chars of reasoning; JSON incomplete:
  ''`. Reproduced directly: at temperature 0 the model's analysis fell into
  a repetition loop ("The next message is the same request again?"),
  recovered on its own, and needed 5,988 tokens against the 4,096 budget.
  The no-thinking retry sent the identical 19,824 chars: the gpt-oss
  template has no `enable_thinking` switch and pastes
  `reasoning_effort` into the system prompt, so `none` is noise and the
  retry was the same seeded request at the same budget. `reasoning_effort:
  low` did fit (175 tokens) but changed the verdict (`failed: no` versus
  `yes` with room), so lowering effort is not a budget fix. Both reasoning
  judges now follow the drafter: cut off while thinking, once more with
  12,288 tokens, and only then the no-thinking pass; the no-thinking pass
  keeps the 4,096 budget it had. Latency is summed over attempts (earlier
  `llm_json_nothink` rows record only their last call). The rerun judged
  the item with reasoning kept (`llm_json`, `failed: yes` at 0.97).
- Labeling: items already labeled but missing answers for a newly drafted
  arm come first and are asked only those steps. `--item ID` relabels in
  full with the current labels shown; `--review` shows them and asks keep
  or redo. `history` records only values that moved; a key that was
  unanswered (`null`) and is now filled in is not a change. Each run of
  questions clears the viewport and draws only what it is about (the turn
  with its projection, one draft with the evidence it cites, the drafts side
  by side for `preferred`), so nothing has to be scrolled back to; `p` at
  any question pages the full projection and `?` the rubric. A typed
  choice wins over a control key, so `b` at `preferred` picks draft B. The
  rubric is a term/description grid that wraps to the terminal; it is shown
  once when the session opens and waits for Enter, because the first item
  clears the screen. Each screen lists the keys once and prompts name their
  choices once.
- Turn screen (2026-11-07): labeling a 12-tool-call item on a 215-column
  terminal took about 85 lines (rubric, header, the projection as stored),
  so the question was never on screen with the evidence. The header's facts
  now sit on one line (project, who started the session, closed by,
  outcome, tool calls with the error count) and the projection is
  condensed: the header line and blank lines are dropped, a tool call and
  its result share one row labeled `[n]→[m]` with the arguments clipped in
  the middle to 72 characters (the key at the front and the file name at
  the back say more than the directory between them) and an `ok` result
  clipped to 100, an error result kept whole and shown in red, and every
  other entry opens with its kind in bold. Unknown entry shapes and the
  omission markers are printed as they are. The stored text, which `p`
  pages, stays the evidence of record and the thing a draft's citations
  point at. On the recorded item the turn screen went from 60 lines to 42
  at 215 columns and from 107 to 88 at 100 columns.
- CLI output: progress lines stay on stdout (dim); problems (a skipped
  draft or verdict, a spooled event batch, import failures grouped at the
  end) go to stderr so they survive a redirect. `draft` counts failures in
  its headline. `report` prints `written to <path>` on stderr, so
  `report --raw > file` holds only the markdown, and warns when the
  terminal is narrower than the tables (about 150 columns) need.
- Report: Stage A and Stage B are separate tables (per item and per draft);
  the write table lists each backend's gates and thresholds; criterion 1
  reports the majority-class rate; criterion 6 is evaluated per reference
  drafter on held-out for `questions_v2` runs only; the `incomplete` slice
  appears only when the label exists.

### Charmbracelet labeling interface, built 2026-11-07

This replaces the Rich labeling screens (now in
[archive/judge-earlier-versions.md](archive/judge-earlier-versions.md#labeling-mem-judge-label)),
not the label
definitions or the model's questions. `labels_v2`, `questions_v2`,
citations, seeded item/draft order, review, backfill, edit history, and
historical reports are unchanged.

- Python's `LabelController` owns ordering, branching, validation, and
  persistence. Its screen payload explicitly selects display fields.
  Drafter identities, slot mappings, model verdicts and probabilities
  never cross into the frontend. Subject headings identify the turn,
  one blinded draft, or the comparison. Draft screens include lessons
  and their cited evidence.
- One Go process serves a labeling session. Bubble Tea owns lifecycle,
  keyboard routing and modes; Huh supplies a single embedded selection
  or note field bound to frontend-owned values; Bubbles supplies
  viewports and keyboard help; Lip Gloss supplies measured layouts and
  a shared terminal-aware palette. Dependencies are pinned released v2
  packages in the packaged `label_ui/go.mod` and `go.sum`.
- The question and answer list stay visible while Page Up/Down scrolls
  evidence. `p` opens the full stored projection and `?` the complete
  guide, both in-app. Esc or q returns without changing the highlight
  or evidence offset. Back is always navigation, including at comparison;
  choose draft B with arrows and Enter. Changing an answer removes
  dependent answers that no longer apply. Notes have Add/edit and
  Continue actions; letters are literal text, Enter accepts and Esc
  cancels. Summary and review actions use the same selection controls.
- Version 1 NDJSON travels over subprocess pipes with increasing request
  IDs, an 8 MiB frame limit, and one action in flight. Go uses the
  controlling terminal separately. Both boundaries reject invalid
  messages without echoing their contents. Display text makes terminal
  controls visible and preserves literal brackets. Saves publish a
  private, fsynced label snapshot by atomic replacement. Only a successful
  Python write allows a Saved notice; a failed write stays at the summary
  for retry. An interrupted frontend does not undo saved items.
- Brief opening instructions replace the long opening rubric. The full
  guide preserves unclear versus incomplete, grounded versus faithful,
  and human versus parent-agent instructions. Display names such as
  Project documentation replace internal identifiers; stored values and
  model prompts do not change.
- `--plain` uses numbered linear prompts and explicit navigation commands,
  without cursor controls or Go. Pipes and dumb terminals choose it
  automatically. It shares the controller, not a second set of branches.
  Its note input uses `/cancel`, since it reads lines rather than raw keys.
  A note is one line without tabs: either interface reads a typed tab as a
  space, and a note saved earlier with a tab is shown with a space while
  its stored text stays as it was.
- Wheels and source distributions include the small Go module. First
  interactive use with pending work builds it under a private platform
  cache, keyed by source/dependency bytes, protocol, platform and Go
  toolchain. Builds hold a file lock, use `-mod=readonly`, and atomically
  publish the executable. Cache hits do not rebuild. Missing Go and
  build failures explicitly suggest `--plain`; they never silently
  switch interfaces. See README for cache locations and commands.
- Width changes restore terminal tab stops before repainting, using the
  workaround documented in Bubble Tea issue
  [#1838](https://github.com/charmbracelet/bubbletea/issues/1838), which
  remains open for the pinned renderer. This protects terminals that
  discard tab stops when a window narrows.

### exp4 design: structured drafts and per-claim Stage B

Decided and implemented 2026-11-08. New runs use `drafter_v3`,
`questions_v3`, and `labels_v3`. Stage A, candidate sampling, projections,
thresholds, freeze, and the ledger schema do not change.

The drafter returns a structured lesson, or `{"lesson": null}` to decline:

```json
{
  "lesson": {
    "rule": "Stop mem serve before running migrations on the same home.",
    "kind": "project_fact",
    "claims": [
      {"text": "The migration failed with database is locked.", "cited": [7]},
      {"text": "mem serve was still running on that home.", "cited": [2, 4]}
    ]
  }
}
```

`rule` is one sentence saying what to do differently or what fact to
remember. `kind` uses the same five options as the item's `lesson_kind`
label. A lesson has 1 to 4 claims, each one sentence about this turn
with its own cited entry numbers. Uncited claims are allowed; neither
the judge nor the labeler skips their questions or supplies automatic
answers. The drafter still sees only `for_drafter`, with the same
`closed_by` guidance and no next message.

The `drafter_v3` prompt leads with the decline. Its first wording gave
the lesson template first and appended `or {"lesson": null}`; on 16
items from this repository's own sessions (10 clean, 6 with a tool
error) both `ollama:gemma4:26b` and `openrouter:openai/gpt-6-luna`
then declined 0/16, where `drafter_v2` had gemma declining 6/16 and
gpt-6-luna 0/16 (1/200 in exp3). The schema was not the cause: the
same prompt with no schema also declined 0/16 on both models, and
under every wording tried the schema and no-schema decline counts
matched on both models, so OpenRouter's strict mode shows no sign of
putting the schema in the model's context. The shipped wording says
first that most turns hold nothing worth keeping (a routine request,
no project-specific fact, advice any competent agent follows), names
the exceptions (a project-specific fact, a tool quirk, a correction a
future session would otherwise repeat), adds one sentence that a
refusal the harness itself issued, such as a tool call cancelled until
a plan is approved, is not a lesson about a tool, and only then
describes the lesson shape; the user message opens with
`Return {"lesson": null} unless ...`. On the same 16 items this
declines 8/16 with gemma and 4/16 with gpt-6-luna; both keep the arXiv
export API, the transitive Pydantic dependency, and the Ollama
`done_reason` facts, and gpt-6-luna drops the one-off "prioritize X"
decisions. The harness sentence is short on purpose: a paragraph
spelling out harness behavior took gemma to 14/16 declines, project
facts included, while gemma still wrote up the Spec Mode refusal under
every wording tried; gpt-6-luna responds to the sentence and draws a
different, grounded fact from that turn instead. Stage B's `enforced`
gate remains the backstop for the local model. No real run holds
first-wording drafts, so the version is unchanged.

`judge/lesson.py` owns frozen, extra-forbidding pydantic models
`Claim`, `Lesson`, and `DraftReply`. Sentence validation collapses
whitespace, caps text at 300 characters, and rejects empty text.
Citation validation accepts bracketed numbers, drops junk and
non-positive numbers, and removes duplicates. Validation errors report
locations and messages without including reply text.

The sampler schema comes from `DraftReply.model_json_schema()`, with
titles and descriptions removed. It does not duplicate the model in
a handwritten schema. A regression test pins its grammar-safe keyword
set. Ollama receives this schema as `format`; llama-server receives
`response_format: json_schema` with the name `lesson`. OpenRouter
receives the same schema with `strict: true`. An OpenRouter HTTP 400
or 422, the codes a provider's rejection of the schema arrives as, gets
one retry with `json_object`; that draft records
`schema_enforced: false`, but pydantic still validates the reply.
Other HTTP failures, including auth, credit, and rate-limit codes, do
not trigger this fallback. The initial token
budget is 700; a cut-off reply gets 2,100 tokens, with both completions'
latency and cost counted.

Draft rows keep the existing fields and add `rule`, `kind`, `claims`,
`schema_enforced`, and `drafter_version`. The canonical text is:

```text
Rule: Stop mem serve before running migrations on the same home.
Kind: project_fact
Claims:
1. The migration failed with database is locked. [7]
2. mem serve was still running on that home. [2, 4]
```

An uncited claim ends in `[no citation]`. This deterministic rendering
is the lesson shown to the judge and labeler, and the text hashed by
the unchanged `draft_hash`. The draft's `cited` field is the union of
its claims' citations.

Stage B asks `grounded.1`, `faithful.1`, and so on for each claim,
then `generalizable`, `actionable`, and `enforced` for the whole draft.
`grounded.i` asks whether the cited lines support claim i;
`faithful.i` asks whether it reads them correctly. Verdicts keep these
answers and derive `grounded` and `faithful` by taking the minimum
per-claim p(yes), tagged `method: min_over_claims` with zero added
latency and cost. Verdict rows record the claim count. Decision events
copy the per-claim answers without copying lesson text. The write rule
and all draft-level metrics, including criterion 5, still use the
derived draft-level values. Legacy rows without claims use the old
Stage B template.

Label steps are `A.c1.grounded`, `A.c1.faithful`, and so on, followed
by `A.generalizable` and `A.enforced_by`. A draft label contains a
`claims` list of `{grounded, faithful}` entries as well as its derived
draft-level values: all true means true, any false means false,
otherwise null. Backfill restores claim answers from this list.
History uses paths such as `drafts.<drafter>.claims.1.grounded`,
numbered from 1 like the steps and the judge's per-claim answers.

The screen subject remains `Draft A`. Its card says `Claim 2 of 3`,
marks that claim with `▶`, and shows only that claim's cited evidence,
including missing-citation notes. Draft-level and comparison screens
show the citation union and no focus. Plain mode prints the focus
line too. Label-UI protocol 2 requires a `focus` string on every
displayed draft; the protocol bump invalidates the cached Go helper.

A Claims report section appears only for runs with structured draft
rows. Per drafter it reports mean claims/draft (declines count as zero),
uncited-claim share, absent-citation share against stored projections,
labeled-claim grounded and faithful rates, and kind agreement with the
item's `lesson_kind` label. Per backend it reports claim-level
accuracy, precision, recall, and F1. Held-out backend metrics remain
locked until freeze. Unavailable projections are reported as unchecked,
not counted as absent evidence.

`draft` refuses any run containing another drafter version, including
adding a v3 arm beside v2 drafts. `draft`, `decide`, and `label` refuse
older question sets. This makes exp3 (3 of 200 items labeled at the
change) read-only along with exp1 and exp2. Finishing exp3 labeling
requires the older code; this change neither redrafts nor relabels it.
Reports retain the older question definitions, write rule, and labels.

### Audit corrections (2026-11-08)

Both labeling frontends now support the controller's four blinded slots,
A through D. Go previously rejected C/D despite Python producing them.
Comparison choices say "All equally" for three or four drafts and retain
the existing stored value. More than four drafts produces an error before
the frontend opens; no drafts are silently omitted. This is a labeling
limit, not a limit on drafting or reporting.

Protocol 2 remains strict about valid, unique slots and matching draft
subjects. Source changes invalidate the cached helper without another
protocol bump. Tests cover plain-mode persistence, navigation, claim focus,
narrow and wide layouts, and synthetic Python-to-Go exchanges over a PTY.

Judge accounting and reporting corrections:

- Stage A latency and cost belong only to the first successfully saved
  verdict for an item. Later drafts in the same decide call use the same
  zero-cost copies as resumed calls. Existing verdict files are not rewritten.
- A missing or null frozen cascade gate means no cascade result is
  available. Reporting no longer fits that gate on held-out labels or
  includes its unsupported criterion 3 conclusion.
- New threshold fitting includes all-positive and all-negative choices,
  including probability endpoints. Tie-breaking is unchanged; existing
  frozen thresholds and gates are read as stored.
- Drafter ratings and preferences require matching current draft hashes.
  Changing a draft invalidates its old draft-level ratings, not the item's
  still-valid turn labels or recorded production costs.
- Next-user linking uses a reverse pass within each session. Resume
  selection builds the pending item-ID set once. Paired-discrimination
  reporting streams comparisons instead of retaining every pair; comparison
  time remains quadratic.

Regression tests reproduce the original failures with synthetic inputs.
Synthetic frozen reports for questions v1, v2, and v3 match the original
implementation byte for byte. This audit did not reopen private historical
runs or modify their files.

### Exact-rule admission, as built (questions v4 / labels v4)

New runs use `questions_v4` and `labels_v4`. The structured lesson schema,
`drafter_v3`, per-claim grounded/faithful questions, seeded splits and
projection format are unchanged. No database migration is needed for this
judge release. Deploy it separately from ledger changes: **finish any
ongoing v3 labeling on the old application before upgrading**. This version
refuses draft, decide and label (including review/backfill) on v1–v3 runs;
it does not upgrade their drafts, labels or frozen thresholds. Reports
retain those versions' original write ground truth and output definitions.
Synthetic frozen v1–v3 report hashes match the pre-v4 implementation.

Stage B now asks a whole-draft `rule_supported` question after the
per-claim checks:

> Do the supported claims and cited evidence justify this exact rule,
> including its scope and conditions, without an unsupported causal leap
> or overgeneralization?

Grounded, faithful, reusable (`generalizable` levels 1 or 2), unenforced
(`enforced = none`) and rule supported must all clear their thresholds.
Stage A failure remains descriptive, not a write prerequisite. The new
gate defaults to 0.5 with the existing 0.15 uncertainty band. Missing or
unparseable rule answers cannot authorize a write, even after threshold
fitting: a parser's uniform fallback is not evidence. (Since 2026-11-10
this holds for every gate, and such a verdict is a review rather than a
discard; see "Unanswered gates" below.) An explicit 0.5
rule-support distribution goes to review even if a fitted cut is lower.
The rules baseline returns that uncertainty rather than treating token
overlap as proof of a causal rule.

Both labeling interfaces add `A.rule_supported` and `A.store_exact`
(likewise for B–D), with Yes, No and Unclear. The latter asks **“Would you
store this exact draft?”**, without repairing it. Each screen is about
the identified draft, shows its complete lesson and citation union, and
has no claim focus or model verdict. The exact-rule question follows
claim checks; the storage question follows reuse and enforcement.
Back, review, redo, skip, resume and hash invalidation use the shared
controller. A draft changed while labeling is open is rejected at save.
No Go wire-schema change is needed; questions/options are display data
in protocol 2.

The new fields live under `drafts.<drafter>` and are boolean or null.
An explicit Unclear is null plus the field name in that draft's `unknown`
list, so it stays unscored but is not repeatedly asked on resume.
Absent answers are backfilled. Corrections retain the existing history
paths, including `drafts.<drafter>.rule_supported` and `.store_exact`.
Changed draft hashes invalidate both new labels and claim ratings.
Changing, adding or removing an arm also reasks the comparison rather
than silently carrying an old preference onto the new drafts.

For v4 only, write precision and miss rate use the blinded `store_exact`
label, not the turn's `worth_remembering`. The latter remains an
opportunity/discovery measure, including decline diagnostics. A valuable
turn with a bad draft is not a correct write. Missing, unclear, declined
and stale-hash admission labels are unscored: precision includes only
scored writes and miss rate only explicitly approved exact drafts.
Reports expose scored admission rows, scored writes and unscored rows
alongside action counts, and score `rule_supported` separately. Drafter
ratings use independent nonmissing denominators rather than treating a
missing rating as false.

Calibration alone fits the new gate, using the human rule-support label;
held-out verdict metrics stay locked until freeze. Frozen cuts and cascade
gates are reused as stored, never fitted on held-out labels. Cost accounting
includes the new question, preserves zero-cost claim aggregates and reused
Stage A answers, and does not drop costs for unscored admission rows.
V4's per-item table sums paid USD across all draft verdicts; its Stage B
table shows only the draft-question subset, not another total to add.
Decision events include version, hashes, claim count, Stage A reuse, the
new answer, action, thresholds and costs, never the rule, claims or state
text.

Regressions use synthetic runs, mocked HTTP, both plain and Go interfaces,
rendered narrow/wide screens and real Python-to-Go PTY exchanges. They cover
true claims under an unsupported rule, valuable turns with rejected drafts,
missing/unparseable gate answers, unclear label persistence, edit history,
hash invalidation, calibration/held-out separation, costs and historical
report reproduction. No private runs or model experiments are part of this
change.

### Unanswered gates, as built (2026-11-10)

A judge can be asked a question and give no usable answer: its reply omits
the question, names none of the options, gives a score with no digit, or
carries no probability that can be read. Until this change only
`rule_supported` was protected. For every other question the parser put a
uniform or default distribution in the answer's place, and the write rule
read it as evidence. A reply that omitted `generalizable` cleared that gate
at 0.67 and could produce `write_lesson`.

A question the judge did not usably answer is now **unanswered**. Its
stored distribution is a placeholder that nothing reads as a probability.

- The mark is the answer's `method`, which ends in `_missing` (no entry,
  or no distribution), `_unmatched` (an answer that is none of the
  options) or `_no_probability` (no finite numeric probability). The JSON
  judges (`llm`, `llamacpp`) no longer read a score answer without a digit
  as level 0 or a missing probability as 0.5, and only a string or a
  number can name an option.
- Any backend's distribution with no mass on a listed option is marked
  the same way. That covers a Jev entry with no value or with empty
  probabilities (a value that is not a number is a `JudgeError`), readout
  samples that name no option (`sampling:8_missing`) and a question the
  rules baseline has no rule for (`rules_missing`).
- With per-claim questions, one unanswered claim leaves the whole
  `grounded` or `faithful` gate unanswered (`min_over_claims_missing`).
- An unanswered write gate makes the verdict a `review`, with the reason
  `<gate>: unanswered`. Like the uncertainty band, it outranks another
  gate that plainly fails. A missing or unparseable `rule_supported`
  answer was a discard before; it is a review now.
- A declined draft is still a discard. A gate the verdict was never asked
  (an item with no draft, an older question set) still reads
  `<gate>: no answer` and cannot clear, as before.
- An unanswered question that gates nothing (`actionable`, Stage A) does
  not hold a write.
- The model is not asked again. The verdict is recorded once with the
  mark, and a resumed decide does not repeat it.
- Reports leave an unanswered question out of every metric and out of
  threshold fitting. When a backend has verdicts that are a review for
  this reason, one line under the write-decision table gives the count
  per backend. With none, the report is byte for byte what it was.
- Admission keeps the distinction. An example's unanswered gate is stored
  as null, the draft goes to human review whatever its score, and it does
  not help choose the write cut.

A read-only scan of the `method` fields in the existing runs' verdict
files found none of these marks, and the reports of the two labelled runs
(exp2, exp3) rebuild byte for byte from copies. A verdict already stored
with a silent default cannot be told apart from a real answer and is left
as it is. Regressions are in `tests/test_judge_unanswered.py`, with
mocked HTTP and synthetic runs.

### One question set per report, as built (2026-11-11)

A verdict's write gates depend on the question set it was judged with.
`mem judge report` used to read every verdict of a run through the run's
own `questions_version`, whatever the verdict itself recorded. It now
refuses a run that holds a verdict judged with another question set, or
with none recorded, and names the verdict file. Nothing is frozen and no
report is written.

No command produces such a run: draft, decide and label refuse a run built
under an older question set, and a run's version does not change. Every
verdict in the stored runs (exp1 to exp4) carries its run's version, and
the reports of exp2 and exp3 rebuild byte for byte from copies.

### The verdict module, as built (2026-11-11)

Everything that builds or reads a verdict is now in `judge/verdict.py`.
No behaviour changed: new rows keep their keys and order, and stored rows
of every question set read as before.

- `ask_turn` asks the turn questions (Stage A). `judge` asks the draft
  questions (Stage B) and returns the verdict. Given answers the turn
  already has, it does not ask the turn questions again.
- `Verdict` reads one stored row in place and never changes it. It gives
  the probability that a question clears, the write gates of the question
  set the verdict was judged with, the action recorded when it was judged,
  the action at any other thresholds, the turn's answers for the next
  draft of the same turn, and the ledger copy (`decision_record`).
- `derive_action` is the write rule, moved here from `questions.py`.

The decide loop, the report and admission read verdicts through this
module instead of each reading a row's `answers` by hand. The loops stay
where they were: `decide_items` still resumes, replaces stale rows, fails
fast and batches ledger events, and admission's daily run is still
`admission/run.py`.

Checked against the previous code on the same inputs: the rows and ledger
copies of a synthetic decide run (two drafters, declined, claimed and
failing drafts, late drafts on resume) are byte for byte the same apart
from the clock, and so are admission's imported examples. The synthetic
reports for questions v1 to v4, open and frozen, and the reports of exp2
and exp3 rebuilt from copies are unchanged. Tests at the module's
interface are in `tests/test_judge_verdict.py`.

### Turn questions and draft questions, renamed (2026-11-11)

The code now says turn questions and draft questions where this spec says
Stage A and Stage B: `TURN_QUESTIONS`, `DRAFT_QUESTIONS`,
`draft_questions_for`, `evaluate_turns` and `evaluate_drafts`. Nothing
else changed. Report headings and the stored row key `stage_a_reused`
keep the old names, so reports and stored rows read as before, and the
earlier sections of this spec keep them too.

### Appending after a cut-off row, as built (2026-11-11)

A run killed while it appends a draft or a verdict can leave the last row
of the file without its end. Reading has always ignored such a fragment,
so the resumed run does that row again. But the append then joined the
new row onto the fragment, and from then on the file could not be read:
every later command stopped with "invalid JSON" until someone repaired the
file by hand.

An append now looks at the end of the file first. It cuts a fragment off,
and it gives a whole row that only lacks its line ending that ending, so
the new row starts on its own line. After an append, reading returns what
it returned before plus the new rows. Admission's own store is unchanged:
it refuses to append to a file it cannot read. Regressions are in
`tests/test_judge_store.py`, one of them a resumed `decide`.

### Label history counts a move off Unclear, as built (2026-11-11)

A reviewer can answer Unclear to "Did the agent fail?", to rule support
and to "Would you store this exact draft?". The answer is stored as null:
for the turn with `incomplete` false, for a draft with the field named in
its `unknown` list. `history` skipped every key whose old value was null.
That is right for a question not yet answered, but it also skipped a
correction from Unclear to Yes or No, so the report's count of changed
labels missed those corrections.

A move off Unclear is now a change, recorded with the null it was stored
as. In `history`, a null old value therefore always means Unclear. The
`unknown` list is no longer recorded as a value of its own, so one changed
answer is one changed value. Stored labels are untouched, and the reports
of earlier runs count their stored history as before.

### The run store, as built (2026-11-11)

Every command reads and writes a run's files through `RunStore`
(`judge/store.py`). The rules that keep the files whole are stated at the
top of that module and applied nowhere else:

- A file that is replaced (`run.json`, `items.jsonl`, `labels.jsonl`, a
  projection, `report.md`) is written beside itself, flushed to disk, and
  moved into place. A reader or a crash sees the old file or the new one,
  never part of one.
- A file that grows (`drafts.jsonl`, a verdict file, `imports.jsonl`) gets
  one row at a time, as "Appending after a cut-off row" describes.
- One command at a time adds rows to a file. A second `draft` on a run, a
  `decide` that names a backend another `decide` is judging, a second
  `candidates` for a run, and a second `import` into a home are refused at
  once, with a message that names the file. Before, the two commands cut
  into each other's rows and paid for the same model calls twice. `decide`
  takes every verdict file it will write before it calls a model, and runs
  a backend that is named twice once. A `decide` on other backends and a
  labeling session still run beside it.
- `run.json` and `labels.jsonl` are changed by commands that run side by
  side, so each change reads the file, merges, and writes it back under a
  lock. `draft` and `report --freeze` each keep the fields the other set.

Two labeling sessions on one run used to erase each other's saves, because
each wrote the labels it had read when it opened plus its own. A save now
adds its one label to the labels on disk at that moment. If another
session has saved the same item since this one read it, the save is
refused: the first label is kept, and the session ends with the reason.
An item another session finished is left out of this session when its
turn comes, with a notice.

A run's directories are 0700 and its files 0600, `report.md` and the
verdict files included. A file or directory made before this change gets
that mode the next time a command writes it. The locks are empty
`<file>.lock` files beside the files they guard, and can be deleted when
no judge command is running.

Admission keeps its files through the same code and adds its own rules;
see [admission-spec.md](admission-spec.md). Regressions are in
`tests/test_judge_store.py`, `test_judge_pipeline.py`, `test_label_ui.py`,
and `test_admission_fit.py`.

### Model calls, as built (2026-11-11)

The reasoning judge, the drafter, and the entity resolver behind
`mem graph resolve` each ask a model for a JSON object. Each had its own
copy of what a model server is sent, how a model reply is read, when to
ask again, and how the object is found in the text, and the copies had
drifted. One module, `src/agent_memory/model_call.py`, now makes the call
for all three. A caller hands it a model server (Ollama, llama-server, or
OpenRouter) and says what to ask: the prompt, the token budget, the shape
the reply is held to, the wider budget a cut-off reply may get, and
whether a last attempt without thinking is allowed. The module sends the
attempts, reads each model reply, sums latency and cost over every
attempt, and returns the object or one of three failures: the server was
unavailable, the last reply was cut off, or it held no JSON object. It
sits outside `judge/` because loading the graph must not load the judge.

`llm:` and `llamacpp:` are one judge asking two servers, and the three
drafter kinds are one drafter. `judge/backends/http.py` and
`judge/backends/llamacpp.py` are gone. `readout`, `jev`, and admission's
embedder do not ask for a JSON object, so they call `endpoints.py`
themselves and turn its errors into their own.

Backend and drafter names, request bodies, recorded methods, and stored
rows are unchanged. What changed:

- The judge chose its next attempt from words in an error message. Any
  failure whose text held "http" made the last attempt leave the thinking
  switch out instead of saying off. A reply that quoted a link did that,
  and so did any error carrying the server's URL; Ollama then let a
  thinking model think again, the fault first fixed on 2026-11-06. A
  missing model whose name held "think", or a transport reply that was
  not JSON, bought a second request that could not succeed. What happened
  now chooses the attempt, as "Backends" describes, and only Ollama's own
  words in an HTTP 400 count as a refused switch.
- A request refused over the thinking switch was not billed. Its latency
  is now part of the answer's.
- A llama.cpp draft that is cut off says how much thinking came before
  it; the drafter had read OpenRouter's field name, so the count was
  always missing. A local llama.cpp draft records no cost, where it
  copied a `usage.cost` that llama-server does not report.
- A hosted draft whose usage block cannot be read fails that item with an
  error. It used to end the run with a traceback.
- Errors say "thinking" where they said "reasoning" (`judge ran out of
  tokens after N chars of thinking`), and a draft with no JSON is reported
  as the drafter's, not the judge's. Error text quoted in earlier notes
  is left as it was printed.

`mem graph resolve` gained the judge's three attempts and keeps the
merges of the blocks that worked; see "Entity resolution keeps what
worked" in [phase-1-spec.md](phase-1-spec.md). Regressions are in
`tests/test_model_call.py`, `test_judge.py`, and `test_graph.py`.

### A failed item is skipped, as built (2026-11-11)

`draft`, `decide`, and `live` skip an item whose draft or verdict fails
and write nothing for it, so the next command asks it again. Until now
that held only once a drafter or backend had written something in the
run. Before that, the first failure ended the command. When the failure
was the item's own (a turn too long for the model, a reply cut off every
time), every rerun ended at the same item and the run could not be
finished.

A command now stops early only when the fault cannot be the items'. It
stops after three failures in a row when:

- the model server could not be reached each time: nothing accepted the
  connection, or the request was not allowed to leave the machine (the
  egress opt-in is off, or a hosted key is missing); or
- this drafter or backend has written nothing in the run yet, whatever
  the three failures were. A model that cannot produce one usable row in
  three tries is unfit or misconfigured.

A reply of any kind starts the first count again, and a written row
starts both. A timeout, an HTTP rejection, and a reply that cannot be
used all count as the item's failure, because each can be the item's
doing: a long turn times out, and a prompt past the context window is
rejected. Counted against the server, three such items at the head of a
run's unfinished work would stop every rerun before it reached the items
behind them. `endpoints.py` raises `Unreachable` for a server that was
not asked, `model_call.py` passes it on as `ModelUnreachable`, and the
judge's backends and drafter as `ServerUnreachable`.

A command that stops says why and repeats the last error, for example
`llm:gemma4:26b: stopped, the model server could not be reached 3 times
in a row: ...`. Before it stops it records what it wrote: the drafter's
entry in `run.json`, and the decision events of the verdicts on disk.
The drafters or backends named after the one that stopped are not run,
and `live` does not go on to judge after a drafter stopped.

`draft`, `decide`, and `live` exit 1 when any item was skipped. They do
so after doing all the rest, and nothing written is taken back; the exit
code tells whatever runs next that the step is not finished. They exit 0
only when nothing is left to retry. `decide_items` returns the verdicts
written and the failures for each backend (`DecideSummary`) where it
returned the written counts alone.

Not covered: a model server that accepts the connection and never
answers. Each remaining item then waits out its timeout, twice, and is
skipped. Regressions are in `tests/test_judge.py`, `test_endpoints.py`,
and `test_model_call.py`.

### A freeze asks for the calibration labels, as built (2026-11-11)

Thresholds are chosen once. `report --freeze` chose them on whatever
calibration items had labels at that moment, and neither the run nor the
report said how many that was. A freeze started too early could not be
taken back, and nothing showed that it had been early.

`report --freeze` now refuses while a calibration item is not fully
labeled, names how many are short, and leaves `run.json` as it was. An
item is fully labeled when a labeling session would ask nothing more
about it. It is short when:

- it has no label; or
- its label has questions left, which happens when a drafter is added
  after the item was labeled, so nobody has answered for the new draft.

This is the labeler's own test of what is left to ask
(`labeling.questions_left`), so a labeling session opens every such
item and `mem judge label` is how to finish them. Held-out items do not
count: they take no part in choosing thresholds. For a run built with an
older question set only the first case applies. Its labels answer the
questions of their day, and labeling no longer opens such a run.

`--freeze --partial` freezes on the labels there are. The run then
records it under `frozen.partial` (`calibration`, the size of the split,
and `unlabeled`, how many items were short), and every report of that
run says so beside the freeze time. A freeze with nothing short is
stored as before, so earlier runs and their reports are unchanged.
`--partial` without `--freeze` is a usage error.

Tried on private copies of the stored runs: exp3, with 118 of its 120
calibration items unlabeled, was refused where it would have frozen.

Not covered: a calibration item that has a label but no verdict from
some backend. That backend's thresholds are still chosen without it, and
nothing says so. Regressions are in `tests/test_judge.py`.
