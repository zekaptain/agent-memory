# Judge experiment spec: decision models as failure and lesson triage

Status: approved and implemented, 2026-10-26; exp3 changes implemented
2026-11-07 (see "As built" at the end).
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
  (default `http://localhost:11434`). If the thinking pass runs out of
  tokens or returns no JSON, the same question is asked once more with
  thinking off and the grammar dropped, recorded as `llm_json_nothink`.
- `llamacpp:<model>`: the same reasoning baseline served by llama.cpp's
  `llama-server` over its OpenAI-compatible `/v1/chat/completions` at
  `LLAMACPP_HOST` (default `http://localhost:9931`, llama-server's own
  default port). Identical prompt, rubric, JSON parsing, and no-thinking
  retry; the request uses `max_tokens`, `response_format: {"type":
  "json_object"}`, `reasoning_format: deepseek` (thoughts go to
  `reasoning_content`, not `content`), and `chat_template_kwargs:
  {"enable_thinking": ...}` plus `reasoning_effort: none` on the retry.
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
| `draft --run ID [--drafter SPEC ...] [--limit N]` | draft lessons, one arm per `--drafter` (default `ollama:gemma4:26b`); resumable per drafter |
| `decide --run ID --backend SPEC [--backend SPEC ...] [--no-events]` | run backends; resumable per backend |
| `label --run ID [--limit N] [--seed 1] [--item ID ...] [--review]` | blinded review; `--item` relabels in full, `--review` steps through labeled items |
| `report --run ID [--freeze] [--raw]` | score, optionally freeze, write `report.md` |
| `live --run ID --since DATE [--backend SPEC ...] [--drafter SPEC ...] [--day D]` | one live-week step |
| `status --run ID` | items per stratum, drafts, verdicts, labels, frozen or not |
| `purge --run ID` | delete projection text; verdicts and labels stay |

`--live DAY` on `draft`, `decide`, `label`, `report`, and `status` targets a
live-week sub-run.

## Layout

`src/agent_memory/judge/`: `store.py` (run directory, home guard),
`provenance.py` (session callers, message provenance), `triggers.py`
(segmentation, triggers, sampling), `projection.py`,
`questions.py`, `drafter.py`, `backends/{base,http,rules,readout,llm,llamacpp,jev}.py`,
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
rewrites its row; `should_write` and the labels-only report section;
`closed_by` derivation and the `for_drafter` rendering; drafter kinds,
decline, and the remote opt-in; two drafters per item through labeling
(A/B order, per-draft answers, `preferred`, backfill of a new arm,
`--item` with back and redo, `--review`, `history`); decide judging each
draft once with Stage A reused; a `questions_v1` run still reports and is
refused by draft, decide, and label.

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
- A fifth backend, `llamacpp:<model>` (see Backends), was added 2026-11-06
  so a second reasoning judge could run from llama-server on a different
  model; it shares the `llm` prompt, parser, and retry. The same day the
  `llm` retry was changed to send `think: false` explicitly (see the exp2
  notes).
- The exp3 changes (2026-11-07) are listed under "exp3 changes, built"
  below the exp3 design.

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

Results (labels 2026-11-05, passes 2026-10-26 to 2026-11-06, frozen
2026-11-07 01:05 UTC). Reference judge `llm:gemma4:26b`. `report.md` in
the run directory holds every table, including the calibration split cut
by who started the session; the numbers below are the ones the criteria
turn on.

Labels on their own (200 items):

- `failed` 60 (30%; calibration 35/120, held-out 25/80). `supported` 194
  (97%): the Stage B drafts almost always cite real evidence, so this
  question has 6 negatives in the whole run and 2 in held-out, and its
  accuracy and AUROC columns carry little information. `category`: none
  136, tool_misuse 34, wrong_assumption 15, user_redirect 8,
  environment_or_provider 4, stale_or_missing_knowledge 3. `generalizable`:
  0 → 17, 1 → 70, 2 → 113. `worth_remembering` 135.
- `enforced_by`: none 89, harness 98, project_docs 9, tooling 4. Of the 100
  drafts from agent-started sessions, 77 restate what the harness (the
  Mission prompts) already enforces and 22 are enforced by nothing; of the
  100 from human-started sessions, 21 are harness-enforced, 9 live in
  project docs, 3 in tooling, and 67 are enforced by nothing.
- Items that should be written (worth remembering and enforced by nothing):
  81 of 200, 62 of them from human-started sessions. Only 33 of the 81 are
  also labeled `failed`. The action rule writes only when `failed` clears
  its threshold, so 48 of the 81 cannot be reached by any backend under the
  current rule and the miss-rate floor is 0.59. Most lessons worth keeping
  came from turns that did not fail.
- 122 labeled items had a tool error; in 53 (43%) the error was intended
  and used, and 54 (44%) were labeled failed. A bare T3 trigger is right
  about failure 44% of the time.

Passes (200 verdicts each; latency is wall time per item for all eight
questions):

- `rules` and `readout:gemma4:26b` on 2026-10-26. Readout 2.7 s per item
  (max 3.6), 9 min in all; every answer from logprobs, no sampling
  fallback.
- `llm:gemma4:26b`: 37.6 s mean (median 33, p90 62, max 80), 125 min in
  all. 98 of 1,600 answers (6%) are `llm_json_nothink`.
- `llamacpp:ggml-org/gpt-oss-20b-GGUF:MXFP4`, added 2026-11-06 as a second
  reasoning judge on a different server and model (llama-server in router
  mode on port 9931; same prompt, parser, and retry as `llm`): 13.3 s mean
  (median 13, p90 17, max 27), 44 min in all. 8 of 1,600 answers are
  `llm_json_nothink`.
- The `llm` pass had stopped at 191 on 2026-10-26; on resume 7 items failed
  both attempts with `judge ran out of tokens; JSON incomplete: ''`. The
  retry removed the `think` key, but Ollama (0.40) thinks by default for
  any model with the thinking capability, so the retry thought again and
  spent the whole 4,096-token budget in `thinking` with empty `content`.
  Checked directly: with `think` omitted gemma4 still fills `thinking`;
  only `think: false` stops it. Fixed 2026-11-06: the retry (and a backend
  built with `think=False`) sends `think: false` explicitly, the key is
  removed only when the server rejected it, and the error now reports how
  much went to reasoning. On one of the seven, Stage A took 56 s and
  14,562 chars of thinking to `length`, then 2.3 s with thinking off. The
  nine remaining items then completed.

Frozen thresholds (accuracy-maximising cuts on the 120 calibration items;
`recurrence` 0.50 and band 0.15 everywhere; gate is the cascade gate
against `llm`):

| backend | τ failed | τ supported | gate |
|---|---|---|---|
| `llm:gemma4:26b` | 0.85 | 0.50 | — |
| `readout:gemma4:26b` | 1.00 | 0.00 | 0.78 |
| `llamacpp:…gpt-oss-20b-GGUF:MXFP4` | 0.97 | 0.07 | 0.80 |
| `rules` | 0.75 | 0.50 | 0.90 |

Two of the cuts are degenerate. At τ failed 1.00 readout never predicts
failed (0 of 200); at τ supported 0.00 and 0.07 readout and gpt-oss always
predict supported. When a backend's probabilities rank the items poorly,
the accuracy-maximising cut on a 70/30 label is the majority class, and
that is what froze. Confidences are as concentrated as exp1 predicted:
median `failed` confidence is 1.00 for `llm` (64% of items ≥ 0.99) and
for readout (84% ≥ 0.99), 0.95 for gpt-oss (98% ≥ 0.90).

Held-out split (80 items; majority-class rates: not failed 0.688,
supported 0.975):

| backend | failed acc | failed AUROC | Brier | ECE | conf→err AUROC | supported acc | category acc | generalizable acc | paired | s/item |
|---|---|---|---|---|---|---|---|---|---|---|
| `llm:gemma4:26b` | 0.700 | 0.700 | 0.302 | 0.301 | 0.535 | 0.963 | 0.537 | 0.588 | 0.766 | 36.9 |
| `readout:gemma4:26b` | 0.688 | 0.694 | 0.326 | 0.335 | 0.512 | 0.975 | 0.700 | 0.600 | 0.753 | 2.8 |
| `llamacpp:…gpt-oss-20b` | 0.662 | 0.597 | 0.472 | 0.473 | 0.454 | 0.925 | 0.350 | 0.588 | 0.643 | 13.0 |
| `rules` | 0.688 | 0.730 | 0.199 | 0.135 | 0.693 | 0.963 | 0.525 | 0.287 | 0.807 | 0.0 |

Calibration split (120 items; majority-class rates: not failed 0.708,
supported 0.967):

| backend | failed acc | failed AUROC | Brier | ECE | conf→err AUROC | supported acc | category acc | generalizable acc | paired | s/item |
|---|---|---|---|---|---|---|---|---|---|---|
| `llm:gemma4:26b` | 0.742 | 0.753 | 0.242 | 0.235 | 0.586 | 0.958 | 0.608 | 0.583 | 0.723 | 38.1 |
| `readout:gemma4:26b` | 0.708 | 0.720 | 0.368 | 0.379 | 0.496 | 0.967 | 0.733 | 0.558 | 0.711 | 2.7 |
| `llamacpp:…gpt-oss-20b` | 0.725 | 0.650 | 0.506 | 0.511 | 0.442 | 0.967 | 0.350 | 0.558 | 0.635 | 13.6 |
| `rules` | 0.692 | 0.698 | 0.216 | 0.162 | 0.651 | 0.967 | 0.475 | 0.392 | 0.771 | 0.0 |

What the `failed` accuracies are made of, at the frozen cuts on held-out
(25 failed, 55 clean): `llm` predicts failed on 33 (17 right, 16 wrong, 8
missed; recall 0.68, precision 0.52); gpt-oss on 14 (6, 8, 19 missed);
`rules` on 6 (3, 3, 22 missed); readout on 0. The two reasoning judges
agree on `failed` at 0.5 for 121 of 200 items. `rules` has the best
ranking (AUROC 0.730) and the best calibration (Brier 0.199, ECE 0.135)
of the four; the model judges are over-confident. Where `llm` and readout
are ≥ 0.90 confident on `failed` (75 of 80 items each) they are wrong on
31% and 33%; gpt-oss is ≥ 0.90 confident on 79 of 80 and wrong on 52%
(Brier 0.47, ECE 0.47). At 0.5 gpt-oss calls 58 of the 80 turns failed
(accuracy 0.49); the frozen 0.97 cut is what lifts it to 0.662.

Write decisions at the frozen thresholds. Calibration: `llm` writes 3, all
on already-enforced lessons (precision 0.00); the other three write
nothing. Held-out: `llm` writes 1, correct (precision 1.00, miss rate 0.97
over the 33 that should be written); the others write nothing (miss rate
1.00). Reviews: `rules` 0, `llm` 26/17, readout 58/37, gpt-oss 94/58
(calibration/held-out). Under the current action rule the backends do not
write; the 0.59 floor above says the rule, not only the judges, is the
reason.

Cascade to `llm` on `failed`, held-out: readout at gate 0.78 escalates 5%,
accuracy 0.675 at 13% of the reference cost; gpt-oss at 0.80 escalates 0%,
0.662 at 35%; `rules` at 0.90 escalates 57%, 0.700 at 57%.

Success criteria on the held-out split, as the report computes them:

1. Within 3 points on `failed` and `supported` at ≤10% latency: readout
   yes (0.688/0.975 vs 0.700/0.963 at 7.5% of the latency); `rules` yes
   (0.688/0.963 at zero cost); gpt-oss no (0.662/0.925 at 35%).
2. Confidence AUROC ≥ 0.75: no for every backend (best `rules` 0.693; the
   reference itself 0.535).
3. Cascade within 1 point at ≤50% cost: no for every backend.
4. `rules` within 3 points of the reference on `failed`: yes (0.688 vs
   0.700).

Reading. Criteria 1 and 4 pass only arithmetically. The reference is one
item better than the majority class on held-out (56 vs 55 of 80),
readout's pass is the constant "not failed" classifier, and `rules`
predicts failed six times. Nothing here separates failed turns from clean
ones well enough to triage on: AUROC 0.60 to 0.73 across the four, with
the rule-based floor on top. The decision-model role in Phase 4 is
therefore treated as limited to write-time admission screening, per the
closing rule above, with two amendments to carry forward: the action rule
should not require `failed` for a write (48 of the 81 lessons worth
keeping came from clean turns), and `enforced_by` is the label that
mattered most for agent-started sessions, so an admission screen should
ask it. The live week stands as specified and its results are appended
below when available.

### Run `exp3`: changes from `exp2`

Decided 2026-11-07 from the exp2 labeling notes and numbers. exp2 stays
as the baseline; exp3 is a new sample with a new seed and the versions
below, so nothing in exp2 is re-scored under new rules.

What exp2 showed that the design did not allow for:

- 108 of 200 items had no recorded outcome: the turn was closed by the
  next user message (56 by a person, 17 by a parent agent) or by the end
  of the session (35). The header said `outcome=none`, the rubric had no
  way to say "this turn did not reach an end-state", and 40% of those
  items were labeled failed against 14% of completed turns.
- The drafter saw NEXT MESSAGE. When a person interrupted, the request and
  the unrelated follow-up were fused into one lesson.
- The drafter was told to write a lesson even when the turn showed no
  failure, and did not know whether the turn had finished.
- `category` was `none` on 136 items because every option is a failure
  cause and 48 of the 81 lessons worth keeping came from clean turns.
- `supported` was `n` on 6 items, yet the reviewer recalls lessons that
  misread the evidence. "Every claim traces to a cited line" is literal;
  it does not catch a lesson that cites the right lines and reads them
  wrong.
- Labeling could not go back a question, confirm an item, reopen one, or
  record a note, so input slips stayed in the file.
- The write rule required `failed`, which put a 0.59 floor under every
  miss rate before any judge was consulted.

Changes, by component:

**Candidates (`triggers_v3`).** Each turn records `closed_by`: `outcome`,
`next_message`, or `session_end`. T0 controls must be `completed`;
triggered strata keep turns closed any way. Stratum rules otherwise
unchanged. `closed_by` is added to the shared facts for `rules` and the
report.

**Projection (`projection_v3`).** The header gains `closed_by=…`. A
`for_drafter` rendering is the same text without the NEXT MESSAGE line;
the judge keeps the full projection. The sidecar is unchanged, so cited
line numbers mean the same thing in both renderings.

**Drafter (`drafter_v2`).** Two arms on the same items, so Stage B
compares drafters on identical turns:

- `ollama:<model>` as before (gemma4:26b), and a frontier model through
  OpenRouter's OpenAI-compatible chat endpoint, spec `openrouter:<model>`,
  key from `OPENROUTER_API_KEY`, allowed only with
  `AGENT_MEMORY_JUDGE_ALLOW_REMOTE=1` (projections are already redacted;
  this arm sends them off the machine). `llamacpp:<model>` is accepted as
  a third local kind for parity with the judges. The frontier model is
  pinned at run time and recorded in `run.json`; cost per draft is
  recorded from the gateway's usage.
- Input is the `for_drafter` rendering. The instructions change in three
  ways: the lesson must come from this turn's own evidence; when
  `closed_by` is `next_message` or `session_end` the request may not have
  been carried out and the lesson, if any, is about the attempt; and the
  drafter may return `{"lesson": null}` to say the turn holds nothing
  worth keeping. A declined draft is stored and is scored as a
  prediction of `worth_remembering = n`.
- `drafts.jsonl` holds one row per (item, drafter); every draft hash is
  recorded on the verdicts and labels it produced.

**Questions (`questions_v2`).** Stage A, projection only:

| id | type | change |
|---|---|---|
| `failed` | noul | unchanged |
| `incomplete` | noul | new: the turn ended before the agent finished (cut off by the next message or the session end), so success or failure cannot be read from it |
| `user_corrected` | noul | unchanged |
| `env_confirmed` | noul | unchanged |
| `category` | choice | unchanged options; scored only on items labeled failed |
| `recurrence` | noul | unchanged |

Stage B, projection plus one draft, asked once per draft:

| id | type | change |
|---|---|---|
| `grounded` | noul | was `supported`: every claim traces to the cited evidence |
| `faithful` | noul | new: the lesson's account of what happened in the turn is right, not a misreading of the evidence it cites |
| `generalizable` | score 0..2 | unchanged |
| `actionable` | noul | unchanged |
| `enforced` | choice | new: none, harness, project_docs, tooling; mirrors the `enforced_by` label. The header's `session_started_by` is most of the signal for `harness`; whether a judge can answer the rest from the projection is a question exp3 asks |

Derived action (v2): `write_lesson` when `grounded`, `faithful`,
`generalizable ≥ 1` (probability mass on levels 1 and 2), and `enforced =
none` all clear their thresholds; `review` when any sits inside the band;
`discard` otherwise, and always `discard` for a declined draft. `failed`
and `recurrence` no longer gate a write. With two drafts per item the
action is per draft.

**Labeling.** Per item: `failed` gains `i` (incomplete), stored as
`failed: null, incomplete: true`; `category` is asked only when `failed =
y`; `worth_remembering` as before; `lesson_kind` is asked when
`worth_remembering = y`, with options `project_fact` (where things are,
conventions, commands), `tool_quirk`, `process` (verify before claiming,
read before editing, and the like), `environment` (machine or setup fact),
`user_preference`; `error_intended` as before. Per draft, both drafts
shown together in a seeded order as A and B with the drafter hidden:
`grounded`, `faithful`, `generalizable`, `enforced_by`; then `preferred`
(A, B, same, neither) once per item. A declined draft is shown as
"(no lesson)" and asks nothing. An optional free-text `note` closes the
item. Controls: `b` back one question, a one-line summary before saving
with `r` to redo the item, `s` skip, `q` quit. `mem judge label --item ID`
reopens items; `--review` steps through labeled items showing current
values. A changed label keeps its previous value in a `history` list on
the row, so the file stays append-only in spirit and the report can count
how many labels moved.

**Decide.** Stage A is judged once per item and Stage B once per draft;
the verdict row is per (item, draft) and carries the shared Stage A
answers, so Stage A is not paid twice. Ledger events are per verdict row
as before, keyed on backend, item, questions version, projection hash,
and draft hash.

**Report.** Stage A metrics on unique items; Stage B metrics per drafter
and per backend. New: a drafter section (decline rate and its precision
against `worth_remembering`; `grounded`, `faithful`, `generalizable`, and
`enforced_by` rates per drafter; `preferred` share; cost and latency per
draft), slices by `closed_by` and by the `incomplete` label, the
label-change count from `history`, and the write decision scored against
the same `should write` label (worth remembering, enforced by nothing),
which the v2 rule can now reach. Freeze is unchanged: per-backend cuts on
calibration, held-out scored after.

**Sample.** `--n 200 --seed 3`, `human_share` 50%, `clean_share` 25%;
both drafters on all 200. Labeling load is about 12 answers per item (4
to 5 Stage A, 4 per draft, `preferred`, `note`), roughly twice exp2.

**Success criteria.** Criteria 1 to 4 stand, with two additions stated
now: (5) a drafter arm is adopted for Phase 4 if its drafts are
`faithful` on at least 90% of labeled items and `preferred` at least as
often as the other arm, at a recorded cost per draft; and (6) the v2
write rule, with the reference judge's answers, reaches write precision
≥ 0.7 and miss rate ≤ 0.5 on held-out. Criterion 1 is read against the
majority-class rate as well as the reference, since exp2 passed it with a
constant classifier.

### exp3 changes, built 2026-11-07

The exp3 design above is implemented as written, with these details and
deviations:

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
- Labeling: items already labeled but missing answers for a newly drafted
  arm come first and are asked only those steps. `--item ID` relabels in
  full with the current labels shown; `--review` shows them and asks keep
  or redo. `history` records only values that moved; a key that was
  unanswered (`null`) and is now filled in is not a change.
- Report: Stage A and Stage B are separate tables (per item and per draft);
  the write table lists each backend's gates and thresholds; criterion 1
  reports the majority-class rate; criterion 6 is evaluated per reference
  drafter on held-out for `questions_v2` runs only; the `incomplete` slice
  appears only when the label exists.
