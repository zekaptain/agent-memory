# Judge experiment spec: runs exp1 to exp4

This text was moved out of
[judge-experiment-spec.md](../judge-experiment-spec.md) on 2026-11-11. It
holds the success criteria, the first test plan, and the logs of the runs
exp1 to exp4: the designs as decided at the time, the results, and the data
notes. It is kept for the record and is not the current contract; where it
says "above" or "below" it means the spec as it stood when the text was
written.

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

## Run `exp1` (2026-10-26): pilot, superseded by `exp2`

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

## Run `exp2` (2026-10-26): `--n 200 --seed 2`

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

## Run `exp3`: changes from `exp2`

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
item. Controls: `b` back one question, `p` to page the full projection,
`?` to page the rubric, a one-line summary before saving with `r` to redo
the item, `s` skip, `q` quit. `mem judge label --item ID`
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

## exp4 changes, as built

The amendment above is implemented in the shared lesson model, all
three drafter transports, the judge pipeline, both label frontends,
and the report. Pydantic is now a direct dependency; no resolved
package version changed. Regression tests use synthetic turns and
mocked model HTTP, with real Quack integration for per-claim decision
events. No hosted model calls or live-run writes are needed to test
the implementation.

Historical checks on temporary copies reproduced exp2 and exp3's raw
reports exactly, apart from the generated timestamp. Exp1 still refuses
to render because it has no labeled verdicts. SHA-256 checks confirmed
that the original exp1, exp2, and exp3 run files were unchanged.
