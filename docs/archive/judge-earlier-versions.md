# Judge experiment spec: earlier versions

This text was moved out of
[judge-experiment-spec.md](../judge-experiment-spec.md) on 2026-11-11. It
holds the first pipeline, `drafter_v1`, `questions_v1`, and the first
labeling screens with their rubric. The code still runs the older question
sets and write rules to reproduce the reports of stored runs; new runs use
the current definitions in the spec. The text is kept for the record and is
not the current contract.

## Pipeline

```
transcripts -> mem judge import -> experiment ledger
            -> candidates (provenance_v1, triggers_v2, seeded sample) -> projection_v2
            -> draft (drafter_v1)
            -> decide (rules | readout | jev | llm) -> verdicts + decision events
            -> label (blinded)  -> report (calibrate, freeze, held-out)
            -> live (new sessions, frozen thresholds)
```

## Drafter (`drafter_v1`)

A local Ollama model writes a lesson of at most two sentences and lists the
evidence line numbers it relied on, for every item including clean controls,
so Stage B has "unsupported lesson from a clean turn" negatives. Output is
JSON; the lesson is normalised to one line and capped at 400 characters.
Drafting is local only.

## Questions (`questions_v1`)

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
