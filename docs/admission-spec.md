# Admission: captured turns to production memory

Implemented. Admission lives in `src/agent_memory/admission/`; the judge
experiment remains separate.
Migration 0007 registers the `admission` agent that admission writes as.
Existing memory types and client authorization cover the writes.

## Before admission: human calibration

There are no usable exact-draft labels in the historical runs. Exp2 has
200 v1 turn labels, exp3 has three, and exp4 has 600 drafts and 600 Jev
verdicts without human labels. None answers v4's `store_exact` question.
Admission cannot fit or enable an approver from those runs.

Build a new v4 run in the experiment home. These commands are a human
workflow, not an installation step or authorization to call models:

```sh
uv run mem --home "$HOME/.agent-memory-judge" --port 9495 judge candidates --run exp5 --n 200 --seed 5
uv run mem --home "$HOME/.agent-memory-judge" --port 9495 judge draft --run exp5 --drafter ollama:gemma4:26b
uv run mem --home "$HOME/.agent-memory-judge" --port 9495 judge draft --run exp5 --drafter llamacpp:ggml-org/gpt-oss-20b-GGUF:MXFP4
uv run mem --home "$HOME/.agent-memory-judge" --port 9495 judge decide --run exp5 --backend llm:gemma4:26b
uv run mem --home "$HOME/.agent-memory-judge" --port 9495 judge label --run exp5
uv run mem --home "$HOME/.agent-memory-judge" --port 9495 judge report --run exp5 --freeze
```

The experiment server must already be running on 9495, with captured
sessions imported. Finish drafting and calibration labeling before freezing.
See [the judge how-to](judge-how-to.md) for setup. A Jev judge for
comparison is optional and requires explicit consent and
`AGENT_MEMORY_JUDGE_ALLOW_REMOTE=1`. Jev answers typed questions; it does
not learn from labels. Admission learns through its logistic model and
neighbor vote.

## Production command order

Stop the production server before `mem migrate` if the `admission` agent
is missing, then restart it. Every admission command checks for that agent
before doing work. Use production home and port for admission, not 9495:

```sh
uv run mem --home "$HOME/.agent-memory" --port 9494 admit import-labels \
  --from "$HOME/.agent-memory-judge/judge/exp5" --backend llm:gemma4:26b
uv run mem --home "$HOME/.agent-memory" --port 9494 admit fit --backend llm:gemma4:26b
uv run mem --home "$HOME/.agent-memory" --port 9494 admit status
uv run mem --home "$HOME/.agent-memory" --port 9494 admit enable --shadow
uv run mem --home "$HOME/.agent-memory" --port 9494 admit run
uv run mem --home "$HOME/.agent-memory" --port 9494 admit review --day 2026-11-10
```

Install the local models before running: default drafter and judge are
`gemma4:26b` on Ollama; the embedder is `nomic-embed-text`.
`--embedder ollama:MODEL` on import and fit must match. Drafting also
supports a loopback llama.cpp server via `--drafter llamacpp:MODEL`.
`run --backend` is a consistency check, not a way to switch a fitted
approver to a different judge. Each day pins its approver, neighbor
snapshot, drafter, and mode; changing configuration applies to a new day.

## Storage and fitting

Private files under `<home>/admission/`:

```text
state.json             mode, watermark, approver version, minimum writes
approver.json          fitted coefficients, thresholds, model identities, metrics
examples.jsonl         examples from imported labels and from admission review
days/<day>/
  run.json             pinned configuration
  examples.jsonl       pinned neighbor set
  items.jsonl          deterministic items and evidence sidecars
  projections/*.txt    redacted evidence
  drafts.jsonl
  verdicts/*.jsonl
  scores.jsonl
  admitted.jsonl
  reviews.jsonl        durable save intents and saved answers
```

Directories are 0700 and files 0600. Admission commands serialize local
mutations with a nonblocking file lock. Concurrent capture remains possible.
The day's `RunStore` uses its existing `root=` argument; it does not bypass
or weaken an experiment guard.

Admission keeps its files with the code that keeps a judge run's files
(`judge/store.py`), and is stricter in three ways. Nothing is written
through a symbolic link, and no link is followed to change a mode.
Everything is redacted before it is written. A file is replaced whole even
to add one row, so it is never left with a cut-off row; a last row that
does not parse is therefore damage, and it is refused where a judge run
would cut it off and write it again.

Import requires a frozen v4 run. It reads the run as the judge's own
commands do, so a last draft or verdict row that a killed command cut off
is left out, and it writes nothing into the run. Only explicit boolean
`store_exact` labels with current draft and projection hashes enter the
example set.
The human `rule_supported` label is retained. Embeddings contain only
rule, kind and claim texts, never the projection. Embedding requests use
`endpoints.post_json` and the existing egress boundary.

The model uses the judge's grounded, faithful, rule-supported,
generalizable and unenforced probabilities, lesson-kind and caller
indicators, a similarity-weighted vote of the eight nearest labels, and
maximum similarity. Standardized, L2-regularized logistic regression uses
deterministic fixed-step gradient descent, without another ML dependency.

Calibration fits coefficients and thresholds; held-out labels do neither.
Neighbor features exclude the training item's own drafts, and held-out
evaluation uses calibration neighbors only. Importing an item in different
experiment runs does not give it a second grouping identity; conflicting
splits are refused. An admission review answer for a previously imported
item replaces its experiment training influence for that judge and
embedder. The write threshold is the lowest calibration score reaching
0.85 precision. The discard threshold is the highest score whose discard
miss rate does not exceed the frozen judge's calibration miss rate. If
the data cannot establish a safe write threshold, nothing is
automatically written. Stored reasons call the two the write cut and the
review cut.

Three guards cannot be learned away: a frozen judge `discard` never becomes
an approver `write`; similarity below the floor (default 0.35) never
becomes a `write`; and a draft with a write gate the judge left unanswered
goes to review whatever its score, and does not help choose the write
threshold (see "Unanswered gates" in the judge spec). Scores between the
two thresholds, the review band, go to review. Reports include precision,
miss rate, coverage, Brier score and AUROC for both splits.

Auto mode requires held-out precision of at least **0.8** over at least
**20 scored held-out writes**:

```sh
uv run mem admit enable --auto
```

`--min-writes N` changes the count, not the precision the held-out bar
asks for. No scored writes is not perfect precision. Imports, human
feedback and refits return the mode to shadow. Changed examples require a
refit before another run or auto enablement; nothing automatically enables
the new approver. Older pinned days cannot borrow a newer approver's auto
permission.

## Daily processing and ledger mapping

Admission reads captured production events directly; it never imports
transcripts. It segments turns and takes all of them as items
(`n=None`), clean controls included. The watermark starts at the
beginning of captured history. Later runs include sessions started since
the watermark and older sessions with newly recorded events, so ongoing
or late-imported capture is not silently lost. A session's start is the
one `session_summary` gives: its first event, where that is earlier than
the start its row carries.

Existing `item_id_for` identities prevent repeated admission. A rerun
can add newly captured turns to the same day but skips completed items.
Older unfinished days resume before new work. Draft, verdict and score
checkpoints survive model failures. The watermark advances only after
processing succeeds. Ledger writes use deterministic identities and one
transaction; a crash between ledger commit and file checkpoint is safe
to retry.

Each `write` or `review` creates a memory through `LifecyclePlan.remember`:

| Field | Value |
|---|---|
| content | Redacted rule alone |
| type | `fact` for project_fact/tool_quirk/environment, `procedure` for process, `preference` for user_preference |
| scope | Source session project, or global when absent |
| trust_level | `tool`, quarantined by the existing status view |
| confidence | Approver probability of storing the exact draft |
| metadata.lesson | Kind and redacted claims |
| metadata.admission | Approver version, score, reasons, draft hash, item, backend, drafter |
| memory_sources | `derived_from` opening event and every cited event in the sidecar |

Content-hash deduplication confirms an existing rule and adds its sources
instead of creating a second memory. A `decision` event under agent
`admission`, harness session `admission-<day>`, records probabilities and
identities without rule, claim, or projection text. It holds two results
side by side: the judge's action and reasons as judged (`payload.judge`),
and the approver's action, score and reasons (`payload.admission`). Its
content names both actions. Until 2026-11-11 the approver's action and
reasons were written over the judge's in `payload.judge`, and
`payload.admission` had no action; events recorded before then keep that
shape.

In auto mode only a `write` also gets an `approve`, actor `admission`,
with approver version and score in its reason. `review` remains
quarantined. Discards stay in day files and make no ledger writes.

Admission has written as its own agent since migration 0007 (2026-11-11).
Until then the decision event, the session that created the memory, and
the approver's `approve` were recorded under agent and actor `judge`, in
sessions of scope `judge`; a ledger that admitted lessons before then
keeps those rows. An admission session now has scope `admission`, says
`via: mem admit` in its metadata, and has a capture observation in source
mode `record`. `mem admit run` passes over the sessions of both agents
when it looks for new turns. The reasons are in
[ADR 0009](adr/0009-admission-writes-as-its-own-agent.md).

## Admission review and feedback

`mem admit review` uses the existing Go frontend, or `--plain` for
numbered prompts. It does not use the experiment's `LabelController`.
It has a controller of its own, and the loop that shows labeling's
screens shows this one's too (`drive` in `judge/label_frontend.py`).
The rule, kind, claims and cited excerpts stay visible. The judge, and
the approver's action and score, are hidden.

One question asks "Store this exact lesson?" with Yes, No, Unclear,
followed by an optional note and save confirmation:

- Yes records `approve`, actor `human`.
- No records `retire`, actor `human`.
- Unclear records the answer but leaves ledger status alone.
- A sampled discard the human answers Yes to is first admitted, then
  approved. One answered No stays out of the ledger.

`mem review approve` and `mem review retire` change the same memories,
also as `human`, but they are not admission review. They hide nothing,
ask nothing about the lesson, and record no example for the approver.
They are for memories that admission did not write.

The default seeded discard sample is 30; `--discards N` adjusts it.
`--limit N` bounds a sitting. Back, redo, skip, quit and resume work in
both frontends. `--again` shows answered lessons again, retaining
history. Interrupted saves finish from their private durable intent on
resume. Notes are redacted before persistence.

Retirement is terminal in the ledger. Correcting a No to Yes therefore
creates a fresh human-approved memory linked to the retired one, rather
than pretending an approve action can reactivate a retired row. Its
examples keep the original grouping identity and split. A discard
answered No uses a stable draft identity until it has a ledger identity.

Every Yes or No appends a human example. Correcting an earlier answer to
Unclear withdraws its training influence without changing ledger status.
Production examples use a stable 60/40 split grouped by memory identity;
corrections replace a memory's previous training influence rather than
multiplying its weight. Run `mem admit fit` after admission review to
refresh the approver and its held-out estimate. Estimates from the
sample of discards are conditional on the sampled traffic, not an
unbiased full-stream miss-rate estimate.

## Optional daily scheduling on macOS

Nothing installs a scheduler automatically. After the production server,
local model service, calibration and shadow run work manually:

```sh
mkdir -p "$HOME/Library/LaunchAgents"
sed -e "s|__MEM__|$(command -v mem)|g" -e "s|__HOME__|$HOME|g" \
  scripts/com.agent-memory.admit.plist \
  > "$HOME/Library/LaunchAgents/com.agent-memory.admit.plist"
launchctl bootstrap "gui/$(id -u)" \
  "$HOME/Library/LaunchAgents/com.agent-memory.admit.plist"
```

The template runs at 03:00 local time against the default production home
and 9494. It does not start model services or turn auto mode on.
Edit its home, port and model settings for your installation. Logs go to
`~/.agent-memory/admit.log`. Stop it with:

```sh
launchctl bootout "gui/$(id -u)/com.agent-memory.admit"
```

## Boundaries and as-built notes

`ensure_experiment_home` is unchanged. `mem judge` never writes lessons,
and no module under `judge/` imports admission. Admission composes
guard-free judge building blocks: it judges through `judge/verdict.py`
(`ask_turn`, `judge`) and takes a verdict's ledger copy from
`Verdict.decision_record`, which always describes an experiment's event.
Admission replaces the agent, the session, and the source key with its
own before it writes the event, so experiment events and reports are
untouched.

Local drafters are required. Hosted judges remain optional behind
`AGENT_MEMORY_JUDGE_ALLOW_REMOTE=1`; a remote embedder host uses the same
opt-in rule. No projection text enters examples or approver metadata.
Redaction runs on model drafts and again before ledger insertion.
Synthetic regression runs and throwaway servers are the validation
fixtures; no production labeling or enablement is part of implementation.
