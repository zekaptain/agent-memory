# Admission: captured turns to production memory

Implemented. Admission lives in `src/agent_memory/admission/`; the judge
experiment remains separate.
No migration is added. Existing memory types, the `judge` agent from 0005,
and client authorization cover the writes.

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
See [the judge how-to](judge-how-to.md) for setup. A Jev comparison arm is
optional and requires explicit consent and
`AGENT_MEMORY_JUDGE_ALLOW_REMOTE=1`. Jev answers typed questions; it does
not learn from labels. Admission learns through its logistic model and
neighbor vote.

## Production command order

Stop the production server before `mem migrate` if the `judge` agent is
missing, then restart it. Every admission command checks for that agent
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
approver.json          fitted coefficients, cuts, model identities, metrics
examples.jsonl         imported human labels and production review history
days/<day>/
  run.json             pinned configuration
  examples.jsonl       pinned neighbor set
  items.jsonl          deterministic candidates and evidence sidecars
  projections/*.txt    redacted evidence
  drafts.jsonl
  verdicts/*.jsonl
  scores.jsonl
  admitted.jsonl
  reviews.jsonl        durable save intents and completed reviews
```

Directories are 0700 and files 0600. Admission commands serialize local
mutations with a nonblocking file lock. Concurrent capture remains possible.
The day's `RunStore` uses its existing `root=` argument; it does not bypass
or weaken an experiment guard.

Import requires a frozen v4 run. Only explicit boolean `store_exact`
labels with current draft and projection hashes enter the example set.
The human `rule_supported` label is retained. Embeddings contain only
rule, kind and claim texts, never the projection. Embedding requests use
`endpoints.post_json` and the existing egress boundary.

The model uses the judge's grounded, faithful, rule-supported,
generalizable and unenforced probabilities, lesson-kind and caller
indicators, a similarity-weighted vote of the eight nearest labels, and
maximum similarity. Standardized, L2-regularized logistic regression uses
deterministic fixed-step gradient descent, without another ML dependency.

Calibration fits coefficients and cuts; held-out labels do neither.
Neighbor features exclude the training item's own drafts, and held-out
evaluation uses calibration neighbors only. Importing an item in different
experiment runs does not give it a second grouping identity; conflicting
splits are refused. Production review of a previously imported item
replaces its experiment training influence for that backend and embedder.
The high cut is the lowest
calibration cut reaching 0.85 precision. The low cut is the highest cut
whose discard miss rate does not exceed the frozen judge's calibration
miss rate. If the data cannot establish a safe write cut, nothing is
automatically written.

Three guards cannot be learned away: a frozen judge `discard` never becomes
an approver `write`; similarity below the floor (default 0.35) never
becomes a `write`; and a draft with a write gate the judge left unanswered
goes to review whatever its score, and does not help choose the write cut
(see "Unanswered gates" in the judge spec). Scores between cuts go to
review. Reports include precision, miss rate, coverage, Brier score and
AUROC for both splits.

Auto mode requires held-out precision of at least **0.8** over at least
**20 scored held-out writes**:

```sh
uv run mem admit enable --auto
```

`--min-writes N` changes the count, not the precision bar. No scored
writes is not perfect precision. Imports, human feedback and refits return
the mode to shadow. Changed examples require a refit before another run or
auto enablement; nothing automatically enables the new approver.
Older pinned days cannot borrow a newer model's auto permission.

## Daily processing and ledger mapping

Admission reads captured production events directly; it never imports
transcripts. It segments turns and uses all deterministic candidates
(`n=None`), including clean controls. The watermark starts at the
beginning of captured history. Later runs include sessions started since
the watermark and older sessions with newly recorded events, so ongoing
or late-imported capture is not silently lost.

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

Content-hash deduplication confirms an existing rule and adds provenance
instead of creating a second memory. A `decision` event under agent
`judge`, harness session `admission-<day>`, records probabilities and
identities without rule, claim, or projection text. In auto mode only a
`write` also gets an `approve`, actor `judge`, with approver version and
score in its reason. `review` remains quarantined. Discards stay in day
files and make no ledger writes.

## Human review and feedback

`mem admit review` uses the existing Go frontend, or `--plain` for
numbered prompts. It does not use the experiment's `LabelController`.
The rule, kind, claims and cited excerpts stay visible. The judge arm,
approver verdict and probability are hidden.

One question asks "Store this exact lesson?" with Yes, No, Unclear,
followed by an optional note and save confirmation:

- Yes records `approve`, actor `human`.
- No records `retire`, actor `human`.
- Unclear records a review but leaves ledger status alone.
- A sampled discard approved by the human is first admitted, then
  approved. A rejected discard stays out of the ledger.

The default seeded discard sample is 30; `--discards N` adjusts it.
`--limit N` bounds a sitting. Back, redo, skip, quit and resume work in
both frontends. `--again` reopens completed reviews, retaining history.
Interrupted saves finish from their private durable intent on resume.
Notes are redacted before persistence.

Retirement is terminal in the ledger. Correcting a No to Yes therefore
creates a fresh human-approved memory linked to the retired one, rather
than pretending an approve action can reactivate a retired row. Its
examples keep the original grouping identity and split. A rejected
discard uses a stable draft identity until it has a ledger identity.

Every Yes or No appends a human example. Correcting an earlier label to
Unclear withdraws its training influence without changing ledger status.
Production examples use a stable 60/40 split grouped by memory identity;
corrections replace a memory's previous training influence rather than
multiplying its weight. Run
`mem admit fit` after reviews to refresh the fitted model and held-out
estimate. Review sampling estimates are conditional on the sampled
traffic, not an unbiased full-stream miss-rate estimate.

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
guard-free judge building blocks. The only shared pipeline extension
lets `decision_record` accept a harness session and project; defaults
preserve historical experiment events and report behavior.

Local drafters are required. Hosted judge arms remain optional behind
`AGENT_MEMORY_JUDGE_ALLOW_REMOTE=1`; a remote embedder host uses the same
opt-in rule. No projection text enters examples or approver metadata.
Redaction runs on model drafts and again before ledger insertion.
Synthetic regression runs and throwaway servers are the validation
fixtures; no production labeling or enablement is part of implementation.
