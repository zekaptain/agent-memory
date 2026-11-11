# Running judge experiments

Run these commands from the repository root. The order is:

```text
init → serve → import → candidates → draft → decide → label → report → report --freeze
```

After freezing, you can test the same backends on new sessions with `live`.
Drafted lessons stay in experiment files. They are never written into the
memory ledger.

## 1. Start the experiment server

Use a separate home and port, not your production ledger. In terminal 1:

```sh
uv sync
uv run mem --home "$HOME/.agent-memory-judge" --port 9495 init
uv run mem --home "$HOME/.agent-memory-judge" --port 9495 serve
```

Leave `serve` running. It is supposed to stay in the foreground.
`init` creates the database and applies migrations; it is safe to rerun
with this server stopped, but it does not reset anything. The first
initialization needs network access to download DuckDB extensions.

A different home does not select a different port. Use the same home
and port for the server and every client command.

## 2. Import Droid sessions

In terminal 2, also at the repository root:

```sh
export AGENT_MEMORY_HOME="$HOME/.agent-memory-judge"
export AGENT_MEMORY_PORT=9495
RUN=trial-01

uv run mem judge import --sessions-dir "$HOME/.factory/sessions"
```

Choose an unused run name. Repeat the environment setup if you open another
client terminal. These variables keep the commands below short. Without
them, put the root options before `judge`:

```sh
uv run mem --home "$HOME/.agent-memory-judge" --port 9495 judge import \
  --sessions-dir "$HOME/.factory/sessions"
```

Import options:

| Option | Use |
|---|---|
| `--sessions-dir PATH` | Required. Directory containing Droid transcript files. |
| `--project SLUG` | Only import files whose parent directory has this exact project slug. Use the directory name under `~/.factory/sessions`, not the repository's display name. |
| `--since 2026-11-01` | Only import files modified on or after this date. Also accepts an ISO datetime; no timezone means UTC. This is a file modification filter, not a turn-date filter. |
| `--limit 20` | Import at most 20 transcripts this invocation. |

To restrict your first import to a small recent batch, use this instead
of the unfiltered import:

```sh
uv run mem judge import --sessions-dir "$HOME/.factory/sessions" \
  --since 2026-11-01 --limit 20
```

Import is safe to repeat: unchanged files are skipped and existing events
are not duplicated. Check the import summary and any errors before continuing.
Import has no `--run`: it fills the experiment ledger shared by runs in
this home. Import filters do not remove data imported earlier.

## 3. Create the sample

```sh
uv run mem judge candidates --run "$RUN" \
  --n 200 --seed 1 --clean-share 0.25 --human-share 0.5
```

This samples turns from the imported ledger and saves their evidence
projections. It also assigns calibration and held-out splits automatically.

| Option | Default | Use |
|---|---|---|
| `--run ID` | Required | Name of the new run. |
| `--n N` | `200` | Requested number of turns. Use a smaller sample for a practice run. |
| `--seed N` | `1` | Reproducible sampling and split assignment for the same input pool. |
| `--clean-share F` | `0.25` | Target fraction of clean control turns, from 0 to 1. |
| `--human-share F` | `0.5` | Target fraction from sessions started by a person, from 0 to 1. |

Check the printed pool and chosen counts. Available turns can limit the
sample and its target fractions. A clean control has no matching failure
trigger; it is not a human judgment that the turn succeeded.

Run `candidates` once per run name. To change the sample, use a new name.
`purge` does not reset a run.

## 4. Draft lessons

Have Ollama running with the chosen model installed, or configure another
transport from [Model choices](#model-choices) before continuing.

```sh
uv run mem judge draft --run "$RUN" --drafter ollama:gemma4:26b
```

Each item gets a structured lesson with a rule and supporting claims, or
a decline if the drafter finds nothing worth keeping. A decline counts
as a finished draft.

- Omitting `--drafter` uses `ollama:gemma4:26b`.
- Repeat `--drafter` to compare models. Each drafter processes every item.
- Add `--limit 5` to process at most five pending items per drafter.
  Rerun without the limit to finish.

For example, with a second model served by llama-server:

```sh
uv run mem judge draft --run "$RUN" \
  --drafter ollama:gemma4:26b --drafter llamacpp:MODEL_ID
```

Replace `MODEL_ID` with your server's model ID or alias. Finish every
intended drafter before starting the full judging pass. Rerunning `draft`
skips completed items and retries missing or failed ones.

An item whose draft fails is skipped with a warning, and the command
goes on to the next item. When it has skipped any, it exits 1 after
writing the rest, so a script can tell that drafting is not finished.
Run `draft` again to retry the skipped items. The command stops early in
two cases only, each after three failures in a row: the model server
cannot be reached, or the drafter has not written a draft in this run
yet. The error says which, and what the last failure was.

Run one `draft` at a time on a run. A second one started while the first
is at work is refused at once, whichever drafters it names. To use two
drafters, name both in one command.

## 5. Run the judges

Start with the rules baseline and a local reasoning judge:

```sh
uv run mem judge decide --run "$RUN" \
  --backend rules --backend llm:gemma4:26b
```

The turn questions judge the turn once per backend. The draft questions
judge each non-declined draft, including each claim's grounding and
faithfulness. The report calls them Stage A and Stage B. With multiple
drafters, expect multiple verdict rows per item.

- `--backend SPEC` is required and repeatable. See [Model choices](#model-choices).
- `--limit 5` processes at most five pending items per backend, including
  their drafts. Rerun without the limit to finish.
- `--no-events` keeps verdict files but skips recording decision events
  in the experiment ledger. Without it, keep the experiment server running.

Rerunning the same command skips current verdicts and retries missing or
failed ones. A failed item is skipped with a warning, and a command that
skipped any exits 1 after judging the rest; run it again to retry them.
`decide` stops early in the same two cases as `draft`: three failures in
a row because the model server cannot be reached, or from a backend that
has not written a verdict in this run yet. The backends named after the
one that stopped are not run.

Two `decide` commands can run on one run at the same time when they name
different backends. A `decide` that names a backend another `decide` is
already judging is refused at once, before it calls a model.

## 6. Add human labels

```sh
uv run mem judge label --run "$RUN" --limit 10
```

Review the turn, then each draft and its claims. The interface hides
model verdicts and drafter identities. Labels must be your judgments,
not another model's answers.

Use arrows and Enter to answer, Page Up/Down to scroll evidence, `b` to
go back, `p` for the full projection, and `?` for the guide. Review the
summary and save each item. Skipping or quitting does not save unfinished
answers for the current item.

The full-screen interface requires Go 1.26.4 or newer and may download
pinned dependencies on its first build. To avoid Go and use numbered
prompts instead:

```sh
uv run mem judge label --run "$RUN" --plain --limit 10
```

| Option | Use |
|---|---|
| `--limit N` | Stop after N items this session. Omit it to continue until done or you quit. |
| `--seed N` | Item and blinded draft order; default `1`. Keep it fixed when resuming. |
| `--review` | Step through saved labels to keep or redo them. |
| `--item ITEM_ID` | Label or fully relabel this item, showing existing answers. Repeat for multiple items. Find IDs in the run's `items.jsonl`. |
| `--plain` | Use linear prompts without the Go interface. Pipes and `TERM=dumb` select this automatically. |

Repeat `label` to resume. It skips completed items and asks only missing
questions when backfilling. Finish the intended labels before freezing;
the simplest approach is to label the whole sample.

Two sessions can label one run at the same time, and each keeps the labels
the other saves. Give them different items with `--item` or different
orders with `--seed`: two sessions with the same seed start on the same
item. If both save one item, the first label is kept and the second
session ends with a message; start it again to go on with the items that
are left. An item the other session finished is left out when its turn
comes.

## 7. Check progress and preview the report

```sh
uv run mem judge status --run "$RUN"
uv run mem judge report --run "$RUN"
```

`status` shows item, draft, verdict, and label counts, plus whether the run
is frozen. Before moving on, check that every intended drafter has covered
the sample and every intended backend has its expected verdicts.

`report` needs at least one labeled verdict. It writes:

```text
~/.agent-memory-judge/judge/trial-01/report.md
```

It shows calibration results before freeze; held-out backend scores stay
locked. For Markdown output instead of wide terminal tables:

```sh
uv run mem judge report --run "$RUN" --raw
```

You can preview the report repeatedly. It includes turn-level scores,
draft-level scores, claim metrics, and write/review/discard decisions.

## 8. Freeze thresholds and score held-out items

Only do this after finishing the intended drafts, verdicts, and calibration
labels:

```sh
uv run mem judge report --run "$RUN" --freeze
```

This chooses per-backend thresholds using calibration labels, locks them
in `run.json`, and reports held-out scores with those thresholds.
Freeze is one-time: a second `--freeze` is refused. Use a new run for a
revised experiment rather than editing frozen metadata.

`--freeze` is refused while calibration items are not fully labeled: an
item with no label, or one whose label has questions left because a
drafter was added after it was labeled. The message says how many there
are, and nothing is frozen. Run `mem judge label` to finish them; a
session opens them along with any held-out items still to label. To
freeze on the labels there are anyway:

```sh
uv run mem judge report --run "$RUN" --freeze --partial
```

A partial freeze cannot be undone either. The run records how many
calibration items it went without, and every report of the run says so.
`--freeze` does not check verdicts: use `status` to see that every
backend has judged the sample.

Later, refresh the report without freezing again:

```sh
uv run mem judge report --run "$RUN"
```

## Model choices

Use these specifications with `--drafter` or `--backend`. Replace
`MODEL_ID` and `PROVIDER/MODEL` with models available on your service.
Local services must already be running; `mem judge` does not start
them or download models.

| Role | Specification | Service |
|---|---|---|
| Drafter | `ollama:MODEL_ID` | Ollama; default drafter is `ollama:gemma4:26b`. |
| Drafter | `llamacpp:MODEL_ID` | llama-server's OpenAI-compatible endpoint. |
| Drafter | `openrouter:PROVIDER/MODEL` | Hosted OpenRouter chat model; requires opt-in below. |
| Judge | `rules` | Deterministic baseline; no model service needed. |
| Judge | `readout:MODEL_ID` | Ollama choice-probability readout, with a sampling fallback. |
| Judge | `llm:MODEL_ID` | Ollama reasoning judge. |
| Judge | `llamacpp:MODEL_ID` | llama-server reasoning judge. |
| Judge | `jev` or `jev:MODEL_ID` | Hosted OpenRouter decision model; requires opt-in below. |

Set `OLLAMA_HOST` if Ollama is not at `http://localhost:11434`.
Set `LLAMACPP_HOST` if llama-server is not at `http://localhost:9931`.
Set these in the client shell before running `draft` or `decide`.

Include an `llm:` backend if you want the report's reference-judge
comparison and cascade metrics. The report selects the `llm:` backend
with the most labeled verdicts; `llamacpp:` does not fill that role.

Hosted calls send redacted session evidence outside your machine and may
cost money. Only enable them if you intend to share that data. Load
`OPENROUTER_API_KEY` into your environment using your normal secret-handling
method, then opt in:

```sh
export AGENT_MEMORY_JUDGE_ALLOW_REMOTE=1
```

For example, before freezing, you can add a hosted drafter or judge:

```sh
uv run mem judge draft --run "$RUN" --drafter openrouter:PROVIDER/MODEL
uv run mem judge decide --run "$RUN" --backend jev
```

If you add a drafter, finish its drafts, rerun your intended judges, and
resume labeling to fill in the new draft questions before freezing.

## 9. Optional: test on new sessions

After freeze, use `live` on sessions collected for the live test. Replace
these example dates with your live period's start and batch date:

```sh
uv run mem judge live --run "$RUN" \
  --since 2026-11-09 --day 2026-11-09 \
  --sessions-dir "$HOME/.factory/sessions" \
  --drafter ollama:gemma4:26b \
  --backend rules --backend llm:gemma4:26b
```

This imports recently modified transcripts, selects previously unseen
turns from sessions **started** on or after `--since`, drafts lessons,
and judges them using the parent's frozen thresholds. Use the same model
specifications you calibrated.

- `--since` is required; it accepts an ISO date or datetime.
- `--day` names the batch. It defaults to today's UTC date.
- `--sessions-dir` defaults to `~/.factory/sessions`.
- `--backend` is repeatable; omitting it runs only `rules`.
- `--drafter` is repeatable; omitting it skips drafting, even if the parent
  run had drafters.

Rerunning an existing day resumes its fixed item list; it does not append
new turns. Choose a new `--day` for another batch. Keep `--since` at the
live period's start if you want to include unseen turns from earlier
sessions in that period. `live` exits 1 when a draft or verdict of the
day failed and was skipped; rerun the same day to retry it.

Inspect, label, and report the batch with `--live DAY`:

```sh
uv run mem judge status --run "$RUN" --live 2026-11-09
uv run mem judge label --run "$RUN" --live 2026-11-09
uv run mem judge report --run "$RUN" --live 2026-11-09
```

`--live DAY` also works on `draft` and `decide` for resuming those stages.
Do not freeze the child again: it already inherits the frozen thresholds.

## 10. Import human labels into admission

After human-labeling and freezing a v4 run, switch to the production home
and port to import its exact-draft labels:

```sh
uv run mem --home "$HOME/.agent-memory" --port 9494 admit import-labels \
  --from "$HOME/.agent-memory-judge/judge/exp5" --backend llm:gemma4:26b
```

This reads experiment files without changing them. It does not make
`mem judge` write lessons or enable automatic approval. Follow the
[admission spec](admission-spec.md) for fitting, shadow runs and review.

## Resume, stop, or clean up

- Resume `draft`, `decide`, or `label` with the same run and options.
  Saved work stays in place. Do not change drafts underneath a running judge.
- Use a new `RUN` and start at `candidates` to try a different sample.
  Reimport only if you need new transcripts.
- Runs made with older question sets are read-only for `draft`, `decide`,
  and `label`. Start a new run with the current code. Historical reports
  can still be generated when the run has labeled verdicts.
- Stop the experiment server with Ctrl-C in terminal 1. This does not
  delete the experiment. Restart the same `serve` command to continue.

When you no longer need the evidence projections:

```sh
uv run mem judge purge --run "$RUN"
```

`purge` deletes projection text for the run and all its live batches.
It keeps drafts, verdicts, labels, metadata, and imported ledger data.
It is not a full privacy wipe or a reset, and drafting, judging, and
labeling cannot continue without the projections.

For any command's full option list, append `--help`, for example
`uv run mem judge decide --help`. For design details and historical runs,
see the [experiment spec](judge-experiment-spec.md).
