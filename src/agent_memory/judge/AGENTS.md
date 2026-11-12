# Working on the judge experiment

This guide applies to changes under `judge/`, to the `mem judge` commands
in `cli.py`, and to running those commands, on top of the root `AGENTS.md`
and `src/agent_memory/AGENTS.md`. It holds the lessons from previous runs.
The spec is `docs/judge-experiment-spec.md`, and `docs/judge-how-to.md`
gives the experiment commands step by step. Paths here are relative to
`src/agent_memory/` unless they start with `docs/` or `tests/`.

## Separate from production

The experiment must stay separate from production. Never remove
`ensure_experiment_home`. Experiment commands never write lessons into the
memory ledger; lessons stay in run files. Nothing under `judge/` imports
admission. Verdicts use agent `judge` and may be recorded as `decision`
events without copying projection or lesson text into them.

## Running experiment commands

A separate home does not automatically select a different port. Use the
same explicit home and port for the experiment server and its clients, for
example:

```sh
# Foreground server, in one terminal after initializing this home.
uv run mem --home "$HOME/.agent-memory-judge" --port 9495 serve
# Client, in another terminal.
uv run mem --home "$HOME/.agent-memory-judge" --port 9495 judge status --run exp3
```

`serve` intentionally stays running.

## Turns, questions, and runs

- Distinguish a human request, parent-agent instructions, and harness
  notifications. A transcript's `user` role does not prove a human wrote
  it. Read full blob-backed messages when needed to tell who wrote one.
- Keep interrupted/incomplete turns distinct from failures. Preserve
  `closed_by` and clean controls. The drafter's view excludes the next
  message so unrelated steering is not fused into a lesson about the turn.
- The turn questions are asked once per turn; the draft questions once
  per draft. Reports and stored rows still call them Stage A and Stage B.
  Current write gates are grounded, faithful, exact rule supported,
  generalizable, and not already enforced. A gate the judge leaves
  unanswered is never read as a probability; it sends the verdict to
  review. A failed tool call can be intentional; a useful lesson need
  not describe a failure. Harness rules, project docs, and tooling may
  already enforce a lesson.
- Preserve seeded sampling, calibration/held-out separation, versioned
  projections/questions/drafters, hashes, and resumable per-item/per-drafter
  outputs. Older question-set runs remain read-only for draft/decide/label,
  but their reports must still reproduce the historical results. A report
  refuses a run whose verdicts were judged with another question set.
- Finish drafting before a long decide run. Do not change its drafts
  underneath it. Labeling and deciding can run independently, but freeze
  thresholds only after the intended calibration labels are complete.
  `report --freeze` refuses while a calibration item has no label or a
  label with questions left; `--partial` overrides that and the run
  records how many items were short. Keep "fully labeled" meaning what a
  sitting would still ask (`labeling.questions_left`), not a second
  definition.
- Read and write a run's files through `RunStore` only. The rules that
  keep them whole are at the top of `judge/store.py`: one command at a
  time adds rows to a file, so a second `draft`, or a `decide` for a
  judge another `decide` is asking, is refused at once; a label save adds
  one label to what is on disk and is refused if another sitting saved
  that item first. Admission keeps its files through the same code, with
  its stricter rules in `admission/store.py`.
- A draft or verdict that fails is skipped, and the next command asks it
  again; `draft`, `decide`, and `live` then exit 1. A command stops early
  only when the fault cannot be the items': after three failures in a row
  with the model server unreachable, or from a drafter or judge that has
  written nothing in the run yet. Do not let one item's failure end a run,
  and do not count a timeout or an HTTP rejection against the server:
  either can be the item's doing.

## Labeling

- Labeling is human ground truth, not an LLM's substitute judgment. Keep
  labels blinded, correctable, and resumable with edit history. Each group
  of questions must clearly identify its subject (turn, draft A/B, or
  comparison), with the relevant lesson visible. Test the rendered output
  and navigation, not only label serialization.
- Labeling and admission review each have their own controller, and one
  loop shows either's screens (`drive` in `judge/label_frontend.py`). It
  chooses the plain interface or the Go helper, stops the controller on
  an interrupt, and closes the helper. Do not write that loop a second
  time; a test that replaces a frontend replaces it in `label_frontend`.

## The Go labeling helper

The default labeling interface is a Go helper under `judge/label_ui/`; its
`go.mod` pins the Go version, and `mem judge label --plain` avoids it.
When you touch the Go labeling helper, also run:

```sh
cd src/agent_memory/judge/label_ui
gofmt -l .
go vet ./...
go test ./...
```
