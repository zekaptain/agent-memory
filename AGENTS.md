# Working on agent-memory

## Purpose and scope

This is a local, shared, auditable memory ledger for coding agents, built on
DuckDB 2.0 and served over Quack. Preserve the ability to answer: what was an
agent shown, where did it come from, and what did the agent actually use?
Provenance matters as much as retrieval quality.

Start with `README.md` for usage, `docs/architecture.md` for the design,
`docs/roadmap.md` for planned work, known defects, and open decisions, and
`GLOSSARY.md` for the project's terms.
Storage and capture (Phases 0 and 1), the relation graph, the judge
experiment, production admission, and the lifecycle, coverage, and
attribution hardening are implemented. Do not confuse planned harness
integrations with
shipped adapters: `adapters/` currently implements Droid and Vibe; the wider
harness roster includes MCP-only use and future hook work.

Read the relevant spec before changing a subsystem:

- `docs/phase-1-spec.md`: capture, spool, adapters, and MCP tools.
- `docs/judge-experiment-spec.md`: the judge design for new runs and its
  dated "As built" notes. The logs of exp1 to exp4 are in
  `docs/archive/judge-experiments.md`, and the definitions only stored
  runs still use are in `docs/archive/judge-earlier-versions.md`. Check
  the later amendments and current code before implementing.
- `docs/admission-spec.md`: production admission, fitted approver, human
  review, and calibration/held-out safeguards.
- `docs/logic-hardening-spec.md`: lifecycle, coverage, attribution,
  `mem audit check` preflight, and rollout policy.
- `docs/judge-how-to.md`: step-by-step experiment commands.
- `docs/memory-utility-evaluation-spec.md`: design only for the memory
  usefulness study; nothing is implemented from it.
- `docs/parametric-consolidation-spec.md`: a draft research proposal, not
  implemented functionality or authorization to add its proposed migration.
- `docs/agent-memory-literature-review.md` and `docs/paper/`: research
  grounding, separate from the runtime's feature contract.
- `docs/adr/`: decisions that are hard to reverse and would surprise a
  reader. Read the relevant one before undoing what it records.
- `docs/archive/`: superseded design text and experiment logs, kept for
  the record. Nothing there is a contract.

## Development commands

Use `uv`, Python 3.12+, Typer for CLI work, Rich for judge terminal output,
Ruff for lint/format, and Pyrefly for typing. Do not substitute another
package manager, CLI framework, or type checker. The default labeling
interface is a Go helper under `judge/label_ui/`; its `go.mod` pins the Go
version, and `mem judge label --plain` avoids it.

Run from the repository root:

```sh
uv sync
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run pyrefly check
uv run mem smoke
```

When you touch the Go labeling helper, also run:

```sh
cd src/agent_memory/judge/label_ui
gofmt -l .
go vet ./...
go test ./...
```

Use `uv run ruff format <changed-files>` to format only files you touched.
Ruff targets Python 3.12 with a 100-character line limit. Follow the existing
typed functions, dataclasses, and small subsystem modules.

Tests use real DuckDB databases and Quack servers, not just mocks.
Initialization installs the `fts` and `quack` extensions, so a fresh
environment needs network access. `tests/conftest.py` supplies temporary
homes, direct connections, live servers on free ports, and an unreachable
server configuration; reuse these and `tests/support.py`.
`mem smoke` uses a throwaway home; `mem smoke --live` checks an existing
server read-only. Never use the user's live ledger as a test fixture.

For code changes, run focused regression tests first, then the checks above
as appropriate. Report failures and skipped checks honestly. For docs-only
changes, verify paths, commands, and claims without starting live services.

## Code map

All runtime paths below are under `src/agent_memory/`.

| Location | Responsibility | Main tests |
| --- | --- | --- |
| `config.py`, `db.py`, `migrations/` | Configuration, migrations, authorization, Quack, export | `test_config`, `test_schema`, `test_migrations`, `test_attribution_migrations`, `test_authorize`, `test_server` |
| `lifecycle.py` | Shared append-only memory writes, correction proposals, bounded due work | `test_lifecycle` |
| `audit.py` | Read-only `mem audit check` diagnostics, legacy replacement reconstruction | `test_audit_check` |
| `endpoints.py` | The HTTP transport to model servers and the opt-in check for sending data off-machine | `test_endpoints` |
| `model_call.py` | A model call that must return a JSON object: its attempts, the three model servers, latency and cost over every attempt | `test_model_call` |
| `adapters/`, `capture/` | Harness input to records, redaction, blobs, idempotent writes, spool/replay, capture identity, the session and event rows every writer shares, observed coverage, answer attribution | `test_adapters`, `test_capture`, `test_writer`, `test_spool`, `test_capture_identity`, `test_ledger_session`, `test_observed_capture`, `test_answer_bindings`, `test_audit_persistence` |
| `mcp_server/` | Stdio tools, the recorded recall path and its query strategies, writes, reviews, citations | `test_mcp` |
| `graph/` | Relation metadata, traversal SQL, graph recall strategies, entity merges, entity resolution | `test_graph` |
| `judge/` | Import, provenance, candidates, projection, draft, decide, label (including the Go helper in `label_ui/`), report | `test_judge`, `test_judge_pipeline`, `test_judge_verdict`, `test_judge_unanswered`, `test_judge_core_regressions`, `test_judge_store`, `test_judge_admission`, `test_label_ui` |
| `admission/` | Production lesson admission, fitting and blinded review | `test_admission`, `test_admission_fit` |
| `cli.py`, `smoke.py` | Typer entry point and end-to-end check | `test_cli`, `test_cli_capture` |

Test names in the table refer to `tests/<name>.py`. The Go helper's own
tests sit beside its sources in `judge/label_ui/`.
Entry points are `mem` (`agent_memory.cli:main`) and `mem-mcp`
(`agent_memory.mcp_server:main`).

## Ledger and capture invariants

- Ledger tables are append-only. Corrections use superseding memories or
  review actions, never an in-place update or delete. Current state is
  derived through views such as `memory_status` and `active_memories`.
- Add schema changes as new `NNNN_name.sql` files in `migrations/`.
  Applied migrations are checksummed and immutable. The root
  `example-memory-db-schema.sql` is inspiration, not the runtime schema.
- `mem serve` owns the database file during normal operation. Clients use
  `MemoryClient` over Quack. Direct admin access and migrations require
  stopping that server; do not bypass the file lock.
- Quack in DuckDB 2.0.0 does not forward bound parameters. Preserve the
  client's parameterized local staging tables and streamed inserts.
  `VARIANT` values are encoded through JSON. Insert parents before children
  in separate statements where same-table foreign keys require it.
- Keep client writable-table lists and the `mem_authorize` migration policy
  consistent. Do not weaken the policy to make a query pass. Text that must
  enter SQL uses the existing `sql_literal` helper, never raw interpolation.
- Redact before any persistent write, including spool, blobs, metadata,
  and projections; redact spooled data again at replay. Keep tokens out of
  errors and traceback locals. Preserve loopback-only serving, private file
  permissions, and the server's external-access/configuration lockdown.
- Capture IDs are deterministic so transcript reimports, overlapping hooks,
  retries, and spool replay do not duplicate events. A server outage should
  queue redacted records rather than break the agent's hook. Delivery never
  drops a record: one the ledger rejects is kept under `spool/failed/` with
  a `capture_error` event while the rest of its batch is written. `mem
  event` runs as a hook, so it prints nothing on stdout and never exits 2.
  Add regression coverage for replay, partial failures, and parent/result
  linkage.
- Session and event rows come from `capture/session.py`, whoever writes
  them: the capture writer, the MCP tools, `mem graph`, and admission. A
  new writer opens a `LedgerSession` and builds its events with
  `event_row`; it does not assemble either row itself.
- Tool-derived memories start quarantined. Every recall reads active
  memories and records retrieval provenance, including misses, whatever
  its strategy: a query, a walk of relations from an entity, or the
  listing of every relation that `mem graph entities` and `resolve` read.
  Add a new way of choosing memories as a strategy for `run_recall`, never
  as a read beside it. Preserve `retrieval_items`, citations, and
  `why_answer` coverage.
- Relations are ordinary `fact` memories with normalized triple keys and
  `metadata.graph`. Entity merges supersede or retire relations and preserve
  sources and aliases. Do not introduce separate graph tables casually.

## Judge workflow and lessons from previous runs

The experiment must stay separate from production. Never remove
`ensure_experiment_home`. Experiment commands never write lessons into the
memory ledger; lessons stay in run files. Admission lives in `admission/`:
it reads human v4 experiment labels and writes production lessons through
the lifecycle, quarantined unless human-approved or enabled by the held-out
auto gate. Nothing under `judge/` imports admission. Verdicts use agent
`judge` and may be recorded as `decision` events without copying state or
lesson text into them. Admission writes as agent `admission`, never as
`judge`.

`--home` and `--port` are root options, before the subcommand. A separate
home does not automatically select a different port. Use the same explicit
home and port for the experiment server and its clients, for example:

```sh
# Foreground server, in one terminal after initializing this home.
uv run mem --home "$HOME/.agent-memory-judge" --port 9495 serve
# Client, in another terminal.
uv run mem --home "$HOME/.agent-memory-judge" --port 9495 judge status --run exp3
```

`serve` intentionally stays running. `init` is idempotent, not a reset;
do not delete existing homes or runs to resolve a workflow mistake.

- Distinguish a human request, parent-agent instructions, and harness
  notifications. A transcript's `user` role does not prove a human wrote
  it. Read full blob-backed messages when needed for provenance.
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
- Read and write a run's files through `RunStore` only. The rules that
  keep them whole are at the top of `judge/store.py`: one command at a
  time adds rows to a file, so a second `draft`, or a `decide` for a
  backend already being judged, is refused at once; a label save adds one
  label to what is on disk and is refused if another session saved that
  item first. Admission keeps its files through the same code, with its
  stricter rules in `admission/store.py`.
- Labeling is human ground truth, not an LLM's substitute judgment. Keep
  labels blinded, correctable, and resumable with edit history. Each group
  of questions must clearly identify its subject (turn, draft A/B, or
  comparison), with the relevant lesson visible. Test the rendered output
  and navigation, not only label serialization.
- Model calls must handle timeouts, empty output, and token exhaustion
  without losing finished items. A call that must return a JSON object
  goes through `model_call.py`, which owns its attempts: the judge, the
  drafter, and `mem graph resolve` hand it a model server and say what to
  ask. Do not build a request or a further attempt beside it, and never
  choose an attempt from the words of an error. Preserve the bound on
  attempts, the ability to run a failed item again, and latency/cost
  accounting across attempts. Repeating a seeded request with the same
  insufficient budget is not a fix.
- A draft or verdict that fails is skipped, and the next command asks it
  again; `draft`, `decide`, and `live` then exit 1. A command stops early
  only when the fault cannot be the items': after three failures in a row
  with the model server unreachable, or from a drafter or backend that has
  written nothing in the run yet. Do not let one item's failure end a run,
  and do not count a timeout or an HTTP rejection against the server:
  either can be the item's doing.
- Prefer local backends. Hosted judge/drafter calls require explicit user
  permission and `AGENT_MEMORY_JUDGE_ALLOW_REMOTE=1`; redaction does not make
  session text public. Mock model HTTP in regression tests rather than
  making paid or private-data-bearing calls.

## Change discipline and research

Inspect the working tree before editing and leave unrelated user changes
alone. Reproduce reported bugs and add regression tests through the affected
CLI, MCP, capture, or judge path. When adding MCP tools, update registration
assertions and usage docs too; graph work previously exposed this gap.

Update the relevant spec's implemented/as-built notes when behavior changes.
Use the terms in `GLOSSARY.md`, and update it when a term changes. Add a
record under `docs/adr/` only for a decision that is hard to reverse,
surprising without context, and the result of a real trade-off. A defect
you find and do not fix goes into `docs/roadmap.md`.
Existing commit messages use Conventional Commits, for example
`fix(judge): ...` and `test(graph): ...`; explain the failure and why the
change fixes it in the body when useful.

For research edits, verify claims and bibliography metadata against primary
sources. Separate peer-reviewed results, preprints, and vendor claims;
state what remains unverified rather than inheriting an earlier session's
confidence. If editing the paper, rebuild `docs/paper/paper.pdf` from the
LaTeX/BibTeX sources and check citations and layout warnings. Do not commit
private transcripts, experiment homes, tokens, or model outputs as fixtures;
use small synthetic/redacted examples.
