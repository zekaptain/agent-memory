# Working on agent-memory

## Purpose and scope

This is a local, shared, auditable memory ledger for coding agents, built on
DuckDB 2.0 and served over Quack. Preserve the ability to answer: what was an
agent shown, where did it come from, and what did the agent actually use?
Provenance matters as much as retrieval quality.

Start with `README.md` for usage and `PLAN.md` for architecture and roadmap.
Storage and capture (Phases 0 and 1), the relation graph, and the judge
experiment are implemented. Do not confuse planned harness integrations with
shipped adapters: `adapters/` currently implements Droid and Vibe; the wider
harness roster includes MCP-only use and future hook work.

Read the relevant spec before changing a subsystem:

- `docs/phase-1-spec.md`: capture, spool, adapters, and MCP tools.
- `docs/judge-experiment-spec.md`: judge design, historical runs, exp3, and
  "As built" notes. Earlier sections describe older versions; check the
  later amendments and current code before implementing.
- `docs/parametric-consolidation-spec.md`: a draft research proposal, not
  implemented functionality or authorization to add its proposed migration.
- `docs/agent-memory-literature-review.md` and `docs/paper/`: research
  grounding, separate from the runtime's feature contract.

## Development commands

Use `uv`, Python 3.12+, Typer for CLI work, Rich for judge terminal output,
Ruff for lint/format, and Pyrefly for typing. Do not substitute another
package manager, CLI framework, or type checker.

Run from the repository root:

```sh
uv sync
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run pyrefly check
uv run mem smoke
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
| `config.py`, `db.py`, `migrations/` | Configuration, migrations, authorization, Quack, export | `test_config`, `test_schema`, `test_migrations`, `test_authorize`, `test_server` |
| `adapters/`, `capture/` | Harness input to records, redaction, blobs, idempotent writes, spool/replay | `test_adapters`, `test_capture`, `test_writer`, `test_spool` |
| `mcp_server/` | Stdio tools, recall ranking, writes, reviews, citations | `test_mcp` |
| `graph/` | Relation metadata, traversal SQL, entity resolution | `test_graph` |
| `judge/` | Import, provenance, candidates, projection, draft, decide, label, report | `test_judge`, `test_judge_pipeline` |
| `cli.py`, `smoke.py` | Typer entry point and end-to-end check | `test_cli`, `test_cli_capture` |

Test names in the table refer to `tests/<name>.py`.
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
  queue redacted records rather than break the agent's hook. Add regression
  coverage for replay, partial failures, and parent/result linkage.
- Tool-derived memories start quarantined. Recall and graph traversal use
  active memories and record retrieval provenance, including misses.
  Preserve `retrieval_items`, citations, and `why_answer` coverage.
- Relations are ordinary `fact` memories with normalized triple keys and
  `metadata.graph`. Entity merges supersede or retire relations and preserve
  sources and aliases. Do not introduce separate graph tables casually.

## Judge workflow and lessons from previous runs

The experiment must stay separate from production. Never remove
`ensure_experiment_home` or write drafted lessons into the memory ledger:
lessons stay in run files; verdicts may be recorded as `decision` events
without copying state text into them.

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
- Stage A evaluates the turn once; Stage B evaluates each draft. Current
  write gates are grounded, faithful, generalizable, and not already
  enforced. A failed tool call can be intentional; a useful lesson need
  not describe a failure. Harness rules, project docs, and tooling may
  already enforce a lesson.
- Preserve seeded sampling, calibration/held-out separation, versioned
  projections/questions/drafters, hashes, and resumable per-item/per-drafter
  outputs. Older question-set runs remain read-only for draft/decide/label,
  but their reports must still reproduce the historical results.
- Finish drafting before a long decide run. Do not change its drafts
  underneath it. Labeling and deciding can run independently, but freeze
  thresholds only after the intended calibration labels are complete.
- Labeling is human ground truth, not an LLM's substitute judgment. Keep
  labels blinded, correctable, and resumable with edit history. Each group
  of questions must clearly identify its subject (turn, draft A/B, or
  comparison), with the relevant lesson visible. Test the rendered output
  and navigation, not only label serialization.
- Model calls must handle timeouts, empty output, and token exhaustion
  without losing finished items. Preserve bounded retries, retryability of
  failed items, and latency/cost accounting across attempts. Repeating a
  seeded request with the same insufficient budget is not a fix.
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
