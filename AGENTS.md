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
integrations with shipped adapters: `adapters/` implements Droid only, the
one harness checked against real payloads; the wider harness roster
includes MCP-only use and future hook work.

## Guides for each part

This file holds what binds every session, including one that only runs
commands or edits documents. What binds a change to one part of the
repository is in an `AGENTS.md` in that part's directory. A harness may
load such a guide by itself when it opens a file there; do not count on
that. Read the guide yourself in these cases:

| Before you | Read |
| --- | --- |
| change code under `src/agent_memory/` | `src/agent_memory/AGENTS.md`: schema, export, Quack, redaction, the rows every writer shares, approval, recall, relations, model calls |
| change `adapters/` or `hooks.py`, or what a Droid hook records | `src/agent_memory/adapters/AGENTS.md` |
| change `capture/`, `mem event`, `mem ingest`, or the spool | `src/agent_memory/capture/AGENTS.md` |
| add an MCP tool or change a tool's inputs | `src/agent_memory/mcp_server/AGENTS.md` |
| change `judge/` or its Go labeling helper, or run a `mem judge` command | `src/agent_memory/judge/AGENTS.md` |
| change `admission/`, or run a `mem admit` command | `src/agent_memory/admission/AGENTS.md` and the judge guide |
| add or change a test | `tests/AGENTS.md` and the guide of the code under test |
| edit a document under `docs/`, or act on what one proposes | `docs/AGENTS.md`: what each document is, and which are drafts or records |

A command in `cli.py` follows the guide of its subsystem. Read the relevant
spec before changing a subsystem; each guide names it.

A rule is written in one guide: here if it binds every session, otherwise
in the directory whose code it binds. Change it there, and do not repeat it
in a second guide.

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

The Go labeling helper has checks of its own, listed in
`src/agent_memory/judge/AGENTS.md`. CI (`.github/workflows/ci.yml`) runs
both lists on every pull request and on every push to `main`: Ruff and
Pyrefly once, the tests and `mem smoke` on Linux and on macOS with the
Python that `.python-version` names, and the Go checks once. When a command
in either list changes, change it there too.

Use `uv run ruff format <changed-files>` to format only files you touched.
Ruff targets Python 3.12 with a 100-character line limit. Follow the existing
typed functions, dataclasses, and small subsystem modules.

Pyrefly reports no errors and ten warnings. Each of the ten is about a
`str()` or an `int()` around a value from a harness payload, a stored row,
or a tool call, where the annotation says more than the data can promise.
Leave those conversions, and treat an eleventh warning as new.

Initialization installs the `fts` and `quack` extensions, so a fresh
environment needs network access.

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
| `review.py` | `mem review`: the memories awaiting approval, and a human's approval or retirement of one | `test_review` |
| `hooks.py` | `mem hooks`: mem's capture command in Droid's hooks file, written, checked, and taken out, with every other hook left as it is | `test_hooks` |
| `endpoints.py` | The HTTP transport to model servers and the opt-in check for sending data off-machine | `test_endpoints` |
| `model_call.py` | A model call that must return a JSON object: its attempts, the three model servers, latency and cost over every attempt | `test_model_call` |
| `adapters/`, `capture/` | Harness input to records, redaction, blobs, idempotent writes, spool/replay, capture identity, the session and event rows every writer shares, observed coverage, attribution | `test_adapters`, `test_capture`, `test_writer`, `test_spool`, `test_capture_identity`, `test_ledger_session`, `test_observed_capture`, `test_answer_bindings`, `test_audit_persistence` |
| `mcp_server/` | Stdio tools, the recorded recall path and its query strategies, writes, review actions, citations | `test_mcp` |
| `graph/` | Relation metadata, traversal SQL, graph recall strategies, entity merges, entity resolution | `test_graph` |
| `judge/` | Import, callers and authors, candidates, projection, draft, decide, label (including the Go helper in `label_ui/`), report | `test_judge`, `test_judge_pipeline`, `test_judge_verdict`, `test_judge_unanswered`, `test_judge_core_regressions`, `test_judge_store`, `test_judge_admission`, `test_label_ui` |
| `admission/` | Production lesson admission, fitting, and admission review | `test_admission`, `test_admission_fit` |
| `cli.py`, `smoke.py` | Typer entry point and end-to-end check | `test_cli`, `test_cli_capture` |

Test names in the table refer to `tests/<name>.py`. The Go helper's own
tests sit beside its sources in `judge/label_ui/`.
Entry points are `mem` (`agent_memory.cli:main`) and `mem-mcp`
(`agent_memory.mcp_server:main`).

## Rules for every session

- Ledger tables are append-only. Corrections use superseding memories or
  review actions, never an in-place update or delete. Current status is
  derived through views such as `memory_status` and `active_memories`.
- `mem serve` owns the database file during normal operation. Clients use
  `MemoryClient` over Quack. Direct admin access and migrations require
  stopping that server; do not bypass the file lock.
- Never use the user's live ledger as a test fixture. `mem smoke` uses a
  throwaway home; `mem smoke --live` checks an existing server read-only.
- `init` is idempotent, not a reset; do not delete existing homes or runs
  to resolve a workflow mistake.
- Prefer local model servers. Hosted judge/drafter calls require explicit
  user permission and `AGENT_MEMORY_JUDGE_ALLOW_REMOTE=1`; redaction does
  not make session text public.
- Do not commit private transcripts, experiment homes, tokens, or model
  outputs as fixtures; use small synthetic/redacted examples.

## Change discipline

Inspect the working tree before editing and leave unrelated user changes
alone. Reproduce reported bugs and add regression tests through the affected
CLI, MCP, capture, or judge path.

Update the relevant spec's implemented/as-built notes when behavior changes.
Use the terms in `GLOSSARY.md`, and update it when a term changes. Add a
record under `docs/adr/` only for a decision that is hard to reverse,
surprising without context, and the result of a real trade-off, and read
the relevant record before undoing what it records. A defect you find and
do not fix goes into `docs/roadmap.md`.
Existing commit messages use Conventional Commits, for example
`fix(judge): ...` and `test(graph): ...`; explain the failure and why the
change fixes it in the body when useful.
