# Working on the runtime

This guide applies to every change under `src/agent_memory/`, on top of
the root `AGENTS.md`. `adapters/`, `capture/`, `mcp_server/`, `judge/`,
and `admission/` each add a guide of their own. Paths here are relative
to this directory unless they start with `docs/` or `tests/`.

Read `docs/logic-hardening-spec.md` before changing lifecycle, coverage,
attribution, the `mem audit check` preflight, or rollout policy.

## Storage

- Add schema changes as new `NNNN_name.sql` files in `migrations/`.
  Applied migrations are checksummed and immutable. The root
  `example-memory-db-schema.sql` is inspiration, not the runtime schema.
- A Parquet export holds the ledger as it was at one moment. Copy tables
  only through `export_snapshot` (one transaction on the database file) or
  `MemoryClient.export` (a client's statements each see the server as it
  is then, so it compares its copies with one count of every table and
  copies again). Both rely on every table being append-only.
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

## Writers and readers

- Session and event rows come from `capture/session.py`, whoever writes
  them: the capture writer, the MCP tools, `mem graph`, `mem review`, and
  admission. A new writer opens a `LedgerSession` and builds its events
  with `event_row`; it does not assemble either row itself.
- A session's row holds what the first record received of the session
  knew, and is never changed: a hook's row has no title, no model, and
  the time of that hook. Read a session's start, model, project, title,
  caller, and harness version from `session_summary`, which completes
  them from the session's observations and events, not from the row.
- Only the `human` agent approves a memory. An agent that could approve
  what it wrote from tool output would hold the key to its own
  quarantine. The check (`lifecycle.check_asker`) is made where an
  approval is asked for: the `review` tool and spool replay. Do not move
  it into `LifecyclePlan.review`, which also writes the approval an
  entity merge carries over and admission's approvals. `mem-mcp` refuses
  to run as `human`, and `mem review approve` takes its answer from a
  terminal only. Do not add a route by which an agent's session approves.
- Tool-derived memories start quarantined. Every recall reads active
  memories and records a retrieval, a miss included, whatever its
  strategy: a query, a walk of relations from an entity, or the listing
  of every relation that `mem graph entities` and `resolve` read.
  Add a new way of choosing memories as a strategy for `run_recall`, never
  as a read beside it. Preserve `retrieval_items`, citations, and
  `why_answer` coverage.
- Relations are ordinary `fact` memories with normalized triple keys and
  `metadata.graph`. Entity merges supersede or retire relations and preserve
  sources and aliases. Do not introduce separate graph tables casually.

## Commands and model calls

- `--home` and `--port` are root options, before the subcommand. Every
  command resolves them through `config.command_config`; the judge and
  admission commands add their own checks to what it returns.
- Model calls must handle timeouts, empty output, and token exhaustion
  without losing finished items. A call that must return a JSON object
  goes through `model_call.py`, which owns its attempts: the judge, the
  drafter, and `mem graph resolve` hand it a model server and say what to
  ask. Do not build a request or a further attempt beside it, and never
  choose an attempt from the words of an error. Preserve the bound on
  attempts, the ability to run a failed item again, and latency/cost
  accounting across attempts. Repeating a seeded request with the same
  insufficient budget is not a fix.
