# agent-memory

A local, shared, auditable memory database for AI coding agents, built on
DuckDB 2.0 and served over Quack. Every agent harness writes to one
append-only ledger, so you can ask why an agent gave an answer: what it was
shown, where that came from, and which agent recorded it.

Status: Phases 0 and 1 are done: the storage layer, the capture path
(`mem event`, `mem ingest`, the spool) and the in-session tools (`mem-mcp`).
Installing hooks into each harness is Phase 2. [PLAN.md](PLAN.md) has the
design, the DuckDB 2.0 behaviors it relies on, and the roadmap;
[docs/phase-1-spec.md](docs/phase-1-spec.md) describes the capture pipeline
and the MCP tools.

Supported harnesses: Factory Droid, Mistral Vibe, and oh-my-pi (full hooks),
Amp (plugin API), and JetBrains Junie (MCP only, recorded as partial capture).

## Requirements

- macOS or Linux
- [uv](https://docs.astral.sh/uv/)
- Python 3.12 or newer (uv installs it if needed)
- Network access the first time you run `mem init`, to download the DuckDB
  `fts` and `quack` extensions

## Quick start

```sh
uv sync
uv run mem init      # creates ~/.agent-memory, a token, and the database
uv run mem serve     # leave this running
```

In another shell:

```sh
uv run mem smoke --live
uv run mem sql "SELECT agent_id, capture_tier FROM agents"
```

`mem smoke` with no flags builds a throwaway database, serves it on a free
port, and runs the full end-to-end check, including writes. It never touches
`~/.agent-memory`.

## Commands

| Command | What it does |
|---------|--------------|
| `mem init` | Create the data directory, token, and database, and apply migrations. Safe to re-run. |
| `mem serve` | Serve the database over Quack on localhost until interrupted. Writes a Parquet snapshot every 24 hours (`--export-interval-hours`, 0 to disable) and keeps the newest 14 (`--export-keep`). A failed export is logged and retried, not fatal. |
| `mem sql QUERY` | Run a query through the server. `-` reads the query from stdin. `--json` prints JSON. |
| `mem sql --direct QUERY` | Open the database file instead of using the server (server must be stopped). Read-only unless you add `--write`. |
| `mem migrate` | Apply pending migrations (server must be stopped). `--status` lists them. |
| `mem export` | Write every table to Parquet under `export/<timestamp>/` and point `export/latest` at it. `--direct` works without the server; `--keep N` prunes older snapshots. |
| `mem smoke` | End-to-end check of the server, policy, and client. `--live` checks the running server, read-only. |
| `mem event --harness droid\|vibe` | Record one hook payload read as JSON from stdin. Exits 0 once the records are in the ledger or the spool. `--keep-raw` keeps the whole payload; `--record-pre-tool` also records pre-tool hooks. |
| `mem ingest --harness droid --transcript PATH` | Import a Droid session transcript (`~/.factory/sessions/<cwd-slug>/<id>.jsonl`). Re-running is a no-op; `--since-key <message id>` skips what a hook already knows about; `--include-thinking` also records thinking blocks. |
| `mem ingest --spool` | Replay records that were spooled while the server was down. `--purge-done DAYS` deletes old replayed files. |
| `mem spool status` | Pending spool files and lines per harness. |
| `mem-mcp` | The stdio MCP server (`recall`, `remember`, `cite`, `review`, `relate`, `graph`, `status`). Needs `AGENT_MEMORY_AGENT`; see below. |
| `mem graph entities` | Every entity in the active relations of a scope, with type and mention count. |
| `mem graph show ENTITY` | The relations within `--hops` of an entity (logged as a `graph_hop_v1` retrieval). |
| `mem graph resolve` | Ask a local Ollama model which entity names are the same thing; `--apply` records the merges as supersedes history. |

Global options: `--home` (default `$AGENT_MEMORY_HOME`, then `~/.agent-memory`)
and `--port` (default `$AGENT_MEMORY_PORT`, then 9494). `AGENT_MEMORY_HOST`
may be `localhost`, `127.0.0.1`, or `::1`.

## Capturing sessions

Hooks pipe their stdin to `mem event`, and a turn-end hook imports the
transcript, because no Droid or Vibe hook carries the assistant's reply:

```sh
# in a hook script
mem event --harness droid
# at turn end (Droid Stop hook); $TRANSCRIPT_PATH comes from the hook payload
mem ingest --harness droid --transcript "$TRANSCRIPT_PATH"
```

Everything is redacted before it is written anywhere (API keys, tokens, JWTs,
private keys, URL credentials, `password=...` style assignments, and the
server token itself). Content over 16 KiB goes to `blobs/` and the event keeps
a 4 KiB prefix plus the hash. When the server is down the records go to
`spool/` and the command still exits 0; `mem serve` replays the spool when it
starts, or run `mem ingest --spool`.

Import an existing Droid session to see real data:

```sh
uv run mem ingest --harness droid --transcript ~/.factory/sessions/<cwd-slug>/<id>.jsonl
uv run mem sql "SELECT * FROM session_summary"
```

## In-session tools (mem-mcp)

Register `mem-mcp` as a stdio MCP server in each harness, with the harness
named in the environment:

```json
{
  "mcpServers": {
    "agent-memory": {
      "command": "mem-mcp",
      "env": {"AGENT_MEMORY_AGENT": "factory-droid", "AGENT_MEMORY_PROJECT": "my-project"}
    }
  }
}
```

`AGENT_MEMORY_AGENT` is one of `factory-droid`, `mistral-vibe`, `oh-my-pi`,
`amp`, `jetbrains-junie`. `AGENT_MEMORY_PROJECT` defaults to the working
directory's name and sets the default recall scope. `AGENT_MEMORY_SESSION`
(the harness session id) joins the tools to the session the hooks record;
without it the server opens its own session, marked `partial`.

Tools: `recall(query, scopes, limit, types)` ranks active memories with FTS
and logs the retrieval; `remember(content, type, ...)` stores a memory (tool
derived ones start quarantined; identical content is confirmed, not
duplicated); `cite(retrieval_id, memory_ids)` records what an answer used;
`review(action, memory_id, reason)` retires, approves, quarantines, confirms,
or flags a contradiction; `status()` reports reachability and the spool. When
the server is down, `recall` returns `degraded: true` and writes are queued to
the spool.

## Relations and the graph

`relate(subject, predicate, object, ...)` records one typed edge, for example
`("payments-api", "owned by", "Dana")`. It is stored as an ordinary `fact`
memory: the key is the normalized triple (`payments-api|owned by|dana`), the
display names, optional entity types, and merged aliases live in
`metadata.graph`, and `memory_sources` points at the events it came from. The
same triple in the same scope is confirmed, not duplicated, and tool-derived
edges start quarantined like any other tool memory. There are no graph tables;
the graph is rebuilt from metadata at query time.

`graph(entity, hops)` walks the active edges outward from an entity with a
recursive CTE (1 to 3 hops, undirected, case- and punctuation-insensitive) and
returns them as items and as `(subject) --[predicate]--> (object)` triples, plus
a `retrieval_id` for `cite`. Every walk is logged in `retrievals` with
strategy `graph_hop_v1`, including misses, so `why_answer` covers graph-backed
answers the same way it covers text recall.

Entity resolution is append-only. `mem graph resolve --scope <project>` lists
the entities in a scope, asks a local Ollama model (default `qwen3:8b`, set
`--model` and `--host`) to cluster surface forms per entity type, and prints
the proposed merges. With `--apply`, each merge supersedes every relation that
mentions the alias with the same relation under the canonical name (the alias
is kept in the new row's metadata and its sources are copied), or retires the
alias relation when the canonical one already exists. Names the model drops stay
as they are; nothing is rewritten in place, and a bad merge is undone by
superseding again.

```sh
uv run mem graph entities --scope my-project
uv run mem graph show payments-api --hops 2 --scope my-project
uv run mem graph resolve --scope my-project            # proposal only
uv run mem graph resolve --scope my-project --apply
```

## Asking why

Every answer event can be traced back through the views:

```sh
# Recent sessions, with capture tier and counts
uv run mem sql "SELECT * FROM session_summary ORDER BY started_at DESC LIMIT 10"

# What an answer was based on, and where each memory came from
uv run mem sql "SELECT via, rank, memory_content, status_at_recall, memory_status_now,
                       source_kind, source_agent_id, source_content
                FROM why_answer WHERE answer_event_id = '<event id>'"

# Everything that happened to memories, in order
uv run mem sql "SELECT * FROM memory_history ORDER BY ts DESC LIMIT 20"
```

For notebooks or long analysis, query the Parquet export. It needs neither the
server nor the token:

```sh
duckdb -c "FROM '$HOME/.agent-memory/export/latest/events.parquet' LIMIT 5"
```

## Data directory

```
~/.agent-memory/          mode 0700
  memory.duckdb           the database; only mem serve opens it
  token                   Quack bearer token, mode 0600
  export/<timestamp>/     Parquet snapshots, one file per table
  export/latest           symlink to the newest snapshot
  spool/<agent>/*.jsonl   records waiting for the server; done/ and failed/ after replay
  blobs/<xx>/<sha256>.txt event content over 16 KiB, content-addressed
```

## Run the server at login (macOS)

```sh
uv tool install --editable .
mem init
sed -e "s|__MEM__|$(command -v mem)|g" -e "s|__HOME__|$HOME|g" \
    scripts/com.agent-memory.serve.plist > ~/Library/LaunchAgents/com.agent-memory.serve.plist
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.agent-memory.serve.plist
mem smoke --live
```

Logs go to `~/.agent-memory/serve.log`. To stop the service, for example
before `mem migrate`:

```sh
launchctl bootout gui/$(id -u)/com.agent-memory.serve
```

## Security

- The server only listens on localhost and requires the token. Traffic is
  plain HTTP on the loopback interface; the token is never written to settings
  or error messages.
- Clients may only read and insert into ledger tables. The server's policy
  rejects `UPDATE`, `DELETE`, DDL, settings changes, file access, comments, and
  multiple statements in one text.
- Every table is append-only, enforced by triggers. Corrections are new rows: a
  superseding memory or a `retire` action.
- Memories derived from tool output start quarantined until approved.
- Secrets are redacted in the capture pipeline before anything is written to
  the spool or the ledger. `events.redactions` counts what was removed.
- The recall query is the one piece of agent text that enters SQL; it is
  literal-quoted and the policy strips literals before classifying the
  statement, so it cannot change what the statement does.

## Development

### Labeling judge experiments

`uv run mem --home /path/to/experiment judge label --run RUN` opens a
full-screen, blinded review. Use arrows and Enter to answer, Page Up/Down
to scroll evidence, `b` to go back, `p` for the full projection, and `?`
for the guide. Notes are literal text; Enter accepts and Esc cancels.
Review each item's summary before saving. Skip and quit never save the
current item's unfinished answers.

The interactive interface needs Go 1.26.4 or newer (pinned in
`src/agent_memory/judge/label_ui/go.mod`). Its pinned dependencies may
need network access on the first build. Sources ship in the Python
wheel and source distribution. The first launch with work to label
builds a private cached helper; unchanged launches reuse it. The cache
is under `~/Library/Caches/agent-memory/label-ui` on macOS and
`${XDG_CACHE_HOME:-~/.cache}/agent-memory/label-ui` on Linux. Source,
dependency, protocol, platform, or Go toolchain changes trigger a new
build. No experiment text is passed in command arguments or logs.

Use `judge label --run RUN --plain` for numbered, linear prompts with
no Go dependency or cursor controls. Pipes and `TERM=dumb` select plain
mode automatically. Enter a choice number or a navigation command
(`back`, `evidence`, `help`, `redo`, `skip`, `quit`). In the note input,
letters are text and `/cancel` cancels the edit. Both interfaces share
the same Python branching, blinding, resume, backfill, review, and
label-history rules. Other judge commands still use Rich.

Never label against the production home. See
`docs/judge-experiment-spec.md` for experiment setup.

### Checks

```sh
uv sync
uv run pytest            # unit, schema, policy, and end-to-end Quack tests
uv run ruff check .      # lint
uv run ruff format .     # format
uv run pyrefly check     # type check
uv run mem smoke         # end-to-end check against a throwaway server
```

For the Go labeling helper, run `gofmt -l .`, `go test ./...`,
`go vet ./...`, and `go build -mod=readonly -o /tmp/agent-memory-label-ui .`
from `src/agent_memory/judge/label_ui`.

Schema changes go in a new file under `src/agent_memory/migrations/`
(`NNNN_name.sql`). Applied migrations are checksummed and must not be edited.
