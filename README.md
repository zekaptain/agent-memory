# agent-memory

A local memory database shared by your AI coding agents. It uses DuckDB 2.0
and serves local clients over Quack.

The ledger keeps a history instead of overwriting records. Use it to find:

- What an agent was shown.
- Where that information came from.
- Which memories the agent used in its answer.

Storage, capture commands, MCP tools, the relation graph, and judge
experiments are implemented. Automatic hook installation is still planned.

Start with the [quick start](#quick-start), then
[import a session](#import-a-droid-session) or
[connect your agent through MCP](#connect-your-agent-through-mcp).
For experiments, use the [judge how-to](docs/judge-how-to.md).

## Requirements

- macOS or Linux
- [uv](https://docs.astral.sh/uv/)
- Python 3.12 or newer (uv installs it if needed)
- Network access the first time you run `mem init`, to download the DuckDB
  `fts` and `quack` extensions

## Quick start

Run these commands from the repository root.

### 1. Install dependencies and initialize the database

```sh
uv sync
uv run mem init
```

This creates `~/.agent-memory`, a private token, and the database.
`init` is safe to rerun with the server stopped. It applies pending
migrations; it does not reset your data.

### 2. Start the server

```sh
uv run mem serve
```

Leave this terminal open. The server runs until you stop it with Ctrl-C.

### 3. Check the connection

Open another terminal at the repository root:

```sh
uv run mem smoke --live
uv run mem sql "SELECT agent_id, capture_tier FROM agents"
```

The smoke check is read-only. The query lists the registered agents.

To run a full end-to-end check without using your database, run
`uv run mem smoke` without `--live`. It uses a throwaway home and a free port.

### Optional: install commands for use outside this repository

Hook scripts, MCP clients, and the login service need `mem` and `mem-mcp`
on their `PATH`:

```sh
uv tool install --editable .
```

After installation, you can use `mem` instead of `uv run mem`.

## Choose a home and port

Defaults work for normal use. To change them:

| Setting | CLI option | Environment variable | Default |
|---|---|---|---|
| Data directory | `--home PATH` | `AGENT_MEMORY_HOME` | `~/.agent-memory` |
| Server port | `--port N` | `AGENT_MEMORY_PORT` | `9494` |
| Server host | None | `AGENT_MEMORY_HOST` | `localhost` |

CLI options take precedence over environment variables. Put `--home` and
`--port` before the subcommand:

```sh
uv run mem --home "$HOME/.agent-memory" --port 9494 smoke --live
```

Use the same settings for the server and its clients. A different home
does not select a different port. The host must be `localhost`, `127.0.0.1`,
or `::1`.

Add `--help` to any command to see its options, for example
`uv run mem ingest --help`.

## Capture sessions

| Harness | Capture available now |
|---|---|
| Factory Droid | Hook payloads and transcript import. |
| Mistral Vibe | Hook payloads. Transcript import is not implemented yet. |
| oh-my-pi and Amp | MCP tools only; capture adapters are planned. |
| JetBrains Junie | MCP tools only, recorded as partial capture. |

### Import a Droid session

Keep the server running. Replace `PROJECT_SLUG` and `SESSION_ID` with an
existing directory and file under `~/.factory/sessions`:

```sh
uv run mem ingest --harness droid \
  --transcript "$HOME/.factory/sessions/PROJECT_SLUG/SESSION_ID.jsonl"
uv run mem sql "SELECT * FROM session_summary"
```

- Reimporting the same transcript does not duplicate events.
- `--since-key MESSAGE_ID` skips entries already known to a hook.
- `--include-thinking` also imports thinking blocks.

### Wire capture into hooks

Hook installation is not automatic. After installing the commands on your
`PATH`, use these patterns in your Droid hook scripts:

```sh
# Read the hook's JSON payload from stdin.
mem event --harness droid

# In a Stop hook, import the assistant's reply from the transcript.
# Read TRANSCRIPT_PATH from the hook payload before calling this.
mem ingest --harness droid --transcript "$TRANSCRIPT_PATH"
```

Neither Droid nor Vibe hooks carry the assistant's reply. Droid transcript
import fills that gap; Vibe transcript support is still pending.

For `mem event`, use `--keep-raw` to retain the full payload or
`--record-pre-tool` to record pre-tool hooks as well.
See the [capture spec](docs/phase-1-spec.md) for payload and adapter details.

### Recover queued records

If the server is down, capture saves records under `spool/` instead.
The command exits 0 once the records are saved in the ledger or the spool.

Starting `mem serve` replays pending records. You can also check and replay
them yourself:

```sh
uv run mem spool status
uv run mem ingest --spool
```

Add `--purge-done DAYS` to the replay command to remove old replayed files.

Before saving, capture redacts known secret patterns, including API keys,
tokens, JWTs, private keys, URL credentials, and password assignments.
Content over 16 KiB goes to `blobs/`; the event keeps a 4 KiB prefix and hash.

## Connect your agent through MCP

Install `mem-mcp` on the harness's `PATH` using
`uv tool install --editable .`, then add a stdio server to its MCP
configuration. For example:

```json
{
  "mcpServers": {
    "agent-memory": {
      "command": "mem-mcp",
      "env": {
        "AGENT_MEMORY_AGENT": "factory-droid",
        "AGENT_MEMORY_PROJECT": "my-project"
      }
    }
  }
}
```

Set the environment for your harness:

| Variable | Use |
|---|---|
| `AGENT_MEMORY_AGENT` | Required. One of `factory-droid`, `mistral-vibe`, `oh-my-pi`, `amp`, or `jetbrains-junie`. |
| `AGENT_MEMORY_PROJECT` | Default recall scope. Defaults to the working directory's name. |
| `AGENT_MEMORY_SESSION` | Harness session ID, when available. Joins MCP activity to the captured session. Without it, MCP creates a separate session marked `partial`. |

If you changed the database home or port, include `AGENT_MEMORY_HOME` and
`AGENT_MEMORY_PORT` in the MCP environment too.

| Tool | What it does |
|---|---|
| `recall` | Search active memories with full-text ranking and log what was retrieved. |
| `remember` | Store a memory. Identical content is confirmed, not duplicated. |
| `cite` | Record which retrieved memories an answer used. |
| `review` | Approve, quarantine, retire, confirm, or flag a contradiction. |
| `relate` | Record a relation between two entities. |
| `graph` | Retrieve relations around an entity. |
| `status` | Check the server connection and spool. |

Memories derived from tools start quarantined. If the server is down,
`recall` returns `degraded: true` and writes are queued to the spool.

## Relations and the graph

Use the MCP `relate` tool to record facts such as
`("payments-api", "owned by", "Dana")`.

### Browse relations

Replace `my-project` and `payments-api` with your scope and entity:

```sh
uv run mem graph entities --scope my-project
uv run mem graph show payments-api --hops 2 --scope my-project
```

Traversal follows active relations in either direction, from 1 to 3 hops.
Name matching ignores case and punctuation. Results include a `retrieval_id`
for `cite`; even searches with no results are logged for `why_answer`.

### Merge names that refer to the same entity

Have Ollama running with `qwen3:8b` installed. Use `--model` and `--host`
to choose a different model or local endpoint.

First, preview the proposed merges:

```sh
uv run mem graph resolve --scope my-project
```

After reviewing the proposal, rerun with `--apply` to record merges:

```sh
uv run mem graph resolve --scope my-project --apply
```

Nothing is rewritten in place:

- Relations are ordinary `fact` memories, not separate graph tables.
  Repeating the same triple in the same scope confirms it.
- A merge supersedes relations under the canonical name, preserving aliases
  and sources. If the canonical relation exists, it retires the duplicate.
- Unmatched names stay unchanged. Correct a bad merge by superseding again.

See [PLAN.md](PLAN.md#data-model-migration-0001) for storage details.

## Asking why

Use these queries to inspect the ledger. Replace `<event id>` with the
answer event ID you want to trace:

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

Add `--json` to `mem sql` for JSON output. Use `-` as the query to read SQL
from stdin.

If you have the DuckDB CLI installed, you can query an export without the
server or token. First create the export while the server is running:

```sh
uv run mem export
duckdb -c "FROM '$HOME/.agent-memory/export/latest/events.parquet' LIMIT 5"
```

## Data directory

```text
~/.agent-memory/          mode 0700
  memory.duckdb           the database; only mem serve opens it
  token                   Quack bearer token, mode 0600
  export/<timestamp>/     Parquet snapshots, one file per table
  export/latest           symlink to the newest snapshot
  spool/<agent>/*.jsonl   records waiting for the server; done/ and failed/ after replay
  blobs/<xx>/<sha256>.txt event content over 16 KiB, content-addressed
```

## Backups and maintenance

The running server automatically writes a Parquet snapshot every 24 hours
and keeps the newest 14. Failed exports are logged and retried.

| Task | Command or option |
|---|---|
| Export now | `uv run mem export` |
| Keep only N snapshots after export | `uv run mem export --keep N` |
| Change the automatic interval | `mem serve --export-interval-hours N` (`0` disables it) |
| Change automatic retention | `mem serve --export-keep N` |

Exports go to `export/<timestamp>/`; `export/latest` points to the newest.

Stop the server before checking or applying migrations:

```sh
uv run mem migrate --status
uv run mem migrate
```

The server must also be stopped for direct file access:

- `mem sql --direct QUERY` reads the file without the server.
  Add `--write` only when you intend to write.
- `mem export --direct` exports without the server.

## Run the server at login (macOS)

Stop any manually started server first. Run these steps from the repository root.

1. Install the commands and initialize the home:

   ```sh
   uv tool install --editable .
   mem init
   ```

2. Create the login service file:

   ```sh
   mkdir -p "$HOME/Library/LaunchAgents"
   sed -e "s|__MEM__|$(command -v mem)|g" -e "s|__HOME__|$HOME|g" \
       scripts/com.agent-memory.serve.plist \
       > "$HOME/Library/LaunchAgents/com.agent-memory.serve.plist"
   ```

3. Start the service and check it:

   ```sh
   launchctl bootstrap "gui/$(id -u)" \
       "$HOME/Library/LaunchAgents/com.agent-memory.serve.plist"
   mem smoke --live
   ```

Logs go to `~/.agent-memory/serve.log`. To stop the service:

```sh
launchctl bootout "gui/$(id -u)/com.agent-memory.serve"
```

For a migration, stop the service, run `mem migrate`, then repeat step 3.

## Security

- The server listens only on loopback and requires a private token.
  Traffic is plain HTTP; the token is not printed in settings or errors.
- Clients can read and insert ledger records, not update or delete them.
  The policy also blocks schema changes, settings changes, file access,
  comments, and multiple SQL statements in one text.
- Triggers enforce the append-only ledger. Corrections use superseding
  memories or review actions.
- Tool-derived memories remain quarantined until approved.
- Capture redacts secrets before writing to the ledger, blobs, or spool.
  `events.redactions` counts removals.
- Recall text is literal-quoted before entering SQL. The policy strips
  literals before checking the statement.

## Judge experiments

See [Running judge experiments](docs/judge-how-to.md) for the command order,
options, model choices, and how to resume a run. Experiments use a separate
home and port; drafted lessons never enter the memory ledger.

- The default labeling interface is full-screen and blinded.
  It requires Go 1.26.4 or newer; its first build may need network access.
- Add `--plain` to `judge label` for numbered prompts without Go or cursor
  controls. Pipes and `TERM=dumb` select this mode automatically.
- Both interfaces use the same labeling, resume, and edit-history rules.
  Never label against the production home.

## Development

### Checks

From the repository root:

```sh
uv sync
uv run pytest            # unit, schema, policy, and end-to-end Quack tests
uv run ruff check .      # lint
uv run ruff format --check .
uv run pyrefly check     # type check
uv run mem smoke         # end-to-end check against a throwaway server
```

To fix formatting, run `uv run ruff format` with the paths of files you changed.

For the Go labeling helper, start at the repository root:

```sh
cd src/agent_memory/judge/label_ui
gofmt -l .
go test ./...
go vet ./...
go build -mod=readonly -o /tmp/agent-memory-label-ui .
```

### Labeling helper cache

The Python wheel and source distribution include the Go sources.
The first labeling session with pending work builds a private cached helper:

- macOS: `~/Library/Caches/agent-memory/label-ui`
- Linux: `${XDG_CACHE_HOME:-~/.cache}/agent-memory/label-ui`

Unchanged launches reuse it. Source, dependency, protocol, platform, or
Go toolchain changes trigger a rebuild. Experiment text is not passed
in command arguments or logs.

### Schema changes

Add a new `NNNN_name.sql` file under `src/agent_memory/migrations/`.
Applied migrations are checksummed and must not be edited.

## Further reading

- [PLAN.md](PLAN.md): architecture, DuckDB behavior, and roadmap.
- [Capture spec](docs/phase-1-spec.md): payloads, adapters, spool, and MCP tools.
- [Judge how-to](docs/judge-how-to.md): step-by-step experiment commands.
- [Judge experiment spec](docs/judge-experiment-spec.md): design and historical runs.
