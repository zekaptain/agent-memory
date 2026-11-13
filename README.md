# agent-memory

A local memory database shared by your AI coding agents. It uses DuckDB 2.0
and serves local clients over Quack.

The ledger keeps a history instead of overwriting records. Use it to find:

- What an agent was shown.
- Where that information came from.
- Which memories the agent declared it used in its reply.

Storage, capture, a hook installer for Factory Droid, MCP tools, the
relation graph, judge experiments, and admission are implemented.

Start with the [quick start](#quick-start), then
[wire capture into Droid's hooks](#wire-capture-into-hooks),
[import a session](#import-a-droid-session), or
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

For an existing home, run the read-only preflight and follow the
[migration safety steps](#migration-safety) before applying new migrations.

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

Hooks, MCP clients, and the login service run `mem` and `mem-mcp` from
outside this checkout. Install a copy of both, built from the checkout
and held to the dependency versions in `uv.lock`:

```sh
uv export --no-dev --no-emit-project --no-hashes --frozen --quiet \
    --output-file /tmp/agent-memory-constraints.txt
uv tool install --reinstall --python "$(cat .python-version)" \
    --constraints /tmp/agent-memory-constraints.txt --from . agent-memory
```

Without the constraints, `uv tool install` resolves the dependencies
anew, to versions the tests may not have run on.

After installation, you can use `mem` instead of `uv run mem`. The copy
does not change when the checkout does, so a hook and the server keep
running the version you installed. Run the two commands again to bring
the copy up to date. If the update adds a migration, stop the server
first and run `mem migrate` before you start it again: `mem serve`
refuses to start while a migration is pending.

`uv tool install --editable .` installs commands that run the checkout's
code as it is at that moment. That suits development. A hook or a server
installed that way runs whatever the checkout has checked out.

Hooks do not need the `PATH`: `mem hooks install` writes the full path
of the `mem` it runs from, so run it from the installed `mem`.

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
| Factory Droid | Hooks, written by `mem hooks install`, and transcript import. |
| oh-my-pi and Amp | MCP tools only; no capture adapter yet. |
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
- `--since-key MESSAGE_ID` imports only what follows that entry.
- `--include-thinking` also imports thinking blocks.

Use this for a session from before the hooks were installed. With the
hooks in place a session is imported as it goes, without its thinking
blocks; `--include-thinking` on its transcript adds them.

### Wire capture into hooks

`mem hooks install` writes capture into Droid's hooks file,
`~/.factory/hooks.json`. Install the commands first (see
[install commands](#optional-install-commands-for-use-outside-this-repository)),
so that a hook runs the installed copy and not this checkout's `.venv`.
Initialize the home and start the server too (see the
[quick start](#quick-start)). Without a server, a hook keeps what it
records in the spool.

```sh
mem hooks install
mem hooks status
```

`install` adds one hook at each of six hook events: `SessionStart`,
`SessionEnd`, `UserPromptSubmit`, `PreCompact`, `Notification`, and
`Stop`. Each runs the same command, `mem event --harness droid`, written
with the full path of `mem` and the home and port in use, because a hook
runs with Droid's environment and not your shell's. To change the home,
the port, or a flag, run `install` again:

```sh
mem --home "$HOME/.agent-memory" --port 9494 hooks install --keep-raw
```

- Every other hook in the file stays as it is. `install` replaces only a
  hook whose whole command is `mem ... event ...`. One that runs
  `mem event` inside a longer command is left alone and reported, because
  capture then runs twice at that hook event.
- The file is rewritten as indented JSON. Its hooks and their order are
  kept; its line layout is not.
- `mem hooks status` shows each hook event as installed, missing, or
  different from what `install` would write. It exits 1 unless all are
  installed. `mem hooks uninstall` takes mem's hooks out again.
- `--file PATH` names another hooks file, such as a project's
  `.factory/hooks.json`. `--mem PATH` names the `mem` a hook runs.
- `install` changes nothing in a file that is not in Droid's format. It
  also refuses to create `hooks.json` while `settings.json` beside it
  holds hooks, because Droid reads those only when there is no
  `hooks.json`.
- A Droid session that is already running may use the change at once or
  keep the hooks it started with. `/hooks` in Droid shows the hooks a
  session runs.

For Droid the transcript is the record of replies, tool calls,
compactions, and what a subagent reports when its task is finished. A
Droid hook does not carry the reply and does not name the tool call or
the task it reports. The hooks therefore record a prompt, a
notification, the start and end of a session, and the request to compact
(as a note; the compaction comes from the transcript). At `Stop` and
`SessionEnd`, `mem event` also imports what the transcript has gained
since its last import. No hook is installed at `PostToolUse` or
`SubagentStop`, where there is nothing to record.

Droid writes a turn's outcome just after `Stop` has run. That outcome
arrives with the next turn's `Stop`, or with `SessionEnd`, which the
Droid app raises 30 minutes after a session was last used.

`--keep-raw` keeps the full payload with each event, redacted like
everything else. A payload kept this way stays with its record while the
record waits in the spool. `--record-pre-tool` also installs a hook at
`PreToolUse` and records a note for each tool call. That costs one
process per call and is meant for debugging.
See the [capture spec](docs/phase-1-spec.md) for payload and adapter details.

`mem event` is made to run as a hook. It prints nothing on stdout, because a
harness can add a hook's stdout to the agent's context, and it reports
problems on stderr. It exits 0 once every record is kept somewhere and 1
when nothing could be kept. It never exits 2, which a harness reads as an
instruction to block the agent.

### Recover queued records

If the server is down, capture saves records under `spool/` instead.
Starting `mem serve` replays pending records. You can also check and replay
them yourself:

```sh
uv run mem spool status
uv run mem ingest --spool
```

Add `--purge-done DAYS` to the replay command to remove old replayed files.

A record the ledger rejects does not hold up the rest of its batch. It is
kept under `spool/failed/`, and the ledger records the reason as a
`capture_error` event in the same session. `mem spool status` lists rejected
records next to pending ones. Once the cause is fixed, return them to the
spool and replay:

```sh
uv run mem spool retry
uv run mem ingest --spool
```

Before saving, capture redacts known secret patterns, including API keys,
tokens, JWTs, private keys, URL credentials, and password assignments.
Content over 16 KiB goes to `blobs/`; the event keeps a 4 KiB prefix and hash.

## Connect your agent through MCP

Install `mem-mcp` on the harness's `PATH` (see
[install commands](#optional-install-commands-for-use-outside-this-repository)),
then add a stdio server to its MCP configuration. For example:

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
| `AGENT_MEMORY_AGENT` | Required. One of `factory-droid`, `oh-my-pi`, `amp`, or `jetbrains-junie`. `mem-mcp` does not start as `human`, the agent that `mem review` writes as. |
| `AGENT_MEMORY_PROJECT` | Default recall scope. Defaults to the working directory's name. |
| `AGENT_MEMORY_SESSION` | Harness session ID, when available. Joins MCP activity to the captured session. MCP-only coverage remains partial even when this ID is present. |
| `AGENT_MEMORY_MCP_SERVER_NAME` | Optional capture setting when the MCP server is named something other than `agent-memory`. Attribution requires that exact server identity; a bare `cite` without recorded server identity stays unresolved. |

Droid's MCP `env` expands shell variables only, not the current Droid
session ID. `AGENT_MEMORY_SESSION` therefore cannot supply that session
automatically. MCP uses its own partial session instead. Captured `cite`
results carry the returned placeholder ID, which links the reply without
requiring the MCP and Droid sessions to match.

If you changed the database home or port, include `AGENT_MEMORY_HOME` and
`AGENT_MEMORY_PORT` in the MCP environment too.

| Tool | What it does |
|---|---|
| `recall` | Search active memories with full-text ranking and log what was retrieved. |
| `remember` | Store a memory. Identical content is confirmed, not duplicated. |
| `cite` | Record which retrieved memories a reply used. |
| `review` | Quarantine, retire, confirm, or flag a contradiction. It refuses to approve. |
| `relate` | Record a relation between two entities. |
| `graph` | Retrieve relations around an entity. |
| `status` | Check the server connection, the spool, and rejected records. |

Memories derived from tools start quarantined. Only a human approves one,
with `mem review` (see
[Memories awaiting approval](#memories-awaiting-approval)). The
`review` tool refuses `approve` and tells the agent to ask you. If the
server is down, `recall` returns `degraded: true` and writes are queued
to the spool.

A correction proposes a replacement. The previous fact stays available
until the replacement is approved when required, valid, and accepted.
Acceptance permanently supersedes the previous fact. Retiring, expiring,
or quarantining the replacement does not restore it. A future-dated fact is
not eligible before `valid_from`; `valid_to` and `expires_at` are exclusive
upper bounds. Confirmation is evidence, not approval.

Queued write IDs remain traceable if replay finds a duplicate. Inspect
`memory_deduplications` for their canonical identity; ordinary supersession
does not redirect later review actions.

## Memories awaiting approval

A quarantined memory is not recalled until a human approves it. With the
server running, list the memories awaiting approval, then approve or
retire one by its ID:

```sh
uv run mem review list --scope my-project
uv run mem review approve <memory-id> --reason "checked against the runbook"
uv run mem review retire <memory-id> --reason "the service was renamed"
```

`list` prints each memory awaiting approval with its scope, the agent
that wrote it, why it awaits approval, and any memory that approving it
would replace. Without `--scope` it reads every scope. `approve` shows
the memory and asks for `y` typed at a terminal. It refuses piped input,
and it refuses a memory that is not awaiting approval. `retire` withdraws
any memory for good and does not ask. Both require `--reason` and are
recorded as a review action by the `human` agent.

For the lessons that admission wrote, use admission review
(`mem admit review`) instead (see
[Admit lessons into memory](#admit-lessons-into-memory)). It also keeps
your answer as an example for the fitted approver.

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
`entities` reads every relation in the scopes, and that read is logged as a
retrieval too.

### Merge names that refer to the same entity

Have Ollama running with `qwen3:8b` installed. Use `--model` and `--host`
to choose a different model or local endpoint.

First, preview the proposed merges:

```sh
uv run mem graph resolve --scope my-project
```

After reading the proposal, rerun with `--apply` to record merges:

```sh
uv run mem graph resolve --scope my-project --apply
```

Each run logs the relations it read as a retrieval and prints its id. A
recorded merge names that retrieval, so you can see later what the proposal
was drawn from.

The model is asked about one block of entities at a time. If it returns
no usable clusters for a block, the merges from the other blocks are still
shown, and recorded with `--apply`. The command then names the block that
failed and why, and exits 1. If a block keeps failing, a smaller
`--block-size` gives the model less to cluster at once. If the model
server cannot be reached, the remaining blocks are not tried and the
output says how many.

Nothing is rewritten in place:

- Relations are ordinary `fact` memories, not separate graph tables.
  Repeating the same triple in the same scope confirms it.
- A merge accepts relations under the canonical name, preserving aliases
  and sources even when the canonical relation already exists. A relation
  between the two merged names is retired.
- Unmatched names stay unchanged. Correct a bad merge by superseding again.

See the [architecture](docs/architecture.md#data-model) for storage details.

## Asking why

Use these queries to inspect the ledger. Replace `<event id>` with the
event ID of the reply you want to trace:

```sh
# Recent sessions, with capture tier and counts
uv run mem sql "SELECT * FROM session_summary ORDER BY started_at DESC LIMIT 10"

# What a reply was based on, and where each memory came from
uv run mem sql "SELECT via, rank, memory_content, status_at_recall, memory_status_now,
                       source_kind, source_agent_id, source_content
                FROM why_answer WHERE answer_event_id = '<event id>'"

# Everything that happened to memories, in order
uv run mem sql "SELECT * FROM memory_history ORDER BY ts DESC LIMIT 20"
```

Add `--json` to `mem sql` for JSON output. Use `-` as the query to read SQL
from stdin.

Harness tier describes capability. `session_summary.capture_completeness`
comes from observations, while `recorded_capture_completeness` preserves the
original session assertion. Transcript imports report their coverage
boundary and exclusions; they do not prove coverage of later activity.
Old full-capture assertions without evidence are labeled unverified.
`session_summary.started_at` is the earliest the ledger knows of a
session: its row's start or its first event, whichever is earlier.
`recorded_started_at` is the row's own. The summary's `model`, `project`,
`title`, `calling_session_id`, and `harness_version` are what a
transcript import said of the session where one did: a hook knows few
of them.

`cite` can return a placeholder before the reply has been imported.
`answer_bindings` records a reply only when captured turn evidence uniquely
identifies it. `why_answer` exposes the effective reply, original
placeholder, and raw link IDs. Ambiguous or MCP-only attribution remains
unresolved. Cited zero-result searches remain visible with `returned = 0`.
Returned, captured in context, and declared used are separate observations,
not proof of the model's reasoning.

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
  spool/<agent>/*.jsonl   records waiting for the server; done/ after replay
  spool/failed/<agent>/   records the ledger rejected, kept until mem spool retry
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
Each one holds the ledger as it was at one moment. `mem export` checks
that no table grew while it copied them and copies them again if one did.
If the ledger is written to during three tries in a row, it exits 1 and
leaves `latest` where it was; run it again when no agent is writing.

Stop the server before checking or applying migrations:

```sh
uv run mem migrate --status
uv run mem migrate
```

The server must also be stopped for direct file access:

- `mem sql --direct QUERY` reads the file without the server.
  Add `--write` only when you intend to write.
- `mem export --direct` exports without the server.

### Migration safety

Run preflight before upgrading an existing home:

```sh
# Against the running server, read-only.
uv run mem audit check --json
# Or against an old database with its server stopped, read-only.
uv run mem audit check --direct --json
```

Diagnostics contain IDs and reason codes, not transcript text. Ambiguous
legacy correction histories need a person to check them; an old
`supersedes` edge is not proof of approval or acceptance.

The check exits 1 when it finds an error and 2 when it could not complete.
Warnings about capture coverage and attribution are listed without failing
it, because a session that only used the MCP tools always has them. Add
`--strict` to exit 1 on a warning too.

A live rollout is a separate operation: read the preflight, verify a
consistent backup of the database **and blobs**, stop the server, migrate,
restart, and verify read-only. Parquet exports alone do not back up blobs.
Restore the backup with its matching application version for rollback.
Do not delete a home or run reverse migrations to resolve a failed upgrade.

## Run the server at login (macOS)

Stop any manually started server first. Run these steps from the repository root.

1. Install the commands (see
   [install commands](#optional-install-commands-for-use-outside-this-repository))
   and initialize the home:

   ```sh
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

For a migration, stop the service, install the commands again, run
`mem migrate`, then repeat step 3.

## Security

- The server listens only on loopback and requires a private token.
  Traffic is plain HTTP; the token is not printed in settings or errors.
- The shared token authenticates access, not each agent's claimed identity.
- Clients can read and insert ledger records, not update or delete them.
  The policy also blocks schema changes, settings changes, file access,
  comments, and multiple SQL statements in one text.
- Triggers enforce the append-only ledger. Corrections use superseding
  memories or review actions.
- Tool-derived memories remain quarantined until a human approves them
  with `mem review`. The MCP tools and spool replay refuse an approval
  from any other agent. Every client holds the same token, so this binds
  agents that use the tools, not a process that writes to the ledger
  directly.
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

New runs use v4 questions and labels, including whole-rule support and a
separate human judgment of whether to store the exact draft. Turn-level
worth is not draft admission ground truth. Historical v1–v3 reports keep
their original definitions; their draft/decide/label operations are
read-only in the new version. Finish ongoing v3 labeling with the previous
application version before upgrading the judge. Ledger migrations do not
rewrite run files.

## Admit lessons into memory

Admission is separate from the sealed judge experiment. It reads captured
production turns and writes their lessons as quarantined memories, which
then await approval. Auto mode is off by default.

First create, human-label and freeze a **v4** experiment run such as
`exp5`. Older turn labels and unlabeled model verdicts cannot train the
approver. Then, with the production server running:

```sh
uv run mem admit import-labels --from "$HOME/.agent-memory-judge/judge/exp5"
uv run mem admit fit
uv run mem admit status
uv run mem admit enable --shadow
uv run mem admit run
uv run mem admit review --day 2026-11-10
```

Defaults use local Ollama models: `gemma4:26b` for drafting and judging,
`nomic-embed-text` for lesson embeddings. All commands need the
`admission` agent row, which migration 0007 adds, and `mem admit run`
reads sessions through the view of migration 0009. If either is missing,
stop the server, run `mem migrate`, then restart it.

Admission review approves or retires exact lessons and samples 30
discarded ones by default. Add `--plain` for numbered prompts, or
`--again` to correct saved answers. Refit afterwards to refresh the
approver and its held-out estimate. `mem admit enable --auto` refuses
unless held-out precision is at least 0.8 over 20 scored writes. A refit
returns to shadow mode.

See the [admission spec](docs/admission-spec.md) for calibration commands,
provenance, safety rules, and an opt-in daily launchd template. No
production run or scheduler is enabled by installing the code.

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

CI runs all of these on every pull request and on every push to `main`,
the tests and `mem smoke` on both Linux and macOS. The workflow is
[.github/workflows/ci.yml](.github/workflows/ci.yml).

### Labeling helper cache

The Python wheel and source distribution include the Go sources.
The first sitting with pending work builds a private cached helper:

- macOS: `~/Library/Caches/agent-memory/label-ui`
- Linux: `${XDG_CACHE_HOME:-~/.cache}/agent-memory/label-ui`

Unchanged launches reuse it. Source, dependency, protocol, platform, or
Go toolchain changes trigger a rebuild. Experiment text is not passed
in command arguments or logs.

### Schema changes

Add a new `NNNN_name.sql` file under `src/agent_memory/migrations/`.
Applied migrations are checksummed and must not be edited.

## Further reading

- [Architecture](docs/architecture.md): runtime, data model, security
  model, and DuckDB behavior.
- [Roadmap](docs/roadmap.md): planned work, known defects, and open
  decisions.
- [Glossary](GLOSSARY.md): the project's terms.
- [Decision records](docs/adr/): choices that are hard to reverse, with
  their reasons.
- [Capture spec](docs/phase-1-spec.md): payloads, adapters, spool, and MCP tools.
- [Judge how-to](docs/judge-how-to.md): step-by-step experiment commands.
- [Judge experiment spec](docs/judge-experiment-spec.md): the design for new
  runs. The [logs of earlier runs](docs/archive/judge-experiments.md) are
  archived.
- [Admission spec](docs/admission-spec.md): calibration, the fitted approver,
  admission review, and safeguards.
- [Contract hardening](docs/logic-hardening-spec.md): lifecycle, coverage,
  attribution, preflight, and rollout policy.
- [Memory usefulness evaluation](docs/memory-utility-evaluation-spec.md):
  design only, with no harness or experiment run in this change.
