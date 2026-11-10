"""The `mem` command."""

import json
import signal
import sys
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Annotated, Any

import duckdb
import typer

from . import __version__, audit
from .adapters import HARNESS_NAMES, UnsupportedTranscript, get_adapter
from .admission.cli import admit_app
from .capture.pipeline import Delivery, deliver, replay_spool
from .capture.spool import Spool
from .config import Config, ConfigError, load_config
from .db import (
    AgentMemoryError,
    MemoryClient,
    MemoryServer,
    applied_migrations,
    available_migrations,
    export_snapshot,
    initialize_home,
    migrate,
    open_direct,
)
from .graph.cli import graph_app
from .judge.cli import judge_app

app = typer.Typer(
    name="mem",
    help="Shared, auditable memory for AI coding agents.",
    no_args_is_help=True,
    add_completion=False,
    # Local variables in a traceback can include the server token.
    pretty_exceptions_show_locals=False,
)


@dataclass(frozen=True)
class _GlobalOptions:
    home: Path | None
    port: int | None


def _config(ctx: typer.Context) -> Config:
    # Resolved per command rather than in the callback, so `mem <cmd> --help`
    # works even when the environment is misconfigured.
    opts = ctx.obj
    if not isinstance(opts, _GlobalOptions):
        raise RuntimeError("mem callback did not run")
    return load_config(home=opts.home, port=opts.port)


def _show_version(value: bool) -> None:
    if value:
        print(f"agent-memory {__version__}")
        raise typer.Exit()


@app.callback()
def _root(
    ctx: typer.Context,
    home: Annotated[
        Path | None,
        typer.Option(
            help="Data directory. Defaults to $AGENT_MEMORY_HOME, then ~/.agent-memory.",
            show_default=False,
        ),
    ] = None,
    port: Annotated[
        int | None,
        typer.Option(
            help="Quack port. Defaults to $AGENT_MEMORY_PORT, then 9494.", show_default=False
        ),
    ] = None,
    version: Annotated[
        bool,
        typer.Option(
            "--version", callback=_show_version, is_eager=True, help="Show the version and exit."
        ),
    ] = False,
) -> None:
    ctx.obj = _GlobalOptions(home=home, port=port)


@app.command()
def init(ctx: typer.Context) -> None:
    """Create the data directory, token, and database. Safe to re-run."""
    cfg = _config(ctx)
    result = initialize_home(cfg)
    print(f"home      {cfg.home}")
    print(f"database  {cfg.db_path}")
    print(f"token     {cfg.token_path} ({'created' if result.created_token else 'kept existing'})")
    applied = ", ".join(m.label for m in result.applied) or "none (up to date)"
    print(f"migrated  {applied}")
    print("\nNext: mem serve   (then, in another shell: mem smoke --live)")


@app.command("migrate")
def migrate_command(
    ctx: typer.Context,
    status: Annotated[
        bool, typer.Option("--status", help="Show migration status without applying.")
    ] = False,
) -> None:
    """Apply pending migrations. The server must be stopped."""
    cfg = _config(ctx)
    con = open_direct(cfg.db_path, read_only=status)
    try:
        if status:
            applied = applied_migrations(con)
            for m in available_migrations():
                state = "applied" if m.version in applied else "pending"
                if m.version in applied and applied[m.version] != m.checksum:
                    state = "CHANGED SINCE APPLIED"
                print(f"{m.label:<30} {state}")
            return
        done = migrate(con)
    finally:
        con.close()
    print("applied: " + (", ".join(m.label for m in done) or "nothing to do"))


@app.command()
def serve(
    ctx: typer.Context,
    export_interval_hours: Annotated[
        float,
        typer.Option(min=0, help="Write a Parquet snapshot this often; 0 disables."),
    ] = 24.0,
    export_keep: Annotated[
        int, typer.Option(min=0, help="Snapshots to keep; older ones are deleted. 0 keeps all.")
    ] = 14,
) -> None:
    """Serve the database to local clients over Quack until interrupted."""
    cfg = _config(ctx)
    server = MemoryServer(cfg, export_interval_hours=export_interval_hours, export_keep=export_keep)
    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())
    server.start()
    print(f"serving {cfg.db_path} on {cfg.uri}", flush=True)
    try:
        _drain_spool(cfg)
        server.serve_forever(
            stop,
            on_export=lambda path: print(f"exported {path}", flush=True),
            on_error=lambda exc: print(f"export failed: {exc}", file=sys.stderr, flush=True),
        )
    finally:
        server.stop()
        print("stopped", flush=True)


@app.command()
def sql(
    ctx: typer.Context,
    query: Annotated[str, typer.Argument(help="SQL to run, or - to read it from stdin.")],
    direct: Annotated[
        bool,
        typer.Option(
            "--direct", help="Open the database file instead of using the server (must be stopped)."
        ),
    ] = False,
    write: Annotated[bool, typer.Option("--write", help="With --direct, open read-write.")] = False,
    as_json: Annotated[bool, typer.Option("--json", help="Print rows as JSON.")] = False,
    max_width: Annotated[
        int, typer.Option(min=0, help="Truncate cells to this width; 0 means no limit.")
    ] = 60,
) -> None:
    """Run a query through the server, or directly against the file with --direct."""
    if write and not direct:
        raise typer.BadParameter("only applies with --direct.", param_hint="--write")
    cfg = _config(ctx)
    text = sys.stdin.read() if query == "-" else query
    if direct:
        con = open_direct(cfg.db_path, read_only=not write)
        try:
            cursor = con.execute(text)
            columns = [d[0] for d in cursor.description] if cursor.description else []
            rows = cursor.fetchall() if columns else []
        finally:
            con.close()
    else:
        with MemoryClient.from_config(cfg) as client:
            columns, rows = client.query(text)
    if as_json:
        records = [dict(zip(columns, row, strict=True)) for row in rows]
        print(json.dumps(records, default=str, ensure_ascii=False, indent=2))
    else:
        print(render_table(columns, rows, max_width))


@app.command()
def export(
    ctx: typer.Context,
    out: Annotated[
        Path | None,
        typer.Option(help="Export directory. Defaults to <home>/export.", show_default=False),
    ] = None,
    direct: Annotated[
        bool,
        typer.Option(
            "--direct", help="Open the database file instead of using the server (must be stopped)."
        ),
    ] = False,
    keep: Annotated[
        int, typer.Option(min=0, help="Delete older snapshots beyond this many. 0 keeps all.")
    ] = 0,
) -> None:
    """Write every table to Parquet and point <out>/latest at the snapshot."""
    cfg = _config(ctx)
    target = (out or cfg.export_dir).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)
    if direct:
        con = open_direct(cfg.db_path, read_only=True)
        try:
            path = export_snapshot(con, target, keep=keep)
        finally:
            con.close()
    else:
        with MemoryClient.from_config(cfg) as client:
            path = client.export(target, keep=keep)
    print(f"exported {path}")


@app.command()
def smoke(
    ctx: typer.Context,
    live: Annotated[
        bool,
        typer.Option(
            "--live", help="Check the running server instead of a throwaway one. Read-only."
        ),
    ] = False,
) -> None:
    """End-to-end check of the server, its policy, and the client."""
    from .smoke import run_ephemeral, run_live

    checks = run_live(_config(ctx)) if live else run_ephemeral()
    width = max(len(c.name) for c in checks)
    for c in checks:
        print(f"{'PASS' if c.ok else 'FAIL'}  {c.name:<{width}}  {c.detail}")
    failed = sum(not c.ok for c in checks)
    print(f"\n{len(checks) - failed}/{len(checks)} checks passed")
    if failed:
        raise typer.Exit(1)


# Capture


HARNESS_HELP = "Which harness produced the input: " + " or ".join(HARNESS_NAMES) + "."


def _drain_spool(cfg: Config) -> None:
    """Replay what accumulated while the server was down. Never fatal for the server."""
    try:
        result = replay_spool(cfg)
    except Exception as exc:
        print(f"spool replay failed: {exc}", file=sys.stderr, flush=True)
        return
    if result.files:
        print(
            f"spool replayed: {result.files} file(s), {result.inserted} rows inserted, "
            f"{result.skipped} already present, {result.failed_lines} line(s) failed",
            flush=True,
        )


def _report_delivery(delivery: Delivery, what: str) -> None:
    if delivery.written is not None:
        w = delivery.written
        print(
            f"{what}: {w.sessions} session(s), {w.events} event(s), {w.memories} memory(ies), "
            f"{w.actions} action(s) inserted; {w.skipped} already present"
        )
        if w.unresolved_parents:
            print(f"  {len(w.unresolved_parents)} result(s) without a known call", file=sys.stderr)
    elif delivery.spooled:
        reason = f" ({delivery.error})" if delivery.error else ""
        print(f"{what}: spooled to {', '.join(str(p) for p in delivery.spooled)}{reason}")
    if not delivery.ok:
        raise typer.Exit(1)


@app.command()
def event(
    ctx: typer.Context,
    harness: Annotated[str, typer.Option(help=HARNESS_HELP)],
    spool_only: Annotated[
        bool, typer.Option("--spool-only", help="Write to the spool without trying the server.")
    ] = False,
    keep_raw: Annotated[
        bool, typer.Option("--keep-raw", help="Keep the full hook payload under payload.raw.")
    ] = False,
    record_pre_tool: Annotated[
        bool, typer.Option("--record-pre-tool", help="Also record pre-tool hooks as notes.")
    ] = False,
) -> None:
    """Record one hook payload read as JSON from stdin. Exits 0 once it is saved anywhere."""
    cfg = _config(ctx)
    try:
        payload = json.loads(sys.stdin.read())
    except json.JSONDecodeError as exc:
        raise typer.BadParameter(f"stdin is not JSON: {exc}") from None
    if not isinstance(payload, dict):
        raise typer.BadParameter("stdin must hold one JSON object")
    try:
        adapter = get_adapter(harness, record_pre_tool=record_pre_tool)
        records = adapter.from_hook(payload)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from None
    if keep_raw:
        records = [replace(r, raw=payload) for r in records]
    if not records:
        print(f"{payload.get('hook_event_name')}: nothing to record")
        return
    delivery = deliver(cfg, records, spool_only=spool_only, keep_raw=keep_raw)
    _report_delivery(delivery, str(payload.get("hook_event_name")))


@app.command()
def ingest(
    ctx: typer.Context,
    harness: Annotated[
        str | None, typer.Option(help=HARNESS_HELP + " Required with --transcript.")
    ] = None,
    transcript: Annotated[
        Path | None,
        typer.Option(help="Transcript file to import.", exists=True, dir_okay=False),
    ] = None,
    spool: Annotated[
        bool, typer.Option("--spool", help="Replay pending spool files instead.")
    ] = False,
    since_key: Annotated[
        str | None,
        typer.Option(help="Skip transcript entries up to and including this message id."),
    ] = None,
    include_thinking: Annotated[
        bool, typer.Option("--include-thinking", help="Also record assistant thinking blocks.")
    ] = False,
    keep_raw: Annotated[
        bool, typer.Option("--keep-raw", help="Keep raw entries under payload.raw.")
    ] = False,
    spool_only: Annotated[
        bool, typer.Option("--spool-only", help="Write to the spool without trying the server.")
    ] = False,
    purge_done: Annotated[
        float | None,
        typer.Option(help="With --spool, delete replayed files older than this many days."),
    ] = None,
) -> None:
    """Import a transcript, or replay the spool. Exits 0 once the records are saved anywhere."""
    cfg = _config(ctx)
    if spool == (transcript is not None):
        raise typer.BadParameter("pass exactly one of --transcript PATH or --spool")
    if spool:
        result = replay_spool(cfg, keep_raw=keep_raw)
        print(
            f"spool: {result.files} file(s), {result.inserted} rows inserted, "
            f"{result.skipped} already present, {result.failed_lines} line(s) failed"
        )
        for error in result.errors:
            print(f"  {error}", file=sys.stderr)
        for path in result.failed_paths:
            print(f"  failed lines kept in {path}", file=sys.stderr)
        if purge_done is not None:
            removed = Spool(cfg.spool_dir).purge_done(purge_done)
            print(f"purged {len(removed)} replayed file(s)")
        return
    if transcript is None or harness is None:
        raise typer.BadParameter("--transcript needs --harness")
    try:
        adapter = get_adapter(harness)
        records = adapter.from_transcript(
            transcript, since_key=since_key, include_thinking=include_thinking
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from None
    except UnsupportedTranscript as exc:
        raise AgentMemoryError(str(exc)) from None
    if not records:
        print(f"{transcript.name}: nothing new to record")
        return
    delivery = deliver(cfg, records, spool_only=spool_only, keep_raw=keep_raw)
    _report_delivery(delivery, transcript.name)


audit_app = typer.Typer(help="Read-only ledger diagnostics.", no_args_is_help=True)
app.add_typer(audit_app, name="audit")


@audit_app.command("check")
def audit_check(
    ctx: typer.Context,
    direct: Annotated[
        bool,
        typer.Option(
            "--direct",
            help="Read the stopped database file, including old schemas, without migrating.",
        ),
    ] = False,
    as_json: Annotated[
        bool, typer.Option("--json", help="Print a deterministic report with IDs and reason codes.")
    ] = False,
) -> None:
    """Check ledger integrity and provenance without writes.

    Exit 0: clean; 1: findings (including coverage warnings); 2: audit could not complete.
    Coverage gaps can describe an unfinished session, not necessarily a capture failure.
    """
    try:
        cfg = _config(ctx)
        if direct:
            con = open_direct(cfg.db_path, read_only=True)
            try:
                report = audit.check(con)
            finally:
                con.close()
        else:
            with MemoryClient.from_config(cfg) as client:
                report = audit.check(client)
    except (AgentMemoryError, ConfigError, duckdb.Error, OSError, ValueError, TypeError):
        # Connection errors can embed credentials, paths, or rejected SQL values.
        report = audit.read_failure()
    if as_json:
        print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2, sort_keys=True))
    else:
        counts = report.to_dict()["summary"]
        print(f"audit: {report.status} ({counts['errors']} errors, {counts['warnings']} warnings)")
        for finding in report.findings:
            related = f" related={','.join(finding.related_ids)}" if finding.related_ids else ""
            print(
                f"{finding.severity}: {finding.code} "
                f"{finding.entity_type}={finding.entity_id}{related}"
            )
        if report.exit_code == 2:
            print(
                "Check configuration, connectivity, and schema. For a stopped old database, "
                "use --direct; this command never initializes or migrates it."
            )
    if report.exit_code:
        raise typer.Exit(report.exit_code)


spool_app = typer.Typer(help="Inspect the spool of records waiting for the server.")
app.add_typer(spool_app, name="spool")
app.add_typer(judge_app, name="judge")
app.add_typer(graph_app, name="graph")
app.add_typer(admit_app, name="admit")


@spool_app.command("status")
def spool_status(ctx: typer.Context) -> None:
    """Pending files and lines per harness."""
    cfg = _config(ctx)
    rows = Spool(cfg.spool_dir).status()
    if not rows:
        print("spool is empty")
        return
    out = [
        (s.agent_id, s.files, s.lines, time.strftime("%Y-%m-%d %H:%M", time.localtime(s.oldest)))
        for s in rows
    ]
    print(render_table(["agent", "files", "lines", "oldest"], out, 0))


def render_table(columns: Sequence[str], rows: Sequence[Sequence[Any]], max_width: int) -> str:
    if not columns:
        return "OK"

    def cell(value: Any) -> str:
        text = "NULL" if value is None else str(value).replace("\n", "\\n")
        if max_width and len(text) > max_width:
            return text[: max_width - 1] + "…"
        return text

    body = [[cell(v) for v in row] for row in rows]
    widths = [max([len(col)] + [len(r[i]) for r in body]) for i, col in enumerate(columns)]
    lines = [
        " | ".join(col.ljust(w) for col, w in zip(columns, widths, strict=True)),
        "-+-".join("-" * w for w in widths),
    ]
    lines += [" | ".join(v.ljust(w) for v, w in zip(r, widths, strict=True)) for r in body]
    lines.append(f"({len(rows)} row{'' if len(rows) == 1 else 's'})")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return its exit code. Expected failures print one line, not a traceback."""
    try:
        app(args=argv, prog_name="mem")
    except SystemExit as exc:
        if exc.code is None or isinstance(exc.code, int):
            return exc.code or 0
        print(exc.code, file=sys.stderr)
        return 1
    except (AgentMemoryError, ConfigError) as exc:
        print(f"mem: {exc}", file=sys.stderr)
        return 1
    except duckdb.Error as exc:
        print(f"mem: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
