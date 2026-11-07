"""`mem judge`: the experiment commands. Registered on the main app from cli.py."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.markup import escape
from rich.table import Table

from ..config import Config, load_config
from .backends.base import REMOTE_ENV
from .labeling import label_items
from .pipeline import (
    build_candidates,
    decide_items,
    draft_items,
    import_sessions,
    live_step,
    purge_run,
)
from .report import build_report
from .store import JudgeError, RunStore, ensure_experiment_home
from .triggers import CALLER_KINDS, STRATUM_PRIORITY, closed_by_of

judge_app = typer.Typer(
    help="Judge experiment: decision models as failure and lesson triage. Separate home only.",
    no_args_is_help=True,
)

console = Console()
err_console = Console(stderr=True)

RUN = Annotated[str, typer.Option("--run", help="Run id, for example exp1.")]
LIVE_DAY = Annotated[
    str | None, typer.Option("--live", help="Operate on live/<DAY> instead of the main run.")
]


def _config(ctx: typer.Context) -> Config:
    opts = ctx.obj
    cfg = load_config(home=getattr(opts, "home", None), port=getattr(opts, "port", None))
    ensure_experiment_home(cfg)
    return cfg


# The rendered report's widest tables (Stage A has 13 columns) need about this many columns.
REPORT_WIDTH = 150


def _log(message: str) -> None:
    """Progress: stdout, dim, so a long run can be followed without drowning the result."""
    console.print(message, markup=False, highlight=False, style="dim")


def _warn(message: str) -> None:
    """A problem the user has to act on: stderr, so it survives a redirect and a pipe."""
    err_console.print(message, markup=False, highlight=False, style="yellow")


def _errors(errors: list[str]) -> None:
    for error in errors:
        err_console.print(error, markup=False, highlight=False, style="red")


def _store(cfg: Config, run: str, live: str | None) -> RunStore:
    store = RunStore(cfg, run)
    return store.child(live) if live else store


def _parse_since(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        raise typer.BadParameter(f"{value!r} is not an ISO date or datetime") from None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _strata_table(title: str, *columns: tuple[str, dict[str, int]]) -> Table:
    table = Table(title=title, show_edge=False)
    table.add_column("stratum")
    for name, _ in columns:
        table.add_column(name, justify="right")
    for s in STRATUM_PRIORITY:
        table.add_row(s, *[str(counts.get(s, 0)) for _, counts in columns])
    table.add_row("total", *[str(sum(counts.values())) for _, counts in columns], style="bold")
    return table


@judge_app.command("import")
def import_command(
    ctx: typer.Context,
    sessions_dir: Annotated[
        Path, typer.Option(help="Droid sessions directory, usually ~/.factory/sessions.")
    ],
    limit: Annotated[int | None, typer.Option(help="Import at most this many transcripts.")] = None,
    project: Annotated[
        str | None, typer.Option(help="Only transcripts under this project slug.")
    ] = None,
    since: Annotated[
        str | None, typer.Option(help="Only transcripts modified at or after this ISO date.")
    ] = None,
) -> None:
    """Import Droid transcripts into the experiment ledger. Needs `mem serve` on this home."""
    cfg = _config(ctx)
    with console.status("importing transcripts..."):
        summary = import_sessions(
            cfg,
            sessions_dir,
            limit=limit,
            project=project,
            since=_parse_since(since) if since else None,
            log=_log,
        )
    table = Table(show_header=False, box=None)
    table.add_row("files imported", str(summary.files))
    table.add_row("files skipped", str(summary.skipped_files))
    if summary.errors:
        table.add_row("files failed", str(len(summary.errors)), style="red")
    table.add_row("records", str(summary.records))
    table.add_row("rows inserted", str(summary.inserted))
    table.add_row("rows already present", str(summary.skipped_rows))
    console.print(table)
    _errors(summary.errors)


@judge_app.command()
def candidates(
    ctx: typer.Context,
    run: RUN,
    n: Annotated[int, typer.Option(min=1, help="Sample size.")] = 200,
    seed: Annotated[int, typer.Option(help="Sampling seed.")] = 1,
    clean_share: Annotated[
        float, typer.Option(min=0.0, max=1.0, help="Share of the sample drawn from clean turns.")
    ] = 0.25,
    human_share: Annotated[
        float,
        typer.Option(min=0.0, max=1.0, help="Share of the sample from sessions a person started."),
    ] = 0.5,
) -> None:
    """Segment turns, apply triggers, sample, and write projections for a new run."""
    cfg = _config(ctx)
    with console.status("segmenting turns and projecting..."):
        summary = build_candidates(
            cfg,
            run,
            n=n,
            seed=seed,
            clean_share=clean_share,
            human_share=human_share,
            log=_log,
        )
    console.print(
        f"{summary.items} items, {summary.redactions} redaction(s) in projections; "
        f"{summary.harness_turns} harness-opened turn(s) excluded from the pool"
    )
    console.print(_strata_table(f"run {run}", ("pool", summary.pool), ("chosen", summary.chosen)))
    callers = Table(title="session started by", show_edge=False)
    callers.add_column("caller")
    callers.add_column("pool", justify="right")
    callers.add_column("chosen", justify="right")
    for kind in CALLER_KINDS:
        callers.add_row(
            kind, str(summary.pool_callers.get(kind, 0)), str(summary.chosen_callers.get(kind, 0))
        )
    console.print(callers)


DRAFTER_HELP = (
    "ollama:<model> | llamacpp:<model> | openrouter:<model> (needs OPENROUTER_API_KEY and "
    f"{REMOTE_ENV}=1). Repeatable; each drafter drafts every item."
)


@judge_app.command()
def draft(
    ctx: typer.Context,
    run: RUN,
    drafter: Annotated[list[str] | None, typer.Option(help=DRAFTER_HELP)] = None,
    limit: Annotated[int | None, typer.Option(help="Draft at most this many items.")] = None,
    live: LIVE_DAY = None,
) -> None:
    """Write a candidate lesson (or a decline) for every item, per drafter. Resumable."""
    cfg = _config(ctx)
    store = _store(cfg, run, live)
    for spec in drafter or ["ollama:gemma4:26b"]:
        summary = draft_items(cfg, run, spec, limit=limit, store=store, log=_log, warn=_warn)
        line = f"drafted {summary.written} item(s) with {spec}"
        if summary.failed:
            line += f"; {summary.failed} failed, rerun to retry"
        console.print(line, markup=False, highlight=False)


@judge_app.command()
def decide(
    ctx: typer.Context,
    run: RUN,
    backend: Annotated[
        list[str],
        typer.Option(
            help="rules | readout:<model> | llm:<model> | llamacpp:<model> | jev[:<model>]. "
            "Repeatable."
        ),
    ],
    limit: Annotated[int | None, typer.Option(help="Decide at most this many items.")] = None,
    no_events: Annotated[
        bool, typer.Option("--no-events", help="Do not record decision events in the ledger.")
    ] = False,
    live: LIVE_DAY = None,
) -> None:
    """Run one or more backends over the items. Resumable per backend."""
    cfg = _config(ctx)
    store = _store(cfg, run, live)
    written = decide_items(
        cfg,
        run,
        backend,
        limit=limit,
        store=store,
        record_events=not no_events,
        log=_log,
        warn=_warn,
    )
    table = Table(show_header=True, show_edge=False)
    table.add_column("backend")
    table.add_column("new verdicts", justify="right")
    for name, count in written.items():
        table.add_row(name, str(count))
    console.print(table)


@judge_app.command()
def label(
    ctx: typer.Context,
    run: RUN,
    limit: Annotated[int | None, typer.Option(help="Stop after this many items.")] = None,
    seed: Annotated[int, typer.Option(help="Order seed, also for the A/B draft order.")] = 1,
    item: Annotated[
        list[str] | None,
        typer.Option("--item", help="Label (or relabel in full) just these item ids. Repeatable."),
    ] = None,
    review: Annotated[
        bool,
        typer.Option("--review", help="Step through labeled items showing their current labels."),
    ] = False,
    live: LIVE_DAY = None,
) -> None:
    """Review items in the terminal. Verdicts are never shown. Resumable."""
    cfg = _config(ctx)
    store = _store(cfg, run, live)
    store.read_run()
    label_items(store, console=console, limit=limit, seed=seed, item_ids=item or (), review=review)


@judge_app.command()
def report(
    ctx: typer.Context,
    run: RUN,
    freeze: Annotated[
        bool,
        typer.Option("--freeze", help="Choose thresholds on the calibration split and lock them."),
    ] = False,
    live: LIVE_DAY = None,
    raw: Annotated[
        bool, typer.Option("--raw", help="Print markdown instead of rendering.")
    ] = False,
) -> None:
    """Score verdicts against labels and write report.md."""
    cfg = _config(ctx)
    store = _store(cfg, run, live)
    text = build_report(store, freeze=freeze)
    if raw:
        print(text)
    else:
        console.print(Markdown(text))
        if console.width < REPORT_WIDTH:
            _warn(
                f"the tables above were squeezed into {console.width} columns and need about "
                f"{REPORT_WIDTH}; widen the terminal, use --raw, or open the file"
            )
    # stderr, so `--raw > report.md` holds only the report; soft_wrap keeps the path whole.
    err_console.print(
        f"written to {store.report_path}",
        markup=False,
        highlight=False,
        soft_wrap=True,
        style="dim",
    )


@judge_app.command()
def live(
    ctx: typer.Context,
    run: RUN,
    since: Annotated[str, typer.Option(help="Judge sessions started at or after this ISO date.")],
    sessions_dir: Annotated[Path, typer.Option(help="Droid sessions directory.")] = Path(
        "~/.factory/sessions"
    ),
    backend: Annotated[
        list[str] | None, typer.Option(help="Backends to run with frozen thresholds. Repeatable.")
    ] = None,
    drafter: Annotated[
        list[str] | None, typer.Option(help=DRAFTER_HELP + " Omit to skip drafting.")
    ] = None,
    day: Annotated[
        str | None, typer.Option(help="Live day label; defaults to today (UTC).")
    ] = None,
) -> None:
    """One day of the live week: import new sessions and judge them with frozen thresholds."""
    cfg = _config(ctx)
    backends = backend or ["rules"]
    summary = live_step(
        cfg,
        run,
        sessions_dir=sessions_dir,
        since=_parse_since(since),
        backends=backends,
        drafter_specs=drafter or (),
        day=day,
        log=_log,
        warn=_warn,
    )
    console.print(
        f"live day {summary.day}: imported {summary.imported.files} file(s), "
        + (f"{summary.candidates.items} new item(s), " if summary.candidates else "")
        + f"drafted {summary.drafted}, verdicts {summary.decided}",
        markup=False,
        highlight=False,
    )
    _errors(summary.imported.errors)


@judge_app.command()
def status(ctx: typer.Context, run: RUN, live: LIVE_DAY = None) -> None:
    """What the run holds so far."""
    cfg = _config(ctx)
    store = _store(cfg, run, live)
    meta = store.read_run()
    items = store.read_jsonl(store.items_path)
    counts: dict[str, int] = {}
    for item in items:
        counts[item["stratum"]] = counts.get(item["stratum"], 0) + 1
    console.print(_strata_table(f"run {run}", ("items", counts)))
    table = Table(show_header=False, box=None)
    closed: dict[str, int] = {}
    for item in items:
        closed[closed_by_of(item)] = closed.get(closed_by_of(item), 0) + 1
    table.add_row("closed by", ", ".join(f"{k} {v}" for k, v in sorted(closed.items())))
    drafts: dict[str, list[int]] = {}
    for row in store.read_jsonl(store.drafts_path):
        slot = drafts.setdefault(str(row.get("drafter") or "unknown"), [0, 0])
        slot[0] += 1
        slot[1] += row.get("lesson") is None
    for name, (count, declined) in sorted(drafts.items()):
        table.add_row(f"drafts {escape(name)}", f"{count} ({declined} declined)")
    table.add_row("labels", str(len(store.read_jsonl(store.labels_path))))
    if store.verdicts_dir.exists():
        for path in sorted(store.verdicts_dir.glob("*.jsonl")):
            table.add_row(f"verdicts {escape(path.stem)}", str(len(store.read_jsonl(path))))
    table.add_row("frozen", "yes" if meta.get("frozen") else "no")
    console.print(table)


@judge_app.command()
def purge(ctx: typer.Context, run: RUN) -> None:
    """Delete projection text for a run (and its live days). Verdicts and labels stay."""
    cfg = _config(ctx)
    removed = purge_run(cfg, run)
    console.print(f"removed {removed} projection file(s)")


__all__ = ["JudgeError", "judge_app"]
