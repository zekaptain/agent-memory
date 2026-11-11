"""The production admission commands. Experiment commands remain sealed."""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import typer

from ..config import Config, load_config
from ..db import MemoryClient, sql_literal
from ..judge.store import JudgeError, lock
from . import approver
from .examples import import_labels
from .fit import fit_approver
from .ledger import ADMISSION_AGENT
from .review import review_day
from .run import DEFAULT_DRAFTER, run_day
from .store import AdmissionStore

admit_app = typer.Typer(
    help="Admit lessons drafted from captured turns into production memory.", no_args_is_help=True
)
DEFAULT_BACKEND = "llm:gemma4:26b"
DEFAULT_EMBEDDER = "ollama:nomic-embed-text"


def _config(ctx: typer.Context) -> Config:
    opts = ctx.obj
    cfg = load_config(home=getattr(opts, "home", None), port=getattr(opts, "port", None))
    with MemoryClient.from_config(cfg) as client:
        _, rows = client.query(
            f"SELECT agent_id FROM agents WHERE agent_id = {sql_literal(ADMISSION_AGENT)}"
        )
    if not rows:
        raise JudgeError(
            f"admission agent is missing; stop the server, run mem --home {cfg.home} migrate, "
            "then restart the server"
        )
    return cfg


@contextmanager
def _locked(cfg: Config) -> Iterator[AdmissionStore]:
    """One admission command at a time. A second one is refused at once."""
    store = AdmissionStore(cfg)
    with lock(store.root / ".lock", wait=False) as held:
        if not held:
            raise JudgeError("another admission command is running; finish it before retrying")
        yield store


@admit_app.command("import-labels")
def import_command(
    ctx: typer.Context,
    source: Annotated[Path, typer.Option("--from", help="Frozen v4 experiment run directory.")],
    backend: Annotated[
        str, typer.Option(help="Judge arm whose gates to import.")
    ] = DEFAULT_BACKEND,
    embedder: Annotated[str, typer.Option()] = DEFAULT_EMBEDDER,
) -> None:
    """Import exact-draft human labels, not model judgments."""
    cfg = _config(ctx)
    with _locked(cfg):
        count = import_labels(
            cfg, source.expanduser().resolve(), backend=backend, embedder=embedder
        )
    print(f"Imported {count} labeled draft(s).")


@admit_app.command("fit")
def fit_command(
    ctx: typer.Context,
    backend: Annotated[str, typer.Option()] = DEFAULT_BACKEND,
    embedder: Annotated[str, typer.Option()] = DEFAULT_EMBEDDER,
    similarity_floor: Annotated[float, typer.Option(min=0, max=1)] = 0.35,
) -> None:
    """Fit on calibration labels and measure untouched held-out labels."""
    cfg = _config(ctx)
    with _locked(cfg):
        model = fit_approver(
            cfg, backend=backend, embedder=embedder, similarity_floor=similarity_floor
        )
    print(f"Fitted approver {model['version']}.")
    print(json.dumps(model["metrics"], indent=2, sort_keys=True))


@admit_app.command("status")
def status_command(ctx: typer.Context) -> None:
    """Show mode, fitted metrics and the capture watermark."""
    cfg = _config(ctx)
    store = AdmissionStore(cfg)
    result = {**store.read_state(), "examples": len(store.read_examples())}
    if store.approver_path.exists():
        model = store.read_approver()
        result.update(
            backend=model["backend"], embedder=model["embedder"], metrics=model["metrics"]
        )
    print(json.dumps(result, indent=2, sort_keys=True))


@admit_app.command("enable")
def enable_command(
    ctx: typer.Context,
    shadow: Annotated[bool, typer.Option("--shadow")] = False,
    auto: Annotated[bool, typer.Option("--auto")] = False,
    min_writes: Annotated[int, typer.Option(min=1, help="Minimum scored held-out writes.")] = 20,
) -> None:
    """Choose shadow or guarded automatic approval."""
    cfg = _config(ctx)
    if shadow == auto:
        raise typer.BadParameter("choose exactly one of --shadow or --auto")
    with _locked(cfg) as store:
        model = store.read_approver()
        approver.validate_model(model)
        if auto:
            approver.validate_examples(model, store.read_examples())
        if auto and not approver.can_auto(model, min_writes=min_writes):
            raise JudgeError(
                f"auto requires held-out precision >= 0.8 over at least {min_writes} scored "
                "held-out writes; collect human labels and run mem admit fit"
            )
        store.write_state(
            mode="auto" if auto else "shadow",
            approver_version=model["version"],
            min_writes=min_writes,
        )
    print("Admission mode: " + ("auto" if auto else "shadow"))


@admit_app.command("run")
def run_command(
    ctx: typer.Context,
    day: Annotated[str | None, typer.Option(help="UTC day; defaults to today.")] = None,
    drafter: Annotated[str, typer.Option()] = DEFAULT_DRAFTER,
    backend: Annotated[str | None, typer.Option(help="Must match the fitted backend.")] = None,
) -> None:
    """Draft and judge captured turns, then quarantine surviving lessons."""
    cfg = _config(ctx)
    with _locked(cfg):
        result = run_day(
            cfg, day=day, drafter=drafter, backend=backend, log=lambda s: print(s, flush=True)
        )
    print(f"Admission {result['day']}: {result['processed']} processed ({result['mode']}).")


@admit_app.command("review")
def review_command(
    ctx: typer.Context,
    day: Annotated[str | None, typer.Option(help="UTC admission day; defaults to today.")] = None,
    discards: Annotated[int, typer.Option(min=0)] = 30,
    seed: Annotated[int, typer.Option()] = 1,
    limit: Annotated[int | None, typer.Option(min=1)] = None,
    plain: Annotated[bool, typer.Option("--plain")] = False,
    again: Annotated[bool, typer.Option("--again", help="Reopen completed reviews.")] = False,
) -> None:
    """Review blinded lessons, including a seeded sample of rejected drafts."""
    cfg = _config(ctx)
    with _locked(cfg):
        result = review_day(
            cfg,
            day or datetime.now(UTC).strftime("%Y-%m-%d"),
            discards=discards,
            seed=seed,
            limit=limit,
            plain=plain,
            again=again,
        )
    print(f"Reviewed {result['reviewed']}, skipped {result['skipped']}.")
