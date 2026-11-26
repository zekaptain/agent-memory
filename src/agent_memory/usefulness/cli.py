"""CLI commands for the usefulness experiment (`mem --home <exp_home> usefulness ...`)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from ..config import command_config
from .memory_set import estimate_run_disk_usage, make_memory_set, verify_memory_set

usefulness_app = typer.Typer(
    help="Usefulness experiment commands (always require a separate experiment --home).",
    no_args_is_help=True,
)


@usefulness_app.command("memory-set")
def memory_set_command(
    ctx: typer.Context,
    source: Annotated[
        Path,
        typer.Option(
            "--source",
            help="Stopped source home to copy into <experiment_home>/memory-set.",
            exists=True,
            file_okay=False,
        ),
    ],
    cutoff: Annotated[
        str,
        typer.Option(
            "--cutoff",
            help="Cutoff date (YYYY-MM-DD) or timezone-aware ISO-8601 timestamp.",
        ),
    ],
    last_day: Annotated[
        str,
        typer.Option(
            "--last-day",
            help="Last day of the run (YYYY-MM-DD or timezone-aware ISO-8601 timestamp) "
            "for counting validity-boundary crossings.",
        ),
    ],
    trials: Annotated[
        int | None,
        typer.Option(
            "--trials",
            min=0,
            help="Optional number of memory-arm trials to estimate baseline disk and copy time.",
        ),
    ] = None,
    as_json: Annotated[
        bool, typer.Option("--json", help="Print the manifest summary as JSON.")
    ] = False,
) -> None:
    """Copy a stopped ledger home and its blobs into the experiment's memory set and hash it."""
    cfg = command_config(ctx)
    result = make_memory_set(source, cfg, cutoff=cutoff, last_day=last_day)
    estimate = (
        estimate_run_disk_usage(result.manifest, trials, copy_seconds_per_trial=result.copy_seconds)
        if trials is not None
        else None
    )
    if as_json:
        payload = {
            "memory_set_dir": str(result.memory_set_dir),
            "manifest_path": str(result.manifest_path),
            "manifest_sha256": result.manifest_sha256,
            "copy_seconds": result.copy_seconds,
            "manifest": result.manifest,
        }
        if estimate is not None:
            payload["disk_estimate"] = estimate.to_dict()
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        return

    m = result.manifest
    vc = m["validity_changes"]
    print(f"memory set  {result.memory_set_dir}")
    print(f"manifest    {result.manifest_path}")
    print(f"sha256      {m['memory_set_sha256']}")
    print(f"database    {m['database']['bytes']} bytes  sha256={m['database']['sha256']}")
    print(
        f"blobs       {m['blobs']['file_count']} file(s), "
        f"{m['blobs']['total_bytes']} bytes  sha256={m['blobs']['sha256']}"
    )
    print(f"migrations  level {m['migration_level']}")
    print(f"cutoff      {m['cutoff']}  last_day={m['last_day']}")
    print(f"memories    total={m['counts']['total']}  by_status={m['counts']['by_status']}")
    print(
        f"validity    crossing={vc['total_crossing']} "
        f"(starts={vc['starts_in_window']}, ends={vc['ends_in_window']}, "
        f"non_terminal={vc['non_terminal_crossing']})"
    )
    print(f"copied in   {result.copy_seconds:.3f}s")
    if estimate is not None:
        gib = estimate.baseline_total_bytes / (1024**3)
        print(
            f"estimate    {estimate.trial_count} memory-arm trial(s): "
            f"{estimate.baseline_total_bytes} bytes ({gib:.2f} GiB), "
            f"~{estimate.estimated_total_copy_seconds:.1f}s total copy"
        )


@usefulness_app.command("verify-memory-set")
def verify_memory_set_command(
    ctx: typer.Context,
    as_json: Annotated[
        bool, typer.Option("--json", help="Print the verified manifest as JSON.")
    ] = False,
) -> None:
    """Re-hash the experiment home's memory set and confirm it matches its manifest."""
    cfg = command_config(ctx)
    manifest = verify_memory_set(cfg)
    if as_json:
        print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
        return
    print(
        f"verified    {cfg.home / 'memory-set'}  "
        f"sha256={manifest['memory_set_sha256']}  "
        f"total_bytes={manifest['total_bytes']}"
    )
