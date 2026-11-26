"""CLI commands for the usefulness experiment (`mem --home <exp_home> usefulness ...`)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from ..config import command_config
from .checks import DEFAULT_CHECK_TIMEOUT_S, run_trial_checks
from .memory_set import estimate_run_disk_usage, make_memory_set, verify_memory_set
from .runner import (
    DEFAULT_AUTO_LEVEL,
    DEFAULT_DROID_MODEL,
    DEFAULT_WALL_TIME_LIMIT_S,
    VALID_ARMS,
    DroidHarnessAdapter,
    TrialConfig,
    run_trial,
)
from .store import UsefulnessError

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


def _resolve_task_arg(home: Path, task_arg: str) -> Path:
    candidate = Path(task_arg).expanduser()
    if candidate.is_dir():
        return candidate
    in_home = home.expanduser() / "tasks" / task_arg
    if in_home.is_dir():
        return in_home
    raise UsefulnessError(f"task directory not found: {task_arg!r} (checked path and {in_home})")


@usefulness_app.command("run-trial")
def run_trial_command(
    ctx: typer.Context,
    run_id: Annotated[
        str,
        typer.Option("--run-id", help="Run identifier under <experiment_home>/runs/<run_id>."),
    ],
    task: Annotated[
        str,
        typer.Option(
            "--task",
            help="Task directory path, or task id under <experiment_home>/tasks/<task_id>.",
        ),
    ],
    arm: Annotated[
        str,
        typer.Option("--arm", help="Arm to run: 'memory' or 'no_memory'."),
    ],
    model: Annotated[
        str,
        typer.Option("--model", help="Model id passed to the harness."),
    ] = DEFAULT_DROID_MODEL,
    reasoning_effort: Annotated[
        str | None,
        typer.Option("--reasoning-effort", help="Optional reasoning effort passed to the harness."),
    ] = None,
    model_base_url: Annotated[
        str | None,
        typer.Option(
            "--model-base-url",
            help="Base URL of the model server. Loopback URLs run without egress opt-in; "
            "omitting it treats the model as hosted.",
        ),
    ] = None,
    source_home: Annotated[
        Path | None,
        typer.Option(
            "--source-home",
            help="Stopped throwaway home or memory-set directory to clone into the memory arm's "
            "trial home. Omit to initialize a fresh empty trial home.",
        ),
    ] = None,
    wall_time_limit: Annotated[
        float,
        typer.Option(
            "--wall-time-limit",
            help="External wall-time limit in seconds before stopping the harness.",
        ),
    ] = DEFAULT_WALL_TIME_LIMIT_S,
    auto_level: Annotated[
        str,
        typer.Option("--auto-level", help="Droid --auto autonomy level (low, medium, high)."),
    ] = DEFAULT_AUTO_LEVEL,
    trial_id: Annotated[
        str | None,
        typer.Option("--trial-id", help="Optional explicit trial identifier."),
    ] = None,
    droid_bin: Annotated[
        str,
        typer.Option("--droid-bin", help="Droid executable path or name."),
    ] = "droid",
    as_json: Annotated[
        bool,
        typer.Option("--json", help="Print the full trial record as JSON."),
    ] = False,
) -> None:
    """Run one trial of a task in a fresh clone and write its immutable trial record."""
    if arm not in VALID_ARMS:
        raise UsefulnessError(f"invalid arm {arm!r}: expected one of {VALID_ARMS}")
    cfg = command_config(ctx)
    task_dir = _resolve_task_arg(cfg.home, task)
    trial_cfg = TrialConfig(
        arm=arm,
        model=model,
        reasoning_effort=reasoning_effort,
        wall_time_limit_s=wall_time_limit,
        auto_level=auto_level,
        model_base_url=model_base_url,
        source_home=source_home,
        trial_id=trial_id,
    )
    adapter = DroidHarnessAdapter(droid_bin=droid_bin)
    record = run_trial(cfg, run_id, task_dir, trial_cfg, adapter=adapter)

    if as_json:
        print(json.dumps(record.to_dict(), ensure_ascii=False, indent=2, sort_keys=True))
        return

    print(f"trial       {record.trial_id}")
    print(f"task        {record.task_id}  arm={record.arm}")
    print(
        f"status      {record.status}"
        + (f" ({record.failure_category})" if record.failure_category else "")
    )
    print(
        f"wall time   {record.wall_time.total_s:.2f}s "
        f"(clone={record.wall_time.clone_s:.2f}s, setup={record.wall_time.setup_s:.2f}s, "
        f"harness={record.wall_time.harness_s:.2f}s, teardown={record.wall_time.teardown_s:.2f}s)"
    )
    print(
        f"changes     {len(record.repo_changes.changed_files)} file(s), "
        f"tool_calls={len(record.tool_calls)}"
    )
    print(
        f"tokens      in={record.tokens.input_tokens} out={record.tokens.output_tokens} "
        f"total={record.tokens.total_tokens}"
    )
    print(f"record      {record.record_path}")


@usefulness_app.command("run-checks")
def run_checks_command(
    ctx: typer.Context,
    run_id: Annotated[
        str,
        typer.Option("--run-id", help="Run identifier under <experiment_home>/runs/<run_id>."),
    ],
    trial_id: Annotated[
        str,
        typer.Option("--trial-id", help="Finished trial identifier to check."),
    ],
    task: Annotated[
        str | None,
        typer.Option(
            "--task",
            help="Optional task directory path (defaults to <experiment_home>/tasks/<task_id>).",
        ),
    ] = None,
    timeout: Annotated[
        float,
        typer.Option(
            "--timeout",
            help="Default timeout in seconds for checks that do not declare their own timeout_s.",
        ),
    ] = DEFAULT_CHECK_TIMEOUT_S,
    as_json: Annotated[
        bool,
        typer.Option("--json", help="Print the full checks result as JSON."),
    ] = False,
) -> None:
    """Install hidden tests and run a task's checks on a finished trial's clone."""
    cfg = command_config(ctx)
    task_dir = _resolve_task_arg(cfg.home, task) if task is not None else None
    result = run_trial_checks(
        cfg,
        run_id,
        trial_id,
        task_or_dir=task_dir,
        default_timeout_s=timeout,
    )
    if as_json:
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2, sort_keys=True))
        return

    print(f"trial       {result.trial_id}")
    print(f"task        {result.task_id}")
    print(
        f"checks      {result.overall_status} "
        f"(passed={result.passed_count}, failed={result.failed_count}, "
        f"could_not_run={result.could_not_run_count})"
    )
    if result.hidden_tests_installed:
        print(f"hidden      installed {len(result.hidden_tests_installed)} file(s)")
    elif result.hidden_tests_error:
        print(f"hidden      error: {result.hidden_tests_error}")
    print(f"duration    {result.total_duration_s:.2f}s")
    print(f"results     {result.checks_path}")
