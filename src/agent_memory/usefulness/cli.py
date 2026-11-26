"""CLI commands for the usefulness experiment (`mem --home <exp_home> usefulness ...`)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from ..config import command_config
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
