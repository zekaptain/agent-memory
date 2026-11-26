"""Post-trial task check execution for the usefulness experiment (MEM-24).

Runs a task's declared check commands against a finished trial's repository
clone, installing hidden tests only at this point (never while the agent is
working), identically across both arms, and writes the results to a separate
write-once ``checks.json`` alongside ``record.json`` without modifying the
agent's trial record.
"""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from ..config import Config
from ..judge.store import utc_now_iso
from .store import UsefulnessError, UsefulnessRunStore, read_json_file, sha256_file, write_once_json
from .tasks import (
    CheckCommand,
    TaskSpec,
    install_hidden_tests,
    load_task,
    load_verified_task,
    verify_task_against_manifest,
)

TRIAL_CHECKS_SCHEMA_VERSION = 1
DEFAULT_CHECK_TIMEOUT_S = 300.0
DEFAULT_KILL_GRACE_S = 5.0
MAX_OUTPUT_BYTES = 262_144  # 256 KiB cap per stream in checks.json

CheckStatus = Literal["passed", "failed", "could_not_run"]


@dataclass(frozen=True)
class CheckOutcome:
    """Result of executing one ``CheckCommand`` on a finished trial's clone."""

    name: str
    command: tuple[str, ...]
    timeout_s: float
    status: CheckStatus
    exit_code: int | None
    duration_s: float
    stdout: str
    stderr: str
    stdout_bytes: int
    stderr_bytes: int
    stdout_truncated: bool
    stderr_truncated: bool
    reason: str | None = None
    error_detail: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "command": list(self.command),
            "timeout_s": self.timeout_s,
            "status": self.status,
            "exit_code": self.exit_code,
            "duration_s": self.duration_s,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "stdout_bytes": self.stdout_bytes,
            "stderr_bytes": self.stderr_bytes,
            "stdout_truncated": self.stdout_truncated,
            "stderr_truncated": self.stderr_truncated,
            "reason": self.reason,
            "error_detail": self.error_detail,
        }


@dataclass(frozen=True)
class TrialChecksResult:
    """Immutable check results persisted to ``<trial_dir>/checks.json``."""

    schema_version: int
    trial_id: str
    run_id: str
    task_id: str
    task_sha256: str
    record_sha256: str
    overall_status: CheckStatus
    passed_count: int
    failed_count: int
    could_not_run_count: int
    hidden_tests_installed: tuple[str, ...]
    hidden_tests_error: str | None
    checks: tuple[CheckOutcome, ...]
    started_at: str
    finished_at: str
    total_duration_s: float
    checks_path: Path
    checks_sha256: str

    def to_dict(self) -> dict[str, Any]:
        # Deliberately omits the trial's arm so checks.json can be shown directly
        # to a blind evaluator without leaking arm identity.
        return {
            "schema_version": self.schema_version,
            "trial_id": self.trial_id,
            "run_id": self.run_id,
            "task_id": self.task_id,
            "task_sha256": self.task_sha256,
            "record_sha256": self.record_sha256,
            "overall_status": self.overall_status,
            "passed_count": self.passed_count,
            "failed_count": self.failed_count,
            "could_not_run_count": self.could_not_run_count,
            "hidden_tests_installed": list(self.hidden_tests_installed),
            "hidden_tests_error": self.hidden_tests_error,
            "checks": [c.to_dict() for c in self.checks],
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "total_duration_s": self.total_duration_s,
        }


def _sanitized_check_env() -> dict[str, str]:
    """Return a clean environment identical across both arms (no ledger/harness vars)."""
    env: dict[str, str] = {}
    for key, value in os.environ.items():
        if key.startswith("AGENT_MEMORY_") or key.startswith("FACTORY_"):
            continue
        env[key] = value
    env["GIT_TERMINAL_PROMPT"] = "0"
    return env


def _cap_output(
    raw: bytes | str | None, max_bytes: int = MAX_OUTPUT_BYTES
) -> tuple[str, int, bool]:
    if raw is None:
        return "", 0, False
    data = raw.encode("utf-8", errors="replace") if isinstance(raw, str) else raw
    total = len(data)
    if total <= max_bytes:
        return data.decode("utf-8", errors="replace"), total, False
    head = data[:max_bytes].decode("utf-8", errors="replace")
    return head, total, True


def _terminate_process_group(proc: subprocess.Popen[bytes], grace_s: float) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    except OSError:
        proc.terminate()
    try:
        proc.wait(timeout=grace_s)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    except OSError:
        proc.kill()
    with contextlib.suppress(subprocess.TimeoutExpired):
        proc.wait(timeout=grace_s)


def _execute_one_check(
    check: CheckCommand,
    clone_dir: Path,
    env: Mapping[str, str],
    *,
    default_timeout_s: float,
    kill_grace_s: float,
) -> CheckOutcome:
    timeout_s = check.timeout_s if check.timeout_s is not None else default_timeout_s
    t0 = time.monotonic()
    try:
        proc = subprocess.Popen(
            list(check.command),
            cwd=str(clone_dir),
            env=dict(env),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
    except FileNotFoundError as exc:
        duration_s = round(time.monotonic() - t0, 6)
        return CheckOutcome(
            name=check.name,
            command=check.command,
            timeout_s=timeout_s,
            status="could_not_run",
            exit_code=None,
            duration_s=duration_s,
            stdout="",
            stderr="",
            stdout_bytes=0,
            stderr_bytes=0,
            stdout_truncated=False,
            stderr_truncated=False,
            reason="command_not_found",
            error_detail=str(exc),
        )
    except PermissionError as exc:
        duration_s = round(time.monotonic() - t0, 6)
        return CheckOutcome(
            name=check.name,
            command=check.command,
            timeout_s=timeout_s,
            status="could_not_run",
            exit_code=None,
            duration_s=duration_s,
            stdout="",
            stderr="",
            stdout_bytes=0,
            stderr_bytes=0,
            stdout_truncated=False,
            stderr_truncated=False,
            reason="permission_denied",
            error_detail=str(exc),
        )
    except OSError as exc:
        duration_s = round(time.monotonic() - t0, 6)
        return CheckOutcome(
            name=check.name,
            command=check.command,
            timeout_s=timeout_s,
            status="could_not_run",
            exit_code=None,
            duration_s=duration_s,
            stdout="",
            stderr="",
            stdout_bytes=0,
            stderr_bytes=0,
            stdout_truncated=False,
            stderr_truncated=False,
            reason="os_error",
            error_detail=str(exc),
        )

    timed_out = False
    try:
        out_raw, err_raw = proc.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        _terminate_process_group(proc, kill_grace_s)
        try:
            tail_out, tail_err = proc.communicate(timeout=kill_grace_s)
        except subprocess.TimeoutExpired:
            tail_out, tail_err = b"", b""
        partial_out = exc.stdout if isinstance(exc.stdout, bytes) else b""
        partial_err = exc.stderr if isinstance(exc.stderr, bytes) else b""
        out_raw = partial_out + tail_out
        err_raw = partial_err + tail_err

    duration_s = round(time.monotonic() - t0, 6)
    stdout_text, stdout_bytes, stdout_trunc = _cap_output(out_raw)
    stderr_text, stderr_bytes, stderr_trunc = _cap_output(err_raw)

    if timed_out:
        return CheckOutcome(
            name=check.name,
            command=check.command,
            timeout_s=timeout_s,
            status="could_not_run",
            exit_code=None,
            duration_s=duration_s,
            stdout=stdout_text,
            stderr=stderr_text,
            stdout_bytes=stdout_bytes,
            stderr_bytes=stderr_bytes,
            stdout_truncated=stdout_trunc,
            stderr_truncated=stderr_trunc,
            reason="timeout",
            error_detail=f"check exceeded timeout of {timeout_s:g}s",
        )

    rc = proc.returncode
    if rc == 0:
        return CheckOutcome(
            name=check.name,
            command=check.command,
            timeout_s=timeout_s,
            status="passed",
            exit_code=0,
            duration_s=duration_s,
            stdout=stdout_text,
            stderr=stderr_text,
            stdout_bytes=stdout_bytes,
            stderr_bytes=stderr_bytes,
            stdout_truncated=stdout_trunc,
            stderr_truncated=stderr_trunc,
        )
    if rc < 0:
        return CheckOutcome(
            name=check.name,
            command=check.command,
            timeout_s=timeout_s,
            status="could_not_run",
            exit_code=rc,
            duration_s=duration_s,
            stdout=stdout_text,
            stderr=stderr_text,
            stdout_bytes=stdout_bytes,
            stderr_bytes=stderr_bytes,
            stdout_truncated=stdout_trunc,
            stderr_truncated=stderr_trunc,
            reason="terminated_by_signal",
            error_detail=f"process terminated by signal {-rc}",
        )
    return CheckOutcome(
        name=check.name,
        command=check.command,
        timeout_s=timeout_s,
        status="failed",
        exit_code=rc,
        duration_s=duration_s,
        stdout=stdout_text,
        stderr=stderr_text,
        stdout_bytes=stdout_bytes,
        stderr_bytes=stderr_bytes,
        stdout_truncated=stdout_trunc,
        stderr_truncated=stderr_trunc,
    )


def _resolve_task_for_checks(
    store: UsefulnessRunStore,
    record: Mapping[str, Any],
    task_or_dir: TaskSpec | Path | None,
    *,
    verify_manifest: bool,
) -> tuple[TaskSpec, Mapping[str, str] | None]:
    expected_task_id = str(record["task_id"])
    expected_task_sha = str(record["task_sha256"])

    manifest: dict[str, Any] | None = None
    if verify_manifest and store.task_manifest_path.exists():
        manifest = store.read_task_manifest()

    if task_or_dir is None:
        task_path = store.tasks_dir / expected_task_id
        task = (
            load_verified_task(task_path, manifest)[0]
            if manifest is not None
            else load_task(task_path)
        )
    elif isinstance(task_or_dir, TaskSpec):
        task = task_or_dir
        if manifest is not None:
            verify_task_against_manifest(task, manifest)
    else:
        task = (
            load_verified_task(task_or_dir, manifest)[0]
            if manifest is not None
            else load_task(task_or_dir)
        )

    if task.task_id != expected_task_id:
        raise UsefulnessError(
            f"task id mismatch: trial record names {expected_task_id!r}, got {task.task_id!r}"
        )
    if task.task_sha256 != expected_task_sha:
        raise UsefulnessError(
            f"task {task.task_id!r} changed since trial ran: "
            f"record has {expected_task_sha}, disk has {task.task_sha256}"
        )

    expected_hidden_hashes: Mapping[str, str] | None = None
    if manifest is not None:
        entry = manifest.get("tasks", {}).get(task.task_id)
        if isinstance(entry, Mapping):
            raw_hidden = entry.get("hidden_test_files")
            if isinstance(raw_hidden, Mapping):
                expected_hidden_hashes = {str(k): str(v) for k, v in raw_hidden.items()}
    return task, expected_hidden_hashes


def _is_clone_conflict_error(exc: UsefulnessError) -> bool:
    msg = str(exc)
    return (
        "path already exists in clone" in msg
        or "clone contains a symlink at hidden test path" in msg
    )


def trial_checks_path(store: UsefulnessRunStore, trial_id: str) -> Path:
    """Return the path to a trial's write-once ``checks.json`` file."""
    return store.trials_dir / trial_id / "checks.json"


def run_trial_checks(
    experiment_home: Config | Path,
    run_id: str,
    trial_id: str,
    *,
    task_or_dir: TaskSpec | Path | None = None,
    default_timeout_s: float = DEFAULT_CHECK_TIMEOUT_S,
    kill_grace_s: float = DEFAULT_KILL_GRACE_S,
    verify_manifest: bool = True,
) -> TrialChecksResult:
    """Install hidden tests on a finished trial's clone, run all task checks, and save results.

    - Runs the exact same check commands and clean environment in both arms.
    - Installs hidden tests only now (after the agent has finished).
    - Keeps every check's result in a separate write-once ``<trial_dir>/checks.json``
      and leaves ``<trial_dir>/record.json`` untouched.
    - Distinguishes ``could_not_run`` (missing binary, permission error, OS error,
      timeout, killed by signal, or clone conflict with a hidden test path) from
      ``failed`` (command exited with a non-zero exit code).
    """
    if default_timeout_s <= 0:
        raise UsefulnessError("default_timeout_s must be positive")
    if kill_grace_s <= 0:
        raise UsefulnessError("kill_grace_s must be positive")

    store = UsefulnessRunStore(experiment_home, run_id)
    trial_dir = store.trials_dir / trial_id
    record_path = trial_dir / "record.json"
    if not record_path.is_file():
        raise UsefulnessError(
            f"trial {trial_id!r} has no finished record.json at {record_path}; "
            "checks run only after a trial finishes"
        )

    checks_path = trial_checks_path(store, trial_id)
    if checks_path.exists():
        raise UsefulnessError(f"checks already recorded for trial {trial_id!r}: {checks_path}")

    record_sha256 = sha256_file(record_path)
    record = read_json_file(record_path)
    if str(record.get("trial_id")) != trial_id:
        raise UsefulnessError(
            f"trial_id mismatch in {record_path}: expected {trial_id!r}, "
            f"got {record.get('trial_id')!r}"
        )

    task, expected_hidden_hashes = _resolve_task_for_checks(
        store, record, task_or_dir, verify_manifest=verify_manifest
    )

    paths_map = record.get("paths") or {}
    clone_raw = paths_map.get("clone_dir") if isinstance(paths_map, Mapping) else None
    clone_dir = Path(str(clone_raw)) if clone_raw else (trial_dir / "repo")
    if not clone_dir.is_dir():
        raise UsefulnessError(f"trial clone directory does not exist: {clone_dir}")

    started_at = utc_now_iso()
    t_start = time.monotonic()

    hidden_installed: tuple[str, ...] = ()
    hidden_error: str | None = None
    try:
        installed_paths = install_hidden_tests(
            task, clone_dir, expected_hashes=expected_hidden_hashes
        )
        clone_root = clone_dir.resolve()
        hidden_installed = tuple(
            p.resolve().relative_to(clone_root).as_posix() for p in installed_paths
        )
    except UsefulnessError as exc:
        if not _is_clone_conflict_error(exc):
            raise
        hidden_error = str(exc)

    outcomes: list[CheckOutcome] = []
    if hidden_error is not None:
        for check in task.checks:
            timeout_s = check.timeout_s if check.timeout_s is not None else default_timeout_s
            outcomes.append(
                CheckOutcome(
                    name=check.name,
                    command=check.command,
                    timeout_s=timeout_s,
                    status="could_not_run",
                    exit_code=None,
                    duration_s=0.0,
                    stdout="",
                    stderr="",
                    stdout_bytes=0,
                    stderr_bytes=0,
                    stdout_truncated=False,
                    stderr_truncated=False,
                    reason="hidden_test_conflict",
                    error_detail=hidden_error,
                )
            )
    else:
        check_env = _sanitized_check_env()
        for check in task.checks:
            outcomes.append(
                _execute_one_check(
                    check,
                    clone_dir,
                    check_env,
                    default_timeout_s=default_timeout_s,
                    kill_grace_s=kill_grace_s,
                )
            )

    total_duration_s = round(time.monotonic() - t_start, 6)
    finished_at = utc_now_iso()

    passed_count = sum(1 for o in outcomes if o.status == "passed")
    failed_count = sum(1 for o in outcomes if o.status == "failed")
    could_not_run_count = sum(1 for o in outcomes if o.status == "could_not_run")
    if could_not_run_count > 0:
        overall_status: CheckStatus = "could_not_run"
    elif failed_count > 0:
        overall_status = "failed"
    else:
        overall_status = "passed"

    body: dict[str, Any] = {
        "schema_version": TRIAL_CHECKS_SCHEMA_VERSION,
        "trial_id": trial_id,
        "run_id": run_id,
        "task_id": task.task_id,
        "task_sha256": task.task_sha256,
        "record_sha256": record_sha256,
        "overall_status": overall_status,
        "passed_count": passed_count,
        "failed_count": failed_count,
        "could_not_run_count": could_not_run_count,
        "hidden_tests_installed": list(hidden_installed),
        "hidden_tests_error": hidden_error,
        "checks": [o.to_dict() for o in outcomes],
        "started_at": started_at,
        "finished_at": finished_at,
        "total_duration_s": total_duration_s,
    }
    checks_sha256 = write_once_json(checks_path, body)

    return TrialChecksResult(
        schema_version=TRIAL_CHECKS_SCHEMA_VERSION,
        trial_id=trial_id,
        run_id=run_id,
        task_id=task.task_id,
        task_sha256=task.task_sha256,
        record_sha256=record_sha256,
        overall_status=overall_status,
        passed_count=passed_count,
        failed_count=failed_count,
        could_not_run_count=could_not_run_count,
        hidden_tests_installed=hidden_installed,
        hidden_tests_error=hidden_error,
        checks=tuple(outcomes),
        started_at=started_at,
        finished_at=finished_at,
        total_duration_s=total_duration_s,
        checks_path=checks_path,
        checks_sha256=checks_sha256,
    )
