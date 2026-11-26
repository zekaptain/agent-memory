"""Task definitions, per-part hashing, task-set manifests, splits, and hidden tests.

A task lives in its own directory under ``<experiment_home>/tasks/<task_id>/``:

- ``task.json``: repository, commit, project, work_done_at, task_family, strata,
  and checks (non-empty list of named commands).
- ``request.md``: the request given to the agent.
- ``required_behavior.md``: the list of required behavior written before any trial.
- ``hidden_tests/`` (optional): test files that must stay out of the clone while
  the agent works and are installed only after the agent finishes.

A task manifest is written once per run, hashes every part of every included task,
records task families and excluded tasks with reason codes, and records a split
that places every calibration task strictly before every held-out task in time
with no task family in both.
"""

from __future__ import annotations

import os
import re
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..judge.store import utc_now_iso
from .store import (
    UsefulnessError,
    read_json_file,
    sha256_bytes,
    sha256_file,
    sha256_json,
)

TASK_MANIFEST_SCHEMA_VERSION = 1

_TASK_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")
_COMMIT_SHA_RE = re.compile(r"[0-9a-fA-F]{7,64}")
_DATE_ONLY_RE = re.compile(r"\d{4}-\d{2}-\d{2}")


@dataclass(frozen=True)
class CheckCommand:
    """One executable check command for a task."""

    name: str
    command: tuple[str, ...]
    timeout_s: float | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"name": self.name, "command": list(self.command)}
        if self.timeout_s is not None:
            out["timeout_s"] = self.timeout_s
        return out


@dataclass(frozen=True)
class TaskSpec:
    """Validated task definition loaded from disk."""

    task_id: str
    root: Path
    repository: str
    commit: str
    project: str
    work_done_at: datetime
    task_family: str
    strata: tuple[str, ...]
    request: str
    required_behavior: str
    checks: tuple[CheckCommand, ...]
    hidden_tests: Mapping[str, str]  # relative POSIX path -> SHA-256
    parts: Mapping[str, str]  # part name -> SHA-256
    task_sha256: str


@dataclass(frozen=True)
class ExcludedTask:
    """A task intentionally left out of the manifest's splits with a reason code."""

    task_id: str
    reason_code: str
    task_family: str | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "task_id": self.task_id,
            "reason_code": self.reason_code,
        }
        if self.task_family is not None:
            out["task_family"] = self.task_family
        return out


@dataclass(frozen=True)
class TaskSplit:
    """Validated chronological and family-disjoint calibration / held-out split."""

    calibration: tuple[str, ...]
    held_out: tuple[str, ...]
    calibration_families: tuple[str, ...]
    held_out_families: tuple[str, ...]
    latest_calibration_work_done_at: str
    earliest_held_out_work_done_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "calibration": list(self.calibration),
            "held_out": list(self.held_out),
            "calibration_families": list(self.calibration_families),
            "held_out_families": list(self.held_out_families),
            "latest_calibration_work_done_at": self.latest_calibration_work_done_at,
            "earliest_held_out_work_done_at": self.earliest_held_out_work_done_at,
        }


def parse_work_done_at(value: Any, *, field_name: str = "work_done_at") -> datetime:
    """Parse an ISO-8601 date (YYYY-MM-DD), timezone-aware timestamp, or datetime into UTC."""
    if isinstance(value, datetime):
        if value.tzinfo is None:
            raise UsefulnessError(f"{field_name} datetime must include a timezone offset")
        return value.astimezone(UTC)
    if not isinstance(value, str) or not value.strip():
        raise UsefulnessError(f"missing or empty {field_name}")
    text = value.strip()
    if _DATE_ONLY_RE.fullmatch(text):
        try:
            parsed_date = datetime.fromisoformat(text)
        except ValueError as exc:
            raise UsefulnessError(f"invalid {field_name} {value!r}") from exc
        return parsed_date.replace(tzinfo=UTC)
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise UsefulnessError(f"invalid {field_name} {value!r}") from exc
    if parsed.tzinfo is None:
        raise UsefulnessError(
            f"{field_name} {value!r} must include a timezone offset (or use YYYY-MM-DD)"
        )
    return parsed.astimezone(UTC)


def _format_utc(dt: datetime) -> str:
    return dt.astimezone(UTC).isoformat()


def _require_non_empty_str(data: Mapping[str, Any], field: str, task_id: str) -> str:
    val = data.get(field)
    if not isinstance(val, str) or not val.strip():
        raise UsefulnessError(f"task {task_id!r} is missing required part: {field}")
    return val.strip()


def _parse_strata(data: Mapping[str, Any], task_id: str) -> tuple[str, ...]:
    raw = data.get("strata")
    if not isinstance(raw, list) or not raw:
        raise UsefulnessError(f"task {task_id!r} is missing required part: strata")
    cleaned: list[str] = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, str) or not item.strip():
            raise UsefulnessError(f"task {task_id!r} has an empty stratum in strata")
        norm = item.strip()
        if norm in seen:
            raise UsefulnessError(f"task {task_id!r} has duplicate stratum {norm!r} in strata")
        seen.add(norm)
        cleaned.append(norm)
    return tuple(sorted(cleaned))


def _parse_checks(data: Mapping[str, Any], task_id: str) -> tuple[CheckCommand, ...]:
    raw = data.get("checks")
    if not isinstance(raw, Mapping):
        raise UsefulnessError(f"task {task_id!r} is missing required part: checks")
    commands = raw.get("commands")
    if not isinstance(commands, list) or not commands:
        raise UsefulnessError(f"task {task_id!r} is missing required part: checks.commands")
    parsed: list[CheckCommand] = []
    seen_names: set[str] = set()
    for index, entry in enumerate(commands):
        if not isinstance(entry, Mapping):
            raise UsefulnessError(
                f"task {task_id!r}: checks.commands[{index}] must be an object "
                "with name and command"
            )
        name = entry.get("name")
        if not isinstance(name, str) or not name.strip():
            raise UsefulnessError(
                f"task {task_id!r} is missing required part: checks.commands[{index}].name"
            )
        clean_name = name.strip()
        if clean_name in seen_names:
            raise UsefulnessError(
                f"task {task_id!r} has duplicate check command name {clean_name!r}"
            )
        seen_names.add(clean_name)

        cmd = entry.get("command")
        if (
            not isinstance(cmd, list)
            or not cmd
            or not all(isinstance(arg, str) and arg != "" for arg in cmd)
        ):
            raise UsefulnessError(
                f"task {task_id!r} is missing required part: checks.commands[{index}].command"
            )
        timeout = entry.get("timeout_s")
        timeout_val: float | None = None
        if timeout is not None:
            if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout <= 0:
                raise UsefulnessError(
                    f"task {task_id!r}: checks.commands[{index}].timeout_s must be positive"
                )
            timeout_val = float(timeout)
        parsed.append(CheckCommand(name=clean_name, command=tuple(cmd), timeout_s=timeout_val))
    return tuple(parsed)


def _scan_hidden_tests(task_dir: Path, task_id: str) -> dict[str, str]:
    hidden_root = task_dir / "hidden_tests"
    if not hidden_root.exists() and not hidden_root.is_symlink():
        return {}
    if hidden_root.is_symlink():
        raise UsefulnessError(f"task {task_id!r}: symlinks are not allowed for hidden_tests")
    if not hidden_root.is_dir():
        raise UsefulnessError(f"task {task_id!r}: hidden_tests must be a real directory")
    resolved_root = hidden_root.resolve()
    files: dict[str, str] = {}
    for root_str, dirnames, filenames in os.walk(hidden_root, followlinks=False):
        dirnames.sort()
        filenames.sort()
        cur_dir = Path(root_str)
        for dname in dirnames:
            dpath = cur_dir / dname
            if dpath.is_symlink():
                raise UsefulnessError(
                    f"task {task_id!r}: symlinks are not allowed in hidden_tests ({dpath.name})"
                )
        for fname in filenames:
            fpath = cur_dir / fname
            if fpath.is_symlink():
                raise UsefulnessError(
                    f"task {task_id!r}: symlinks are not allowed in hidden_tests ({fname})"
                )
            if not fpath.is_file():
                raise UsefulnessError(
                    f"task {task_id!r}: hidden_tests may only contain regular files ({fname})"
                )
            resolved = fpath.resolve()
            if not resolved.is_relative_to(resolved_root):
                raise UsefulnessError(
                    f"task {task_id!r}: hidden_tests file escapes directory ({fname})"
                )
            rel = fpath.relative_to(hidden_root).as_posix()
            if rel.startswith("/") or ".." in Path(rel).parts:
                raise UsefulnessError(f"task {task_id!r}: invalid hidden_tests path {rel!r}")
            files[rel] = sha256_file(fpath)
    return dict(sorted(files.items()))


def load_task(task_dir: Path) -> TaskSpec:
    """Load, validate, and hash one task directory.

    Refuses any task with a missing or invalid required part, naming the part.
    """
    if not task_dir.exists() or not task_dir.is_dir():
        raise UsefulnessError(f"task directory does not exist: {task_dir}")
    task_id = task_dir.name
    if not _TASK_ID_RE.fullmatch(task_id):
        raise UsefulnessError(
            f"bad task id {task_id!r}: letters, digits, dot, dash, underscore only"
        )

    meta_path = task_dir / "task.json"
    if not meta_path.exists():
        raise UsefulnessError(f"task {task_id!r} is missing required part: task.json")
    raw_meta_bytes = meta_path.read_bytes()
    meta = read_json_file(meta_path)

    declared_id = meta.get("task_id")
    if declared_id is not None and declared_id != task_id:
        raise UsefulnessError(
            f"task {task_id!r}: task.json task_id {declared_id!r} does not match directory name"
        )

    repository = _require_non_empty_str(meta, "repository", task_id)
    commit = _require_non_empty_str(meta, "commit", task_id)
    if not _COMMIT_SHA_RE.fullmatch(commit):
        raise UsefulnessError(
            f"task {task_id!r}: commit {commit!r} must be a fixed hex git commit SHA"
        )
    project = _require_non_empty_str(meta, "project", task_id)
    if "work_done_at" not in meta or meta.get("work_done_at") is None:
        raise UsefulnessError(f"task {task_id!r} is missing required part: work_done_at")
    work_done_at = parse_work_done_at(
        meta["work_done_at"], field_name=f"task {task_id!r} work_done_at"
    )
    task_family = _require_non_empty_str(meta, "task_family", task_id)
    strata = _parse_strata(meta, task_id)
    checks = _parse_checks(meta, task_id)

    request_path = task_dir / "request.md"
    if not request_path.exists():
        raise UsefulnessError(f"task {task_id!r} is missing required part: request (request.md)")
    request_bytes = request_path.read_bytes()
    try:
        request_text = request_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise UsefulnessError(f"task {task_id!r}: request.md is not valid UTF-8") from exc
    if not request_text.strip():
        raise UsefulnessError(f"task {task_id!r} is missing required part: request (request.md)")

    behavior_path = task_dir / "required_behavior.md"
    if not behavior_path.exists():
        raise UsefulnessError(
            f"task {task_id!r} is missing required part: required_behavior (required_behavior.md)"
        )
    behavior_bytes = behavior_path.read_bytes()
    try:
        behavior_text = behavior_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise UsefulnessError(f"task {task_id!r}: required_behavior.md is not valid UTF-8") from exc
    if not behavior_text.strip():
        raise UsefulnessError(
            f"task {task_id!r} is missing required part: required_behavior (required_behavior.md)"
        )

    hidden_tests = _scan_hidden_tests(task_dir, task_id)

    request_hash = sha256_bytes(request_bytes)
    behavior_hash = sha256_bytes(behavior_bytes)
    hidden_hash = sha256_json(hidden_tests)
    checks_hash = sha256_json(
        {
            "commands": [c.to_dict() for c in checks],
            "required_behavior": behavior_hash,
            "hidden_tests": hidden_hash,
        }
    )
    parts: dict[str, str] = {
        "repository_commit": sha256_json({"repository": repository, "commit": commit}),
        "request": request_hash,
        "checks": checks_hash,
        "required_behavior": behavior_hash,
        "hidden_tests": hidden_hash,
        "project": sha256_json(project),
        "work_done_at": sha256_json(_format_utc(work_done_at)),
        "task_family": sha256_json(task_family),
        "strata": sha256_json(list(strata)),
        "task_json": sha256_bytes(raw_meta_bytes),
    }
    task_sha256 = sha256_json({"parts": parts, "hidden_test_files": hidden_tests})

    return TaskSpec(
        task_id=task_id,
        root=task_dir.resolve(),
        repository=repository,
        commit=commit,
        project=project,
        work_done_at=work_done_at,
        task_family=task_family,
        strata=strata,
        request=request_text,
        required_behavior=behavior_text,
        checks=checks,
        hidden_tests=hidden_tests,
        parts=parts,
        task_sha256=task_sha256,
    )


def validate_split(
    tasks: Sequence[TaskSpec],
    calibration_ids: Sequence[str],
    held_out_ids: Sequence[str],
) -> TaskSplit:
    """Verify a calibration / held-out split is complete, chronological, and family-disjoint."""
    by_id = {t.task_id: t for t in tasks}
    if len(by_id) != len(tasks):
        raise UsefulnessError("duplicate task_id in task set")

    cal_set = set(calibration_ids)
    held_set = set(held_out_ids)
    if len(cal_set) != len(calibration_ids):
        raise UsefulnessError("duplicate task_id in calibration split")
    if len(held_set) != len(held_out_ids):
        raise UsefulnessError("duplicate task_id in held_out split")

    overlap_ids = sorted(cal_set & held_set)
    if overlap_ids:
        raise UsefulnessError(f"tasks assigned to both splits: {', '.join(overlap_ids)}")

    unknown = sorted((cal_set | held_set) - set(by_id))
    if unknown:
        raise UsefulnessError(f"split names unknown tasks: {', '.join(unknown)}")

    missing = sorted(set(by_id) - (cal_set | held_set))
    if missing:
        raise UsefulnessError(f"tasks not assigned to any split: {', '.join(missing)}")

    if not cal_set:
        raise UsefulnessError("split must include at least one calibration task")
    if not held_set:
        raise UsefulnessError("split must include at least one held-out task")

    cal_tasks = [by_id[tid] for tid in sorted(cal_set)]
    held_tasks = [by_id[tid] for tid in sorted(held_set)]

    cal_families = {t.task_family for t in cal_tasks}
    held_families = {t.task_family for t in held_tasks}
    shared_families = sorted(cal_families & held_families)
    if shared_families:
        raise UsefulnessError(
            "split places the same task family in both calibration and held_out: "
            + ", ".join(shared_families)
        )

    latest_cal = max(cal_tasks, key=lambda t: t.work_done_at)
    earliest_held = min(held_tasks, key=lambda t: t.work_done_at)
    if not (latest_cal.work_done_at < earliest_held.work_done_at):
        raise UsefulnessError(
            "split violates chronological order: calibration task "
            f"{latest_cal.task_id!r} ({_format_utc(latest_cal.work_done_at)}) is not strictly "
            f"before held-out task {earliest_held.task_id!r} "
            f"({_format_utc(earliest_held.work_done_at)})"
        )

    return TaskSplit(
        calibration=tuple(sorted(cal_set)),
        held_out=tuple(sorted(held_set)),
        calibration_families=tuple(sorted(cal_families)),
        held_out_families=tuple(sorted(held_families)),
        latest_calibration_work_done_at=_format_utc(latest_cal.work_done_at),
        earliest_held_out_work_done_at=_format_utc(earliest_held.work_done_at),
    )


def split_tasks(
    tasks: Sequence[TaskSpec],
    *,
    calibration_fraction: float | None = None,
    split_after: str | datetime | None = None,
) -> TaskSplit:
    """Split ``tasks`` into calibration and held-out sets by time and task family.

    Every calibration task is strictly earlier in ``work_done_at`` than every
    held-out task, and no task family appears in both splits. Refuses any task
    set that admits no such split.
    """
    if calibration_fraction is not None and split_after is not None:
        raise UsefulnessError("specify at most one of calibration_fraction or split_after")
    if not tasks:
        raise UsefulnessError("cannot split an empty task set")

    by_family: dict[str, list[TaskSpec]] = {}
    for task in tasks:
        by_family.setdefault(task.task_family, []).append(task)
    if len(by_family) < 2:
        raise UsefulnessError(
            "cannot split task set: at least two distinct task families are required"
        )

    if split_after is not None:
        boundary = parse_work_done_at(split_after, field_name="split_after")
        cal_ids: list[str] = []
        held_ids: list[str] = []
        for fam, fam_tasks in sorted(by_family.items()):
            earliest = min(t.work_done_at for t in fam_tasks)
            latest = max(t.work_done_at for t in fam_tasks)
            if latest <= boundary:
                cal_ids.extend(t.task_id for t in fam_tasks)
            elif earliest > boundary:
                held_ids.extend(t.task_id for t in fam_tasks)
            else:
                raise UsefulnessError(
                    f"task family {fam!r} straddles split_after {_format_utc(boundary)}"
                )
        return validate_split(tasks, cal_ids, held_ids)

    target_fraction = 0.3 if calibration_fraction is None else float(calibration_fraction)
    if isinstance(calibration_fraction, bool) or not (0.0 < target_fraction < 1.0):
        raise UsefulnessError("calibration_fraction must be strictly between 0 and 1")

    # Every valid chronological + family-disjoint split is a contiguous prefix
    # when families are ordered by their earliest task timestamp.
    ordered_families = sorted(
        by_family.items(),
        key=lambda item: (
            min(t.work_done_at for t in item[1]),
            max(t.work_done_at for t in item[1]),
            item[0],
        ),
    )
    g = len(ordered_families)
    mins = [min(t.work_done_at for t in fam_tasks) for _, fam_tasks in ordered_families]
    maxs = [max(t.work_done_at for t in fam_tasks) for _, fam_tasks in ordered_families]
    sizes = [len(fam_tasks) for _, fam_tasks in ordered_families]

    prefix_max: list[datetime] = []
    running_max = maxs[0]
    for m in maxs:
        if m > running_max:
            running_max = m
        prefix_max.append(running_max)

    suffix_min: list[datetime] = [mins[-1]] * g
    running_min = mins[-1]
    for idx in range(g - 1, -1, -1):
        if mins[idx] < running_min:
            running_min = mins[idx]
        suffix_min[idx] = running_min

    total_tasks = len(tasks)
    valid_cuts: list[tuple[float, int, int]] = []
    cal_count = 0
    for cut in range(1, g):
        cal_count += sizes[cut - 1]
        if prefix_max[cut - 1] < suffix_min[cut]:
            distance = abs((cal_count / total_tasks) - target_fraction)
            valid_cuts.append((distance, cal_count, cut))

    if not valid_cuts:
        raise UsefulnessError(
            "cannot split task set: overlapping task-family time spans or tied timestamps "
            "leave no chronological family-disjoint boundary"
        )

    valid_cuts.sort()
    best_cut = valid_cuts[0][2]
    cal_ids = [t.task_id for _, fam_tasks in ordered_families[:best_cut] for t in fam_tasks]
    held_ids = [t.task_id for _, fam_tasks in ordered_families[best_cut:] for t in fam_tasks]
    return validate_split(tasks, cal_ids, held_ids)


def _read_excluded_metadata(task_dir: Path) -> tuple[str | None, str | None]:
    meta_path = task_dir / "task.json"
    if not meta_path.exists():
        return None, None
    meta = read_json_file(meta_path)
    reason = meta.get("exclude_reason")
    family = meta.get("task_family")
    clean_reason = reason.strip() if isinstance(reason, str) and reason.strip() else None
    clean_family = family.strip() if isinstance(family, str) and family.strip() else None
    return clean_reason, clean_family


def load_task_set(
    tasks_dir: Path,
    *,
    excluded: Mapping[str, str] | None = None,
) -> tuple[list[TaskSpec], list[ExcludedTask]]:
    """Load every task directory under ``tasks_dir``, separating excluded tasks."""
    if not tasks_dir.exists() or not tasks_dir.is_dir():
        raise UsefulnessError(f"tasks directory does not exist: {tasks_dir}")

    excluded_map: dict[str, str] = {}
    if excluded is not None:
        for tid, reason in excluded.items():
            if not isinstance(tid, str) or not _TASK_ID_RE.fullmatch(tid):
                raise UsefulnessError(f"bad excluded task id {tid!r}")
            if not isinstance(reason, str) or not reason.strip():
                raise UsefulnessError(f"excluded task {tid!r} must have a non-empty reason_code")
            excluded_map[tid] = reason.strip()

    task_dirs = [
        p for p in sorted(tasks_dir.iterdir()) if p.is_dir() and not p.name.startswith(".")
    ]
    seen_dirs = {p.name for p in task_dirs}

    included: list[TaskSpec] = []
    excluded_rows: list[ExcludedTask] = []

    for task_dir in task_dirs:
        tid = task_dir.name
        inline_reason, inline_family = _read_excluded_metadata(task_dir)
        explicit_reason = excluded_map.get(tid)
        if (
            explicit_reason is not None
            and inline_reason is not None
            and explicit_reason != inline_reason
        ):
            raise UsefulnessError(
                f"task {tid!r} has conflicting exclusion reasons: "
                f"{explicit_reason!r} vs {inline_reason!r}"
            )
        reason_code = explicit_reason or inline_reason
        if reason_code is not None:
            excluded_rows.append(
                ExcludedTask(task_id=tid, reason_code=reason_code, task_family=inline_family)
            )
            continue
        included.append(load_task(task_dir))

    # Also allow listing excluded task ids whose directories were omitted or removed,
    # keeping their reason codes in the manifest.
    for tid in sorted(set(excluded_map) - seen_dirs):
        excluded_rows.append(ExcludedTask(task_id=tid, reason_code=excluded_map[tid]))

    excluded_rows.sort(key=lambda row: row.task_id)
    return included, excluded_rows


def build_task_manifest(
    tasks_dir: Path,
    *,
    excluded: Mapping[str, str] | None = None,
    calibration_ids: Sequence[str] | None = None,
    held_out_ids: Sequence[str] | None = None,
    calibration_fraction: float | None = None,
    split_after: str | datetime | None = None,
    cutoff: str | datetime | None = None,
    created_at: str | None = None,
) -> dict[str, Any]:
    """Validate, split, and hash a task set into a task manifest object."""
    included, excluded_rows = load_task_set(tasks_dir, excluded=excluded)
    if not included:
        raise UsefulnessError(f"no included tasks found under {tasks_dir}")

    cutoff_dt: datetime | None = None
    if cutoff is not None:
        cutoff_dt = parse_work_done_at(cutoff, field_name="cutoff")
        before_cutoff = [t.task_id for t in included if not (t.work_done_at > cutoff_dt)]
        if before_cutoff:
            raise UsefulnessError(
                f"tasks on or before cutoff {_format_utc(cutoff_dt)}: {', '.join(before_cutoff)}"
            )

    if calibration_ids is not None or held_out_ids is not None:
        if calibration_ids is None or held_out_ids is None:
            raise UsefulnessError("specify both calibration_ids and held_out_ids together")
        split = validate_split(included, calibration_ids, held_out_ids)
    else:
        split = split_tasks(
            included,
            calibration_fraction=calibration_fraction,
            split_after=split_after,
        )

    split_by_task = {tid: "calibration" for tid in split.calibration}
    split_by_task.update({tid: "held_out" for tid in split.held_out})

    tasks_obj: dict[str, dict[str, Any]] = {}
    families_obj: dict[str, dict[str, Any]] = {}
    by_family: dict[str, list[TaskSpec]] = {}

    for task in sorted(included, key=lambda t: t.task_id):
        by_family.setdefault(task.task_family, []).append(task)
        tasks_obj[task.task_id] = {
            "task_id": task.task_id,
            "split": split_by_task[task.task_id],
            "task_family": task.task_family,
            "strata": list(task.strata),
            "project": task.project,
            "repository": task.repository,
            "commit": task.commit,
            "work_done_at": _format_utc(task.work_done_at),
            "check_names": [c.name for c in task.checks],
            "hidden_test_files": dict(task.hidden_tests),
            "parts": dict(task.parts),
            "task_sha256": task.task_sha256,
        }

    for family, fam_tasks in sorted(by_family.items()):
        fam_split = split_by_task[fam_tasks[0].task_id]
        families_obj[family] = {
            "split": fam_split,
            "task_ids": sorted(t.task_id for t in fam_tasks),
            "earliest_work_done_at": _format_utc(min(t.work_done_at for t in fam_tasks)),
            "latest_work_done_at": _format_utc(max(t.work_done_at for t in fam_tasks)),
        }

    return {
        "schema_version": TASK_MANIFEST_SCHEMA_VERSION,
        "created_at": created_at if created_at is not None else utc_now_iso(),
        "cutoff": _format_utc(cutoff_dt) if cutoff_dt is not None else None,
        "tasks": tasks_obj,
        "families": families_obj,
        "excluded": [row.to_dict() for row in excluded_rows],
        "split": split.to_dict(),
    }


def verify_task_against_manifest(task: TaskSpec, manifest: Mapping[str, Any]) -> str:
    """Verify ``task`` matches its entry in ``manifest`` and return its split name.

    Refuses any task missing from the manifest, listed as excluded, or changed
    in any part or hidden test file.
    """
    tasks_map = manifest.get("tasks")
    if not isinstance(tasks_map, Mapping):
        raise UsefulnessError("invalid task manifest: missing tasks mapping")

    excluded_list = manifest.get("excluded") or []
    excluded_ids = {
        row["task_id"]
        for row in excluded_list
        if isinstance(row, Mapping) and isinstance(row.get("task_id"), str)
    }
    if task.task_id in excluded_ids:
        raise UsefulnessError(f"task {task.task_id!r} is excluded by the task manifest")

    entry = tasks_map.get(task.task_id)
    if not isinstance(entry, Mapping):
        raise UsefulnessError(f"task {task.task_id!r} is not listed in the task manifest")

    expected_parts = entry.get("parts")
    if not isinstance(expected_parts, Mapping):
        raise UsefulnessError(
            f"invalid task manifest entry for {task.task_id!r}: missing parts hashes"
        )

    changed: list[str] = []
    for part_name in sorted(set(expected_parts) | set(task.parts)):
        if expected_parts.get(part_name) != task.parts.get(part_name):
            changed.append(part_name)

    expected_hidden = entry.get("hidden_test_files")
    if (
        not isinstance(expected_hidden, Mapping) or dict(expected_hidden) != dict(task.hidden_tests)
    ) and "hidden_tests" not in changed:
        changed.append("hidden_tests")

    if entry.get("task_sha256") != task.task_sha256:
        changed.append("task_sha256")

    if changed:
        raise UsefulnessError(
            f"task {task.task_id!r} no longer matches manifest (changed: {', '.join(changed)})"
        )

    split_name = entry.get("split")
    if split_name not in ("calibration", "held_out"):
        raise UsefulnessError(f"invalid task manifest entry for {task.task_id!r}: bad split")
    return str(split_name)


def load_verified_task(task_dir: Path, manifest: Mapping[str, Any]) -> tuple[TaskSpec, str]:
    """Load a task from ``task_dir`` and verify every part against ``manifest``."""
    task = load_task(task_dir)
    split_name = verify_task_against_manifest(task, manifest)
    return task, split_name


def _safe_clone_target(clone_dir: Path, rel_posix: str) -> Path:
    clone_root = clone_dir.resolve()
    rel = Path(rel_posix)
    if rel.is_absolute() or ".." in rel.parts or not rel.parts:
        raise UsefulnessError(f"unsafe hidden test target path: {rel_posix!r}")
    # Check each existing prefix is not a symlink escaping clone_dir.
    cur = clone_root
    for part in rel.parts:
        cur = cur / part
        if cur.is_symlink():
            raise UsefulnessError(f"clone contains a symlink at hidden test path {rel_posix!r}")
    resolved = cur.resolve()
    if not resolved.is_relative_to(clone_root):
        raise UsefulnessError(f"hidden test target escapes clone directory: {rel_posix!r}")
    return cur


def assert_hidden_tests_absent(task: TaskSpec, clone_dir: Path) -> None:
    """Refuse to start an agent if any of ``task``'s hidden tests exist in ``clone_dir``."""
    if not clone_dir.exists() or not clone_dir.is_dir():
        raise UsefulnessError(f"clone directory does not exist: {clone_dir}")
    present: list[str] = []
    for rel in sorted(task.hidden_tests):
        target = _safe_clone_target(clone_dir, rel)
        if target.exists() or target.is_symlink():
            present.append(rel)
    if present:
        raise UsefulnessError(
            f"task {task.task_id!r}: hidden tests must not be in the clone while the agent "
            f"works (found: {', '.join(present)})"
        )


def install_hidden_tests(
    task: TaskSpec,
    clone_dir: Path,
    *,
    expected_hashes: Mapping[str, str] | None = None,
) -> list[Path]:
    """Copy ``task``'s hidden tests into ``clone_dir`` after the agent has finished.

    Verifies every hidden test file's SHA-256 against ``task.hidden_tests`` (and
    ``expected_hashes`` when given) and refuses to overwrite files already in
    the clone.
    """
    if not clone_dir.exists() or not clone_dir.is_dir():
        raise UsefulnessError(f"clone directory does not exist: {clone_dir}")
    if not task.hidden_tests:
        return []

    hidden_root = task.root / "hidden_tests"
    if hidden_root.is_symlink() or not hidden_root.is_dir():
        raise UsefulnessError(f"task {task.task_id!r}: hidden_tests must be a real directory")
    resolved_hidden_root = hidden_root.resolve()
    hashes_to_check = expected_hashes if expected_hashes is not None else task.hidden_tests
    if dict(hashes_to_check) != dict(task.hidden_tests):
        raise UsefulnessError(
            f"task {task.task_id!r}: hidden_tests do not match expected manifest hashes"
        )

    staged: list[tuple[Path, Path]] = []
    for rel, expected_sha in sorted(task.hidden_tests.items()):
        src = hidden_root / rel
        if src.is_symlink() or not src.is_file():
            raise UsefulnessError(
                f"task {task.task_id!r}: hidden test missing or not a file: {rel}"
            )
        if not src.resolve().is_relative_to(resolved_hidden_root):
            raise UsefulnessError(f"task {task.task_id!r}: hidden test escapes directory: {rel}")
        actual_sha = sha256_file(src)
        if actual_sha != expected_sha:
            raise UsefulnessError(
                f"task {task.task_id!r}: hidden test {rel!r} changed on disk since manifest"
            )
        dst = _safe_clone_target(clone_dir, rel)
        if dst.exists() or dst.is_symlink():
            raise UsefulnessError(
                f"task {task.task_id!r}: cannot install hidden test {rel!r}, path already exists "
                "in clone"
            )
        staged.append((src, dst))

    installed: list[Path] = []
    for src, dst in staged:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        installed.append(dst)
    return installed
