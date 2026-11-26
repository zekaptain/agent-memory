"""Deterministic pair scheduling, resumable execution, and retry policy (MEM-22).

Fixes the order of pairs and the order of the two arms within each pair from a
registered integer seed and a run's task manifest. Persists the schedule once
in ``schedule.json`` and tracks execution progress in an append-only
``schedule-events.jsonl`` log so interrupted runs continue with unmade trials
while keeping and marking interrupted trials without overwriting them in place.
Whether a pair is run again as a new attempt is decided strictly by the run's
registered ``RetryRule``.
"""

from __future__ import annotations

import hashlib
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from ..config import Config
from ..judge.store import utc_now_iso
from .runner import (
    DEFAULT_AUTO_LEVEL,
    DEFAULT_BASE_TOOLS,
    DEFAULT_DROID_MODEL,
    DEFAULT_MEMORY_SERVER_NAME,
    DEFAULT_WALL_TIME_LIMIT_S,
    FAILURE_CATEGORIES,
    VALID_ARMS,
    Arm,
    DroidHarnessAdapter,
    HarnessAdapter,
    TrialConfig,
    TrialRecord,
    check_trial_egress,
    resolve_trial_registration,
    run_trial,
)
from .store import (
    UsefulnessError,
    UsefulnessRunStore,
    ensure_experiment_home,
    read_json_file,
    sha256_file,
    write_once_json,
)
from .tasks import load_verified_task

SCHEDULE_SCHEMA_VERSION = 1
INTERRUPTED_MARKER_SCHEMA_VERSION = 1

SplitFilter = Literal["calibration", "held_out", "all"]
VALID_SPLIT_FILTERS: tuple[SplitFilter, ...] = ("calibration", "held_out", "all")

SlotStatus = Literal["pending", "completed", "failed", "interrupted"]
RetryDecision = Literal["pending", "not_needed", "scheduled", "declined"]


def _require_int(value: Any, name: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise UsefulnessError(f"{name} must be an integer, got {value!r}")
    if minimum is not None and value < minimum:
        raise UsefulnessError(f"{name} must be >= {minimum}, got {value}")
    return value


def _validate_arm_order(raw: Sequence[Any]) -> tuple[Arm, Arm]:
    if len(raw) != 2:
        raise UsefulnessError(f"arm_order must contain exactly 2 arms, got {list(raw)!r}")
    a, b = raw[0], raw[1]
    if a not in VALID_ARMS or b not in VALID_ARMS or a == b:
        raise UsefulnessError(f"arm_order must be a permutation of {VALID_ARMS}, got {list(raw)!r}")
    arm_a: Arm = "memory" if a == "memory" else "no_memory"
    arm_b: Arm = "memory" if b == "memory" else "no_memory"
    return (arm_a, arm_b)


def _draw_arm_order(rng: random.Random) -> tuple[Arm, Arm]:
    if rng.random() < 0.5:
        return ("memory", "no_memory")
    return ("no_memory", "memory")


@dataclass(frozen=True)
class RetryRule:
    """Explicit rule deciding whether a pair attempt is run again as a new attempt.

    Defaults to never retrying (``max_pair_retries=0``). When retries are
    enabled, only pair attempts whose every non-completed arm has a failure
    category listed in ``retryable_failure_categories`` qualify, up to
    ``max_pair_retries`` additional attempts beyond attempt 1.
    """

    max_pair_retries: int = 0
    retryable_failure_categories: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        max_retries = _require_int(self.max_pair_retries, "max_pair_retries", minimum=0)
        cats: list[str] = []
        seen: set[str] = set()
        for raw_cat in self.retryable_failure_categories:
            if not isinstance(raw_cat, str) or raw_cat not in FAILURE_CATEGORIES:
                raise UsefulnessError(
                    f"invalid retryable failure category {raw_cat!r}: "
                    f"expected one of {FAILURE_CATEGORIES}"
                )
            if raw_cat not in seen:
                seen.add(raw_cat)
                cats.append(raw_cat)
        normalized = tuple(sorted(cats))
        if max_retries > 0 and not normalized:
            raise UsefulnessError(
                "retry rule with max_pair_retries > 0 must declare at least one "
                "retryable_failure_categories entry"
            )
        object.__setattr__(self, "max_pair_retries", max_retries)
        object.__setattr__(self, "retryable_failure_categories", normalized)

    def evaluate(
        self,
        attempt: int,
        arm_outcomes: Sequence[tuple[str, str | None]],
    ) -> tuple[bool, str, tuple[str, ...]]:
        """Evaluate whether a finished pair attempt should be retried.

        Returns ``(should_retry, reason, failing_categories)``.
        """
        _require_int(attempt, "attempt", minimum=1)
        non_completed = [
            (status, failure_category)
            for status, failure_category in arm_outcomes
            if status != "completed"
        ]
        if not non_completed:
            return (False, "all_arms_completed", ())

        failing_categories = tuple(
            sorted({cat if cat is not None else "unknown" for _, cat in non_completed})
        )
        if self.max_pair_retries == 0:
            return (False, "retries_disabled", failing_categories)
        if attempt > self.max_pair_retries:
            return (False, "max_pair_retries_reached", failing_categories)

        retryable = set(self.retryable_failure_categories)
        if all(cat in retryable for _, cat in non_completed):
            return (True, "retryable_failure", failing_categories)
        return (False, "non_retryable_failure_category", failing_categories)

    def should_retry(
        self,
        attempt: int,
        arm_outcomes: Sequence[tuple[str, str | None]],
    ) -> bool:
        """Return True iff the pair attempt qualifies for another attempt under this rule."""
        retry, _, _ = self.evaluate(attempt, arm_outcomes)
        return retry

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_pair_retries": self.max_pair_retries,
            "retryable_failure_categories": list(self.retryable_failure_categories),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RetryRule:
        if not isinstance(data, Mapping):
            raise UsefulnessError("invalid retry_rule: expected a JSON object")
        raw_cats = data.get("retryable_failure_categories", ())
        if not isinstance(raw_cats, list | tuple):
            raise UsefulnessError("invalid retry_rule.retryable_failure_categories: expected list")
        return cls(
            max_pair_retries=data.get("max_pair_retries", 0),
            retryable_failure_categories=tuple(raw_cats),
        )


@dataclass(frozen=True)
class ScheduledPair:
    """One scheduled (task, repeat) pair with its fixed attempt-1 arm order."""

    pair_index: int
    pair_id: str
    task_id: str
    task_family: str
    task_sha256: str
    split: str
    repeat_index: int
    arm_order: tuple[Arm, Arm]

    def to_dict(self) -> dict[str, Any]:
        return {
            "pair_index": self.pair_index,
            "pair_id": self.pair_id,
            "task_id": self.task_id,
            "task_family": self.task_family,
            "task_sha256": self.task_sha256,
            "split": self.split,
            "repeat_index": self.repeat_index,
            "arm_order": list(self.arm_order),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ScheduledPair:
        if not isinstance(data, Mapping):
            raise UsefulnessError("invalid scheduled pair: expected a JSON object")
        arm_order = _validate_arm_order(data.get("arm_order", ()))
        return cls(
            pair_index=_require_int(data.get("pair_index"), "pair_index", minimum=0),
            pair_id=str(data["pair_id"]),
            task_id=str(data["task_id"]),
            task_family=str(data["task_family"]),
            task_sha256=str(data["task_sha256"]),
            split=str(data["split"]),
            repeat_index=_require_int(data.get("repeat_index"), "repeat_index", minimum=0),
            arm_order=arm_order,
        )


@dataclass(frozen=True)
class RunSchedule:
    """Immutable seeded schedule of pairs for one usefulness run."""

    schema_version: int
    created_at: str
    run_id: str
    seed: int
    split: SplitFilter
    task_manifest_sha256: str
    default_repeats: int
    task_repeats: Mapping[str, int]
    retry_rule: RetryRule
    pairs: tuple[ScheduledPair, ...]

    def pair_by_id(self, pair_id: str) -> ScheduledPair:
        for pair in self.pairs:
            if pair.pair_id == pair_id:
                return pair
        raise UsefulnessError(f"unknown pair_id {pair_id!r} in schedule for run {self.run_id!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "created_at": self.created_at,
            "run_id": self.run_id,
            "seed": self.seed,
            "split": self.split,
            "task_manifest_sha256": self.task_manifest_sha256,
            "default_repeats": self.default_repeats,
            "task_repeats": dict(sorted(self.task_repeats.items())),
            "retry_rule": self.retry_rule.to_dict(),
            "pairs": [p.to_dict() for p in self.pairs],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RunSchedule:
        if not isinstance(data, Mapping):
            raise UsefulnessError("invalid schedule.json: expected a JSON object")
        version = _require_int(data.get("schema_version"), "schema_version", minimum=1)
        if version != SCHEDULE_SCHEMA_VERSION:
            raise UsefulnessError(
                f"unsupported schedule schema_version {version}; expected {SCHEDULE_SCHEMA_VERSION}"
            )
        raw_split = data.get("split")
        if raw_split not in VALID_SPLIT_FILTERS:
            raise UsefulnessError(
                f"invalid schedule split {raw_split!r}: expected one of {VALID_SPLIT_FILTERS}"
            )
        split: SplitFilter
        if raw_split == "calibration":
            split = "calibration"
        elif raw_split == "held_out":
            split = "held_out"
        else:
            split = "all"

        raw_task_repeats = data.get("task_repeats", {})
        if not isinstance(raw_task_repeats, Mapping):
            raise UsefulnessError("invalid schedule task_repeats: expected a JSON object")
        task_repeats = {
            str(k): _require_int(v, f"task_repeats[{k!r}]", minimum=1)
            for k, v in sorted(raw_task_repeats.items())
        }

        raw_pairs = data.get("pairs")
        if not isinstance(raw_pairs, list) or not raw_pairs:
            raise UsefulnessError("invalid schedule pairs: expected a non-empty list")
        pairs = tuple(ScheduledPair.from_dict(item) for item in raw_pairs)

        return cls(
            schema_version=version,
            created_at=str(data["created_at"]),
            run_id=str(data["run_id"]),
            seed=_require_int(data.get("seed"), "seed"),
            split=split,
            task_manifest_sha256=str(data["task_manifest_sha256"]),
            default_repeats=_require_int(data.get("default_repeats"), "default_repeats", minimum=1),
            task_repeats=task_repeats,
            retry_rule=RetryRule.from_dict(data.get("retry_rule", {})),
            pairs=pairs,
        )


def pair_id_for_repeat(task_id: str, repeat_index: int) -> str:
    """Canonical pair identifier for ``(task_id, repeat_index)``."""
    idx = _require_int(repeat_index, "repeat_index", minimum=0)
    return f"{task_id}__r{idx + 1}"


def trial_id_for_slot(pair_id: str, attempt: int, arm: Arm) -> str:
    """Canonical trial identifier for ``(pair_id, attempt, arm)``.

    Includes ``attempt`` so retrying a pair never collides with or overwrites an
    earlier attempt's trial directory or record.
    """
    att = _require_int(attempt, "attempt", minimum=1)
    if arm not in VALID_ARMS:
        raise UsefulnessError(f"invalid arm {arm!r}: expected one of {VALID_ARMS}")
    return f"{pair_id}__a{att}__{arm}"


def arm_order_for_attempt(seed: int, pair: ScheduledPair, attempt: int) -> tuple[Arm, Arm]:
    """Deterministic within-pair arm order for ``attempt``.

    Attempt 1 uses ``pair.arm_order`` drawn directly when building the schedule.
    Retry attempts (``attempt >= 2``) derive an independent deterministic RNG
    from ``(seed, pair.pair_id, attempt)`` so retrying one pair never shifts the
    arm order of any other pair.
    """
    _require_int(seed, "seed")
    att = _require_int(attempt, "attempt", minimum=1)
    if att == 1:
        return pair.arm_order
    material = f"{seed}:{pair.pair_id}:attempt:{att}".encode()
    derived_seed = int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
    return _draw_arm_order(random.Random(derived_seed))


def build_schedule(
    task_manifest: Mapping[str, Any],
    *,
    run_id: str,
    seed: int,
    task_manifest_sha256: str,
    split: SplitFilter = "held_out",
    repeats: int = 1,
    task_repeats: Mapping[str, int] | None = None,
    retry_rule: RetryRule | None = None,
    created_at: str | None = None,
) -> RunSchedule:
    """Build a deterministic ``RunSchedule`` from ``task_manifest`` and ``seed``.

    The same ``seed``, ``task_manifest``, ``split``, and repeat counts always
    produce the exact same order of pairs and within-pair arm orders, regardless
    of dictionary insertion order in the manifest.
    """
    _require_int(seed, "seed")
    if split not in VALID_SPLIT_FILTERS:
        raise UsefulnessError(f"invalid split {split!r}: expected one of {VALID_SPLIT_FILTERS}")
    default_repeats = _require_int(repeats, "repeats", minimum=1)
    if not isinstance(task_manifest_sha256, str) or not task_manifest_sha256.strip():
        raise UsefulnessError("task_manifest_sha256 must be a non-empty string")

    tasks_map = task_manifest.get("tasks")
    if not isinstance(tasks_map, Mapping) or not tasks_map:
        raise UsefulnessError("task manifest has no tasks")

    selected_entries: list[tuple[str, Mapping[str, Any]]] = []
    for task_id in sorted(tasks_map):
        entry = tasks_map[task_id]
        if not isinstance(entry, Mapping):
            raise UsefulnessError(f"invalid task manifest entry for {task_id!r}")
        entry_split = entry.get("split")
        if not isinstance(entry_split, str):
            raise UsefulnessError(f"task {task_id!r} in manifest is missing its split")
        if split == "all" or entry_split == split:
            selected_entries.append((str(task_id), entry))

    if not selected_entries:
        raise UsefulnessError(f"task manifest has no tasks in split {split!r}")

    selected_ids = {tid for tid, _ in selected_entries}
    overrides: dict[str, int] = {}
    if task_repeats is not None:
        for tid, count in sorted(task_repeats.items()):
            if tid not in selected_ids:
                raise UsefulnessError(
                    f"task_repeats names task {tid!r} which is not in split {split!r}"
                )
            overrides[tid] = _require_int(count, f"task_repeats[{tid!r}]", minimum=1)

    # Canonical sorted list of (task_id, repeat_index, manifest_entry) before seeded shuffle.
    canonical_items: list[tuple[str, int, Mapping[str, Any]]] = []
    for task_id, entry in selected_entries:
        count = overrides.get(task_id, default_repeats)
        for repeat_index in range(count):
            canonical_items.append((task_id, repeat_index, entry))

    rng = random.Random(seed)
    rng.shuffle(canonical_items)

    pairs: list[ScheduledPair] = []
    for pair_index, (task_id, repeat_index, entry) in enumerate(canonical_items):
        arm_order = _draw_arm_order(rng)
        pairs.append(
            ScheduledPair(
                pair_index=pair_index,
                pair_id=pair_id_for_repeat(task_id, repeat_index),
                task_id=task_id,
                task_family=str(entry.get("task_family", "")),
                task_sha256=str(entry.get("task_sha256", "")),
                split=str(entry["split"]),
                repeat_index=repeat_index,
                arm_order=arm_order,
            )
        )

    effective_rule = retry_rule if retry_rule is not None else RetryRule()
    return RunSchedule(
        schema_version=SCHEDULE_SCHEMA_VERSION,
        created_at=created_at if created_at is not None else utc_now_iso(),
        run_id=run_id,
        seed=seed,
        split=split,
        task_manifest_sha256=task_manifest_sha256,
        default_repeats=default_repeats,
        task_repeats=overrides,
        retry_rule=effective_rule,
        pairs=tuple(pairs),
    )


def create_run_schedule(
    experiment_home: Config | Path,
    run_id: str,
    *,
    seed: int,
    split: SplitFilter = "held_out",
    repeats: int = 1,
    task_repeats: Mapping[str, int] | None = None,
    retry_rule: RetryRule | None = None,
) -> tuple[RunSchedule, str]:
    """Create and persist a write-once ``schedule.json`` for ``run_id``.

    Requires ``task-manifest.json`` to already exist in the run directory.
    Returns ``(schedule, schedule_sha256)``.
    """
    store = UsefulnessRunStore(experiment_home, run_id)
    manifest = store.read_task_manifest()
    manifest_sha = store.task_manifest_sha256()
    schedule = build_schedule(
        manifest,
        run_id=run_id,
        seed=seed,
        task_manifest_sha256=manifest_sha,
        split=split,
        repeats=repeats,
        task_repeats=task_repeats,
        retry_rule=retry_rule,
    )
    schedule_sha = store.write_schedule(schedule.to_dict())
    return (schedule, schedule_sha)


def load_run_schedule(
    experiment_home: Config | Path,
    run_id: str,
    *,
    verify_manifest: bool = True,
) -> tuple[RunSchedule, str]:
    """Load ``schedule.json`` for ``run_id`` and optionally verify the task manifest hash."""
    store = UsefulnessRunStore(experiment_home, run_id)
    schedule = RunSchedule.from_dict(store.read_schedule())
    if schedule.run_id != run_id:
        raise UsefulnessError(
            f"schedule run_id {schedule.run_id!r} does not match requested run {run_id!r}"
        )
    if verify_manifest:
        actual_manifest_sha = store.task_manifest_sha256()
        if actual_manifest_sha != schedule.task_manifest_sha256:
            raise UsefulnessError(
                f"task-manifest.json SHA-256 ({actual_manifest_sha}) does not match "
                f"schedule's recorded task_manifest_sha256 ({schedule.task_manifest_sha256})"
            )
    return (schedule, store.schedule_sha256())


# ---------------------------------------------------------------------------
# Resumable execution state, interruption marking, and retry queue
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScheduleRunConfig:
    """Run-level settings passed to ``run_trial`` for every scheduled trial."""

    model: str = DEFAULT_DROID_MODEL
    reasoning_effort: str | None = None
    wall_time_limit_s: float = DEFAULT_WALL_TIME_LIMIT_S
    auto_level: str = DEFAULT_AUTO_LEVEL
    base_tools: tuple[str, ...] = DEFAULT_BASE_TOOLS
    memory_server_name: str = DEFAULT_MEMORY_SERVER_NAME
    model_base_url: str | None = None
    source_home: Path | None = None
    kill_grace_s: float = 10.0
    memory_tool_wording: Mapping[str, Any] | None = None
    memory_tool_wording_file: Path | None = None
    repo_commit: str | None = None
    hardware_class: str | None = None
    harness_version: str | None = None

    def to_trial_config(self, *, arm: Arm, trial_id: str) -> TrialConfig:
        return TrialConfig(
            arm=arm,
            model=self.model,
            reasoning_effort=self.reasoning_effort,
            wall_time_limit_s=self.wall_time_limit_s,
            auto_level=self.auto_level,
            base_tools=self.base_tools,
            memory_server_name=self.memory_server_name,
            model_base_url=self.model_base_url,
            source_home=self.source_home if arm == "memory" else None,
            verify_manifest=True,
            trial_id=trial_id,
            kill_grace_s=self.kill_grace_s,
            memory_tool_wording=self.memory_tool_wording,
            memory_tool_wording_file=self.memory_tool_wording_file,
            repo_commit=self.repo_commit,
            hardware_class=self.hardware_class,
            harness_version=self.harness_version,
        )


@dataclass(frozen=True)
class TrialSlotState:
    """Reconciled state of one scheduled ``(pair_id, attempt, arm)`` trial slot."""

    pair_id: str
    task_id: str
    repeat_index: int
    attempt: int
    arm: Arm
    trial_id: str
    status: SlotStatus
    failure_category: str | None = None
    record_path: Path | None = None
    record_sha256: str | None = None
    interrupted_marker_path: Path | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "pair_id": self.pair_id,
            "task_id": self.task_id,
            "repeat_index": self.repeat_index,
            "attempt": self.attempt,
            "arm": self.arm,
            "trial_id": self.trial_id,
            "status": self.status,
            "failure_category": self.failure_category,
            "record_path": str(self.record_path) if self.record_path is not None else None,
            "record_sha256": self.record_sha256,
            "interrupted_marker_path": (
                str(self.interrupted_marker_path)
                if self.interrupted_marker_path is not None
                else None
            ),
        }


@dataclass(frozen=True)
class PairAttemptState:
    """Reconciled state of one attempt of a scheduled pair."""

    pair_id: str
    task_id: str
    repeat_index: int
    attempt: int
    arm_order: tuple[Arm, Arm]
    trials: tuple[TrialSlotState, TrialSlotState]
    retry_decision: RetryDecision
    retry_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "pair_id": self.pair_id,
            "task_id": self.task_id,
            "repeat_index": self.repeat_index,
            "attempt": self.attempt,
            "arm_order": list(self.arm_order),
            "trials": [t.to_dict() for t in self.trials],
            "retry_decision": self.retry_decision,
            "retry_reason": self.retry_reason,
        }


@dataclass(frozen=True)
class ScheduleProgress:
    """Full reconciled progress snapshot of a scheduled run."""

    schedule: RunSchedule
    schedule_sha256: str
    attempts: tuple[PairAttemptState, ...]
    pending_slots: tuple[TrialSlotState, ...]
    completed_trials: int
    failed_trials: int
    interrupted_trials: int
    retried_pairs: int

    @property
    def is_complete(self) -> bool:
        return len(self.pending_slots) == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.schedule.run_id,
            "seed": self.schedule.seed,
            "split": self.schedule.split,
            "schedule_sha256": self.schedule_sha256,
            "task_manifest_sha256": self.schedule.task_manifest_sha256,
            "retry_rule": self.schedule.retry_rule.to_dict(),
            "scheduled_pairs": len(self.schedule.pairs),
            "total_pair_attempts": len(self.attempts),
            "completed_trials": self.completed_trials,
            "failed_trials": self.failed_trials,
            "interrupted_trials": self.interrupted_trials,
            "pending_trials": len(self.pending_slots),
            "retried_pairs": self.retried_pairs,
            "is_complete": self.is_complete,
            "pending_slots": [slot.to_dict() for slot in self.pending_slots],
            "attempts": [att.to_dict() for att in self.attempts],
        }


@dataclass(frozen=True)
class ScheduleRunResult:
    """Outcome of a ``run_schedule`` invocation (full, bounded, or interrupted)."""

    progress: ScheduleProgress
    executed_trial_ids: tuple[str, ...] = field(default_factory=tuple)
    interrupted: bool = False
    stopped_by_max_trials: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "executed_trial_ids": list(self.executed_trial_ids),
            "interrupted": self.interrupted,
            "stopped_by_max_trials": self.stopped_by_max_trials,
            "progress": self.progress.to_dict(),
        }


def _interrupted_marker_path(store: UsefulnessRunStore, trial_id: str) -> Path:
    return store.trials_dir / trial_id / "interrupted.json"


def _write_interrupted_marker(
    store: UsefulnessRunStore,
    *,
    pair: ScheduledPair,
    attempt: int,
    arm: Arm,
    trial_id: str,
    reason: str,
) -> Path:
    marker_path = _interrupted_marker_path(store, trial_id)
    if not marker_path.exists():
        write_once_json(
            marker_path,
            {
                "schema_version": INTERRUPTED_MARKER_SCHEMA_VERSION,
                "recorded_at": utc_now_iso(),
                "run_id": store.run_id,
                "pair_id": pair.pair_id,
                "task_id": pair.task_id,
                "repeat_index": pair.repeat_index,
                "attempt": attempt,
                "arm": arm,
                "trial_id": trial_id,
                "status": "interrupted",
                "failure_category": "interrupted",
                "reason": reason,
            },
        )
    return marker_path


def _inspect_or_reconcile(
    store: UsefulnessRunStore,
    schedule: RunSchedule,
    schedule_sha: str,
    *,
    reconcile: bool,
) -> ScheduleProgress:
    """Compute progress from ``schedule.json``, ``schedule-events.jsonl``, and ``trials/``.

    When ``reconcile=True``, marks any started or partial trial missing
    ``record.json`` as interrupted (with a write-once ``interrupted.json`` marker
    and an appended ``trial_interrupted`` event), backfills any unlogged
    ``trial_finished`` events for existing ``record.json`` files, and evaluates
    and persists retry decisions for finished pair attempts.
    """
    events = store.schedule_events()
    started_ids: set[str] = set()
    finished_logged_ids: set[str] = set()
    interrupted_logged_ids: set[str] = set()
    retry_decisions_logged: dict[tuple[str, int], tuple[RetryDecision, str | None]] = {}
    retry_queue: list[tuple[str, int, tuple[Arm, Arm]]] = []
    seen_attempts: set[tuple[str, int]] = set()

    for pair in schedule.pairs:
        seen_attempts.add((pair.pair_id, 1))

    for ev in events:
        kind = ev.get("event")
        tid = ev.get("trial_id")
        if kind == "trial_started" and isinstance(tid, str):
            started_ids.add(tid)
        elif kind == "trial_finished" and isinstance(tid, str):
            finished_logged_ids.add(tid)
        elif kind == "trial_interrupted" and isinstance(tid, str):
            interrupted_logged_ids.add(tid)
        elif kind == "pair_retry_scheduled":
            pid = str(ev["pair_id"])
            prev_att = _require_int(ev["previous_attempt"], "previous_attempt", minimum=1)
            next_att = _require_int(ev["next_attempt"], "next_attempt", minimum=2)
            retry_decisions_logged[(pid, prev_att)] = ("scheduled", str(ev.get("reason", "")))
            if (pid, next_att) not in seen_attempts:
                arm_order = _validate_arm_order(ev["arm_order"])
                seen_attempts.add((pid, next_att))
                retry_queue.append((pid, next_att, arm_order))
        elif kind == "pair_retry_declined":
            pid = str(ev["pair_id"])
            att = _require_int(ev["attempt"], "attempt", minimum=1)
            reason = str(ev.get("reason", ""))
            retry_decisions_logged[(pid, att)] = ("declined", reason)

    ordered_attempts: list[tuple[ScheduledPair, int, tuple[Arm, Arm]]] = [
        (pair, 1, pair.arm_order) for pair in schedule.pairs
    ]
    for pid, att, arm_order in retry_queue:
        ordered_attempts.append((schedule.pair_by_id(pid), att, arm_order))

    attempt_states: list[PairAttemptState] = []
    pending_slots: list[TrialSlotState] = []
    completed_trials = 0
    failed_trials = 0
    interrupted_trials = 0
    retried_pairs = 0

    idx = 0
    while idx < len(ordered_attempts):
        pair, attempt, arm_order = ordered_attempts[idx]
        idx += 1

        slots: list[TrialSlotState] = []
        for arm in arm_order:
            trial_id = trial_id_for_slot(pair.pair_id, attempt, arm)
            trial_dir = store.trials_dir / trial_id
            record_path = trial_dir / "record.json"
            marker_path = _interrupted_marker_path(store, trial_id)

            if record_path.is_file():
                rec = read_json_file(record_path)
                raw_status = str(rec.get("status", "failed"))
                raw_failure = rec.get("failure_category")
                failure_cat = raw_failure if isinstance(raw_failure, str) else None
                rec_sha = sha256_file(record_path)

                if raw_status == "completed":
                    slot_status: SlotStatus = "completed"
                    completed_trials += 1
                elif failure_cat == "interrupted":
                    slot_status = "interrupted"
                    interrupted_trials += 1
                else:
                    slot_status = "failed"
                    failed_trials += 1

                if reconcile and trial_id not in finished_logged_ids:
                    store.append_schedule_event(
                        {
                            "event": "trial_finished",
                            "recorded_at": utc_now_iso(),
                            "pair_id": pair.pair_id,
                            "task_id": pair.task_id,
                            "repeat_index": pair.repeat_index,
                            "attempt": attempt,
                            "arm": arm,
                            "trial_id": trial_id,
                            "status": raw_status,
                            "failure_category": failure_cat,
                            "record_sha256": rec_sha,
                        }
                    )
                    finished_logged_ids.add(trial_id)

                slots.append(
                    TrialSlotState(
                        pair_id=pair.pair_id,
                        task_id=pair.task_id,
                        repeat_index=pair.repeat_index,
                        attempt=attempt,
                        arm=arm,
                        trial_id=trial_id,
                        status=slot_status,
                        failure_category=failure_cat,
                        record_path=record_path,
                        record_sha256=rec_sha,
                        interrupted_marker_path=marker_path if marker_path.is_file() else None,
                    )
                )
            elif (
                trial_id in started_ids or trial_id in interrupted_logged_ids or trial_dir.exists()
            ):
                # Interrupted before record.json could be written: keep and mark, never overwrite.
                reason = "trial interrupted before record.json was written"
                if reconcile:
                    written_marker = _write_interrupted_marker(
                        store,
                        pair=pair,
                        attempt=attempt,
                        arm=arm,
                        trial_id=trial_id,
                        reason=reason,
                    )
                    if trial_id not in interrupted_logged_ids:
                        store.append_schedule_event(
                            {
                                "event": "trial_interrupted",
                                "recorded_at": utc_now_iso(),
                                "pair_id": pair.pair_id,
                                "task_id": pair.task_id,
                                "repeat_index": pair.repeat_index,
                                "attempt": attempt,
                                "arm": arm,
                                "trial_id": trial_id,
                                "failure_category": "interrupted",
                                "reason": reason,
                                "marker_path": str(written_marker),
                            }
                        )
                        interrupted_logged_ids.add(trial_id)
                    effective_marker: Path | None = written_marker
                else:
                    effective_marker = marker_path if marker_path.is_file() else None

                interrupted_trials += 1
                slots.append(
                    TrialSlotState(
                        pair_id=pair.pair_id,
                        task_id=pair.task_id,
                        repeat_index=pair.repeat_index,
                        attempt=attempt,
                        arm=arm,
                        trial_id=trial_id,
                        status="interrupted",
                        failure_category="interrupted",
                        record_path=None,
                        record_sha256=None,
                        interrupted_marker_path=effective_marker,
                    )
                )
            else:
                slot = TrialSlotState(
                    pair_id=pair.pair_id,
                    task_id=pair.task_id,
                    repeat_index=pair.repeat_index,
                    attempt=attempt,
                    arm=arm,
                    trial_id=trial_id,
                    status="pending",
                )
                slots.append(slot)
                pending_slots.append(slot)

        slot_pair = (slots[0], slots[1])
        both_settled = slot_pair[0].status != "pending" and slot_pair[1].status != "pending"

        decision: RetryDecision
        decision_reason: str | None
        if not both_settled:
            decision = "pending"
            decision_reason = None
        else:
            existing = retry_decisions_logged.get((pair.pair_id, attempt))
            if existing is not None:
                decision, decision_reason = existing
                if decision == "scheduled":
                    retried_pairs += 1
            else:
                arm_outcomes = [
                    (slot_pair[0].status, slot_pair[0].failure_category),
                    (slot_pair[1].status, slot_pair[1].failure_category),
                ]
                should_retry, reason, failing_cats = schedule.retry_rule.evaluate(
                    attempt, arm_outcomes
                )
                if reason == "all_arms_completed":
                    decision = "not_needed"
                    decision_reason = reason
                elif should_retry:
                    decision = "scheduled"
                    decision_reason = reason
                    retried_pairs += 1
                    next_attempt = attempt + 1
                    next_arms = arm_order_for_attempt(schedule.seed, pair, next_attempt)
                    if (pair.pair_id, next_attempt) not in seen_attempts:
                        seen_attempts.add((pair.pair_id, next_attempt))
                        ordered_attempts.append((pair, next_attempt, next_arms))
                    if reconcile:
                        store.append_schedule_event(
                            {
                                "event": "pair_retry_scheduled",
                                "recorded_at": utc_now_iso(),
                                "pair_id": pair.pair_id,
                                "task_id": pair.task_id,
                                "repeat_index": pair.repeat_index,
                                "previous_attempt": attempt,
                                "next_attempt": next_attempt,
                                "arm_order": list(next_arms),
                                "reason": reason,
                                "triggering_failure_categories": list(failing_cats),
                            }
                        )
                        retry_decisions_logged[(pair.pair_id, attempt)] = (
                            decision,
                            decision_reason,
                        )
                else:
                    decision = "declined"
                    decision_reason = reason
                    if reconcile:
                        store.append_schedule_event(
                            {
                                "event": "pair_retry_declined",
                                "recorded_at": utc_now_iso(),
                                "pair_id": pair.pair_id,
                                "task_id": pair.task_id,
                                "repeat_index": pair.repeat_index,
                                "attempt": attempt,
                                "reason": reason,
                                "failure_categories": list(failing_cats),
                            }
                        )
                        retry_decisions_logged[(pair.pair_id, attempt)] = (
                            decision,
                            decision_reason,
                        )

        attempt_states.append(
            PairAttemptState(
                pair_id=pair.pair_id,
                task_id=pair.task_id,
                repeat_index=pair.repeat_index,
                attempt=attempt,
                arm_order=arm_order,
                trials=slot_pair,
                retry_decision=decision,
                retry_reason=decision_reason,
            )
        )

    return ScheduleProgress(
        schedule=schedule,
        schedule_sha256=schedule_sha,
        attempts=tuple(attempt_states),
        pending_slots=tuple(pending_slots),
        completed_trials=completed_trials,
        failed_trials=failed_trials,
        interrupted_trials=interrupted_trials,
        retried_pairs=retried_pairs,
    )


def inspect_schedule_progress(
    experiment_home: Config | Path,
    run_id: str,
    *,
    verify_manifest: bool = True,
) -> ScheduleProgress:
    """Read-only inspection of a run's schedule progress, pending slots, and retries."""
    store = UsefulnessRunStore(experiment_home, run_id)
    schedule, schedule_sha = load_run_schedule(
        experiment_home, run_id, verify_manifest=verify_manifest
    )
    return _inspect_or_reconcile(store, schedule, schedule_sha, reconcile=False)


def run_schedule(
    experiment_home: Config | Path,
    run_id: str,
    run_config: ScheduleRunConfig,
    *,
    tasks_dir: Path | None = None,
    adapter: HarnessAdapter | None = None,
    max_trials: int | None = None,
) -> ScheduleRunResult:
    """Execute pending trials of a run's schedule in seeded order, resuming cleanly.

    On entry, reconciles any previously interrupted trial (keeping its directory
    and marking it without overwriting in place) and evaluates retry decisions
    under the schedule's ``RetryRule``. Executes pending trial slots one by one,
    stopping cleanly on ``KeyboardInterrupt`` (or a trial returning
    ``failure_category == "interrupted"``) or when ``max_trials`` is reached.
    """
    if max_trials is not None:
        _require_int(max_trials, "max_trials", minimum=1)

    store = UsefulnessRunStore(experiment_home, run_id)
    schedule, schedule_sha = load_run_schedule(experiment_home, run_id, verify_manifest=True)
    manifest = store.read_task_manifest()
    effective_tasks_dir = (
        tasks_dir.expanduser().resolve() if tasks_dir is not None else store.tasks_dir
    )

    # Preflight egress, held-out registration gate, and source_home before
    # recording any trial_started event so a configuration refusal never leaves
    # a false interrupted marker.
    check_trial_egress(run_config.model, run_config.model_base_url)
    if schedule.split == "held_out" and not store.registration_path.exists():
        raise UsefulnessError(
            f"cannot run held-out schedule for run {run_id!r}: "
            f"no registration at {store.registration_path}"
        )
    if run_config.source_home is not None:
        resolved_src = ensure_experiment_home(run_config.source_home)
        if not (resolved_src / "memory.duckdb").is_file():
            raise UsefulnessError(
                f"source home has no database file: {resolved_src / 'memory.duckdb'}"
            )
        if (resolved_src / "memory.duckdb.wal").exists():
            raise UsefulnessError(
                f"source home has a WAL file ({resolved_src / 'memory.duckdb.wal'}); "
                "stop its server before copying into a trial"
            )

    active_adapter: HarnessAdapter = adapter if adapter is not None else DroidHarnessAdapter()
    executed: list[str] = []
    interrupted = False
    stopped_by_max = False

    while True:
        progress = _inspect_or_reconcile(store, schedule, schedule_sha, reconcile=True)
        if not progress.pending_slots:
            return ScheduleRunResult(
                progress=progress,
                executed_trial_ids=tuple(executed),
                interrupted=interrupted,
                stopped_by_max_trials=stopped_by_max,
            )

        if max_trials is not None and len(executed) >= max_trials:
            stopped_by_max = True
            return ScheduleRunResult(
                progress=progress,
                executed_trial_ids=tuple(executed),
                interrupted=interrupted,
                stopped_by_max_trials=stopped_by_max,
            )

        slot = progress.pending_slots[0]
        pair = schedule.pair_by_id(slot.pair_id)
        task_dir = effective_tasks_dir / pair.task_id
        if not task_dir.is_dir():
            raise UsefulnessError(
                f"task directory missing for scheduled task {pair.task_id!r}: {task_dir}"
            )
        # Verify the task manifest and any frozen registration before logging
        # trial_started so a refusal never creates a false interrupted marker.
        verified_task, task_split = load_verified_task(task_dir, manifest)

        trial_cfg = run_config.to_trial_config(arm=slot.arm, trial_id=slot.trial_id)
        resolve_trial_registration(
            experiment_home,
            run_id,
            verified_task.task_id,
            task_split,
            trial_cfg,
            active_adapter,
        )
        store.append_schedule_event(
            {
                "event": "trial_started",
                "recorded_at": utc_now_iso(),
                "pair_id": slot.pair_id,
                "task_id": slot.task_id,
                "repeat_index": slot.repeat_index,
                "attempt": slot.attempt,
                "arm": slot.arm,
                "trial_id": slot.trial_id,
            }
        )

        try:
            record: TrialRecord = run_trial(
                experiment_home,
                run_id,
                verified_task,
                trial_cfg,
                adapter=active_adapter,
            )
        except KeyboardInterrupt:
            interrupted = True
            record_path = store.trials_dir / slot.trial_id / "record.json"
            if not record_path.is_file():
                marker = _write_interrupted_marker(
                    store,
                    pair=pair,
                    attempt=slot.attempt,
                    arm=slot.arm,
                    trial_id=slot.trial_id,
                    reason="run interrupted by KeyboardInterrupt before record.json was written",
                )
                store.append_schedule_event(
                    {
                        "event": "trial_interrupted",
                        "recorded_at": utc_now_iso(),
                        "pair_id": slot.pair_id,
                        "task_id": slot.task_id,
                        "repeat_index": slot.repeat_index,
                        "attempt": slot.attempt,
                        "arm": slot.arm,
                        "trial_id": slot.trial_id,
                        "failure_category": "interrupted",
                        "reason": (
                            "run interrupted by KeyboardInterrupt before record.json was written"
                        ),
                        "marker_path": str(marker),
                    }
                )
            executed.append(slot.trial_id)
            final_progress = _inspect_or_reconcile(store, schedule, schedule_sha, reconcile=False)
            return ScheduleRunResult(
                progress=final_progress,
                executed_trial_ids=tuple(executed),
                interrupted=True,
                stopped_by_max_trials=False,
            )

        store.append_schedule_event(
            {
                "event": "trial_finished",
                "recorded_at": utc_now_iso(),
                "pair_id": slot.pair_id,
                "task_id": slot.task_id,
                "repeat_index": slot.repeat_index,
                "attempt": slot.attempt,
                "arm": slot.arm,
                "trial_id": record.trial_id,
                "status": record.status,
                "failure_category": record.failure_category,
                "record_sha256": record.record_sha256,
            }
        )
        executed.append(record.trial_id)

        if record.failure_category == "interrupted":
            interrupted = True
            # Stop immediately on interruption without launching further trials;
            # on resume, the interrupted trial stays kept and marked and unmade
            # trials continue (and retry_rule governs any retry attempt).
            final_progress = _inspect_or_reconcile(store, schedule, schedule_sha, reconcile=False)
            return ScheduleRunResult(
                progress=final_progress,
                executed_trial_ids=tuple(executed),
                interrupted=True,
                stopped_by_max_trials=False,
            )
