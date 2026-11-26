"""Blind trial grading and post-grading memory-use attribution (MEM-25).

Drives the shared labeling helper (``judge.label_frontend.drive``) for two
distinct, strictly separated phases of the usefulness experiment:

1. **Blind grading** (``BlindGradingController``):
   - Shows only the task request, required behavior, task checks and what they
     gave, repository changes, and the trial's last reply. Never shows the arm,
     ``trial_id``, memory tool calls, recall accounting, or anything from a
     ledger (and scrubs trial directory paths / IDs from check outputs).
   - Orders trials deterministically from a seed so the order does not tell the
     arm and never places the two trials of a pair back-to-back (whenever at
     least two pairs are present).
   - Appends every grade and every pre-finalization correction to
     ``grades.jsonl`` (preserving full edit history) until ``finalize_grades``
     writes write-once ``grades-final.json``.

2. **Post-grading memory-use attribution** (``MemoryAttributionController``):
   - Refuses to open until every finished trial of the run is graded and
     ``grades-final.json`` is written and verified against ``grades.jsonl``.
   - Shows the unblinded memory-arm trial's recalls, citations, and recalled
     memories (read read-only from the trial's own home) so the evaluator can
     link wrong actions, attempted retired/expired/out-of-scope memory use, and
     confident wrong replies to memory use.
   - Writes only to append-only ``attributions.jsonl`` and never changes any
     grade.
"""

from __future__ import annotations

import hashlib
import random
import uuid
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb

from ..config import Config
from ..judge.label_controller import Action, Screen
from ..judge.label_frontend import drive
from ..judge.label_questions import Option, display_safe
from ..judge.store import JudgeError, utc_now_iso
from .store import (
    UsefulnessError,
    UsefulnessRunStore,
    read_json_file,
    sha256_file,
)
from .tasks import TaskSpec, load_task, load_verified_task

GRADE_SCHEMA_VERSION = 1
GRADES_FINAL_SCHEMA_VERSION = 1
ATTRIBUTION_SCHEMA_VERSION = 1

MAX_NOTE_BYTES = 4096
MAX_EVIDENCE_SECTION_CHARS = 120_000
MAX_PROJECTION_CHARS = 2_000_000

GRADE_FAILURE_CATEGORIES: tuple[str, ...] = (
    "ordinary_wrong_answer",
    "timeout",
    "tool_outage",
    "model_refusal",
    "evaluator_failure",
    "interrupted",
)

GRADE_FAILURE_LABELS: dict[str, str] = {
    "ordinary_wrong_answer": "Ordinary wrong answer",
    "timeout": "Timeout (wall time or budget limit)",
    "tool_outage": "Tool or model server outage",
    "model_refusal": "Model refusal",
    "evaluator_failure": "Evaluator, check, or environment failure",
    "interrupted": "Interrupted before finishing",
}

BLIND_GRADING_INTRO = (
    "Grade each trial blind to the arm.\nAnswers save per trial. ? opens the grading guide."
)

BLIND_GRADING_GUIDE = """Blind trial grading guide

What is shown
You see the task request, the required behavior, the task's checks and what they gave,
the changes to the repository, and the agent's last reply.
The arm, calls to memory tools, and anything from a ledger are hidden.
Trials are shown in a seeded blind order that does not put a pair's two trials back to back.

Met checks
Yes: the trial met its task's checks and required behavior.
No: the trial did not meet its checks or required behavior.

Failure kind (when Met checks is No)
Choose the primary failure kind:
- Ordinary wrong answer: the agent finished, but checks or required behavior were not met.
- Timeout: the trial hit its wall-time or budget limit.
- Tool or model server outage: a tool or model server outage prevented completion.
- Model refusal: the model refused the task.
- Evaluator, check, or environment failure: checks could not run, or a task/environment defect.
- Interrupted before finishing: the trial was interrupted before completing.

Confident wrong reply (when Met checks is No)
Yes: the final reply presented the task as done or a fact as known with no stated doubt
or failure, while the checks show it was wrong.
No: the agent stated doubt, stated failure, or gave no reply.

Arm leaked
Answer Yes if anything in the shown material (request, checks, diff, or reply) gave away
whether this trial had memory tools or not. The report counts these trials separately.

Navigation and history
Back restores the previous answer and clears dependent answers when changed.
A note is optional literal text. Every save appends a revision to grades.jsonl so earlier
grades remain in history until grades are finalized.
p opens the full trial bundle. Skip leaves this trial for later. Quit keeps saved grades.
"""

ATTRIBUTION_INTRO = (
    "Link failed or confident-wrong memory-arm trials to recalled memories.\n"
    "Grades are final and will not be changed. ? opens the attribution guide."
)

ATTRIBUTION_GUIDE = """Post-grading memory-use attribution guide

When this opens
Every grade of the run is final (grades-final.json is written and verified).
Nothing done here changes any grade; attributions are appended to attributions.jsonl.

What is shown
You see the finalized grade, the trial's recall accounting, citations, the recalled
memories read read-only from the trial's own home, memory tool calls, and the trial output.

Questions
- Harmful memory use: Yes if a declared or observed use of a memory led to a concrete
  wrong action in the trial.
- Attempted invalid memory use: Yes if the trial attempted to use a retired, expired, or
  out-of-scope memory.
- Confident wrong reply linked to memory (only when the blind grade marked a confident
  wrong reply): Yes if the wrong part of the final reply traces to a declared or observed
  use of a memory.

Navigation
Back edits the previous answer. Add a note to record specific memory IDs or evidence.
p opens the full unblinded trial details. Quit keeps earlier saved attributions.
"""

_YN_OPTIONS = [Option("y", "Yes"), Option("n", "No")]
_ARM_LEAKED_OPTIONS = [Option("n", "No"), Option("y", "Yes, material gave the arm away")]


@dataclass(frozen=True)
class TrialGradingTarget:
    """One finished trial discovered in a run, together with its verified task and checks."""

    trial_id: str
    task_id: str
    arm: str
    pair_key: tuple[str, int]
    trial_dir: Path
    record_path: Path
    record_sha256: str
    record: Mapping[str, Any]
    checks_path: Path | None
    checks_sha256: str | None
    checks: Mapping[str, Any] | None
    task: TaskSpec


@dataclass(frozen=True)
class GradingSummary:
    """Summary returned after a blind grading sitting."""

    graded: int
    skipped: int
    changed: int
    stopped: bool
    remaining: int


@dataclass(frozen=True)
class AttributionSummary:
    """Summary returned after a post-grading memory-use attribution sitting."""

    attributed: int
    skipped: int
    changed: int
    stopped: bool
    remaining: int


@dataclass(frozen=True)
class GradingStatus:
    """Read-only, arm-blind grading progress snapshot for a run."""

    run_id: str
    total_trials: int
    graded_trials: int
    ungraded_trials: int
    total_grade_revisions: int
    arm_leaked_count: int
    grades_final: bool
    grades_final_sha256: str | None
    attributions_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "total_trials": self.total_trials,
            "graded_trials": self.graded_trials,
            "ungraded_trials": self.ungraded_trials,
            "total_grade_revisions": self.total_grade_revisions,
            "arm_leaked_count": self.arm_leaked_count,
            "grades_final": self.grades_final,
            "grades_final_sha256": self.grades_final_sha256,
            "attributions_count": self.attributions_count,
        }


# ---------------------------------------------------------------------------
# Trial discovery and blind ordering
# ---------------------------------------------------------------------------


def _pair_key_for_trial(
    trial_id: str,
    task_id: str,
    arm: str,
    schedule_pair_ids: set[str],
) -> tuple[str, int]:
    """Derive a stable ``(pair_id, attempt)`` grouping key for a trial."""
    suffix = f"__{arm}"
    if schedule_pair_ids and trial_id.endswith(suffix):
        stem = trial_id[: -len(suffix)]
        pair_id, sep, attempt_part = stem.rpartition("__a")
        if sep and pair_id in schedule_pair_ids and attempt_part.isdigit():
            return (pair_id, int(attempt_part))
    return (task_id, 0)


def discover_run_trials(
    experiment_home: Config | Path,
    run_id: str,
    *,
    tasks_dir: Path | None = None,
    require_checks: bool = True,
) -> list[TrialGradingTarget]:
    """Load every finished trial (``record.json``) in ``run_id`` with its task and checks."""
    store = UsefulnessRunStore(experiment_home, run_id)
    effective_tasks_dir = (
        tasks_dir.expanduser().resolve() if tasks_dir is not None else store.tasks_dir
    )
    manifest = store.read_task_manifest() if store.task_manifest_path.exists() else None
    schedule_pair_ids: set[str] = set()
    if store.schedule_path.exists():
        sched = store.read_schedule()
        for pair in sched.get("pairs") or ():
            if isinstance(pair, Mapping) and isinstance(pair.get("pair_id"), str):
                schedule_pair_ids.add(pair["pair_id"])

    if not store.trials_dir.is_dir():
        return []

    task_cache: dict[str, TaskSpec] = {}
    targets: list[TrialGradingTarget] = []
    for trial_dir in sorted(p for p in store.trials_dir.iterdir() if p.is_dir()):
        record_path = trial_dir / "record.json"
        if not record_path.is_file():
            continue
        record = read_json_file(record_path)
        record_sha = sha256_file(record_path)
        trial_id = record.get("trial_id")
        task_id = record.get("task_id")
        arm = record.get("arm")
        if (
            not isinstance(trial_id, str)
            or not trial_id
            or not isinstance(task_id, str)
            or not task_id
            or arm not in ("memory", "no_memory")
        ):
            raise UsefulnessError(f"malformed trial record at {record_path}")

        if task_id not in task_cache:
            tdir = effective_tasks_dir / task_id
            if not tdir.is_dir():
                raise UsefulnessError(f"task directory missing for task {task_id!r}: {tdir}")
            if manifest is not None:
                task_spec, _ = load_verified_task(tdir, manifest)
            else:
                task_spec = load_task(tdir)
            task_cache[task_id] = task_spec
        task = task_cache[task_id]
        rec_task_sha = record.get("task_sha256")
        if isinstance(rec_task_sha, str) and rec_task_sha and rec_task_sha != task.task_sha256:
            raise UsefulnessError(
                f"task {task_id!r} hash mismatch between disk ({task.task_sha256}) "
                f"and trial record ({rec_task_sha})"
            )

        checks_path = trial_dir / "checks.json"
        checks_data: dict[str, Any] | None = None
        checks_sha: str | None = None
        if checks_path.is_file():
            checks_data = read_json_file(checks_path)
            checks_sha = sha256_file(checks_path)
            if checks_data.get("trial_id") != trial_id or checks_data.get("task_id") != task_id:
                raise UsefulnessError(f"checks.json metadata mismatch in {checks_path}")
            if (
                isinstance(checks_data.get("record_sha256"), str)
                and checks_data["record_sha256"] != record_sha
            ):
                raise UsefulnessError(
                    f"checks.json record_sha256 does not match record.json in {trial_dir}"
                )
        elif require_checks:
            raise UsefulnessError(
                f"trial {trial_id!r} has no checks.json at {checks_path}; "
                "run `mem usefulness run-checks` before blind grading"
            )

        pair_key = _pair_key_for_trial(trial_id, task_id, arm, schedule_pair_ids)
        targets.append(
            TrialGradingTarget(
                trial_id=trial_id,
                task_id=task_id,
                arm=arm,
                pair_key=pair_key,
                trial_dir=trial_dir,
                record_path=record_path,
                record_sha256=record_sha,
                record=record,
                checks_path=checks_path if checks_data is not None else None,
                checks_sha256=checks_sha,
                checks=checks_data,
                task=task,
            )
        )
    return targets


def _no_adjacent_pairs(items: Sequence[TrialGradingTarget]) -> bool:
    return all(items[i].pair_key != items[i + 1].pair_key for i in range(len(items) - 1))


def _no_adjacent_tasks(items: Sequence[TrialGradingTarget]) -> bool:
    return all(items[i].task_id != items[i + 1].task_id for i in range(len(items) - 1))


def order_trials_blind(
    trials: Sequence[TrialGradingTarget],
    *,
    seed: int = 1,
) -> list[TrialGradingTarget]:
    """Order trials deterministically from ``seed`` without revealing arm or pairing adjacently.

    Guarantees:
    - Independent of filesystem order and uncorrelated with ``arm``.
    - Whenever feasible (i.e. largest pair group has size ``<= ceil(N / 2)``,
      which always holds when at least two pairs are present), no two trials of
      the same pair appear consecutively.
    - Also avoids consecutive trials of the same task when feasible.
    """
    n = len(trials)
    if n <= 1:
        return list(trials)

    # Sort canonically by a seeded hash of trial_id so input order has zero
    # correlation with arm names or filesystem directory ordering.
    base = sorted(
        trials,
        key=lambda t: hashlib.sha256(f"{seed}:{t.trial_id}".encode()).hexdigest(),
    )

    pair_counts: dict[tuple[str, int], int] = defaultdict(int)
    task_counts: dict[str, int] = defaultdict(int)
    for t in base:
        pair_counts[t.pair_key] += 1
        task_counts[t.task_id] += 1

    can_separate_pairs = len(pair_counts) >= 2 and max(pair_counts.values()) <= (n + 1) // 2
    can_separate_tasks = len(task_counts) >= 2 and max(task_counts.values()) <= (n + 1) // 2

    rng = random.Random(seed)
    if not can_separate_pairs:
        # Only one pair (or impossible pigeonhole distribution): still randomize arm order.
        out = list(base)
        rng.shuffle(out)
        return out

    # 1. Try direct seeded shuffles first (uniform over valid permutations).
    if can_separate_tasks:
        for _ in range(128):
            cand = list(base)
            rng.shuffle(cand)
            if _no_adjacent_pairs(cand) and _no_adjacent_tasks(cand):
                return cand
    for _ in range(128):
        cand = list(base)
        rng.shuffle(cand)
        if _no_adjacent_pairs(cand):
            return cand

    # 2. Deterministic randomized greedy fallback with pigeonhole feasibility guard.
    buckets: dict[tuple[str, int], list[TrialGradingTarget]] = defaultdict(list)
    for t in base:
        buckets[t.pair_key].append(t)
    for b in buckets.values():
        rng.shuffle(b)

    result: list[TrialGradingTarget] = []
    last_pair: tuple[str, int] | None = None
    last_task: str | None = None
    remaining = n
    while remaining > 0:
        eligible = [k for k, v in buckets.items() if v and k != last_pair]
        if not eligible:
            # Should not happen when can_separate_pairs is True; defensive fallback.
            eligible = [k for k, v in buckets.items() if v]
        # Any bucket with count > (remaining - 1) // 2 MUST be chosen now to keep
        # a valid non-adjacent continuation feasible.
        max_len = max(len(buckets[k]) for k in eligible)
        if max_len > (remaining - 1) // 2:
            candidates = [k for k in eligible if len(buckets[k]) == max_len]
        else:
            candidates = eligible
            if last_task is not None:
                diff_task = [k for k in candidates if buckets[k][-1].task_id != last_task]
                if diff_task:
                    candidates = diff_task
        chosen_key = rng.choice(candidates)
        chosen = buckets[chosen_key].pop()
        result.append(chosen)
        last_pair = chosen.pair_key
        last_task = chosen.task_id
        remaining -= 1

    return result


# ---------------------------------------------------------------------------
# Blind evidence formatting (strictly no arm, no trial_id, no memory tools/ledger)
# ---------------------------------------------------------------------------


def _truncate_text(text: str, limit: int, label: str) -> str:
    if len(text) <= limit:
        return text
    omitted = len(text) - limit
    return f"{text[:limit]}\n\n[... {label} truncated: {omitted} chars omitted ...]"


def _scrub_blind_text(
    text: str,
    *,
    trial_dir: Path,
    trials_dir: Path,
    trial_id: str,
) -> str:
    """Remove trial paths and ``trial_id`` (which may embed the arm name) from displayed text."""
    if not text:
        return text
    replacements = [
        (str(trial_dir.resolve()), "<trial-dir>"),
        (str(trial_dir), "<trial-dir>"),
        (str(trials_dir.resolve()), "<trials-dir>"),
        (str(trials_dir), "<trials-dir>"),
        (trial_id, "<trial>"),
    ]
    out = text
    for needle, token in replacements:
        if needle:
            out = out.replace(needle, token)
    return out


def _format_checks_section(
    target: TrialGradingTarget,
    *,
    trials_dir: Path,
    max_stream_chars: int,
) -> str:
    def scrub(val: str) -> str:
        return _scrub_blind_text(
            val,
            trial_dir=target.trial_dir,
            trials_dir=trials_dir,
            trial_id=target.trial_id,
        )

    if target.checks is None:
        lines = ["Checks have not been run for this trial. Declared checks:"]
        for chk in target.task.checks:
            lines.append(f"- {chk.name}: {' '.join(chk.command)}")
        return "\n".join(lines)

    overall = str(target.checks.get("overall_status", "unknown"))
    passed = target.checks.get("passed_count", 0)
    failed = target.checks.get("failed_count", 0)
    cnr = target.checks.get("could_not_run_count", 0)
    lines = [
        f"Overall: {overall} (passed={passed}, failed={failed}, could_not_run={cnr})",
    ]
    hidden_installed = target.checks.get("hidden_tests_installed") or []
    if hidden_installed:
        lines.append("Hidden tests installed: " + ", ".join(str(p) for p in hidden_installed))
    hidden_err = target.checks.get("hidden_tests_error")
    if hidden_err:
        lines.append("Hidden tests error: " + scrub(str(hidden_err)))

    for chk in target.checks.get("checks") or ():
        if not isinstance(chk, Mapping):
            continue
        name = str(chk.get("name", "check"))
        status = str(chk.get("status", "unknown"))
        exit_code = chk.get("exit_code")
        duration = chk.get("duration_s")
        cmd_raw = chk.get("command")
        cmd_str = (
            scrub(" ".join(str(part) for part in cmd_raw)) if isinstance(cmd_raw, list) else ""
        )
        header = f"\n[{name}] status={status} exit_code={exit_code}"
        if isinstance(duration, int | float):
            header += f" duration={duration:.2f}s"
        lines.append(header)
        if cmd_str:
            lines.append(f"command: {cmd_str}")
        if chk.get("reason"):
            lines.append(f"reason: {scrub(str(chk['reason']))}")
        if chk.get("error_detail"):
            lines.append(f"error: {scrub(str(chk['error_detail']))}")
        stdout = scrub(str(chk.get("stdout") or ""))
        stderr = scrub(str(chk.get("stderr") or ""))
        if stdout.strip():
            lines.append("stdout:\n" + _truncate_text(stdout.rstrip(), max_stream_chars, "stdout"))
        if stderr.strip():
            lines.append("stderr:\n" + _truncate_text(stderr.rstrip(), max_stream_chars, "stderr"))
    return "\n".join(lines)


def _format_repo_changes_section(
    target: TrialGradingTarget,
    *,
    trials_dir: Path,
    max_diff_chars: int,
) -> str:
    def scrub(val: str) -> str:
        return _scrub_blind_text(
            val,
            trial_dir=target.trial_dir,
            trials_dir=trials_dir,
            trial_id=target.trial_id,
        )

    rc = target.record.get("repo_changes")
    if not isinstance(rc, Mapping):
        return "(repository changes unavailable)"
    changed = [str(f) for f in (rc.get("changed_files") or ())]
    untracked = [str(f) for f in (rc.get("untracked_files") or ())]
    diff_stat = scrub(str(rc.get("diff_stat") or "")).strip()
    diff = scrub(str(rc.get("diff") or "")).strip()
    if not rc.get("has_changes") and not changed and not untracked and not diff:
        return "(no repository changes)"

    lines: list[str] = []
    if changed:
        lines.append("Changed files: " + ", ".join(changed))
    if untracked:
        lines.append("Untracked files: " + ", ".join(untracked))
    if diff_stat:
        lines.append("Diff stat:\n" + diff_stat)
    if diff:
        lines.append("Diff:\n" + _truncate_text(diff, max_diff_chars, "diff"))
    return "\n".join(lines) if lines else "(no repository changes)"


def _format_blind_trial_bundle(
    target: TrialGradingTarget,
    *,
    trials_dir: Path,
    max_section_chars: int,
) -> str:
    """Format the strictly blind trial material shown to the evaluator."""

    def scrub(val: str) -> str:
        return _scrub_blind_text(
            val,
            trial_dir=target.trial_dir,
            trials_dir=trials_dir,
            trial_id=target.trial_id,
        )

    sections: list[str] = []
    # Neutral trial termination notice if execution itself failed (no failure_detail
    # in case it contains raw harness paths, and never arm/tool_calls/ledger info).
    rec_status = target.record.get("status")
    rec_fail_cat = target.record.get("failure_category")
    if rec_status == "failed":
        cat_text = f" ({rec_fail_cat})" if isinstance(rec_fail_cat, str) and rec_fail_cat else ""
        sections.append(f"=== TRIAL EXECUTION ===\nStatus: failed{cat_text}")

    sections.append(
        "=== REQUEST ===\n"
        + _truncate_text(
            target.task.request.strip() or "(empty request)", max_section_chars, "request"
        )
    )
    sections.append(
        "=== REQUIRED BEHAVIOR ===\n"
        + _truncate_text(
            target.task.required_behavior.strip() or "(empty required behavior)",
            max_section_chars,
            "required_behavior",
        )
    )
    sections.append(
        "=== CHECKS ===\n"
        + _format_checks_section(target, trials_dir=trials_dir, max_stream_chars=max_section_chars)
    )
    sections.append(
        "=== REPOSITORY CHANGES ===\n"
        + _format_repo_changes_section(
            target, trials_dir=trials_dir, max_diff_chars=max_section_chars
        )
    )
    last_reply = target.record.get("last_reply")
    if isinstance(last_reply, str) and last_reply.strip():
        reply_text = _truncate_text(scrub(last_reply.strip()), max_section_chars, "last_reply")
    else:
        reply_text = "(no final reply)"
    sections.append("=== LAST REPLY ===\n" + reply_text)

    return "\n\n".join(sections)


def format_blind_trial_evidence(target: TrialGradingTarget, *, trials_dir: Path) -> str:
    """Compact blind trial evidence for the main evidence viewport."""
    return _format_blind_trial_bundle(
        target, trials_dir=trials_dir, max_section_chars=MAX_EVIDENCE_SECTION_CHARS
    )


def format_blind_trial_full(target: TrialGradingTarget, *, trials_dir: Path) -> str:
    """Full blind trial evidence for the pager (``p`` / ``evidence``)."""
    text = _format_blind_trial_bundle(
        target, trials_dir=trials_dir, max_section_chars=MAX_PROJECTION_CHARS // 4
    )
    return _truncate_text(text, MAX_PROJECTION_CHARS, "full trial bundle")


def _blind_metadata(target: TrialGradingTarget, *, revision: int | None = None) -> str:
    overall = str(target.checks.get("overall_status", "not_run")) if target.checks else "not_run"
    parts = [
        f"Task: {target.task_id}",
        f"Project: {target.task.project}",
        f"Checks: {overall}",
    ]
    if target.record.get("status") == "failed":
        cat = target.record.get("failure_category") or "failed"
        parts.append(f"Execution: failed ({cat})")
    if revision is not None:
        parts.append(f"Saved rev: {revision}")
    return " · ".join(parts)


# ---------------------------------------------------------------------------
# Grade storage, history, status, and finalization
# ---------------------------------------------------------------------------


def _validate_note(note: str) -> str:
    if (
        not isinstance(note, str)
        or any(ch in "\r\n" or display_safe(ch) != ch for ch in note)
        or len(note.encode("utf-8")) > MAX_NOTE_BYTES
    ):
        raise UsefulnessError(
            f"note must be one line of plain text, at most {MAX_NOTE_BYTES} bytes"
        )
    return note


def _is_recurrence_task(task: TaskSpec) -> bool:
    return any(s.strip().lower() == "recurrence" for s in task.strata)


def latest_grades(store: UsefulnessRunStore) -> dict[str, dict[str, Any]]:
    """Return the latest grade revision per ``trial_id`` from ``grades.jsonl``."""
    latest: dict[str, dict[str, Any]] = {}
    for row in store.grades():
        tid = row.get("trial_id")
        if isinstance(tid, str) and tid:
            latest[tid] = row
    return latest


def grade_history(
    store: UsefulnessRunStore,
    trial_id: str | None = None,
) -> list[dict[str, Any]]:
    """Return full append-only grade revision history, optionally filtered to one trial."""
    rows = store.grades()
    if trial_id is None:
        return rows
    return [row for row in rows if row.get("trial_id") == trial_id]


def latest_attributions(store: UsefulnessRunStore) -> dict[str, dict[str, Any]]:
    """Return the latest memory-use attribution revision per ``trial_id``."""
    latest: dict[str, dict[str, Any]] = {}
    for row in store.attributions():
        tid = row.get("trial_id")
        if isinstance(tid, str) and tid:
            latest[tid] = row
    return latest


def record_grade(
    experiment_home: Config | Path,
    run_id: str,
    trial_id: str,
    *,
    met_checks: bool,
    failure_category: str | None = None,
    confident_wrong_reply: bool = False,
    repeated_earlier_error: bool | None = None,
    arm_leaked: bool = False,
    note: str = "",
    tasks_dir: Path | None = None,
    require_checks: bool = True,
) -> dict[str, Any]:
    """Validate and append one grade revision to ``grades.jsonl`` (refused once finalized)."""
    store = UsefulnessRunStore(experiment_home, run_id)
    if store.is_grades_final:
        raise UsefulnessError(
            f"grades are already final for run {run_id!r}; no further grades or corrections allowed"
        )
    if type(met_checks) is not bool:
        raise UsefulnessError("met_checks must be a boolean")
    if type(confident_wrong_reply) is not bool:
        raise UsefulnessError("confident_wrong_reply must be a boolean")
    if type(arm_leaked) is not bool:
        raise UsefulnessError("arm_leaked must be a boolean")
    if repeated_earlier_error is not None and type(repeated_earlier_error) is not bool:
        raise UsefulnessError("repeated_earlier_error must be a boolean or None")
    if met_checks:
        if failure_category is not None:
            raise UsefulnessError("failure_category must be None when met_checks is True")
        if confident_wrong_reply:
            raise UsefulnessError("confident_wrong_reply must be False when met_checks is True")
    else:
        if failure_category not in GRADE_FAILURE_CATEGORIES:
            raise UsefulnessError(
                f"invalid failure_category {failure_category!r}: "
                f"expected one of {GRADE_FAILURE_CATEGORIES}"
            )
    _validate_note(note)

    targets = {
        t.trial_id: t
        for t in discover_run_trials(
            experiment_home, run_id, tasks_dir=tasks_dir, require_checks=require_checks
        )
    }
    if trial_id not in targets:
        raise UsefulnessError(f"trial {trial_id!r} not found in run {run_id!r}")
    target = targets[trial_id]
    if _is_recurrence_task(target.task) and repeated_earlier_error is None:
        raise UsefulnessError(
            f"trial {trial_id!r} is in the recurrence stratum and requires repeated_earlier_error"
        )

    prev = latest_grades(store).get(trial_id)
    prev_rev = int(prev["revision"]) if prev is not None else 0
    row: dict[str, Any] = {
        "schema_version": GRADE_SCHEMA_VERSION,
        "grade_id": f"grade-{uuid.uuid4().hex[:12]}",
        "revision": prev_rev + 1,
        "supersedes_grade_id": prev.get("grade_id") if prev is not None else None,
        "run_id": run_id,
        "trial_id": target.trial_id,
        "task_id": target.task_id,
        "record_sha256": target.record_sha256,
        "checks_sha256": target.checks_sha256,
        "met_checks": met_checks,
        "failure_category": failure_category,
        "confident_wrong_reply": confident_wrong_reply,
        "repeated_earlier_error": repeated_earlier_error,
        "arm_leaked": arm_leaked,
        "note": note,
        "graded_at": utc_now_iso(),
    }
    store.append_grade(row)
    return row


def inspect_grading_status(
    experiment_home: Config | Path,
    run_id: str,
    *,
    tasks_dir: Path | None = None,
    require_checks: bool = False,
) -> GradingStatus:
    """Return read-only, arm-blind grading status for ``run_id``."""
    store = UsefulnessRunStore(experiment_home, run_id)
    targets = discover_run_trials(
        experiment_home, run_id, tasks_dir=tasks_dir, require_checks=require_checks
    )
    all_rows = store.grades()
    latest = latest_grades(store)
    graded_ids = {t.trial_id for t in targets if t.trial_id in latest}
    arm_leaked_count = sum(1 for tid in graded_ids if latest[tid].get("arm_leaked") is True)
    final_sha = store.grades_final_sha256() if store.is_grades_final else None
    attributions_count = len(latest_attributions(store)) if store.attributions_path.exists() else 0
    return GradingStatus(
        run_id=run_id,
        total_trials=len(targets),
        graded_trials=len(graded_ids),
        ungraded_trials=len(targets) - len(graded_ids),
        total_grade_revisions=len(all_rows),
        arm_leaked_count=arm_leaked_count,
        grades_final=store.is_grades_final,
        grades_final_sha256=final_sha,
        attributions_count=attributions_count,
    )


def finalize_grades(
    experiment_home: Config | Path,
    run_id: str,
    *,
    tasks_dir: Path | None = None,
    require_checks: bool = True,
) -> dict[str, Any]:
    """Freeze all grades for ``run_id`` in write-once ``grades-final.json``.

    Refuses if any finished trial lacks a grade or if grades are already final.
    """
    store = UsefulnessRunStore(experiment_home, run_id)
    if store.is_grades_final:
        raise UsefulnessError(
            f"grades are already final for run {run_id!r}: {store.grades_final_path}"
        )
    targets = discover_run_trials(
        experiment_home, run_id, tasks_dir=tasks_dir, require_checks=require_checks
    )
    if not targets:
        raise UsefulnessError(
            f"cannot finalize grades for run {run_id!r}: no finished trials found"
        )
    latest = latest_grades(store)
    missing = [t for t in targets if t.trial_id not in latest]
    if missing:
        raise UsefulnessError(
            f"cannot finalize grades for run {run_id!r}: "
            f"{len(missing)} of {len(targets)} trial(s) still ungraded"
        )
    grades_sha = sha256_file(store.grades_path)
    final_grades: dict[str, dict[str, Any]] = {}
    arm_leaked_count = 0
    for target in sorted(targets, key=lambda t: t.trial_id):
        g = latest[target.trial_id]
        if g.get("record_sha256") != target.record_sha256:
            raise UsefulnessError(
                f"grade for trial {target.trial_id!r} references a different record_sha256"
            )
        if g.get("arm_leaked") is True:
            arm_leaked_count += 1
        final_grades[target.trial_id] = {
            "grade_id": g["grade_id"],
            "revision": g["revision"],
            "task_id": target.task_id,
            "record_sha256": target.record_sha256,
            "checks_sha256": target.checks_sha256,
            "met_checks": g["met_checks"],
            "failure_category": g["failure_category"],
            "confident_wrong_reply": g["confident_wrong_reply"],
            "repeated_earlier_error": g["repeated_earlier_error"],
            "arm_leaked": g["arm_leaked"],
            "note": g.get("note", ""),
            "graded_at": g["graded_at"],
        }

    doc: dict[str, Any] = {
        "schema_version": GRADES_FINAL_SCHEMA_VERSION,
        "run_id": run_id,
        "finalized_at": utc_now_iso(),
        "grades_sha256": grades_sha,
        "trial_count": len(final_grades),
        "total_grade_revisions": len(store.grades()),
        "arm_leaked_count": arm_leaked_count,
        "final_grades": final_grades,
    }
    final_sha = store.write_grades_final(doc)
    return {
        **doc,
        "grades_final_sha256": final_sha,
        "grades_final_path": str(store.grades_final_path),
    }


def verify_finalized_grades(store: UsefulnessRunStore) -> tuple[dict[str, Any], str]:
    """Verify ``grades-final.json`` exists and still matches ``grades.jsonl`` byte for byte."""
    if not store.is_grades_final:
        raise UsefulnessError(
            f"grades are not final for run {store.run_id!r}; "
            "finalize every grade with `mem usefulness finalize-grades` first"
        )
    final_doc = store.read_grades_final()
    final_sha = store.grades_final_sha256()
    if not store.grades_path.is_file():
        raise UsefulnessError(f"grades.jsonl missing for finalized run {store.run_id!r}")
    live_grades_sha = sha256_file(store.grades_path)
    if live_grades_sha != final_doc.get("grades_sha256"):
        raise UsefulnessError(
            f"grades.jsonl hash ({live_grades_sha}) does not match "
            f"grades-final.json ({final_doc.get('grades_sha256')})"
        )
    return final_doc, final_sha


# ---------------------------------------------------------------------------
# BlindGradingController (shared label_frontend.drive controller)
# ---------------------------------------------------------------------------


def _grade_to_raw_answers(row: Mapping[str, Any], *, is_recurrence: bool) -> dict[str, str]:
    ans: dict[str, str] = {
        "met_checks": "y" if row["met_checks"] else "n",
        "arm_leaked": "y" if row.get("arm_leaked") else "n",
    }
    if not row["met_checks"]:
        if isinstance(row.get("failure_category"), str):
            ans["failure_category"] = row["failure_category"]
        ans["confident_wrong_reply"] = "y" if row.get("confident_wrong_reply") else "n"
    if is_recurrence and isinstance(row.get("repeated_earlier_error"), bool):
        ans["repeated_earlier_error"] = "y" if row["repeated_earlier_error"] else "n"
    if row.get("note"):
        ans["note"] = str(row["note"])
    return ans


def _grade_steps_for(answers: Mapping[str, str], *, is_recurrence: bool) -> list[str]:
    steps = ["met_checks"]
    if answers.get("met_checks") == "n":
        steps.append("failure_category")
        steps.append("confident_wrong_reply")
    if is_recurrence:
        steps.append("repeated_earlier_error")
    steps.append("arm_leaked")
    steps.append("note")
    return steps


def _grade_question(step: str) -> tuple[str, list[Option]]:
    if step == "met_checks":
        return "Did this trial meet the task's checks and required behavior?", _YN_OPTIONS
    if step == "failure_category":
        return "Which kind of failure was this?", [
            Option(cat, GRADE_FAILURE_LABELS[cat]) for cat in GRADE_FAILURE_CATEGORIES
        ]
    if step == "confident_wrong_reply":
        return (
            "Did the final reply present the task as done or a fact as known\n"
            "without stating doubt or failure?",
            _YN_OPTIONS,
        )
    if step == "repeated_earlier_error":
        return "Did this trial repeat the task's pre-specified earlier error?", _YN_OPTIONS
    if step == "arm_leaked":
        return "Did anything in the shown material give the arm away?", _ARM_LEAKED_OPTIONS
    if step == "note":
        return "Add a note?", [Option("edit", "Add/edit note"), Option("continue", "Continue")]
    if step == "summary":
        return "Save this grade?", [Option("save", "Save"), Option("redo", "Redo")]
    if step == "review":
        return "Keep the current grade?", [Option("keep", "Keep"), Option("redo", "Redo")]
    raise JudgeError(f"unknown blind grading step: {step}")


def _format_grade_summary_lines(
    answers: Mapping[str, str],
    *,
    is_recurrence: bool,
    previous_revision: int | None = None,
) -> str:
    met = answers.get("met_checks") == "y"
    lines: list[str] = []
    if previous_revision is not None:
        lines.append(f"Current saved revision: {previous_revision}")
    lines.append(f"Met checks: {'Yes' if met else 'No'}")
    if not met:
        cat = answers.get("failure_category", "")
        lines.append(f"Failure kind: {GRADE_FAILURE_LABELS.get(cat, cat)}")
        cwr = answers.get("confident_wrong_reply") == "y"
        lines.append(f"Confident wrong reply: {'Yes' if cwr else 'No'}")
    if is_recurrence:
        rep = answers.get("repeated_earlier_error") == "y"
        lines.append(f"Repeated earlier error: {'Yes' if rep else 'No'}")
    leaked = answers.get("arm_leaked") == "y"
    lines.append(f"Arm leaked by material: {'Yes' if leaked else 'No'}")
    if answers.get("note"):
        lines.append("Note: " + display_safe(answers["note"]))
    return "\n".join(lines)


class BlindGradingController:
    """Controller for blind trial grading, driven by ``judge.label_frontend.drive``."""

    def __init__(
        self,
        experiment_home: Config | Path,
        run_id: str,
        *,
        seed: int = 1,
        limit: int | None = None,
        review: bool = False,
        trial_ids: Sequence[str] = (),
        tasks_dir: Path | None = None,
        require_checks: bool = True,
    ) -> None:
        self.store = UsefulnessRunStore(experiment_home, run_id)
        if self.store.is_grades_final:
            raise UsefulnessError(
                f"grades are already final for run {run_id!r} ({self.store.grades_final_path}); "
                "blind grading and corrections are closed"
            )
        self.seed = seed
        self.review = review
        self.all_targets = discover_run_trials(
            experiment_home, run_id, tasks_dir=tasks_dir, require_checks=require_checks
        )
        self.existing = latest_grades(self.store)
        by_id = {t.trial_id: t for t in self.all_targets}

        if trial_ids:
            missing = [tid for tid in trial_ids if tid not in by_id]
            if missing:
                raise UsefulnessError(f"unknown trial_id(s) in run {run_id!r}: {missing}")
            selected = [by_id[tid] for tid in dict.fromkeys(trial_ids)]
        elif review:
            selected = [t for t in self.all_targets if t.trial_id in self.existing]
        else:
            selected = [t for t in self.all_targets if t.trial_id not in self.existing]

        self.todo = order_trials_blind(selected, seed=seed)
        if limit is not None:
            if limit < 1:
                raise UsefulnessError("limit must be at least 1")
            self.todo = self.todo[:limit]

        self.index = 0
        self.graded = 0
        self.skipped = 0
        self.changed = 0
        self.stopped = False
        self.notice = BLIND_GRADING_INTRO
        self.answers: dict[str, str] = {}
        self.position = 0
        self.mode = ""
        self._evidence = ""
        self._full_bundle = ""
        self._start()

    @property
    def done(self) -> bool:
        return self.stopped or self.index >= len(self.todo)

    @property
    def target(self) -> TrialGradingTarget:
        return self.todo[self.index]

    @property
    def is_recurrence(self) -> bool:
        return _is_recurrence_task(self.target.task)

    @property
    def plan(self) -> list[str]:
        return _grade_steps_for(self.answers, is_recurrence=self.is_recurrence)

    @property
    def step(self) -> str:
        if self.mode == "review":
            return "review"
        return self.plan[self.position] if self.position < len(self.plan) else "summary"

    def _start(self) -> None:
        if self.done:
            return
        t = self.target
        prev = self.existing.get(t.trial_id)
        self._evidence = format_blind_trial_evidence(t, trials_dir=self.store.trials_dir)
        self._full_bundle = format_blind_trial_full(t, trials_dir=self.store.trials_dir)
        self.answers = (
            _grade_to_raw_answers(prev, is_recurrence=self.is_recurrence)
            if prev is not None
            else {}
        )
        self.position = 0
        self.mode = "review" if self.review and prev is not None else ""

    def screen(self) -> Screen:
        if self.done:
            raise JudgeError("this grading sitting has ended")
        t = self.target
        step = self.step
        prev = self.existing.get(t.trial_id)
        prev_rev = int(prev["revision"]) if prev is not None else None
        title, options = _grade_question(step)

        if step == "review":
            subject = "Review"
            evidence = (
                _format_grade_summary_lines(
                    self.answers, is_recurrence=self.is_recurrence, previous_revision=prev_rev
                )
                + "\n\n"
                + self._evidence
            )
            selected = "keep"
        elif step == "summary":
            subject = "Summary"
            evidence = (
                _format_grade_summary_lines(
                    self.answers, is_recurrence=self.is_recurrence, previous_revision=prev_rev
                )
                + "\n\n"
                + self._evidence
            )
            selected = "save"
        elif step == "note":
            subject = "Notes"
            note_preview = (
                f"Note: {display_safe(self.answers['note'])}\n\n"
                if self.answers.get("note")
                else ""
            )
            evidence = note_preview + self._evidence
            selected = "continue" if "note" in self.answers else options[0].id
        else:
            subject = "Trial"
            evidence = self._evidence
            selected = self.answers.get(step, options[0].id)

        return Screen(
            item=self.index + 1,
            total=len(self.todo),
            subject=subject,
            key=f"{self.index + 1}:{step}",
            metadata=display_safe(_blind_metadata(t, revision=prev_rev)),
            evidence=display_safe(evidence),
            projection=display_safe(self._full_bundle),
            question=title,
            options=options,
            selected=selected,
            note=display_safe(self.answers.get("note", "")),
            can_back=step != "review" and self.position > 0,
            can_redo=step in ("summary", "review"),
            notice=self.notice,
            guide=BLIND_GRADING_GUIDE,
            drafts=[],
        )

    def apply(self, action: Action) -> None:
        if self.done:
            raise JudgeError("this grading sitting has ended")
        screen = self.screen()
        step = self.step
        self.notice = ""
        if action.action not in ("answer", "note") and action.value:
            raise JudgeError("navigation action must not contain a value")
        if action.action == "quit":
            self.stopped = True
            return
        if action.action == "skip":
            self.skipped += 1
            self._advance()
            return
        if action.action == "back" and screen.can_back:
            self.position -= 1
            return
        if action.action == "redo" and screen.can_redo:
            self.mode = ""
            self.position = 0
            return
        if action.action == "note" and step == "note":
            try:
                _validate_note(action.value)
            except UsefulnessError as exc:
                raise JudgeError(str(exc)) from None
            self.answers["note"] = action.value
            return
        if action.action == "answer" and action.value in {o.id for o in screen.options}:
            if step == "review":
                if action.value == "keep":
                    self._advance()
                else:
                    self.mode = ""
                    self.position = 0
                return
            if step == "summary":
                if action.value == "redo":
                    self.position = 0
                else:
                    self._save()
                return
            if step == "note":
                if action.value != "continue":
                    raise JudgeError("edit notes through the note input")
                self.position += 1
                return
            self.answers[step] = action.value
            # Drop dependent answers when switching met_checks to Yes.
            if step == "met_checks" and action.value == "y":
                self.answers.pop("failure_category", None)
                self.answers.pop("confident_wrong_reply", None)
            self.position += 1
            return
        raise JudgeError("action is not available on this screen")

    def _advance(self) -> None:
        self.index += 1
        self.position = 0
        self.mode = ""
        self.answers = {}
        self._start()

    def _save(self) -> None:
        t = self.target
        prev = self.existing.get(t.trial_id)
        prev_rev = int(prev["revision"]) if prev is not None else 0
        met = self.answers["met_checks"] == "y"
        failure_cat = None if met else self.answers["failure_category"]
        cwr = False if met else self.answers["confident_wrong_reply"] == "y"
        rep = self.answers["repeated_earlier_error"] == "y" if self.is_recurrence else None
        leaked = self.answers["arm_leaked"] == "y"
        note = self.answers.get("note", "")
        row: dict[str, Any] = {
            "schema_version": GRADE_SCHEMA_VERSION,
            "grade_id": f"grade-{uuid.uuid4().hex[:12]}",
            "revision": prev_rev + 1,
            "supersedes_grade_id": prev.get("grade_id") if prev is not None else None,
            "run_id": self.store.run_id,
            "trial_id": t.trial_id,
            "task_id": t.task_id,
            "record_sha256": t.record_sha256,
            "checks_sha256": t.checks_sha256,
            "met_checks": met,
            "failure_category": failure_cat,
            "confident_wrong_reply": cwr,
            "repeated_earlier_error": rep,
            "arm_leaked": leaked,
            "note": note,
            "graded_at": utc_now_iso(),
        }
        try:
            self.store.append_grade(row)
        except (OSError, UsefulnessError):
            self.notice = "Save failed. Earlier saved grades are safe. Retry Save or quit."
            return
        self.existing[t.trial_id] = row
        self.graded += 1
        if prev is not None:
            self.changed += 1
        self.notice = "Saved."
        self._advance()

    def summary(self) -> GradingSummary:
        remaining = sum(1 for t in self.all_targets if t.trial_id not in self.existing)
        return GradingSummary(
            graded=self.graded,
            skipped=self.skipped,
            changed=self.changed,
            stopped=self.stopped,
            remaining=remaining,
        )

    def result_text(self) -> str:
        s = self.summary()
        verb = "Stopped." if s.stopped else "Done."
        parts = [f"Graded {s.graded}"]
        if s.changed:
            parts.append(f"updated {s.changed}")
        if s.skipped:
            parts.append(f"skipped {s.skipped}")
        parts.append(f"{s.remaining} ungraded remaining")
        return f"{verb} " + ", ".join(parts) + "."


# ---------------------------------------------------------------------------
# Post-grading memory-use attribution (locked until grades-final.json exists)
# ---------------------------------------------------------------------------


def _read_trial_memories_readonly(
    trial_home_raw: Any,
    memory_ids: Sequence[str],
) -> list[dict[str, Any]]:
    """Read recalled/cited memories read-only from a trial's own ``memory.duckdb``."""
    if not memory_ids or not isinstance(trial_home_raw, str) or not trial_home_raw:
        return []
    db_path = Path(trial_home_raw) / "memory.duckdb"
    if not db_path.is_file():
        return []
    unique_ids = list(dict.fromkeys(memory_ids))
    conn: duckdb.DuckDBPyConnection | None = None
    try:
        conn = duckdb.connect(str(db_path), read_only=True)
        placeholders = ", ".join("?" for _ in unique_ids)
        rows = conn.execute(
            f"""
            SELECT CAST(m.memory_id AS VARCHAR), m.type, m.scope, m.key, m.content,
                   m.trust_level, m.valid_from, m.valid_to, m.expires_at, ms.status
            FROM memories m
            LEFT JOIN memory_status ms ON ms.memory_id = m.memory_id
            WHERE CAST(m.memory_id AS VARCHAR) IN ({placeholders})
            """,
            unique_ids,
        ).fetchall()
    except duckdb.Error:
        return []
    finally:
        if conn is not None:
            conn.close()

    by_id: dict[str, dict[str, Any]] = {}
    for r in rows:
        by_id[str(r[0])] = {
            "memory_id": str(r[0]),
            "type": str(r[1]) if r[1] is not None else "",
            "scope": str(r[2]) if r[2] is not None else "",
            "key": str(r[3]) if r[3] is not None else "",
            "content": str(r[4]) if r[4] is not None else "",
            "trust_level": str(r[5]) if r[5] is not None else "",
            "valid_from": str(r[6]) if r[6] is not None else None,
            "valid_to": str(r[7]) if r[7] is not None else None,
            "expires_at": str(r[8]) if r[8] is not None else None,
            "status": str(r[9]) if r[9] is not None else "unknown",
        }
    return [by_id[mid] for mid in unique_ids if mid in by_id]


def _format_attribution_bundle(
    target: TrialGradingTarget,
    final_grade: Mapping[str, Any],
    *,
    trials_dir: Path,
    max_section_chars: int,
) -> str:
    """Format unblinded trial details, recalls, citations, and memories for attribution."""
    lines: list[str] = [
        "=== FINALIZED GRADE (READ-ONLY) ===",
        f"Trial: {target.trial_id}  Task: {target.task_id}  Arm: {target.arm}",
        f"Met checks: {'Yes' if final_grade['met_checks'] else 'No'}",
    ]
    if not final_grade["met_checks"]:
        cat = str(final_grade.get("failure_category") or "")
        lines.append(f"Failure kind: {GRADE_FAILURE_LABELS.get(cat, cat)}")
        lines.append(
            f"Confident wrong reply: {'Yes' if final_grade.get('confident_wrong_reply') else 'No'}"
        )
    if final_grade.get("arm_leaked"):
        lines.append("Arm leaked during blind grading: Yes")
    if final_grade.get("note"):
        lines.append(f"Blind grade note: {final_grade['note']}")

    # Recall accounting and recalled memories from trial home
    ra = target.record.get("recall_accounting")
    memory_ids: list[str] = []
    lines.append("\n=== RECALLS AND CITATIONS ===")
    if isinstance(ra, Mapping):
        lines.append(
            f"Recalls: total={ra.get('total_recalls', 0)} "
            f"(returned={ra.get('returned_count', 0)}, miss={ra.get('miss_count', 0)}, "
            f"degraded={ra.get('degraded_count', 0)}, "
            f"missing_record={ra.get('missing_record_count', 0)})"
        )
        for idx, rec in enumerate(ra.get("recalls") or (), 1):
            if not isinstance(rec, Mapping):
                continue
            rmids = [str(m) for m in (rec.get("returned_memory_ids") or ())]
            memory_ids.extend(rmids)
            lines.append(
                f"  [{idx}] op={rec.get('operation')} outcome={rec.get('outcome')} "
                f"returned={rec.get('returned_count')} ids={rmids or '[]'}"
            )
        cit = ra.get("citations")
        if isinstance(cit, Mapping):
            cited_ids = [str(m) for m in (cit.get("cited_memory_ids") or ())]
            memory_ids.extend(cited_ids)
            lines.append(
                f"Citations: made={cit.get('citations_made', 0)} "
                f"returned_cited={cit.get('returned_memories_cited_count', 0)} "
                f"cited_ids={cited_ids or '[]'}"
            )
    else:
        lines.append("(no recall accounting recorded)")

    paths = target.record.get("paths")
    trial_home_raw = paths.get("trial_home") if isinstance(paths, Mapping) else None
    memories = _read_trial_memories_readonly(trial_home_raw, memory_ids)
    if memories:
        lines.append("\n=== RECALLED / CITED MEMORIES (FROM TRIAL HOME) ===")
        for mem in memories:
            bounds: list[str] = []
            if mem["valid_from"]:
                bounds.append(f"from={mem['valid_from']}")
            if mem["valid_to"]:
                bounds.append(f"to={mem['valid_to']}")
            if mem["expires_at"]:
                bounds.append(f"expires={mem['expires_at']}")
            validity = f" ({', '.join(bounds)})" if bounds else ""
            lines.append(
                f"- {mem['memory_id']} [{mem['status']}] type={mem['type']} "
                f"scope={mem['scope']} trust={mem['trust_level']}{validity}"
            )
            if mem["key"]:
                lines.append(f"  key: {mem['key']}")
            if mem["content"]:
                lines.append(
                    "  content: " + _truncate_text(mem["content"].strip(), 2000, "memory content")
                )

    # Tool calls mentioning memory tools
    mem_calls = [
        tc
        for tc in (target.record.get("tool_calls") or ())
        if isinstance(tc, Mapping) and "memory" in str(tc.get("name", "")).lower()
    ]
    if mem_calls:
        lines.append("\n=== MEMORY TOOL CALLS ===")
        for tc in mem_calls:
            lines.append(f"- {tc.get('name')} (is_error={tc.get('is_error')})")

    # Trial output (request, checks, repo changes, last reply)
    lines.append(
        "\n"
        + _format_blind_trial_bundle(
            target, trials_dir=trials_dir, max_section_chars=max_section_chars
        )
    )
    return "\n".join(lines)


def record_attribution(
    experiment_home: Config | Path,
    run_id: str,
    trial_id: str,
    *,
    harmful_memory_use: bool,
    attempted_invalid_memory_use: bool = False,
    confident_wrong_linked_to_memory: bool = False,
    note: str = "",
    tasks_dir: Path | None = None,
    require_checks: bool = False,
) -> dict[str, Any]:
    """Append one post-grading memory-use attribution to ``attributions.jsonl``.

    Refuses before ``grades-final.json`` exists or if ``grades.jsonl`` no longer
    matches ``grades-final.json``. Changes no grade.
    """
    store = UsefulnessRunStore(experiment_home, run_id)
    final_doc, final_sha = verify_finalized_grades(store)
    if type(harmful_memory_use) is not bool:
        raise UsefulnessError("harmful_memory_use must be a boolean")
    if type(attempted_invalid_memory_use) is not bool:
        raise UsefulnessError("attempted_invalid_memory_use must be a boolean")
    if type(confident_wrong_linked_to_memory) is not bool:
        raise UsefulnessError("confident_wrong_linked_to_memory must be a boolean")
    _validate_note(note)

    targets = {
        t.trial_id: t
        for t in discover_run_trials(
            experiment_home, run_id, tasks_dir=tasks_dir, require_checks=require_checks
        )
    }
    if trial_id not in targets:
        raise UsefulnessError(f"trial {trial_id!r} not found in run {run_id!r}")
    target = targets[trial_id]
    if target.arm != "memory":
        raise UsefulnessError(
            f"trial {trial_id!r} is in arm {target.arm!r}; "
            "memory-use attribution applies only to memory-arm trials"
        )
    final_grades = final_doc.get("final_grades") or {}
    if trial_id not in final_grades:
        raise UsefulnessError(f"trial {trial_id!r} is absent from grades-final.json")
    fg = final_grades[trial_id]
    if confident_wrong_linked_to_memory and not fg.get("confident_wrong_reply"):
        raise UsefulnessError(
            f"trial {trial_id!r} was not graded as a confident wrong reply; "
            "confident_wrong_linked_to_memory must be False"
        )

    prev = latest_attributions(store).get(trial_id)
    prev_rev = int(prev["revision"]) if prev is not None else 0
    row: dict[str, Any] = {
        "schema_version": ATTRIBUTION_SCHEMA_VERSION,
        "attribution_id": f"attr-{uuid.uuid4().hex[:12]}",
        "revision": prev_rev + 1,
        "supersedes_attribution_id": prev.get("attribution_id") if prev is not None else None,
        "run_id": run_id,
        "trial_id": target.trial_id,
        "task_id": target.task_id,
        "arm": target.arm,
        "grades_final_sha256": final_sha,
        "grade_id": fg["grade_id"],
        "grade_revision": fg["revision"],
        "harmful_memory_use": harmful_memory_use,
        "attempted_invalid_memory_use": attempted_invalid_memory_use,
        "confident_wrong_linked_to_memory": confident_wrong_linked_to_memory,
        "note": note,
        "attributed_at": utc_now_iso(),
    }
    store.append_attribution(row)
    return row


def _attribution_steps_for(*, is_confident_wrong: bool) -> list[str]:
    steps = ["harmful_memory_use", "attempted_invalid_memory_use"]
    if is_confident_wrong:
        steps.append("confident_wrong_linked_to_memory")
    steps.append("note")
    return steps


def _attribution_question(step: str) -> tuple[str, list[Option]]:
    if step == "harmful_memory_use":
        return (
            "Was a declared or observed use of a memory linked to a concrete wrong action?",
            _YN_OPTIONS,
        )
    if step == "attempted_invalid_memory_use":
        return (
            "Did this trial attempt to use a retired, expired, or out-of-scope memory?",
            _YN_OPTIONS,
        )
    if step == "confident_wrong_linked_to_memory":
        return (
            "Was the wrong part of the confident wrong reply linked to a declared\n"
            "or observed use of a memory?",
            _YN_OPTIONS,
        )
    if step == "note":
        return "Add a note?", [Option("edit", "Add/edit note"), Option("continue", "Continue")]
    if step == "summary":
        return "Save this memory-use attribution?", [Option("save", "Save"), Option("redo", "Redo")]
    if step == "review":
        return "Keep the current attribution?", [Option("keep", "Keep"), Option("redo", "Redo")]
    raise JudgeError(f"unknown attribution step: {step}")


def _attribution_to_raw_answers(
    row: Mapping[str, Any], *, is_confident_wrong: bool
) -> dict[str, str]:
    ans: dict[str, str] = {
        "harmful_memory_use": "y" if row.get("harmful_memory_use") else "n",
        "attempted_invalid_memory_use": "y" if row.get("attempted_invalid_memory_use") else "n",
    }
    if is_confident_wrong:
        ans["confident_wrong_linked_to_memory"] = (
            "y" if row.get("confident_wrong_linked_to_memory") else "n"
        )
    if row.get("note"):
        ans["note"] = str(row["note"])
    return ans


def _format_attribution_summary_lines(
    answers: Mapping[str, str],
    *,
    is_confident_wrong: bool,
    previous_revision: int | None = None,
) -> str:
    lines: list[str] = []
    if previous_revision is not None:
        lines.append(f"Current saved attribution revision: {previous_revision}")
    lines.append(
        "Harmful memory use (wrong action): "
        + ("Yes" if answers.get("harmful_memory_use") == "y" else "No")
    )
    lines.append(
        "Attempted retired/expired/out-of-scope memory use: "
        + ("Yes" if answers.get("attempted_invalid_memory_use") == "y" else "No")
    )
    if is_confident_wrong:
        lines.append(
            "Confident wrong reply linked to memory: "
            + ("Yes" if answers.get("confident_wrong_linked_to_memory") == "y" else "No")
        )
    if answers.get("note"):
        lines.append("Note: " + display_safe(answers["note"]))
    return "\n".join(lines)


class MemoryAttributionController:
    """Post-grading memory-use attribution controller for ``judge.label_frontend.drive``.

    Refuses to open until every grade of the run is final (``grades-final.json``),
    and writes only to ``attributions.jsonl`` without altering any grade.
    """

    def __init__(
        self,
        experiment_home: Config | Path,
        run_id: str,
        *,
        seed: int = 1,
        limit: int | None = None,
        review: bool = False,
        include_passed: bool = False,
        trial_ids: Sequence[str] = (),
        tasks_dir: Path | None = None,
        require_checks: bool = False,
    ) -> None:
        self.store = UsefulnessRunStore(experiment_home, run_id)
        self.final_doc, self.grades_final_sha256 = verify_finalized_grades(self.store)
        self.final_grades: dict[str, dict[str, Any]] = dict(
            self.final_doc.get("final_grades") or {}
        )
        self.review = review
        all_targets = discover_run_trials(
            experiment_home, run_id, tasks_dir=tasks_dir, require_checks=require_checks
        )
        memory_targets = [
            t for t in all_targets if t.arm == "memory" and t.trial_id in self.final_grades
        ]
        if include_passed:
            eligible = memory_targets
        else:
            eligible = [
                t
                for t in memory_targets
                if not self.final_grades[t.trial_id]["met_checks"]
                or self.final_grades[t.trial_id].get("confident_wrong_reply")
            ]
        self.eligible = eligible
        self.existing = latest_attributions(self.store)
        by_id = {t.trial_id: t for t in memory_targets}

        if trial_ids:
            missing = [tid for tid in trial_ids if tid not in by_id]
            if missing:
                raise UsefulnessError(
                    f"unknown or non-memory trial_id(s) in run {run_id!r}: {missing}"
                )
            selected = [by_id[tid] for tid in dict.fromkeys(trial_ids)]
        elif review:
            selected = [t for t in eligible if t.trial_id in self.existing]
        else:
            selected = [t for t in eligible if t.trial_id not in self.existing]

        rng = random.Random(seed)
        self.todo = sorted(selected, key=lambda t: t.trial_id)
        rng.shuffle(self.todo)
        if limit is not None:
            if limit < 1:
                raise UsefulnessError("limit must be at least 1")
            self.todo = self.todo[:limit]

        self.index = 0
        self.attributed = 0
        self.skipped = 0
        self.changed = 0
        self.stopped = False
        self.notice = ATTRIBUTION_INTRO
        self.answers: dict[str, str] = {}
        self.position = 0
        self.mode = ""
        self._evidence = ""
        self._full_bundle = ""
        self._start()

    @property
    def done(self) -> bool:
        return self.stopped or self.index >= len(self.todo)

    @property
    def target(self) -> TrialGradingTarget:
        return self.todo[self.index]

    @property
    def current_grade(self) -> Mapping[str, Any]:
        return self.final_grades[self.target.trial_id]

    @property
    def is_confident_wrong(self) -> bool:
        return bool(self.current_grade.get("confident_wrong_reply"))

    @property
    def plan(self) -> list[str]:
        return _attribution_steps_for(is_confident_wrong=self.is_confident_wrong)

    @property
    def step(self) -> str:
        if self.mode == "review":
            return "review"
        return self.plan[self.position] if self.position < len(self.plan) else "summary"

    def _start(self) -> None:
        if self.done:
            return
        t = self.target
        fg = self.current_grade
        prev = self.existing.get(t.trial_id)
        self._evidence = _format_attribution_bundle(
            t, fg, trials_dir=self.store.trials_dir, max_section_chars=MAX_EVIDENCE_SECTION_CHARS
        )
        self._full_bundle = _truncate_text(
            _format_attribution_bundle(
                t, fg, trials_dir=self.store.trials_dir, max_section_chars=MAX_PROJECTION_CHARS // 4
            ),
            MAX_PROJECTION_CHARS,
            "full attribution bundle",
        )
        self.answers = (
            _attribution_to_raw_answers(prev, is_confident_wrong=self.is_confident_wrong)
            if prev is not None
            else {}
        )
        self.position = 0
        self.mode = "review" if self.review and prev is not None else ""

    def screen(self) -> Screen:
        if self.done:
            raise JudgeError("this attribution sitting has ended")
        t = self.target
        step = self.step
        prev = self.existing.get(t.trial_id)
        prev_rev = int(prev["revision"]) if prev is not None else None
        title, options = _attribution_question(step)

        grade_tag = (
            "passed"
            if self.current_grade["met_checks"]
            else self.current_grade.get("failure_category")
        )
        meta_parts = [
            f"Trial: {t.trial_id}",
            f"Task: {t.task_id}",
            "Arm: memory",
            f"Grade: {grade_tag}",
        ]
        if prev_rev is not None:
            meta_parts.append(f"Saved rev: {prev_rev}")

        if step == "review":
            subject = "Review"
            evidence = (
                _format_attribution_summary_lines(
                    self.answers,
                    is_confident_wrong=self.is_confident_wrong,
                    previous_revision=prev_rev,
                )
                + "\n\n"
                + self._evidence
            )
            selected = "keep"
        elif step == "summary":
            subject = "Summary"
            evidence = (
                _format_attribution_summary_lines(
                    self.answers,
                    is_confident_wrong=self.is_confident_wrong,
                    previous_revision=prev_rev,
                )
                + "\n\n"
                + self._evidence
            )
            selected = "save"
        elif step == "note":
            subject = "Notes"
            note_preview = (
                f"Note: {display_safe(self.answers['note'])}\n\n"
                if self.answers.get("note")
                else ""
            )
            evidence = note_preview + self._evidence
            selected = "continue" if "note" in self.answers else options[0].id
        else:
            subject = "Trial"
            evidence = self._evidence
            selected = self.answers.get(step, options[0].id)

        return Screen(
            item=self.index + 1,
            total=len(self.todo),
            subject=subject,
            key=f"{self.index + 1}:{step}",
            metadata=display_safe(" · ".join(meta_parts)),
            evidence=display_safe(evidence),
            projection=display_safe(self._full_bundle),
            question=title,
            options=options,
            selected=selected,
            note=display_safe(self.answers.get("note", "")),
            can_back=step != "review" and self.position > 0,
            can_redo=step in ("summary", "review"),
            notice=self.notice,
            guide=ATTRIBUTION_GUIDE,
            drafts=[],
        )

    def apply(self, action: Action) -> None:
        if self.done:
            raise JudgeError("this attribution sitting has ended")
        screen = self.screen()
        step = self.step
        self.notice = ""
        if action.action not in ("answer", "note") and action.value:
            raise JudgeError("navigation action must not contain a value")
        if action.action == "quit":
            self.stopped = True
            return
        if action.action == "skip":
            self.skipped += 1
            self._advance()
            return
        if action.action == "back" and screen.can_back:
            self.position -= 1
            return
        if action.action == "redo" and screen.can_redo:
            self.mode = ""
            self.position = 0
            return
        if action.action == "note" and step == "note":
            try:
                _validate_note(action.value)
            except UsefulnessError as exc:
                raise JudgeError(str(exc)) from None
            self.answers["note"] = action.value
            return
        if action.action == "answer" and action.value in {o.id for o in screen.options}:
            if step == "review":
                if action.value == "keep":
                    self._advance()
                else:
                    self.mode = ""
                    self.position = 0
                return
            if step == "summary":
                if action.value == "redo":
                    self.position = 0
                else:
                    self._save()
                return
            if step == "note":
                if action.value != "continue":
                    raise JudgeError("edit notes through the note input")
                self.position += 1
                return
            self.answers[step] = action.value
            self.position += 1
            return
        raise JudgeError("action is not available on this screen")

    def _advance(self) -> None:
        self.index += 1
        self.position = 0
        self.mode = ""
        self.answers = {}
        self._start()

    def _save(self) -> None:
        t = self.target
        fg = self.current_grade
        prev = self.existing.get(t.trial_id)
        prev_rev = int(prev["revision"]) if prev is not None else 0
        row: dict[str, Any] = {
            "schema_version": ATTRIBUTION_SCHEMA_VERSION,
            "attribution_id": f"attr-{uuid.uuid4().hex[:12]}",
            "revision": prev_rev + 1,
            "supersedes_attribution_id": prev.get("attribution_id") if prev is not None else None,
            "run_id": self.store.run_id,
            "trial_id": t.trial_id,
            "task_id": t.task_id,
            "arm": t.arm,
            "grades_final_sha256": self.grades_final_sha256,
            "grade_id": fg["grade_id"],
            "grade_revision": fg["revision"],
            "harmful_memory_use": self.answers["harmful_memory_use"] == "y",
            "attempted_invalid_memory_use": self.answers["attempted_invalid_memory_use"] == "y",
            "confident_wrong_linked_to_memory": (
                self.answers["confident_wrong_linked_to_memory"] == "y"
                if self.is_confident_wrong
                else False
            ),
            "note": self.answers.get("note", ""),
            "attributed_at": utc_now_iso(),
        }
        try:
            self.store.append_attribution(row)
        except (OSError, UsefulnessError):
            self.notice = "Save failed. Earlier saved attributions are safe. Retry Save or quit."
            return
        self.existing[t.trial_id] = row
        self.attributed += 1
        if prev is not None:
            self.changed += 1
        self.notice = "Saved."
        self._advance()

    def summary(self) -> AttributionSummary:
        remaining = sum(1 for t in self.eligible if t.trial_id not in self.existing)
        return AttributionSummary(
            attributed=self.attributed,
            skipped=self.skipped,
            changed=self.changed,
            stopped=self.stopped,
            remaining=remaining,
        )

    def result_text(self) -> str:
        s = self.summary()
        verb = "Stopped." if s.stopped else "Done."
        parts = [f"Attributed {s.attributed}"]
        if s.changed:
            parts.append(f"updated {s.changed}")
        if s.skipped:
            parts.append(f"skipped {s.skipped}")
        parts.append(f"{s.remaining} remaining")
        return f"{verb} " + ", ".join(parts) + "."


# ---------------------------------------------------------------------------
# Session entry points using the shared judge.label_frontend.drive loop
# ---------------------------------------------------------------------------


def grade_session(
    experiment_home: Config | Path,
    run_id: str,
    *,
    plain: bool = False,
    seed: int = 1,
    limit: int | None = None,
    review: bool = False,
    trial_ids: Sequence[str] = (),
    tasks_dir: Path | None = None,
    require_checks: bool = True,
    progress: Callable[[str], None] | None = None,
) -> GradingSummary:
    """Run an interactive blind trial grading sitting via the shared labeling frontend."""
    controller = BlindGradingController(
        experiment_home,
        run_id,
        seed=seed,
        limit=limit,
        review=review,
        trial_ids=trial_ids,
        tasks_dir=tasks_dir,
        require_checks=require_checks,
    )
    drive(controller, plain=plain, progress=progress)
    print(controller.result_text())
    return controller.summary()


def attribution_session(
    experiment_home: Config | Path,
    run_id: str,
    *,
    plain: bool = False,
    seed: int = 1,
    limit: int | None = None,
    review: bool = False,
    include_passed: bool = False,
    trial_ids: Sequence[str] = (),
    tasks_dir: Path | None = None,
    require_checks: bool = False,
    progress: Callable[[str], None] | None = None,
) -> AttributionSummary:
    """Run a post-grading memory-use attribution sitting via the shared labeling frontend."""
    controller = MemoryAttributionController(
        experiment_home,
        run_id,
        seed=seed,
        limit=limit,
        review=review,
        include_passed=include_passed,
        trial_ids=trial_ids,
        tasks_dir=tasks_dir,
        require_checks=require_checks,
    )
    drive(controller, plain=plain, progress=progress)
    print(controller.result_text())
    return controller.summary()
