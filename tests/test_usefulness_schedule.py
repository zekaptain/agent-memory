"""Tests for usefulness pair scheduling, resumable execution, and retry rules (MEM-22)."""

from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from agent_memory.cli import app
from agent_memory.endpoints import Unreachable
from agent_memory.usefulness import (
    HarnessInvocation,
    HarnessResult,
    RetryRule,
    ScheduleRunConfig,
    ScriptedHarnessAdapter,
    UsefulnessError,
    UsefulnessRunStore,
    arm_order_for_attempt,
    build_schedule,
    build_task_manifest,
    create_run_schedule,
    inspect_schedule_progress,
    load_run_schedule,
    run_schedule,
    sha256_file,
    trial_id_for_slot,
)


def _git(repo: Path, *args: str) -> str:
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "Test",
        "GIT_AUTHOR_EMAIL": "test@example.com",
        "GIT_COMMITTER_NAME": "Test",
        "GIT_COMMITTER_EMAIL": "test@example.com",
    }
    proc = subprocess.run(
        ["git", *args],
        cwd=repo,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    return proc.stdout.strip()


def _make_repo(tmp_path: Path) -> tuple[Path, str]:
    repo = tmp_path / "src-repo"
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    (repo / "app.py").write_text("def answer():\n    return 1\n", encoding="utf-8")
    _git(repo, "add", "app.py")
    _git(repo, "commit", "-m", "initial")
    commit = _git(repo, "rev-parse", "HEAD")
    return repo, commit


def _write_task(
    tasks_root: Path,
    task_id: str,
    *,
    repo: Path,
    commit: str,
    work_done_at: str,
    task_family: str,
) -> Path:
    task_dir = tasks_root / task_id
    task_dir.mkdir(parents=True)
    meta = {
        "schema_version": 1,
        "task_id": task_id,
        "task_family": task_family,
        "strata": ["general"],
        "project": "demo-proj",
        "repository": str(repo),
        "commit": commit,
        "work_done_at": work_done_at,
        "checks": {
            "commands": [
                {"name": "ok", "command": [sys.executable, "-c", "print('ok')"]},
            ],
        },
    }
    (task_dir / "task.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (task_dir / "request.md").write_text(f"Complete {task_id}.\n", encoding="utf-8")
    (task_dir / "required_behavior.md").write_text("Pass when ok succeeds.\n", encoding="utf-8")
    return task_dir


def _setup_run_with_tasks(
    tmp_path: Path,
    run_id: str = "run-sched",
) -> tuple[Path, UsefulnessRunStore, dict[str, object]]:
    repo, commit = _make_repo(tmp_path)
    exp_home = tmp_path / "exp"
    tasks_dir = exp_home / "tasks"
    # 1 calibration family, 3 held-out families.
    _write_task(
        tasks_dir,
        "t-cal-1",
        repo=repo,
        commit=commit,
        work_done_at="2026-11-10T12:00:00Z",
        task_family="fam-cal",
    )
    _write_task(
        tasks_dir,
        "t-h1",
        repo=repo,
        commit=commit,
        work_done_at="2026-11-20T12:00:00Z",
        task_family="fam-h1",
    )
    _write_task(
        tasks_dir,
        "t-h2",
        repo=repo,
        commit=commit,
        work_done_at="2026-11-21T12:00:00Z",
        task_family="fam-h2",
    )
    _write_task(
        tasks_dir,
        "t-h3",
        repo=repo,
        commit=commit,
        work_done_at="2026-11-22T12:00:00Z",
        task_family="fam-h3",
    )

    store = UsefulnessRunStore(exp_home, run_id)
    manifest = build_task_manifest(tasks_dir, split_after="2026-11-15T00:00:00Z")
    store.write_task_manifest(manifest)
    return exp_home, store, manifest


def test_schedule_deterministic_from_seed_and_manifest_with_repeats(tmp_path: Path) -> None:
    """Same seed and manifest fix pair order and arm order; repeated tasks yield that many pairs."""
    exp_home, store, manifest = _setup_run_with_tasks(tmp_path)
    manifest_sha = store.task_manifest_sha256()

    # Re-order dictionary keys inside manifest["tasks"] to verify independence from dict order.
    reversed_manifest = dict(manifest)
    tasks_map = manifest["tasks"]
    assert isinstance(tasks_map, dict)
    reversed_manifest["tasks"] = dict(reversed(list(tasks_map.items())))

    s1 = build_schedule(
        manifest,
        run_id="run-sched",
        seed=20261126,
        task_manifest_sha256=manifest_sha,
        split="held_out",
        repeats=1,
        task_repeats={"t-h2": 3},
        created_at="2026-11-26T00:00:00Z",
    )
    s2 = build_schedule(
        reversed_manifest,
        run_id="run-sched",
        seed=20261126,
        task_manifest_sha256=manifest_sha,
        split="held_out",
        repeats=1,
        task_repeats={"t-h2": 3},
        created_at="2026-11-26T00:00:00Z",
    )

    assert s1.to_dict() == s2.to_dict()
    assert [(p.pair_id, p.arm_order) for p in s1.pairs] == [
        (p.pair_id, p.arm_order) for p in s2.pairs
    ]

    # t-h1 (1) + t-h2 (3) + t-h3 (1) = 5 held-out pairs; calibration task excluded.
    assert len(s1.pairs) == 5
    assert [p.task_id for p in s1.pairs].count("t-h2") == 3
    assert [p.task_id for p in s1.pairs].count("t-h1") == 1
    assert [p.task_id for p in s1.pairs].count("t-h3") == 1
    assert all(p.split == "held_out" for p in s1.pairs)
    assert sorted(p.repeat_index for p in s1.pairs if p.task_id == "t-h2") == [0, 1, 2]
    assert sorted(p.pair_id for p in s1.pairs if p.task_id == "t-h2") == [
        "t-h2__r1",
        "t-h2__r2",
        "t-h2__r3",
    ]

    # Different seeds produce different pair orders or arm orders.
    different_found = False
    for alt_seed in range(1, 20):
        s_alt = build_schedule(
            manifest,
            run_id="run-sched",
            seed=alt_seed,
            task_manifest_sha256=manifest_sha,
            split="held_out",
            repeats=1,
            task_repeats={"t-h2": 3},
        )
        if [(p.pair_id, p.arm_order) for p in s_alt.pairs] != [
            (p.pair_id, p.arm_order) for p in s1.pairs
        ]:
            different_found = True
            break
    assert different_found

    # Write-once persistence and validation guards.
    persisted, sha = create_run_schedule(
        exp_home,
        "run-sched",
        seed=20261126,
        split="held_out",
        repeats=1,
        task_repeats={"t-h2": 3},
    )
    loaded, loaded_sha = load_run_schedule(exp_home, "run-sched")
    assert loaded_sha == sha
    assert loaded.pairs == persisted.pairs

    with pytest.raises(UsefulnessError, match="already exists"):
        create_run_schedule(exp_home, "run-sched", seed=20261126, split="held_out")

    # Invalid inputs are refused.
    with pytest.raises(UsefulnessError, match="must be an integer"):
        build_schedule(manifest, run_id="r", seed=True, task_manifest_sha256=manifest_sha)  # type: ignore[arg-type]
    with pytest.raises(UsefulnessError, match="not in split"):
        build_schedule(
            manifest,
            run_id="r",
            seed=1,
            task_manifest_sha256=manifest_sha,
            split="held_out",
            task_repeats={"t-cal-1": 2},
        )
    with pytest.raises(UsefulnessError, match="must declare at least one"):
        RetryRule(max_pair_retries=1, retryable_failure_categories=())


def test_interrupted_run_keeps_and_marks_interrupted_trial_and_resumes_unmade_trials(
    tmp_path: Path,
) -> None:
    """Interrupted trials (both caught and hard-crashed) are kept, marked, and never overwritten."""
    exp_home, store, _ = _setup_run_with_tasks(tmp_path, run_id="run-resume")
    schedule, _ = create_run_schedule(
        exp_home,
        "run-resume",
        seed=99,
        split="held_out",
        repeats=1,
        retry_rule=RetryRule(max_pair_retries=0),
    )

    expected_order = [
        trial_id_for_slot(pair.pair_id, 1, arm) for pair in schedule.pairs for arm in pair.arm_order
    ]
    assert len(expected_order) == 6

    run_cfg = ScheduleRunConfig(
        model="local-model",
        model_base_url="http://127.0.0.1:8080/v1",
    )

    # 1. Preflight refusal (hosted model without opt-in) must NOT mark any trial interrupted.
    hosted_cfg = ScheduleRunConfig(model="claude-sonnet-4-6", model_base_url=None)
    with pytest.raises(Unreachable, match="AGENT_MEMORY_JUDGE_ALLOW_REMOTE"):
        run_schedule(
            exp_home,
            "run-resume",
            hosted_cfg,
            adapter=ScriptedHarnessAdapter(
                lambda inv: HarnessResult(status="completed", last_reply="ok")
            ),
        )
    assert store.schedule_events() == []
    assert inspect_schedule_progress(exp_home, "run-resume").interrupted_trials == 0

    # 2. First execution: trial 0 completes, trial 1 raises KeyboardInterrupt inside the harness.
    seen_pass1: list[str] = []

    def agent_pass1(inv: HarnessInvocation) -> HarnessResult:
        seen_pass1.append(inv.trial_id)
        if inv.trial_id == expected_order[1]:
            raise KeyboardInterrupt("user hit Ctrl-C")
        return HarnessResult(status="completed", last_reply="done")

    res1 = run_schedule(
        exp_home,
        "run-resume",
        run_cfg,
        adapter=ScriptedHarnessAdapter(agent_pass1),
    )
    assert res1.interrupted is True
    assert seen_pass1 == expected_order[:2]
    assert res1.executed_trial_ids == tuple(expected_order[:2])

    # Interrupted trial's record.json is written and marked failure_category="interrupted".
    interrupted_rec_path = store.trials_dir / expected_order[1] / "record.json"
    assert interrupted_rec_path.is_file()
    interrupted_bytes_before = interrupted_rec_path.read_bytes()
    interrupted_sha_before = sha256_file(interrupted_rec_path)
    interrupted_data = json.loads(interrupted_bytes_before)
    assert interrupted_data["status"] == "failed"
    assert interrupted_data["failure_category"] == "interrupted"

    # 3. Also simulate a HARD CRASH on trial 2 (directory + partial artifact, NO record.json,
    #    with trial_started already logged in schedule-events.jsonl).
    hard_crash_tid = expected_order[2]
    hard_crash_dir = store.trials_dir / hard_crash_tid
    hard_crash_dir.mkdir(parents=True)
    partial_file = hard_crash_dir / "partial-trace.txt"
    partial_file.write_text("partial work before SIGKILL\n", encoding="utf-8")
    pair2 = schedule.pairs[1]
    store.append_schedule_event(
        {
            "event": "trial_started",
            "recorded_at": "2026-11-26T00:01:00Z",
            "pair_id": pair2.pair_id,
            "task_id": pair2.task_id,
            "repeat_index": pair2.repeat_index,
            "attempt": 1,
            "arm": pair2.arm_order[0],
            "trial_id": hard_crash_tid,
        }
    )

    # 4. Resume with max_trials=1 first, then finish the remaining trials.
    seen_pass2: list[str] = []

    def agent_resume(inv: HarnessInvocation) -> HarnessResult:
        seen_pass2.append(inv.trial_id)
        return HarnessResult(status="completed", last_reply="resumed")

    res2 = run_schedule(
        exp_home,
        "run-resume",
        run_cfg,
        adapter=ScriptedHarnessAdapter(agent_resume),
        max_trials=1,
    )
    assert res2.interrupted is False
    assert res2.stopped_by_max_trials is True
    assert seen_pass2 == [expected_order[3]]

    # Hard-crashed trial directory and partial file are kept intact and marked via interrupted.json.
    assert partial_file.read_text(encoding="utf-8") == "partial work before SIGKILL\n"
    marker_path = hard_crash_dir / "interrupted.json"
    assert marker_path.is_file()
    marker_data = json.loads(marker_path.read_text(encoding="utf-8"))
    assert marker_data["trial_id"] == hard_crash_tid
    assert marker_data["status"] == "interrupted"
    assert marker_data["failure_category"] == "interrupted"
    assert not (hard_crash_dir / "record.json").exists()

    # 5. Resume to completion.
    seen_pass3: list[str] = []

    def agent_finish(inv: HarnessInvocation) -> HarnessResult:
        seen_pass3.append(inv.trial_id)
        return HarnessResult(status="completed", last_reply="finished")

    res3 = run_schedule(
        exp_home,
        "run-resume",
        run_cfg,
        adapter=ScriptedHarnessAdapter(agent_finish),
    )
    assert res3.progress.is_complete is True
    assert seen_pass3 == expected_order[4:]

    # Neither interrupted trial was ever re-run in place, and trial 1's record is byte-identical.
    assert expected_order[1] not in seen_pass2 + seen_pass3
    assert hard_crash_tid not in seen_pass2 + seen_pass3
    assert interrupted_rec_path.read_bytes() == interrupted_bytes_before
    assert sha256_file(interrupted_rec_path) == interrupted_sha_before

    # Final progress counts reflect 4 completed, 2 interrupted, 0 failed, 0 retries.
    assert res3.progress.completed_trials == 4
    assert res3.progress.interrupted_trials == 2
    assert res3.progress.failed_trials == 0
    assert res3.progress.retried_pairs == 0


def test_retry_rule_strictly_governs_pair_retries(tmp_path: Path) -> None:
    """Whether a pair is run again follows RetryRule and nothing else, with fresh attempt IDs."""
    exp_home, store, _ = _setup_run_with_tasks(tmp_path, run_id="run-retry")
    rule = RetryRule(
        max_pair_retries=1,
        retryable_failure_categories=("tool_outage",),
    )
    schedule, _ = create_run_schedule(
        exp_home,
        "run-retry",
        seed=7,
        split="held_out",
        repeats=1,
        retry_rule=rule,
    )
    p0, p1, p2 = schedule.pairs

    # Scenario:
    # - p0 attempt 1: arm 0 completes, arm 1 fails with tool_outage -> RETRIED as attempt 2!
    #   p0 attempt 2: both arms fail with tool_outage -> cap reached (max_pair_retries=1), NO a3!
    # - p1 attempt 1: arm 0 fails with timeout (non-retryable), arm 1 fails with tool_outage ->
    #   NOT retried because timeout is not retryable.
    # - p2 attempt 1: both arms complete -> NOT retried.
    executed: list[str] = []

    def agent(inv: HarnessInvocation) -> HarnessResult:
        executed.append(inv.trial_id)
        if inv.trial_id == trial_id_for_slot(p0.pair_id, 1, p0.arm_order[1]):
            return HarnessResult(
                status="failed",
                last_reply=None,
                failure_category="tool_outage",
                failure_detail="transient upstream 503",
            )
        if inv.trial_id.startswith(f"{p0.pair_id}__a2__"):
            return HarnessResult(
                status="failed",
                last_reply=None,
                failure_category="tool_outage",
                failure_detail="still 503 on retry",
            )
        if inv.trial_id == trial_id_for_slot(p1.pair_id, 1, p1.arm_order[0]):
            return HarnessResult(
                status="failed",
                last_reply=None,
                failure_category="timeout",
                failure_detail="wall time exhausted",
            )
        if inv.trial_id == trial_id_for_slot(p1.pair_id, 1, p1.arm_order[1]):
            return HarnessResult(
                status="failed",
                last_reply=None,
                failure_category="tool_outage",
                failure_detail="outage on other arm",
            )
        return HarnessResult(status="completed", last_reply="ok")

    run_cfg = ScheduleRunConfig(
        model="local-model",
        model_base_url="http://127.0.0.1:8080/v1",
    )
    res = run_schedule(exp_home, "run-retry", run_cfg, adapter=ScriptedHarnessAdapter(agent))

    assert res.progress.is_complete is True
    expected_p0_a2_arms = arm_order_for_attempt(7, p0, 2)
    expected_sequence = [
        trial_id_for_slot(p0.pair_id, 1, p0.arm_order[0]),
        trial_id_for_slot(p0.pair_id, 1, p0.arm_order[1]),
        trial_id_for_slot(p1.pair_id, 1, p1.arm_order[0]),
        trial_id_for_slot(p1.pair_id, 1, p1.arm_order[1]),
        trial_id_for_slot(p2.pair_id, 1, p2.arm_order[0]),
        trial_id_for_slot(p2.pair_id, 1, p2.arm_order[1]),
        trial_id_for_slot(p0.pair_id, 2, expected_p0_a2_arms[0]),
        trial_id_for_slot(p0.pair_id, 2, expected_p0_a2_arms[1]),
    ]
    assert executed == expected_sequence

    # Attempt 1 records for p0 are retained alongside attempt 2 records (no overwrite in place).
    for tid in expected_sequence:
        assert (store.trials_dir / tid / "record.json").is_file()

    # Check exact retry decisions recorded per pair attempt.
    by_key = {(a.pair_id, a.attempt): a for a in res.progress.attempts}
    assert by_key[(p0.pair_id, 1)].retry_decision == "scheduled"
    assert by_key[(p0.pair_id, 1)].retry_reason == "retryable_failure"
    assert by_key[(p0.pair_id, 2)].retry_decision == "declined"
    assert by_key[(p0.pair_id, 2)].retry_reason == "max_pair_retries_reached"
    assert by_key[(p1.pair_id, 1)].retry_decision == "declined"
    assert by_key[(p1.pair_id, 1)].retry_reason == "non_retryable_failure_category"
    assert by_key[(p2.pair_id, 1)].retry_decision == "not_needed"
    assert res.progress.retried_pairs == 1
    assert res.progress.completed_trials == 3
    assert res.progress.failed_trials == 5


def test_interrupted_pair_retried_only_when_rule_allows_interrupted(tmp_path: Path) -> None:
    """An interrupted pair is run again as attempt 2 iff RetryRule permits 'interrupted'."""
    exp_home, store, _ = _setup_run_with_tasks(tmp_path, run_id="run-int-retry")
    schedule, _ = create_run_schedule(
        exp_home,
        "run-int-retry",
        seed=123,
        split="calibration",
        repeats=1,
        retry_rule=RetryRule(
            max_pair_retries=1,
            retryable_failure_categories=("interrupted",),
        ),
    )
    assert len(schedule.pairs) == 1
    pair = schedule.pairs[0]
    run_cfg = ScheduleRunConfig(
        model="local-model",
        model_base_url="http://127.0.0.1:8080/v1",
    )

    # First pass interrupts on arm 0 of attempt 1.
    def interrupt_once(inv: HarnessInvocation) -> HarnessResult:
        raise KeyboardInterrupt("stop now")

    r1 = run_schedule(
        exp_home,
        "run-int-retry",
        run_cfg,
        adapter=ScriptedHarnessAdapter(interrupt_once),
    )
    assert r1.interrupted is True
    a1_arm0_tid = trial_id_for_slot(pair.pair_id, 1, pair.arm_order[0])
    a1_arm0_sha = sha256_file(store.trials_dir / a1_arm0_tid / "record.json")

    # Resume runs attempt 1's unmade arm 1, then schedules and runs attempt 2 (both arms).
    resumed_ids: list[str] = []

    def complete_all(inv: HarnessInvocation) -> HarnessResult:
        resumed_ids.append(inv.trial_id)
        return HarnessResult(status="completed", last_reply="ok")

    r2 = run_schedule(
        exp_home,
        "run-int-retry",
        run_cfg,
        adapter=ScriptedHarnessAdapter(complete_all),
    )
    assert r2.progress.is_complete is True
    a2_arms = arm_order_for_attempt(123, pair, 2)
    assert resumed_ids == [
        trial_id_for_slot(pair.pair_id, 1, pair.arm_order[1]),
        trial_id_for_slot(pair.pair_id, 2, a2_arms[0]),
        trial_id_for_slot(pair.pair_id, 2, a2_arms[1]),
    ]
    # Attempt 1's interrupted trial record stayed untouched.
    assert sha256_file(store.trials_dir / a1_arm0_tid / "record.json") == a1_arm0_sha
    assert r2.progress.retried_pairs == 1
    assert r2.progress.interrupted_trials == 1
    assert r2.progress.completed_trials == 3


def test_schedule_cli_commands_end_to_end(tmp_path: Path) -> None:
    """CLI schedule, schedule-status, and run-schedule work end-to-end with a fake droid binary."""
    exp_home, _, _ = _setup_run_with_tasks(tmp_path, run_id="run-cli")

    fake_droid = tmp_path / "fake-droid"
    fake_droid.write_text(
        f"#!{sys.executable}\n"
        "import json, sys\n"
        "if '--version' in sys.argv:\n"
        "    print('0.99.0-fake')\n"
        "    sys.exit(0)\n"
        "print(json.dumps({'type': 'system', 'subtype': 'init', 'session_id': 's1', "
        "'model': 'local-model'}))\n"
        "print(json.dumps({'type': 'completion', 'finalText': 'Done.', 'numTurns': 1, "
        "'durationMs': 12, 'is_error': False}))\n",
        encoding="utf-8",
    )
    fake_droid.chmod(fake_droid.stat().st_mode | stat.S_IXUSR)

    runner = CliRunner()
    res_create = runner.invoke(
        app,
        [
            "--home",
            str(exp_home),
            "usefulness",
            "schedule",
            "--run-id",
            "run-cli",
            "--seed",
            "42",
            "--split",
            "calibration",
            "--repeats",
            "2",
            "--max-pair-retries",
            "1",
            "--retry-category",
            "tool_outage",
            "--json",
        ],
    )
    assert res_create.exit_code == 0, res_create.output
    created_payload = json.loads(res_create.output)
    assert len(created_payload["schedule"]["pairs"]) == 2
    assert created_payload["schedule"]["retry_rule"] == {
        "max_pair_retries": 1,
        "retryable_failure_categories": ["tool_outage"],
    }

    res_status = runner.invoke(
        app,
        [
            "--home",
            str(exp_home),
            "usefulness",
            "schedule-status",
            "--run-id",
            "run-cli",
            "--json",
        ],
    )
    assert res_status.exit_code == 0, res_status.output
    status_payload = json.loads(res_status.output)
    assert status_payload["pending_trials"] == 4
    assert status_payload["is_complete"] is False

    res_run = runner.invoke(
        app,
        [
            "--home",
            str(exp_home),
            "usefulness",
            "run-schedule",
            "--run-id",
            "run-cli",
            "--model",
            "local-model",
            "--model-base-url",
            "http://127.0.0.1:8080/v1",
            "--droid-bin",
            str(fake_droid),
            "--max-trials",
            "2",
            "--json",
        ],
    )
    assert res_run.exit_code == 0, res_run.output
    run_payload = json.loads(res_run.output)
    assert len(run_payload["executed_trial_ids"]) == 2
    assert run_payload["stopped_by_max_trials"] is True
    assert run_payload["progress"]["completed_trials"] == 2
    assert run_payload["progress"]["pending_trials"] == 2
