"""Tests for blind trial grading and post-grading memory-use attribution (MEM-25)."""

from __future__ import annotations

import errno
import fcntl
import io
import json
import os
import pty
import select
import shutil
import struct
import subprocess
import sys
import termios
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from agent_memory.cli import app
from agent_memory.config import Config
from agent_memory.db import MemoryServer, initialize_home
from agent_memory.judge import label_build, label_frontend
from agent_memory.judge.label_controller import Action, Screen
from agent_memory.judge.label_frontend import PlainFrontend, encode_frame
from agent_memory.judge.label_questions import Option
from agent_memory.judge.store import JudgeError
from agent_memory.mcp_server.tools import MemoryTools
from agent_memory.smoke import free_port
from agent_memory.usefulness import (
    GRADE_FAILURE_CATEGORIES,
    BlindGradingController,
    HarnessInvocation,
    HarnessResult,
    MemoryAttributionController,
    ScriptedHarnessAdapter,
    ToolCallRecord,
    TrialConfig,
    UsefulnessError,
    UsefulnessRunStore,
    build_task_manifest,
    create_run_schedule,
    discover_run_trials,
    finalize_grades,
    grade_history,
    inspect_grading_status,
    latest_attributions,
    latest_grades,
    make_memory_set,
    order_trials_blind,
    record_attribution,
    record_grade,
    run_trial,
    run_trial_checks,
    sha256_file,
    verify_finalized_grades,
)


def _git(repo: Path, *args: str) -> str:
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "Test",
        "GIT_AUTHOR_EMAIL": "test@example.com",
        "GIT_COMMITTER_NAME": "Test",
        "GIT_COMMITTER_EMAIL": "test@example.com",
        "GIT_TERMINAL_PROMPT": "0",
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
    return repo, _git(repo, "rev-parse", "HEAD")


def _write_task(
    tasks_root: Path,
    task_id: str,
    *,
    repo: Path,
    commit: str,
    work_done_at: str,
    task_family: str,
    strata: list[str] | None = None,
    check_script: str | None = None,
) -> Path:
    task_dir = tasks_root / task_id
    task_dir.mkdir(parents=True)
    # Check script intentionally prints cwd (which contains trial_id/arm) to test scrubbing!
    script = check_script or (
        "import os, sys; "
        "print('cwd_marker=' + os.getcwd()); "
        "import app; "
        "sys.exit(0 if app.answer() == 2 else 1)"
    )
    meta = {
        "schema_version": 1,
        "task_id": task_id,
        "task_family": task_family,
        "strata": strata if strata is not None else ["general"],
        "project": "demo-proj",
        "repository": str(repo),
        "commit": commit,
        "work_done_at": work_done_at,
        "checks": {
            "commands": [
                {"name": "unit", "command": [sys.executable, "-c", script]},
            ],
        },
    }
    (task_dir / "task.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (task_dir / "request.md").write_text(
        f"Update answer() to return 2 for {task_id}.\n", encoding="utf-8"
    )
    (task_dir / "required_behavior.md").write_text(
        "app.answer() returns 2 and unit check passes.\n", encoding="utf-8"
    )
    return task_dir


def _setup_graded_run(
    tmp_path: Path,
    *,
    run_id: str = "run-grade",
    num_cal_tasks: int = 2,
    include_recurrence: bool = False,
) -> dict[str, Any]:
    """Create a run with calibration tasks, a memory set, finished trials, and checks.json."""
    repo, commit = _make_repo(tmp_path)
    exp_home = tmp_path / "exp"
    tasks_dir = exp_home / "tasks"

    # Source home with a secret sentinel memory so we can verify:
    # 1) blind grading NEVER leaks it
    # 2) post-grading attribution CAN read it read-only from the trial home
    source_home = tmp_path / "source-ledger"
    cfg_src = Config(home=source_home, port=free_port())
    initialize_home(cfg_src)
    srv = MemoryServer(cfg_src, export_interval_hours=0)
    srv.start()
    try:
        tools = MemoryTools(cfg_src, "factory-droid", project="demo-proj")
        try:
            created = tools.remember(
                content="SECRET_LEDGER_MEMORY_SENTINEL: always return 99 instead of 2.",
                memory_type="decision",
                scope="demo-proj",
                trust_level="user",
            )
            mem_id = str(created["memory_id"])
        finally:
            tools.close()
    finally:
        srv.stop()
    ms = make_memory_set(
        source_home,
        exp_home,
        cutoff="2026-10-01T00:00:00Z",
        last_day="2026-10-31T23:59:59Z",
    )

    task_ids: list[str] = []
    for idx in range(1, num_cal_tasks + 1):
        tid = f"t-cal-{idx}"
        strata = ["recurrence"] if (include_recurrence and idx == 1) else ["general"]
        _write_task(
            tasks_dir,
            tid,
            repo=repo,
            commit=commit,
            work_done_at=f"2026-11-0{idx}T12:00:00Z",
            task_family=f"fam-cal-{idx}",
            strata=strata,
        )
        task_ids.append(tid)
    # Add 1 held-out task so the chronological split is valid.
    _write_task(
        tasks_dir,
        "t-held-1",
        repo=repo,
        commit=commit,
        work_done_at="2026-11-20T12:00:00Z",
        task_family="fam-held-1",
        strata=["exact_value"],
    )

    store = UsefulnessRunStore(exp_home, run_id)
    manifest = build_task_manifest(tasks_dir, split_after="2026-11-15T00:00:00Z")
    store.write_task_manifest(manifest)
    schedule, _ = create_run_schedule(exp_home, run_id, seed=42, split="calibration", repeats=1)

    # Run both arms for every calibration task with predictable pass/fail outcomes:
    # - t-cal-1: memory arm fails (writes wrong return 99 + confident reply + recalls memory),
    #            no_memory arm passes (writes return 2).
    # - t-cal-2+: both arms pass (return 2).
    trial_records = []
    for pair in schedule.pairs:
        for arm in pair.arm_order:
            tid = f"{pair.pair_id}__a1__{arm}"

            def _make_agent(t_id: str, arm_name: str, pair_task: str):
                def _run(inv: HarnessInvocation) -> HarnessResult:
                    if pair_task == "t-cal-1" and arm_name == "memory":
                        (inv.clone_dir / "app.py").write_text(
                            "def answer():\n    return 99\n", encoding="utf-8"
                        )
                        return HarnessResult(
                            status="completed",
                            last_reply="All done, answer() is updated and verified.",
                            tool_calls=(
                                ToolCallRecord(
                                    sequence=1,
                                    name="mcp__agent-memory__memory_recall",
                                    is_error=False,
                                ),
                            ),
                        )
                    (inv.clone_dir / "app.py").write_text(
                        "def answer():\n    return 2\n", encoding="utf-8"
                    )
                    return HarnessResult(
                        status="completed",
                        last_reply="Updated answer() to return 2.",
                        tool_calls=(
                            (
                                ToolCallRecord(
                                    sequence=1,
                                    name="mcp__agent-memory__memory_recall",
                                    is_error=False,
                                ),
                            )
                            if arm_name == "memory"
                            else ()
                        ),
                    )

                return _run

            tcfg = TrialConfig(
                arm=arm,
                model="local-model",
                model_base_url="http://127.0.0.1:8080/v1",
                source_home=ms.memory_set_dir if arm == "memory" else None,
                trial_id=tid,
            )
            rec = run_trial(
                exp_home,
                run_id,
                tasks_dir / pair.task_id,
                tcfg,
                adapter=ScriptedHarnessAdapter(_make_agent(tid, arm, pair.task_id)),
            )
            # For t-cal-1 memory arm, attach synthetic recall_accounting referencing mem_id
            # inside its record.json BEFORE running checks so checks.json hashes the final record.
            if pair.task_id == "t-cal-1" and arm == "memory":
                raw = json.loads(rec.record_path.read_text(encoding="utf-8"))
                raw["recall_accounting"] = {
                    "total_recalls": 1,
                    "returned_count": 1,
                    "miss_count": 0,
                    "degraded_count": 0,
                    "missing_record_count": 0,
                    "recalls": [
                        {
                            "outcome": "returned",
                            "source": "transcript_and_ledger",
                            "operation": "recall",
                            "call_id": "call-1",
                            "retrieval_id": "ret-1",
                            "returned_count": 1,
                            "returned_memory_ids": [mem_id],
                            "strategy": "hybrid",
                            "latency_ms": 4.2,
                        }
                    ],
                    "citations": {
                        "citations_made": 1,
                        "returned_memories_cited_count": 1,
                        "cited_memory_ids": [mem_id],
                        "returned_memory_ids": [mem_id],
                        "returned_memories_cited_ids": [mem_id],
                        "unbound_placeholder_count": 0,
                        "single_citation_reply_count": 1,
                    },
                }
                rec.record_path.write_text(
                    json.dumps(raw, indent=2, sort_keys=True) + "\n", encoding="utf-8"
                )
            run_trial_checks(exp_home, run_id, tid)
            trial_records.append(rec)

    return {
        "exp_home": exp_home,
        "run_id": run_id,
        "store": store,
        "tasks_dir": tasks_dir,
        "schedule": schedule,
        "mem_id": mem_id,
        "trial_records": trial_records,
    }


def _complete_trial_pass(ctrl: BlindGradingController, *, arm_leaked: bool = False) -> None:
    while ctrl.step != "summary":
        step = ctrl.step
        if step == "met_checks":
            ctrl.apply(Action("answer", "y"))
        elif step == "repeated_earlier_error":
            ctrl.apply(Action("answer", "n"))
        elif step == "arm_leaked":
            ctrl.apply(Action("answer", "y" if arm_leaked else "n"))
        elif step == "note":
            ctrl.apply(Action("answer", "continue"))
        else:
            raise AssertionError(f"unexpected step {step}")
    ctrl.apply(Action("answer", "save"))


def _complete_trial_fail(
    ctrl: BlindGradingController,
    *,
    category: str = "ordinary_wrong_answer",
    confident_wrong: bool = True,
    repeated_earlier_error: bool = False,
    arm_leaked: bool = False,
    note: str | None = None,
) -> None:
    while ctrl.step != "summary":
        step = ctrl.step
        if step == "met_checks":
            ctrl.apply(Action("answer", "n"))
        elif step == "failure_category":
            ctrl.apply(Action("answer", category))
        elif step == "confident_wrong_reply":
            ctrl.apply(Action("answer", "y" if confident_wrong else "n"))
        elif step == "repeated_earlier_error":
            ctrl.apply(Action("answer", "y" if repeated_earlier_error else "n"))
        elif step == "arm_leaked":
            ctrl.apply(Action("answer", "y" if arm_leaked else "n"))
        elif step == "note":
            if note is not None:
                ctrl.apply(Action("note", note))
            ctrl.apply(Action("answer", "continue"))
        else:
            raise AssertionError(f"unexpected step {step}")
    ctrl.apply(Action("answer", "save"))


def test_blind_grading_shows_required_material_and_hides_arm_tools_and_ledger(
    tmp_path: Path,
) -> None:
    """Blind screens show request, checks, repo changes, last reply; never arm/tools/ledger."""
    fx = _setup_graded_run(tmp_path, num_cal_tasks=2)
    ctrl = BlindGradingController(fx["exp_home"], fx["run_id"], seed=7)

    frames: list[dict[str, Any]] = []
    trial_ids_in_run = [r.trial_id for r in fx["trial_records"]]
    while not ctrl.done:
        scr = ctrl.screen()
        assert encode_frame(
            {"version": 2, "request_id": 1, "type": "screen", "screen": scr.payload()}
        )
        frames.append(scr.payload())
        if ctrl.step == "met_checks":
            # Fail one trial so we exercise failure_category & confident_wrong_reply screens too
            ctrl.apply(Action("answer", "n" if scr.item == 1 else "y"))
        elif ctrl.step == "failure_category":
            ctrl.apply(Action("answer", "ordinary_wrong_answer"))
        elif ctrl.step == "confident_wrong_reply":
            ctrl.apply(Action("answer", "y"))
        elif ctrl.step == "arm_leaked":
            ctrl.apply(Action("answer", "n"))
        elif ctrl.step == "note":
            ctrl.apply(Action("answer", "continue"))
        elif ctrl.step == "summary":
            ctrl.apply(Action("answer", "save"))

    boundary = json.dumps(frames, ensure_ascii=False)

    # Required content IS present across the blind screens:
    assert "=== REQUEST ===" in boundary
    assert "Update answer() to return 2" in boundary
    assert "=== REQUIRED BEHAVIOR ===" in boundary
    assert "=== CHECKS ===" in boundary
    assert "=== REPOSITORY CHANGES ===" in boundary
    assert "app.py" in boundary
    assert "=== LAST REPLY ===" in boundary
    assert "All done, answer() is updated and verified." in boundary
    # Even though the check script printed os.getcwd(), the trial_dir path was scrubbed!
    assert "cwd_marker=<trial-dir>/repo" in boundary

    # Subjects use "Trial", "Notes", "Summary":
    assert {f["subject"] for f in frames} == {"Trial", "Notes", "Summary"}

    # Forbidden material NEVER appears anywhere in any frame:
    for forbidden in (
        *trial_ids_in_run,
        "no_memory",
        "__memory",
        "mcp__agent-memory",
        "memory_recall",
        "SECRET_LEDGER_MEMORY_SENTINEL",
        fx["mem_id"],
        "recall_accounting",
        "trial-home",
        "ret-1",
    ):
        assert forbidden not in boundary, f"blinding leak detected: {forbidden!r}"


def test_blind_order_does_not_reveal_arm_and_separates_pair_trials(tmp_path: Path) -> None:
    """Order is deterministic per seed, never puts a pair back-to-back, and does not reveal arm."""
    fx = _setup_graded_run(tmp_path, num_cal_tasks=3)
    targets = discover_run_trials(fx["exp_home"], fx["run_id"])
    assert len(targets) == 6

    first_arms: set[str] = set()
    arm_patterns: set[tuple[str, ...]] = set()
    for seed in range(1, 51):
        ordered = order_trials_blind(targets, seed=seed)
        ordered_rev = order_trials_blind(list(reversed(targets)), seed=seed)
        # Deterministic regardless of input order
        assert [t.trial_id for t in ordered] == [t.trial_id for t in ordered_rev]
        assert len(ordered) == len(targets)
        # No two trials of the same pair appear consecutively
        for i in range(len(ordered) - 1):
            assert ordered[i].pair_key != ordered[i + 1].pair_key, (
                f"seed={seed} placed pair {ordered[i].pair_key} consecutively"
            )
            assert ordered[i].task_id != ordered[i + 1].task_id
        first_arms.add(ordered[0].arm)
        arm_patterns.add(tuple(t.arm for t in ordered))

    # Order does not lock the first arm or collapse to strict alternation
    assert first_arms == {"memory", "no_memory"}
    assert len(arm_patterns) > 4
    # At least some seeds have two adjacent trials of the same arm (proving non-alternating order)
    assert any(any(pat[i] == pat[i + 1] for i in range(len(pat) - 1)) for pat in arm_patterns)


def test_blind_order_separates_two_pairs_four_trials(tmp_path: Path) -> None:
    """With 2 pairs (4 trials), greedy/shuffle always prevents adjacent same-pair trials."""
    fx = _setup_graded_run(tmp_path, num_cal_tasks=2)
    targets = discover_run_trials(fx["exp_home"], fx["run_id"])
    assert len(targets) == 4
    for seed in range(1, 40):
        ordered = order_trials_blind(targets, seed=seed)
        for i in range(len(ordered) - 1):
            assert ordered[i].pair_key != ordered[i + 1].pair_key


def test_grading_corrections_preserve_history_and_navigation(tmp_path: Path) -> None:
    """Grades support failure kinds, arm-leaked flag, back cleanup, notes, and revision history."""
    fx = _setup_graded_run(tmp_path, num_cal_tasks=2, include_recurrence=True)
    store: UsefulnessRunStore = fx["store"]

    ctrl = BlindGradingController(fx["exp_home"], fx["run_id"], seed=3)
    # Trial 1: start as fail, go back and switch to pass (branch cleanup), mark arm_leaked, add note
    first_tid = ctrl.target.trial_id
    ctrl.apply(Action("answer", "n"))
    assert ctrl.step == "failure_category"
    ctrl.apply(Action("answer", "timeout"))
    assert ctrl.step == "confident_wrong_reply"
    ctrl.apply(Action("back"))
    assert ctrl.step == "failure_category"
    assert ctrl.screen().selected == "timeout"
    ctrl.apply(Action("back"))
    assert ctrl.step == "met_checks"
    ctrl.apply(Action("answer", "y"))
    # Switching to Yes clears failure_category and confident_wrong_reply
    assert "failure_category" not in ctrl.answers
    assert "confident_wrong_reply" not in ctrl.answers
    if ctrl.step == "repeated_earlier_error":
        ctrl.apply(Action("answer", "n"))
    assert ctrl.step == "arm_leaked"
    ctrl.apply(Action("answer", "y"))
    assert ctrl.step == "note"
    ctrl.apply(Action("note", "q [bold] literal note"))
    ctrl.apply(Action("answer", "continue"))
    assert ctrl.step == "summary"
    ctrl.apply(Action("answer", "save"))

    # Finish remaining trials (one with timeout failure + confident_wrong=False, rest pass)
    second_tid = ctrl.target.trial_id
    _complete_trial_fail(
        ctrl,
        category="timeout",
        confident_wrong=False,
        repeated_earlier_error=True,
        arm_leaked=False,
        note="hit limit",
    )
    while not ctrl.done:
        _complete_trial_pass(ctrl)

    assert ctrl.summary().graded == 4
    assert ctrl.summary().remaining == 0

    rev1_rows = grade_history(store, first_tid)
    assert len(rev1_rows) == 1
    assert rev1_rows[0]["revision"] == 1
    assert rev1_rows[0]["met_checks"] is True
    assert rev1_rows[0]["arm_leaked"] is True
    assert rev1_rows[0]["note"] == "q [bold] literal note"

    # Now open review mode BEFORE finalization and correct first_tid from pass -> fail!
    review_ctrl = BlindGradingController(
        fx["exp_home"], fx["run_id"], seed=3, review=True, trial_ids=[first_tid]
    )
    assert review_ctrl.step == "review"
    assert review_ctrl.screen().subject == "Review"
    assert review_ctrl.screen().can_redo is True
    review_ctrl.apply(Action("answer", "redo"))
    _complete_trial_fail(
        review_ctrl,
        category="ordinary_wrong_answer",
        confident_wrong=True,
        repeated_earlier_error=False,
        arm_leaked=False,
        note="corrected after re-reading diff",
    )
    assert review_ctrl.done
    assert review_ctrl.summary().changed == 1

    # Full history is kept in grades.jsonl; latest_grades reflects revision 2
    hist = grade_history(store, first_tid)
    assert len(hist) == 2
    assert hist[0]["revision"] == 1
    assert hist[0]["met_checks"] is True
    assert hist[1]["revision"] == 2
    assert hist[1]["supersedes_grade_id"] == hist[0]["grade_id"]
    assert hist[1]["met_checks"] is False
    assert hist[1]["failure_category"] == "ordinary_wrong_answer"
    assert hist[1]["confident_wrong_reply"] is True
    assert hist[1]["arm_leaked"] is False
    assert latest_grades(store)[first_tid]["grade_id"] == hist[1]["grade_id"]
    assert latest_grades(store)[second_tid]["failure_category"] == "timeout"

    # Check status is arm-blind and reflects 5 total revisions across 4 trials
    status = inspect_grading_status(fx["exp_home"], fx["run_id"])
    assert status.total_trials == 4
    assert status.graded_trials == 4
    assert status.ungraded_trials == 0
    assert status.total_grade_revisions == 5
    assert status.grades_final is False


def test_finalization_gate_locks_grades_and_unlocks_attribution_without_altering_grades(
    tmp_path: Path,
) -> None:
    """Attribution refuses before finalize, opens after, reads DB read-only, alters no grade."""
    fx = _setup_graded_run(tmp_path, num_cal_tasks=2)
    store: UsefulnessRunStore = fx["store"]
    targets = discover_run_trials(fx["exp_home"], fx["run_id"])
    mem_fail_target = next(t for t in targets if t.task_id == "t-cal-1" and t.arm == "memory")

    # 1. Cannot finalize while trials are ungraded; cannot open attribution before finalize.
    with pytest.raises(UsefulnessError, match="still ungraded"):
        finalize_grades(fx["exp_home"], fx["run_id"])
    with pytest.raises(UsefulnessError, match="not final"):
        MemoryAttributionController(fx["exp_home"], fx["run_id"])
    with pytest.raises(UsefulnessError, match="not final"):
        record_attribution(
            fx["exp_home"],
            fx["run_id"],
            mem_fail_target.trial_id,
            harmful_memory_use=True,
        )

    # 2. Grade all trials (leaving t-cal-1 memory arm as failed + confident_wrong_reply).
    for t in targets:
        if t.trial_id == mem_fail_target.trial_id:
            record_grade(
                fx["exp_home"],
                fx["run_id"],
                t.trial_id,
                met_checks=False,
                failure_category="ordinary_wrong_answer",
                confident_wrong_reply=True,
                arm_leaked=False,
            )
        else:
            record_grade(
                fx["exp_home"],
                fx["run_id"],
                t.trial_id,
                met_checks=True,
            )

    # Still cannot open attribution before explicit finalization!
    with pytest.raises(UsefulnessError, match="not final"):
        MemoryAttributionController(fx["exp_home"], fx["run_id"])

    # 3. Finalize grades (write-once).
    final_info = finalize_grades(fx["exp_home"], fx["run_id"])
    assert final_info["trial_count"] == 4
    grades_bytes_before = store.grades_path.read_bytes()
    grades_final_bytes_before = store.grades_final_path.read_bytes()

    # Second finalization is refused; further blind grading or corrections are refused!
    with pytest.raises(UsefulnessError, match="already final"):
        finalize_grades(fx["exp_home"], fx["run_id"])
    with pytest.raises(UsefulnessError, match="already final"):
        BlindGradingController(fx["exp_home"], fx["run_id"])
    with pytest.raises(UsefulnessError, match="already final"):
        record_grade(
            fx["exp_home"],
            fx["run_id"],
            mem_fail_target.trial_id,
            met_checks=True,
        )

    # 4. Now MemoryAttributionController opens ONLY the failed/confident-wrong memory trial!
    trial_home_db = Path(mem_fail_target.record["paths"]["trial_home"]) / "memory.duckdb"
    trial_db_sha_before = sha256_file(trial_home_db)

    attr_ctrl = MemoryAttributionController(fx["exp_home"], fx["run_id"], seed=1)
    assert len(attr_ctrl.todo) == 1
    assert attr_ctrl.target.trial_id == mem_fail_target.trial_id

    scr = attr_ctrl.screen()
    assert scr.subject == "Trial"
    # Unblinded attribution screen shows recalls, citations, AND the recalled memory from trial DB!
    assert "SECRET_LEDGER_MEMORY_SENTINEL" in scr.evidence
    assert fx["mem_id"] in scr.evidence
    assert "mcp__agent-memory__memory_recall" in scr.evidence

    # Step through attribution questions
    assert attr_ctrl.step == "harmful_memory_use"
    attr_ctrl.apply(Action("answer", "y"))
    assert attr_ctrl.step == "attempted_invalid_memory_use"
    attr_ctrl.apply(Action("answer", "n"))
    assert attr_ctrl.step == "confident_wrong_linked_to_memory"
    attr_ctrl.apply(Action("answer", "y"))
    assert attr_ctrl.step == "note"
    attr_ctrl.apply(Action("note", f"linked to {fx['mem_id']}"))
    attr_ctrl.apply(Action("answer", "continue"))
    assert attr_ctrl.step == "summary"
    attr_ctrl.apply(Action("answer", "save"))
    assert attr_ctrl.done
    assert attr_ctrl.summary().attributed == 1

    # Verify attribution saved AND grades.jsonl + grades-final.json + trial DB are UNCHANGED!
    attrs = latest_attributions(store)
    assert mem_fail_target.trial_id in attrs
    row = attrs[mem_fail_target.trial_id]
    assert row["harmful_memory_use"] is True
    assert row["attempted_invalid_memory_use"] is False
    assert row["confident_wrong_linked_to_memory"] is True
    assert row["grades_final_sha256"] == final_info["grades_final_sha256"]
    assert store.grades_path.read_bytes() == grades_bytes_before
    assert store.grades_final_path.read_bytes() == grades_final_bytes_before
    assert sha256_file(trial_home_db) == trial_db_sha_before

    # 5. Tampering with grades.jsonl after finalization is detected and refused.
    store.grades_path.write_bytes(grades_bytes_before + b" ")
    with pytest.raises(UsefulnessError, match="does not match"):
        verify_finalized_grades(store)
    with pytest.raises(UsefulnessError, match="does not match"):
        MemoryAttributionController(fx["exp_home"], fx["run_id"])


def test_plain_frontend_and_cli_end_to_end(tmp_path: Path) -> None:
    """PlainFrontend and `mem usefulness` CLI drive blind grading, finalize, and attribution."""
    fx = _setup_graded_run(tmp_path, num_cal_tasks=2)
    runner = CliRunner()

    # Drive 1 trial through PlainFrontend directly to verify help/evidence/note/save
    ctrl = BlindGradingController(fx["exp_home"], fx["run_id"], seed=1, limit=1)
    first_tid = ctrl.target.trial_id
    out = io.StringIO()
    # 1=Yes (met_checks), 1=No (arm_leaked), 1=edit note -> text -> 2=continue,
    # p=full bundle, ?=guide, 1=save
    plain_input = "?\np\n1\n1\n1\nplain note text\n2\n1\n"
    pf = PlainFrontend(io.StringIO(plain_input), out)
    while not ctrl.done:
        ctrl.apply(pf.exchange(ctrl.screen()))
    rendered = out.getvalue()
    assert "Label | Item 1 of 1 | Trial" in rendered
    assert "Label | Item 1 of 1 | Notes" in rendered
    assert "Label | Item 1 of 1 | Summary" in rendered
    assert "Blind trial grading guide" in rendered
    assert "=== REQUEST ===" in rendered
    assert latest_grades(fx["store"])[first_tid]["note"] == "plain note text"

    # Grade the remaining 3 trials through the CLI `--plain`:
    # For each remaining trial, answer fail on the first remaining one (with confident wrong reply)
    # and pass on the others.
    # Let's check which trial is the memory-arm failure and grade all remaining via CLI `--plain`!
    res_status = runner.invoke(
        app,
        [
            "--home",
            str(fx["exp_home"]),
            "usefulness",
            "grades-status",
            "--run-id",
            fx["run_id"],
            "--json",
        ],
    )
    assert res_status.exit_code == 0, res_status.output
    st = json.loads(res_status.output)
    assert st["graded_trials"] == 1
    assert st["ungraded_trials"] == 3

    # Grade all 3 remaining trials via CLI `--plain`:
    # For each: 2 (No met_checks), 1 (ordinary_wrong_answer), 1 (Yes confident_wrong_reply),
    # 1 (No arm_leaked), 2 (Continue note), 1 (Save).
    cli_answers = "2\n1\n1\n1\n2\n1\n" * 3
    res_grade = runner.invoke(
        app,
        ["--home", str(fx["exp_home"]), "usefulness", "grade", "--run-id", fx["run_id"], "--plain"],
        input=cli_answers,
    )
    assert res_grade.exit_code == 0, res_grade.output
    assert "Graded 3" in res_grade.output

    # Finalize via CLI
    res_fin = runner.invoke(
        app,
        [
            "--home",
            str(fx["exp_home"]),
            "usefulness",
            "finalize-grades",
            "--run-id",
            fx["run_id"],
            "--json",
        ],
    )
    assert res_fin.exit_code == 0, res_fin.output
    fin_data = json.loads(res_fin.output)
    assert fin_data["trial_count"] == 4

    # Attribute via CLI `--plain` (at least 1 failed memory-arm trial exists)
    attr_answers = "1\n2\n1\n2\n1\n" * 4
    res_attr = runner.invoke(
        app,
        [
            "--home",
            str(fx["exp_home"]),
            "usefulness",
            "attribute",
            "--run-id",
            fx["run_id"],
            "--plain",
        ],
        input=attr_answers,
    )
    assert res_attr.exit_code == 0, res_attr.output
    assert "Attributed" in res_attr.output


def test_screen_accepts_trial_subject_and_rejects_unknown_subject() -> None:
    """Python protocol Screen accepts 'Trial' and still rejects unknown subjects."""
    scr = Screen(
        item=1,
        total=1,
        subject="Trial",
        key="1:met_checks",
        metadata="Task: t-1",
        evidence="REQUEST\nDo X.",
        projection="REQUEST\nDo X.",
        question="Did this trial meet checks?",
        options=[Option("y", "Yes"), Option("n", "No")],
    )
    assert scr.payload()["subject"] == "Trial"
    with pytest.raises(JudgeError, match="Invalid labeling subject"):
        Screen(
            item=1,
            total=1,
            subject="Trial 2",
            key="1:met_checks",
            metadata="Task: t-1",
            evidence="REQUEST\nDo X.",
            projection="REQUEST\nDo X.",
            question="Did this trial meet checks?",
            options=[Option("y", "Yes")],
        )


def test_all_grade_failure_categories_validated(tmp_path: Path) -> None:
    """Every registered failure category works; unknown categories and bad combos are refused."""
    fx = _setup_graded_run(tmp_path, num_cal_tasks=2)
    target = discover_run_trials(fx["exp_home"], fx["run_id"])[0]
    for cat in GRADE_FAILURE_CATEGORIES:
        row = record_grade(
            fx["exp_home"],
            fx["run_id"],
            target.trial_id,
            met_checks=False,
            failure_category=cat,
        )
        assert row["failure_category"] == cat

    with pytest.raises(UsefulnessError, match="invalid failure_category"):
        record_grade(
            fx["exp_home"],
            fx["run_id"],
            target.trial_id,
            met_checks=False,
            failure_category="made_up_category",
        )
    with pytest.raises(UsefulnessError, match="failure_category must be None"):
        record_grade(
            fx["exp_home"],
            fx["run_id"],
            target.trial_id,
            met_checks=True,
            failure_category="timeout",
        )
    with pytest.raises(UsefulnessError, match="confident_wrong_reply must be False"):
        record_grade(
            fx["exp_home"],
            fx["run_id"],
            target.trial_id,
            met_checks=True,
            confident_wrong_reply=True,
        )


def test_real_go_helper_roundtrip_with_trial_subject(tmp_path: Path) -> None:
    """Real Python GoFrontend + compiled Go label-ui binary render and accept a 'Trial' screen."""
    if shutil.which("go") is None:
        pytest.skip("real Go label-ui round trip needs Go")
    binary = label_build.frontend_binary(root=tmp_path / "label-ui-cache")
    fx = _setup_graded_run(tmp_path, num_cal_tasks=2)
    ctrl = BlindGradingController(fx["exp_home"], fx["run_id"], seed=1)
    screen = ctrl.screen()
    assert screen.subject == "Trial"

    path = tmp_path / "trial_screen.json"
    path.write_text(json.dumps(screen.payload()), encoding="utf-8")
    body = """
import fcntl, json, sys, termios
from pathlib import Path
from agent_memory.judge.label_controller import Action, DisplayDraft, Screen
from agent_memory.judge.label_frontend import GoFrontend
from agent_memory.judge.label_questions import Option
fcntl.ioctl(sys.stdin.fileno(), termios.TIOCSCTTY, 0)
data = json.loads(Path(sys.argv[2]).read_text())
data["drafts"] = [DisplayDraft(**d) for d in data["drafts"]]
data["options"] = [Option(**o) for o in data["options"]]
screen = Screen(**data)
frontend = GoFrontend(Path(sys.argv[1]))
try:
    assert frontend.exchange(screen) == Action("answer", screen.selected)
    frontend.finish("Finished Trial round trip.")
finally:
    frontend.close()
print("TRIAL_GO_ROUNDTRIP_OK", flush=True)
"""
    master, slave = pty.openpty()
    fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", 40, 100, 0, 0))
    process = subprocess.Popen(
        [sys.executable, "-c", body, str(binary), str(path)],
        stdin=slave,
        stdout=slave,
        stderr=slave,
        start_new_session=True,
        env={**os.environ, "TERM": "xterm-256color", "NO_COLOR": "1"},
    )
    os.close(slave)
    output, submitted = bytearray(), False
    deadline = time.monotonic() + 15
    try:
        while time.monotonic() < deadline:
            ready, _, _ = select.select([master], [], [], 0.2)
            if ready:
                try:
                    chunk = os.read(master, 4096)
                except OSError as exc:
                    if exc.errno == errno.EIO:
                        break
                    raise
                if not chunk:
                    break
                output.extend(chunk)
                if not submitted and b"Did this trial meet" in output:
                    os.write(master, b"\r")
                    submitted = True
            if process.poll() is not None and not ready:
                break
        process.wait(timeout=5)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        os.close(master)

    assert process.returncode == 0, output.decode("utf-8", errors="replace")
    assert b"Trial" in output
    assert b"TRIAL_GO_ROUNDTRIP_OK" in output


def test_save_failure_leaves_earlier_grades_intact_without_leaking_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If saving a grade fails on disk, earlier grades stay intact and raw OS error is hidden."""
    fx = _setup_graded_run(tmp_path, num_cal_tasks=2)
    ctrl = BlindGradingController(fx["exp_home"], fx["run_id"], seed=1)
    _complete_trial_pass(ctrl)
    saved_bytes = fx["store"].grades_path.read_bytes()

    while ctrl.step != "summary":
        if ctrl.step == "met_checks":
            ctrl.apply(Action("answer", "y"))
        elif ctrl.step == "arm_leaked":
            ctrl.apply(Action("answer", "n"))
        elif ctrl.step == "note":
            ctrl.apply(Action("answer", "continue"))

    def _boom(_row: Mapping[str, Any]) -> None:
        raise OSError("PRIVATE_DISK_DETAIL")

    monkeypatch.setattr(ctrl.store, "append_grade", _boom)
    ctrl.apply(Action("answer", "save"))
    assert ctrl.step == "summary"
    assert "Save failed" in ctrl.screen().notice
    assert "PRIVATE_DISK_DETAIL" not in ctrl.screen().notice
    assert fx["store"].grades_path.read_bytes() == saved_bytes


def test_drive_used_directly_by_grade_and_attribution_sessions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """grade_session and attribution_session route through judge.label_frontend.drive."""
    fx = _setup_graded_run(tmp_path, num_cal_tasks=2)
    seen_subjects: list[str] = []

    class FakeFrontend:
        def __init__(self, _binary: Path) -> None:
            pass

        def exchange(self, screen: Screen) -> Action:
            seen_subjects.append(screen.subject)
            return Action("quit")

        def finish(self, message: str) -> None:
            pass

        def close(self) -> None:
            pass

    monkeypatch.setattr(label_frontend, "GoFrontend", FakeFrontend)
    monkeypatch.setattr(label_frontend, "automatic_plain", lambda: False)
    monkeypatch.setattr(label_frontend, "frontend_binary", lambda **kwargs: tmp_path / "helper")

    from agent_memory.usefulness import grade_session

    summary = grade_session(fx["exp_home"], fx["run_id"], plain=False)
    assert summary.stopped is True
    assert seen_subjects == ["Trial"]
