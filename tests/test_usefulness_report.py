"""Tests for the local aggregate report of a usefulness run (MEM-27)."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from agent_memory.cli import app as cli_app
from agent_memory.config import Config
from agent_memory.db import initialize_home, open_direct
from agent_memory.mcp_server import TOOL_NAMES
from agent_memory.usefulness import (
    DEFAULT_BASE_TOOLS,
    BinaryPair,
    ReportResult,
    RetryRule,
    UsefulnessError,
    UsefulnessRunStore,
    append_deviation,
    build_report,
    build_task_manifest,
    create_run_schedule,
    default_trial_config_overrides,
    finalize_grades,
    freeze_registration,
    make_memory_set,
    paired_success_summary,
    record_attribution,
    record_grade,
    sha256_file,
    write_once_json,
)

CUSTOM_INSTRUCTIONS = "Registered report test instructions (v1)."
CUSTOM_DESCRIPTIONS: dict[str, str] = {
    name: f"Registered report test description for {name} (v1)." for name in TOOL_NAMES
}

CANARY_TASK_REQUEST = "CANARY_SECRET_TASK_REQUEST_PROMPT_9911"
CANARY_REQUIRED_BEHAVIOR = "CANARY_SECRET_REQUIRED_BEHAVIOR_9922"
CANARY_HIDDEN_TEST = "CANARY_SECRET_HIDDEN_TEST_ASSERTION_9933"
CANARY_CHECK_OUTPUT = "CANARY_SECRET_CHECK_STDOUT_9944"
CANARY_LAST_REPLY = "CANARY_SECRET_ASSISTANT_LAST_REPLY_9955"
CANARY_DIFF_TEXT = "CANARY_SECRET_REPO_DIFF_PATCH_9966"
CANARY_TOOL_ARG = "CANARY_SECRET_TOOL_ARGUMENT_9977"
CANARY_FAILURE_DETAIL = "CANARY_SECRET_FAILURE_STACKTRACE_9988"
CANARY_MEMORY_TEXT = "CANARY_SECRET_LEDGER_MEMORY_BODY_9999"
CANARY_GRADE_NOTE = "CANARY_SECRET_GRADER_FREE_NOTE_8811"
CANARY_ATTR_NOTE = "CANARY_SECRET_ATTRIBUTION_NOTE_8822"


def _git(repo: Path, *args: str) -> str:
    env = os.environ.copy()
    env.update(
        {
            "GIT_AUTHOR_NAME": "Test Author",
            "GIT_AUTHOR_EMAIL": "test@example.com",
            "GIT_COMMITTER_NAME": "Test Author",
            "GIT_COMMITTER_EMAIL": "test@example.com",
            "GIT_TERMINAL_PROMPT": "0",
        }
    )
    proc = subprocess.run(
        ["git", *args],
        cwd=str(repo),
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    return proc.stdout.strip()


def _make_git_repo(tmp_path: Path) -> tuple[Path, str]:
    repo = tmp_path / "source-repo"
    repo.mkdir(parents=True)
    _git(repo, "init", "-b", "main")
    (repo / "app.py").write_text("def value():\n    return 1\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "initial")
    return repo, _git(repo, "rev-parse", "HEAD")


def _make_source_ledger_with_validity_crossings(tmp_path: Path) -> Path:
    """Create a stopped source home containing memories whose validity crosses the run window."""
    source_home = tmp_path / "source-ledger"
    initialize_home(Config(home=source_home))
    con = open_direct(source_home / "memory.duckdb")
    try:
        # 1. Active memory valid throughout (does not cross window)
        con.execute(
            """
            INSERT INTO memories (
                memory_id, type, scope, content, content_hash, trust_level, confidence,
                created_at, valid_from, valid_to, expires_at
            ) VALUES (
                '00000000-0000-0000-0000-000000000001', 'fact', 'project:demo', ?, 'hash-1',
                'user', 0.9,
                TIMESTAMPTZ '2026-09-01 00:00:00+00',
                TIMESTAMPTZ '2026-09-01 00:00:00+00',
                NULL, NULL
            )
            """,
            [CANARY_MEMORY_TEXT],
        )
        # 2. Memory whose validity starts inside [2026-10-01, 2026-10-31]
        con.execute(
            """
            INSERT INTO memories (
                memory_id, type, scope, content, content_hash, trust_level, confidence,
                created_at, valid_from, valid_to, expires_at
            ) VALUES (
                '00000000-0000-0000-0000-000000000002', 'procedure', 'project:demo', ?, 'hash-2',
                'user', 0.8,
                TIMESTAMPTZ '2026-09-15 00:00:00+00',
                TIMESTAMPTZ '2026-10-10 00:00:00+00',
                NULL, NULL
            )
            """,
            [f"{CANARY_MEMORY_TEXT} starts in window"],
        )
        # 3. Memory whose validity ends inside [2026-10-01, 2026-10-31]
        con.execute(
            """
            INSERT INTO memories (
                memory_id, type, scope, content, content_hash, trust_level, confidence,
                created_at, valid_from, valid_to, expires_at
            ) VALUES (
                '00000000-0000-0000-0000-000000000003', 'decision', 'project:demo', ?, 'hash-3',
                'user', 0.8,
                TIMESTAMPTZ '2026-09-15 00:00:00+00',
                TIMESTAMPTZ '2026-09-15 00:00:00+00',
                NULL,
                TIMESTAMPTZ '2026-10-20 00:00:00+00'
            )
            """,
            [f"{CANARY_MEMORY_TEXT} expires in window"],
        )
        # 4. Retired memory whose valid_to also falls in window (total=3, non_terminal=2)
        con.execute(
            """
            INSERT INTO memories (
                memory_id, type, scope, content, content_hash, trust_level, confidence,
                created_at, valid_from, valid_to, expires_at
            ) VALUES (
                '00000000-0000-0000-0000-000000000004', 'fact', 'project:demo', ?, 'hash-4',
                'user', 0.5,
                TIMESTAMPTZ '2026-09-10 00:00:00+00',
                TIMESTAMPTZ '2026-09-10 00:00:00+00',
                TIMESTAMPTZ '2026-10-25 00:00:00+00',
                NULL
            )
            """,
            [f"{CANARY_MEMORY_TEXT} retired"],
        )
        con.execute(
            """
            INSERT INTO memory_actions (action_id, memory_id, actor, action, reason, created_at)
            VALUES (
                '10000000-0000-0000-0000-000000000004',
                '00000000-0000-0000-0000-000000000004',
                'human', 'retire', 'obsolete', TIMESTAMPTZ '2026-09-20 00:00:00+00'
            )
            """
        )
    finally:
        con.close()
    blobs = source_home / "blobs"
    blobs.mkdir(parents=True, exist_ok=True)
    (blobs / "sample.txt").write_text(f"blob {CANARY_MEMORY_TEXT}\n", encoding="utf-8")
    return source_home


def _write_task(
    tasks_dir: Path,
    task_id: str,
    *,
    repo: Path,
    commit: str,
    work_done_at: str,
    task_family: str,
    strata: list[str],
    project: str = "demo",
    exclude_reason: str | None = None,
    with_hidden_test: bool = False,
) -> Path:
    tdir = tasks_dir / task_id
    tdir.mkdir(parents=True, exist_ok=True)
    task_json: dict[str, Any] = {
        "task_id": task_id,
        "repository": str(repo),
        "commit": commit,
        "project": project,
        "work_done_at": work_done_at,
        "task_family": task_family,
        "strata": strata,
        "checks": {
            "commands": [{"name": "unit", "command": [sys.executable, "-c", "print('ok')"]}]
        },
    }
    if exclude_reason is not None:
        task_json["exclude_reason"] = exclude_reason
    (tdir / "task.json").write_text(json.dumps(task_json, indent=2) + "\n", encoding="utf-8")
    (tdir / "request.md").write_text(
        f"{CANARY_TASK_REQUEST} for {task_id}.\n",
        encoding="utf-8",
    )
    (tdir / "required_behavior.md").write_text(
        f"{CANARY_REQUIRED_BEHAVIOR} for {task_id}.\n",
        encoding="utf-8",
    )
    if with_hidden_test:
        ht = tdir / "hidden_tests"
        ht.mkdir(parents=True, exist_ok=True)
        (ht / "test_hidden.py").write_text(
            f"# {CANARY_HIDDEN_TEST}\ndef test_ok():\n    assert True\n",
            encoding="utf-8",
        )
    return tdir


def _write_trial_record(
    store: UsefulnessRunStore,
    *,
    trial_id: str,
    task_id: str,
    task_sha256: str,
    arm: str,
    model: str = "local-model-a",
    reasoning_effort: str | None = "high",
    registration_sha256: str | None = None,
    status: str = "completed",
    failure_category: str | None = None,
    total_tokens: int | None = 1000,
    input_tokens: int | None = 700,
    output_tokens: int | None = 300,
    thinking_tokens: int | None = None,
    factory_credits: float | None = 0.05,
    wall_total_s: float = 12.0,
    wall_harness_s: float = 10.0,
    wall_clone_s: float = 1.0,
    wall_setup_s: float = 0.5,
    wall_teardown_s: float = 0.5,
    recall_accounting: Mapping[str, Any] | None = None,
    write_checks: bool = True,
) -> tuple[Path, str]:
    trial_dir = store.trials_dir / trial_id
    trial_dir.mkdir(parents=True, exist_ok=True)
    record_path = trial_dir / "record.json"
    doc: dict[str, Any] = {
        "schema_version": 1,
        "trial_id": trial_id,
        "run_id": store.run_id,
        "task_id": task_id,
        "task_sha256": task_sha256,
        "registration_sha256": registration_sha256,
        "arm": arm,
        "status": status,
        "failure_category": failure_category,
        "failure_detail": CANARY_FAILURE_DETAIL if failure_category else None,
        "exit_code": 0 if status == "completed" else 1,
        "harness": "scripted",
        "harness_version": "scripted-1.0.0",
        "harness_session_id": f"sess-{trial_id}",
        "model": model,
        "reasoning_effort": reasoning_effort,
        "num_turns": 2,
        "last_reply": f"{CANARY_LAST_REPLY} ({trial_id})",
        "repo_changes": {
            "base_commit": "0" * 40,
            "head_commit": "0" * 40,
            "changed_files": ["app.py"],
            "untracked_files": [],
            "diff_stat": " app.py | 2 +-",
            "diff": f"+# {CANARY_DIFF_TEXT}\n",
            "has_changes": True,
        },
        "wall_time": {
            "clone_s": wall_clone_s,
            "setup_s": wall_setup_s,
            "harness_s": wall_harness_s,
            "teardown_s": wall_teardown_s,
            "total_s": wall_total_s,
            "harness_reported_duration_ms": wall_harness_s * 1000.0,
        },
        "tool_calls": [
            {
                "sequence": 1,
                "call_id": "call-1",
                "name": "Read",
                "arguments": {"secret": CANARY_TOOL_ARG},
                "is_error": False,
            }
        ],
        "tokens": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cache_read_input_tokens": None,
            "cache_creation_input_tokens": None,
            "thinking_tokens": thinking_tokens,
            "total_tokens": total_tokens,
            "factory_credits": factory_credits,
            "ttft_ms": 120.0,
        },
        "recall_accounting": dict(recall_accounting) if recall_accounting is not None else None,
        "started_at": "2026-10-15T12:00:00Z",
        "finished_at": "2026-10-15T12:00:12Z",
        "paths": {"clone_dir": str(trial_dir / "repo")},
        "environment": {"harness": "scripted"},
    }
    rec_sha = write_once_json(record_path, doc)
    if write_checks:
        write_once_json(
            trial_dir / "checks.json",
            {
                "schema_version": 1,
                "run_id": store.run_id,
                "trial_id": trial_id,
                "task_id": task_id,
                "task_sha256": task_sha256,
                "record_sha256": rec_sha,
                "checked_at": "2026-10-15T12:01:00Z",
                "overall_status": "passed",
                "passed_count": 1,
                "failed_count": 0,
                "could_not_run_count": 0,
                "total_duration_s": 0.1,
                "hidden_tests_installed": [],
                "hidden_tests_error": None,
                "checks": [
                    {
                        "name": "unit",
                        "command": ["python", "-c", "print('ok')"],
                        "status": "passed",
                        "exit_code": 0,
                        "duration_s": 0.1,
                        "stdout": CANARY_CHECK_OUTPUT,
                        "stderr": "",
                        "error_kind": None,
                        "error_message": None,
                    }
                ],
            },
        )
    return record_path, rec_sha


def _recall_accounting(
    *,
    made_no_recall: bool = False,
    returned_count: int = 1,
    miss_count: int = 0,
    degraded_count: int = 0,
    missing_record_count: int = 0,
    retrieval_latency_ms: float | None = 45.0,
    citations_made: int = 1,
    returned_memory_ids: Sequence[str] = ("m-1", "m-2"),
    returned_memories_cited_count: int = 1,
    placeholder_count: int = 1,
    bound_placeholder_count: int = 1,
    unbound_placeholder_count: int = 0,
    bound_reply_count: int = 1,
    single_citation_reply_count: int = 1,
    multi_citation_reply_count: int = 0,
) -> dict[str, Any]:
    total = (
        0 if made_no_recall else returned_count + miss_count + degraded_count + missing_record_count
    )
    return {
        "made_no_recall": made_no_recall,
        "total_recalls": total,
        "returned_count": 0 if made_no_recall else returned_count,
        "miss_count": 0 if made_no_recall else miss_count,
        "degraded_count": 0 if made_no_recall else degraded_count,
        "missing_record_count": 0 if made_no_recall else missing_record_count,
        "total_retrieval_latency_ms": None if made_no_recall else retrieval_latency_ms,
        "trial_session_ids": ["s-1"],
        "recalls": [],
        "citations": {
            "citations_made": citations_made,
            "cite_calls_in_transcript": citations_made,
            "degraded_cite_calls": 0,
            "citation_links": citations_made,
            "retrieval_links": citations_made,
            "cited_memory_links": returned_memories_cited_count,
            "context_event_links": 0,
            "returned_memory_ids": [] if made_no_recall else list(returned_memory_ids),
            "cited_memory_ids": list(returned_memory_ids[:returned_memories_cited_count]),
            "returned_memories_cited_count": returned_memories_cited_count,
            "returned_memories_cited_ids": list(
                returned_memory_ids[:returned_memories_cited_count]
            ),
            "placeholder_count": placeholder_count,
            "bound_placeholder_count": bound_placeholder_count,
            "unbound_placeholder_count": unbound_placeholder_count,
            "unbound_placeholder_ids": [],
            "bound_reply_count": bound_reply_count,
            "single_citation_reply_count": single_citation_reply_count,
            "single_citation_reply_ids": [],
            "multi_citation_reply_count": multi_citation_reply_count,
            "multi_citation_reply_ids": [],
            "citations": [],
        },
        "transcript_import": {
            "status": "imported",
            "harness": "droid",
            "transcript_path": None,
            "records_parsed": 4,
            "inserted": 4,
            "skipped": 0,
            "spooled": 0,
            "failed_lines": 0,
            "error": None,
        },
    }


def _registration_spec(
    *,
    sample_size: int,
    strata_rules: Mapping[str, str],
    models: Sequence[Mapping[str, Any]] | None = None,
    pooling: str = "single model; no cross-model pooling",
    exploratory_slices: Sequence[str] = ("exact_value",),
    bootstrap_seed: int = 20261126,
    bootstrap_resamples: int = 200,
    stop_at: Any | None = None,
    safety_stop_harmful_uses: int = 3,
) -> dict[str, Any]:
    model_entries = (
        list(models)
        if models is not None
        else [
            {
                "model": "local-model-a",
                "reasoning_effort": "high",
                "model_base_url": "http://127.0.0.1:11434",
            }
        ]
    )
    return {
        "primary_endpoint": {
            "endpoint": "paired_blind_success_rate_diff",
            "practical_benefit_threshold": 0.10,
        },
        "task_manifest": {
            "population": "coding tasks after cutoff",
            "inclusion_exclusion_rules": ["verifiable local checks"],
            "sampling_seed": 42,
            "sample_size": sample_size,
            "trials_per_task": 1,
            "strata_rules": dict(strata_rules),
        },
        "models": {
            "models": model_entries,
            "harness_settings": {"auto_level": "high", "seed_supported": False},
            "pooling": pooling,
        },
        "memory_tools": {
            "instructions": CUSTOM_INSTRUCTIONS,
            "descriptions": dict(CUSTOM_DESCRIPTIONS),
        },
        "calibration_settings": {"recall_limit": 8},
        "environment_manifest": {
            "harness": "scripted",
            "harness_version": "scripted-1.0.0",
            "repo_commit": "a1b2c3d4e5f60718293a4b5c6d7e8f9012345678",
            "allowed_base_tools": list(DEFAULT_BASE_TOOLS),
            "memory_server_name": "agent-memory",
            "hardware_class": "darwin-arm64",
            "config_overrides": default_trial_config_overrides("high"),
        },
        "limits": {
            "tokens": None,
            "wall_time_s": 60.0,
            "tool_calls": None,
            "retries": {"max_pair_retries": 0, "retryable_failure_categories": []},
            "enforced": ["wall_time_s", "retries"],
        },
        "invalidation_rule": "Invalidate only on pre-registered environment defect blind to arm.",
        "analysis": {
            "bootstrap_seed": bootstrap_seed,
            "bootstrap_resamples": bootstrap_resamples,
            "secondary_measures": "All secondary measures reported with denominator and 95% CI.",
            "exploratory_slices": list(exploratory_slices),
        },
        "stopping_rule": {
            "stop_at": stop_at
            if stop_at is not None
            else {"kind": "sample_size", "held_out_pairs": sample_size},
            "safety_stop_harmful_uses": safety_stop_harmful_uses,
            "sequential_error_control": "none (single fixed look at sample_size)",
        },
    }


def _build_rich_held_out_run(tmp_path: Path, run_id: str = "held-run") -> dict[str, Any]:
    """Build a complete, realistic held-out run covering every stratum and secondary measure."""
    source_home = _make_source_ledger_with_validity_crossings(tmp_path)
    exp_home = tmp_path / "exp-home"
    ms_res = make_memory_set(
        source_home,
        exp_home,
        cutoff=datetime(2026, 10, 1, 0, 0, tzinfo=UTC),
        last_day=datetime(2026, 10, 31, 23, 59, 59, tzinfo=UTC),
    )

    repo, commit = _make_git_repo(tmp_path)
    tasks_dir = exp_home / "tasks"
    _write_task(
        tasks_dir,
        "cal-01",
        repo=repo,
        commit=commit,
        work_done_at="2026-10-03T10:00:00Z",
        task_family="fam-cal",
        strata=["bugfix"],
    )
    _write_task(
        tasks_dir,
        "h-exact",
        repo=repo,
        commit=commit,
        work_done_at="2026-10-12T10:00:00Z",
        task_family="fam-1",
        strata=["exact_value"],
        with_hidden_test=True,
    )
    _write_task(
        tasks_dir,
        "h-recur",
        repo=repo,
        commit=commit,
        work_done_at="2026-10-14T10:00:00Z",
        task_family="fam-2",
        strata=["recurrence"],
    )
    _write_task(
        tasks_dir,
        "h-abstain",
        repo=repo,
        commit=commit,
        work_done_at="2026-10-16T10:00:00Z",
        task_family="fam-3",
        strata=["abstention"],
    )
    _write_task(
        tasks_dir,
        "h-bug",
        repo=repo,
        commit=commit,
        work_done_at="2026-10-18T10:00:00Z",
        task_family="fam-4",
        strata=["bugfix"],
    )
    _write_task(
        tasks_dir,
        "h-excluded",
        repo=repo,
        commit=commit,
        work_done_at="2026-10-19T10:00:00Z",
        task_family="fam-excl",
        strata=["bugfix"],
        exclude_reason="external_credentials_required",
    )

    store = UsefulnessRunStore(exp_home, run_id)
    tm = build_task_manifest(
        tasks_dir,
        cutoff="2026-10-01T00:00:00Z",
        calibration_ids=["cal-01"],
        held_out_ids=["h-exact", "h-recur", "h-abstain", "h-bug"],
    )
    store.write_task_manifest(tm)

    spec = _registration_spec(
        sample_size=4,
        strata_rules={
            "bugfix": "Fixes a reproducible bug.",
            "exact_value": "Hinges on an exact value.",
            "recurrence": "Tests whether an earlier verified error recurs.",
            "abstention": "Needs no prior memory.",
        },
        exploratory_slices=("exact_value",),
        bootstrap_seed=20261126,
        bootstrap_resamples=200,
    )
    reg_res = freeze_registration(exp_home, run_id, spec, created_at="2026-10-10T00:00:00Z")
    reg_sha = reg_res.registration_sha256

    create_run_schedule(exp_home, run_id, seed=7, split="held_out", repeats=1)

    # Pair 1: h-exact -> memory passes (recalled + cited), no_memory fails (confident wrong)
    _write_trial_record(
        store,
        trial_id="h-exact__r1__a1__memory",
        task_id="h-exact",
        task_sha256=tm["tasks"]["h-exact"]["task_sha256"],
        arm="memory",
        registration_sha256=reg_sha,
        total_tokens=1200,
        wall_total_s=14.0,
        wall_harness_s=11.0,
        recall_accounting=_recall_accounting(
            returned_count=1,
            citations_made=1,
            returned_memory_ids=("m-1", "m-2"),
            returned_memories_cited_count=1,
            placeholder_count=1,
            bound_placeholder_count=1,
            unbound_placeholder_count=0,
            bound_reply_count=1,
            single_citation_reply_count=1,
        ),
    )
    _write_trial_record(
        store,
        trial_id="h-exact__r1__a1__no_memory",
        task_id="h-exact",
        task_sha256=tm["tasks"]["h-exact"]["task_sha256"],
        arm="no_memory",
        registration_sha256=reg_sha,
        total_tokens=1000,
        wall_total_s=10.0,
        wall_harness_s=8.0,
    )

    # Pair 2: h-recur -> memory passes (no repeat error, recalled with miss+returned),
    # no_memory fails with repeated_earlier_error=True (ordinary_wrong_answer, stated failure)
    _write_trial_record(
        store,
        trial_id="h-recur__r1__a1__memory",
        task_id="h-recur",
        task_sha256=tm["tasks"]["h-recur"]["task_sha256"],
        arm="memory",
        registration_sha256=reg_sha,
        total_tokens=1500,
        wall_total_s=16.0,
        wall_harness_s=13.0,
        recall_accounting=_recall_accounting(
            returned_count=1,
            miss_count=1,
            degraded_count=0,
            missing_record_count=1,
            citations_made=2,
            returned_memory_ids=("m-3",),
            returned_memories_cited_count=1,
            placeholder_count=2,
            bound_placeholder_count=1,
            unbound_placeholder_count=1,
            bound_reply_count=1,
            single_citation_reply_count=0,
            multi_citation_reply_count=1,
        ),
    )
    _write_trial_record(
        store,
        trial_id="h-recur__r1__a1__no_memory",
        task_id="h-recur",
        task_sha256=tm["tasks"]["h-recur"]["task_sha256"],
        arm="no_memory",
        registration_sha256=reg_sha,
        total_tokens=None,  # Missing token count must stay None, never 0!
        input_tokens=None,
        output_tokens=None,
        wall_total_s=15.0,
        wall_harness_s=12.0,
    )

    # Pair 3: h-abstain -> memory made_no_recall (appropriate non-use!) & passes;
    # no_memory times out (non-ordinary failure category)
    _write_trial_record(
        store,
        trial_id="h-abstain__r1__a1__memory",
        task_id="h-abstain",
        task_sha256=tm["tasks"]["h-abstain"]["task_sha256"],
        arm="memory",
        registration_sha256=reg_sha,
        total_tokens=800,
        wall_total_s=9.0,
        wall_harness_s=7.0,
        recall_accounting=_recall_accounting(
            made_no_recall=True,
            citations_made=0,
            returned_memory_ids=(),
            returned_memories_cited_count=0,
            placeholder_count=0,
            bound_placeholder_count=0,
            unbound_placeholder_count=0,
            bound_reply_count=0,
            single_citation_reply_count=0,
        ),
    )
    _write_trial_record(
        store,
        trial_id="h-abstain__r1__a1__no_memory",
        task_id="h-abstain",
        task_sha256=tm["tasks"]["h-abstain"]["task_sha256"],
        arm="no_memory",
        registration_sha256=reg_sha,
        status="failed",
        failure_category="timeout",
        total_tokens=2000,
        wall_total_s=60.0,
        wall_harness_s=58.0,
    )

    # Pair 4: h-bug -> memory fails (confident wrong reply linked to harmful memory, arm_leaked!)
    # and degraded recall; no_memory passes
    _write_trial_record(
        store,
        trial_id="h-bug__r1__a1__memory",
        task_id="h-bug",
        task_sha256=tm["tasks"]["h-bug"]["task_sha256"],
        arm="memory",
        registration_sha256=reg_sha,
        total_tokens=1100,
        wall_total_s=11.0,
        wall_harness_s=9.0,
        recall_accounting=_recall_accounting(
            returned_count=1,
            degraded_count=1,
            citations_made=1,
            returned_memory_ids=("m-4",),
            returned_memories_cited_count=1,
            placeholder_count=1,
            bound_placeholder_count=1,
            unbound_placeholder_count=0,
            bound_reply_count=1,
            single_citation_reply_count=1,
        ),
    )
    _write_trial_record(
        store,
        trial_id="h-bug__r1__a1__no_memory",
        task_id="h-bug",
        task_sha256=tm["tasks"]["h-bug"]["task_sha256"],
        arm="no_memory",
        registration_sha256=reg_sha,
        total_tokens=900,
        wall_total_s=10.0,
        wall_harness_s=8.0,
    )

    # Grade all 8 trials (with canary note in grades)
    record_grade(
        exp_home,
        run_id,
        "h-exact__r1__a1__memory",
        met_checks=True,
        note=CANARY_GRADE_NOTE,
    )
    record_grade(
        exp_home,
        run_id,
        "h-exact__r1__a1__no_memory",
        met_checks=False,
        failure_category="ordinary_wrong_answer",
        confident_wrong_reply=True,
        note=CANARY_GRADE_NOTE,
    )
    record_grade(
        exp_home,
        run_id,
        "h-recur__r1__a1__memory",
        met_checks=True,
        repeated_earlier_error=False,
    )
    record_grade(
        exp_home,
        run_id,
        "h-recur__r1__a1__no_memory",
        met_checks=False,
        failure_category="ordinary_wrong_answer",
        confident_wrong_reply=False,
        repeated_earlier_error=True,
    )
    record_grade(
        exp_home,
        run_id,
        "h-abstain__r1__a1__memory",
        met_checks=True,
    )
    record_grade(
        exp_home,
        run_id,
        "h-abstain__r1__a1__no_memory",
        met_checks=False,
        failure_category="timeout",
        confident_wrong_reply=False,
    )
    record_grade(
        exp_home,
        run_id,
        "h-bug__r1__a1__memory",
        met_checks=False,
        failure_category="ordinary_wrong_answer",
        confident_wrong_reply=True,
        arm_leaked=True,
        note=CANARY_GRADE_NOTE,
    )
    record_grade(
        exp_home,
        run_id,
        "h-bug__r1__a1__no_memory",
        met_checks=True,
    )

    finalize_grades(exp_home, run_id)

    # Record post-grading memory-use attribution for the failed memory-arm trial
    record_attribution(
        exp_home,
        run_id,
        "h-bug__r1__a1__memory",
        harmful_memory_use=True,
        attempted_invalid_memory_use=False,
        confident_wrong_linked_to_memory=True,
        note=CANARY_ATTR_NOTE,
    )

    # Append a dated deviation
    append_deviation(
        exp_home,
        run_id,
        "Paused between pair 2 and pair 3 for laptop reboot.",
        date="2026-10-16T08:00:00Z",
        metadata={"kind": "schedule_pause"},
    )

    return {
        "exp_home": exp_home,
        "run_id": run_id,
        "store": store,
        "tm": tm,
        "reg_res": reg_res,
        "ms_res": ms_res,
    }


def test_report_primary_secondary_exploratory_and_integrity_sections(tmp_path: Path) -> None:
    fx = _build_rich_held_out_run(tmp_path)
    result = build_report(fx["exp_home"], fx["run_id"])
    assert isinstance(result, ReportResult)
    r = result.report
    md = result.markdown

    # Provenance & single-model scope limitation
    assert r["run_id"] == fx["run_id"]
    assert r["split"] == "held_out"
    assert r["includes_held_out"] is True
    assert r["stopping_status"]["stopped_by"] == "sample_size"
    assert r["grades_status"]["grades_final"] is True
    assert r["models"]["multi_model"] is False
    assert r["primary_endpoint"]["by_model"] == {}
    assert "local-model-a" in r["models"]["single_model_limitation"]
    assert "not an estimate for a weaker or stronger model" in md

    # Primary registered aggregate matches independent paired_success_summary exactly:
    # Pairs: h-abstain (T,F), h-bug (F,T), h-exact (T,F), h-recur (T,F) => memory 3/4, no_memory 1/4
    expected_agg = paired_success_summary(
        [
            BinaryPair(family="fam-3", memory=True, no_memory=False, task_id="h-abstain"),
            BinaryPair(family="fam-4", memory=False, no_memory=True, task_id="h-bug"),
            BinaryPair(family="fam-1", memory=True, no_memory=False, task_id="h-exact"),
            BinaryPair(family="fam-2", memory=True, no_memory=False, task_id="h-recur"),
        ],
        seed=20261126,
        n_resamples=200,
    )
    agg = r["primary_endpoint"]["registered_aggregate"]
    assert agg["exploratory"] is False
    assert agg["n_pairs"] == 4
    assert agg["memory_successes"] == 3
    assert agg["no_memory_successes"] == 1
    assert agg["difference"]["estimate"] == pytest.approx(expected_agg.difference.estimate)
    assert agg["difference"]["interval"]["low"] == pytest.approx(
        expected_agg.difference.interval.low  # type: ignore[union-attr]
    )
    assert agg["difference"]["interval"]["high"] == pytest.approx(
        expected_agg.difference.interval.high  # type: ignore[union-attr]
    )
    assert agg["exact_mcnemar"]["p_value"] == pytest.approx(expected_agg.exact_mcnemar.p_value)

    # Every stratum present; exact_value marked exploratory per registration
    by_stratum = r["primary_endpoint"]["by_stratum"]
    assert sorted(by_stratum) == ["abstention", "bugfix", "exact_value", "recurrence"]
    assert by_stratum["exact_value"]["exploratory"] is True
    assert by_stratum["exact_value"]["slice_label"] == "exploratory"
    assert by_stratum["bugfix"]["exploratory"] is False
    assert "Stratum: `exact_value` [EXPLORATORY]" in md

    # Exploratory slices explicitly labeled; recalled_pairs_only excludes h-abstain (made_no_recall)
    exp_recalled = r["exploratory_slices"]["recalled_pairs_only"]
    assert exp_recalled["exploratory"] is True
    assert exp_recalled["n_pairs"] == 3
    assert exp_recalled["memory_successes"] == 2
    assert exp_recalled["no_memory_successes"] == 1
    exp_no_leak = r["exploratory_slices"]["excluding_arm_leaked_pairs"]
    assert exp_no_leak["exploratory"] is True
    assert exp_no_leak["n_pairs"] == 3
    assert exp_no_leak["memory_successes"] == 3
    assert exp_no_leak["no_memory_successes"] == 0
    assert "## Exploratory Slices [EXPLORATORY]" in md

    # Secondary measures all carry denominators and intervals
    sec = r["secondary_measures"]
    assert sec["repeat_error_rate"]["memory_arm"]["numerator"] == 0
    assert sec["repeat_error_rate"]["memory_arm"]["denominator"] == 1
    assert sec["repeat_error_rate"]["memory_arm"]["wilson"] is not None
    assert sec["repeat_error_rate"]["no_memory_arm"]["numerator"] == 1
    assert sec["repeat_error_rate"]["no_memory_arm"]["denominator"] == 1

    assert sec["harmful_memory_use"]["combined"]["numerator"] == 1
    assert sec["harmful_memory_use"]["combined"]["denominator"] == 4
    assert sec["harmful_memory_use"]["combined"]["rate"]["interval"] is not None
    assert sec["harmful_memory_use"]["passing_trials_not_opened_in_attribution"] == 3

    cwr = sec["confident_wrong_reply"]
    # 3 memory-arm trials had declared/observed memory use (h-exact, h-recur, h-bug); 1 linked CWR
    assert cwr["memory_linked_among_memory_use"]["numerator"] == 1
    assert cwr["memory_linked_among_memory_use"]["denominator"] == 3
    assert cwr["memory_linked_among_memory_use"]["wilson"] is not None
    assert cwr["all_trials_memory_arm"]["numerator"] == 1
    assert cwr["all_trials_memory_arm"]["denominator"] == 4
    assert cwr["all_trials_no_memory_arm"]["numerator"] == 1
    assert cwr["all_trials_no_memory_arm"]["denominator"] == 4
    assert cwr["stated_failures_no_memory_arm"]["numerator"] == 2
    assert cwr["stated_failures_no_memory_arm"]["denominator"] == 4

    # Tokens preserve missing value (h-recur no_memory had None -> 1 missing pair, 3 complete)
    tok = sec["total_tokens"]["total_tokens"]
    assert tok["n_total_pairs"] == 4
    assert tok["n_pairs"] == 3
    assert tok["n_missing_pairs"] == 1
    assert tok["n_missing_no_memory"] == 1
    assert tok["difference_median"]["interval"] is not None
    assert tok["ratio_of_means"]["interval"] is not None
    assert "p90" in tok["difference_quantiles"]
    assert "p95" in tok["difference_quantiles"]

    # Recall & retrieval counts reported separately
    rr = sec["recall_and_retrieval"]
    assert rr["made_no_recall"]["numerator"] == 1
    assert rr["made_no_recall"]["denominator"] == 4
    assert rr["with_returned_memories"]["numerator"] == 3
    assert rr["with_miss"]["numerator"] == 1
    assert rr["with_degraded"]["numerator"] == 1
    assert rr["with_missing_record"]["numerator"] == 1

    # Citations
    cites = sec["citations"]
    assert cites["returned_memories_cited_rate"]["numerator"] == 3
    assert cites["returned_memories_cited_rate"]["denominator"] == 4
    assert cites["unbound_placeholder_rate"]["numerator"] == 1
    assert cites["unbound_placeholder_rate"]["denominator"] == 4
    assert cites["single_citation_reply_rate"]["numerator"] == 2
    assert cites["single_citation_reply_rate"]["denominator"] == 3

    # Abstention
    abst = sec["abstention"]
    assert abst["appropriate_non_use"]["numerator"] == 1
    assert abst["appropriate_non_use"]["denominator"] == 1
    assert abst["unnecessary_recall"]["numerator"] == 0
    assert abst["unnecessary_recall"]["denominator"] == 1

    # Non-ordinary failure categories, arm-leaked trials, deviations, memory validity crossings
    fc = r["failure_categories"]
    assert fc["graded_non_ordinary_total"] == 1
    assert fc["graded_non_ordinary"]["timeout"]["total"] == 1
    assert fc["graded_non_ordinary"]["timeout"]["trials"][0]["trial_id"] == (
        "h-abstain__r1__a1__no_memory"
    )
    assert fc["graded_ordinary_wrong_answer"]["total"] == 3

    assert r["arm_leaked"]["total"] == 1
    assert r["arm_leaked"]["trials"][0]["trial_id"] == "h-bug__r1__a1__memory"

    assert r["deviations"]["count"] == 1
    assert "laptop reboot" in r["deviations"]["items"][0]["reason"]
    assert "laptop reboot" in md

    vc = r["memory_set"]["validity_changes"]
    assert vc["total_crossing"] == 3
    assert vc["starts_in_window"] == 1
    assert vc["ends_in_window"] == 2
    assert vc["non_terminal_crossing"] == 2
    assert "**3** total crossing" in md

    # Excluded task recorded with reason code
    assert r["task_population"]["excluded_tasks"] == [
        {"task_id": "h-excluded", "reason_code": "external_credentials_required"}
    ]


def test_report_contains_no_task_transcript_or_memory_text_and_ignores_task_files(
    tmp_path: Path,
) -> None:
    fx = _build_rich_held_out_run(tmp_path)
    # Remove the tasks/ directory altogether to prove build_report never reads task text files.
    shutil.rmtree(fx["exp_home"] / "tasks")

    result = build_report(fx["exp_home"], fx["run_id"])
    combined = result.markdown + "\n" + result.json_text

    canaries = [
        CANARY_TASK_REQUEST,
        CANARY_REQUIRED_BEHAVIOR,
        CANARY_HIDDEN_TEST,
        CANARY_CHECK_OUTPUT,
        CANARY_LAST_REPLY,
        CANARY_DIFF_TEXT,
        CANARY_TOOL_ARG,
        CANARY_FAILURE_DETAIL,
        CANARY_MEMORY_TEXT,
        CANARY_GRADE_NOTE,
        CANARY_ATTR_NOTE,
        str(fx["exp_home"]),  # No local filesystem paths leaked
    ]
    for canary in canaries:
        assert canary not in combined, f"private text leaked into report: {canary!r}"


def test_report_is_byte_for_byte_deterministic_and_cli_round_trip(tmp_path: Path) -> None:
    fx = _build_rich_held_out_run(tmp_path)
    first = build_report(fx["exp_home"], fx["run_id"])
    second = build_report(fx["exp_home"], fx["run_id"])

    assert first.markdown.encode("utf-8") == second.markdown.encode("utf-8")
    assert first.json_text.encode("utf-8") == second.json_text.encode("utf-8")
    assert first.markdown_sha256 == second.markdown_sha256
    assert first.json_sha256 == second.json_sha256
    assert fx["store"].report_path.read_bytes() == first.markdown.encode("utf-8")
    assert fx["store"].report_json_path.read_bytes() == first.json_text.encode("utf-8")

    runner = CliRunner()
    md_1 = runner.invoke(
        cli_app,
        ["--home", str(fx["exp_home"]), "usefulness", "report", "--run-id", fx["run_id"]],
    )
    md_2 = runner.invoke(
        cli_app,
        ["--home", str(fx["exp_home"]), "usefulness", "report", "--run-id", fx["run_id"]],
    )
    assert md_1.exit_code == 0
    assert md_2.exit_code == 0
    assert md_1.stdout.encode("utf-8") == md_2.stdout.encode("utf-8")
    assert md_1.stdout == first.markdown

    js_1 = runner.invoke(
        cli_app,
        [
            "--home",
            str(fx["exp_home"]),
            "usefulness",
            "report",
            "--run-id",
            fx["run_id"],
            "--json",
            "--no-write",
        ],
    )
    assert js_1.exit_code == 0
    assert js_1.stdout == first.json_text
    assert json.loads(js_1.stdout)["run_id"] == fx["run_id"]


def test_report_multi_model_reports_each_model_and_refuses_missing_model(tmp_path: Path) -> None:
    source_home = _make_source_ledger_with_validity_crossings(tmp_path)
    exp_home = tmp_path / "exp-home"
    make_memory_set(
        source_home,
        exp_home,
        cutoff="2026-10-01T00:00:00Z",
        last_day="2026-10-31T23:59:59Z",
    )
    repo, commit = _make_git_repo(tmp_path)
    tasks_dir = exp_home / "tasks"
    _write_task(
        tasks_dir,
        "cal-01",
        repo=repo,
        commit=commit,
        work_done_at="2026-10-03T10:00:00Z",
        task_family="fam-cal",
        strata=["bugfix"],
    )
    _write_task(
        tasks_dir,
        "h-01",
        repo=repo,
        commit=commit,
        work_done_at="2026-10-12T10:00:00Z",
        task_family="fam-1",
        strata=["exact_value"],
    )
    _write_task(
        tasks_dir,
        "h-02",
        repo=repo,
        commit=commit,
        work_done_at="2026-10-14T10:00:00Z",
        task_family="fam-2",
        strata=["exact_value"],
    )
    run_id = "multi-model-run"
    store = UsefulnessRunStore(exp_home, run_id)
    tm = build_task_manifest(
        tasks_dir,
        cutoff="2026-10-01T00:00:00Z",
        calibration_ids=["cal-01"],
        held_out_ids=["h-01", "h-02"],
    )
    store.write_task_manifest(tm)
    spec = _registration_spec(
        sample_size=2,
        strata_rules={"bugfix": "bug", "exact_value": "exact"},
        models=[
            {"model": "model-alpha", "model_base_url": "http://127.0.0.1:11434"},
            {"model": "model-beta", "model_base_url": "http://127.0.0.1:11434"},
        ],
        pooling="equal-weight pooling across registered models plus per-model reporting",
        stop_at={"kind": "sample_size", "held_out_pairs": 4},
    )
    reg_res = freeze_registration(exp_home, run_id, spec, created_at="2026-10-10T00:00:00Z")
    reg_sha = reg_res.registration_sha256

    # Run model-alpha on both tasks first; model-beta has not run yet -> must refuse!
    for tid in ("h-01", "h-02"):
        for arm in ("memory", "no_memory"):
            _write_trial_record(
                store,
                trial_id=f"{tid}__alpha__{arm}",
                task_id=tid,
                task_sha256=tm["tasks"][tid]["task_sha256"],
                arm=arm,
                model="model-alpha",
                reasoning_effort=None,
                registration_sha256=reg_sha,
                recall_accounting=_recall_accounting() if arm == "memory" else None,
            )

    # Also add model-beta trials on both tasks; model-alpha gains +100pp, model-beta gains 0pp
    for tid in ("h-01", "h-02"):
        for arm in ("memory", "no_memory"):
            _write_trial_record(
                store,
                trial_id=f"{tid}__beta__{arm}",
                task_id=tid,
                task_sha256=tm["tasks"][tid]["task_sha256"],
                arm=arm,
                model="model-beta",
                reasoning_effort=None,
                registration_sha256=reg_sha,
                recall_accounting=_recall_accounting() if arm == "memory" else None,
            )

    for tid in ("h-01", "h-02"):
        record_grade(exp_home, run_id, f"{tid}__alpha__memory", met_checks=True)
        record_grade(
            exp_home,
            run_id,
            f"{tid}__alpha__no_memory",
            met_checks=False,
            failure_category="ordinary_wrong_answer",
        )
        record_grade(exp_home, run_id, f"{tid}__beta__memory", met_checks=True)
        record_grade(exp_home, run_id, f"{tid}__beta__no_memory", met_checks=True)
    finalize_grades(exp_home, run_id)

    result = build_report(exp_home, run_id)
    r = result.report
    assert r["models"]["multi_model"] is True
    assert r["models"]["single_model_limitation"] is None
    assert sorted(r["primary_endpoint"]["by_model"]) == ["model-alpha", "model-beta"]
    assert r["primary_endpoint"]["by_model"]["model-alpha"]["difference"]["estimate"] == (
        pytest.approx(1.0)
    )
    assert r["primary_endpoint"]["by_model"]["model-beta"]["difference"]["estimate"] == (
        pytest.approx(0.0)
    )
    assert r["primary_endpoint"]["registered_aggregate"]["difference"]["estimate"] == (
        pytest.approx(0.5)
    )
    assert "### By Model" in result.markdown
    assert "Model: `model-alpha`" in result.markdown
    assert "Model: `model-beta`" in result.markdown


def test_report_refuses_held_out_before_stop_or_final_grades_and_allows_calibration(
    tmp_path: Path,
) -> None:
    source_home = _make_source_ledger_with_validity_crossings(tmp_path)
    exp_home = tmp_path / "exp-home"
    make_memory_set(
        source_home,
        exp_home,
        cutoff="2026-10-01T00:00:00Z",
        last_day="2026-10-31T23:59:59Z",
    )
    repo, commit = _make_git_repo(tmp_path)
    tasks_dir = exp_home / "tasks"
    _write_task(
        tasks_dir,
        "cal-01",
        repo=repo,
        commit=commit,
        work_done_at="2026-10-03T10:00:00Z",
        task_family="fam-cal",
        strata=["bugfix"],
    )
    _write_task(
        tasks_dir,
        "h-01",
        repo=repo,
        commit=commit,
        work_done_at="2026-10-12T10:00:00Z",
        task_family="fam-h1",
        strata=["exact_value"],
    )
    _write_task(
        tasks_dir,
        "h-02",
        repo=repo,
        commit=commit,
        work_done_at="2026-10-14T10:00:00Z",
        task_family="fam-h2",
        strata=["exact_value"],
    )

    run_id = "gated-run"
    store = UsefulnessRunStore(exp_home, run_id)
    tm = build_task_manifest(
        tasks_dir,
        cutoff="2026-10-01T00:00:00Z",
        calibration_ids=["cal-01"],
        held_out_ids=["h-01", "h-02"],
    )
    store.write_task_manifest(tm)

    # Pre-freeze calibration trials can be reported via split="calibration"
    _write_trial_record(
        store,
        trial_id="cal-01__r1__a1__memory",
        task_id="cal-01",
        task_sha256=tm["tasks"]["cal-01"]["task_sha256"],
        arm="memory",
        recall_accounting=_recall_accounting(),
    )
    _write_trial_record(
        store,
        trial_id="cal-01__r1__a1__no_memory",
        task_id="cal-01",
        task_sha256=tm["tasks"]["cal-01"]["task_sha256"],
        arm="no_memory",
    )
    record_grade(exp_home, run_id, "cal-01__r1__a1__memory", met_checks=True)
    record_grade(
        exp_home,
        run_id,
        "cal-01__r1__a1__no_memory",
        met_checks=False,
        failure_category="ordinary_wrong_answer",
    )

    cal_report = build_report(
        exp_home,
        run_id,
        split="calibration",
        bootstrap_seed=9,
        bootstrap_resamples=100,
        write=False,
    )
    assert cal_report.report["split"] == "calibration"
    assert cal_report.report["includes_held_out"] is False
    assert cal_report.report["primary_endpoint"]["registered_aggregate"]["n_pairs"] == 1

    # Held-out refused before freeze
    with pytest.raises(UsefulnessError, match="registration"):
        build_report(exp_home, run_id, split="held_out", write=False)

    # Freeze registration and schedule 2 held-out pairs
    spec = _registration_spec(
        sample_size=2,
        strata_rules={"bugfix": "bug", "exact_value": "exact"},
    )
    reg_res = freeze_registration(exp_home, run_id, spec, created_at="2026-10-10T00:00:00Z")
    reg_sha = reg_res.registration_sha256
    create_run_schedule(exp_home, run_id, seed=5, split="held_out", repeats=1)

    # Run only 1 of 2 scheduled pairs -> still pending slots -> refused even if graded!
    _write_trial_record(
        store,
        trial_id="h-01__r1__a1__memory",
        task_id="h-01",
        task_sha256=tm["tasks"]["h-01"]["task_sha256"],
        arm="memory",
        registration_sha256=reg_sha,
        recall_accounting=_recall_accounting(),
    )
    _write_trial_record(
        store,
        trial_id="h-01__r1__a1__no_memory",
        task_id="h-01",
        task_sha256=tm["tasks"]["h-01"]["task_sha256"],
        arm="no_memory",
        registration_sha256=reg_sha,
    )
    record_grade(exp_home, run_id, "h-01__r1__a1__memory", met_checks=True)
    record_grade(exp_home, run_id, "h-01__r1__a1__no_memory", met_checks=True)

    # Refused because grades are not finalized yet
    with pytest.raises(UsefulnessError, match="final"):
        build_report(exp_home, run_id, split="held_out", write=False)

    # Even if grades were finalized early on only 1 pair, refused because schedule has pending slots
    finalize_grades(exp_home, run_id)
    with pytest.raises(UsefulnessError, match="pending trial"):
        build_report(exp_home, run_id, split="held_out", write=False)

    # Run second pair after grades-final.json was already written -> refused as ungraded finished
    _write_trial_record(
        store,
        trial_id="h-02__r1__a1__memory",
        task_id="h-02",
        task_sha256=tm["tasks"]["h-02"]["task_sha256"],
        arm="memory",
        registration_sha256=reg_sha,
        recall_accounting=_recall_accounting(),
    )
    _write_trial_record(
        store,
        trial_id="h-02__r1__a1__no_memory",
        task_id="h-02",
        task_sha256=tm["tasks"]["h-02"]["task_sha256"],
        arm="no_memory",
        registration_sha256=reg_sha,
    )
    with pytest.raises(UsefulnessError, match=r"not covered by grades-final\.json"):
        build_report(exp_home, run_id, split="held_out", write=False)


def test_report_allows_safety_stop_and_resource_budget_stop_with_interrupted_trial(
    tmp_path: Path,
) -> None:
    source_home = _make_source_ledger_with_validity_crossings(tmp_path)
    exp_home = tmp_path / "exp-home"
    make_memory_set(
        source_home,
        exp_home,
        cutoff="2026-10-01T00:00:00Z",
        last_day="2026-10-31T23:59:59Z",
    )
    repo, commit = _make_git_repo(tmp_path)
    tasks_dir = exp_home / "tasks"
    _write_task(
        tasks_dir,
        "cal-01",
        repo=repo,
        commit=commit,
        work_done_at="2026-10-03T10:00:00Z",
        task_family="fam-cal",
        strata=["bugfix"],
    )
    _write_task(
        tasks_dir,
        "h-01",
        repo=repo,
        commit=commit,
        work_done_at="2026-10-12T10:00:00Z",
        task_family="fam-1",
        strata=["exact_value"],
    )
    _write_task(
        tasks_dir,
        "h-02",
        repo=repo,
        commit=commit,
        work_done_at="2026-10-14T10:00:00Z",
        task_family="fam-2",
        strata=["exact_value"],
    )

    # --- Case 1: Safety stop halts run after 1 harmful use (even with pending slots + interrupted)
    run_safe = "safety-stop-run"
    store = UsefulnessRunStore(exp_home, run_safe)
    tm = build_task_manifest(
        tasks_dir,
        cutoff="2026-10-01T00:00:00Z",
        calibration_ids=["cal-01"],
        held_out_ids=["h-01", "h-02"],
    )
    store.write_task_manifest(tm)
    spec = _registration_spec(
        sample_size=2,
        strata_rules={"bugfix": "bug", "exact_value": "exact"},
        safety_stop_harmful_uses=1,
    )
    reg_res = freeze_registration(exp_home, run_safe, spec, created_at="2026-10-10T00:00:00Z")
    create_run_schedule(exp_home, run_safe, seed=3, split="held_out", repeats=1)

    _write_trial_record(
        store,
        trial_id="h-01__r1__a1__memory",
        task_id="h-01",
        task_sha256=tm["tasks"]["h-01"]["task_sha256"],
        arm="memory",
        registration_sha256=reg_res.registration_sha256,
        recall_accounting=_recall_accounting(),
    )
    # Partner arm hard-interrupted before writing record.json (marker has no model field)
    write_once_json(
        store.trials_dir / "h-01__r1__a1__no_memory" / "interrupted.json",
        {
            "schema_version": 1,
            "recorded_at": "2026-10-15T12:05:00Z",
            "run_id": run_safe,
            "pair_id": "h-01__r1",
            "task_id": "h-01",
            "repeat_index": 0,
            "attempt": 1,
            "arm": "no_memory",
            "trial_id": "h-01__r1__a1__no_memory",
            "status": "interrupted",
            "failure_category": "interrupted",
            "reason": "process killed",
        },
    )
    record_grade(
        exp_home,
        run_safe,
        "h-01__r1__a1__memory",
        met_checks=False,
        failure_category="ordinary_wrong_answer",
    )
    finalize_grades(exp_home, run_safe)
    record_attribution(
        exp_home,
        run_safe,
        "h-01__r1__a1__memory",
        harmful_memory_use=True,
        attempted_invalid_memory_use=False,
        confident_wrong_linked_to_memory=False,
    )

    safe_report = build_report(exp_home, run_safe, split="held_out")
    assert safe_report.report["stopping_status"]["stopped_by"] == "safety_stop"
    # Interrupted partner inherited local-model-a (no phantom "unknown" model) and counted missing
    assert safe_report.report["models"]["observed"] == ["local-model-a"]
    assert safe_report.report["primary_endpoint"]["registered_aggregate"]["n_missing_pairs"] == 1
    assert (
        safe_report.report["failure_categories"]["execution_failures"]["interrupted"]["total"] == 1
    )

    # --- Case 2: Resource budget stop halts run before schedule finishes ---
    run_budget = "budget-stop-run"
    store_b = UsefulnessRunStore(exp_home, run_budget)
    store_b.write_task_manifest(tm)
    spec_b = _registration_spec(
        sample_size=2,
        strata_rules={"bugfix": "bug", "exact_value": "exact"},
        stop_at={"kind": "budget", "max_total_tokens": 5000},
        safety_stop_harmful_uses=5,
    )
    reg_b = freeze_registration(exp_home, run_budget, spec_b, created_at="2026-10-10T00:00:00Z")
    create_run_schedule(exp_home, run_budget, seed=3, split="held_out", repeats=1)
    _write_trial_record(
        store_b,
        trial_id="h-01__r1__a1__memory",
        task_id="h-01",
        task_sha256=tm["tasks"]["h-01"]["task_sha256"],
        arm="memory",
        registration_sha256=reg_b.registration_sha256,
        total_tokens=3000,
        recall_accounting=_recall_accounting(),
    )
    _write_trial_record(
        store_b,
        trial_id="h-01__r1__a1__no_memory",
        task_id="h-01",
        task_sha256=tm["tasks"]["h-01"]["task_sha256"],
        arm="no_memory",
        registration_sha256=reg_b.registration_sha256,
        total_tokens=2500,
    )
    record_grade(exp_home, run_budget, "h-01__r1__a1__memory", met_checks=True)
    record_grade(exp_home, run_budget, "h-01__r1__a1__no_memory", met_checks=True)
    finalize_grades(exp_home, run_budget)

    budget_report = build_report(exp_home, run_budget, split="held_out")
    assert budget_report.report["stopping_status"]["stopped_by"] == "resource_budget"


def test_report_resolves_retried_pairs_and_refuses_bootstrap_override(tmp_path: Path) -> None:
    source_home = _make_source_ledger_with_validity_crossings(tmp_path)
    exp_home = tmp_path / "exp-home"
    make_memory_set(
        source_home,
        exp_home,
        cutoff="2026-10-01T00:00:00Z",
        last_day="2026-10-31T23:59:59Z",
    )
    repo, commit = _make_git_repo(tmp_path)
    tasks_dir = exp_home / "tasks"
    _write_task(
        tasks_dir,
        "cal-01",
        repo=repo,
        commit=commit,
        work_done_at="2026-10-03T10:00:00Z",
        task_family="fam-cal",
        strata=["bugfix"],
    )
    _write_task(
        tasks_dir,
        "h-01",
        repo=repo,
        commit=commit,
        work_done_at="2026-10-12T10:00:00Z",
        task_family="fam-1",
        strata=["exact_value"],
    )
    run_id = "retry-run"
    store = UsefulnessRunStore(exp_home, run_id)
    tm = build_task_manifest(
        tasks_dir,
        cutoff="2026-10-01T00:00:00Z",
        calibration_ids=["cal-01"],
        held_out_ids=["h-01"],
    )
    store.write_task_manifest(tm)
    spec = _registration_spec(
        sample_size=1,
        strata_rules={"bugfix": "bug", "exact_value": "exact"},
        bootstrap_seed=777,
        bootstrap_resamples=150,
    )
    reg_res = freeze_registration(exp_home, run_id, spec, created_at="2026-10-10T00:00:00Z")
    reg_sha = reg_res.registration_sha256
    create_run_schedule(
        exp_home,
        run_id,
        seed=1,
        split="held_out",
        repeats=1,
        retry_rule=RetryRule(max_pair_retries=1, retryable_failure_categories=("tool_outage",)),
    )

    # Attempt 1: memory hit tool_outage, no_memory completed
    _write_trial_record(
        store,
        trial_id="h-01__r1__a1__memory",
        task_id="h-01",
        task_sha256=tm["tasks"]["h-01"]["task_sha256"],
        arm="memory",
        registration_sha256=reg_sha,
        status="failed",
        failure_category="tool_outage",
        total_tokens=400,
        wall_total_s=5.0,
        recall_accounting=_recall_accounting(made_no_recall=True, citations_made=0),
    )
    _write_trial_record(
        store,
        trial_id="h-01__r1__a1__no_memory",
        task_id="h-01",
        task_sha256=tm["tasks"]["h-01"]["task_sha256"],
        arm="no_memory",
        registration_sha256=reg_sha,
        total_tokens=900,
        wall_total_s=8.0,
    )
    # Attempt 2: both arms completed cleanly
    _write_trial_record(
        store,
        trial_id="h-01__r1__a2__memory",
        task_id="h-01",
        task_sha256=tm["tasks"]["h-01"]["task_sha256"],
        arm="memory",
        registration_sha256=reg_sha,
        total_tokens=1100,
        wall_total_s=12.0,
        recall_accounting=_recall_accounting(),
    )
    _write_trial_record(
        store,
        trial_id="h-01__r1__a2__no_memory",
        task_id="h-01",
        task_sha256=tm["tasks"]["h-01"]["task_sha256"],
        arm="no_memory",
        registration_sha256=reg_sha,
        total_tokens=1000,
        wall_total_s=10.0,
    )

    # Grade all 4 trials (including attempt 1)
    record_grade(
        exp_home,
        run_id,
        "h-01__r1__a1__memory",
        met_checks=False,
        failure_category="tool_outage",
    )
    record_grade(exp_home, run_id, "h-01__r1__a1__no_memory", met_checks=True)
    record_grade(exp_home, run_id, "h-01__r1__a2__memory", met_checks=True)
    record_grade(
        exp_home,
        run_id,
        "h-01__r1__a2__no_memory",
        met_checks=False,
        failure_category="ordinary_wrong_answer",
    )
    finalize_grades(exp_home, run_id)

    # Cannot override registered bootstrap_seed or bootstrap_resamples on held-out
    with pytest.raises(UsefulnessError, match="bootstrap_seed"):
        build_report(exp_home, run_id, bootstrap_seed=999, write=False)
    with pytest.raises(UsefulnessError, match="bootstrap_resamples"):
        build_report(exp_home, run_id, bootstrap_resamples=500, write=False)

    res = build_report(exp_home, run_id)
    r = res.report
    # Resolved primary endpoint uses attempt 2 (1 pair, memory=True, no_memory=False)
    agg = r["primary_endpoint"]["registered_aggregate"]
    assert agg["n_pairs"] == 1
    assert agg["memory_successes"] == 1
    assert agg["no_memory_successes"] == 0
    assert r["task_population"]["superseded_retry_trials"] == 2
    # Attempt 1 tool_outage still listed in non-ordinary failure categories & retry time accounted
    assert r["failure_categories"]["graded_non_ordinary"]["tool_outage"]["total"] == 1
    assert r["secondary_measures"]["latency"]["retry_time_s"]["total_s"] == pytest.approx(13.0)
    assert "total_tokens_including_retries" in r["secondary_measures"]["total_tokens"]
    assert "total_s_including_retries" in r["secondary_measures"]["latency"]
    assert sha256_file(store.report_path) == res.markdown_sha256
    assert sha256_file(store.report_json_path) == res.json_sha256
