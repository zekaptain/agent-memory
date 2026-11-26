"""Tests for usefulness registration, freeze enforcement, wording, and deviations (MEM-23)."""

from __future__ import annotations

import asyncio
import copy
import json
import os
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from agent_memory.cli import main as cli_main
from agent_memory.config import Config
from agent_memory.db import MemoryServer, initialize_home
from agent_memory.mcp_server import (
    DEFAULT_TOOL_DESCRIPTIONS,
    INSTRUCTIONS,
    MCP_WORDING_ENV,
    TOOL_NAMES,
    default_tool_wording,
)
from agent_memory.smoke import free_port
from agent_memory.usefulness import (
    DEFAULT_BASE_TOOLS,
    REGISTRATION_SCHEMA_VERSION,
    DroidHarnessAdapter,
    HarnessInvocation,
    HarnessResult,
    ScriptedHarnessAdapter,
    TrialConfig,
    UsefulnessError,
    UsefulnessRunStore,
    append_deviation,
    build_environment_manifest,
    build_memory_tool_wording,
    build_task_manifest,
    default_trial_config_overrides,
    freeze_registration,
    list_deviations,
    load_registration,
    make_memory_set,
    memory_tool_wording_sha256,
    run_trial,
    sha256_file,
    verify_registration_artifacts,
    verify_registration_for_trial,
)

CUSTOM_INSTRUCTIONS = (
    "Registered custom experiment instructions (freeze v1, not installed default)."
)
CUSTOM_DESCRIPTIONS: dict[str, str] = {
    name: f"Registered custom description for {name} (freeze v1)." for name in TOOL_NAMES
}


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
) -> Path:
    tdir = tasks_dir / task_id
    tdir.mkdir(parents=True, exist_ok=True)
    (tdir / "task.json").write_text(
        json.dumps(
            {
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
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (tdir / "request.md").write_text(f"Solve task {task_id}.\n", encoding="utf-8")
    (tdir / "required_behavior.md").write_text("Unit check passes.\n", encoding="utf-8")
    return tdir


def _setup_run_fixtures(tmp_path: Path, run_id: str = "run-001") -> dict[str, Any]:
    """Create a source home, memory set, task manifest (1 cal + 1 held-out), and spec."""
    source_home = tmp_path / "source-ledger"
    initialize_home(Config(home=source_home))
    blobs = source_home / "blobs"
    blobs.mkdir(parents=True, exist_ok=True)
    (blobs / "note.txt").write_text("synthetic blob content\n", encoding="utf-8")

    exp_home = tmp_path / "exp-home"
    ms_manifest = make_memory_set(
        source_home,
        exp_home,
        cutoff="2026-10-01T00:00:00Z",
        last_day="2026-10-31T23:59:59Z",
    )

    repo, commit = _make_git_repo(tmp_path)
    tasks_dir = exp_home / "tasks"
    cal_dir = _write_task(
        tasks_dir,
        "cal-01",
        repo=repo,
        commit=commit,
        work_done_at="2026-10-05T10:00:00Z",
        task_family="fam-cal",
        strata=["bugfix"],
    )
    held_dir = _write_task(
        tasks_dir,
        "held-01",
        repo=repo,
        commit=commit,
        work_done_at="2026-10-15T10:00:00Z",
        task_family="fam-held",
        strata=["exact_value"],
    )
    tm_manifest = build_task_manifest(
        tasks_dir,
        cutoff="2026-10-01T00:00:00Z",
        calibration_ids=["cal-01"],
        held_out_ids=["held-01"],
    )
    UsefulnessRunStore(exp_home, run_id).write_task_manifest(tm_manifest)

    harness_version = "scripted-1.0.0"
    repo_commit = "a1b2c3d4e5f60718293a4b5c6d7e8f9012345678"
    hardware_class = "darwin-arm64"
    auto_level = "high"

    spec = _valid_registration_spec(
        harness="scripted",
        harness_version=harness_version,
        repo_commit=repo_commit,
        hardware_class=hardware_class,
        auto_level=auto_level,
    )

    return {
        "exp_home": exp_home,
        "source_home": source_home,
        "run_id": run_id,
        "repo": repo,
        "commit": commit,
        "cal_dir": cal_dir,
        "held_dir": held_dir,
        "ms_manifest": ms_manifest,
        "tm_manifest": tm_manifest,
        "spec": spec,
        "harness_version": harness_version,
        "repo_commit": repo_commit,
        "hardware_class": hardware_class,
        "auto_level": auto_level,
    }


def _valid_registration_spec(
    *,
    harness: str = "scripted",
    harness_version: str = "scripted-1.0.0",
    repo_commit: str = "a1b2c3d4e5f60718293a4b5c6d7e8f9012345678",
    hardware_class: str = "darwin-arm64",
    auto_level: str = "high",
    instructions: str = CUSTOM_INSTRUCTIONS,
    descriptions: dict[str, str] | None = None,
) -> dict[str, Any]:
    descs = dict(CUSTOM_DESCRIPTIONS if descriptions is None else descriptions)
    return {
        "primary_endpoint": {
            "endpoint": "paired_blind_success_rate_diff",
            "practical_benefit_threshold": 0.10,
        },
        "task_manifest": {
            "population": "coding tasks from real sessions after memory-set cutoff",
            "inclusion_exclusion_rules": [
                "include tasks with verifiable local checks",
                "exclude tasks requiring external credentials",
            ],
            "sampling_seed": 20261125,
            "sample_size": 1,
            "trials_per_task": 1,
            "strata_rules": {
                "bugfix": "task fixes a defect reproduced by a check",
                "exact_value": "task hinges on an exact value such as a port, flag, or identifier",
            },
        },
        "models": {
            "models": [
                {
                    "model": "local-test-model",
                    "reasoning_effort": "high",
                    "model_base_url": "http://127.0.0.1:11434",
                }
            ],
            "harness_settings": {
                "auto_level": auto_level,
                "seed_supported": False,
                "temperature_supported": False,
            },
            "pooling": "single model; no cross-model pooling",
        },
        "memory_tools": {
            "instructions": instructions,
            "descriptions": descs,
        },
        "calibration_settings": {
            "recall_limit": 8,
            "tool_wording_variant": "freeze-v1",
        },
        "environment_manifest": {
            "harness": harness,
            "harness_version": harness_version,
            "repo_commit": repo_commit,
            "allowed_base_tools": list(DEFAULT_BASE_TOOLS),
            "memory_server_name": "agent-memory",
            "hardware_class": hardware_class,
            "config_overrides": default_trial_config_overrides(auto_level),
        },
        "limits": {
            "tokens": None,
            "wall_time_s": 60.0,
            "tool_calls": None,
            "retries": {"max_retries": 1, "retryable_categories": ["tool_outage"]},
            "enforced": ["wall_time_s", "retries"],
        },
        "invalidation_rule": (
            "A pair is invalid if the task check fails on the reference commit or the harness "
            "suffers a non-retryable environment error."
        ),
        "analysis": {
            "bootstrap_seed": 123456,
            "bootstrap_resamples": 2000,
            "secondary_measures": {
                "wall_time_ratio": "paired ratio of medians with family bootstrap CI",
                "token_ratio": "paired ratio of medians with family bootstrap CI",
            },
            "exploratory_slices": ["exact_value", "bugfix"],
        },
        "stopping_rule": {
            "stop_at": {"kind": "sample_size", "held_out_pairs": 1},
            "safety_stop_harmful_uses": 3,
            "sequential_error_control": "none (single fixed look at sample_size)",
        },
    }


def _trial_config(
    fx: dict[str, Any],
    *,
    arm: str = "no_memory",
    model: str = "local-test-model",
    reasoning_effort: str | None = "high",
    wall_time_limit_s: float = 60.0,
    auto_level: str = "high",
    source_home: Path | None = None,
    verify_manifest: bool = True,
    memory_tool_wording: dict[str, Any] | None = None,
    memory_tool_wording_file: Path | None = None,
    repo_commit: str | None = None,
    hardware_class: str | None = None,
    harness_version: str | None = None,
    base_tools: tuple[str, ...] = DEFAULT_BASE_TOOLS,
    memory_server_name: str = "agent-memory",
) -> TrialConfig:
    return TrialConfig(
        arm=arm,  # type: ignore[arg-type]
        model=model,
        reasoning_effort=reasoning_effort,
        wall_time_limit_s=wall_time_limit_s,
        auto_level=auto_level,
        model_base_url="http://127.0.0.1:11434",
        source_home=source_home,
        verify_manifest=verify_manifest,
        base_tools=base_tools,
        memory_server_name=memory_server_name,
        memory_tool_wording=memory_tool_wording,
        memory_tool_wording_file=memory_tool_wording_file,
        repo_commit=fx["repo_commit"] if repo_commit is None else repo_commit,
        hardware_class=fx["hardware_class"] if hardware_class is None else hardware_class,
        harness_version=fx["harness_version"] if harness_version is None else harness_version,
    )


def _ok_adapter(
    fx: dict[str, Any],
    *,
    name: str = "scripted",
    version: str | None = None,
) -> ScriptedHarnessAdapter:
    def _handler(inv: HarnessInvocation) -> HarnessResult:
        return HarnessResult(
            status="completed",
            last_reply=f"done {inv.task.task_id} ({inv.arm})",
            exit_code=0,
            harness_version=fx["harness_version"] if version is None else version,
            resolved_model=inv.config.model,
            resolved_reasoning_effort=inv.config.reasoning_effort,
            num_turns=1,
        )

    adapter = ScriptedHarnessAdapter(
        _handler,
        version=fx["harness_version"] if version is None else version,
    )
    adapter.name = name
    return adapter


# ---------------------------------------------------------------------------
# 1. Freeze writes one complete registration.json and refuses missing fields
# ---------------------------------------------------------------------------


def test_freeze_writes_complete_registration_and_binds_hashes(tmp_path: Path) -> None:
    fx = _setup_run_fixtures(tmp_path)
    res = freeze_registration(fx["exp_home"], fx["run_id"], fx["spec"])

    store = UsefulnessRunStore(fx["exp_home"], fx["run_id"])
    assert res.registration_path == store.registration_path
    assert res.registration_path.is_file()
    assert res.registration_sha256 == sha256_file(res.registration_path)
    assert res.registration_sha256 == store.registration_sha256()

    reg = res.registration
    assert reg["schema_version"] == REGISTRATION_SCHEMA_VERSION
    assert reg["run_id"] == fx["run_id"]
    assert reg["primary_endpoint"]["practical_benefit_threshold"] == 0.10

    # Bound hashes match live artifacts
    assert reg["task_manifest"]["sha256"] == store.task_manifest_sha256()
    assert reg["task_manifest"]["families"] == fx["tm_manifest"]["families"]
    assert reg["task_manifest"]["split"] == fx["tm_manifest"]["split"]
    assert "exact_value" in reg["task_manifest"]["strata_rules"]

    ms_doc = fx["ms_manifest"].manifest
    assert reg["memory_set"]["sha256"] == ms_doc["memory_set_sha256"]
    assert reg["memory_set"]["cutoff"] == ms_doc["cutoff"]
    assert reg["memory_set"]["last_day"] == ms_doc["last_day"]

    expected_mt_sha = memory_tool_wording_sha256(CUSTOM_INSTRUCTIONS, CUSTOM_DESCRIPTIONS)
    assert reg["memory_tools"]["sha256"] == expected_mt_sha
    assert reg["memory_tools"]["instructions"] == CUSTOM_INSTRUCTIONS
    assert reg["memory_tools"]["descriptions"] == CUSTOM_DESCRIPTIONS

    expected_env = build_environment_manifest(
        harness="scripted",
        harness_version=fx["harness_version"],
        repo_commit=fx["repo_commit"],
        allowed_base_tools=DEFAULT_BASE_TOOLS,
        memory_server_name="agent-memory",
        hardware_class=fx["hardware_class"],
        config_overrides=default_trial_config_overrides(fx["auto_level"]),
    )
    assert reg["environment_manifest"]["sha256"] == expected_env["sha256"]
    assert reg["limits"]["enforced"] == ["wall_time_s", "retries"]
    assert reg["analysis"]["bootstrap_seed"] == 123456
    assert reg["stopping_rule"]["safety_stop_harmful_uses"] == 3

    loaded = load_registration(fx["exp_home"], fx["run_id"])
    assert loaded.registration_sha256 == res.registration_sha256
    assert loaded.registration == reg


@pytest.mark.parametrize(
    "drop_path",
    [
        ("primary_endpoint",),
        ("primary_endpoint", "endpoint"),
        ("primary_endpoint", "practical_benefit_threshold"),
        ("task_manifest",),
        ("task_manifest", "population"),
        ("task_manifest", "inclusion_exclusion_rules"),
        ("task_manifest", "sampling_seed"),
        ("task_manifest", "sample_size"),
        ("task_manifest", "trials_per_task"),
        ("task_manifest", "strata_rules"),
        ("task_manifest", "strata_rules", "exact_value"),
        ("task_manifest", "strata_rules", "bugfix"),
        ("models",),
        ("models", "models"),
        ("models", "harness_settings"),
        ("models", "pooling"),
        ("memory_tools",),
        ("memory_tools", "instructions"),
        ("memory_tools", "descriptions"),
        ("memory_tools", "descriptions", "recall"),
        ("calibration_settings",),
        ("environment_manifest",),
        ("environment_manifest", "harness"),
        ("environment_manifest", "harness_version"),
        ("environment_manifest", "repo_commit"),
        ("environment_manifest", "allowed_base_tools"),
        ("environment_manifest", "memory_server_name"),
        ("environment_manifest", "hardware_class"),
        ("environment_manifest", "config_overrides"),
        ("limits",),
        ("limits", "wall_time_s"),
        ("limits", "enforced"),
        ("invalidation_rule",),
        ("analysis",),
        ("analysis", "bootstrap_seed"),
        ("analysis", "bootstrap_resamples"),
        ("analysis", "secondary_measures"),
        ("stopping_rule",),
        ("stopping_rule", "stop_at"),
        ("stopping_rule", "safety_stop_harmful_uses"),
        ("stopping_rule", "sequential_error_control"),
    ],
)
def test_freeze_refuses_when_required_field_is_missing(
    tmp_path: Path, drop_path: tuple[str, ...]
) -> None:
    fx = _setup_run_fixtures(tmp_path)
    broken = copy.deepcopy(fx["spec"])
    cursor: Any = broken
    for key in drop_path[:-1]:
        cursor = cursor[key]
    del cursor[drop_path[-1]]

    with pytest.raises(UsefulnessError):
        freeze_registration(fx["exp_home"], fx["run_id"], broken)

    store = UsefulnessRunStore(fx["exp_home"], fx["run_id"])
    assert not store.registration_path.exists()


def test_freeze_refuses_invalid_limits_trials_per_task_or_artifact_mismatch(tmp_path: Path) -> None:
    fx = _setup_run_fixtures(tmp_path)

    # Unknown enforced limit name
    bad_limits = copy.deepcopy(fx["spec"])
    bad_limits["limits"]["enforced"] = ["wall_time_s", "unknown_limit"]
    with pytest.raises(UsefulnessError, match="unknown limit"):
        freeze_registration(fx["exp_home"], fx["run_id"], bad_limits)

    # Per-task trials map missing a manifest task
    bad_tpt = copy.deepcopy(fx["spec"])
    bad_tpt["task_manifest"]["trials_per_task"] = {"cal-01": 1}
    with pytest.raises(UsefulnessError, match="missing task"):
        freeze_registration(fx["exp_home"], fx["run_id"], bad_tpt)

    # Explicit task_manifest.sha256 disagrees with run's task-manifest.json
    bad_tm = copy.deepcopy(fx["spec"])
    bad_tm["task_manifest"]["sha256"] = "0" * 64
    with pytest.raises(UsefulnessError, match="does not match"):
        freeze_registration(fx["exp_home"], fx["run_id"], bad_tm)

    # Explicit memory_set.cutoff disagrees with verified memory set
    bad_ms = copy.deepcopy(fx["spec"])
    bad_ms["memory_set"] = {"cutoff": "2020-01-01T00:00:00Z"}
    with pytest.raises(UsefulnessError, match="cutoff"):
        freeze_registration(fx["exp_home"], fx["run_id"], bad_ms)

    # Missing task manifest for another run id
    with pytest.raises(UsefulnessError):
        freeze_registration(fx["exp_home"], "run-without-manifest", fx["spec"])


# ---------------------------------------------------------------------------
# 2. Registration is write-once and cannot be overwritten through the store
# ---------------------------------------------------------------------------


def test_registration_cannot_be_written_twice_or_changed_through_store(tmp_path: Path) -> None:
    fx = _setup_run_fixtures(tmp_path)
    first = freeze_registration(fx["exp_home"], fx["run_id"], fx["spec"])
    original_bytes = first.registration_path.read_bytes()

    with pytest.raises(UsefulnessError, match="already exists and cannot be overwritten"):
        freeze_registration(fx["exp_home"], fx["run_id"], fx["spec"])

    store = UsefulnessRunStore(fx["exp_home"], fx["run_id"])
    with pytest.raises(UsefulnessError, match="already exists and cannot be overwritten"):
        store.write_registration({"mutated": True})

    assert first.registration_path.read_bytes() == original_bytes
    assert store.registration_sha256() == first.registration_sha256


# ---------------------------------------------------------------------------
# 3. Held-out trials refused without registration; record carries sha256
# ---------------------------------------------------------------------------


def test_held_out_trial_refused_without_registration_while_calibration_runs(tmp_path: Path) -> None:
    fx = _setup_run_fixtures(tmp_path)
    store = UsefulnessRunStore(fx["exp_home"], fx["run_id"])

    # Calibration task runs before freeze, with registration_sha256=None.
    cal_rec = run_trial(
        fx["exp_home"],
        fx["run_id"],
        fx["cal_dir"],
        _trial_config(fx, arm="no_memory"),
        adapter=_ok_adapter(fx),
    )
    assert cal_rec.status == "completed"
    assert cal_rec.registration_sha256 is None
    assert (
        json.loads(cal_rec.record_path.read_text(encoding="utf-8"))["registration_sha256"] is None
    )

    # Held-out task is refused before freeze (even if verify_manifest=False).
    with pytest.raises(UsefulnessError, match="no registration"):
        run_trial(
            fx["exp_home"],
            fx["run_id"],
            fx["held_dir"],
            _trial_config(fx, arm="no_memory", verify_manifest=True),
            adapter=_ok_adapter(fx),
        )
    with pytest.raises(UsefulnessError, match="no registration"):
        run_trial(
            fx["exp_home"],
            fx["run_id"],
            fx["held_dir"],
            _trial_config(fx, arm="no_memory", verify_manifest=False),
            adapter=_ok_adapter(fx),
        )
    # Only the calibration trial directory exists.
    assert len(list(store.trials_dir.iterdir())) == 1


def test_held_out_trial_runs_after_freeze_and_records_registration_sha256(tmp_path: Path) -> None:
    fx = _setup_run_fixtures(tmp_path)
    frozen = freeze_registration(fx["exp_home"], fx["run_id"], fx["spec"])

    # No-memory arm
    rec_no_mem = run_trial(
        fx["exp_home"],
        fx["run_id"],
        fx["held_dir"],
        _trial_config(fx, arm="no_memory"),
        adapter=_ok_adapter(fx),
    )
    assert rec_no_mem.status == "completed"
    assert rec_no_mem.registration_sha256 == frozen.registration_sha256
    on_disk = json.loads(rec_no_mem.record_path.read_text(encoding="utf-8"))
    assert on_disk["registration_sha256"] == frozen.registration_sha256

    # Memory arm defaults source_home to the registered memory set and writes wording file.
    seen_wording_path: list[Path | None] = []

    def _capture(inv: HarnessInvocation) -> HarnessResult:
        assert inv.memory_endpoint is not None
        seen_wording_path.append(inv.memory_endpoint.wording_path)
        assert inv.memory_endpoint.wording_sha256 == frozen.registration["memory_tools"]["sha256"]
        return HarnessResult(
            status="completed",
            last_reply="done with memory",
            exit_code=0,
            harness_version=fx["harness_version"],
            resolved_model=inv.config.model,
            resolved_reasoning_effort=inv.config.reasoning_effort,
            num_turns=1,
        )

    rec_mem = run_trial(
        fx["exp_home"],
        fx["run_id"],
        fx["held_dir"],
        _trial_config(fx, arm="memory", source_home=None),
        adapter=ScriptedHarnessAdapter(_capture, version=fx["harness_version"]),
    )
    assert rec_mem.status == "completed"
    assert rec_mem.registration_sha256 == frozen.registration_sha256
    assert (
        rec_mem.environment["memory_tools_wording_sha256"]
        == frozen.registration["memory_tools"]["sha256"]
    )
    assert seen_wording_path[0] is not None
    assert seen_wording_path[0].is_file()
    assert rec_mem.paths["trial_home"] is not None
    assert (Path(rec_mem.paths["trial_home"]) / "blobs" / "note.txt").is_file()


# ---------------------------------------------------------------------------
# 4. Trial refused when memory set, task manifest, environment, or wording drift
# ---------------------------------------------------------------------------


def test_trial_refused_when_memory_set_drifts_from_registration(tmp_path: Path) -> None:
    fx = _setup_run_fixtures(tmp_path)
    freeze_registration(fx["exp_home"], fx["run_id"], fx["spec"])
    store = UsefulnessRunStore(fx["exp_home"], fx["run_id"])

    # Tamper with a blob inside the frozen memory set.
    (store.memory_set_dir / "blobs" / "note.txt").write_text("tampered\n", encoding="utf-8")

    with pytest.raises(UsefulnessError):
        run_trial(
            fx["exp_home"],
            fx["run_id"],
            fx["held_dir"],
            _trial_config(fx, arm="memory"),
            adapter=_ok_adapter(fx),
        )
    assert not store.trials_dir.exists() or not list(store.trials_dir.iterdir())


def test_trial_refused_when_different_source_home_passed_in_memory_arm(tmp_path: Path) -> None:
    fx = _setup_run_fixtures(tmp_path)
    freeze_registration(fx["exp_home"], fx["run_id"], fx["spec"])

    other_home = tmp_path / "other-ledger"
    initialize_home(Config(home=other_home))

    with pytest.raises(UsefulnessError, match="does not match the registered memory set"):
        run_trial(
            fx["exp_home"],
            fx["run_id"],
            fx["held_dir"],
            _trial_config(fx, arm="memory", source_home=other_home),
            adapter=_ok_adapter(fx),
        )


def test_trial_refused_when_task_manifest_or_task_file_drifts(tmp_path: Path) -> None:
    fx = _setup_run_fixtures(tmp_path)
    freeze_registration(fx["exp_home"], fx["run_id"], fx["spec"])
    store = UsefulnessRunStore(fx["exp_home"], fx["run_id"])

    # Case A: task file altered on disk (caught even if verify_manifest=False is passed).
    (fx["held_dir"] / "request.md").write_text("Altered task request.\n", encoding="utf-8")
    with pytest.raises(UsefulnessError):
        run_trial(
            fx["exp_home"],
            fx["run_id"],
            fx["held_dir"],
            _trial_config(fx, arm="no_memory", verify_manifest=False),
            adapter=_ok_adapter(fx),
        )
    # Restore task file, then tamper with task-manifest.json itself.
    (fx["held_dir"] / "request.md").write_text("Solve task held-01.\n", encoding="utf-8")
    tm_text = store.task_manifest_path.read_text(encoding="utf-8")
    store.task_manifest_path.write_text(tm_text + "\n", encoding="utf-8")

    with pytest.raises(UsefulnessError, match="task manifest no longer matches registration"):
        run_trial(
            fx["exp_home"],
            fx["run_id"],
            fx["held_dir"],
            _trial_config(fx, arm="no_memory"),
            adapter=_ok_adapter(fx),
        )
    assert not store.trials_dir.exists() or not list(store.trials_dir.iterdir())


def test_trial_refused_when_environment_or_model_or_limit_drifts(tmp_path: Path) -> None:
    fx = _setup_run_fixtures(tmp_path)
    freeze_registration(fx["exp_home"], fx["run_id"], fx["spec"])

    # Wrong harness name
    with pytest.raises(UsefulnessError, match="harness"):
        run_trial(
            fx["exp_home"],
            fx["run_id"],
            fx["held_dir"],
            _trial_config(fx),
            adapter=_ok_adapter(fx, name="other-harness"),
        )

    # Wrong harness version
    with pytest.raises(UsefulnessError, match="harness_version"):
        run_trial(
            fx["exp_home"],
            fx["run_id"],
            fx["held_dir"],
            _trial_config(fx, harness_version="scripted-2.0.0"),
            adapter=_ok_adapter(fx, version="scripted-2.0.0"),
        )

    # Wrong repo commit
    with pytest.raises(UsefulnessError, match="repo_commit"):
        run_trial(
            fx["exp_home"],
            fx["run_id"],
            fx["held_dir"],
            _trial_config(fx, repo_commit="f" * 40),
            adapter=_ok_adapter(fx),
        )

    # Wrong hardware class
    with pytest.raises(UsefulnessError, match="hardware_class"):
        run_trial(
            fx["exp_home"],
            fx["run_id"],
            fx["held_dir"],
            _trial_config(fx, hardware_class="linux-x86_64"),
            adapter=_ok_adapter(fx),
        )

    # Wrong allowed base tools
    with pytest.raises(UsefulnessError, match="allowed_base_tools"):
        run_trial(
            fx["exp_home"],
            fx["run_id"],
            fx["held_dir"],
            _trial_config(fx, base_tools=("Read", "LS")),
            adapter=_ok_adapter(fx),
        )

    # Wrong memory_server_name
    with pytest.raises(UsefulnessError, match="memory_server_name"):
        run_trial(
            fx["exp_home"],
            fx["run_id"],
            fx["held_dir"],
            _trial_config(fx, memory_server_name="other-memory"),
            adapter=_ok_adapter(fx),
        )

    # Wrong auto_level (config_overrides mismatch)
    with pytest.raises(UsefulnessError, match="config_overrides"):
        run_trial(
            fx["exp_home"],
            fx["run_id"],
            fx["held_dir"],
            _trial_config(fx, auto_level="medium"),
            adapter=_ok_adapter(fx),
        )

    # Unregistered model
    with pytest.raises(UsefulnessError, match="model"):
        run_trial(
            fx["exp_home"],
            fx["run_id"],
            fx["held_dir"],
            _trial_config(fx, model="other-model"),
            adapter=_ok_adapter(fx),
        )

    # Mismatched reasoning_effort
    with pytest.raises(UsefulnessError, match="reasoning_effort"):
        run_trial(
            fx["exp_home"],
            fx["run_id"],
            fx["held_dir"],
            _trial_config(fx, reasoning_effort="low"),
            adapter=_ok_adapter(fx),
        )

    # Mismatched wall_time_limit_s
    with pytest.raises(UsefulnessError, match="wall_time_limit_s"):
        run_trial(
            fx["exp_home"],
            fx["run_id"],
            fx["held_dir"],
            _trial_config(fx, wall_time_limit_s=120.0),
            adapter=_ok_adapter(fx),
        )


def test_trial_refused_when_memory_tool_wording_drifts(tmp_path: Path) -> None:
    fx = _setup_run_fixtures(tmp_path)
    freeze_registration(fx["exp_home"], fx["run_id"], fx["spec"])

    # In-memory wording override with changed description
    drifted_descs = dict(CUSTOM_DESCRIPTIONS)
    drifted_descs["recall"] = "A changed recall description."
    with pytest.raises(UsefulnessError, match="wording no longer matches registration"):
        run_trial(
            fx["exp_home"],
            fx["run_id"],
            fx["held_dir"],
            _trial_config(
                fx,
                arm="memory",
                memory_tool_wording={
                    "instructions": CUSTOM_INSTRUCTIONS,
                    "descriptions": drifted_descs,
                },
            ),
            adapter=_ok_adapter(fx),
        )

    # Wording file on disk with installed defaults (differs from registered custom wording)
    wording_file = tmp_path / "installed-wording.json"
    wording_file.write_text(json.dumps(default_tool_wording()), encoding="utf-8")
    with pytest.raises(UsefulnessError, match=r"wording.*no longer matches registration"):
        run_trial(
            fx["exp_home"],
            fx["run_id"],
            fx["held_dir"],
            _trial_config(fx, arm="memory", memory_tool_wording_file=wording_file),
            adapter=_ok_adapter(fx),
        )

    # Matching wording file succeeds
    matching_file = tmp_path / "registered-wording.json"
    matching_file.write_text(
        json.dumps({"instructions": CUSTOM_INSTRUCTIONS, "descriptions": CUSTOM_DESCRIPTIONS}),
        encoding="utf-8",
    )
    rec = run_trial(
        fx["exp_home"],
        fx["run_id"],
        fx["held_dir"],
        _trial_config(fx, arm="memory", memory_tool_wording_file=matching_file),
        adapter=_ok_adapter(fx),
    )
    assert rec.status == "completed"


# ---------------------------------------------------------------------------
# 5. Trial memory tools serve the registered wording (not installed defaults)
# ---------------------------------------------------------------------------


def _write_fake_droid(tmp_path: Path, version: str = "1.21.0") -> Path:
    """Create a local fake droid script that records mcp.json and exits cleanly."""
    script = tmp_path / "fake-droid"
    script.write_text(
        f"""#!{sys.executable}
import json, os, sys
from pathlib import Path

if "--version" in sys.argv:
    print({version!r})
    sys.exit(0)

# Emit minimal stream-json init + completion events
print(json.dumps({{
    "type": "system",
    "subtype": "init",
    "session_id": "fake-sess-1",
    "model": "local-test-model",
    "reasoning_effort": "high",
}}))
print(json.dumps({{
    "type": "completion",
    "finalText": "fake droid finished",
    "numTurns": 1,
    "durationMs": 12,
    "session_id": "fake-sess-1",
    "usage": {{"input_tokens": 10, "output_tokens": 5}},
}}))
""",
        encoding="utf-8",
    )
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    return script


def test_trial_memory_tools_serve_registered_custom_wording_over_stdio(tmp_path: Path) -> None:
    from mcp.client.session import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client

    assert CUSTOM_INSTRUCTIONS != INSTRUCTIONS
    assert CUSTOM_DESCRIPTIONS != DEFAULT_TOOL_DESCRIPTIONS

    fx = _setup_run_fixtures(tmp_path)
    fake_droid = _write_fake_droid(tmp_path, version="1.21.0")

    spec = _valid_registration_spec(
        harness="droid",
        harness_version="1.21.0",
        repo_commit=fx["repo_commit"],
        hardware_class=fx["hardware_class"],
        auto_level="high",
        instructions=CUSTOM_INSTRUCTIONS,
        descriptions=CUSTOM_DESCRIPTIONS,
    )
    frozen = freeze_registration(fx["exp_home"], fx["run_id"], spec)

    adapter = DroidHarnessAdapter(
        droid_bin=str(fake_droid),
        mcp_command=(sys.executable, "-m", "agent_memory.mcp_server"),
    )
    cfg = _trial_config(
        fx,
        arm="memory",
        harness_version="1.21.0",
    )
    rec = run_trial(fx["exp_home"], fx["run_id"], fx["held_dir"], cfg, adapter=adapter)
    assert rec.status == "completed"
    assert rec.registration_sha256 == frozen.registration_sha256

    mcp_config_path = Path(rec.paths["mcp_config_path"])  # type: ignore[arg-type]
    mcp_cfg = json.loads(mcp_config_path.read_text(encoding="utf-8"))
    server_cfg = mcp_cfg["mcpServers"]["agent-memory"]
    wording_env_path = Path(server_cfg["env"][MCP_WORDING_ENV])
    assert wording_env_path.is_file()
    assert wording_env_path == Path(rec.paths["memory_tool_wording_path"])  # type: ignore[arg-type]

    # Start a live MemoryServer on the trial home and spawn mem-mcp with the trial's mcp.json env.
    trial_home = Path(rec.paths["trial_home"])  # type: ignore[arg-type]
    live_port = free_port()
    trial_cfg = Config(home=trial_home, port=live_port)
    live_server = MemoryServer(trial_cfg)
    live_server.start()
    try:
        stdio_env = {
            **os.environ,
            **server_cfg["env"],
            "AGENT_MEMORY_PORT": str(live_port),
        }
        params = StdioServerParameters(
            command=server_cfg["command"],
            args=server_cfg["args"],
            env=stdio_env,
        )

        async def inspect_mcp() -> None:
            async with (
                stdio_client(params) as (read, write),
                ClientSession(read, write) as session,
            ):
                init_result = await session.initialize()
                assert init_result.instructions == CUSTOM_INSTRUCTIONS
                assert init_result.instructions != INSTRUCTIONS

                tools_list = (await session.list_tools()).tools
                by_name = {t.name: t.description for t in tools_list}
                assert set(by_name.keys()) == set(TOOL_NAMES)
                for name in TOOL_NAMES:
                    assert by_name[name] == CUSTOM_DESCRIPTIONS[name]
                    assert by_name[name] != DEFAULT_TOOL_DESCRIPTIONS[name]

                status = await session.call_tool("status", {})
                assert status.is_error is False
                assert status.structured_content is not None
                assert status.structured_content["reachable"] is True

        asyncio.run(inspect_mcp())
    finally:
        live_server.stop()


def test_mem_mcp_refuses_invalid_wording_file(tmp_path: Path) -> None:
    home = tmp_path / "home"
    initialize_home(Config(home=home))
    bad_wording = tmp_path / "bad-wording.json"
    bad_wording.write_text(json.dumps({"instructions": "only instructions"}), encoding="utf-8")

    env = {
        **os.environ,
        "AGENT_MEMORY_HOME": str(home),
        "AGENT_MEMORY_AGENT": "factory-droid",
        MCP_WORDING_ENV: str(bad_wording),
    }
    proc = subprocess.run(
        [sys.executable, "-m", "agent_memory.mcp_server"],
        env=env,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 1
    assert "descriptions" in proc.stderr


# ---------------------------------------------------------------------------
# 6. Deviations are appended with date and reason; earlier ones never rewritten
# ---------------------------------------------------------------------------


def test_deviations_append_only_with_date_and_reason(tmp_path: Path) -> None:
    fx = _setup_run_fixtures(tmp_path)
    store = UsefulnessRunStore(fx["exp_home"], fx["run_id"])

    assert list_deviations(fx["exp_home"], fx["run_id"]) == []

    first = append_deviation(
        fx["exp_home"],
        fx["run_id"],
        "Provider maintenance delayed batch 2 by 4 hours",
        date="2026-11-26T12:00:00Z",
    )
    assert first["recorded_at"] == "2026-11-26T12:00:00Z"
    assert first["reason"] == "Provider maintenance delayed batch 2 by 4 hours"
    lines_after_first = store.deviations_path.read_text(encoding="utf-8").splitlines()
    assert len(lines_after_first) == 1

    second = append_deviation(
        fx["exp_home"],
        fx["run_id"],
        "Retried pair 3 after transient rate limit",
        metadata={"pair_index": 3},
    )
    assert second["recorded_at"]
    assert second["reason"] == "Retried pair 3 after transient rate limit"
    assert second["metadata"] == {"pair_index": 3}

    lines_after_second = store.deviations_path.read_text(encoding="utf-8").splitlines()
    assert len(lines_after_second) == 2
    assert lines_after_second[0] == lines_after_first[0]

    rows = list_deviations(fx["exp_home"], fx["run_id"])
    assert len(rows) == 2
    assert rows[0]["reason"] == first["reason"]
    assert rows[1]["reason"] == second["reason"]

    with pytest.raises(UsefulnessError, match="non-empty string"):
        append_deviation(fx["exp_home"], fx["run_id"], "   ")
    assert store.deviations_path.read_text(encoding="utf-8").splitlines() == lines_after_second


# ---------------------------------------------------------------------------
# 7. CLI end-to-end: freeze, verify-registration, deviate, deviations
# ---------------------------------------------------------------------------


def test_cli_freeze_verify_and_deviations(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    fx = _setup_run_fixtures(tmp_path)
    spec_path = tmp_path / "freeze-spec.json"
    spec_path.write_text(json.dumps(fx["spec"], indent=2), encoding="utf-8")

    # Freeze
    rc = cli_main(
        [
            "--home",
            str(fx["exp_home"]),
            "usefulness",
            "freeze",
            "--run-id",
            fx["run_id"],
            "--spec",
            str(spec_path),
            "--json",
        ]
    )
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    reg_sha = out["registration_sha256"]
    assert len(reg_sha) == 64

    # Second freeze fails
    rc_dup = cli_main(
        [
            "--home",
            str(fx["exp_home"]),
            "usefulness",
            "freeze",
            "--run-id",
            fx["run_id"],
            "--spec",
            str(spec_path),
        ]
    )
    assert rc_dup != 0
    capsys.readouterr()

    # Verify with matching wording file
    good_wording = tmp_path / "good-wording.json"
    good_wording.write_text(
        json.dumps({"instructions": CUSTOM_INSTRUCTIONS, "descriptions": CUSTOM_DESCRIPTIONS}),
        encoding="utf-8",
    )
    rc_verify = cli_main(
        [
            "--home",
            str(fx["exp_home"]),
            "usefulness",
            "verify-registration",
            "--run-id",
            fx["run_id"],
            "--wording-file",
            str(good_wording),
            "--json",
        ]
    )
    assert rc_verify == 0
    v_out = json.loads(capsys.readouterr().out)
    assert v_out["registration_sha256"] == reg_sha

    # Verify fails with mismatched wording file
    bad_wording = tmp_path / "bad-wording.json"
    bad_wording.write_text(json.dumps(default_tool_wording()), encoding="utf-8")
    rc_bad = cli_main(
        [
            "--home",
            str(fx["exp_home"]),
            "usefulness",
            "verify-registration",
            "--run-id",
            fx["run_id"],
            "--wording-file",
            str(bad_wording),
        ]
    )
    assert rc_bad != 0
    capsys.readouterr()

    # Deviate & list deviations
    rc_dev1 = cli_main(
        [
            "--home",
            str(fx["exp_home"]),
            "usefulness",
            "deviate",
            "--run-id",
            fx["run_id"],
            "--reason",
            "First logged deviation",
            "--date",
            "2026-11-26T15:00:00Z",
            "--json",
        ]
    )
    assert rc_dev1 == 0
    dev1 = json.loads(capsys.readouterr().out)
    assert dev1["recorded_at"] == "2026-11-26T15:00:00Z"

    rc_dev2 = cli_main(
        [
            "--home",
            str(fx["exp_home"]),
            "usefulness",
            "deviate",
            "--run-id",
            fx["run_id"],
            "--reason",
            "Second logged deviation",
        ]
    )
    assert rc_dev2 == 0
    capsys.readouterr()

    rc_list = cli_main(
        [
            "--home",
            str(fx["exp_home"]),
            "usefulness",
            "deviations",
            "--run-id",
            fx["run_id"],
            "--json",
        ]
    )
    assert rc_list == 0
    listed = json.loads(capsys.readouterr().out)
    assert [r["reason"] for r in listed] == [
        "First logged deviation",
        "Second logged deviation",
    ]


def test_verify_registration_artifacts_and_helper_unit_checks(tmp_path: Path) -> None:
    fx = _setup_run_fixtures(tmp_path)
    frozen = freeze_registration(fx["exp_home"], fx["run_id"], fx["spec"])

    # Artifact check passes
    checked = verify_registration_artifacts(fx["exp_home"], fx["run_id"])
    assert checked.registration_sha256 == frozen.registration_sha256

    # build_memory_tool_wording hash stability
    w1 = build_memory_tool_wording(CUSTOM_INSTRUCTIONS, CUSTOM_DESCRIPTIONS)
    w2 = build_memory_tool_wording(
        CUSTOM_INSTRUCTIONS,
        dict(reversed(list(CUSTOM_DESCRIPTIONS.items()))),
    )
    assert w1["sha256"] == w2["sha256"]

    # Direct verify_registration_for_trial passes for matching trial inputs
    verified = verify_registration_for_trial(
        fx["exp_home"],
        fx["run_id"],
        harness="scripted",
        harness_version=fx["harness_version"],
        repo_commit=fx["repo_commit"],
        hardware_class=fx["hardware_class"],
        allowed_base_tools=DEFAULT_BASE_TOOLS,
        memory_server_name="agent-memory",
        config_overrides=default_trial_config_overrides("high"),
        model="local-test-model",
        reasoning_effort="high",
        wall_time_limit_s=60.0,
        arm="memory",
    )
    assert verified.registration_sha256 == frozen.registration_sha256
    assert verified.memory_tools_instructions == CUSTOM_INSTRUCTIONS
    assert verified.memory_tools_descriptions == CUSTOM_DESCRIPTIONS
