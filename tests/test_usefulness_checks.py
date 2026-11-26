"""Tests for post-trial task check execution (MEM-24)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from agent_memory.cli import main as cli_main
from agent_memory.usefulness import (
    HarnessInvocation,
    HarnessResult,
    ScriptedHarnessAdapter,
    TrialConfig,
    UsefulnessError,
    UsefulnessRunStore,
    build_task_manifest,
    run_trial,
    run_trial_checks,
    sha256_file,
)


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


def _make_repo(tmp_path: Path) -> tuple[Path, str]:
    repo = tmp_path / "src-repo"
    repo.mkdir(parents=True)
    _git(repo, "init", "-b", "main")
    (repo / "app.py").write_text("def answer():\n    return 1\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "base")
    return repo, _git(repo, "rev-parse", "HEAD")


def _write_task(
    tasks_dir: Path,
    task_id: str,
    *,
    repo: Path,
    commit: str,
    commands: list[dict[str, object]],
    hidden_tests: dict[str, str] | None = None,
    work_done_at: str = "2026-11-20T12:00:00Z",
    task_family: str = "fam-1",
) -> Path:
    task_dir = tasks_dir / task_id
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "task.json").write_text(
        json.dumps(
            {
                "task_id": task_id,
                "repository": str(repo),
                "commit": commit,
                "project": "demo",
                "work_done_at": work_done_at,
                "task_family": task_family,
                "strata": ["ordinary"],
                "checks": {"commands": commands},
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (task_dir / "request.md").write_text("Make answer() return 42.\n", encoding="utf-8")
    (task_dir / "required_behavior.md").write_text("- answer() returns 42\n", encoding="utf-8")
    if hidden_tests:
        for rel, body in hidden_tests.items():
            target = task_dir / "hidden_tests" / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(body, encoding="utf-8")
    return task_dir


def test_checks_install_hidden_tests_post_run_same_command_both_arms_and_leave_record_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Checks run identically across arms, stage hidden tests post-run, and leave record intact."""
    monkeypatch.setenv("AGENT_MEMORY_HOME", "/should/be/stripped")
    monkeypatch.setenv("FACTORY_SESSION_ID", "should-be-stripped")

    repo, commit = _make_repo(tmp_path)
    exp_home = tmp_path / "exp"
    hidden_code = (
        "import os, sys\n"
        "sys.path.insert(0, os.getcwd())\n"
        "from app import answer\n"
        "assert 'AGENT_MEMORY_HOME' not in os.environ\n"
        "assert 'FACTORY_SESSION_ID' not in os.environ\n"
        "assert answer() == 42, f'got {answer()}'\n"
        "print('HIDDEN_OK')\n"
    )
    t_cal = _write_task(
        exp_home / "tasks",
        "t-check",
        repo=repo,
        commit=commit,
        commands=[
            {"name": "hidden-check", "command": [sys.executable, "tests/hidden_verify.py"]},
        ],
        hidden_tests={"tests/hidden_verify.py": hidden_code},
        work_done_at="2026-11-18T12:00:00Z",
        task_family="fam-a",
    )
    _write_task(
        exp_home / "tasks",
        "t-later",
        repo=repo,
        commit=commit,
        commands=[{"name": "ok", "command": [sys.executable, "-c", "print('ok')"]}],
        work_done_at="2026-11-22T12:00:00Z",
        task_family="fam-b",
    )

    store = UsefulnessRunStore(exp_home, "run-checks-1")
    manifest = build_task_manifest(exp_home / "tasks", calibration_fraction=0.5)
    store.write_task_manifest(manifest)

    seen_hidden_during_agent: dict[str, bool] = {}

    def agent(inv: HarnessInvocation) -> HarnessResult:
        # Hidden test must NOT exist in the clone while the agent works!
        seen_hidden_during_agent[inv.arm] = (inv.clone_dir / "tests" / "hidden_verify.py").exists()
        if inv.arm == "memory":
            (inv.clone_dir / "app.py").write_text(
                "def answer():\n    return 42\n", encoding="utf-8"
            )
        else:
            # Leave app.py returning 1 so the no_memory arm fails the hidden check.
            (inv.clone_dir / "app.py").write_text("def answer():\n    return 7\n", encoding="utf-8")
        return HarnessResult(status="completed", last_reply="Done.")

    adapter = ScriptedHarnessAdapter(agent)
    rec_mem = run_trial(
        exp_home,
        "run-checks-1",
        t_cal,
        TrialConfig(
            arm="memory",
            model="local-model",
            model_base_url="http://127.0.0.1:8080/v1",
            trial_id="trial-mem",
        ),
        adapter=adapter,
    )
    rec_nomem = run_trial(
        exp_home,
        "run-checks-1",
        t_cal,
        TrialConfig(
            arm="no_memory",
            model="local-model",
            model_base_url="http://127.0.0.1:8080/v1",
            trial_id="trial-nomem",
        ),
        adapter=adapter,
    )

    assert seen_hidden_during_agent == {"memory": False, "no_memory": False}

    # Snapshot exact bytes and SHA-256 of both record.json files before checks run.
    mem_rec_bytes_before = rec_mem.record_path.read_bytes()
    nomem_rec_bytes_before = rec_nomem.record_path.read_bytes()
    mem_rec_sha = sha256_file(rec_mem.record_path)
    nomem_rec_sha = sha256_file(rec_nomem.record_path)

    res_mem = run_trial_checks(exp_home, "run-checks-1", "trial-mem")
    res_nomem = run_trial_checks(exp_home, "run-checks-1", "trial-nomem")

    # Same command ran in both arms; memory passed, no_memory failed.
    assert res_mem.checks[0].command == res_nomem.checks[0].command
    assert res_mem.overall_status == "passed"
    assert res_mem.passed_count == 1
    assert res_mem.failed_count == 0
    assert res_mem.could_not_run_count == 0
    assert res_mem.checks[0].status == "passed"
    assert res_mem.checks[0].exit_code == 0
    assert "HIDDEN_OK" in res_mem.checks[0].stdout
    assert res_mem.hidden_tests_installed == ("tests/hidden_verify.py",)
    assert res_mem.record_sha256 == mem_rec_sha

    assert res_nomem.overall_status == "failed"
    assert res_nomem.passed_count == 0
    assert res_nomem.failed_count == 1
    assert res_nomem.could_not_run_count == 0
    assert res_nomem.checks[0].status == "failed"
    assert res_nomem.checks[0].exit_code == 1
    assert "got 7" in res_nomem.checks[0].stderr
    assert res_nomem.hidden_tests_installed == ("tests/hidden_verify.py",)
    assert res_nomem.record_sha256 == nomem_rec_sha

    # Hidden tests now exist in the clones (only after checks ran).
    assert (Path(rec_mem.paths["clone_dir"] or "") / "tests" / "hidden_verify.py").is_file()
    assert (Path(rec_nomem.paths["clone_dir"] or "") / "tests" / "hidden_verify.py").is_file()

    # Neither record.json changed by even a single byte.
    assert rec_mem.record_path.read_bytes() == mem_rec_bytes_before
    assert rec_nomem.record_path.read_bytes() == nomem_rec_bytes_before

    # checks.json omits `arm` (blind-safe) and is write-once (cannot be overwritten).
    checks_json = json.loads(res_mem.checks_path.read_text(encoding="utf-8"))
    assert "arm" not in checks_json
    assert checks_json == res_mem.to_dict()
    with pytest.raises(UsefulnessError, match="already recorded"):
        run_trial_checks(exp_home, "run-checks-1", "trial-mem")


def test_checks_distinguish_could_not_run_from_failed(tmp_path: Path) -> None:
    """Missing binary, timeout, and signal termination are could_not_run, not failed."""
    repo, commit = _make_repo(tmp_path)
    exp_home = tmp_path / "exp"
    task_dir = _write_task(
        exp_home / "tasks",
        "t-outcomes",
        repo=repo,
        commit=commit,
        commands=[
            {"name": "pass-cmd", "command": [sys.executable, "-c", "print('ok')"]},
            {"name": "fail-cmd", "command": [sys.executable, "-c", "import sys; sys.exit(2)"]},
            {"name": "missing-bin", "command": ["/nonexistent/binary/path/xyz-check"]},
            {
                "name": "timeout-cmd",
                "command": [
                    sys.executable,
                    "-c",
                    "import time; print('started', flush=True); time.sleep(30)",
                ],
                "timeout_s": 0.4,
            },
            {
                "name": "signal-cmd",
                "command": [
                    sys.executable,
                    "-c",
                    "import os, signal; os.kill(os.getpid(), signal.SIGKILL)",
                ],
            },
        ],
    )

    run_trial(
        exp_home,
        "run-outcomes",
        task_dir,
        TrialConfig(
            arm="no_memory",
            model="local-model",
            model_base_url="http://127.0.0.1:8080/v1",
            trial_id="trial-out",
        ),
        adapter=ScriptedHarnessAdapter(
            lambda inv: HarnessResult(status="completed", last_reply="Done.")
        ),
    )

    result = run_trial_checks(exp_home, "run-outcomes", "trial-out", kill_grace_s=0.5)
    assert result.overall_status == "could_not_run"
    assert result.passed_count == 1
    assert result.failed_count == 1
    assert result.could_not_run_count == 3

    by_name = {c.name: c for c in result.checks}
    assert by_name["pass-cmd"].status == "passed"
    assert by_name["pass-cmd"].exit_code == 0
    assert by_name["pass-cmd"].reason is None

    assert by_name["fail-cmd"].status == "failed"
    assert by_name["fail-cmd"].exit_code == 2
    assert by_name["fail-cmd"].reason is None

    assert by_name["missing-bin"].status == "could_not_run"
    assert by_name["missing-bin"].exit_code is None
    assert by_name["missing-bin"].reason == "command_not_found"

    assert by_name["timeout-cmd"].status == "could_not_run"
    assert by_name["timeout-cmd"].exit_code is None
    assert by_name["timeout-cmd"].reason == "timeout"
    assert "started" in by_name["timeout-cmd"].stdout

    assert by_name["signal-cmd"].status == "could_not_run"
    assert by_name["signal-cmd"].exit_code is not None
    assert by_name["signal-cmd"].exit_code < 0
    assert by_name["signal-cmd"].reason == "terminated_by_signal"


def test_checks_handle_clone_hidden_test_conflict_and_refuse_unfinished_or_tampered_task(
    tmp_path: Path,
) -> None:
    """Clone conflict marks could_not_run; unfinished trials and tampered tasks are refused."""
    repo, commit = _make_repo(tmp_path)
    exp_home = tmp_path / "exp"
    task_dir = _write_task(
        exp_home / "tasks",
        "t-conflict",
        repo=repo,
        commit=commit,
        commands=[{"name": "c1", "command": [sys.executable, "-c", "print('ok')"]}],
        hidden_tests={"tests/hidden.py": "assert True\n"},
    )

    # Unfinished trial (no record.json) is refused.
    with pytest.raises(UsefulnessError, match=r"no finished record\.json"):
        run_trial_checks(exp_home, "run-c", "trial-missing")

    # Agent creates a file at the hidden-test path -> staging conflict recorded as could_not_run.
    def conflicting_agent(inv: HarnessInvocation) -> HarnessResult:
        p = inv.clone_dir / "tests" / "hidden.py"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("# agent wrote conflicting file\n", encoding="utf-8")
        return HarnessResult(status="completed", last_reply="Wrote tests/hidden.py.")

    rec = run_trial(
        exp_home,
        "run-c",
        task_dir,
        TrialConfig(
            arm="no_memory",
            model="local-model",
            model_base_url="http://127.0.0.1:8080/v1",
            trial_id="trial-conflict",
        ),
        adapter=ScriptedHarnessAdapter(conflicting_agent),
    )
    res = run_trial_checks(exp_home, "run-c", "trial-conflict")
    assert res.overall_status == "could_not_run"
    assert res.could_not_run_count == 1
    assert res.checks[0].reason == "hidden_test_conflict"
    assert res.hidden_tests_error is not None
    # Agent's conflicting file was not overwritten.
    assert (Path(rec.paths["clone_dir"] or "") / "tests" / "hidden.py").read_text(
        encoding="utf-8"
    ) == "# agent wrote conflicting file\n"

    # Tampering with the task on disk after a trial ran is refused.
    run_trial(
        exp_home,
        "run-c",
        task_dir,
        TrialConfig(
            arm="no_memory",
            model="local-model",
            model_base_url="http://127.0.0.1:8080/v1",
            trial_id="trial-tamper",
        ),
        adapter=ScriptedHarnessAdapter(
            lambda inv: HarnessResult(status="completed", last_reply="Done.")
        ),
    )
    (task_dir / "hidden_tests" / "tests" / "hidden.py").write_text(
        "assert False\n", encoding="utf-8"
    )
    with pytest.raises(UsefulnessError, match="changed since trial ran"):
        run_trial_checks(exp_home, "run-c", "trial-tamper")


def test_cli_run_checks_command_end_to_end(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`mem --home <exp> usefulness run-checks` runs checks and prints JSON cleanly."""
    repo, commit = _make_repo(tmp_path)
    exp_home = tmp_path / "exp"
    task_dir = _write_task(
        exp_home / "tasks",
        "t-cli-check",
        repo=repo,
        commit=commit,
        commands=[{"name": "unit", "command": [sys.executable, "-c", "print('cli-pass')"]}],
    )
    run_trial(
        exp_home,
        "run-cli",
        task_dir,
        TrialConfig(
            arm="no_memory",
            model="local-model",
            model_base_url="http://127.0.0.1:8080/v1",
            trial_id="trial-cli-1",
        ),
        adapter=ScriptedHarnessAdapter(
            lambda inv: HarnessResult(status="completed", last_reply="Done.")
        ),
    )

    rc = cli_main(
        [
            "--home",
            str(exp_home),
            "usefulness",
            "run-checks",
            "--run-id",
            "run-cli",
            "--trial-id",
            "trial-cli-1",
            "--json",
        ]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["overall_status"] == "passed"
    assert payload["passed_count"] == 1
    assert payload["checks"][0]["name"] == "unit"
    assert "cli-pass" in payload["checks"][0]["stdout"]
