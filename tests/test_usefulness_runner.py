"""Tests for the usefulness single-trial runner, Droid adapter, isolation, and CLI (MEM-18)."""

from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from agent_memory.cli import main as cli_main
from agent_memory.config import DEFAULT_HOME, Config
from agent_memory.db import MemoryClient, initialize_home, open_direct, sql_literal
from agent_memory.endpoints import REMOTE_ENV, Unreachable
from agent_memory.usefulness import (
    DEFAULT_BASE_TOOLS,
    TRIAL_RECORD_SCHEMA_VERSION,
    DroidHarnessAdapter,
    HarnessInvocation,
    HarnessResult,
    ScriptedHarnessAdapter,
    TokenUsage,
    ToolCallRecord,
    TrialConfig,
    UsefulnessError,
    UsefulnessRunStore,
    build_task_manifest,
    check_trial_egress,
    parse_droid_stream_events,
    run_trial,
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


def _make_git_repo(
    tmp_path: Path,
    *,
    with_repo_mcp: bool = True,
    with_hidden_conflict: bool = False,
) -> tuple[Path, str, str]:
    """Create a local git repository with a base commit and a later 'future' commit."""
    repo = tmp_path / "source-repo"
    repo.mkdir(parents=True)
    _git(repo, "init", "-b", "main")
    (repo / "app.py").write_text("def answer():\n    return 1\n", encoding="utf-8")
    if with_repo_mcp:
        (repo / ".factory").mkdir()
        (repo / ".factory" / "mcp.json").write_text(
            json.dumps(
                {
                    "mcpServers": {
                        "agent-memory": {
                            "type": "stdio",
                            "command": "mem-mcp",
                            "env": {"AGENT_MEMORY_HOME": "/should/never/be/used"},
                        },
                        "lint-server": {
                            "type": "stdio",
                            "command": "lint-mcp",
                        },
                    }
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    if with_hidden_conflict:
        (repo / "tests").mkdir(exist_ok=True)
        (repo / "tests" / "test_hidden.py").write_text("assert True\n", encoding="utf-8")

    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "base commit")
    base_sha = _git(repo, "rev-parse", "HEAD")

    # Add a later commit (the real fix) and a future tag, plus an untracked build artifact.
    (repo / "app.py").write_text("def answer():\n    return 42\n", encoding="utf-8")
    _git(repo, "add", "app.py")
    _git(repo, "commit", "-m", "future fix commit")
    future_sha = _git(repo, "rev-parse", "HEAD")
    _git(repo, "tag", "v-future")

    build_dir = repo / "build-cache"
    build_dir.mkdir()
    (build_dir / "stale.artifact").write_text("stale\n", encoding="utf-8")

    return repo, base_sha, future_sha


def _write_task(
    tasks_dir: Path,
    task_id: str,
    *,
    repo: Path,
    commit: str,
    project: str = "demo-proj",
    work_done_at: str = "2026-11-20T12:00:00Z",
    task_family: str = "fam-1",
    request: str = "Make answer() return 42.",
    hidden_tests: dict[str, str] | None = None,
) -> Path:
    task_dir = tasks_dir / task_id
    task_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "task_id": task_id,
        "repository": str(repo),
        "commit": commit,
        "project": project,
        "work_done_at": work_done_at,
        "task_family": task_family,
        "strata": ["ordinary"],
        "checks": {
            "commands": [{"name": "check", "command": [sys.executable, "-c", "print('ok')"]}]
        },
    }
    (task_dir / "task.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    (task_dir / "request.md").write_text(request, encoding="utf-8")
    (task_dir / "required_behavior.md").write_text("- answer() returns 42\n", encoding="utf-8")
    if hidden_tests:
        for rel, body in hidden_tests.items():
            dest = task_dir / "hidden_tests" / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(body, encoding="utf-8")
    return task_dir


def _make_seeded_source_home(path: Path, project: str) -> Path:
    """Create a stopped throwaway home with one active memory scoped to ``project``."""
    cfg = Config(home=path)
    initialize_home(cfg)
    con = open_direct(cfg.db_path)
    try:
        now = datetime(2026, 11, 15, 12, 0, tzinfo=UTC)
        seed_sess_id = str(uuid.uuid4())
        seed_mem_id = str(uuid.uuid4())
        con.execute(
            """
            INSERT INTO sessions (
                session_id, agent_id, capture_completeness, project, scope, started_at
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [seed_sess_id, "factory-droid", "full", project, f"project:{project}", now],
        )
        con.execute(
            """
            INSERT INTO memories (
                memory_id, scope, type, content, content_hash, trust_level, created_by_session
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                seed_mem_id,
                f"project:{project}",
                "fact",
                "answer() must return 42",
                "hash-seed-1",
                "user",
                seed_sess_id,
            ],
        )
        con.execute("CHECKPOINT")
    finally:
        con.close()
    return path


def _write_fake_droid(path: Path, script_body: str) -> Path:
    path.write_text(f"#!{sys.executable}\n{script_body}", encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return path


# ---------------------------------------------------------------------------
# Token usage parsing: missing counts must be None, never 0
# ---------------------------------------------------------------------------


def test_token_usage_records_missing_fields_as_none_not_zero() -> None:
    partial = TokenUsage.from_harness_usage({"input_tokens": 120})
    assert partial.input_tokens == 120
    assert partial.output_tokens is None
    assert partial.cache_read_input_tokens is None
    assert partial.cache_creation_input_tokens is None
    assert partial.thinking_tokens is None
    assert partial.total_tokens is None
    assert partial.ttft_ms is None

    # Pre-call failure (auth error before any model call): Droid emits 0s with no ttft_ms.
    pre_call = TokenUsage.from_harness_usage(
        {"input_tokens": 0, "output_tokens": 0, "factory_credits": 0},
        pre_call_failure=True,
    )
    assert pre_call == TokenUsage()
    assert pre_call.input_tokens is None
    assert pre_call.output_tokens is None
    assert pre_call.total_tokens is None

    # Full usage sums non-None components while keeping missing sub-fields None.
    complete = TokenUsage.from_harness_usage(
        {
            "input_tokens": 100,
            "cache_read_input_tokens": 50,
            "output_tokens": 25,
            "ttft_ms": 312.5,
        }
    )
    assert complete.input_tokens == 100
    assert complete.output_tokens == 25
    assert complete.cache_read_input_tokens == 50
    assert complete.cache_creation_input_tokens is None
    assert complete.thinking_tokens is None
    assert complete.total_tokens == 175
    assert complete.ttft_ms == 312.5


# ---------------------------------------------------------------------------
# Egress and default-home guards
# ---------------------------------------------------------------------------


def test_guards_refuse_default_home_and_hosted_egress_without_opt_in(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(REMOTE_ENV, raising=False)
    repo, base_sha, _ = _make_git_repo(tmp_path)
    exp_home = tmp_path / "exp"
    task_dir = _write_task(exp_home / "tasks", "t-guard", repo=repo, commit=base_sha)

    adapter = ScriptedHarnessAdapter(lambda inv: HarnessResult(status="completed", last_reply="ok"))

    # Default home refused for experiment_home.
    with pytest.raises(UsefulnessError, match="default home"):
        run_trial(
            DEFAULT_HOME,
            "run-1",
            task_dir,
            TrialConfig(arm="no_memory", model_base_url="http://127.0.0.1:11434/v1"),
            adapter=adapter,
        )

    # Hosted model (model_base_url=None) refused without REMOTE_ENV=1.
    with pytest.raises(Unreachable, match=REMOTE_ENV):
        run_trial(
            exp_home,
            "run-1",
            task_dir,
            TrialConfig(arm="no_memory", model="claude-sonnet-4-6"),
            adapter=adapter,
        )

    # Non-loopback model_base_url refused without REMOTE_ENV=1.
    with pytest.raises(Unreachable, match=REMOTE_ENV):
        run_trial(
            exp_home,
            "run-1",
            task_dir,
            TrialConfig(
                arm="no_memory",
                model="claude-sonnet-4-6",
                model_base_url="https://api.example.com/v1",
            ),
            adapter=adapter,
        )

    # Default home refused as source_home even when egress is allowed.
    with pytest.raises(UsefulnessError, match="default home"):
        run_trial(
            exp_home,
            "run-1",
            task_dir,
            TrialConfig(
                arm="memory",
                model_base_url="http://127.0.0.1:11434/v1",
                source_home=DEFAULT_HOME,
            ),
            adapter=adapter,
        )

    # Loopback model_base_url succeeds without REMOTE_ENV=1.
    assert check_trial_egress("local-qwen", "http://127.0.0.1:11434/v1") == "local"
    assert check_trial_egress("local-qwen", "http://localhost:9931") == "local"

    # Setting REMOTE_ENV=1 allows hosted model.
    monkeypatch.setenv(REMOTE_ENV, "1")
    assert check_trial_egress("claude-sonnet-4-6", None) == "remote_opted_in"
    assert adapter.invocations == []


# ---------------------------------------------------------------------------
# End-to-end scripted trials: both arms, fresh clones, repo MCP stripping,
# real MemoryServer isolation, and source_home immutability
# ---------------------------------------------------------------------------


def test_paired_arms_isolated_clones_repo_mcp_sanitization_and_real_trial_server(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(REMOTE_ENV, raising=False)
    repo, base_sha, future_sha = _make_git_repo(tmp_path, with_repo_mcp=True)
    exp_home = tmp_path / "exp-home"
    task_dir = _write_task(
        exp_home / "tasks",
        "t-pair",
        repo=repo,
        commit=base_sha,
        project="demo-proj",
        hidden_tests={"tests/test_secret.py": "def test_secret():\n    assert 42 == 42\n"},
    )
    source_home = _make_seeded_source_home(tmp_path / "throwaway-source", "demo-proj")
    source_db_sha_before = sha256_file(source_home / "memory.duckdb")

    observed: dict[str, dict[str, Any]] = {}
    trial_write_session_id = str(uuid.uuid4())

    def _scripted_agent(inv: HarnessInvocation) -> HarnessResult:
        # Verify clone is at base_sha, future commit is unreachable, stale build artifacts absent,
        # and hidden test is not in the clone.
        head = _git(inv.clone_dir, "rev-parse", "HEAD")
        assert head == base_sha
        future_check = subprocess.run(
            ["git", "cat-file", "-e", future_sha],
            cwd=str(inv.clone_dir),
            capture_output=True,
        )
        assert future_check.returncode != 0
        assert not (inv.clone_dir / "build-cache").exists()
        assert not (inv.clone_dir / "tests" / "test_secret.py").exists()

        # Verify repo .factory/mcp.json had agent-memory stripped in BOTH arms,
        # while non-memory servers stayed intact.
        repo_mcp = json.loads((inv.clone_dir / ".factory" / "mcp.json").read_text("utf-8"))
        assert "agent-memory" not in repo_mcp["mcpServers"]
        assert "lint-server" in repo_mcp["mcpServers"]

        # Verify second trial never sees files created by an earlier trial.
        assert not (inv.clone_dir / "trial-artifact.txt").exists()
        (inv.clone_dir / "trial-artifact.txt").write_text(f"from {inv.arm}\n", encoding="utf-8")

        recalled_contents: list[str] = []
        if inv.arm == "memory":
            assert inv.memory_endpoint is not None
            ep = inv.memory_endpoint
            assert ep.project == "demo-proj"
            assert ep.home != source_home
            # Talk to the trial's live MemoryServer over Quack and verify seeded memory + write.
            with MemoryClient.from_config(Config(home=ep.home, host=ep.host, port=ep.port)) as cli:
                scope_lit = sql_literal(f"project:{ep.project}")
                _, rows = cli.query(
                    f"SELECT content FROM active_memories WHERE scope = {scope_lit}"
                )
                recalled_contents = [str(r[0]) for r in rows]
                cli.insert(
                    {
                        "sessions": [
                            {
                                "session_id": trial_write_session_id,
                                "agent_id": "factory-droid",
                                "capture_completeness": "full",
                                "project": ep.project,
                                "scope": f"project:{ep.project}",
                                "started_at": datetime(2026, 11, 25, 12, 0, tzinfo=UTC),
                            }
                        ]
                    }
                )
        else:
            assert inv.memory_endpoint is None

        (inv.clone_dir / "app.py").write_text("def answer():\n    return 42\n", encoding="utf-8")
        observed[inv.arm] = {
            "allowed_tools": inv.allowed_tools,
            "recalled": recalled_contents,
            "clone_dir": inv.clone_dir,
        }
        return HarnessResult(
            status="completed",
            last_reply=f"Fixed in {inv.arm} arm.",
            tool_calls=(
                ToolCallRecord(
                    sequence=1,
                    call_id="c1",
                    name="Edit",
                    arguments={"file": "app.py"},
                    is_error=False,
                ),
            ),
            # Deliberately omit thinking_tokens and cache counts to verify they stay None.
            tokens=TokenUsage.from_harness_usage({"input_tokens": 200, "output_tokens": 45}),
            num_turns=1,
            exit_code=0,
            harness_session_id=f"sess-{inv.arm}",
            harness_reported_duration_ms=123.4,
        )

    adapter = ScriptedHarnessAdapter(_scripted_agent)

    rec_no_mem = run_trial(
        exp_home,
        "run-1",
        task_dir,
        TrialConfig(
            arm="no_memory",
            model="local-model",
            model_base_url="http://127.0.0.1:11434/v1",
            trial_id="t-pair-nomem",
        ),
        adapter=adapter,
    )
    rec_mem = run_trial(
        exp_home,
        "run-1",
        task_dir,
        TrialConfig(
            arm="memory",
            model="local-model",
            model_base_url="http://127.0.0.1:11434/v1",
            source_home=source_home,
            trial_id="t-pair-mem",
        ),
        adapter=adapter,
    )

    assert rec_no_mem.status == "completed", rec_no_mem.failure_detail
    assert rec_mem.status == "completed", rec_mem.failure_detail
    # Distinct clones and arm tool symmetry.
    assert observed["no_memory"]["clone_dir"] != observed["memory"]["clone_dir"]
    assert observed["no_memory"]["allowed_tools"] == DEFAULT_BASE_TOOLS
    assert observed["memory"]["allowed_tools"] == (
        *DEFAULT_BASE_TOOLS,
        "ToolSearch",
        "MCP:agent-memory",
    )
    assert observed["memory"]["recalled"] == ["answer() must return 42"]

    # Check record fields, clean repo_changes (no sanitized MCP diff), wall_time, tokens.
    for rec in (rec_no_mem, rec_mem):
        assert rec.schema_version == TRIAL_RECORD_SCHEMA_VERSION
        assert rec.status == "completed"
        assert rec.failure_category is None
        assert rec.repo_changes.base_commit == base_sha
        assert rec.repo_changes.changed_files == ("app.py", "trial-artifact.txt")
        assert rec.repo_changes.untracked_files == ("trial-artifact.txt",)
        assert "return 42" in rec.repo_changes.diff
        assert f"from {rec.arm}" in rec.repo_changes.diff
        assert ".factory/mcp.json" not in rec.repo_changes.changed_files
        assert rec.wall_time.total_s >= 0.0
        assert rec.wall_time.clone_s >= 0.0
        assert rec.wall_time.setup_s >= 0.0
        assert rec.wall_time.harness_s >= 0.0
        assert rec.wall_time.teardown_s >= 0.0
        assert rec.wall_time.harness_reported_duration_ms == 123.4
        assert rec.tokens.input_tokens == 200
        assert rec.tokens.output_tokens == 45
        assert rec.tokens.total_tokens == 245
        assert rec.tokens.thinking_tokens is None
        assert rec.tokens.cache_read_input_tokens is None
        assert rec.environment["sanitized_repo_mcp_files"] == [
            {"path": ".factory/mcp.json", "removed_servers": ["agent-memory"]}
        ]
        # Persisted JSON matches and preserves nulls.
        persisted = json.loads(rec.record_path.read_text("utf-8"))
        assert persisted["tokens"]["thinking_tokens"] is None
        assert persisted["tokens"]["input_tokens"] == 200

    assert rec_no_mem.paths["trial_home"] is None
    assert rec_mem.paths["trial_home"] is not None

    # MemoryServer was stopped cleanly (we can open trial-home DB directly) and contains
    # the trial's write, while source_home is byte-for-byte unchanged.
    trial_home = Path(rec_mem.paths["trial_home"])
    con = open_direct(trial_home / "memory.duckdb", read_only=True)
    try:
        sess_ids = [str(r[0]) for r in con.execute("SELECT session_id FROM sessions").fetchall()]
    finally:
        con.close()
    assert trial_write_session_id in sess_ids
    assert sha256_file(source_home / "memory.duckdb") == source_db_sha_before


def test_hidden_test_in_base_commit_and_manifest_tamper_and_wal_are_refused(
    tmp_path: Path,
) -> None:
    exp_home = tmp_path / "exp"
    adapter = ScriptedHarnessAdapter(lambda inv: HarnessResult(status="completed", last_reply="ok"))

    # 1. Hidden test already present in base commit is refused.
    repo_conflict, base_conflict, _ = _make_git_repo(tmp_path / "c", with_hidden_conflict=True)
    t_conflict = _write_task(
        exp_home / "tasks",
        "t-conflict",
        repo=repo_conflict,
        commit=base_conflict,
        hidden_tests={"tests/test_hidden.py": "assert True\n"},
    )
    with pytest.raises(UsefulnessError, match="hidden tests must not be in the clone"):
        run_trial(
            exp_home,
            "run-conflict",
            t_conflict,
            TrialConfig(arm="no_memory", model_base_url="http://127.0.0.1:11434/v1"),
            adapter=adapter,
        )

    # 2. Tampered task fails manifest verification when task-manifest.json exists.
    repo_ok, base_ok, _ = _make_git_repo(tmp_path / "ok")
    tasks_dir = exp_home / "tasks-m"
    t1 = _write_task(
        tasks_dir,
        "t-cal",
        repo=repo_ok,
        commit=base_ok,
        work_done_at="2026-11-10T00:00:00Z",
        task_family="f1",
    )
    _write_task(
        tasks_dir,
        "t-ho",
        repo=repo_ok,
        commit=base_ok,
        work_done_at="2026-11-20T00:00:00Z",
        task_family="f2",
    )
    store = UsefulnessRunStore(exp_home, "run-manifest")
    manifest = build_task_manifest(tasks_dir)
    store.write_task_manifest(manifest)

    # Modify request.md after manifest was written.
    (t1 / "request.md").write_text("Tampered request\n", encoding="utf-8")
    with pytest.raises(UsefulnessError, match="no longer matches manifest"):
        run_trial(
            exp_home,
            "run-manifest",
            t1,
            TrialConfig(arm="no_memory", model_base_url="http://127.0.0.1:11434/v1"),
            adapter=adapter,
        )

    # 3. Source home with a WAL file is refused.
    wal_home = tmp_path / "wal-source"
    initialize_home(Config(home=wal_home))
    (wal_home / "memory.duckdb.wal").write_bytes(b"wal")
    with pytest.raises(UsefulnessError, match="WAL file"):
        run_trial(
            exp_home,
            "run-wal",
            t_conflict,
            TrialConfig(
                arm="memory",
                model_base_url="http://127.0.0.1:11434/v1",
                source_home=wal_home,
            ),
            adapter=adapter,
        )


# ---------------------------------------------------------------------------
# DroidHarnessAdapter real-subprocess tests with a fake droid script
# ---------------------------------------------------------------------------


def test_droid_adapter_isolation_mcp_stream_json_and_transcript(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, base_sha, _ = _make_git_repo(tmp_path)
    exp_home = tmp_path / "exp"
    task_dir = _write_task(
        exp_home / "tasks", "t-droid", repo=repo, commit=base_sha, project="proj-x"
    )

    # Create a fake user ~/.factory dir containing auth files AND hooks/settings/mcp that
    # must NOT be linked into the trial harness_home.
    fake_user_factory = tmp_path / "user-factory"
    fake_user_factory.mkdir()
    (fake_user_factory / "auth.v2.file").write_text("tok", encoding="utf-8")
    (fake_user_factory / "hooks.json").write_text('{"SessionStart": ["bad"]}', encoding="utf-8")
    (fake_user_factory / "mcp.json").write_text('{"mcpServers": {"bad": {}}}', encoding="utf-8")
    (fake_user_factory / "settings.json").write_text('{"cloudSessionSync": true}', encoding="utf-8")

    # Simulate outer environment variables that must be stripped.
    monkeypatch.setenv("FACTORY_SESSION_ID", "outer-session-should-be-stripped")
    monkeypatch.setenv("AGENT_MEMORY_HOME", str(tmp_path / "outer-prod-home"))
    monkeypatch.setenv("AGENT_MEMORY_PORT", "9494")
    monkeypatch.delenv("FACTORY_API_KEY", raising=False)

    capture_file = tmp_path / "captured-invocation.json"
    fake_droid = _write_fake_droid(
        tmp_path / "fake-droid",
        f"""
import json, os, pathlib, sys

if sys.argv[1:] == ["--version"]:
    print("1.21.0-fake")
    sys.exit(0)

args = sys.argv[1:]
settings_path = args[args.index("--settings") + 1]
prompt_path = args[args.index("-f") + 1]
only_tools = args[args.index("--only-tools") + 1]
auto_level = args[args.index("--auto") + 1]
cwd_arg = args[args.index("--cwd") + 1]

factory_home = pathlib.Path(os.environ["FACTORY_HOME_OVERRIDE"])
mcp_cfg = json.loads((factory_home / ".factory" / "mcp.json").read_text("utf-8"))
settings_cfg = json.loads(pathlib.Path(settings_path).read_text("utf-8"))

info = {{
    "args": args,
    "only_tools": only_tools,
    "auto_level": auto_level,
    "prompt": pathlib.Path(prompt_path).read_text("utf-8"),
    "settings": settings_cfg,
    "mcp": mcp_cfg,
    "factory_home_files": sorted(p.name for p in (factory_home / ".factory").iterdir()),
    "env_stripped": {{
        k: (k in os.environ)
        for k in ("FACTORY_SESSION_ID", "AGENT_MEMORY_HOME", "AGENT_MEMORY_PORT")
    }},
    "auto_update": os.environ.get("FACTORY_DROID_AUTO_UPDATE_ENABLED"),
}}
pathlib.Path({str(capture_file)!r}).write_text(json.dumps(info), encoding="utf-8")

# Write a transcript at the Droid MEM-17 path.
sanitized = str(pathlib.Path(cwd_arg).resolve()).replace("/", "-")
sess_dir = factory_home / ".factory" / "sessions" / sanitized
sess_dir.mkdir(parents=True, exist_ok=True)
(sess_dir / "sess-fake-1.jsonl").write_text('{{"type":"session_start"}}\\n', encoding="utf-8")

# Edit the repo so repo_changes is populated.
(pathlib.Path(cwd_arg) / "app.py").write_text("def answer():\\n    return 42\\n", encoding="utf-8")

events = [
    {{
        "type": "system",
        "subtype": "init",
        "session_id": "sess-fake-1",
        "model": "claude-sonnet-4-6",
        "reasoning_effort": "high",
    }},
    {{
        "type": "tool_call",
        "id": "tc-1",
        "toolName": "agent-memory___recall",
        "parameters": {{"query": "answer"}},
    }},
    {{"type": "tool_result", "id": "tc-1", "isError": False}},
    {{"type": "message", "role": "assistant", "text": "Updated answer() to return 42."}},
    {{
        "type": "completion",
        "session_id": "sess-fake-1",
        "finalText": "Done: answer() returns 42.",
        "numTurns": 2,
        "durationMs": 456.0,
        "usage": {{
            "input_tokens": 300,
            "output_tokens": 50,
            "cache_read_input_tokens": 100,
            "ttft_ms": 80.5,
        }},
    }},
]
for ev in events:
    print(json.dumps(ev), flush=True)
""",
    )

    adapter = DroidHarnessAdapter(
        droid_bin=str(fake_droid),
        auth_source_dir=fake_user_factory,
    )
    record = run_trial(
        exp_home,
        "run-droid",
        task_dir,
        TrialConfig(
            arm="memory",
            model="claude-sonnet-4-6",
            reasoning_effort="high",
            model_base_url="http://127.0.0.1:11434/v1",
            trial_id="t-droid-mem",
        ),
        adapter=adapter,
    )

    captured = json.loads(capture_file.read_text("utf-8"))
    assert captured["auto_level"] == "high"
    assert "--disable-builtin-skills" in captured["args"]
    assert captured["only_tools"] == (
        "Read,LS,Glob,Grep,Edit,Create,Execute,ToolSearch,MCP:agent-memory"
    )
    assert captured["settings"] == {
        "hooks": {"hooksDisabled": True},
        "cloudSessionSync": False,
    }
    assert captured["auto_update"] == "false"
    assert captured["env_stripped"] == {
        "FACTORY_SESSION_ID": False,
        "AGENT_MEMORY_HOME": False,
        "AGENT_MEMORY_PORT": False,
    }
    # Auth file was symlinked; user hooks.json was NOT.
    assert "auth.v2.file" in captured["factory_home_files"]
    assert "hooks.json" not in captured["factory_home_files"]
    # Memory MCP server env points at trial home & task.project.
    mcp_env = captured["mcp"]["mcpServers"]["agent-memory"]["env"]
    assert mcp_env["AGENT_MEMORY_HOME"] == record.paths["trial_home"]
    assert mcp_env["AGENT_MEMORY_PROJECT"] == "proj-x"
    assert mcp_env["AGENT_MEMORY_AGENT"] == "factory-droid"

    assert record.status == "completed"
    assert record.failure_category is None
    assert record.harness == "droid"
    assert record.harness_version == "1.21.0-fake"
    assert record.harness_session_id == "sess-fake-1"
    assert record.last_reply == "Done: answer() returns 42."
    assert record.num_turns == 2
    assert len(record.tool_calls) == 1
    assert record.tool_calls[0].name == "agent-memory___recall"
    assert record.tool_calls[0].is_error is False
    assert record.tokens.input_tokens == 300
    assert record.tokens.output_tokens == 50
    assert record.tokens.cache_read_input_tokens == 100
    assert record.tokens.thinking_tokens is None
    assert record.tokens.total_tokens == 450
    assert record.paths["transcript_path"] is not None
    assert Path(record.paths["transcript_path"]).is_file()


def test_droid_adapter_enforces_wall_time_limit_kills_process_group_and_keeps_record(
    tmp_path: Path,
) -> None:
    repo, base_sha, _ = _make_git_repo(tmp_path)
    exp_home = tmp_path / "exp"
    task_dir = _write_task(exp_home / "tasks", "t-timeout", repo=repo, commit=base_sha)
    child_pid_file = tmp_path / "grandchild.pid"

    fake_slow_droid = _write_fake_droid(
        tmp_path / "fake-slow-droid",
        f"""
import json, pathlib, subprocess, sys, time

if sys.argv[1:] == ["--version"]:
    print("1.21.0-fake")
    sys.exit(0)

print(
    json.dumps({{"type": "system", "subtype": "init", "session_id": "sess-timeout"}}),
    flush=True,
)
print(
    json.dumps(
        {{
            "type": "tool_call",
            "id": "tc-slow",
            "toolName": "Execute",
            "parameters": {{"command": "sleep 30"}},
        }}
    ),
    flush=True,
)

child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
pathlib.Path({str(child_pid_file)!r}).write_text(str(child.pid), encoding="utf-8")
time.sleep(30)
""",
    )

    adapter = DroidHarnessAdapter(droid_bin=str(fake_slow_droid))
    t0 = time.monotonic()
    record = run_trial(
        exp_home,
        "run-timeout",
        task_dir,
        TrialConfig(
            arm="no_memory",
            model="local-model",
            model_base_url="http://127.0.0.1:11434/v1",
            wall_time_limit_s=0.4,
            kill_grace_s=0.2,
            trial_id="t-timeout-1",
        ),
        adapter=adapter,
    )
    elapsed = time.monotonic() - t0

    assert elapsed < 5.0
    assert record.status == "failed"
    assert record.failure_category == "timeout"
    assert record.failure_detail is not None
    assert "wall-time limit" in record.failure_detail
    assert record.harness_session_id == "sess-timeout"
    assert len(record.tool_calls) == 1
    assert record.tool_calls[0].name == "Execute"
    assert record.tokens.input_tokens is None
    assert record.tokens.output_tokens is None
    assert record.record_path.is_file()

    # Confirm the grandchild process in the process group was also terminated.
    if child_pid_file.exists():
        child_pid = int(child_pid_file.read_text("utf-8").strip())
        # Give the OS a moment to reap the killed child.
        for _ in range(20):
            try:
                os.kill(child_pid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.05)
        with pytest.raises(ProcessLookupError):
            os.kill(child_pid, 0)


@pytest.mark.parametrize(
    ("error_msg", "expected_category"),
    [
        ("Authentication failed: invalid API key", "auth_error"),
        ("insufficient permission to proceed. Re-run with --auto high", "permission_denied"),
        ("failed to connect to MCP server agent-memory", "tool_outage"),
        ("Model refused to fulfill request due to content policy", "model_refusal"),
        ("unexpected internal crash", "harness_error"),
    ],
)
def test_droid_adapter_classifies_failures_and_keeps_record(
    tmp_path: Path, error_msg: str, expected_category: str
) -> None:
    repo, base_sha, _ = _make_git_repo(tmp_path)
    exp_home = tmp_path / "exp"
    task_dir = _write_task(exp_home / "tasks", "t-fail", repo=repo, commit=base_sha)

    fake_fail_droid = _write_fake_droid(
        tmp_path / "fake-fail-droid",
        f"""
import json, sys

if sys.argv[1:] == ["--version"]:
    print("1.21.0-fake")
    sys.exit(0)

print(json.dumps({{
    "type": "result",
    "subtype": "failure",
    "is_error": True,
    "num_turns": 0,
    "result": {error_msg!r},
    "usage": {{"input_tokens": 0, "output_tokens": 0}},
}}), flush=True)
sys.exit(1)
""",
    )

    record = run_trial(
        exp_home,
        "run-fail",
        task_dir,
        TrialConfig(
            arm="no_memory",
            model="local-model",
            model_base_url="http://127.0.0.1:11434/v1",
            trial_id=f"t-fail-{expected_category}",
        ),
        adapter=DroidHarnessAdapter(droid_bin=str(fake_fail_droid)),
    )
    assert record.status == "failed"
    assert record.failure_category == expected_category
    assert record.last_reply is None
    # Pre-call failure with no ttft_ms must record missing tokens (None), not 0.
    assert record.tokens.input_tokens is None
    assert record.tokens.output_tokens is None
    assert record.record_path.is_file()


def test_droid_adapter_missing_binary_records_environment_error(tmp_path: Path) -> None:
    repo, base_sha, _ = _make_git_repo(tmp_path)
    exp_home = tmp_path / "exp"
    task_dir = _write_task(exp_home / "tasks", "t-nobin", repo=repo, commit=base_sha)

    record = run_trial(
        exp_home,
        "run-nobin",
        task_dir,
        TrialConfig(
            arm="no_memory",
            model="local-model",
            model_base_url="http://127.0.0.1:11434/v1",
            trial_id="t-nobin-1",
        ),
        adapter=DroidHarnessAdapter(droid_bin=str(tmp_path / "does-not-exist-droid")),
    )
    assert record.status == "failed"
    assert record.failure_category == "environment_error"
    assert record.record_path.is_file()


def test_parse_droid_stream_events_handles_malformed_lines_and_message_fallback() -> None:
    parsed = parse_droid_stream_events(
        [
            "not json at all",
            json.dumps({"type": "system", "subtype": "init", "session_id": "s1", "model": "m1"}),
            json.dumps({"type": "message", "role": "assistant", "text": "Fallback reply"}),
        ]
    )
    assert parsed.session_id == "s1"
    assert parsed.model == "m1"
    assert parsed.last_reply == "Fallback reply"


# ---------------------------------------------------------------------------
# CLI: `mem --home <exp_home> usefulness run-trial`
# ---------------------------------------------------------------------------


def test_cli_run_trial_end_to_end_and_guards(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.delenv(REMOTE_ENV, raising=False)
    repo, base_sha, _ = _make_git_repo(tmp_path)
    exp_home = tmp_path / "exp"
    _write_task(exp_home / "tasks", "t-cli", repo=repo, commit=base_sha)

    fake_droid = _write_fake_droid(
        tmp_path / "fake-cli-droid",
        """
import json, sys
if sys.argv[1:] == ["--version"]:
    print("1.21.0-fake")
    sys.exit(0)
print(
    json.dumps(
        {
            "type": "completion",
            "finalText": "CLI ok",
            "numTurns": 1,
            "durationMs": 12.0,
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }
    ),
    flush=True,
)
""",
    )

    # 1. Default home refused.
    code = cli_main(
        [
            "--home",
            str(DEFAULT_HOME),
            "usefulness",
            "run-trial",
            "--run-id",
            "r1",
            "--task",
            str(exp_home / "tasks" / "t-cli"),
            "--arm",
            "no_memory",
            "--model-base-url",
            "http://127.0.0.1:11434/v1",
            "--droid-bin",
            str(fake_droid),
        ]
    )
    assert code == 1
    assert "default home" in capsys.readouterr().err

    # 2. Hosted model refused without REMOTE_ENV=1.
    code = cli_main(
        [
            "--home",
            str(exp_home),
            "usefulness",
            "run-trial",
            "--run-id",
            "r1",
            "--task",
            "t-cli",
            "--arm",
            "no_memory",
            "--droid-bin",
            str(fake_droid),
        ]
    )
    assert code == 1
    assert REMOTE_ENV in capsys.readouterr().err

    # 3. Successful run via task id and loopback model-base-url with --json.
    code = cli_main(
        [
            "--home",
            str(exp_home),
            "usefulness",
            "run-trial",
            "--run-id",
            "r1",
            "--task",
            "t-cli",
            "--arm",
            "no_memory",
            "--model-base-url",
            "http://127.0.0.1:11434/v1",
            "--droid-bin",
            str(fake_droid),
            "--trial-id",
            "cli-trial-1",
            "--json",
        ]
    )
    assert code == 0
    out = json.loads(capsys.readouterr().out)
    assert out["trial_id"] == "cli-trial-1"
    assert out["status"] == "completed"
    assert out["last_reply"] == "CLI ok"
    assert out["tokens"]["input_tokens"] == 10
    assert out["tokens"]["output_tokens"] == 5
    assert out["tokens"]["thinking_tokens"] is None
