"""Single-trial runner for the usefulness experiment (MEM-18).

Runs one trial of a task in either the ``memory`` or ``no_memory`` arm inside a
fresh git clone at the task's commit, isolated from user and repository harness
configuration, enforcing wall-time limits and egress/default-home guards.

The harness is invoked behind ``HarnessAdapter``: ``DroidHarnessAdapter`` wraps
headless ``droid exec`` based on the MEM-17 probe findings, and
``ScriptedHarnessAdapter`` provides an in-process stand-in for tests without
calling any model.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import secrets
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

from ..config import Config, ConfigError
from ..db import MemoryClient, MemoryServer, initialize_home
from ..endpoints import REMOTE_ENV, Unreachable, is_loopback, remote_allowed, require_egress
from ..judge.store import private_dir, utc_now_iso
from ..mcp_server import MCP_WORDING_ENV, load_tool_wording_file, validate_tool_wording
from ..smoke import free_port
from .accounting import (
    RecallAccounting,
    TrialLedgerBaseline,
    account_trial_recall,
    capture_trial_baseline,
    import_trial_transcript,
)
from .registration import (
    default_trial_config_overrides,
    detect_hardware_class,
    detect_repo_commit,
    memory_tool_wording_sha256,
    verify_registration_for_trial,
)
from .store import UsefulnessError, UsefulnessRunStore, write_once_json
from .tasks import TaskSpec, assert_hidden_tests_absent, load_task, load_verified_task

TRIAL_RECORD_SCHEMA_VERSION = 1

Arm = Literal["memory", "no_memory"]
VALID_ARMS: tuple[Arm, ...] = ("memory", "no_memory")
VALID_AUTO_LEVELS: tuple[str, ...] = ("low", "medium", "high")

DEFAULT_MEMORY_SERVER_NAME = "agent-memory"
DEFAULT_MEMORY_AGENT_ID = "factory-droid"
DEFAULT_BASE_TOOLS: tuple[str, ...] = ("Read", "LS", "Glob", "Grep", "Edit", "Create", "Execute")
DEFAULT_AUTO_LEVEL = "high"
DEFAULT_WALL_TIME_LIMIT_S = 600.0
DEFAULT_DROID_MODEL = "claude-sonnet-4-6"

FAILURE_CATEGORIES: tuple[str, ...] = (
    "timeout",
    "permission_denied",
    "auth_error",
    "tool_outage",
    "model_refusal",
    "harness_error",
    "environment_error",
    "interrupted",
)

# Relative paths inside a task repository where harness MCP servers may be configured.
_REPO_MCP_CONFIG_PATHS: tuple[str, ...] = (".factory/mcp.json", ".mcp.json")

# Auth files symlinked into isolated FACTORY_HOME_OVERRIDE/.factory/ on macOS/Linux
# when FACTORY_API_KEY is not set. Hooks, settings, MCP configs, skills, and droids
# are never linked or copied from the user's home.
_DROID_AUTH_FILES: tuple[str, ...] = (
    "auth.v2.file",
    "auth.v2.key",
    "auth.v2.loginkeychain",
)

# Environment variables stripped from the harness subprocess so neither arm can reach
# the user's production home or inherit an outer Droid session id.
_STRIPPED_HARNESS_ENV: tuple[str, ...] = (
    "FACTORY_SESSION_ID",
    "AGENT_MEMORY_HOME",
    "AGENT_MEMORY_PORT",
    "AGENT_MEMORY_HOST",
    "AGENT_MEMORY_AGENT",
    "AGENT_MEMORY_PROJECT",
    "AGENT_MEMORY_SESSION",
    MCP_WORDING_ENV,
)

_TRIAL_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
_MEMORY_SERVER_MARKERS: tuple[str, ...] = (
    "agent-memory",
    "agent_memory",
    "mem-mcp",
)


def _opt_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


def _opt_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


@dataclass(frozen=True)
class TokenUsage:
    """Token counts reported by the harness. Unavailable counts are ``None``, never ``0``."""

    input_tokens: int | None = None
    output_tokens: int | None = None
    cache_read_input_tokens: int | None = None
    cache_creation_input_tokens: int | None = None
    thinking_tokens: int | None = None
    total_tokens: int | None = None
    factory_credits: float | None = None
    ttft_ms: float | None = None

    @classmethod
    def from_harness_usage(
        cls,
        raw: Mapping[str, Any] | None,
        *,
        pre_call_failure: bool = False,
    ) -> TokenUsage:
        """Parse a harness usage mapping, preserving every missing field as ``None``."""
        if not isinstance(raw, Mapping):
            return cls()
        ttft_ms = _opt_float(raw.get("ttft_ms"))
        # MEM-17: on pre-call failures (e.g. auth error before any model call), Droid
        # emits 0 for token counts while omitting ttft_ms. Record missing rather than 0.
        if pre_call_failure and ttft_ms is None:
            return cls()

        input_tokens = _opt_int(raw.get("input_tokens"))
        output_tokens = _opt_int(raw.get("output_tokens"))
        cache_read = _opt_int(raw.get("cache_read_input_tokens"))
        cache_creation = _opt_int(raw.get("cache_creation_input_tokens"))
        thinking = _opt_int(raw.get("thinking_tokens"))
        explicit_total = _opt_int(raw.get("total_tokens"))
        credits_val = _opt_float(raw.get("factory_credits"))

        total: int | None
        if explicit_total is not None:
            total = explicit_total
        elif input_tokens is not None and output_tokens is not None:
            total = (
                input_tokens
                + (cache_read or 0)
                + (cache_creation or 0)
                + output_tokens
                + (thinking or 0)
            )
        else:
            total = None

        return cls(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_input_tokens=cache_read,
            cache_creation_input_tokens=cache_creation,
            thinking_tokens=thinking,
            total_tokens=total,
            factory_credits=credits_val,
            ttft_ms=ttft_ms,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_read_input_tokens": self.cache_read_input_tokens,
            "cache_creation_input_tokens": self.cache_creation_input_tokens,
            "thinking_tokens": self.thinking_tokens,
            "total_tokens": self.total_tokens,
            "factory_credits": self.factory_credits,
            "ttft_ms": self.ttft_ms,
        }


@dataclass(frozen=True)
class ToolCallRecord:
    """One tool invocation observed from the harness stream."""

    sequence: int
    name: str
    call_id: str | None = None
    arguments: Mapping[str, Any] | None = None
    is_error: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence,
            "call_id": self.call_id,
            "name": self.name,
            "arguments": dict(self.arguments) if self.arguments is not None else None,
            "is_error": self.is_error,
        }


@dataclass(frozen=True)
class WallTimeBreakdown:
    """Wall time of a trial in seconds, itemized by stage."""

    clone_s: float
    setup_s: float
    harness_s: float
    teardown_s: float
    total_s: float
    harness_reported_duration_ms: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "clone_s": self.clone_s,
            "setup_s": self.setup_s,
            "harness_s": self.harness_s,
            "teardown_s": self.teardown_s,
            "total_s": self.total_s,
            "harness_reported_duration_ms": self.harness_reported_duration_ms,
        }


@dataclass(frozen=True)
class RepoChanges:
    """Repository changes observed in the trial clone relative to ``base_commit``."""

    base_commit: str
    head_commit: str
    changed_files: tuple[str, ...]
    untracked_files: tuple[str, ...]
    diff_stat: str
    diff: str
    has_changes: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "base_commit": self.base_commit,
            "head_commit": self.head_commit,
            "changed_files": list(self.changed_files),
            "untracked_files": list(self.untracked_files),
            "diff_stat": self.diff_stat,
            "diff": self.diff,
            "has_changes": self.has_changes,
        }


@dataclass(frozen=True)
class MemoryEndpoint:
    """Connection settings for the memory arm's dedicated per-trial ``MemoryServer``."""

    home: Path
    host: str
    port: int
    project: str
    agent_id: str = DEFAULT_MEMORY_AGENT_ID
    server_name: str = DEFAULT_MEMORY_SERVER_NAME
    wording_path: Path | None = None
    wording_sha256: str | None = None


@dataclass(frozen=True)
class TrialConfig:
    """Registered settings for running a single trial."""

    arm: Arm
    model: str = DEFAULT_DROID_MODEL
    reasoning_effort: str | None = None
    wall_time_limit_s: float = DEFAULT_WALL_TIME_LIMIT_S
    base_tools: tuple[str, ...] = DEFAULT_BASE_TOOLS
    auto_level: str = DEFAULT_AUTO_LEVEL
    model_base_url: str | None = None
    source_home: Path | None = None
    trial_id: str | None = None
    verify_manifest: bool = True
    memory_server_name: str = DEFAULT_MEMORY_SERVER_NAME
    kill_grace_s: float = 5.0
    memory_tool_wording: Mapping[str, Any] | None = None
    memory_tool_wording_file: Path | None = None
    repo_commit: str | None = None
    hardware_class: str | None = None
    harness_version: str | None = None


@dataclass(frozen=True)
class HarnessInvocation:
    """Inputs handed to a ``HarnessAdapter`` for one trial."""

    trial_id: str
    arm: Arm
    task: TaskSpec
    clone_dir: Path
    harness_home: Path
    config: TrialConfig
    memory_endpoint: MemoryEndpoint | None
    allowed_tools: tuple[str, ...]


@dataclass(frozen=True)
class HarnessResult:
    """Outcome returned by a ``HarnessAdapter``."""

    status: Literal["completed", "failed"]
    last_reply: str | None
    tool_calls: tuple[ToolCallRecord, ...] = ()
    tokens: TokenUsage = field(default_factory=TokenUsage)
    num_turns: int | None = None
    failure_category: str | None = None
    failure_detail: str | None = None
    exit_code: int | None = None
    harness_session_id: str | None = None
    harness_reported_duration_ms: float | None = None
    transcript_path: Path | None = None
    settings_path: Path | None = None
    mcp_config_path: Path | None = None
    harness_version: str | None = None
    resolved_model: str | None = None
    resolved_reasoning_effort: str | None = None
    raw_events_path: Path | None = None
    transcript_harness: str | None = None


class HarnessAdapter(Protocol):
    """Protocol implemented by headless coding-agent harness adapters."""

    name: str

    def run(self, invocation: HarnessInvocation) -> HarnessResult: ...


@dataclass(frozen=True)
class TrialRecord:
    """Immutable record persisted to ``<trial_dir>/record.json`` for one trial."""

    schema_version: int
    trial_id: str
    run_id: str
    task_id: str
    task_sha256: str
    arm: Arm
    status: Literal["completed", "failed"]
    failure_category: str | None
    failure_detail: str | None
    exit_code: int | None
    harness: str
    harness_version: str | None
    harness_session_id: str | None
    model: str
    reasoning_effort: str | None
    num_turns: int | None
    last_reply: str | None
    repo_changes: RepoChanges
    wall_time: WallTimeBreakdown
    tool_calls: tuple[ToolCallRecord, ...]
    tokens: TokenUsage
    started_at: str
    finished_at: str
    paths: Mapping[str, str | None]
    environment: Mapping[str, Any]
    record_path: Path
    record_sha256: str
    registration_sha256: str | None = None
    recall_accounting: RecallAccounting | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "trial_id": self.trial_id,
            "run_id": self.run_id,
            "task_id": self.task_id,
            "task_sha256": self.task_sha256,
            "registration_sha256": self.registration_sha256,
            "arm": self.arm,
            "status": self.status,
            "failure_category": self.failure_category,
            "failure_detail": self.failure_detail,
            "exit_code": self.exit_code,
            "harness": self.harness,
            "harness_version": self.harness_version,
            "harness_session_id": self.harness_session_id,
            "model": self.model,
            "reasoning_effort": self.reasoning_effort,
            "num_turns": self.num_turns,
            "last_reply": self.last_reply,
            "repo_changes": self.repo_changes.to_dict(),
            "wall_time": self.wall_time.to_dict(),
            "tool_calls": [tc.to_dict() for tc in self.tool_calls],
            "tokens": self.tokens.to_dict(),
            "recall_accounting": (
                self.recall_accounting.to_dict() if self.recall_accounting is not None else None
            ),
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "paths": dict(self.paths),
            "environment": dict(self.environment),
        }


# ---------------------------------------------------------------------------
# Egress, git clone, repo sanitization, and trial-home setup
# ---------------------------------------------------------------------------


def check_trial_egress(model: str, model_base_url: str | None) -> str:
    """Verify whether a trial's model is allowed to run under the egress policy.

    A model with no ``model_base_url`` is treated as hosted (off-machine) and
    requires ``AGENT_MEMORY_JUDGE_ALLOW_REMOTE=1``. A loopback ``model_base_url``
    is allowed without opting in to egress; any non-loopback URL goes through
    ``endpoints.require_egress``.
    """
    if model_base_url is not None:
        require_egress(model_base_url, f"Usefulness trial for model {model!r}")
        return "local" if is_loopback(model_base_url) else "remote_opted_in"
    if not remote_allowed():
        raise Unreachable(
            f"Usefulness trial for hosted model {model!r} sends data to a non-loopback "
            f"service. Set {REMOTE_ENV}=1 to allow it, or pass a loopback model_base_url."
        )
    return "remote_opted_in"


def _git_env() -> dict[str, str]:
    env = os.environ.copy()
    env["GIT_TERMINAL_PROMPT"] = "0"
    return env


def _run_git(
    args: Sequence[str],
    *,
    cwd: Path | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(
        ["git", *args],
        cwd=str(cwd) if cwd is not None else None,
        env=_git_env(),
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if check and proc.returncode != 0:
        detail = (proc.stderr or proc.stdout).strip()
        raise UsefulnessError(f"git {' '.join(args)} failed (exit {proc.returncode}): {detail}")
    return proc


def _clone_task_repo(task: TaskSpec, clone_dir: Path) -> str:
    """Create a fresh clone of ``task.repository`` at ``task.commit``.

    Removes remote references and any branch/tag pointing outside ``task.commit``'s
    ancestry so future commits after the task commit are not visible in the clone.
    """
    if clone_dir.exists():
        raise UsefulnessError(f"clone directory already exists: {clone_dir}")
    clone_dir.parent.mkdir(parents=True, exist_ok=True)
    _run_git(["clone", "--no-hardlinks", "--no-checkout", task.repository, str(clone_dir)])
    _run_git(["checkout", "--detach", task.commit], cwd=clone_dir)

    base_commit = _run_git(["rev-parse", "HEAD"], cwd=clone_dir).stdout.strip()
    if not base_commit.lower().startswith(task.commit.lower()):
        raise UsefulnessError(
            f"task {task.task_id!r}: checked out commit {base_commit} does not match {task.commit}"
        )

    # Strip origin and any branches/remotes/future tags so the agent cannot inspect
    # commits made after task.commit (such as the real fix).
    _run_git(["remote", "remove", "origin"], cwd=clone_dir, check=False)
    refs_out = _run_git(
        ["for-each-ref", "--format=%(refname)", "refs/heads", "refs/remotes", "refs/tags"],
        cwd=clone_dir,
        check=False,
    ).stdout
    removed_any = False
    for ref in (line.strip() for line in refs_out.splitlines() if line.strip()):
        if ref.startswith("refs/tags/"):
            is_anc = _run_git(
                ["merge-base", "--is-ancestor", ref, "HEAD"], cwd=clone_dir, check=False
            )
            if is_anc.returncode == 0:
                continue
        _run_git(["update-ref", "-d", ref], cwd=clone_dir, check=False)
        removed_any = True
    if removed_any:
        _run_git(["reflog", "expire", "--expire=now", "--all"], cwd=clone_dir, check=False)
        _run_git(["gc", "--prune=now", "--quiet"], cwd=clone_dir, check=False)

    assert_hidden_tests_absent(task, clone_dir)
    return base_commit


def _is_memory_mcp_server(name: str, cfg: Any) -> bool:
    lower_name = name.lower()
    if any(marker in lower_name for marker in _MEMORY_SERVER_MARKERS):
        return True
    if not isinstance(cfg, Mapping):
        return False
    parts: list[str] = []
    cmd = cfg.get("command")
    if isinstance(cmd, str):
        parts.append(cmd)
    args = cfg.get("args")
    if isinstance(args, list):
        parts.extend(str(item) for item in args)
    joined = " ".join(parts).lower()
    return any(marker in joined for marker in _MEMORY_SERVER_MARKERS)


@dataclass(frozen=True)
class _SanitizedMcpFile:
    rel_path: str
    path: Path
    original_bytes: bytes
    sanitized_bytes: bytes
    removed_servers: tuple[str, ...]


def _sanitize_repo_mcp_configs(clone_dir: Path) -> list[_SanitizedMcpFile]:
    """Remove memory MCP server definitions from repository config files in ``clone_dir``.

    Saves original and sanitized bytes so unchanged sanitized files can be restored
    after the harness exits without polluting ``repo_changes``.
    """
    resolved_root = clone_dir.resolve()
    sanitized: list[_SanitizedMcpFile] = []
    for rel in _REPO_MCP_CONFIG_PATHS:
        candidate = clone_dir / rel
        if not candidate.exists() or candidate.is_symlink() or not candidate.is_file():
            continue
        if not candidate.resolve().is_relative_to(resolved_root):
            continue
        original_bytes = candidate.read_bytes()
        try:
            parsed = json.loads(original_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
        if not isinstance(parsed, dict):
            continue
        servers = parsed.get("mcpServers")
        if not isinstance(servers, dict):
            continue
        removed = sorted(
            name for name, entry in servers.items() if _is_memory_mcp_server(str(name), entry)
        )
        if not removed:
            continue
        updated = dict(parsed)
        updated["mcpServers"] = {k: v for k, v in servers.items() if k not in removed}
        sanitized_bytes = (json.dumps(updated, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
        candidate.write_bytes(sanitized_bytes)
        sanitized.append(
            _SanitizedMcpFile(
                rel_path=rel,
                path=candidate,
                original_bytes=original_bytes,
                sanitized_bytes=sanitized_bytes,
                removed_servers=tuple(removed),
            )
        )
    return sanitized


def _restore_untouched_sanitized_files(sanitized: Sequence[_SanitizedMcpFile]) -> None:
    """Restore repo MCP files that the agent did not modify so diffs stay clean."""
    for item in sanitized:
        if item.path.exists() and not item.path.is_symlink():
            with contextlib.suppress(OSError):
                if item.path.read_bytes() == item.sanitized_bytes:
                    item.path.write_bytes(item.original_bytes)


def _capture_repo_changes(clone_dir: Path, base_commit: str) -> RepoChanges:
    head_commit = _run_git(["rev-parse", "HEAD"], cwd=clone_dir).stdout.strip()
    tracked_out = _run_git(["diff", "--name-only", base_commit], cwd=clone_dir).stdout
    tracked_files = [line.strip() for line in tracked_out.splitlines() if line.strip()]

    untracked_out = _run_git(["ls-files", "--others", "--exclude-standard"], cwd=clone_dir).stdout
    untracked_files = sorted(line.strip() for line in untracked_out.splitlines() if line.strip())

    diff_stat = _run_git(["diff", "--stat", base_commit], cwd=clone_dir).stdout
    diff_chunks: list[str] = []
    tracked_diff = _run_git(["diff", base_commit], cwd=clone_dir).stdout
    if tracked_diff:
        diff_chunks.append(tracked_diff)
    for rel in untracked_files:
        proc = _run_git(
            ["diff", "--no-index", "--", os.devnull, rel],
            cwd=clone_dir,
            check=False,
        )
        # git diff --no-index exits 1 when the two paths differ (normal for new files).
        if proc.stdout:
            diff_chunks.append(proc.stdout)

    changed_files = tuple(sorted(set(tracked_files) | set(untracked_files)))
    has_changes = bool(changed_files) or head_commit != base_commit
    return RepoChanges(
        base_commit=base_commit,
        head_commit=head_commit,
        changed_files=changed_files,
        untracked_files=tuple(untracked_files),
        diff_stat=diff_stat,
        diff="".join(diff_chunks),
        has_changes=has_changes,
    )


def _prepare_trial_home(
    trial_home: Path,
    source_home: Path | None,
    port: int,
) -> Config:
    """Create a fresh per-trial ledger home for the memory arm.

    When ``source_home`` is ``None``, initializes a fresh empty ledger home.
    When ``source_home`` is given, copies its stopped ``memory.duckdb`` and
    optional ``blobs/`` directory and mints a fresh per-trial token.
    """
    from .store import ensure_experiment_home

    if trial_home.exists():
        raise UsefulnessError(f"trial home already exists: {trial_home}")
    private_dir(trial_home)
    cfg = Config(home=trial_home, port=port)

    if source_home is None:
        initialize_home(cfg)
        return cfg

    resolved_source = ensure_experiment_home(source_home)
    src_db = resolved_source / "memory.duckdb"
    if not src_db.is_file():
        raise UsefulnessError(f"source home has no database file: {src_db}")
    src_wal = resolved_source / "memory.duckdb.wal"
    if src_wal.exists():
        raise UsefulnessError(
            f"source home has a WAL file ({src_wal}); stop its server before copying into a trial"
        )

    shutil.copy2(src_db, cfg.db_path)
    src_blobs = resolved_source / "blobs"
    if src_blobs.is_dir():
        shutil.copytree(src_blobs, cfg.blob_dir)
    else:
        cfg.blob_dir.mkdir(parents=True, exist_ok=True)
    cfg.export_dir.mkdir(parents=True, exist_ok=True)
    cfg.spool_dir.mkdir(parents=True, exist_ok=True)

    # Fresh token per trial home so no credential is shared with source_home.
    fd = os.open(cfg.token_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(fd, (secrets.token_hex(32) + "\n").encode("ascii"))
    finally:
        os.close(fd)
    return cfg


# ---------------------------------------------------------------------------
# Headless Droid adapter (built from MEM-17 probe findings)
# ---------------------------------------------------------------------------


def _classify_droid_failure(
    exit_code: int | None,
    error_text: str | None,
    stderr_text: str,
) -> str:
    combined = f"{error_text or ''}\n{stderr_text}".lower()
    if any(
        tok in combined
        for tok in (
            "authentication failed",
            "unauthorized",
            "invalid api key",
            "not authenticated",
            "login required",
            "factory_api_key",
        )
    ):
        return "auth_error"
    if any(
        tok in combined
        for tok in (
            "insufficient permission",
            "re-run with --auto high",
            "permission denied",
            "not allowed under current autonomy",
        )
    ):
        return "permission_denied"
    if any(
        tok in combined
        for tok in (
            "mcp server",
            "failed to connect to mcp",
            "tool server",
            "connection refused",
        )
    ):
        return "tool_outage"
    if any(
        tok in combined
        for tok in (
            "content policy",
            "safety filter",
            "refused to fulfill",
            "model refused",
        )
    ):
        return "model_refusal"
    return "harness_error"


def _extract_message_text(event: Mapping[str, Any]) -> str | None:
    text = event.get("text")
    if isinstance(text, str) and text:
        return text
    content = event.get("content")
    if isinstance(content, str) and content:
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, Mapping) and block.get("type") == "text":
                t = block.get("text")
                if isinstance(t, str):
                    parts.append(t)
        joined = "".join(parts)
        return joined or None
    return None


@dataclass
class _ParsedDroidStream:
    session_id: str | None = None
    model: str | None = None
    reasoning_effort: str | None = None
    last_reply: str | None = None
    num_turns: int | None = None
    duration_ms: float | None = None
    tokens: TokenUsage = field(default_factory=TokenUsage)
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    error_text: str | None = None
    terminal_is_error: bool = False
    saw_completion: bool = False


def parse_droid_stream_events(lines: Sequence[str]) -> _ParsedDroidStream:
    """Parse NDJSON lines from ``droid exec -o stream-json`` (or ``-o json``)."""
    parsed = _ParsedDroidStream()
    pending_index_by_id: dict[str, int] = {}
    fallback_assistant_text: str | None = None

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue

        ev_type = event.get("type")
        if isinstance(event.get("session_id"), str) and event["session_id"]:
            parsed.session_id = event["session_id"]

        if ev_type == "system" and event.get("subtype") == "init":
            if isinstance(event.get("model"), str):
                parsed.model = event["model"]
            if isinstance(event.get("reasoning_effort"), str):
                parsed.reasoning_effort = event["reasoning_effort"]

        elif ev_type == "tool_call":
            name = event.get("toolName") or event.get("name")
            if isinstance(name, str) and name:
                call_id_raw = event.get("id") or event.get("callId") or event.get("toolCallId")
                call_id = str(call_id_raw) if call_id_raw is not None else None
                raw_args = event.get("parameters")
                if raw_args is None:
                    raw_args = event.get("arguments")
                if raw_args is None:
                    raw_args = event.get("input")
                args_dict = dict(raw_args) if isinstance(raw_args, Mapping) else None
                rec = ToolCallRecord(
                    sequence=len(parsed.tool_calls) + 1,
                    call_id=call_id,
                    name=name,
                    arguments=args_dict,
                    is_error=None,
                )
                idx = len(parsed.tool_calls)
                parsed.tool_calls.append(rec)
                if call_id is not None:
                    pending_index_by_id[call_id] = idx

        elif ev_type == "tool_result":
            call_id_raw = event.get("id") or event.get("callId") or event.get("toolCallId")
            call_id = str(call_id_raw) if call_id_raw is not None else None
            is_err_raw = event.get("isError")
            if is_err_raw is None:
                is_err_raw = event.get("is_error")
            is_err = is_err_raw if isinstance(is_err_raw, bool) else None
            idx = pending_index_by_id.get(call_id) if call_id is not None else None
            if idx is None and parsed.tool_calls and parsed.tool_calls[-1].is_error is None:
                idx = len(parsed.tool_calls) - 1
            if idx is not None and is_err is not None:
                old = parsed.tool_calls[idx]
                parsed.tool_calls[idx] = ToolCallRecord(
                    sequence=old.sequence,
                    call_id=old.call_id,
                    name=old.name,
                    arguments=old.arguments,
                    is_error=is_err,
                )

        elif ev_type == "message" and event.get("role") == "assistant":
            text = _extract_message_text(event)
            if text is not None:
                fallback_assistant_text = text

        elif ev_type == "completion":
            parsed.saw_completion = True
            final_text = event.get("finalText")
            if isinstance(final_text, str):
                parsed.last_reply = final_text
            parsed.num_turns = _opt_int(event.get("numTurns"))
            parsed.duration_ms = _opt_float(event.get("durationMs"))
            parsed.tokens = TokenUsage.from_harness_usage(event.get("usage"))

        elif ev_type == "result":
            parsed.saw_completion = True
            is_err = bool(event.get("is_error")) or event.get("subtype") == "failure"
            parsed.terminal_is_error = is_err
            parsed.num_turns = _opt_int(event.get("num_turns"))
            parsed.duration_ms = _opt_float(event.get("duration_ms"))
            pre_call = is_err and (parsed.num_turns in (None, 0))
            parsed.tokens = TokenUsage.from_harness_usage(
                event.get("usage"), pre_call_failure=pre_call
            )
            result_text = event.get("result")
            if isinstance(result_text, str):
                if is_err:
                    parsed.error_text = result_text
                else:
                    parsed.last_reply = result_text

        elif ev_type == "error":
            parsed.terminal_is_error = True
            msg = event.get("message") or event.get("error") or event.get("detail")
            if isinstance(msg, str) and msg:
                parsed.error_text = msg

    if parsed.last_reply is None and not parsed.terminal_is_error:
        parsed.last_reply = fallback_assistant_text
    return parsed


def _locate_droid_transcript(
    harness_home: Path,
    clone_dir: Path,
    session_id: str | None,
) -> Path | None:
    if not session_id:
        return None
    sessions_root = harness_home / ".factory" / "sessions"
    if not sessions_root.is_dir():
        return None
    sanitized_cwd = str(clone_dir.resolve()).replace("/", "-")
    direct = sessions_root / sanitized_cwd / f"{session_id}.jsonl"
    if direct.is_file():
        return direct
    for match in sorted(sessions_root.rglob(f"{session_id}.jsonl")):
        if match.is_file():
            return match
    return None


def _terminate_process_group(proc: subprocess.Popen[str], grace_s: float) -> None:
    """Stop a process group cleanly with SIGTERM, escalating to SIGKILL after ``grace_s``."""
    try:
        pgid = os.getpgid(proc.pid)
    except OSError:
        pgid = None

    if pgid is not None:
        with contextlib.suppress(OSError):
            os.killpg(pgid, signal.SIGTERM)
    else:
        with contextlib.suppress(OSError):
            proc.terminate()

    try:
        proc.wait(timeout=max(0.1, grace_s))
        return
    except subprocess.TimeoutExpired:
        pass

    if pgid is not None:
        with contextlib.suppress(OSError):
            os.killpg(pgid, signal.SIGKILL)
    else:
        with contextlib.suppress(OSError):
            proc.kill()
    with contextlib.suppress(subprocess.TimeoutExpired):
        proc.wait(timeout=5.0)


class DroidHarnessAdapter:
    """Runs headless ``droid exec`` isolated per the MEM-17 findings."""

    name = "droid"

    def __init__(
        self,
        *,
        droid_bin: str = "droid",
        auth_source_dir: Path | None = None,
        mcp_command: Sequence[str] | None = None,
    ) -> None:
        self.droid_bin = droid_bin
        self.auth_source_dir = auth_source_dir
        self.mcp_command = (
            tuple(mcp_command)
            if mcp_command is not None
            else (sys.executable, "-m", "agent_memory.mcp_server")
        )

    def version(self) -> str | None:
        return self._detect_version()

    def _detect_version(self) -> str | None:
        try:
            proc = subprocess.run(
                [self.droid_bin, "--version"],
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=10.0,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        if proc.returncode != 0:
            return None
        out = proc.stdout.strip()
        return out.splitlines()[0].strip() if out else None

    def _prepare_factory_home(
        self,
        invocation: HarnessInvocation,
    ) -> tuple[Path, Path, Path]:
        factory_dir = invocation.harness_home / ".factory"
        private_dir(factory_dir)

        # Link auth files only when FACTORY_API_KEY is not set; never link user
        # hooks.json, settings.json, mcp.json, skills/, or droids/.
        if not os.environ.get("FACTORY_API_KEY"):
            auth_dir = (
                self.auth_source_dir
                if self.auth_source_dir is not None
                else Path.home() / ".factory"
            )
            if auth_dir.is_dir():
                for filename in _DROID_AUTH_FILES:
                    src = auth_dir / filename
                    dst = factory_dir / filename
                    if src.exists() and not dst.exists():
                        with contextlib.suppress(OSError):
                            dst.symlink_to(src)

        settings_payload: dict[str, Any] = {
            "hooks": {"hooksDisabled": True},
            "cloudSessionSync": False,
        }
        settings_path = factory_dir / "settings.json"
        settings_path.write_text(
            json.dumps(settings_payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        mcp_servers: dict[str, Any] = {}
        if invocation.memory_endpoint is not None:
            ep = invocation.memory_endpoint
            mcp_env: dict[str, str] = {
                "AGENT_MEMORY_HOME": str(ep.home),
                "AGENT_MEMORY_PORT": str(ep.port),
                "AGENT_MEMORY_HOST": ep.host,
                "AGENT_MEMORY_AGENT": ep.agent_id,
                "AGENT_MEMORY_PROJECT": ep.project,
            }
            if ep.wording_path is not None:
                mcp_env[MCP_WORDING_ENV] = str(ep.wording_path)
            mcp_servers[ep.server_name] = {
                "type": "stdio",
                "command": self.mcp_command[0],
                "args": list(self.mcp_command[1:]),
                "env": mcp_env,
            }
        mcp_config_path = factory_dir / "mcp.json"
        mcp_config_path.write_text(
            json.dumps({"mcpServers": mcp_servers}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        prompt_path = invocation.harness_home / "prompt.md"
        prompt_path.write_text(invocation.task.request, encoding="utf-8")
        return settings_path, mcp_config_path, prompt_path

    def _build_env(self, invocation: HarnessInvocation) -> dict[str, str]:
        env = os.environ.copy()
        for key in _STRIPPED_HARNESS_ENV:
            env.pop(key, None)
        env["FACTORY_HOME_OVERRIDE"] = str(invocation.harness_home)
        env["FACTORY_DROID_AUTO_UPDATE_ENABLED"] = "false"
        return env

    def _build_command(
        self,
        invocation: HarnessInvocation,
        *,
        settings_path: Path,
        prompt_path: Path,
    ) -> list[str]:
        cmd = [
            self.droid_bin,
            "exec",
            "-o",
            "stream-json",
            "--cwd",
            str(invocation.clone_dir),
            "--auto",
            invocation.config.auto_level,
            "--settings",
            str(settings_path),
            "--disable-builtin-skills",
            "--only-tools",
            ",".join(invocation.allowed_tools),
            "-m",
            invocation.config.model,
        ]
        if invocation.config.reasoning_effort is not None:
            cmd.extend(["-r", invocation.config.reasoning_effort])
        cmd.extend(["-f", str(prompt_path)])
        return cmd

    def run(self, invocation: HarnessInvocation) -> HarnessResult:
        settings_path, mcp_config_path, prompt_path = self._prepare_factory_home(invocation)
        version = self._detect_version()
        cmd = self._build_command(invocation, settings_path=settings_path, prompt_path=prompt_path)
        env = self._build_env(invocation)
        raw_events_path = invocation.harness_home / "stream.ndjson"

        try:
            proc = subprocess.Popen(
                cmd,
                cwd=str(invocation.clone_dir),
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                start_new_session=True,
            )
        except OSError as exc:
            return HarnessResult(
                status="failed",
                last_reply=None,
                failure_category="environment_error",
                failure_detail=f"failed to launch {self.droid_bin!r}: {exc}",
                settings_path=settings_path,
                mcp_config_path=mcp_config_path,
                harness_version=version,
            )

        stdout_lines: list[str] = []
        stderr_chunks: list[str] = []
        stdout_pipe = proc.stdout
        stderr_pipe = proc.stderr

        def _drain_stdout() -> None:
            if stdout_pipe is None:
                return
            with raw_events_path.open("w", encoding="utf-8") as raw_fh:
                for line in stdout_pipe:
                    raw_fh.write(line)
                    raw_fh.flush()
                    stdout_lines.append(line)

        def _drain_stderr() -> None:
            if stderr_pipe is None:
                return
            for chunk in iter(lambda: stderr_pipe.read(4096), ""):
                if len(stderr_chunks) < 64:
                    stderr_chunks.append(chunk)

        t_out = threading.Thread(target=_drain_stdout, daemon=True)
        t_err = threading.Thread(target=_drain_stderr, daemon=True)
        t_out.start()
        t_err.start()

        timed_out = False
        interrupted = False
        try:
            proc.wait(timeout=invocation.config.wall_time_limit_s)
        except subprocess.TimeoutExpired:
            timed_out = True
            _terminate_process_group(proc, invocation.config.kill_grace_s)
        except KeyboardInterrupt:
            interrupted = True
            _terminate_process_group(proc, invocation.config.kill_grace_s)

        t_out.join(timeout=5.0)
        t_err.join(timeout=5.0)

        stderr_text = "".join(stderr_chunks).strip()
        parsed = parse_droid_stream_events(stdout_lines)
        transcript_path = _locate_droid_transcript(
            invocation.harness_home, invocation.clone_dir, parsed.session_id
        )

        if timed_out:
            return HarnessResult(
                status="failed",
                last_reply=parsed.last_reply,
                tool_calls=tuple(parsed.tool_calls),
                tokens=parsed.tokens,
                num_turns=parsed.num_turns,
                failure_category="timeout",
                failure_detail=(
                    f"wall-time limit exceeded ({invocation.config.wall_time_limit_s:.1f}s)"
                ),
                exit_code=proc.returncode,
                harness_session_id=parsed.session_id,
                harness_reported_duration_ms=parsed.duration_ms,
                transcript_path=transcript_path,
                settings_path=settings_path,
                mcp_config_path=mcp_config_path,
                harness_version=version,
                resolved_model=parsed.model,
                resolved_reasoning_effort=parsed.reasoning_effort,
                raw_events_path=raw_events_path,
            )

        if interrupted:
            return HarnessResult(
                status="failed",
                last_reply=parsed.last_reply,
                tool_calls=tuple(parsed.tool_calls),
                tokens=parsed.tokens,
                num_turns=parsed.num_turns,
                failure_category="interrupted",
                failure_detail="trial interrupted by signal",
                exit_code=proc.returncode,
                harness_session_id=parsed.session_id,
                harness_reported_duration_ms=parsed.duration_ms,
                transcript_path=transcript_path,
                settings_path=settings_path,
                mcp_config_path=mcp_config_path,
                harness_version=version,
                resolved_model=parsed.model,
                resolved_reasoning_effort=parsed.reasoning_effort,
                raw_events_path=raw_events_path,
            )

        succeeded = proc.returncode == 0 and not parsed.terminal_is_error
        if succeeded:
            return HarnessResult(
                status="completed",
                last_reply=parsed.last_reply,
                tool_calls=tuple(parsed.tool_calls),
                tokens=parsed.tokens,
                num_turns=parsed.num_turns,
                failure_category=None,
                failure_detail=None,
                exit_code=proc.returncode,
                harness_session_id=parsed.session_id,
                harness_reported_duration_ms=parsed.duration_ms,
                transcript_path=transcript_path,
                settings_path=settings_path,
                mcp_config_path=mcp_config_path,
                harness_version=version,
                resolved_model=parsed.model,
                resolved_reasoning_effort=parsed.reasoning_effort,
                raw_events_path=raw_events_path,
            )

        detail = parsed.error_text or stderr_text or f"droid exited with code {proc.returncode}"
        category = _classify_droid_failure(proc.returncode, parsed.error_text, stderr_text)
        return HarnessResult(
            status="failed",
            last_reply=parsed.last_reply,
            tool_calls=tuple(parsed.tool_calls),
            tokens=parsed.tokens,
            num_turns=parsed.num_turns,
            failure_category=category,
            failure_detail=detail[:1000],
            exit_code=proc.returncode,
            harness_session_id=parsed.session_id,
            harness_reported_duration_ms=parsed.duration_ms,
            transcript_path=transcript_path,
            settings_path=settings_path,
            mcp_config_path=mcp_config_path,
            harness_version=version,
            resolved_model=parsed.model,
            resolved_reasoning_effort=parsed.reasoning_effort,
            raw_events_path=raw_events_path,
        )


# ---------------------------------------------------------------------------
# Scripted stand-in adapter for tests (never calls a model)
# ---------------------------------------------------------------------------


class ScriptedHarnessAdapter:
    """In-process scripted stand-in implementing ``HarnessAdapter`` for tests."""

    name = "scripted"

    def __init__(
        self,
        handler: Callable[[HarnessInvocation], HarnessResult],
        *,
        version: str = "scripted-1",
    ) -> None:
        self._handler = handler
        self._version = version
        self.invocations: list[HarnessInvocation] = []

    def version(self) -> str | None:
        return self._version

    def run(self, invocation: HarnessInvocation) -> HarnessResult:
        self.invocations.append(invocation)
        result = self._handler(invocation)
        if result.harness_version is None:
            return HarnessResult(
                status=result.status,
                last_reply=result.last_reply,
                tool_calls=result.tool_calls,
                tokens=result.tokens,
                num_turns=result.num_turns,
                failure_category=result.failure_category,
                failure_detail=result.failure_detail,
                exit_code=result.exit_code,
                harness_session_id=result.harness_session_id,
                harness_reported_duration_ms=result.harness_reported_duration_ms,
                transcript_path=result.transcript_path,
                settings_path=result.settings_path,
                mcp_config_path=result.mcp_config_path,
                harness_version=self._version,
                resolved_model=result.resolved_model,
                resolved_reasoning_effort=result.resolved_reasoning_effort,
                raw_events_path=result.raw_events_path,
            )
        return result


# ---------------------------------------------------------------------------
# Trial runner orchestration
# ---------------------------------------------------------------------------


def _allowed_tools_for_arm(config: TrialConfig) -> tuple[str, ...]:
    base = tuple(tool for tool in config.base_tools if tool)
    if config.arm == "memory":
        extras = [t for t in ("ToolSearch", f"MCP:{config.memory_server_name}") if t not in base]
        return (*base, *extras)
    return base


def _validate_trial_config(config: TrialConfig) -> None:
    if config.arm not in VALID_ARMS:
        raise UsefulnessError(f"invalid arm {config.arm!r}: expected one of {VALID_ARMS}")
    if config.auto_level not in VALID_AUTO_LEVELS:
        raise UsefulnessError(
            f"invalid auto_level {config.auto_level!r}: expected one of {VALID_AUTO_LEVELS}"
        )
    if not config.model.strip():
        raise UsefulnessError("model must be a non-empty string")
    if config.wall_time_limit_s <= 0:
        raise UsefulnessError("wall_time_limit_s must be positive")
    if config.kill_grace_s <= 0:
        raise UsefulnessError("kill_grace_s must be positive")
    if not config.base_tools:
        raise UsefulnessError("base_tools must be non-empty")
    if config.trial_id is not None and not _TRIAL_ID_RE.fullmatch(config.trial_id):
        raise UsefulnessError(
            f"bad trial_id {config.trial_id!r}: letters, digits, dot, dash, underscore only"
        )
    if not config.memory_server_name.strip():
        raise UsefulnessError("memory_server_name must be a non-empty string")


def _resolve_task_for_trial(
    store: UsefulnessRunStore,
    task_or_dir: TaskSpec | Path,
    *,
    verify_manifest: bool,
) -> tuple[TaskSpec, str | None]:
    has_registration = store.registration_path.exists()
    effective_verify = verify_manifest or has_registration
    if has_registration and not store.task_manifest_path.exists():
        raise UsefulnessError(
            f"cannot run trial under registration without task manifest: {store.task_manifest_path}"
        )

    if isinstance(task_or_dir, TaskSpec):
        task = task_or_dir
        if effective_verify and store.task_manifest_path.exists():
            from .tasks import verify_task_against_manifest

            split = verify_task_against_manifest(task, store.read_task_manifest())
            return task, split
    else:
        task_dir = task_or_dir
        if effective_verify and store.task_manifest_path.exists():
            task, split = load_verified_task(task_dir, store.read_task_manifest())
            return task, split
        task = load_task(task_dir)

    if store.task_manifest_path.exists():
        manifest = store.read_task_manifest()
        held_out = (manifest.get("split") or {}).get("held_out") or ()
        if task.task_id in held_out:
            return task, "held_out"
    return task, None


def _resolve_optional_wording(
    config: TrialConfig,
) -> tuple[str, dict[str, str], str] | None:
    if config.memory_tool_wording is not None and config.memory_tool_wording_file is not None:
        raise UsefulnessError(
            "specify at most one of memory_tool_wording or memory_tool_wording_file"
        )
    if config.memory_tool_wording_file is not None:
        try:
            instr, descs = load_tool_wording_file(config.memory_tool_wording_file)
        except ConfigError as exc:
            raise UsefulnessError(str(exc)) from None
        return instr, descs, memory_tool_wording_sha256(instr, descs)
    if config.memory_tool_wording is not None:
        try:
            instr, descs = validate_tool_wording(config.memory_tool_wording)
        except ConfigError as exc:
            raise UsefulnessError(str(exc)) from None
        return instr, descs, memory_tool_wording_sha256(instr, descs)
    return None


def _adapter_version(adapter: HarnessAdapter) -> str | None:
    version_fn = getattr(adapter, "version", None)
    if callable(version_fn):
        ver = version_fn()
        return ver if isinstance(ver, str) else None
    return None


def _generate_trial_id(task_id: str, arm: Arm) -> str:
    return f"{task_id}-{arm}-{uuid.uuid4().hex[:12]}"


def resolve_trial_registration(
    experiment_home: Config | Path,
    run_id: str,
    task_id: str,
    task_split: str | None,
    config: TrialConfig,
    adapter: HarnessAdapter,
) -> tuple[str | None, tuple[str, dict[str, str], str] | None, Path | None]:
    """Enforce the held-out gate and pre-trial registration check.

    Returns ``(registration_sha256, effective_wording, effective_source_home)``.
    Raises ``UsefulnessError`` before any trial directory or schedule event is written
    when a held-out task has no registration or when a frozen registration does not
    match the live trial configuration.
    """
    store = UsefulnessRunStore(experiment_home, run_id)
    registration_sha256: str | None = None
    effective_source_home: Path | None = config.source_home
    effective_wording: tuple[str, dict[str, str], str] | None = None

    if task_split == "held_out" and not store.registration_path.exists():
        raise UsefulnessError(
            f"cannot start held-out trial for task {task_id!r} in run {run_id!r}: "
            f"no registration at {store.registration_path}"
        )

    if store.registration_path.exists():
        live_harness_version = (
            config.harness_version
            if config.harness_version is not None
            else _adapter_version(adapter)
        )
        live_repo_commit = (
            config.repo_commit if config.repo_commit is not None else detect_repo_commit()
        )
        live_hardware_class = (
            config.hardware_class if config.hardware_class is not None else detect_hardware_class()
        )
        verified_reg = verify_registration_for_trial(
            experiment_home,
            run_id,
            harness=adapter.name,
            harness_version=live_harness_version,
            repo_commit=live_repo_commit,
            hardware_class=live_hardware_class,
            allowed_base_tools=config.base_tools,
            memory_server_name=config.memory_server_name,
            config_overrides=default_trial_config_overrides(config.auto_level),
            model=config.model,
            reasoning_effort=config.reasoning_effort,
            wall_time_limit_s=config.wall_time_limit_s,
            arm=config.arm,
            source_home=config.source_home,
            wording_override=config.memory_tool_wording,
            wording_file=config.memory_tool_wording_file,
        )
        registration_sha256 = verified_reg.registration_sha256
        effective_wording = (
            verified_reg.memory_tools_instructions,
            verified_reg.memory_tools_descriptions,
            verified_reg.memory_tools_sha256,
        )
        if config.arm == "memory" and effective_source_home is None:
            effective_source_home = verified_reg.memory_set_dir
    else:
        effective_wording = _resolve_optional_wording(config)

    return registration_sha256, effective_wording, effective_source_home


def run_trial(
    experiment_home: Config | Path,
    run_id: str,
    task_or_dir: TaskSpec | Path,
    config: TrialConfig,
    *,
    adapter: HarnessAdapter | None = None,
) -> TrialRecord:
    """Run one trial of ``task_or_dir`` in ``config.arm`` and write its immutable record.

    Enforces:
    - Refusal of the default production home (``ensure_experiment_home``).
    - Egress opt-in for any model not on this machine (``check_trial_egress``).
    - Registration presence for held-out tasks, plus verification of task
      manifest, memory set, environment manifest, and memory-tool wording
      whenever ``registration.json`` exists in the run.
    - Fresh git clone at ``task.commit`` with hidden tests verified absent and
      repository-configured memory MCP servers stripped in both arms.
    - Isolated per-trial ``MemoryServer`` and trial home in the memory arm only,
      with ``AGENT_MEMORY_PROJECT=<task.project>`` and any registered MCP tool
      wording injected via ``AGENT_MEMORY_MCP_WORDING_FILE``.
    - Wall-time enforcement and retained trial records on failure or timeout.
    """
    _validate_trial_config(config)
    store = UsefulnessRunStore(experiment_home, run_id)
    egress_mode = check_trial_egress(config.model, config.model_base_url)
    task, task_split = _resolve_task_for_trial(
        store, task_or_dir, verify_manifest=config.verify_manifest
    )

    active_adapter: HarnessAdapter = adapter if adapter is not None else DroidHarnessAdapter()
    allowed_tools = _allowed_tools_for_arm(config)

    registration_sha256, effective_wording, effective_source_home = resolve_trial_registration(
        experiment_home,
        run_id,
        task.task_id,
        task_split,
        config,
        active_adapter,
    )

    # Validate source_home early (including refusing DEFAULT_HOME) before creating trial dirs.
    if effective_source_home is not None:
        from .store import ensure_experiment_home

        resolved_src = ensure_experiment_home(effective_source_home)
        if config.arm == "memory":
            if not (resolved_src / "memory.duckdb").is_file():
                raise UsefulnessError(
                    f"source home has no database file: {resolved_src / 'memory.duckdb'}"
                )
            if (resolved_src / "memory.duckdb.wal").exists():
                raise UsefulnessError(
                    f"source home has a WAL file ({resolved_src / 'memory.duckdb.wal'}); "
                    "stop its server before copying into a trial"
                )

    trial_id = config.trial_id or _generate_trial_id(task.task_id, config.arm)
    trial_dir = store.trials_dir / trial_id
    if trial_dir.exists():
        raise UsefulnessError(f"trial directory already exists: {trial_dir}")

    private_dir(trial_dir)
    clone_dir = trial_dir / "repo"
    harness_home = trial_dir / "harness-home"
    private_dir(harness_home)

    started_at = utc_now_iso()
    t_total_start = time.monotonic()

    # 1. Clone repository at task.commit and check hidden tests absent.
    t_clone_start = time.monotonic()
    base_commit = _clone_task_repo(task, clone_dir)
    sanitized_files = _sanitize_repo_mcp_configs(clone_dir)
    clone_s = time.monotonic() - t_clone_start

    # 2. Setup trial home & dedicated MemoryServer (memory arm only).
    t_setup_start = time.monotonic()
    server: MemoryServer | None = None
    trial_cfg: Config | None = None
    trial_home_path: Path | None = None
    trial_baseline: TrialLedgerBaseline | None = None
    memory_endpoint: MemoryEndpoint | None = None
    wording_path: Path | None = None
    wording_sha256: str | None = None
    setup_error: str | None = None

    try:
        if config.arm == "memory":
            if effective_wording is not None:
                w_instr, w_descs, w_sha = effective_wording
                wording_path = harness_home / "memory-tool-wording.json"
                write_once_json(
                    wording_path,
                    {"instructions": w_instr, "descriptions": w_descs},
                )
                wording_sha256 = w_sha
            trial_home_path = trial_dir / "trial-home"
            port = free_port()
            trial_cfg = _prepare_trial_home(trial_home_path, effective_source_home, port)
            server = MemoryServer(trial_cfg)
            server.start()
            baseline_client = MemoryClient.from_config(trial_cfg)
            try:
                trial_baseline = capture_trial_baseline(baseline_client)
            finally:
                baseline_client.close()
            memory_endpoint = MemoryEndpoint(
                home=trial_cfg.home,
                host=trial_cfg.host,
                port=trial_cfg.port,
                project=task.project,
                agent_id=DEFAULT_MEMORY_AGENT_ID,
                server_name=config.memory_server_name,
                wording_path=wording_path,
                wording_sha256=wording_sha256,
            )
    except Exception as exc:
        if server is not None:
            with contextlib.suppress(Exception):
                server.stop()
            server = None
        setup_error = str(exc)
    setup_s = time.monotonic() - t_setup_start

    # 3. Invoke harness adapter (unless setup failed).
    t_harness_start = time.monotonic()
    if setup_error is not None:
        harness_result = HarnessResult(
            status="failed",
            last_reply=None,
            failure_category="environment_error",
            failure_detail=f"trial setup failed: {setup_error}",
        )
    else:
        invocation = HarnessInvocation(
            trial_id=trial_id,
            arm=config.arm,
            task=task,
            clone_dir=clone_dir,
            harness_home=harness_home,
            config=config,
            memory_endpoint=memory_endpoint,
            allowed_tools=allowed_tools,
        )
        try:
            harness_result = active_adapter.run(invocation)
        except KeyboardInterrupt:
            harness_result = HarnessResult(
                status="failed",
                last_reply=None,
                failure_category="interrupted",
                failure_detail="trial interrupted during harness execution",
            )
        except Exception as exc:
            harness_result = HarnessResult(
                status="failed",
                last_reply=None,
                failure_category="environment_error",
                failure_detail=f"harness adapter raised {type(exc).__name__}: {exc}",
            )
    harness_s = time.monotonic() - t_harness_start

    # 4. Teardown: import transcript & account recalls (memory arm), stop server, capture diff.
    t_teardown_start = time.monotonic()
    recall_accounting: RecallAccounting | None = None
    try:
        if config.arm == "memory" and server is not None and trial_cfg is not None:
            transcript_harness = harness_result.transcript_harness or active_adapter.name
            import_summary, parsed_records = import_trial_transcript(
                trial_cfg,
                harness_result.transcript_path,
                harness=transcript_harness,
                memory_server_name=config.memory_server_name,
            )
            accounting_client = MemoryClient.from_config(trial_cfg)
            try:
                recall_accounting = account_trial_recall(
                    accounting_client,
                    transcript_path=harness_result.transcript_path,
                    transcript_records=parsed_records,
                    harness=transcript_harness,
                    memory_server_name=config.memory_server_name,
                    baseline=trial_baseline,
                    transcript_import=import_summary,
                )
            finally:
                accounting_client.close()
    finally:
        if server is not None:
            with contextlib.suppress(Exception):
                server.stop()
    _restore_untouched_sanitized_files(sanitized_files)
    repo_changes = _capture_repo_changes(clone_dir, base_commit)
    teardown_s = time.monotonic() - t_teardown_start
    total_s = time.monotonic() - t_total_start
    finished_at = utc_now_iso()

    wall_time = WallTimeBreakdown(
        clone_s=round(clone_s, 6),
        setup_s=round(setup_s, 6),
        harness_s=round(harness_s, 6),
        teardown_s=round(teardown_s, 6),
        total_s=round(total_s, 6),
        harness_reported_duration_ms=harness_result.harness_reported_duration_ms,
    )

    failure_category = harness_result.failure_category
    if harness_result.status == "failed" and failure_category is None:
        failure_category = "harness_error"
    if harness_result.status == "completed":
        failure_category = None

    paths: dict[str, str | None] = {
        "trial_dir": str(trial_dir),
        "clone_dir": str(clone_dir),
        "harness_home": str(harness_home),
        "trial_home": str(trial_home_path) if trial_home_path is not None else None,
        "transcript_path": (
            str(harness_result.transcript_path)
            if harness_result.transcript_path is not None
            else None
        ),
        "settings_path": (
            str(harness_result.settings_path) if harness_result.settings_path is not None else None
        ),
        "mcp_config_path": (
            str(harness_result.mcp_config_path)
            if harness_result.mcp_config_path is not None
            else None
        ),
        "raw_events_path": (
            str(harness_result.raw_events_path)
            if harness_result.raw_events_path is not None
            else None
        ),
        "memory_tool_wording_path": str(wording_path) if wording_path is not None else None,
    }

    environment: dict[str, Any] = {
        "allowed_tools": list(allowed_tools),
        "auto_level": config.auto_level,
        "hooks_disabled": True,
        "cloud_session_sync": False,
        "builtin_skills_disabled": True,
        "wall_time_limit_s": config.wall_time_limit_s,
        "model_base_url": config.model_base_url,
        "egress_mode": egress_mode,
        "project": task.project if config.arm == "memory" else None,
        "memory_server_name": config.memory_server_name if config.arm == "memory" else None,
        "memory_tools_wording_sha256": wording_sha256 if config.arm == "memory" else None,
        "sanitized_repo_mcp_files": [
            {"path": item.rel_path, "removed_servers": list(item.removed_servers)}
            for item in sanitized_files
        ],
    }

    record_path = trial_dir / "record.json"
    record_body: dict[str, Any] = {
        "schema_version": TRIAL_RECORD_SCHEMA_VERSION,
        "trial_id": trial_id,
        "run_id": run_id,
        "task_id": task.task_id,
        "task_sha256": task.task_sha256,
        "registration_sha256": registration_sha256,
        "arm": config.arm,
        "status": harness_result.status,
        "failure_category": failure_category,
        "failure_detail": harness_result.failure_detail,
        "exit_code": harness_result.exit_code,
        "harness": active_adapter.name,
        "harness_version": harness_result.harness_version,
        "harness_session_id": harness_result.harness_session_id,
        "model": harness_result.resolved_model or config.model,
        "reasoning_effort": (
            harness_result.resolved_reasoning_effort
            if harness_result.resolved_reasoning_effort is not None
            else config.reasoning_effort
        ),
        "num_turns": harness_result.num_turns,
        "last_reply": harness_result.last_reply,
        "repo_changes": repo_changes.to_dict(),
        "wall_time": wall_time.to_dict(),
        "tool_calls": [tc.to_dict() for tc in harness_result.tool_calls],
        "tokens": harness_result.tokens.to_dict(),
        "recall_accounting": (
            recall_accounting.to_dict() if recall_accounting is not None else None
        ),
        "started_at": started_at,
        "finished_at": finished_at,
        "paths": paths,
        "environment": environment,
    }
    record_sha256 = write_once_json(record_path, record_body)

    return TrialRecord(
        schema_version=TRIAL_RECORD_SCHEMA_VERSION,
        trial_id=trial_id,
        run_id=run_id,
        task_id=task.task_id,
        task_sha256=task.task_sha256,
        registration_sha256=registration_sha256,
        arm=config.arm,
        status=harness_result.status,
        failure_category=failure_category,
        failure_detail=harness_result.failure_detail,
        exit_code=harness_result.exit_code,
        harness=active_adapter.name,
        harness_version=harness_result.harness_version,
        harness_session_id=harness_result.harness_session_id,
        model=harness_result.resolved_model or config.model,
        reasoning_effort=(
            harness_result.resolved_reasoning_effort
            if harness_result.resolved_reasoning_effort is not None
            else config.reasoning_effort
        ),
        num_turns=harness_result.num_turns,
        last_reply=harness_result.last_reply,
        repo_changes=repo_changes,
        wall_time=wall_time,
        tool_calls=harness_result.tool_calls,
        tokens=harness_result.tokens,
        started_at=started_at,
        finished_at=finished_at,
        paths=paths,
        environment=environment,
        record_path=record_path,
        record_sha256=record_sha256,
        recall_accounting=recall_accounting,
    )
