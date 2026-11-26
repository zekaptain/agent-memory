"""Persistent NDJSON stdio bridge (`mem bridge`) and session-start context CLI (`mem context`)."""

from __future__ import annotations

import contextlib
import json
import math
import os
import sys
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any, TextIO

from .capture.session import Observed
from .config import Config, ConfigError
from .lifecycle import HUMAN_AGENT, LifecycleError
from .mcp_server import ranking
from .mcp_server.tools import DEFAULT_LIMIT, MemoryTools, ToolError

DEFAULT_BRIDGE_AGENT = "oh-my-pi"
DEFAULT_CACHE_SIZE = 32
BRIDGE = Observed("mem-bridge", "record", "mem_bridge_v1", ())
CONTEXT = Observed("mem context", "record", "mem_cli_v1")
SUPPORTED_OPS = frozenset({"recall", "remember", "cite", "review", "status", "session_context"})

_RECALL_KEYS = frozenset({"query", "scopes", "limit", "types"})
_REMEMBER_KEYS = frozenset(
    {
        "content",
        "type",
        "scope",
        "key",
        "confidence",
        "importance",
        "trust_level",
        "supersedes",
        "source_event_ids",
        "valid_from",
        "expires_at",
        "valid_to",
    }
)
_CITE_KEYS = frozenset({"retrieval_id", "memory_ids", "context_event_ids", "answer_event_id"})
_REVIEW_KEYS = frozenset({"action", "memory_id", "reason", "related_memory_id"})
_STATUS_KEYS: frozenset[str] = frozenset()
_SESSION_CONTEXT_KEYS = frozenset({"scopes", "max_items", "max_chars", "types"})


class _BridgeArgError(ValueError):
    """Bad envelope or operation arguments."""


class _UnknownOpError(ValueError):
    """Unsupported operation name."""


def resolve_bridge_agent(agent_id: str | None = None) -> str:
    agent = (
        agent_id
        if agent_id is not None
        else (os.environ.get("AGENT_MEMORY_AGENT") or DEFAULT_BRIDGE_AGENT)
    )
    if not agent or not agent.strip():
        raise ConfigError("agent_id must be a non-empty registered agent")
    if agent == HUMAN_AGENT:
        raise ConfigError(
            f"agent_id={HUMAN_AGENT} is the human at the mem CLI; "
            "name the harness (for example oh-my-pi)."
        )
    return agent


def _is_valid_id(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, str):
        return True
    if isinstance(value, int):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    return False


def _check_allowed_keys(args: Mapping[str, Any], allowed: frozenset[str], op: str) -> None:
    extra = sorted(set(args) - allowed)
    if extra:
        raise _BridgeArgError(f"unexpected arguments for {op}: {', '.join(extra)}")


def _req_str(args: Mapping[str, Any], key: str) -> str:
    if key not in args or not isinstance(args[key], str):
        raise _BridgeArgError(f"{key} must be a string")
    return args[key]


def _opt_str(args: Mapping[str, Any], key: str) -> str | None:
    value = args.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise _BridgeArgError(f"{key} must be a string or null")
    return value


def _opt_str_default(args: Mapping[str, Any], key: str, default: str) -> str:
    value = args.get(key)
    if value is None:
        return default
    if not isinstance(value, str):
        raise _BridgeArgError(f"{key} must be a string")
    return value


def _opt_str_list(args: Mapping[str, Any], key: str) -> list[str] | None:
    value = args.get(key)
    if value is None:
        return None
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise _BridgeArgError(f"{key} must be an array of strings or null")
    return list(value)


def _opt_int(args: Mapping[str, Any], key: str, default: int) -> int:
    value = args.get(key)
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int):
        raise _BridgeArgError(f"{key} must be an integer")
    return value


def _opt_float(args: Mapping[str, Any], key: str, default: float) -> float:
    value = args.get(key)
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _BridgeArgError(f"{key} must be a number")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise _BridgeArgError(f"{key} must be a finite number")
    return numeric


def _dispatch(tools: MemoryTools, op: str, args: Mapping[str, Any]) -> dict[str, Any]:
    if op == "recall":
        _check_allowed_keys(args, _RECALL_KEYS, op)
        return tools.recall(
            query=_req_str(args, "query"),
            scopes=_opt_str_list(args, "scopes"),
            limit=_opt_int(args, "limit", DEFAULT_LIMIT),
            types=_opt_str_list(args, "types"),
        )
    if op == "remember":
        _check_allowed_keys(args, _REMEMBER_KEYS, op)
        return tools.remember(
            content=_req_str(args, "content"),
            memory_type=_req_str(args, "type"),
            scope=_opt_str(args, "scope"),
            key=_opt_str(args, "key"),
            confidence=_opt_float(args, "confidence", 0.7),
            importance=_opt_float(args, "importance", 0.5),
            trust_level=_opt_str_default(args, "trust_level", "agent"),
            supersedes=_opt_str(args, "supersedes"),
            source_event_ids=_opt_str_list(args, "source_event_ids"),
            valid_from=_opt_str(args, "valid_from"),
            expires_at=_opt_str(args, "expires_at"),
            valid_to=_opt_str(args, "valid_to"),
        )
    if op == "cite":
        _check_allowed_keys(args, _CITE_KEYS, op)
        return tools.cite(
            retrieval_id=_opt_str(args, "retrieval_id"),
            memory_ids=_opt_str_list(args, "memory_ids"),
            context_event_ids=_opt_str_list(args, "context_event_ids"),
            answer_event_id=_opt_str(args, "answer_event_id"),
        )
    if op == "review":
        _check_allowed_keys(args, _REVIEW_KEYS, op)
        return tools.review(
            action=_req_str(args, "action"),
            memory_id=_req_str(args, "memory_id"),
            reason=_req_str(args, "reason"),
            related_memory_id=_opt_str(args, "related_memory_id"),
        )
    if op == "status":
        _check_allowed_keys(args, _STATUS_KEYS, op)
        return tools.status()
    if op == "session_context":
        _check_allowed_keys(args, _SESSION_CONTEXT_KEYS, op)
        return tools.session_context(
            scopes=_opt_str_list(args, "scopes"),
            max_items=_opt_int(args, "max_items", ranking.SESSION_START_DEFAULT_MAX_ITEMS),
            max_chars=_opt_int(args, "max_chars", ranking.SESSION_START_DEFAULT_MAX_CHARS),
            types=_opt_str_list(args, "types"),
        )
    raise _UnknownOpError(f"unsupported op: {op}")


def _error_response(req_id: Any, code: str, message: str) -> dict[str, Any]:
    return {"id": req_id, "ok": False, "error": {"code": code, "message": message}}


class BridgeServer:
    """Stateful NDJSON request handler with an LRU cache of MemoryTools instances."""

    def __init__(
        self,
        cfg: Config,
        agent_id: str,
        *,
        cache_size: int = DEFAULT_CACHE_SIZE,
        stderr: TextIO | None = None,
    ) -> None:
        self.cfg = cfg
        self.agent_id = resolve_bridge_agent(agent_id)
        self.cache_size = max(1, cache_size)
        self._stderr = stderr
        self._cache: OrderedDict[tuple[str, str | None, str | None], MemoryTools] = OrderedDict()

    def close(self) -> None:
        while self._cache:
            _, tools = self._cache.popitem(last=False)
            with contextlib.suppress(Exception):
                tools.close()

    def _tools_for(
        self, harness_session_id: str, project: str | None, cwd: str | None
    ) -> MemoryTools:
        key = (harness_session_id, project, cwd)
        tools = self._cache.get(key)
        if tools is not None:
            self._cache.move_to_end(key)
            return tools
        if len(self._cache) >= self.cache_size:
            _, evicted = self._cache.popitem(last=False)
            with contextlib.suppress(Exception):
                evicted.close()
        tools = MemoryTools(
            self.cfg,
            self.agent_id,
            harness_session_id=harness_session_id,
            project=project,
            cwd=cwd,
            observed=BRIDGE,
        )
        self._cache[key] = tools
        return tools

    def handle_line(self, raw_line: str) -> dict[str, Any] | None:
        line = raw_line.strip()
        if not line:
            return None
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            return _error_response(None, "invalid_json", f"invalid JSON: {exc}")
        if not isinstance(payload, dict):
            return _error_response(None, "invalid_args", "request must be a JSON object")

        raw_id = payload.get("id")
        req_id = raw_id if _is_valid_id(raw_id) else None
        if "id" not in payload or req_id is None:
            return _error_response(None, "invalid_args", "id must be a string or number")

        op = payload.get("op")
        if not isinstance(op, str) or not op.strip():
            return _error_response(req_id, "invalid_args", "op must be a non-empty string")

        hsid = payload.get("harness_session_id")
        if not isinstance(hsid, str) or not hsid.strip():
            return _error_response(
                req_id, "invalid_args", "harness_session_id must be a non-empty string"
            )

        cwd = payload.get("cwd")
        if cwd is not None and not isinstance(cwd, str):
            return _error_response(req_id, "invalid_args", "cwd must be a string or null")
        cwd_val = cwd if cwd else None

        project = payload.get("project")
        if project is not None and not isinstance(project, str):
            return _error_response(req_id, "invalid_args", "project must be a string or null")
        project_val = project if project else None

        args = payload.get("args")
        if args is None:
            args = {}
        if not isinstance(args, dict):
            return _error_response(req_id, "invalid_args", "args must be an object")

        if op not in SUPPORTED_OPS:
            return _error_response(req_id, "unknown_op", f"unsupported op: {op}")

        try:
            tools = self._tools_for(hsid, project_val, cwd_val)
            result = _dispatch(tools, op, args)
        except _BridgeArgError as exc:
            return _error_response(req_id, "invalid_args", str(exc))
        except _UnknownOpError as exc:
            return _error_response(req_id, "unknown_op", str(exc))
        except (ToolError, LifecycleError) as exc:
            return _error_response(req_id, "validation_error", str(exc))
        except Exception as exc:
            err_stream = sys.stderr if self._stderr is None else self._stderr
            with contextlib.suppress(OSError):
                print(f"mem bridge: {type(exc).__name__}: {exc}", file=err_stream, flush=True)
            return _error_response(req_id, "internal_error", f"{type(exc).__name__}: {exc}")
        return {"id": req_id, "ok": True, "result": result}


def run_bridge(
    cfg: Config,
    *,
    agent_id: str | None = None,
    cache_size: int = DEFAULT_CACHE_SIZE,
    stdin: TextIO | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    effective_agent = resolve_bridge_agent(agent_id)
    in_stream = sys.stdin if stdin is None else stdin
    out_stream = sys.stdout if stdout is None else stdout
    err_stream = sys.stderr if stderr is None else stderr
    server = BridgeServer(cfg, effective_agent, cache_size=cache_size, stderr=err_stream)
    try:
        while True:
            raw_line = in_stream.readline()
            if raw_line == "":
                break
            response = server.handle_line(raw_line)
            if response is None:
                continue
            try:
                out_stream.write(json.dumps(response, ensure_ascii=False) + "\n")
                out_stream.flush()
            except BrokenPipeError:
                break
    finally:
        server.close()
    return 0


def run_context(
    cfg: Config,
    *,
    project: str | None = None,
    cwd: str | None = None,
    scopes: Sequence[str] | None = None,
    max_items: int = ranking.SESSION_START_DEFAULT_MAX_ITEMS,
    max_chars: int = ranking.SESSION_START_DEFAULT_MAX_CHARS,
    agent_id: str = HUMAN_AGENT,
    harness_session_id: str | None = None,
    as_json: bool = False,
    stdout: TextIO | None = None,
) -> dict[str, Any]:
    out = sys.stdout if stdout is None else stdout
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    effective_session = harness_session_id or f"context-{stamp}"
    tools = MemoryTools(
        cfg,
        agent_id,
        harness_session_id=effective_session,
        project=project,
        cwd=cwd,
        observed=CONTEXT,
    )
    try:
        result = tools.session_context(
            scopes=scopes,
            max_items=max_items,
            max_chars=max_chars,
        )
    finally:
        tools.close()
    if as_json:
        out.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        out.flush()
    elif not result["degraded"] and result["text"]:
        out.write(result["text"] + "\n")
        out.flush()
    return result
