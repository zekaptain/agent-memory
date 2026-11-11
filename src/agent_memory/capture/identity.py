"""Capture identity rules: choose source keys here, hash them in ``ids``."""

from __future__ import annotations

import json
import uuid
from collections.abc import Mapping
from datetime import datetime
from typing import Literal

from . import ids


def prompt_key(message_id: str | None, ts: datetime) -> str:
    return str(message_id) if message_id else f"prompt:unkeyed:{ts.isoformat()}"


def reply_key(message_id: str) -> str:
    return str(message_id)


def thinking_key(message_id: str) -> str:
    return f"{reply_key(message_id)}#thinking"


def tool_call_key(tool_use_id: str) -> str:
    return str(tool_use_id)


def tool_result_key(tool_use_id: str) -> str:
    return str(tool_use_id)


def session_start_key(id: str) -> str:
    return id


def session_marker_key(event: str, source: str | None, id: str) -> str:
    return f"{event}:{source}:{id}" if source else note_key(event, id)


def note_key(event: str, id_or_ts: str) -> str:
    return f"{event}:{id_or_ts}"


def binding_id(placeholder_event_id: str, previous_binding_id: str | None = None) -> str:
    """Keep root and correction identities stable across replay and reconciliation."""
    return ids.binding_id(
        json.dumps([placeholder_event_id, previous_binding_id], ensure_ascii=False)
    )


def mcp_harness_session(
    agent_id: str, environ: Mapping[str, str]
) -> tuple[str, Literal["env", "own"]]:
    if session := environ.get("AGENT_MEMORY_SESSION"):
        return session, "env"
    return f"mcp-{agent_id}-{uuid.uuid4()}", "own"
