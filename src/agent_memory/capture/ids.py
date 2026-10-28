"""Deterministic ids, so replaying a capture produces the same rows."""

from __future__ import annotations

import uuid

NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, "agent-memory")


def session_id(agent_id: str, harness_session_id: str) -> str:
    return str(uuid.uuid5(NAMESPACE, f"{agent_id}/{harness_session_id}"))


def event_id(agent_id: str, harness_session_id: str, kind: str, source_key: str) -> str:
    return str(uuid.uuid5(NAMESPACE, f"{agent_id}/{harness_session_id}/{kind}/{source_key}"))


def memory_id(agent_id: str, harness_session_id: str, source_key: str) -> str:
    return str(uuid.uuid5(NAMESPACE, f"{agent_id}/{harness_session_id}/memory/{source_key}"))


def action_id(agent_id: str, harness_session_id: str, source_key: str) -> str:
    return str(uuid.uuid5(NAMESPACE, f"{agent_id}/{harness_session_id}/action/{source_key}"))
