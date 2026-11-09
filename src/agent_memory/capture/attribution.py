"""Conservative answer attribution from captured tool results and explicit turns."""

from __future__ import annotations

import json
import os
import uuid
from collections import defaultdict
from collections.abc import Sequence
from typing import Any

import duckdb

from ..db import MemoryClient, sql_literal
from . import ids
from .observations import fingerprint
from .record import Record

METHOD = "captured_cite_turn_v1"


def _memory_tool(payload: dict[str, Any], operation: str) -> bool:
    """Require the configured MCP server identity, not a generic tool-name suffix."""
    server = os.environ.get("AGENT_MEMORY_MCP_SERVER_NAME", "agent-memory")
    name = payload.get("tool_name")
    if payload.get("mcp_server") == server and name == operation:
        return True
    return name in {
        f"{server}.{operation}",
        f"{server}__{operation}",
        f"{server}___{operation}",
        f"mcp__{server}__{operation}",
        f"tools.{server}.{operation}",
    }


def _uuid(value: Any) -> str | None:
    try:
        return str(uuid.UUID(str(value)))
    except (ValueError, TypeError):
        return None


def _result_id(value: Any, field: str) -> str | None:
    """Decode structured MCP results or their captured text envelope."""
    if isinstance(value, str):
        try:
            return _result_id(json.loads(value), field)
        except (ValueError, TypeError):
            return None
    if isinstance(value, list):
        candidates = {_result_id(part, field) for part in value} - {None}
        return candidates.pop() if len(candidates) == 1 else None
    if not isinstance(value, dict):
        return None
    if result := _uuid(value.get(field)):
        return result
    for key in ("structuredContent", "structured_content", "content", "text"):
        if result := _result_id(value.get(key), field):
            return result
    return None


def evidence_from_records(records: Sequence[Record]) -> list[dict[str, Any]]:
    """Require a named complete turn and unique reply, never timestamp adjacency."""
    calls = {r.source_key: r for r in records if r.kind == "tool_call"}
    complete_turns = {
        str(attr["turn_id"])
        for rec in records
        if (attr := rec.attribution or {}).get("turn_id") and attr.get("turn_complete")
    }
    replies: dict[str, list[Record]] = defaultdict(list)
    for rec in records:
        attr = rec.attribution or {}
        turn = attr.get("turn_id")
        if (
            turn
            and rec.kind == "assistant_msg"
            and (rec.payload or {}).get("part") != "thinking"
            and not attr.get("has_tool_calls")
            and attr.get("channel") in (None, "final")
        ):
            replies[str(turn)].append(rec)
    evidence = []
    captured_retrievals: dict[str, list[dict[str, str]]] = defaultdict(list)
    for rec in records:
        if rec.kind != "tool_result" or not (call := calls.get(rec.parent_key or "")):
            continue
        turn = str((call.attribution or {}).get("turn_id") or "")
        if (
            turn
            and (
                _memory_tool(call.payload or {}, "recall")
                or _memory_tool(call.payload or {}, "graph")
            )
            and (retrieval_id := _result_id(rec.content, "retrieval_id"))
        ):
            captured_retrievals[turn].append(
                {
                    "retrieval_id": retrieval_id,
                    "result_event_id": ids.event_id(
                        rec.agent_id, rec.harness_session_id, rec.kind, rec.source_key
                    ),
                }
            )
    for rec in records:
        if rec.kind != "tool_result" or not (call := calls.get(rec.parent_key or "")):
            continue
        if not _memory_tool(call.payload or {}, "cite"):
            continue
        placeholder = _result_id(rec.content, "answer_event_id")
        attr = call.attribution or {}
        turn = str(attr.get("turn_id") or "")
        if not placeholder or not turn or turn not in complete_turns:
            continue
        # An explicit supplied answer is already linked by cite. Do not reinterpret it.
        supplied = (call.payload or {}).get("tool_input") or {}
        if isinstance(supplied, dict) and supplied.get("answer_event_id"):
            continue
        candidates = replies.get(turn, [])
        final = [r for r in candidates if (r.attribution or {}).get("channel") == "final"]
        if final:
            candidates = final
        evidence.append(
            {
                "placeholder_event_id": placeholder,
                "call_event_id": ids.event_id(
                    call.agent_id, call.harness_session_id, call.kind, call.source_key
                ),
                "result_event_id": ids.event_id(
                    rec.agent_id, rec.harness_session_id, rec.kind, rec.source_key
                ),
                "reply_event_ids": [
                    ids.event_id(r.agent_id, r.harness_session_id, r.kind, r.source_key)
                    for r in candidates
                ],
                "turn_id": turn,
                "method": METHOD,
                "captured_retrievals": captured_retrievals.get(turn, []),
            }
        )
    return evidence


def reconcile_answers(client: MemoryClient, session_ids: Sequence[str]) -> int:
    """Append or correct bindings from persisted evidence; safe to repeat after failure."""
    total = 0
    for sid in sorted(set(session_ids)):
        for attempt in range(2):
            try:
                total += _reconcile_explicit(client, sid)
                total += _reconcile_session(client, sid)
                break
            except duckdb.ConstraintException:
                if attempt:
                    raise
    return total


def _reconcile_explicit(client: MemoryClient, sid: str) -> int:
    _, rows = client.query(
        "SELECT p.event_id::VARCHAR, a.event_id::VARCHAR FROM events p "
        "JOIN events a ON a.event_id = try_cast(p.payload.explicit_answer_event_id AS UUID) "
        f"WHERE p.session_id = {sql_literal(sid)} AND a.session_id = p.session_id "
        "AND coalesce(try_cast(p.payload.placeholder AS BOOLEAN), false) "
        "AND a.kind = 'assistant_msg' "
        "AND NOT coalesce(try_cast(a.payload.placeholder AS BOOLEAN), false) "
        "AND NOT EXISTS (SELECT 1 FROM answer_bindings b WHERE b.placeholder_event_id = p.event_id)"
    )
    if rows:
        client.insert(
            {
                "answer_bindings": [
                    {
                        "binding_id": str(
                            uuid.uuid5(
                                uuid.NAMESPACE_URL,
                                "agent-memory/answer-binding/" + fingerprint([placeholder, None]),
                            )
                        ),
                        "placeholder_event_id": placeholder,
                        "root_placeholder_id": placeholder,
                        "answer_event_id": answer,
                        "evidence": {"explicit_answer_event_id": answer},
                        "resolution_method": "explicit_answer_id_v1",
                        "reason_code": "explicit_answer_id",
                    }
                    for placeholder, answer in rows
                ]
            }
        )
    return len(rows)


def _reconcile_session(client: MemoryClient, sid: str) -> int:
    _, observations = client.query(
        "SELECT observation_id::VARCHAR, attribution_evidence::JSON::VARCHAR "
        f"FROM capture_observations WHERE session_id = {sql_literal(sid)} "
        "AND json_array_length(attribution_evidence::JSON) > 0"
    )
    evidence: dict[str, list[dict[str, Any]]] = defaultdict(list)
    wanted: set[str] = set()
    for oid, raw in observations:
        for item in json.loads(raw or "[]"):
            placeholder = _uuid(item.get("placeholder_event_id"))
            if not placeholder:
                continue
            evidence[placeholder].append({"observation_id": oid, **item})
            wanted.add(placeholder)
            wanted.update(
                value
                for key in ("call_event_id", "result_event_id")
                if (value := _uuid(item.get(key)))
            )
            wanted.update(value for mid in item.get("reply_event_ids", []) if (value := _uuid(mid)))
            wanted.update(
                value
                for retrieval in item.get("captured_retrievals", [])
                if (value := _uuid(retrieval.get("result_event_id")))
            )
    if not wanted:
        return 0
    events: dict[str, tuple[str, bool]] = {}
    cite_calls: set[str] = set()
    ordered = sorted(wanted)
    for start in range(0, len(ordered), 500):
        literals = ", ".join(sql_literal(mid) for mid in ordered[start : start + 500])
        _, rows = client.query(
            "SELECT event_id::VARCHAR, kind, "
            "coalesce(try_cast(payload.placeholder AS BOOLEAN), false), payload::JSON::VARCHAR "
            f"FROM events WHERE session_id = {sql_literal(sid)} AND event_id IN ({literals})"
        )
        for mid, kind, placeholder, raw in rows:
            events[str(mid)] = (str(kind), bool(placeholder))
            if kind == "tool_call" and _memory_tool(json.loads(raw or "{}"), "cite"):
                cite_calls.add(str(mid))
    inserts = []
    for placeholder, items in evidence.items():
        if events.get(placeholder) != ("assistant_msg", True):
            continue
        candidates: set[str] = set()
        usable = []
        for item in items:
            if (
                item["call_event_id"] not in cite_calls
                or events.get(item["result_event_id"], (None,))[0] != "tool_result"
            ):
                continue
            # Missing imported candidates cannot turn an ambiguous turn into a unique one.
            candidates.update(item.get("reply_event_ids", []))
            usable.append(item)
        if not usable:
            continue
        answer = next(iter(candidates)) if len(candidates) == 1 else None
        reason = "unique_captured_turn" if answer else "ambiguous_captured_turn"
        if answer and events.get(answer) != ("assistant_msg", False):
            answer = None
            reason = "reply_not_imported"
        captured = sorted(
            {
                retrieval["retrieval_id"]
                for item in usable
                for retrieval in item.get("captured_retrievals", [])
                if events.get(retrieval["result_event_id"], (None,))[0] == "tool_result"
            }
        )
        _, old = client.query(
            "SELECT binding_id::VARCHAR, answer_event_id::VARCHAR, reason_code, "
            "captured_retrieval_ids::VARCHAR[] "
            "FROM effective_answer_bindings "
            f"WHERE placeholder_event_id = {sql_literal(placeholder)}"
        )
        if old and (old[0][1], old[0][2], old[0][3]) == (answer, reason, captured):
            continue
        previous = str(old[0][0]) if old else None
        # A deterministic root and unique correction parent settle concurrent reconciliation.
        binding_id = str(
            uuid.uuid5(
                uuid.NAMESPACE_URL,
                "agent-memory/answer-binding/" + fingerprint([placeholder, previous]),
            )
        )
        inserts.append(
            {
                "binding_id": binding_id,
                "placeholder_event_id": placeholder,
                "answer_event_id": answer,
                "supersedes_binding_id": previous,
                "root_placeholder_id": placeholder if previous is None else None,
                "evidence": usable,
                "resolution_method": METHOD,
                "reason_code": reason,
                "captured_retrieval_ids": captured,
            }
        )
    if inserts:
        client.insert({"answer_bindings": inserts})
    return len(inserts)
