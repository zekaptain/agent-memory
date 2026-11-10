"""Admission writes and reviews, with append-only lifecycle deduplication."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from ..capture import blobs, ids
from ..capture.pipeline import make_redactor
from ..capture.record import Record
from ..config import Config
from ..db import MemoryClient, sql_literal
from ..judge.pipeline import decision_record
from ..lifecycle import LifecyclePlan, stable_id

MEMORY_TYPES = {
    "project_fact": "fact",
    "tool_quirk": "fact",
    "environment": "fact",
    "process": "procedure",
    "user_preference": "preference",
}


def _event_rows(client: MemoryClient, record: Record) -> tuple[str, str, list]:
    sid = ids.session_id(record.agent_id, record.harness_session_id)
    eid = ids.event_id(record.agent_id, record.harness_session_id, record.kind, record.source_key)
    rows: list[tuple[str, list[dict[str, Any]]]] = []
    if not client.query(f"SELECT 1 FROM sessions WHERE session_id = {sql_literal(sid)}")[1]:
        info = record.session
        rows.append(
            (
                "sessions",
                [
                    {
                        "session_id": sid,
                        "agent_id": record.agent_id,
                        "harness_session_id": record.harness_session_id,
                        "project": info.project if info else None,
                        "scope": "judge",
                        "capture_completeness": "partial",
                        "started_at": info.started_at if info else record.ts,
                    }
                ],
            )
        )
    if not client.query(f"SELECT 1 FROM events WHERE event_id = {sql_literal(eid)}")[1]:
        rows.append(
            (
                "events",
                [
                    {
                        "event_id": eid,
                        "session_id": sid,
                        "seq": record.seq,
                        "kind": record.kind,
                        "content": record.content,
                        "content_hash": blobs.content_hash(record.content or ""),
                        "payload": record.payload,
                        "ts": record.ts,
                        "redactions": record.redactions,
                    }
                ],
            )
        )
    return sid, eid, rows


def admit(
    cfg: Config,
    day: str,
    created_at: str,
    item: dict[str, Any],
    draft: dict[str, Any],
    verdict: dict[str, Any],
    result: dict[str, Any],
    *,
    auto: bool = False,
    human_action: dict[str, Any] | None = None,
    replace_memory_id: str | None = None,
) -> dict[str, Any]:
    """Commit the decision, memory, sources and optional approval in one transaction.

    Deterministic identities also cover a crash after commit but before admitted.jsonl.
    A repeated lesson from a different turn confirms its canonical memory.
    """
    if result["action"] == "discard" and human_action is None:
        return {**result, "memory_id": None}
    redactor = make_redactor(cfg)
    metadata = {
        "approver_version": result["approver_version"],
        "score": result["score"],
        "reasons": result["reasons"],
        "draft_hash": verdict["draft_hash"],
        "item_id": item["item_id"],
        "backend": verdict["backend"],
        "drafter": draft["drafter"],
    }
    record = decision_record(
        {**verdict, "action": result["action"], "reasons": result["reasons"]},
        item,
        created_at,
        harness_session_id=f"admission-{day}",
        project=item.get("project") or "unknown",
    )
    record = redactor.record(
        replace(
            record,
            source_key=record.source_key + "/" + result["approver_version"],
            payload={**(record.payload or {}), "admission": metadata},
        )
    )
    with MemoryClient.from_config(cfg) as client:
        sid, eid, rows = _event_rows(client, record)
        plan = LifecyclePlan(client)
        content, _ = redactor.text(draft["rule"])
        lesson, _ = redactor.value({"kind": draft["kind"], "claims": draft["claims"]})
        clean_metadata, _ = redactor.value(metadata)
        cited = {n for claim in draft["claims"] for n in claim["cited"]}
        sources = {item["turn_event_id"]}
        sources.update(
            entry["event_id"]
            for entry in item["sidecar"]
            if entry["n"] in cited and entry.get("event_id")
        )
        mid, duplicate = plan.remember(
            {
                "memory_id": stable_id(
                    human_action["action_id"] if human_action and replace_memory_id else eid,
                    "memory",
                ),
                "scope": item.get("project") or "global",
                "type": MEMORY_TYPES[draft["kind"]],
                "content": content,
                "content_hash": blobs.content_hash(content),
                "confidence": result["score"],
                "trust_level": "tool",
                "created_by_session": sid,
                "metadata": {"lesson": lesson, "admission": clean_metadata},
            },
            event_id=eid,
            actor="human" if human_action else "judge",
            deduplicate=replace_memory_id is None,
            sources=[
                {"event_id": source, "relation": "derived_from"} for source in sorted(sources)
            ],
        )
        if human_action is not None or (auto and result["action"] == "write"):
            action = human_action or {
                "action_id": stable_id(eid, "approve"),
                "action": "approve",
                "reason": f"approver {result['approver_version']} score {result['score']:.6f}",
            }
            plan.review(
                {
                    **action,
                    "memory_id": mid,
                    "actor": "human" if human_action else "judge",
                    "session_id": sid,
                    "event_id": eid,
                    "related_memory_id": replace_memory_id,
                }
            )
        client.insert([*rows, *plan.rows])
    return {**result, "memory_id": mid, "duplicate": duplicate, "event_id": eid}


def review_memory(cfg: Config, memory_id: str, action: dict[str, Any]) -> str:
    """Apply a human action using its stable review identity."""
    clean, _ = make_redactor(cfg).value(action)
    with MemoryClient.from_config(cfg) as client:
        plan = LifecyclePlan(client)
        mid = plan.review({**clean, "memory_id": memory_id, "actor": "human"})
        client.insert(plan.rows)
    return mid


def terminal_memory(cfg: Config, memory_id: str) -> bool:
    with MemoryClient.from_config(cfg) as client:
        plan = LifecyclePlan(client)
        return plan.memory(plan.resolve(memory_id)).get("status") in {
            "retired",
            "superseded",
            "expired",
            "deduplicated",
        }
