"""State projection (`projection_v3`): the bounded, deterministic text a judge sees for a turn.

Built from ledger events only. Event ids never enter the text; a sidecar maps the numbered
lines back to events so a verdict can cite evidence without the state leaving the machine.
User messages are shown without the harness blocks the transcript wraps them in, and the
header says who started the session, how the turn closed, and who wrote the next message.

The drafter sees the same text without the NEXT MESSAGE entry (`drafter_rendering`), so a
lesson cannot fuse the request with an unrelated follow-up. The sidecar is shared, so cited
line numbers mean the same thing in both renderings.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..capture.redact import Redactor
from ..db import MemoryClient, sql_literal
from .provenance import strip_harness_blocks
from .triggers import Candidate, parse_payload, read_blob

PROJECTION_VERSION = "projection_v3"

PROVENANCE_LABEL = {"human": "human", "parent_agent": "parent agent", "harness": "harness"}

USER_CHARS = 1500
ASSISTANT_CHARS = 1500
TOOL_INPUT_CHARS = 200
ERROR_RESULT_CHARS = 300
OK_RESULT_CHARS = 120
NEXT_USER_CHARS = 1000
MAX_TOOL_CALLS = 12
MAX_CHARS = 24_000

CONTENT_SQL = (
    "SELECT event_id::VARCHAR, kind, content, CAST(payload AS JSON)::VARCHAR, "
    "parent_event_id::VARCHAR, ts::VARCHAR, seq, truncated, blob_ref "
    "FROM events WHERE event_id IN ({ids}) ORDER BY ts, seq, recorded_at"
)


@dataclass(frozen=True)
class FullEvent:
    event_id: str
    kind: str
    content: str
    payload: dict[str, Any] | None
    parent_event_id: str | None


@dataclass
class Projection:
    text: str
    sidecar: list[dict[str, Any]] = field(default_factory=list)
    redactions: int = 0

    @property
    def hash(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()


_NEXT_MESSAGE = re.compile(r"\n\[\d+\] NEXT MESSAGE\b")


def drafter_rendering(text: str) -> str:
    """The projection without its NEXT MESSAGE entry. That entry is always the last one, so
    everything from its marker on is dropped; the numbering of the rest is unchanged."""
    matches = list(_NEXT_MESSAGE.finditer(text))
    return text[: matches[-1].start()] if matches else text


def fetch_events(
    client: MemoryClient, event_ids: Sequence[str], blob_dir: Path | None = None
) -> list[FullEvent]:
    """Events by id. A user message truncated inline is completed from its blob, because the
    request often sits after the harness blocks the 4 KiB cut lands in. Tool text stays inline;
    it is clipped far shorter anyway."""
    if not event_ids:
        return []
    ids = ", ".join(sql_literal(e) for e in event_ids)
    _, rows = client.query(CONTENT_SQL.format(ids=ids))
    out = []
    for eid, kind, content, payload, parent, _ts, _seq, truncated, blob_ref in rows:
        text = content or ""
        if kind == "user_msg" and truncated and blob_ref and blob_dir is not None:
            text = read_blob(blob_dir, str(blob_ref)) or text
        out.append(FullEvent(str(eid), str(kind), text, parse_payload(payload), parent or None))
    return out


def _clip(text: str, limit: int) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def _one_line(text: str, limit: int) -> str:
    return _clip(" ".join(text.split()), limit)


def _user_text(content: str, limit: int) -> str:
    body, removed = strip_harness_blocks(content)
    note = f" [{removed} harness block(s) removed]" if removed else ""
    if not body:
        return "(harness only)" + note
    return _clip(body, limit) + note


def build_projection(
    candidate: Candidate,
    events: Sequence[FullEvent],
    next_user: FullEvent | None,
    redactor: Redactor | None = None,
) -> Projection:
    by_id = {e.event_id: e for e in events}
    ordered = [by_id[eid] for eid in candidate.event_ids if eid in by_id]
    calls = [e for e in ordered if e.kind == "tool_call"]
    results_by_call: dict[str, FullEvent] = {}
    for e in ordered:
        if e.kind == "tool_result" and e.parent_event_id:
            results_by_call.setdefault(e.parent_event_id, e)
    user = next((e for e in ordered if e.kind == "user_msg"), None)
    assistant_texts = [e for e in ordered if e.kind == "assistant_msg" and e.content.strip()]
    final_assistant = assistant_texts[-1] if assistant_texts else None

    lines: list[str] = []
    sidecar: list[dict[str, Any]] = []
    counter = 0

    def add(text: str, event: FullEvent | None, kind: str) -> None:
        nonlocal counter
        counter += 1
        lines.append(f"[{counter}] {text}")
        sidecar.append({"n": counter, "event_id": event.event_id if event else None, "kind": kind})

    caller = f"agent:{candidate.caller_role}" if candidate.caller_role else candidate.caller
    header = (
        f"Coding-agent turn. project={candidate.project or 'unknown'} "
        f"session_started_by={caller} closed_by={candidate.closed_by} "
        f"outcome={candidate.outcome_reason or 'none'} "
        f"tool_calls={len(calls)} tool_errors={candidate.facts.get('tool_error_count', 0)}"
    )
    lines.append(header)
    if user is not None:
        who = "REQUEST (parent agent)" if candidate.caller != "human" else "REQUEST (human)"
        add(f"{who}: " + _user_text(user.content, USER_CHARS), user, "user_msg")

    shown = calls
    omitted = 0
    if len(calls) > MAX_TOOL_CALLS:
        half = MAX_TOOL_CALLS // 2
        shown = calls[:half] + calls[-half:]
        omitted = len(calls) - len(shown)
    for index, call in enumerate(shown):
        if omitted and index == MAX_TOOL_CALLS // 2:
            lines.append(f"... {omitted} tool call(s) omitted ...")
        name = str((call.payload or {}).get("tool_name") or "tool")
        tool_input = (call.payload or {}).get("tool_input")
        rendered = (
            json.dumps(tool_input, ensure_ascii=False, sort_keys=True, default=str)
            if tool_input is not None
            else call.content
        )
        add(f"TOOL CALL {name}: {_one_line(rendered, TOOL_INPUT_CHARS)}", call, "tool_call")
        result = results_by_call.get(call.event_id)
        if result is None:
            continue
        is_error = bool((result.payload or {}).get("is_error"))
        if is_error:
            add(
                "TOOL RESULT (error): " + _one_line(result.content, ERROR_RESULT_CHARS),
                result,
                "tool_result",
            )
        else:
            add(
                "TOOL RESULT (ok): " + _one_line(result.content, OK_RESULT_CHARS),
                result,
                "tool_result",
            )
    if final_assistant is not None:
        add(
            "ASSISTANT (final): " + _clip(final_assistant.content, ASSISTANT_CHARS),
            final_assistant,
            "assistant_msg",
        )
    add(f"OUTCOME: {candidate.outcome_reason or 'none recorded'}", None, "outcome")
    if next_user is not None:
        who = PROVENANCE_LABEL.get(candidate.next_provenance or "", "unknown")
        add(
            f"NEXT MESSAGE ({who}): " + _user_text(next_user.content, NEXT_USER_CHARS),
            next_user,
            "user_msg",
        )
    else:
        add("NEXT MESSAGE: (none; last request of the session)", None, "user_msg")

    text = "\n".join(lines)
    if len(text) > MAX_CHARS:
        text = text[: MAX_CHARS - 20].rstrip() + "\n... [projection cut]"
    redactions = 0
    if redactor is not None:
        text, redactions = redactor.text(text)
    return Projection(text=text, sidecar=sidecar, redactions=redactions)


def project_candidate(
    client: MemoryClient,
    candidate: Candidate,
    redactor: Redactor | None = None,
    blob_dir: Path | None = None,
) -> Projection:
    wanted = list(candidate.event_ids)
    if candidate.next_user_event_id:
        wanted.append(candidate.next_user_event_id)
    events = fetch_events(client, wanted, blob_dir)
    next_user = None
    if candidate.next_user_event_id:
        next_user = next((e for e in events if e.event_id == candidate.next_user_event_id), None)
    turn_events = [e for e in events if e.event_id != candidate.next_user_event_id]
    return build_projection(candidate, turn_events, next_user, redactor)
