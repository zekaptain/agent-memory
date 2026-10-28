"""Factory Droid: hooks.json stdin payloads and ~/.factory/sessions transcripts."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from ..capture.record import Record, SessionInfo, now, parse_ts
from .base import UnsupportedTranscript, iter_jsonl, project_of, render_tool_call, text_of

AGENT_ID = "factory-droid"

# Hook events that map to a single event kind. Stop/SubagentStop carry no reply text;
# the Phase 2 hook script follows them with `mem ingest --transcript`.
_HOOK_KINDS = {
    "SessionStart": "session_start",
    "SessionEnd": "session_end",
    "UserPromptSubmit": "user_msg",
    "PreCompact": "compaction",
    "SubagentStop": "note",
    "Notification": "note",
    "Stop": None,
    "PreToolUse": None,
    "PostToolUse": "tool_call",
}

_ASSISTANT_META = ("modelId", "apiProvider", "reasoningEffort", "compactionSummaryId")


class DroidAdapter:
    agent_id = AGENT_ID

    def __init__(self, *, record_pre_tool: bool = False) -> None:
        self.record_pre_tool = record_pre_tool

    # Hooks

    def from_hook(self, payload: dict[str, Any]) -> list[Record]:
        hsid = payload.get("session_id")
        event = payload.get("hook_event_name")
        if not hsid or not event:
            raise ValueError("hook payload needs session_id and hook_event_name")
        ts = now()
        session = SessionInfo(
            cwd=payload.get("cwd"),
            project=project_of(payload.get("cwd")),
            metadata={"transcript_path": payload.get("transcript_path")}
            if payload.get("transcript_path")
            else None,
        )
        base = {"agent_id": AGENT_ID, "harness_session_id": str(hsid), "ts": ts, "session": session}
        key_tail = payload.get("message_id") or payload.get("tool_use_id") or ts.isoformat()
        kind = _HOOK_KINDS.get(event, "note")

        if event == "PostToolUse":
            key = str(payload.get("tool_use_id") or f"{event}:{key_tail}")
            call = Record(
                **base,
                kind="tool_call",
                source_key=key,
                content=render_tool_call(payload.get("tool_name"), payload.get("tool_input")),
                payload={
                    "tool_name": payload.get("tool_name"),
                    "tool_input": payload.get("tool_input"),
                    "hook": event,
                },
            )
            result = Record(
                **base,
                kind="tool_result",
                source_key=key,
                parent_key=key,
                content=text_of(payload.get("tool_response")),
                payload={"tool_name": payload.get("tool_name"), "hook": event},
            )
            return [call, result]
        if event == "PreToolUse":
            if not self.record_pre_tool:
                return []
            return [
                Record(
                    **base,
                    kind="note",
                    source_key=f"{event}:{key_tail}",
                    content=render_tool_call(payload.get("tool_name"), payload.get("tool_input")),
                    payload={"hook": event, "tool_name": payload.get("tool_name")},
                )
            ]
        if kind is None:
            return []
        source_key = f"{event}:{key_tail}"
        if event == "SessionStart":
            source_key = f"{event}:{payload.get('source') or 'startup'}:{key_tail}"
            session = SessionInfo(
                cwd=session.cwd,
                project=session.project,
                started_at=ts,
                metadata={
                    **(session.metadata or {}),
                    "permission_mode": payload.get("permission_mode"),
                },
            )
            base["session"] = session
        content: str | None
        if event == "UserPromptSubmit":
            content = str(payload.get("prompt") or "")
        elif event == "PreCompact":
            content = payload.get("custom_instructions") or None
        elif event == "Notification":
            content = text_of(payload.get("message"))
        else:
            content = None
        extra = {
            k: v
            for k, v in payload.items()
            if k not in {"session_id", "transcript_path", "cwd", "hook_event_name", "prompt"}
        }
        return [
            Record(
                **base,
                kind=kind,
                source_key=source_key,
                content=content,
                payload={"hook": event, **extra},
            )
        ]

    # Transcripts

    def from_transcript(
        self, path: Path, *, since_key: str | None = None, include_thinking: bool = False
    ) -> list[Record]:
        if path.suffix != ".jsonl":
            raise UnsupportedTranscript(f"{path}: expected a .jsonl session transcript")
        hsid = path.stem
        records: list[Record] = []
        session = SessionInfo()
        model: str | None = None
        skipping = since_key is not None
        last_ts: datetime | None = None
        last_seq = 0
        for number, entry, raw in iter_jsonl(path):
            if entry is None:
                records.append(_bad_line(hsid, path, number, raw))
                continue
            entry_type = entry.get("type")
            if entry_type == "session_start":
                session = SessionInfo(
                    cwd=entry.get("cwd"),
                    project=project_of(entry.get("cwd")),
                    harness_version=str(entry["version"]) if "version" in entry else None,
                    started_at=parse_ts(entry["timestamp"]) if entry.get("timestamp") else None,
                    metadata=_session_metadata(entry),
                )
            if skipping:
                if entry.get("id") == since_key:
                    skipping = False
                continue
            # agent_turn_outcome carries neither timestamp nor seq. It takes the previous
            # entry's time plus one millisecond, so file order survives a sort by (ts, seq).
            if entry.get("timestamp"):
                ts = parse_ts(entry["timestamp"])
            elif last_ts is not None:
                ts = last_ts + timedelta(milliseconds=1)
            else:
                ts = parse_ts(session.started_at or now())
            seq = int(entry.get("seq") or last_seq)
            last_ts, last_seq = ts, seq
            base: dict[str, Any] = {
                "agent_id": AGENT_ID,
                "harness_session_id": hsid,
                "ts": ts,
                "seq": seq,
            }
            match entry_type:
                case "session_start":
                    records.append(
                        Record(
                            **base,
                            kind="session_start",
                            source_key=str(entry.get("id") or "session_start"),
                            content=entry.get("title"),
                            payload={"source": "transcript"},
                            session=SessionInfo(
                                cwd=session.cwd,
                                project=session.project,
                                harness_version=session.harness_version,
                                started_at=ts,
                                metadata=session.metadata,
                            ),
                        )
                    )
                case "message":
                    msg = entry.get("message") or {}
                    if not model and msg.get("modelId"):
                        model = str(msg["modelId"])
                    records.extend(
                        _message_records(base, entry, msg, include_thinking=include_thinking)
                    )
                case "compaction_state":
                    records.append(
                        Record(
                            **base,
                            kind="compaction",
                            source_key=str(entry.get("id") or f"compaction:{number}"),
                            content=entry.get("summaryText"),
                            payload={
                                "summaryTokens": entry.get("summaryTokens"),
                                "summaryKind": entry.get("summaryKind"),
                                "removedCount": entry.get("removedCount"),
                                "anchorMessage": entry.get("anchorMessage"),
                            },
                        )
                    )
                case "agent_turn_outcome":
                    records.append(
                        Record(
                            **base,
                            kind="outcome",
                            source_key=str(entry.get("turnId") or f"outcome:{number}"),
                            content=entry.get("reason"),
                            payload={"resultKind": entry.get("resultKind")},
                        )
                    )
                case _:
                    continue
        if model:
            # The transcript names the model on assistant messages, not up front.
            records = [_with_model(r, model) if r.kind == "session_start" else r for r in records]
        if records and not any(r.kind == "session_start" for r in records) and since_key is None:
            records[0] = _with_session(records[0], session)
        return records


def _message_records(
    base: dict[str, Any], entry: dict[str, Any], msg: dict[str, Any], *, include_thinking: bool
) -> list[Record]:
    role = msg.get("role")
    entry_id = str(entry.get("id") or f"message:{base['seq']}")
    content = msg.get("content")
    parts = content if isinstance(content, list) else [{"type": "text", "text": content}]
    out: list[Record] = []
    texts: list[str] = []
    documents: list[dict[str, Any]] = []
    for part in parts:
        if not isinstance(part, dict):
            continue
        ptype = part.get("type")
        if ptype == "text" and part.get("text"):
            texts.append(str(part["text"]))
        elif ptype == "document":
            source = part.get("source") or {}
            documents.append(
                {
                    k: source.get(k)
                    for k in ("type", "media_type", "name", "title")
                    if source.get(k) is not None
                }
            )
        elif ptype == "tool_use":
            key = str(part.get("id") or f"{entry_id}#tool_use")
            out.append(
                Record(
                    **base,
                    kind="tool_call",
                    source_key=key,
                    content=render_tool_call(part.get("name"), part.get("input")),
                    payload={"tool_name": part.get("name"), "tool_input": part.get("input")},
                )
            )
        elif ptype == "tool_result":
            key = str(part.get("tool_use_id") or f"{entry_id}#tool_result")
            out.append(
                Record(
                    **base,
                    kind="tool_result",
                    source_key=key,
                    parent_key=key,
                    content=text_of(part.get("content")),
                    payload={"is_error": part.get("is_error")} if "is_error" in part else None,
                )
            )
        elif ptype == "thinking" and include_thinking and part.get("thinking"):
            out.append(
                Record(
                    **base,
                    kind="assistant_msg",
                    source_key=f"{entry_id}#thinking",
                    content=str(part["thinking"]),
                    payload={"part": "thinking", "durationMs": part.get("durationMs")},
                )
            )
    if texts or documents:
        kind = "user_msg" if role == "user" else "assistant_msg"
        payload: dict[str, Any] = {}
        if role == "assistant":
            payload.update({k: msg[k] for k in _ASSISTANT_META if msg.get(k) is not None})
        if documents:
            payload["documents"] = documents
        if msg.get("userMessageSource"):
            payload["userMessageSource"] = msg["userMessageSource"]
        text = "\n".join(texts)
        if documents and not text:
            text = "[document]"
        out.insert(
            0,
            Record(
                **base,
                kind=kind,
                source_key=entry_id,
                content=text,
                payload=payload or None,
            ),
        )
    return out


def _session_metadata(entry: dict[str, Any]) -> dict[str, Any] | None:
    """Title plus, for a session another session spawned (Task workers, subagents), the
    caller's ids. Those tell a reader that the "user" in this transcript is an agent."""
    meta: dict[str, Any] = {}
    if entry.get("title"):
        meta["title"] = entry["title"]
    if entry.get("callingSessionId"):
        meta["calling_session_id"] = str(entry["callingSessionId"])
    if entry.get("callingToolUseId"):
        meta["calling_tool_use_id"] = str(entry["callingToolUseId"])
    return meta or None


def _bad_line(hsid: str, path: Path, number: int, raw: str) -> Record:
    return Record(
        agent_id=AGENT_ID,
        harness_session_id=hsid,
        kind="capture_error",
        ts=now(),
        source_key=f"transcript:{path.name}:{number}",
        content=f"unparseable transcript line {number}",
        payload={"line": raw[:4096]},
    )


def _with_model(rec: Record, model: str) -> Record:
    info = rec.session or SessionInfo()
    return replace(rec, session=replace(info, model=model))


def _with_session(rec: Record, session: SessionInfo) -> Record:
    return replace(rec, session=session)
