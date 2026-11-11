"""Factory Droid: hooks.json stdin payloads and ~/.factory/sessions transcripts."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from ..capture import identity
from ..capture.attribution import evidence_from_records
from ..capture.record import Record, SessionInfo, now, parse_ts
from .base import UnsupportedTranscript, iter_jsonl, project_of, render_tool_call, text_of

AGENT_ID = "factory-droid"

# What each hook event is recorded as. A hook records what the transcript does not hold, or
# holds only later; the transcript is the record of replies, tool calls, and compactions.
# None is nothing at all. Stop carries no reply text. PostToolUse names no call, so what it
# wrote could not be told from the transcript's record of the same call, and it is not sent
# for a call made inside a script. PreCompact says a compaction was asked for; the
# compaction itself, with its summary, is a transcript entry.
_HOOK_KINDS = {
    "SessionStart": "session_start",
    "SessionEnd": "session_end",
    "UserPromptSubmit": "user_msg",
    "PreCompact": "note",
    "SubagentStop": "note",
    "Notification": "note",
    "Stop": None,
    "PreToolUse": None,
    "PostToolUse": None,
}

# Events keyed on the message id Droid sends with them, when it sends one. The id names
# where in the session the event falls: the message a compaction follows, the reply a
# session is resumed after. Raised again at the same place the event says nothing new, and
# it is recorded once. Any other event is told from the next by when its hook ran, because
# one message can raise several notifications that each say something else.
_KEYED_ON_MESSAGE = frozenset({"SessionStart", "SessionEnd", "PreCompact"})

# The hooks at which the transcript holds what no hook carries: the reply and the tool
# calls of a turn when it stops, and the outcome of the last turn when the session ends.
_TRANSCRIPT_DUE = frozenset({"Stop", "SessionEnd"})

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
        base = {
            "agent_id": AGENT_ID,
            "harness_session_id": str(hsid),
            "ts": ts,
            "session": session,
            "attribution": {
                key: payload[key]
                for key in ("message_id", "turn_id", "channel", "final_message_id")
                if payload.get(key) is not None
            },
        }
        named = payload.get("message_id") if event in _KEYED_ON_MESSAGE else None
        key_tail = str(named or ts.isoformat())
        kind = _HOOK_KINDS.get(event, "note")

        if event == "PreToolUse" and self.record_pre_tool:
            return [
                Record(
                    **base,
                    kind="note",
                    source_key=identity.note_key(event, key_tail),
                    content=render_tool_call(payload.get("tool_name"), payload.get("tool_input")),
                    payload={"hook": event, "tool_name": payload.get("tool_name")},
                )
            ]
        if kind is None:
            return []
        source_key = identity.note_key(event, key_tail)
        if event == "SessionStart":
            # A new session's start is the transcript's first entry too, and Droid gives that
            # entry the session's id. Any other start (resume, clear, compact) is the hook's
            # alone.
            source_key = (
                identity.session_start_key(str(hsid))
                if payload.get("source") == "startup"
                else identity.session_marker_key(
                    event, str(payload.get("source") or "startup"), key_tail
                )
            )
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
            source_key = identity.prompt_key(payload.get("message_id"), ts)
            content = text_of(payload.get("prompt"))
        elif event == "PreCompact":
            content = text_of(payload.get("custom_instructions")) or None
        elif event == "Notification":
            content = text_of(payload.get("message"))
        else:
            content = None
        extra = {
            k: v
            for k, v in payload.items()
            if k not in {"session_id", "transcript_path", "cwd", "hook_event_name", "prompt"}
        }
        if event == "UserPromptSubmit" and not payload.get("message_id"):
            extra["dedupe"] = "unkeyed"
        return [
            Record(
                **base,
                kind=kind,
                source_key=source_key,
                content=content,
                payload={"hook": event, **extra},
            )
        ]

    def transcript_due(self, payload: dict[str, Any]) -> Path | None:
        path = payload.get("transcript_path")
        if path and payload.get("hook_event_name") in _TRANSCRIPT_DUE:
            return Path(str(path))
        return None

    # Transcripts

    def from_transcript(
        self,
        path: Path,
        *,
        since_key: str | None = None,
        since_line: int | None = None,
        include_thinking: bool = False,
    ) -> list[Record]:
        if path.suffix != ".jsonl":
            raise UnsupportedTranscript(f"{path}: expected a .jsonl session transcript")
        hsid = path.stem
        records: list[Record] = []
        session = SessionInfo()
        model: str | None = None
        skipping = since_key is not None
        emit_start = 0
        segment_start = 0
        parse_errors: list[dict[str, Any]] = []
        unsupported: set[str] = set()
        range_start = 1
        range_end = 0
        cursor_end = None
        source_hash = hashlib.sha256()
        last_ts: datetime | None = None
        last_seq = 0

        def skip_through(number: int, entry_id: Any) -> None:
            """Leave out what this line gave, and stop leaving out once it is the line to
            read on from. An earlier read is found by the line it ended on: the id it ended
            on can be a turn's, and the turn's prompt has that id too."""
            nonlocal skipping, emit_start, range_start
            emit_start = len(records)
            if (
                entry_id == since_key
                if since_line is None
                else number == since_line and cursor_end == since_key
            ):
                skipping = False
                range_start = number + 1

        for number, entry, raw in iter_jsonl(path):
            source_hash.update(raw.encode("utf-8"))
            range_end = number
            if entry is None:
                parse_errors.append({"line": number, "reason_code": "invalid_json_object"})
                records.append(_bad_line(hsid, path, number, raw))
                if skipping:
                    skip_through(number, None)
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
            # Two kinds of entry carry no timestamp. An agent_turn_outcome takes the time
            # of the entry before it plus one millisecond. A session_start, the first entry,
            # takes the time of the first entry that has one less a millisecond. File order
            # then survives a sort by (ts, seq), and the file says the same thing however
            # often it is read.
            if entry.get("timestamp"):
                ts = parse_ts(entry["timestamp"])
                if last_ts is None:
                    records[:] = [_placed(rec, ts - timedelta(milliseconds=1)) for rec in records]
                last_ts = ts
            elif last_ts is not None:
                ts = last_ts = last_ts + timedelta(milliseconds=1)
            else:
                ts = now()  # stands until an entry gives a time; a file may give none
            seq = last_seq = int(entry.get("seq") or last_seq)
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
                            source_key=identity.session_start_key(
                                str(entry.get("id") or "session_start")
                            ),
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
                    message_records = _message_records(
                        base, entry, msg, include_thinking=include_thinking
                    )
                    if any(r.kind == "user_msg" for r in message_records):
                        # A new prompt or harness message can interrupt a turn
                        # without an outcome. Do not attach an earlier cite to
                        # a later reply across that boundary.
                        segment_start = len(records)
                    records.extend(message_records)
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
                    turn_id = entry.get("turnId")
                    if turn_id:
                        segment = records[segment_start:]
                        named = {
                            (r.attribution or {}).get("turn_id")
                            for r in segment
                            if (r.attribution or {}).get("turn_id")
                        }
                        # Conflicting explicit turns make an implicit segment ambiguous.
                        if not named or named == {str(turn_id)}:
                            for index in range(segment_start, len(records)):
                                rec = records[index]
                                records[index] = replace(
                                    rec,
                                    attribution={
                                        **(rec.attribution or {}),
                                        "turn_id": str(turn_id),
                                        "turn_complete": entry.get("resultKind") == "text",
                                        "turn_boundary": "agent_turn_outcome",
                                    },
                                )
                    records.append(
                        Record(
                            **base,
                            kind="outcome",
                            source_key=str(entry.get("turnId") or f"outcome:{number}"),
                            content=entry.get("reason"),
                            payload={"resultKind": entry.get("resultKind")},
                        )
                    )
                    segment_start = len(records)
                case _:
                    unsupported.add(str(entry_type))
            cursor_end = entry.get("id") or entry.get("turnId") or cursor_end
            if skipping:
                skip_through(number, entry.get("id"))
        if skipping and since_line is not None:
            # Not the file the earlier read was of. Ids are deterministic, so a read of all
            # of it gives the ledger nothing twice.
            return self.from_transcript(path, include_thinking=include_thinking)
        if skipping:
            raise ValueError("since_key was not found; no coverage boundary can be verified")
        if model:
            # The transcript names the model on assistant messages, not up front.
            records = [_with_model(r, model) if r.kind == "session_start" else r for r in records]
        if records and not any(r.kind == "session_start" for r in records) and since_key is None:
            records[0] = _with_session(records[0], session)
        all_records = records
        records = records[emit_start:]
        if not records:
            return []
        exclusions = ["attachment_content_unavailable", "unsupported_hook_categories"]
        if not include_thinking:
            exclusions.append("thinking_omitted")
        exclusions.extend(f"unsupported_entry_type:{kind}" for kind in sorted(unsupported))
        # Fingerprint exactly the non-empty lines parsed, not a second read of
        # a transcript the harness may still be appending.
        source_fingerprint = "nonempty_jsonl_sha256:" + source_hash.hexdigest()
        observed = [r.ts for r in records if r.kind != "capture_error"]
        metadata = asdict(session)
        metadata["model"] = model
        observation = {
            "source_mode": "transcript_import",
            "adapter_version": "droid_transcript_v2",
            "source_fingerprint": source_fingerprint,
            "covered_from": min(observed) if observed else None,
            "covered_through": max(observed) if observed else None,
            "cursor_start": since_key,
            "cursor_end": cursor_end,
            "range_start": range_start,
            "range_end": range_end,
            "parse_errors": parse_errors,
            "exclusions": exclusions,
            "session_metadata": metadata,
            "attribution_evidence": evidence_from_records(all_records),
        }
        records[0] = replace(records[0], observation=observation)
        return records


def _message_records(
    base: dict[str, Any], entry: dict[str, Any], msg: dict[str, Any], *, include_thinking: bool
) -> list[Record]:
    role = msg.get("role")
    entry_id = str(entry.get("id") or f"message:{base['seq']}")
    content = msg.get("content")
    parts = content if isinstance(content, list) else [{"type": "text", "text": content}]
    turn_id = entry.get("turnId") or msg.get("turnId") or entry.get("turn_id") or msg.get("turn_id")
    channel = msg.get("channel") or entry.get("channel")
    attribution = {
        "message_id": entry_id,
        "turn_id": str(turn_id) if turn_id is not None else None,
        "channel": channel,
        "turn_complete": bool(turn_id and channel == "final"),
        "has_tool_calls": any(
            isinstance(part, dict) and part.get("type") == "tool_use" for part in parts
        ),
    }
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
            key = identity.tool_call_key(str(part.get("id") or f"{entry_id}#tool_use"))
            out.append(
                Record(
                    **base,
                    kind="tool_call",
                    source_key=key,
                    content=render_tool_call(part.get("name"), part.get("input")),
                    payload={
                        "tool_name": part.get("name"),
                        "tool_input": part.get("input"),
                        **({"mcp_server": part["server_name"]} if part.get("server_name") else {}),
                    },
                )
            )
        elif ptype == "tool_result":
            tool_id = str(part.get("tool_use_id") or f"{entry_id}#tool_result")
            key = identity.tool_result_key(tool_id)
            out.append(
                Record(
                    **base,
                    kind="tool_result",
                    source_key=key,
                    parent_key=identity.tool_call_key(tool_id),
                    content=text_of(part.get("content")),
                    payload={"is_error": part.get("is_error")} if "is_error" in part else None,
                )
            )
        elif ptype == "thinking" and include_thinking and part.get("thinking"):
            out.append(
                Record(
                    **base,
                    kind="assistant_msg",
                    source_key=identity.thinking_key(entry_id),
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
                source_key=(
                    identity.prompt_key(entry_id, base["ts"])
                    if role == "user"
                    else identity.reply_key(entry_id)
                ),
                content=text,
                payload=payload or None,
            ),
        )
    return [replace(rec, attribution=attribution) for rec in out]


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


def _placed(rec: Record, ts: datetime) -> Record:
    """The record of an entry that carries no time, at the time its place in the file gives
    it. A line that could not be read keeps the time it was found."""
    if rec.kind == "capture_error":
        return rec
    session = replace(rec.session, started_at=ts) if rec.session else None
    return replace(rec, ts=ts, session=session)


def _with_model(rec: Record, model: str) -> Record:
    info = rec.session or SessionInfo()
    return replace(rec, session=replace(info, model=model))


def _with_session(rec: Record, session: SessionInfo) -> Record:
    return replace(rec, session=session)
