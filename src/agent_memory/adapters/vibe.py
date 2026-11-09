"""Mistral Vibe: hooks.toml payloads (pre_tool, post_tool, post_agent). Transcripts: Phase 2."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..capture.record import Record, SessionInfo, now
from .base import UnsupportedTranscript, project_of, render_tool_call, text_of

AGENT_ID = "mistral-vibe"


class VibeAdapter:
    agent_id = AGENT_ID

    def __init__(self, *, record_pre_tool: bool = False) -> None:
        self.record_pre_tool = record_pre_tool

    def from_hook(self, payload: dict[str, Any]) -> list[Record]:
        hsid = payload.get("session_id")
        event = payload.get("hook_event_name")
        if not hsid or not event:
            raise ValueError("hook payload needs session_id and hook_event_name")
        ts = now()
        metadata = {
            k: payload[k]
            for k in ("parent_session_id", "transcript_path")
            if payload.get(k) is not None
        }
        session = SessionInfo(
            cwd=payload.get("cwd"),
            project=project_of(payload.get("cwd")),
            metadata=metadata or None,
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
        if event == "post_tool":
            key = str(payload.get("tool_call_id") or f"{event}:{ts.isoformat()}")
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
            output = payload.get("tool_output_text")
            if output is None:
                output = text_of(payload.get("tool_output"))
            result = Record(
                **base,
                kind="tool_result",
                source_key=key,
                parent_key=key,
                content=str(output),
                payload={
                    "tool_name": payload.get("tool_name"),
                    "tool_status": payload.get("tool_status"),
                    "hook": event,
                },
            )
            return [call, result]
        if event == "pre_tool":
            if not self.record_pre_tool:
                return []
            key = str(payload.get("tool_call_id") or ts.isoformat())
            return [
                Record(
                    **base,
                    kind="note",
                    source_key=f"{event}:{key}",
                    content=render_tool_call(payload.get("tool_name"), payload.get("tool_input")),
                    payload={"hook": event, "tool_name": payload.get("tool_name")},
                )
            ]
        if event == "post_agent":
            # Carries no text; the hook script runs `mem ingest --transcript` next.
            return []
        extra = {
            k: v
            for k, v in payload.items()
            if k not in {"session_id", "transcript_path", "cwd", "hook_event_name"}
        }
        return [
            Record(
                **base,
                kind="note",
                source_key=f"{event}:{ts.isoformat()}",
                content=None,
                payload={"hook": event, **extra},
            )
        ]

    def from_transcript(
        self, path: Path, *, since_key: str | None = None, include_thinking: bool = False
    ) -> list[Record]:
        raise UnsupportedTranscript(
            f"{path}: Vibe transcript format is not mapped yet (Phase 2); "
            "use hooks, or file the format so the adapter can be completed."
        )
