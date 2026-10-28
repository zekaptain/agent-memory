"""Adapter protocol and helpers shared by the harness adapters."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Protocol

from ..capture.record import Record


class UnsupportedTranscript(Exception):
    """This adapter cannot read transcripts (yet)."""


class Adapter(Protocol):
    agent_id: str

    def from_hook(self, payload: dict[str, Any]) -> list[Record]: ...

    def from_transcript(
        self, path: Path, *, since_key: str | None = None, include_thinking: bool = False
    ) -> list[Record]: ...


def text_of(content: Any) -> str:
    """Flatten a message content field (string or list of typed parts) to text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                if isinstance(part.get("text"), str):
                    parts.append(part["text"])
                else:
                    parts.append(compact_json(part))
        return "\n".join(p for p in parts if p)
    return compact_json(content)


def compact_json(value: Any, limit: int | None = None) -> str:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    if limit is not None and len(text) > limit:
        return text[: limit - 1] + "…"
    return text


def render_tool_call(name: str | None, tool_input: Any) -> str:
    return f"{name or 'tool'} {compact_json(tool_input)}" if tool_input is not None else str(name)


def project_of(cwd: str | None) -> str | None:
    return Path(cwd).name if cwd else None


def iter_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any] | None, str]]:
    """Yield (line number, parsed object or None, raw line) for each non-empty line."""
    with path.open(encoding="utf-8") as fh:
        for number, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                yield number, None, line
                continue
            yield number, obj if isinstance(obj, dict) else None, line
