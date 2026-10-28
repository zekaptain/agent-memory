"""Lesson drafter (`drafter_v1`): a local model writes a short lesson and cites the numbered
evidence lines it relied on. Every item gets a draft, clean controls included, so Stage B
judges learn to reject lessons drawn from turns that did not fail.
"""

from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass
from typing import Any

from .backends import http
from .backends.llm import parse_judge_json
from .store import JudgeError

DRAFTER_VERSION = "drafter_v1"
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
MAX_LESSON_CHARS = 400
MAX_DRAFT_TOKENS = 400

SYSTEM = (
    "You write one lesson for a future AI coding-agent session on the same project, from a "
    "numbered projection of one past turn. The lesson is at most two sentences and says what to "
    "do differently next time, or what fact to remember. Cite the evidence line numbers you "
    "relied on. If the turn shows no failure, still write the most useful lesson you can; the "
    "judge decides whether it is worth keeping. Return only JSON."
)


def drafter_prompt(state: str) -> str:
    return (
        f"EVIDENCE\n{state}\n\n"
        'Return exactly: {"lesson": "<at most two sentences>", "cited": [<line numbers>]}'
    )


@dataclass(frozen=True)
class Draft:
    item_id: str
    lesson: str
    cited: list[int]
    drafter: str
    model_snapshot: str
    latency_ms: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def parse_draft(data: dict[str, Any]) -> tuple[str, list[int]]:
    lesson = str(data.get("lesson") or "").strip()
    if not lesson:
        raise JudgeError("drafter returned an empty lesson")
    lesson = re.sub(r"\s+", " ", lesson)[:MAX_LESSON_CHARS]
    cited: list[int] = []
    raw = data.get("cited") or []
    if not isinstance(raw, list):
        raw = [raw]
    for value in raw:
        try:
            n = int(str(value).strip().strip("[]"))
        except ValueError:
            continue
        if n > 0 and n not in cited:
            cited.append(n)
    return lesson, cited


class OllamaDrafter:
    def __init__(self, model: str, *, host: str = OLLAMA_HOST) -> None:
        self.model = model
        self.host = host
        self.name = f"ollama:{model}"

    def draft(self, item_id: str, state: str) -> Draft:
        body: dict[str, Any] = {
            "model": self.model,
            "stream": False,
            "think": False,
            "format": "json",
            # Two sentences and a list of numbers; the cap stops a model that keeps going.
            "options": {"temperature": 0.2, "seed": 7, "num_predict": MAX_DRAFT_TOKENS},
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": drafter_prompt(state)},
            ],
        }
        response, latency = http.post_json(f"{self.host}/api/chat", body)
        content = str((response.get("message") or {}).get("content") or "")
        lesson, cited = parse_draft(parse_judge_json(content))
        return Draft(
            item_id=item_id,
            lesson=lesson,
            cited=cited,
            drafter=self.name,
            model_snapshot=str(response.get("model") or self.model),
            latency_ms=latency,
        )


def make_drafter(spec: str) -> OllamaDrafter:
    """`ollama:<model>` is the only drafter; drafting stays local."""
    kind, _, model = spec.strip().partition(":")
    if kind != "ollama" or not model:
        raise JudgeError(f"bad drafter spec {spec!r}; use ollama:<model>")
    return OllamaDrafter(model)
