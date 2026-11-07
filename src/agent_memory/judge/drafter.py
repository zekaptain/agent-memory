"""Lesson drafters (`drafter_v2`): a model writes a short lesson from one past turn, cites the
numbered evidence lines it relied on, or declines when the turn holds nothing worth keeping.

The drafter sees the projection without its NEXT MESSAGE entry, so a lesson comes from this
turn's own evidence and cannot fuse the request with an unrelated follow-up. The header's
`closed_by` tells it whether the turn finished; a cut-off turn yields a lesson about the
attempt, not the result. Every item is drafted, clean controls included, so Stage B judges
learn to reject lessons drawn from turns that did not fail.

Three kinds: `ollama:<model>` and `llamacpp:<model>` stay local; `openrouter:<model>` sends
the redacted rendering to a hosted model and needs the remote opt-in. Several drafters can
draft the same run, one row per (item, drafter), so Stage B compares them on identical turns.
"""

from __future__ import annotations

import hashlib
import os
import re
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from typing import Any

from .backends import http
from .backends.base import require_remote
from .backends.jev import KEY_ENV
from .backends.llamacpp import LLAMACPP_HOST, disable_thinking
from .backends.llm import out_of_tokens, parse_judge_json
from .store import JudgeError

DRAFTER_VERSION = "drafter_v2"
DRAFTER_KINDS = ("ollama", "llamacpp", "openrouter")
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MAX_LESSON_CHARS = 400
MAX_DRAFT_TOKENS = 400
# A model that writes past two sentences, or reasons before it answers, can spend the whole
# budget before the JSON closes. On OpenAI-compatible endpoints reasoning tokens share
# max_tokens with the visible answer.
RETRY_DRAFT_TOKENS = 3 * MAX_DRAFT_TOKENS
TEMPERATURE = 0.2
SEED = 7

SYSTEM = (
    "You write one lesson for a future AI coding-agent session on the same project, from a "
    "numbered projection of one past turn: the request, the tool calls and their results, the "
    "agent's final message, and how the turn ended. The lesson is at most two sentences and says "
    "what to do differently next time, or what fact to remember. Use only this turn's own "
    "evidence and cite the line numbers you relied on. The header's closed_by says how the turn "
    "ended: outcome means the agent finished and the result was recorded; next_message or "
    "session_end means the turn was cut off before the agent finished, so the request may not "
    "have been carried out and any lesson is about the attempt, not the result. If the turn "
    'holds nothing worth keeping for a future session, return {"lesson": null}. Return only JSON.'
)


def draft_hash(lesson: str | None) -> str | None:
    """Identifies a draft on verdicts and labels; None for a declined or missing draft."""
    return hashlib.sha256(lesson.encode("utf-8")).hexdigest() if lesson else None


def drafter_prompt(state: str) -> str:
    return (
        f"EVIDENCE\n{state}\n\n"
        'Return exactly: {"lesson": "<at most two sentences>" or null, '
        '"cited": [<line numbers>]}'
    )


@dataclass(frozen=True)
class Draft:
    item_id: str
    lesson: str | None
    cited: list[int]
    drafter: str
    model_snapshot: str
    latency_ms: float
    cost_usd: float = 0.0

    @property
    def declined(self) -> bool:
        return self.lesson is None

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "declined": self.declined}


def parse_draft(data: Mapping[str, Any]) -> tuple[str | None, list[int]]:
    """The lesson text (None when the drafter declined) and the cited line numbers."""
    if "lesson" not in data:
        raise JudgeError("drafter returned no lesson field")
    raw = data["lesson"]
    cited: list[int] = []
    values = data.get("cited") or []
    if not isinstance(values, list):
        values = [values]
    for value in values:
        try:
            n = int(str(value).strip().strip("[]"))
        except ValueError:
            continue
        if n > 0 and n not in cited:
            cited.append(n)
    if raw is None:
        return None, cited
    lesson = str(raw).strip()
    if not lesson:
        raise JudgeError("drafter returned an empty lesson")
    return re.sub(r"\s+", " ", lesson)[:MAX_LESSON_CHARS], cited


@dataclass(frozen=True)
class Attempt:
    """One completion call: what came back and whether the budget ran out first."""

    response: Mapping[str, Any]
    content: str
    reasoning: str
    cut_off: bool
    latency_ms: float


def draft_with_room(once: Callable[[int], Attempt]) -> list[Attempt]:
    """Call at the normal budget; when that is cut off, once more with room to finish. The
    request is seeded, so without the larger budget a rerun would be cut off at the same place.
    Raises when the retry is cut off too."""
    attempts = [once(MAX_DRAFT_TOKENS)]
    if attempts[-1].cut_off:
        attempts.append(once(RETRY_DRAFT_TOKENS))
    last = attempts[-1]
    if last.cut_off:
        raise JudgeError(out_of_tokens(last.content, last.reasoning, who="drafter"))
    return attempts


class OllamaDrafter:
    def __init__(self, model: str, *, host: str = OLLAMA_HOST) -> None:
        self.model = model
        self.host = host
        self.name = f"ollama:{model}"
        self.local = True

    def draft(self, item_id: str, state: str) -> Draft:
        body: dict[str, Any] = {
            "model": self.model,
            "stream": False,
            "think": False,
            "format": "json",
            # Two sentences and a list of numbers; the cap stops a model that keeps going.
            "options": {"temperature": TEMPERATURE, "seed": SEED, "num_predict": MAX_DRAFT_TOKENS},
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": drafter_prompt(state)},
            ],
        }

        def once(budget: int) -> Attempt:
            body["options"]["num_predict"] = budget
            response, latency = http.post_json(f"{self.host}/api/chat", body)
            message = response.get("message") or {}
            return Attempt(
                response=response,
                content=str(message.get("content") or ""),
                reasoning=str(message.get("thinking") or ""),
                cut_off=response.get("done_reason") == "length",
                latency_ms=latency,
            )

        attempts = draft_with_room(once)
        lesson, cited = parse_draft(parse_judge_json(attempts[-1].content))
        return Draft(
            item_id=item_id,
            lesson=lesson,
            cited=cited,
            drafter=self.name,
            model_snapshot=str(attempts[-1].response.get("model") or self.model),
            latency_ms=sum(a.latency_ms for a in attempts),
        )


class OpenAIChatDrafter:
    """A drafter behind an OpenAI-compatible `/v1/chat/completions`: llama-server locally, or
    OpenRouter when the remote opt-in is set."""

    def __init__(
        self,
        kind: str,
        model: str,
        url: str,
        *,
        local: bool,
        extra: Mapping[str, Any] | None = None,
        api_key: str | None = None,
    ) -> None:
        self.kind = kind
        self.model = model
        self.url = url
        self.local = local
        self.extra = dict(extra or {})
        self.api_key = api_key
        self.name = f"{kind}:{model}"

    def _headers(self) -> dict[str, str]:
        if self.local:
            return {}
        require_remote(f"The {self.kind} drafter")
        key = self.api_key or os.environ.get(KEY_ENV)
        if not key:
            raise JudgeError(f"{KEY_ENV} is not set; the {self.kind} drafter needs an API key.")
        return {"Authorization": f"Bearer {key}"}

    def draft(self, item_id: str, state: str) -> Draft:
        headers = self._headers()
        body: dict[str, Any] = {
            "model": self.model,
            "stream": False,
            "temperature": TEMPERATURE,
            "seed": SEED,
            "max_tokens": MAX_DRAFT_TOKENS,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": drafter_prompt(state)},
            ],
            **self.extra,
        }

        def once(budget: int) -> Attempt:
            body["max_tokens"] = budget
            response, latency = http.post_json(self.url, body, headers=headers, timeout=180)
            choices = response.get("choices") or []
            first = choices[0] if choices and isinstance(choices[0], dict) else {}
            message = first.get("message") or {}
            return Attempt(
                response=response,
                content=str(message.get("content") or ""),
                reasoning=str(message.get("reasoning") or ""),
                cut_off=first.get("finish_reason") == "length",
                latency_ms=latency,
            )

        attempts = draft_with_room(once)
        lesson, cited = parse_draft(parse_judge_json(attempts[-1].content))
        return Draft(
            item_id=item_id,
            lesson=lesson,
            cited=cited,
            drafter=self.name,
            model_snapshot=str(attempts[-1].response.get("model") or self.model),
            latency_ms=sum(a.latency_ms for a in attempts),
            # Every attempt is billed, the cut-off one included.
            cost_usd=sum(
                float((a.response.get("usage") or {}).get("cost") or 0.0) for a in attempts
            ),
        )


Drafter = OllamaDrafter | OpenAIChatDrafter


def make_drafter(spec: str) -> Drafter:
    """`ollama:<model>`, `llamacpp:<model>`, or `openrouter:<model>`."""
    kind, _, model = spec.strip().partition(":")
    if kind not in DRAFTER_KINDS or not model:
        raise JudgeError(
            f"bad drafter spec {spec!r}; use ollama:<model>, llamacpp:<model>, or "
            "openrouter:<model>"
        )
    if kind == "ollama":
        return OllamaDrafter(model)
    if kind == "llamacpp":
        extra: dict[str, Any] = {}
        disable_thinking(extra)
        return OpenAIChatDrafter(
            kind, model, f"{LLAMACPP_HOST}/v1/chat/completions", local=True, extra=extra
        )
    # OpenRouter reports the charge in `usage.cost` when asked; that is the per-draft cost the
    # report records.
    return OpenAIChatDrafter(
        kind, model, OPENROUTER_URL, local=False, extra={"usage": {"include": True}}
    )
