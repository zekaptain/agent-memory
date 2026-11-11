"""Lesson drafters (`drafter_v3`): a model writes a structured lesson from one past turn, cites the
numbered evidence lines it relied on, or declines when the turn holds nothing worth keeping.

The prompt leads with the decline. Given the lesson template first and the null reply as an
afterthought, local and hosted models alike filled the template for every turn, routine ones
included; told first that most turns hold nothing and what the exceptions are, they decline
the routine ones. A refusal the harness itself issued, such as a tool call cancelled while a
plan awaits approval, is named as not a lesson: the harness enforces it every time, so there
is nothing for a future session to remember.

The drafter sees the projection without its NEXT MESSAGE entry, so a lesson comes from this
turn's own evidence and cannot fuse the request with an unrelated follow-up. The header's
`closed_by` tells it whether the turn finished; a cut-off turn yields a lesson about the
attempt, not the result. Every item is drafted, clean controls included, so judges answering
the draft questions learn to reject lessons drawn from turns that did not fail.

Three kinds: `ollama:<model>` and `llamacpp:<model>` stay local; `openrouter:<model>` sends
the redacted rendering to a hosted model and needs the remote opt-in. Several drafters can
draft the same run, one row per (item, drafter), so the draft questions compare them on
identical turns.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from typing import Any

from ..endpoints import LLAMACPP_HOST, OLLAMA_HOST
from .backends import http
from .backends.base import require_remote
from .backends.jev import KEY_ENV
from .backends.llamacpp import disable_thinking
from .backends.llm import out_of_tokens, parse_judge_json
from .lesson import LESSON_KINDS, LESSON_SCHEMA, Lesson, parse_lesson, render_lesson
from .store import JudgeError

DRAFTER_VERSION = "drafter_v3"
DRAFTER_KINDS = ("ollama", "llamacpp", "openrouter")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MAX_DRAFT_TOKENS = 700
# A model that writes long claims, or reasons before it answers, can spend the whole
# budget before the JSON closes. On OpenAI-compatible endpoints reasoning tokens share
# max_tokens with the visible answer.
RETRY_DRAFT_TOKENS = 3 * MAX_DRAFT_TOKENS
TEMPERATURE = 0.2
SEED = 7
# OpenRouter passes a provider's rejection of the schema through as a 400; some providers
# say 422. Auth, credit, timeout, and rate-limit codes are not schema problems, so they get
# no second send without the schema.
SCHEMA_REJECTED = frozenset({400, 422})

# One short harness sentence, on purpose. A paragraph spelling out harness behavior made a 26B
# local model decline nearly every turn, project facts included, while still writing up the
# Spec Mode refusal it described; the `enforced` draft question is the backstop for that model.
SYSTEM = (
    "You write one lesson for a future AI coding-agent session on the same project, from a "
    "numbered projection of one past turn: the request, the tool calls and their results, the "
    "agent's final message, and how the turn ended. Most turns hold nothing worth keeping: a "
    "routine request that went fine, a result with no project-specific fact, or advice any "
    "competent agent already follows. Write a lesson only when this turn shows a "
    "project-specific fact, a tool quirk, or a correction that a future session would otherwise "
    "repeat. A refusal the harness itself issued, such as a tool call cancelled until a plan is "
    'approved, is not a lesson about a tool. Otherwise return {"lesson": null}. A lesson has one '
    "rule sentence saying what to do differently next time or what fact to remember, a kind, and "
    "1 to 4 claims. Each claim is one sentence about what happened in this turn, with its own "
    "cited line numbers that show it. Each sentence is at most 300 characters. Use only this "
    "turn's own evidence. The header's closed_by says how the turn ended: outcome means the "
    "agent finished and the result was recorded; next_message or session_end means the turn was "
    "cut off before the agent finished, so the request may not have been carried out and any "
    "lesson is about the attempt, not the result. Return only JSON."
)


def draft_hash(lesson: str | None) -> str | None:
    """Identifies a draft on verdicts and labels; None for a declined or missing draft."""
    return hashlib.sha256(lesson.encode("utf-8")).hexdigest() if lesson else None


def drafter_prompt(state: str) -> str:
    return (
        f"EVIDENCE\n{state}\n\n"
        'Return {"lesson": null} unless this turn holds something worth keeping. Otherwise return '
        '{"lesson": {"rule": "<one sentence>", "kind": "<kind>", '
        '"claims": [{"text": "<one sentence about this turn>", "cited": [<line numbers>]}]}}. '
        "Kinds: "
        + "; ".join(f"{kind}: {description}" for kind, description in LESSON_KINDS.items())
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
    rule: str | None = None
    kind: str | None = None
    claims: list[dict[str, Any]] = field(default_factory=list)
    schema_enforced: bool = True

    @property
    def declined(self) -> bool:
        return self.lesson is None

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "declined": self.declined, "drafter_version": DRAFTER_VERSION}

    @classmethod
    def from_lesson(cls, item_id: str, lesson: Lesson | None, **metrics: Any) -> Draft:
        return cls(
            item_id=item_id,
            lesson=render_lesson(lesson) if lesson is not None else None,
            cited=sorted({n for claim in lesson.claims for n in claim.cited}) if lesson else [],
            rule=lesson.rule if lesson else None,
            kind=lesson.kind if lesson else None,
            claims=[claim.model_dump(mode="json") for claim in lesson.claims] if lesson else [],
            **metrics,
        )


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
            "format": LESSON_SCHEMA,
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
        lesson = parse_lesson(parse_judge_json(attempts[-1].content))
        return Draft.from_lesson(
            item_id,
            lesson=lesson,
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
        schema_enforced = True
        schema: dict[str, Any] = {"name": "lesson", "schema": LESSON_SCHEMA}
        if self.kind == "openrouter":
            schema["strict"] = True
        body: dict[str, Any] = {
            "model": self.model,
            "stream": False,
            "temperature": TEMPERATURE,
            "seed": SEED,
            "max_tokens": MAX_DRAFT_TOKENS,
            "response_format": {"type": "json_schema", "json_schema": schema},
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": drafter_prompt(state)},
            ],
            **self.extra,
        }

        def once(budget: int) -> Attempt:
            nonlocal schema_enforced
            body["max_tokens"] = budget
            try:
                response, latency = http.post_json(self.url, body, headers=headers, timeout=180)
            except http.HttpStatusError as exc:
                if (
                    self.kind != "openrouter"
                    or not schema_enforced
                    or exc.code not in SCHEMA_REJECTED
                ):
                    raise
                body["response_format"] = {"type": "json_object"}
                schema_enforced = False
                response, latency = http.post_json(self.url, body, headers=headers, timeout=180)
                latency += exc.latency_ms
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
        lesson = parse_lesson(parse_judge_json(attempts[-1].content))
        return Draft.from_lesson(
            item_id,
            lesson=lesson,
            schema_enforced=schema_enforced,
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
