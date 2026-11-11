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
from dataclasses import asdict, dataclass, field
from typing import Any

from .. import model_call
from ..model_call import Ask, LlamaCpp, Ollama, OpenRouter, Schema, Server
from .lesson import LESSON_KINDS, LESSON_SCHEMA, Lesson, parse_lesson, render_lesson
from .store import JudgeError

DRAFTER_VERSION = "drafter_v3"
DRAFTER_KINDS = ("ollama", "llamacpp", "openrouter")
MAX_DRAFT_TOKENS = 700
# A model that writes long claims, or thinks before it answers, can spend the whole budget
# before the JSON closes. The request is seeded, so a rerun at the same budget would be cut
# off at the same place.
RETRY_DRAFT_TOKENS = 3 * MAX_DRAFT_TOKENS
TEMPERATURE = 0.2
SEED = 7
OPENAI_CHAT_TIMEOUT = 180

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


class Drafter:
    """A drafter on one model server. Thinking is off, the reply is held to the lesson
    schema, and a cut-off draft is asked for once more with room to finish."""

    def __init__(self, server: Server) -> None:
        self.server = server
        self.name = f"{server.kind}:{server.model}"
        self.local = server.local

    def draft(self, item_id: str, state: str) -> Draft:
        ask = Ask(
            prompt=drafter_prompt(state),
            budget=MAX_DRAFT_TOKENS,
            system=SYSTEM,
            temperature=TEMPERATURE,
            seed=SEED,
            shape=Schema("lesson", LESSON_SCHEMA),
            think=False,
        )
        try:
            result = model_call.call(self.server, ask, who="drafter", wider=RETRY_DRAFT_TOKENS)
        except model_call.ModelCallError as exc:
            raise JudgeError(str(exc)) from None
        return Draft.from_lesson(
            item_id,
            lesson=parse_lesson(result.data),
            schema_enforced=result.shape_enforced,
            drafter=self.name,
            model_snapshot=result.snapshot,
            # Every attempt is billed, the cut-off one included.
            latency_ms=result.latency_ms,
            cost_usd=result.cost_usd,
        )


def make_drafter(spec: str) -> Drafter:
    """`ollama:<model>`, `llamacpp:<model>`, or `openrouter:<model>`."""
    kind, _, model = spec.strip().partition(":")
    if kind not in DRAFTER_KINDS or not model:
        raise JudgeError(
            f"bad drafter spec {spec!r}; use ollama:<model>, llamacpp:<model>, or "
            "openrouter:<model>"
        )
    if kind == "ollama":
        return Drafter(Ollama(model))
    if kind == "llamacpp":
        return Drafter(LlamaCpp(model, timeout=OPENAI_CHAT_TIMEOUT))
    return Drafter(OpenRouter(model, timeout=OPENAI_CHAT_TIMEOUT))
