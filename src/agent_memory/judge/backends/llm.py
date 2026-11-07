"""The reasoning baseline: a local Ollama model prompted with the rubric, answering in JSON.

Its self-reported probability is the confidence. The literature says that number is poorly
calibrated; measuring it against the decision backends is part of the experiment.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from ..questions import Question
from ..store import JudgeError
from . import http
from .base import Answer, JudgeInput, answer_from_probabilities

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
MAX_JUDGE_TOKENS = 4096
# Thinking shares the budget with the answer. A judge cut off mid-thought is asked once more
# with room before thinking is switched off: at temperature 0 the same budget stops at the
# same place, and a template with no thinking switch (gpt-oss) makes the no-think retry the
# same request again. Three times the budget, as for drafts; at gemma4:26b's speed on Ollama
# that still finishes inside the request timeout.
RETRY_JUDGE_TOKENS = 3 * MAX_JUDGE_TOKENS

SYSTEM = (
    "You evaluate one turn of an AI coding agent. The evidence is a numbered projection of the "
    "turn: the user's request, tool calls and their results, the agent's final message, the "
    "recorded outcome, and the user's next message. The header's closed_by says how the turn "
    "ended: at its outcome, cut off by the next message, or at the end of the session. Judge "
    "only from that evidence. Do not trust the agent's own claims of success. Answer every "
    "question. Return only JSON."
)


def render_questions(questions: Sequence[Question]) -> str:
    blocks = []
    for q in questions:
        opts = "\n".join(f"    - {d}" for d in q.option_descriptions())
        blocks.append(f'- id "{q.id}" ({q.type}): {q.instructions}\n  options:\n{opts}')
    return "\n".join(blocks)


def judge_prompt(state: str, questions: Sequence[Question]) -> str:
    schema = ", ".join(
        f'"{q.id}": {{"answer": <one of the option names>, "probability": <0..1 that your answer '
        f"is correct>}}"
        for q in questions
    )
    return (
        f"EVIDENCE\n{state}\n\nQUESTIONS\n{render_questions(questions)}\n\n"
        "For a noul question answer 'yes' or 'no'. For a choice question answer with the option "
        "name. For a score question answer with the level number. Give the probability that your "
        f"answer is correct.\n\nReturn exactly this JSON object: {{{schema}}}"
    )


_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


def parse_judge_json(text: str) -> dict[str, Any]:
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = _JSON_OBJECT.search(text)
        if not match:
            raise JudgeError(f"judge returned no JSON: {text[:200]!r}") from None
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            raise JudgeError(
                f"judge returned unparsable JSON ({exc.msg}): {text[:200]!r}"
            ) from None
    if not isinstance(data, dict):
        raise JudgeError("judge JSON is not an object")
    return data


def answers_from_judge_json(
    data: dict[str, Any],
    questions: Sequence[Question],
    *,
    method: str = "llm_json",
    **extra: Any,
) -> dict[str, Answer]:
    out: dict[str, Answer] = {}
    for q in questions:
        entry = data.get(q.id)
        options = q.options()
        if not isinstance(entry, dict):
            out[q.id] = answer_from_probabilities(q, {}, method=f"{method}_missing", **extra)
            continue
        answer = str(entry.get("answer", "")).strip().lower()
        try:
            p = min(1.0, max(0.0, float(entry.get("probability", 0.5))))
        except (TypeError, ValueError):
            p = 0.5
        if q.type == "score":
            answer = re.sub(r"[^0-9]", "", answer) or "0"
        matched = next((o for o in options if o.lower() == answer), None)
        if matched is None:
            matched = next((o for o in options if answer and o.lower().startswith(answer)), None)
        if matched is None:
            out[q.id] = answer_from_probabilities(q, {}, method=f"{method}_unmatched", **extra)
            continue
        rest = (1 - p) / (len(options) - 1) if len(options) > 1 else 0.0
        probs = {o: (p if o == matched else rest) for o in options}
        out[q.id] = answer_from_probabilities(q, probs, method=method, **extra)
    return out


def out_of_tokens(content: str, reasoning: str, *, who: str = "judge") -> str:
    """Says where the budget went, so an empty answer after a long think is not mistaken for a
    model that produced nothing."""
    where = f" after {len(reasoning)} chars of reasoning" if reasoning else ""
    return f"{who} ran out of tokens{where}; JSON incomplete: {content[-80:]!r}"


def cut_off(exc: JudgeError) -> bool:
    return "ran out of tokens" in str(exc)


def retry_without_thinking(exc: JudgeError) -> bool:
    """The token budget is shared between thinking and the answer. A thinking model that uses
    it all returns empty or cut-off JSON; a server that predates thinking rejects the flag.
    Either way the same question is worth asking again without thinking."""
    message = str(exc).lower()
    return "think" in message or "json" in message


@dataclass(frozen=True)
class Reply:
    """One completion call: the answer text, the thinking before it, whether the budget ran out
    first, the model the server says it used, and how long it took."""

    content: str
    reasoning: str
    cut_off: bool
    snapshot: str
    latency_ms: float


def answers_from_replies(
    replies: Sequence[Reply], questions: Sequence[Question], *, method: str
) -> dict[str, Answer]:
    """Answers from the last reply, with every attempt's time billed to them."""
    last = replies[-1]
    if last.cut_off:
        raise JudgeError(out_of_tokens(last.content, last.reasoning))
    data = parse_judge_json(last.content)
    spent = sum(r.latency_ms for r in replies)
    return answers_from_judge_json(
        data,
        questions,
        method=method,
        latency_ms=spent / max(1, len(questions)),
        model_snapshot=last.snapshot,
    )


def judge_with_room(
    ask: Callable[[str], dict[str, Answer]],
    widen: Callable[[], None],
    without_thinking: Callable[[JudgeError], None],
) -> dict[str, Answer]:
    """Thinking at the normal budget; when cut off, once more with room; when that still fails
    for a reason thinking can cause, once without it. Three calls at most. `ask` takes the
    method name to record; `widen` and `without_thinking` adjust the request in place."""
    try:
        return ask("llm_json")
    except JudgeError as exc:
        if cut_off(exc):
            widen()
            try:
                return ask("llm_json")
            except JudgeError as again:
                exc = again
        if not retry_without_thinking(exc):
            raise
        without_thinking(exc)
        return ask("llm_json_nothink")


class LlmJudgeBackend:
    local = True

    def __init__(self, model: str, *, host: str = OLLAMA_HOST, think: bool = True) -> None:
        self.model = model
        self.host = host
        self.think = think
        self.name = f"llm:{model}"

    def decide(self, inp: JudgeInput, questions: Sequence[Question]) -> dict[str, Answer]:
        body: dict[str, Any] = {
            "model": self.model,
            "stream": False,
            "format": "json",
            # Bounds thinking plus the JSON answer; without it a judge can run past the timeout.
            "options": {"temperature": 0, "num_predict": MAX_JUDGE_TOKENS},
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": judge_prompt(inp.state_with_draft(), questions)},
            ],
        }
        # Ollama thinks by default for any model with the thinking capability, so leaving the
        # flag out is not "off": the model would spend the whole budget in `thinking` and
        # return an empty answer. Off has to be said explicitly.
        if not self.think:
            body["think"] = False
            return self._ask(body, questions, method="llm_json")
        body["think"] = True
        replies: list[Reply] = []

        def ask(method: str) -> dict[str, Answer]:
            replies.append(self._complete(body))
            return answers_from_replies(replies, questions, method=method)

        def widen() -> None:
            body["options"]["num_predict"] = RETRY_JUDGE_TOKENS

        def without_thinking(exc: JudgeError) -> None:
            # Answer without thinking and say so in the method, so the report can count how
            # often reasoning was dropped.
            if "http" in str(exc).lower():
                # The server rejected the flag itself (it predates thinking): drop it.
                body.pop("think", None)
            else:
                body["think"] = False
            # Grammar-constrained JSON with no thinking channel is where Gemma-family models
            # emit whitespace until the budget is gone; free text with the object extracted
            # by regex is the sturdier shape for the retry.
            body.pop("format", None)
            body["options"]["num_predict"] = MAX_JUDGE_TOKENS

        return judge_with_room(ask, widen, without_thinking)

    def _ask(
        self, body: dict[str, Any], questions: Sequence[Question], *, method: str
    ) -> dict[str, Answer]:
        return answers_from_replies([self._complete(body)], questions, method=method)

    def _complete(self, body: dict[str, Any]) -> Reply:
        response, latency = http.post_json(f"{self.host}/api/chat", body)
        message = response.get("message") or {}
        return Reply(
            content=str(message.get("content") or ""),
            reasoning=str(message.get("thinking") or ""),
            cut_off=response.get("done_reason") == "length",
            snapshot=str(response.get("model") or self.model),
            latency_ms=latency,
        )
