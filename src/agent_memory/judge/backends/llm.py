"""The reasoning baseline: a local Ollama model prompted with the rubric, answering in JSON.

Its self-reported probability is the confidence. The literature says that number is poorly
calibrated; measuring it against the decision backends is part of the experiment.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Sequence
from typing import Any

from ..questions import Question
from ..store import JudgeError
from . import http
from .base import Answer, JudgeInput, answer_from_probabilities

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
MAX_JUDGE_TOKENS = 4096

SYSTEM = (
    "You evaluate one turn of an AI coding agent. The evidence is a numbered projection of the "
    "turn: the user's request, tool calls and their results, the agent's final message, the "
    "recorded outcome, and the user's next message. Judge only from that evidence. Do not trust "
    "the agent's own claims of success. Answer every question. Return only JSON."
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
        if not self.think:
            return self._ask(body, questions, method="llm_json")
        body["think"] = True
        try:
            return self._ask(body, questions, method="llm_json")
        except JudgeError as exc:
            # The token budget is shared between thinking and the answer. A thinking model
            # that uses it all returns empty or cut-off JSON; a server that predates thinking
            # rejects the flag. Either way, answer the same question without thinking and
            # say so in the method, so the report can count how often reasoning was dropped.
            message = str(exc).lower()
            if "think" not in message and "json" not in message:
                raise
            body.pop("think")
            # Grammar-constrained JSON with no thinking channel is where Gemma-family models
            # emit whitespace until the budget is gone; free text with the object extracted
            # by regex is the sturdier shape for the retry.
            body.pop("format", None)
            return self._ask(body, questions, method="llm_json_nothink")

    def _ask(
        self, body: dict[str, Any], questions: Sequence[Question], *, method: str
    ) -> dict[str, Answer]:
        response, latency = http.post_json(f"{self.host}/api/chat", body)
        content = str((response.get("message") or {}).get("content") or "")
        if response.get("done_reason") == "length":
            raise JudgeError(f"judge ran out of tokens; JSON incomplete: {content[-80:]!r}")
        data = parse_judge_json(content)
        snapshot = str(response.get("model") or self.model)
        per_question = latency / max(1, len(questions))
        return answers_from_judge_json(
            data, questions, method=method, latency_ms=per_question, model_snapshot=snapshot
        )
