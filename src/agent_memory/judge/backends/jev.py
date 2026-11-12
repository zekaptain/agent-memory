"""Hosted Jev through the OpenRouter Decisions API. Opt-in: it sends redacted text off-machine.

Request and response shapes follow TypeSafe's System One API (noul, choice, score) as served by
OpenRouter at /api/alpha/decisions. The model snapshot the gateway served is recorded with
every answer. Projection text is never logged by this module.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any

from ... import endpoints
from ...model_call import OPENROUTER_KEY_ENV
from ..questions import Question
from ..store import JudgeError, ServerUnreachable
from .base import Answer, JudgeInput, answer_from_probabilities, require_remote

OPENROUTER_URL = "https://openrouter.ai/api/alpha/decisions"
DEFAULT_MODEL = "typesafe/jev-1.14"


def question_body(q: Question) -> dict[str, Any]:
    body: dict[str, Any] = {"type": q.type, "instructions": q.instructions}
    if q.type == "noul" and q.criteria:
        body["criteria"] = {"true": q.criteria["true"], "false": q.criteria["false"]}
    elif q.type == "choice":
        body["criteria"] = dict(q.criteria)
    elif q.type == "score":
        body["criteria"] = list(q.criteria)
    return body


def request_body(model: str, state: str, questions: Sequence[Question]) -> dict[str, Any]:
    return {
        "model": model,
        "state": state,
        "questions": {q.id: question_body(q) for q in questions},
    }


def parse_answers(
    response: dict[str, Any], questions: Sequence[Question], *, latency_ms: float
) -> dict[str, Answer]:
    answers = response.get("answers")
    if not isinstance(answers, dict):
        raise JudgeError("Jev response has no answers object")
    snapshot = str(response.get("model") or "")
    try:
        cost = float((response.get("usage") or {}).get("cost") or 0.0)
    except (AttributeError, TypeError, ValueError):
        raise JudgeError("Jev response has a malformed usage object") from None
    per_question = latency_ms / max(1, len(questions))
    out: dict[str, Answer] = {}
    for q in questions:
        entry = answers.get(q.id)
        if not isinstance(entry, dict):
            raise JudgeError(f"Jev response lacks an answer for {q.id!r}")
        try:
            probs = _probabilities(q, entry)
            confidence = float(entry["confidence"]) if "confidence" in entry else None
            score = float(entry["score"]) if q.type == "score" and "score" in entry else None
        except (AttributeError, OverflowError, TypeError, ValueError):
            # The entry is model output, so the message names only the question.
            raise JudgeError(f"Jev answer for {q.id!r} is malformed") from None
        answer = answer_from_probabilities(
            q,
            probs,
            method="jev",
            latency_ms=per_question,
            cost_usd=cost / max(1, len(questions)),
            model_snapshot=snapshot,
        )
        # Jev's own confidence (for choice and score) is the distribution's concentration;
        # keep the gateway's number when it supplies one.
        if confidence is not None:
            answer.confidence = confidence
        if score is not None:
            answer.value = score
        out[q.id] = answer
    return out


def _probabilities(q: Question, entry: dict[str, Any]) -> dict[str, float]:
    """The distribution one entry states. An entry that states none is no answer."""
    if q.type == "noul":
        if entry.get("noul") is None:
            return {}
        p = float(entry["noul"])
        return {"yes": p, "no": 1 - p}
    raw = entry.get("probabilities") or {}
    if q.type == "choice":
        probs = {str(k): float(v) for k, v in raw.items()}
        if not probs and "choice" in entry:
            probs = {str(entry["choice"]): 1.0}
        return probs
    if isinstance(raw, list):
        probs = {str(i): float(v) for i, v in enumerate(raw)}
    else:
        probs = {str(k): float(v) for k, v in raw.items()}
    if not probs and "score" in entry:
        probs = {str(round(float(entry["score"]))): 1.0}
    return probs


class JevBackend:
    local = False

    def __init__(
        self, model: str = DEFAULT_MODEL, *, url: str = OPENROUTER_URL, api_key: str | None = None
    ) -> None:
        self.model = model
        self.url = url
        self.api_key = api_key
        self.name = f"jev:{model}"

    def _key(self) -> str:
        key = self.api_key or os.environ.get(OPENROUTER_KEY_ENV)
        if not key:
            raise ServerUnreachable(
                f"{OPENROUTER_KEY_ENV} is not set; the jev backend needs an OpenRouter key."
            )
        return key

    def decide(self, inp: JudgeInput, questions: Sequence[Question]) -> dict[str, Answer]:
        require_remote("The jev backend")
        key = self._key()
        body = request_body(self.model, inp.state_with_draft(), questions)
        try:
            response, latency = endpoints.post_json(
                self.url, body, headers={"Authorization": f"Bearer {key}"}, timeout=120
            )
        except endpoints.Unreachable as exc:
            raise ServerUnreachable(str(exc)) from None
        except endpoints.EgressError as exc:
            raise JudgeError(str(exc)) from None
        return parse_answers(response, questions, latency_ms=latency)
