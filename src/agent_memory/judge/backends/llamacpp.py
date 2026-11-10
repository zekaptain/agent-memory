"""The reasoning baseline served by llama.cpp: the same rubric, prompt, and JSON answer shape
as `llm:`, spoken to `llama-server`'s OpenAI-compatible `/v1/chat/completions`.

It runs under its own backend name so one run can hold verdicts from both servers and the
report lays them side by side. Only an `llm:` backend can be the reference judge.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ...endpoints import LLAMACPP_HOST
from ..questions import Question
from ..store import JudgeError
from . import http
from .base import Answer, JudgeInput
from .llm import (
    MAX_JUDGE_TOKENS,
    RETRY_JUDGE_TOKENS,
    SYSTEM,
    Reply,
    answers_from_replies,
    judge_prompt,
    judge_with_room,
)


class LlamaCppJudgeBackend:
    local = True

    def __init__(self, model: str, *, host: str = LLAMACPP_HOST, think: bool = True) -> None:
        # In single-model mode llama-server ignores the request's model and answers with the
        # one it loaded; in router mode the name must be an id from /v1/models and the router
        # loads it on first use. Either way the response's own `model` field is what gets
        # recorded as the snapshot.
        self.model = model
        self.host = host
        self.think = think
        self.name = f"llamacpp:{model}"

    def decide(self, inp: JudgeInput, questions: Sequence[Question]) -> dict[str, Answer]:
        body: dict[str, Any] = {
            "model": self.model,
            "stream": False,
            "temperature": 0,
            # Bounds thinking plus the JSON answer; without it a judge can run past the timeout.
            "max_tokens": MAX_JUDGE_TOKENS,
            "response_format": {"type": "json_object"},
            # Parsed thoughts land in message.reasoning_content, leaving content as the answer.
            "reasoning_format": "deepseek",
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": judge_prompt(inp.state_with_draft(), questions)},
            ],
        }
        if not self.think:
            disable_thinking(body)
            return self._ask(body, questions, method="llm_json")
        # The template kwarg is how Qwen- and Gemma-style templates are switched; templates
        # without the switch ignore it and follow the server's --reasoning setting.
        body["chat_template_kwargs"] = {"enable_thinking": True}
        replies: list[Reply] = []

        def ask(method: str) -> dict[str, Answer]:
            replies.append(self._complete(body))
            return answers_from_replies(replies, questions, method=method)

        def widen() -> None:
            body["max_tokens"] = RETRY_JUDGE_TOKENS

        def without_thinking(exc: JudgeError) -> None:
            disable_thinking(body)
            body["max_tokens"] = MAX_JUDGE_TOKENS
            # Same shape as the Ollama retry: free text with the object extracted by regex,
            # since a constrained grammar with no thinking channel can stall on whitespace.
            body.pop("response_format", None)

        return judge_with_room(ask, widen, without_thinking)

    def _ask(
        self, body: dict[str, Any], questions: Sequence[Question], *, method: str
    ) -> dict[str, Answer]:
        return answers_from_replies([self._complete(body)], questions, method=method)

    def _complete(self, body: dict[str, Any]) -> Reply:
        try:
            response, latency = http.post_json(f"{self.host}/v1/chat/completions", body)
        except JudgeError as exc:
            raise self._with_known_models(exc) from None
        choices = response.get("choices") or []
        first = choices[0] if choices and isinstance(choices[0], dict) else {}
        message = first.get("message") or {}
        return Reply(
            content=str(message.get("content") or ""),
            reasoning=str(message.get("reasoning_content") or ""),
            cut_off=first.get("finish_reason") == "length",
            snapshot=str(response.get("model") or self.model),
            latency_ms=latency,
        )

    def _with_known_models(self, exc: JudgeError) -> JudgeError:
        """A router answers an unknown model id with a bare 'not found'; name the ids it serves
        so the spec can be corrected without a second trip to /v1/models."""
        message = str(exc)
        if "not found" not in message.lower():
            return exc
        try:
            listing = http.get_json(f"{self.host}/v1/models")
        except JudgeError:
            return exc
        ids = [str(m["id"]) for m in listing.get("data") or [] if isinstance(m, dict) and "id" in m]
        if not ids:
            return exc
        return JudgeError(f"{message}. Models this server knows: {', '.join(sorted(ids))}")


def disable_thinking(body: dict[str, Any]) -> None:
    # Both switches, because which one a template honours depends on the model.
    body["chat_template_kwargs"] = {"enable_thinking": False}
    body["reasoning_effort"] = "none"
