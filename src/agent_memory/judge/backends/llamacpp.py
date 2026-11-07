"""The reasoning baseline served by llama.cpp: the same rubric, prompt, and JSON answer shape
as `llm:`, spoken to `llama-server`'s OpenAI-compatible `/v1/chat/completions`.

It runs under its own backend name so one run can hold verdicts from both servers and the
report lays them side by side. Only an `llm:` backend can be the reference judge.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any

from ..questions import Question
from ..store import JudgeError
from . import http
from .base import Answer, JudgeInput
from .llm import (
    MAX_JUDGE_TOKENS,
    SYSTEM,
    answers_from_judge_json,
    judge_prompt,
    out_of_tokens,
    parse_judge_json,
    retry_without_thinking,
)

# 9931 is llama-server's own default port.
LLAMACPP_HOST = os.environ.get("LLAMACPP_HOST", "http://localhost:9931").rstrip("/")


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
        try:
            return self._ask(body, questions, method="llm_json")
        except JudgeError as exc:
            if not retry_without_thinking(exc):
                raise
            disable_thinking(body)
            # Same shape as the Ollama retry: free text with the object extracted by regex,
            # since a constrained grammar with no thinking channel can stall on whitespace.
            body.pop("response_format", None)
            return self._ask(body, questions, method="llm_json_nothink")

    def _ask(
        self, body: dict[str, Any], questions: Sequence[Question], *, method: str
    ) -> dict[str, Answer]:
        try:
            response, latency = http.post_json(f"{self.host}/v1/chat/completions", body)
        except JudgeError as exc:
            raise self._with_known_models(exc) from None
        choices = response.get("choices") or []
        first = choices[0] if choices and isinstance(choices[0], dict) else {}
        message = first.get("message") or {}
        content = str(message.get("content") or "")
        if first.get("finish_reason") == "length":
            raise JudgeError(out_of_tokens(content, str(message.get("reasoning_content") or "")))
        data = parse_judge_json(content)
        snapshot = str(response.get("model") or self.model)
        per_question = latency / max(1, len(questions))
        return answers_from_judge_json(
            data, questions, method=method, latency_ms=per_question, model_snapshot=snapshot
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
