"""LLM-as-Jev readout: a frozen local model answers with one option letter, and the option
distribution is read from its token log-probabilities (Li and Wagle, 2026). Local only.

Ollama's native /api/chat returns per-token logprobs when asked. If a server does not, the
backend falls back to sampling and records which method produced each answer.
"""

from __future__ import annotations

import string
from collections.abc import Sequence
from typing import Any

from ... import endpoints
from ...endpoints import OLLAMA_HOST
from ..questions import Question
from ..store import JudgeError, ServerUnreachable
from .base import Answer, JudgeInput, answer_from_probabilities, softmax

LETTERS = string.ascii_uppercase
SAMPLES = 8

SYSTEM = (
    "You answer one multiple-choice question about a coding agent's turn, using only the "
    "evidence shown. Reply with the single letter of your answer and nothing else."
)


def readout_prompt(state: str, question: Question) -> str:
    lines = [f"{LETTERS[i]}. {desc}" for i, desc in enumerate(question.option_descriptions())]
    return (
        f"EVIDENCE\n{state}\n\nQUESTION\n{question.instructions}\n\n"
        + "\n".join(lines)
        + "\n\nAnswer with one letter."
    )


def _letter_logits(entries: Any, n_options: int) -> dict[str, float]:
    """Pick the first generated token position that carries option letters, and return
    letter -> logprob over the letters present."""
    if not isinstance(entries, list):
        return {}
    valid = set(LETTERS[:n_options])
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        candidates = entry.get("top_logprobs") or []
        if isinstance(entry.get("token"), str) and "logprob" in entry:
            candidates = [*candidates, {"token": entry["token"], "logprob": entry["logprob"]}]
        found: dict[str, float] = {}
        for cand in candidates:
            if not isinstance(cand, dict):
                continue
            token = str(cand.get("token", "")).strip().strip(".)").upper()
            if token in valid:
                try:
                    lp = float(cand["logprob"])
                except (KeyError, TypeError, ValueError):
                    continue
                found[token] = max(lp, found.get(token, float("-inf")))
        if found:
            return found
    return {}


def find_logprobs(response: dict[str, Any]) -> Any:
    for holder in (response, response.get("message") or {}):
        if isinstance(holder, dict) and holder.get("logprobs") is not None:
            return holder["logprobs"]
    return None


def first_letter(text: str, n_options: int) -> str | None:
    for ch in text.strip().upper():
        if ch in LETTERS[:n_options]:
            return ch
        if ch.isalnum():
            return None
    return None


class ReadoutBackend:
    local = True

    def __init__(
        self, model: str, *, host: str = OLLAMA_HOST, top_logprobs: int = 20, samples: int = SAMPLES
    ) -> None:
        self.model = model
        self.host = host
        self.top_logprobs = top_logprobs
        self.samples = samples
        self.name = f"readout:{model}"

    def decide(self, inp: JudgeInput, questions: Sequence[Question]) -> dict[str, Answer]:
        return {q.id: self._one(inp.state_with_draft(), q) for q in questions}

    def _request(self, prompt: str, *, temperature: float, seed: int | None, logprobs: bool) -> Any:
        body: dict[str, Any] = {
            "model": self.model,
            "stream": False,
            "think": False,
            "options": {"temperature": temperature, "num_predict": 4},
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": prompt},
            ],
        }
        if seed is not None:
            body["options"]["seed"] = seed
        if logprobs:
            body["logprobs"] = True
            body["top_logprobs"] = self.top_logprobs
        try:
            return endpoints.post_json(f"{self.host}/api/chat", body)
        except endpoints.Unreachable as exc:
            raise ServerUnreachable(str(exc)) from None
        except endpoints.EgressError as exc:
            raise JudgeError(str(exc)) from None

    def _one(self, state: str, question: Question) -> Answer:
        options = question.options()
        prompt = readout_prompt(state, question)
        response, latency = self._request(prompt, temperature=0.0, seed=None, logprobs=True)
        snapshot = str(response.get("model") or self.model)
        logits = _letter_logits(find_logprobs(response), len(options))
        if logits:
            letter_probs = softmax(logits)
            probs = {options[LETTERS.index(k)]: p for k, p in letter_probs.items()}
            return answer_from_probabilities(
                question, probs, method="logprobs", latency_ms=latency, model_snapshot=snapshot
            )
        # No logprobs from this server: estimate the distribution by sampling.
        counts = dict.fromkeys(options, 0)
        content = str((response.get("message") or {}).get("content") or "")
        letter = first_letter(content, len(options))
        if letter:
            counts[options[LETTERS.index(letter)]] += 1
        total_latency = latency
        for i in range(self.samples - 1):
            resp, lat = self._request(prompt, temperature=1.0, seed=1000 + i, logprobs=False)
            total_latency += lat
            text = str((resp.get("message") or {}).get("content") or "")
            letter = first_letter(text, len(options))
            if letter:
                counts[options[LETTERS.index(letter)]] += 1
        # Add-one smoothing so a unanimous sample is confident, not certain. Samples that
        # named no option are no answer, which smoothing would dress up as a uniform one.
        probs = {o: (c + 0.5) for o, c in counts.items()} if any(counts.values()) else {}
        return answer_from_probabilities(
            question,
            probs,
            method=f"sampling:{self.samples}",
            latency_ms=total_latency,
            model_snapshot=snapshot,
        )
