"""The one interface every judge backend implements, and the answer shape they all return."""

from __future__ import annotations

import math
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

from ..lesson import Claim
from ..questions import Question
from ..store import JudgeError

REMOTE_ENV = "AGENT_MEMORY_JUDGE_ALLOW_REMOTE"


@dataclass
class Answer:
    """A typed decision: the chosen option, a distribution over options, and a confidence.

    Confidence is the largest probability, the signal JEV-as-a-Judge thresholds on. It says
    how concentrated the distribution is, not whether the answer is right.
    """

    value: str | float
    probabilities: dict[str, float]
    confidence: float
    method: str
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    model_snapshot: str = ""
    raw: dict[str, Any] | None = field(default=None, compare=False)

    @property
    def p_true(self) -> float | None:
        return self.probabilities.get("yes")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("raw", None)
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Answer:
        return cls(
            value=data["value"],
            probabilities=dict(data["probabilities"]),
            confidence=float(data["confidence"]),
            method=str(data.get("method", "")),
            latency_ms=float(data.get("latency_ms", 0.0)),
            cost_usd=float(data.get("cost_usd", 0.0)),
            model_snapshot=str(data.get("model_snapshot", "")),
        )


@dataclass(frozen=True)
class JudgeInput:
    item_id: str
    state: str
    facts: dict[str, Any]
    draft: str | None = None
    flags: tuple[str, ...] = ()
    claims: tuple[Claim, ...] = ()

    def state_with_draft(self) -> str:
        if not self.draft:
            return self.state
        return f"{self.state}\n\nDRAFT LESSON:\n{self.draft}"


class DecisionBackend(Protocol):
    name: str
    local: bool

    def decide(self, inp: JudgeInput, questions: Sequence[Question]) -> dict[str, Answer]: ...


def normalize(probabilities: Mapping[str, float], options: Sequence[str]) -> dict[str, float]:
    """Clamp, fill missing options with zero, and renormalise to sum to one."""
    probs = {o: max(0.0, float(probabilities.get(o, 0.0))) for o in options}
    total = sum(probs.values())
    if total <= 0:
        return {o: 1.0 / len(options) for o in options}
    return {o: p / total for o, p in probs.items()}


def answer_from_probabilities(
    question: Question, probabilities: Mapping[str, float], *, method: str, **extra: Any
) -> Answer:
    probs = normalize(probabilities, question.options())
    if question.type == "score":
        value: str | float = sum(int(k) * p for k, p in probs.items())
    else:
        value = max(probs, key=lambda k: probs[k])
    return Answer(
        value=value, probabilities=probs, confidence=max(probs.values()), method=method, **extra
    )


def softmax(logits: Mapping[str, float]) -> dict[str, float]:
    if not logits:
        return {}
    peak = max(logits.values())
    weights = {k: math.exp(v - peak) for k, v in logits.items()}
    total = sum(weights.values())
    return {k: w / total for k, w in weights.items()}


def remote_allowed() -> bool:
    return os.environ.get(REMOTE_ENV, "") == "1"


def require_remote(what: str) -> None:
    if not remote_allowed():
        raise JudgeError(
            f"{what} sends redacted state to a hosted service. Set {REMOTE_ENV}=1 to allow it."
        )


_SPEC = re.compile(r"^(?P<kind>[a-z]+)(?::(?P<arg>.+))?$")


def make_backend(spec: str) -> DecisionBackend:
    """`rules`, `readout:<ollama model>`, `llm:<ollama model>`, `llamacpp:<llama-server model>`,
    `jev[:<openrouter model>]`."""
    match = _SPEC.match(spec.strip())
    if not match:
        raise JudgeError(f"bad backend spec {spec!r}")
    kind, arg = match["kind"], match["arg"]
    if kind == "rules":
        from .rules import RulesBackend

        return RulesBackend()
    if kind == "readout":
        from .readout import ReadoutBackend

        if not arg:
            raise JudgeError("readout needs a model: readout:<ollama model>")
        return ReadoutBackend(model=arg)
    if kind == "llm":
        from .llm import LlmJudgeBackend

        if not arg:
            raise JudgeError("llm needs a model: llm:<ollama model>")
        return LlmJudgeBackend(model=arg)
    if kind == "llamacpp":
        from .llamacpp import LlamaCppJudgeBackend

        if not arg:
            raise JudgeError("llamacpp needs a model: llamacpp:<llama-server model id or alias>")
        return LlamaCppJudgeBackend(model=arg)
    if kind == "jev":
        from .jev import JevBackend

        return JevBackend(model=arg) if arg else JevBackend()
    raise JudgeError(
        f"unknown backend {kind!r}; choose rules, readout:M, llm:M, llamacpp:M, or jev"
    )
