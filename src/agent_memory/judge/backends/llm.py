"""The reasoning baseline: a local model prompted with the rubric, answering in JSON.

Its self-reported probability is the confidence. The literature says that number is poorly
calibrated; measuring it against the decision backends is part of the experiment.

The judge is the same on either server: `llm:` asks Ollama, `llamacpp:` asks llama-server.
Each runs under its own backend name, so one run can hold verdicts from both and the report
lays them side by side. Only an `llm:` backend can be the reference judge.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from typing import Any

from ... import model_call
from ...endpoints import LLAMACPP_HOST, OLLAMA_HOST
from ...model_call import Ask, LlamaCpp, Ollama, Server
from ..questions import Question
from ..store import JudgeError
from .base import Answer, JudgeInput, answer_from_probabilities

MAX_JUDGE_TOKENS = 4096
# Thinking shares the budget with the answer. A judge cut off mid-thought is asked once more
# with room before thinking is switched off: at temperature 0 the same budget stops at the
# same place, and a template with no thinking switch (gpt-oss) makes the attempt without
# thinking the same request again. Three times the budget, as for drafts; at gemma4:26b's
# speed on Ollama that still finishes inside the request timeout.
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


def _read(entry: Any, question: Question) -> tuple[str, float] | str:
    """The option an entry of the reply names and the probability it gives, or why the entry
    is unusable: `missing`, `unmatched`, or `no_probability`. Nothing is guessed in its
    place, because any default would read as the judge's own answer."""
    if not isinstance(entry, dict):
        return "missing"
    raw = entry.get("answer")
    # Only text and numbers can name an option: `str(None)` would name the option "none".
    named = isinstance(raw, (str, int, float)) and not isinstance(raw, bool)
    answer = str(raw).strip().lower() if named else ""
    if question.type == "score":
        answer = re.sub(r"[^0-9]", "", answer)
    options = question.options()
    matched = next((o for o in options if o.lower() == answer), None)
    if matched is None:
        matched = next((o for o in options if answer and o.lower().startswith(answer)), None)
    if matched is None:
        return "unmatched"
    raw_p = entry.get("probability")
    if raw_p is None or isinstance(raw_p, bool):
        return "no_probability"
    try:
        p = float(raw_p)
    except (TypeError, ValueError):
        return "no_probability"
    if not math.isfinite(p):
        return "no_probability"
    return matched, min(1.0, max(0.0, p))


def answers_from_judge_json(
    data: dict[str, Any],
    questions: Sequence[Question],
    *,
    method: str = "llm_json",
    **extra: Any,
) -> dict[str, Answer]:
    """One answer per question. A question the reply does not usably answer gets a uniform
    placeholder whose method ends in the reason, which `questions.unanswered` recognises."""
    out: dict[str, Answer] = {}
    for q in questions:
        read = _read(data.get(q.id), q)
        if isinstance(read, str):
            out[q.id] = answer_from_probabilities(q, {}, method=f"{method}_{read}", **extra)
            continue
        matched, p = read
        options = q.options()
        rest = (1 - p) / (len(options) - 1) if len(options) > 1 else 0.0
        probs = {o: (p if o == matched else rest) for o in options}
        out[q.id] = answer_from_probabilities(q, probs, method=method, **extra)
    return out


class ReasoningJudge:
    """The reasoning judge on one model server: thinking at the normal budget; when cut off,
    once more with room; when that still yields no answers, once without thinking. Three
    attempts at most. With `think=False` there is one attempt and nothing to fall back to."""

    def __init__(self, name: str, server: Server, *, think: bool = True) -> None:
        self.name = name
        self.server = server
        self.think = think
        self.local = server.local

    def decide(self, inp: JudgeInput, questions: Sequence[Question]) -> dict[str, Answer]:
        ask = Ask(
            prompt=judge_prompt(inp.state_with_draft(), questions),
            budget=MAX_JUDGE_TOKENS,
            system=SYSTEM,
            think=self.think,
        )
        try:
            if self.think:
                result = model_call.call(
                    self.server, ask, who="judge", wider=RETRY_JUDGE_TOKENS, drop_thinking=True
                )
            else:
                result = model_call.call(self.server, ask, who="judge")
        except model_call.ModelCallError as exc:
            raise JudgeError(str(exc)) from None
        return answers_from_judge_json(
            result.data,
            questions,
            # Said in the method, so the report can count how often thinking was dropped.
            method="llm_json_nothink" if result.thinking_dropped else "llm_json",
            # Every attempt's time is billed to the answers.
            latency_ms=result.latency_ms / max(1, len(questions)),
            model_snapshot=result.snapshot,
        )


def ollama_judge(model: str, *, host: str = OLLAMA_HOST, think: bool = True) -> ReasoningJudge:
    """The `llm:<model>` backend."""
    return ReasoningJudge(f"llm:{model}", Ollama(model, host), think=think)


def llamacpp_judge(model: str, *, host: str = LLAMACPP_HOST, think: bool = True) -> ReasoningJudge:
    """The `llamacpp:<model>` backend. It asks for thinking apart from the text, so the
    answer is not buried in it."""
    return ReasoningJudge(
        f"llamacpp:{model}", LlamaCpp(model, host, split_thinking=True), think=think
    )
