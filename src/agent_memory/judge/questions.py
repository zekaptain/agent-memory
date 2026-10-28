"""The typed question set (`questions_v1`) and the action derived from its answers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

QUESTIONS_VERSION = "questions_v1"

QuestionType = Literal["noul", "choice", "score"]
Stage = Literal["A", "B"]


@dataclass(frozen=True)
class Question:
    id: str
    type: QuestionType
    instructions: str
    stage: Stage
    # choice: {option: description}; score: ordered level descriptions; noul: {true, false}.
    criteria: Any = None

    def options(self) -> list[str]:
        if self.type == "noul":
            return ["yes", "no"]
        if self.type == "choice":
            return list(self.criteria)
        return [str(i) for i in range(len(self.criteria))]

    def option_descriptions(self) -> list[str]:
        if self.type == "noul":
            crit = self.criteria or {}
            return [
                f"yes: {crit.get('true', 'the statement holds')}",
                f"no: {crit.get('false', 'the statement does not hold')}",
            ]
        if self.type == "choice":
            return [f"{k}: {v}" for k, v in self.criteria.items()]
        return [f"{i}: {desc}" for i, desc in enumerate(self.criteria)]


CATEGORIES: dict[str, str] = {
    "wrong_assumption": "The agent assumed something about the codebase, task, or environment "
    "that was false (wrong file, wrong API, misread intent).",
    "tool_misuse": "The agent used a tool incorrectly or repeatedly hit the same tool error "
    "(bad arguments, wrong command, ignored an error).",
    "stale_or_missing_knowledge": "The agent lacked or relied on outdated project knowledge "
    "(a renamed module, a changed convention, a prior decision it did not know).",
    "environment_or_provider": "The failure came from the environment or the model provider "
    "(rate limit, outage, missing dependency, flaky network), not from the agent's work.",
    "user_redirect": "The user changed scope, interrupted to redirect, or stopped the agent "
    "without the work being wrong.",
    "none": "No failure: the turn achieved what the user asked.",
}

STAGE_A: tuple[Question, ...] = (
    Question(
        "failed",
        "noul",
        "The agent did not accomplish what the user asked in this turn. Judge by the tool "
        "results, the recorded outcome, and the user's next message, not by the agent's own "
        "claims of completion.",
        "A",
        {
            "true": "Tool errors went unresolved, the outcome was cancelled or errored, or the "
            "user's next message shows the work was wrong or incomplete.",
            "false": "The evidence shows the requested work was done, or the user moved on "
            "without objection.",
        },
    ),
    Question(
        "user_corrected",
        "noul",
        "The user's next message corrects, rejects, or redirects the agent's work from this turn.",
        "A",
        {
            "true": "The next message says the result was wrong, asks to revert or redo, or "
            "points out something the agent missed.",
            "false": "The next message is a new request, a thanks, or unrelated.",
        },
    ),
    Question(
        "env_confirmed",
        "noul",
        "A tool result in this turn independently confirms the agent's claimed completion "
        "(tests passed, a command exited cleanly, a check succeeded).",
        "A",
        {
            "true": "At least one tool result after the last change shows success.",
            "false": "No tool result confirms it; the agent only asserted completion.",
        },
    ),
    Question(
        "category",
        "choice",
        "Which best describes what went wrong in this turn, if anything?",
        "A",
        CATEGORIES,
    ),
    Question(
        "recurrence",
        "noul",
        "The cause of the problem in this turn is specific to this project and likely to recur "
        "in future sessions on it.",
        "A",
        {
            "true": "A project convention, a tool or dependency quirk, or a codebase fact that "
            "will trip the agent again.",
            "false": "A one-off slip, a transient outage, or something the user changed their "
            "mind about.",
        },
    ),
)

STAGE_B: tuple[Question, ...] = (
    Question(
        "supported",
        "noul",
        "The DRAFT LESSON is supported by the evidence it cites in this turn.",
        "B",
        {
            "true": "Every claim in the lesson can be traced to a cited tool result, outcome, "
            "or user message shown above.",
            "false": "The lesson asserts something the cited evidence does not show, or cites "
            "nothing relevant.",
        },
    ),
    Question(
        "generalizable",
        "score",
        "How reusable is the DRAFT LESSON beyond this exact turn?",
        "B",
        [
            "One-off: tied to this specific request; would not help another session.",
            "Project-reusable: helps future sessions on this project.",
            "General practice: helps on any project.",
        ],
    ),
    Question(
        "actionable",
        "noul",
        "The DRAFT LESSON says what to do differently next time, not only what happened.",
        "B",
        {
            "true": "It names a concrete action, check, or command to use or avoid.",
            "false": "It only describes the failure or states a vague intention.",
        },
    ),
)

ALL_QUESTIONS: tuple[Question, ...] = STAGE_A + STAGE_B
BY_ID: dict[str, Question] = {q.id: q for q in ALL_QUESTIONS}

DEFAULT_THRESHOLDS: dict[str, float] = {
    "failed": 0.5,
    "recurrence": 0.5,
    "supported": 0.5,
    "band": 0.15,
}

Action = Literal["write_lesson", "review", "discard"]


@dataclass(frozen=True)
class Derived:
    action: Action
    reasons: list[str] = field(default_factory=list)


def derive_action(
    p_true: Mapping[str, float], thresholds: Mapping[str, float] | None = None
) -> Derived:
    """write_lesson when failed, recurrence, and supported all clear their thresholds; review
    when any of the three sits inside the uncertainty band; discard otherwise."""
    t = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    band = t["band"]
    reasons = []
    uncertain = False
    all_above = True
    for key in ("failed", "recurrence", "supported"):
        p = p_true.get(key)
        if p is None:
            reasons.append(f"{key}: no answer")
            all_above = False
            continue
        if abs(p - t[key]) < band:
            uncertain = True
            reasons.append(f"{key}={p:.2f} within {band:.2f} of {t[key]:.2f}")
        elif p < t[key]:
            all_above = False
            reasons.append(f"{key}={p:.2f} below {t[key]:.2f}")
    if uncertain:
        return Derived("review", reasons)
    if all_above:
        return Derived("write_lesson", reasons or ["all thresholds cleared"])
    return Derived("discard", reasons)


def thresholds_from_run(run: Mapping[str, Any], backend: str | None = None) -> dict[str, float]:
    """Defaults, then the run's thresholds, then whatever was frozen for this backend."""
    frozen = (run.get("frozen") or {}).get("backends", {}).get(backend or "", {})
    return {**DEFAULT_THRESHOLDS, **run.get("thresholds", {}), **frozen.get("thresholds", {})}


def questions_for(stage: str | None) -> Sequence[Question]:
    if stage == "A":
        return STAGE_A
    if stage == "B":
        return STAGE_B
    return ALL_QUESTIONS
