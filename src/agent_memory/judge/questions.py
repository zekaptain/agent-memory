"""The typed question set (`questions_v3`) and the action derived from its answers.

Stage A is asked once per item from the projection alone; Stage B once per draft. The write
rule reads only Stage B: a lesson is written when it is grounded, faithful, reusable beyond
the turn, and enforced by nothing else. `failed` and `recurrence` describe the turn and are
scored on their own, but no longer gate a write. `questions_v1` runs (exp1, exp2) keep their
rule through `write_gates`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from .lesson import Claim, Lesson

if TYPE_CHECKING:
    from .backends.base import Answer

QUESTIONS_VERSION = "questions_v3"

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

# Where a lesson may already be enforced. Shared by the `enforced` question and the
# `enforced_by` label, so the judge and the reviewer answer from the same options.
ENFORCED: dict[str, str] = {
    "none": "Nothing else makes the agent behave this way; memory would be the only place.",
    "harness": "The harness already enforces it (Mission prompts, system prompt, subagent "
    "limits). In a session a parent agent started, its instructions count as harness.",
    "project_docs": "The repository already says it (AGENTS.md, contributing guide, conventions).",
    "tooling": "A tool would catch the slip anyway (lint, type check, CI, hook).",
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
        "incomplete",
        "noul",
        "The turn ended before the agent finished: it was cut off by the next message or by the "
        "end of the session, so whether the request was carried out cannot be read from it.",
        "A",
        {
            "true": "The header says closed_by=next_message or closed_by=session_end and the "
            "agent's last visible step is not a finished result: no final message that closes "
            "the request, or work still in progress.",
            "false": "The turn reached its outcome, or the agent's final message and tool "
            "results show the request was carried through (or clearly failed) before the turn "
            "ended.",
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
        "grounded",
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
        "faithful",
        "noul",
        "The DRAFT LESSON's account of what happened in this turn is right: it does not misread "
        "the evidence it cites.",
        "B",
        {
            "true": "The lesson's description of the cause, the error, or the fact matches what "
            "the cited lines actually show.",
            "false": "The lesson misattributes the cause, describes an error or result that did "
            "not happen as stated, or draws a rule the evidence contradicts.",
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
    Question(
        "enforced",
        "choice",
        "Is the DRAFT LESSON already enforced somewhere the agent will meet anyway, so that "
        "memory would add nothing? Judge the lesson, not the turn.",
        "B",
        ENFORCED,
    ),
)

ALL_QUESTIONS: tuple[Question, ...] = STAGE_A + STAGE_B
BY_ID: dict[str, Question] = {q.id: q for q in ALL_QUESTIONS}


def claim_questions(claims: Sequence[Claim]) -> tuple[Question, ...]:
    criteria = {
        "grounded": {
            "true": "This claim can be traced to a relevant cited entry.",
            "false": "The cited entries do not support this claim, "
            "or it cites no relevant evidence.",
        },
        "faithful": {
            "true": "This claim's cause, error, or fact matches what its cited entries show.",
            "false": "This claim misreads its cited entries "
            "or describes something they do not show.",
        },
    }
    out = []
    for i, claim in enumerate(claims, 1):
        lines = ", ".join(map(str, claim.cited)) or "nothing"
        for key, instruction in (
            ("grounded", "is supported by the lines it cites"),
            (
                "faithful",
                "reads those lines correctly: the cause, error, or fact it states matches "
                "what they show",
            ),
        ):
            out.append(
                Question(
                    f"{key}.{i}",
                    "noul",
                    f"Claim {i} of the DRAFT LESSON {instruction}. "
                    f"Claim {i}: {claim.text} (cites {lines})",
                    "B",
                    criteria[key],
                )
            )
    return tuple(out)


def stage_b_for(lesson: Lesson | None) -> tuple[Question, ...]:
    if lesson is None or not lesson.claims:
        return STAGE_B
    return claim_questions(lesson.claims) + tuple(
        q for q in STAGE_B if q.id not in ("grounded", "faithful")
    )


def aggregate_claims(answers: Mapping[str, Answer]) -> dict[str, Answer]:
    """Derived draft gates cost nothing: every claim must clear each gate."""
    from .backends.base import answer_from_probabilities

    out = {}
    for key in ("grounded", "faithful"):
        probabilities = [
            float(answer.probabilities["yes"])
            for qid, answer in answers.items()
            if qid.startswith(key + ".")
        ]
        if probabilities:
            p = min(probabilities)
            out[key] = answer_from_probabilities(
                BY_ID[key], {"yes": p, "no": 1 - p}, method="min_over_claims"
            )
    return out


# The questions whose probability of clearing gates a write, per questions version.
WRITE_GATES: tuple[str, ...] = ("grounded", "faithful", "generalizable", "enforced")
WRITE_GATES_V1: tuple[str, ...] = ("failed", "recurrence", "supported")

DEFAULT_THRESHOLDS: dict[str, float] = {
    "failed": 0.5,
    "recurrence": 0.5,
    "supported": 0.5,
    "grounded": 0.5,
    "faithful": 0.5,
    "generalizable": 0.5,
    "enforced": 0.5,
    "band": 0.15,
}

Action = Literal["write_lesson", "review", "discard"]


@dataclass(frozen=True)
class Derived:
    action: Action
    reasons: list[str] = field(default_factory=list)


def write_gates(questions_version: str | None) -> tuple[str, ...]:
    return WRITE_GATES_V1 if questions_version == "questions_v1" else WRITE_GATES


def gate_probability(qid: str, probabilities: Mapping[str, float]) -> float | None:
    """The probability that a gate is cleared: p(yes) for a noul question, the mass on levels
    1 and 2 for `generalizable`, and p(none) for `enforced`."""
    if qid == "generalizable":
        if "1" not in probabilities and "2" not in probabilities:
            return None
        return float(probabilities.get("1", 0.0)) + float(probabilities.get("2", 0.0))
    if qid == "enforced":
        return probabilities.get("none")
    return probabilities.get("yes")


def gate_probabilities(
    probabilities: Mapping[str, Mapping[str, float]], gates: Sequence[str] = WRITE_GATES
) -> dict[str, float]:
    """`{question id: probabilities}` from a verdict's answers, reduced to the write gates."""
    out: dict[str, float] = {}
    for qid in gates:
        probs = probabilities.get(qid)
        if probs is None:
            continue
        p = gate_probability(qid, probs)
        if p is not None:
            out[qid] = p
    return out


def derive_action(
    p_true: Mapping[str, float],
    thresholds: Mapping[str, float] | None = None,
    *,
    gates: Sequence[str] = WRITE_GATES,
    declined: bool = False,
) -> Derived:
    """write_lesson when every gate clears its threshold; review when any gate sits inside the
    uncertainty band; discard otherwise, and always for a draft the drafter declined."""
    if declined:
        return Derived("discard", ["drafter declined: nothing worth keeping"])
    t = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    band = t["band"]
    reasons = []
    uncertain = False
    all_above = True
    for key in gates:
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
