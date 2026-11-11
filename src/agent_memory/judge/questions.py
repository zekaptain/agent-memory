"""The typed question set (`questions_v4`) and which of its questions gate a write.

Turn questions are asked once per item from the projection alone; draft questions once per
draft. The write rule reads only draft questions: a lesson is written when it is grounded,
faithful, reusable beyond the turn, enforced by nothing else, and its exact rule follows from
the evidence. `failed` and `recurrence` describe the turn, not admission. Historical versions
keep their gates through `write_gates`. How answers become an action is in `verdict.py`.

A question the judge did not usably answer is marked `unanswered` and never read as a
probability: on a write gate it sends the verdict to review.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from .lesson import Claim, Lesson

QUESTIONS_VERSION = "questions_v4"

QuestionType = Literal["noul", "choice", "score"]


@dataclass(frozen=True)
class Question:
    id: str
    type: QuestionType
    instructions: str
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

TURN_QUESTIONS: tuple[Question, ...] = (
    Question(
        "failed",
        "noul",
        "The agent did not accomplish what the user asked in this turn. Judge by the tool "
        "results, the recorded outcome, and the user's next message, not by the agent's own "
        "claims of completion.",
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
        {
            "true": "At least one tool result after the last change shows success.",
            "false": "No tool result confirms it; the agent only asserted completion.",
        },
    ),
    Question(
        "category",
        "choice",
        "Which best describes what went wrong in this turn, if anything?",
        CATEGORIES,
    ),
    Question(
        "recurrence",
        "noul",
        "The cause of the problem in this turn is specific to this project and likely to recur "
        "in future sessions on it.",
        {
            "true": "A project convention, a tool or dependency quirk, or a codebase fact that "
            "will trip the agent again.",
            "false": "A one-off slip, a transient outage, or something the user changed their "
            "mind about.",
        },
    ),
)

DRAFT_QUESTIONS: tuple[Question, ...] = (
    Question(
        "grounded",
        "noul",
        "The DRAFT LESSON is supported by the evidence it cites in this turn.",
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
        {
            "true": "The lesson's description of the cause, the error, or the fact matches what "
            "the cited lines actually show.",
            "false": "The lesson misattributes the cause, describes an error or result that did "
            "not happen as stated, or draws a rule the evidence contradicts.",
        },
    ),
    Question(
        "rule_supported",
        "noul",
        "Do the supported claims and cited evidence justify this EXACT rule, including its "
        "scope and conditions, without an unsupported causal leap or overgeneralization? "
        "Judge the whole DRAFT LESSON, not whether its individual claims are true.",
        {
            "true": "The evidence warrants this exact rule and its scope and conditions.",
            "false": "True claims do not establish the proposed rule, its causal explanation, "
            "scope, or conditions; the rule overgeneralizes or goes beyond the evidence.",
        },
    ),
    Question(
        "generalizable",
        "score",
        "How reusable is the DRAFT LESSON beyond this exact turn?",
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
        ENFORCED,
    ),
)

ALL_QUESTIONS: tuple[Question, ...] = TURN_QUESTIONS + DRAFT_QUESTIONS
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
                    criteria[key],
                )
            )
    return tuple(out)


def draft_questions_for(lesson: Lesson | None) -> tuple[Question, ...]:
    if lesson is None or not lesson.claims:
        return DRAFT_QUESTIONS
    return claim_questions(lesson.claims) + tuple(
        q for q in DRAFT_QUESTIONS if q.id not in ("grounded", "faithful")
    )


# How a backend marks a question its judge did not usably answer: the answer's method ends in
# one of these. The probabilities stored beside it are a uniform placeholder, never evidence.
UNANSWERED: tuple[str, ...] = ("_missing", "_unmatched", "_no_probability")


def unanswered(method: str) -> bool:
    return method.endswith(UNANSWERED)


# The questions whose probability of clearing gates a write, per questions version.
WRITE_GATES_V2: tuple[str, ...] = ("grounded", "faithful", "generalizable", "enforced")
WRITE_GATES: tuple[str, ...] = (*WRITE_GATES_V2, "rule_supported")
WRITE_GATES_V1: tuple[str, ...] = ("failed", "recurrence", "supported")

DEFAULT_THRESHOLDS: dict[str, float] = {
    "failed": 0.5,
    "recurrence": 0.5,
    "supported": 0.5,
    "grounded": 0.5,
    "faithful": 0.5,
    "rule_supported": 0.5,
    "generalizable": 0.5,
    "enforced": 0.5,
    "band": 0.15,
}


def write_gates(questions_version: str | None) -> tuple[str, ...]:
    if questions_version == "questions_v1":
        return WRITE_GATES_V1
    return WRITE_GATES if questions_version == QUESTIONS_VERSION else WRITE_GATES_V2


def thresholds_from_run(run: Mapping[str, Any], backend: str | None = None) -> dict[str, float]:
    """Defaults, then the run's thresholds, then whatever was frozen for this backend."""
    frozen = (run.get("frozen") or {}).get("backends", {}).get(backend or "", {})
    return {**DEFAULT_THRESHOLDS, **run.get("thresholds", {}), **frozen.get("thresholds", {})}
