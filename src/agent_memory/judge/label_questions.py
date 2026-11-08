"""Display wording only. Model questions and stored label definitions stay unchanged."""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass

from .labeling import CATEGORY_ORDER, ENFORCED_ORDER, LESSON_ORDER


def display_safe(value: str) -> str:
    """Make terminal controls visible, including ESC, C1, bidi overrides and carriage returns."""
    return "".join(
        ch
        if ch in "\n\t\u200c\u200d" or unicodedata.category(ch) not in ("Cc", "Cf", "Cs")
        else f"\\u{ord(ch):04x}"
        for ch in value
    )


@dataclass(frozen=True)
class Option:
    id: str
    label: str


NAMES = {
    "wrong_assumption": "Wrong assumption",
    "tool_misuse": "Tool misuse",
    "stale_or_missing_knowledge": "Missing or outdated knowledge",
    "environment_or_provider": "Environment or provider",
    "user_redirect": "User redirect",
    "none": "Nothing else",
    "project_fact": "Project fact",
    "tool_quirk": "Tool behavior",
    "process": "Working practice",
    "environment": "Machine or setup",
    "user_preference": "User preference",
    "harness": "Harness or parent-agent instructions",
    "project_docs": "Project documentation",
    "tooling": "Tools, checks, or CI",
}
YN = [Option("y", "Yes"), Option("n", "No")]


def question(step: str, slots: list[str]) -> tuple[str, list[Option]]:
    key = step.rsplit(".", 1)[-1]
    per_claim = ".c" in step
    if key == "failed":
        return "Did the agent fail?", [*YN, Option("u", "Unclear"), Option("i", "Incomplete")]
    if key == "worth_remembering":
        return "Would a lesson help a future session?", YN
    if key == "error_intended":
        return "Was the tool error expected and used?", YN
    if key == "grounded":
        if per_claim:
            return "Does this claim have evidence in the lines it cites?", YN
        return "Does every claim have evidence?", YN
    if key == "faithful":
        if per_claim:
            return "Does this claim read those lines correctly?", YN
        return "Does the lesson read the evidence correctly?", YN
    if key == "generalizable":
        return "Where would this lesson help?", [
            Option("0", "Only this request"),
            Option("1", "This project"),
            Option("2", "Any project"),
        ]
    if key in ("category", "lesson_kind", "enforced_by"):
        title, keys = {
            "category": ("What went wrong?", CATEGORY_ORDER),
            "lesson_kind": ("What kind of lesson?", LESSON_ORDER),
            "enforced_by": ("What already enforces this lesson?", ENFORCED_ORDER),
        }[key]
        return title, [
            Option(str(i), "No failure" if key == "none" and step == "category" else NAMES[key])
            for i, key in enumerate(keys, 1)
        ]
    if key == "preferred":
        return "Which draft would you keep?", [
            *(Option(slot.lower(), f"Draft {slot}") for slot in slots),
            Option("same", "Both equally"),
            Option("neither", "Neither"),
        ]
    if key == "note":
        return "Add a note?", [Option("edit", "Add/edit note"), Option("continue", "Continue")]
    if key == "summary":
        return "Save these labels?", [Option("save", "Save"), Option("redo", "Redo")]
    if key == "review":
        return "Keep the current labels?", [Option("keep", "Keep"), Option("redo", "Redo")]
    raise ValueError("unknown labeling question")


INTRO = "Judge the evidence, not the agent's claims.\nAnswers save per item. ? opens the guide."
GUIDE = """Labeling guide

Who is speaking
The session may be started by a human or a parent agent. Parent-agent instructions are
harness instructions. A parent's review is not the human requester's judgment.

Turn endings
Outcome means the turn reached a recorded result. Next message or session end may cut it
off. Look at the evidence, not only the agent's completion claim.

Failure
Yes: the agent did not accomplish the request. No: the evidence shows it did.
Unclear: the turn finished, but the evidence cannot settle success or failure.
Incomplete: it was cut off before finishing, so success or failure cannot be read from it.
When failed, choose the main cause.

Worth remembering
Would a lesson help future sessions on this project or any project? One-off slips,
outages and user redirects usually do not. Choose the lesson's kind if worth remembering.

Expected errors
A failed tool call may be intentional: a probe, a negative test, or a search with no match.
Answer Yes only when the error was expected and used.

Drafts
Draft A, B, etc. are in seeded random order. Authors are hidden. No lesson means the author
declined; it has no draft questions. Citations refer to numbered entries in the projection.
Missing citation numbers provide no evidence. p opens the full stored projection.

Grounded versus faithful
Grounded: every claim traces to relevant evidence, especially the cited entries. No if a
cause or rule is asserted without evidence.
Faithful: the lesson reads that evidence correctly, including its cause, error and facts.
A lesson can cite the right entry and still misread it.

For structured lessons, answer Grounded and Faithful for each highlighted claim separately.
Only that claim's citations appear below it. An uncited claim still needs both answers.
The draft is grounded or faithful only when every claim is.

Reuse and existing enforcement
Choose Only this request, This project, or Any project.
Judge the lesson, not the turn: is it already enforced by the harness (including parent-agent
instructions), project documentation (AGENTS.md and conventions), or tools (lint, CI, hooks)?
Nothing else means memory would be the only place.

Comparison, notes and saving
Choose a draft, Both equally, or Neither. A note is optional and literal text.
Back restores the previous answer. Changing it removes answers that no longer apply.
Review the summary before saving. Redo restarts this item. Skip leaves it for another session.
Quit keeps previously saved items; this item's unsaved answers are not written.
"""
