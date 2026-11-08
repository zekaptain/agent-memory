"""Label definitions and conversions shared by both interactive frontends.

Labels are saved per item in labels.jsonl so a session can stop and resume. Order is a seeded
shuffle of the unlabeled items, so stratum and backend output cannot bias the reviewer. The
drafts of an item are shown together as A, B, ... in a seeded order with the drafter hidden;
the row records which drafter sat in each slot. A changed label keeps its previous value in
the row's `history`, so the file stays append-only in spirit and the report can count how
many labels moved.
"""

from __future__ import annotations

import random
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .drafter import draft_hash
from .questions import CATEGORIES, ENFORCED

LABELS_VERSION = "labels_v2"
LABEL_KEYS = (
    "failed",
    "incomplete",
    "category",
    "worth_remembering",
    "lesson_kind",
    "error_intended",
    "drafts",
    "preferred",
    "note",
)
DRAFT_LABEL_KEYS = ("grounded", "faithful", "generalizable", "enforced_by")
CATEGORY_ORDER = tuple(CATEGORIES)
ENFORCED_BY = ENFORCED
ENFORCED_ORDER = tuple(ENFORCED_BY)
LESSON_KINDS = {
    "project_fact": "where things are, conventions, commands, how the project is run",
    "tool_quirk": "how a tool, CLI, or API behaves, and what trips it",
    "process": "how to work: verify before claiming, read before editing, run the tests",
    "environment": "a fact about this machine or setup: ports, paths, installed versions",
    "user_preference": "how this person wants things done or reported",
}
LESSON_ORDER = tuple(LESSON_KINDS)
SLOTS = "ABCD"
PREFERRED_OTHER = ("same", "neither")
_ENTRY = re.compile(r"^\[(\d+)\] ", re.MULTILINE)
EXCERPT_CHARS = 320


def cited_entries(projection: str, cited: Sequence[int]) -> list[tuple[int, str]]:
    """The numbered projection entries a draft cites, each cut to a short excerpt."""
    starts = [(int(m.group(1)), m.start()) for m in _ENTRY.finditer(projection)]
    out = []
    for i, (number, start) in enumerate(starts):
        if number not in cited:
            continue
        end = starts[i + 1][1] if i + 1 < len(starts) else len(projection)
        body = projection[start:end].strip()
        if len(body) > EXCERPT_CHARS:
            body = body[:EXCERPT_CHARS].rstrip() + " …"
        out.append((number, body))
    return out


@dataclass
class LabelSummary:
    labeled: int
    skipped: int
    remaining: int
    changed: int = 0


@dataclass(frozen=True)
class DraftView:
    """One draft as the reviewer sees it: a slot letter instead of the drafter's name."""

    slot: str
    drafter: str
    lesson: str | None
    cited: tuple[int, ...]

    @property
    def declined(self) -> bool:
        return self.lesson is None

    @property
    def hash(self) -> str | None:
        return draft_hash(self.lesson)


def order_drafts(item_id: str, rows: Sequence[Mapping[str, Any]], seed: int) -> list[DraftView]:
    """Seeded per item, so the same drafter does not always sit in slot A."""
    ordered = sorted(rows, key=lambda r: str(r.get("drafter") or ""))
    random.Random(f"{seed}:{item_id}").shuffle(ordered)
    return [
        DraftView(
            slot=SLOTS[i],
            drafter=str(row.get("drafter") or ""),
            lesson=row.get("lesson") or None,
            cited=tuple(int(n) for n in row.get("cited") or []),
        )
        for i, row in enumerate(ordered[: len(SLOTS)])
    ]


def steps_for(
    answers: Mapping[str, str], drafts: Sequence[DraftView], has_tool_errors: bool
) -> list[str]:
    steps = ["failed"]
    if answers.get("failed") == "y":
        steps.append("category")
    steps.append("worth_remembering")
    if answers.get("worth_remembering") == "y":
        steps.append("lesson_kind")
    if has_tool_errors:
        steps.append("error_intended")
    for draft in drafts:
        if not draft.declined:
            steps.extend(f"{draft.slot}.{key}" for key in DRAFT_LABEL_KEYS)
    if len(drafts) >= 2:
        steps.append("preferred")
    steps.append("note")
    return steps


def assemble(
    answers: Mapping[str, str], drafts: Sequence[DraftView], has_tool_errors: bool
) -> dict[str, Any]:
    """The stored row from raw answers. Only answers to steps still in the plan count, so an
    answer made moot by going back (a category after failed became n) is dropped."""
    plan = set(steps_for(answers, drafts, has_tool_errors))
    raw = {k: v for k, v in answers.items() if k in plan}
    failed = raw.get("failed")
    row: dict[str, Any] = {
        "failed": {"y": True, "n": False}.get(failed or ""),
        "incomplete": failed == "i",
        "category": CATEGORY_ORDER[int(raw["category"]) - 1] if "category" in raw else None,
        "worth_remembering": (
            raw["worth_remembering"] == "y" if "worth_remembering" in raw else None
        ),
        "lesson_kind": LESSON_ORDER[int(raw["lesson_kind"]) - 1] if "lesson_kind" in raw else None,
        "error_intended": raw["error_intended"] == "y" if "error_intended" in raw else None,
        "drafts": {},
        "preferred": None,
        "note": raw.get("note") or None,
    }
    for draft in drafts:
        entry: dict[str, Any] = {
            "slot": draft.slot,
            "draft_hash": draft.hash,
            "declined": draft.declined,
        }
        for key in DRAFT_LABEL_KEYS:
            value = raw.get(f"{draft.slot}.{key}")
            if value is None:
                entry[key] = None
            elif key == "generalizable":
                entry[key] = int(value)
            elif key == "enforced_by":
                entry[key] = ENFORCED_ORDER[int(value) - 1]
            else:
                entry[key] = value == "y"
        row["drafts"][draft.drafter] = entry
    preferred = raw.get("preferred")
    if preferred in PREFERRED_OTHER:
        row["preferred"] = preferred
    elif preferred:
        row["preferred"] = next((d.drafter for d in drafts if d.slot.lower() == preferred), None)
    return row


def raw_answers(row: Mapping[str, Any], drafts: Sequence[DraftView]) -> dict[str, str]:
    """The typed answers a stored row corresponds to, so a partial relabel can replan from it."""
    raw: dict[str, str] = {}
    if "failed" in row:
        if row.get("incomplete"):
            raw["failed"] = "i"
        else:
            raw["failed"] = {True: "y", False: "n"}.get(row["failed"], "u")
    if row.get("category") in CATEGORY_ORDER:
        raw["category"] = str(CATEGORY_ORDER.index(row["category"]) + 1)
    if row.get("worth_remembering") is not None:
        raw["worth_remembering"] = "y" if row["worth_remembering"] else "n"
    if row.get("lesson_kind") in LESSON_ORDER:
        raw["lesson_kind"] = str(LESSON_ORDER.index(row["lesson_kind"]) + 1)
    if row.get("error_intended") is not None:
        raw["error_intended"] = "y" if row["error_intended"] else "n"
    entries = row.get("drafts") or {}
    for draft in drafts:
        entry = entries.get(draft.drafter) or {}
        if entry.get("draft_hash") != draft.hash:
            continue  # the draft changed since; its labels no longer apply
        for key in DRAFT_LABEL_KEYS:
            value = entry.get(key)
            if value is None:
                continue
            if key == "generalizable":
                raw[f"{draft.slot}.{key}"] = str(int(value))
            elif key == "enforced_by":
                if value in ENFORCED_ORDER:
                    raw[f"{draft.slot}.{key}"] = str(ENFORCED_ORDER.index(value) + 1)
            else:
                raw[f"{draft.slot}.{key}"] = "y" if value else "n"
    preferred = row.get("preferred")
    if preferred in PREFERRED_OTHER:
        raw["preferred"] = str(preferred)
    elif preferred:
        slot = next((d.slot.lower() for d in drafts if d.drafter == preferred), None)
        if slot:
            raw["preferred"] = slot
    if "note" in row:
        raw["note"] = str(row.get("note") or "")
    return raw


def missing_steps(
    row: Mapping[str, Any], drafts: Sequence[DraftView], has_tool_errors: bool
) -> list[str]:
    raw = raw_answers(row, drafts)
    return [s for s in steps_for(raw, drafts, has_tool_errors) if s not in raw]


def _flatten(row: Mapping[str, Any]) -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in row.items():
        if key in ("item_id", "labeled_at", "relabeled_at", "history", "labels_version"):
            continue
        if key == "drafts" and isinstance(value, dict):
            for drafter, entry in value.items():
                for k, v in (entry or {}).items():
                    if k not in ("slot",):
                        flat[f"drafts.{drafter}.{k}"] = v
            continue
        flat[key] = value
    return flat


def changed_values(previous: Mapping[str, Any], new: Mapping[str, Any]) -> dict[str, Any]:
    """Keys whose value moved, with the old value. Keys that are new, or were unanswered
    (None) before, are not changes."""
    before = _flatten(previous)
    after = _flatten(new)
    return {k: v for k, v in before.items() if v is not None and k in after and after[k] != v}
