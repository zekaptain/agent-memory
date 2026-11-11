"""Label definitions and conversions shared by both interactive frontends.

Labels are saved per item in labels.jsonl so a session can stop and resume. Order is a seeded
shuffle of the unlabeled items, so stratum and backend output cannot bias the reviewer. The
drafts of an item are shown together as A, B, ... in a seeded order with the drafter hidden;
the row records which drafter sat in each slot. A changed label keeps its previous value in
the row's `history`, so the file stays append-only in spirit and the report can count how
many labels moved. Unclear is an answer like Yes or No, so a move off it counts.
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .drafter import draft_hash
from .lesson import LESSON_KINDS as LESSON_KINDS
from .lesson import Claim, lesson_from_row
from .lesson import cited_entries as cited_entries
from .lesson import entry_numbers as entry_numbers
from .questions import CATEGORIES, ENFORCED
from .store import JudgeError

LABELS_VERSION = "labels_v4"
DRAFT_LABEL_KEYS = (
    "grounded",
    "faithful",
    "rule_supported",
    "generalizable",
    "enforced_by",
    "store_exact",
)
UNKNOWN_LABEL_KEYS = ("rule_supported", "store_exact")
CATEGORY_ORDER = tuple(CATEGORIES)
ENFORCED_BY = ENFORCED
ENFORCED_ORDER = tuple(ENFORCED_BY)
LESSON_ORDER = tuple(LESSON_KINDS)
SLOTS = "ABCD"
PREFERRED_OTHER = ("same", "neither")


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
    claims: tuple[Claim, ...] = ()

    @property
    def declined(self) -> bool:
        return self.lesson is None

    @property
    def hash(self) -> str | None:
        return draft_hash(self.lesson)


def validate_draft_count(count: int) -> None:
    if count > len(SLOTS):
        raise JudgeError(f"Labeling supports at most {len(SLOTS)} drafts per item; found {count}.")


def order_drafts(item_id: str, rows: Sequence[Mapping[str, Any]], seed: int) -> list[DraftView]:
    """Seeded per item, so the same drafter does not always sit in slot A."""
    validate_draft_count(len(rows))
    ordered = sorted(rows, key=lambda r: str(r.get("drafter") or ""))
    random.Random(f"{seed}:{item_id}").shuffle(ordered)
    views = []
    for i, row in enumerate(ordered):
        lesson = lesson_from_row(row)
        views.append(
            DraftView(
                slot=SLOTS[i],
                drafter=str(row.get("drafter") or ""),
                lesson=row.get("lesson") or None,
                cited=tuple(int(n) for n in row.get("cited") or []),
                claims=tuple(lesson.claims) if lesson else (),
            )
        )
    return views


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
            if draft.claims:
                steps.extend(
                    f"{draft.slot}.c{i}.{key}"
                    for i in range(1, len(draft.claims) + 1)
                    for key in ("grounded", "faithful")
                )
                steps.extend(
                    f"{draft.slot}.{key}"
                    for key in DRAFT_LABEL_KEYS
                    if key not in ("grounded", "faithful")
                )
            else:
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
                entry[key] = {"y": True, "n": False}.get(value)
        unknown = [key for key in UNKNOWN_LABEL_KEYS if raw.get(f"{draft.slot}.{key}") == "u"]
        if unknown:
            entry["unknown"] = unknown
        if draft.claims:
            entry["claims"] = [
                {
                    key: (
                        raw[f"{draft.slot}.c{i}.{key}"] == "y"
                        if f"{draft.slot}.c{i}.{key}" in raw
                        else None
                    )
                    for key in ("grounded", "faithful")
                }
                for i in range(1, len(draft.claims) + 1)
            ]
            for key in ("grounded", "faithful"):
                values = [claim[key] for claim in entry["claims"]]
                entry[key] = (
                    False
                    if any(value is False for value in values)
                    else True
                    if all(value is True for value in values)
                    else None
                )
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
            if draft.claims and key in ("grounded", "faithful"):
                continue
            value = entry.get(key)
            if value is None:
                if key in UNKNOWN_LABEL_KEYS and key in (entry.get("unknown") or []):
                    raw[f"{draft.slot}.{key}"] = "u"
                continue
            if key == "generalizable":
                raw[f"{draft.slot}.{key}"] = str(int(value))
            elif key == "enforced_by":
                if value in ENFORCED_ORDER:
                    raw[f"{draft.slot}.{key}"] = str(ENFORCED_ORDER.index(value) + 1)
            else:
                raw[f"{draft.slot}.{key}"] = "y" if value else "n"
        if draft.claims:
            for i, claim in enumerate(entry.get("claims") or [], 1):
                for key in ("grounded", "faithful"):
                    if claim.get(key) is not None:
                        raw[f"{draft.slot}.c{i}.{key}"] = "y" if claim[key] else "n"
    same_drafts = {draft.drafter: draft.hash for draft in drafts} == {
        name: entry.get("draft_hash") for name, entry in entries.items()
    }
    preferred = row.get("preferred") if same_drafts else None
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


def has_tool_errors(item: Mapping[str, Any]) -> bool:
    return bool(item.get("facts", {}).get("tool_error_count"))


def questions_left(
    item: Mapping[str, Any], row: Mapping[str, Any], draft_rows: Sequence[Mapping[str, Any]]
) -> list[str]:
    """What a labeling session would still ask about an item that has a label. The slot a
    draft is shown in changes the names of its questions, not whether any is left."""
    drafts = order_drafts(str(item["item_id"]), draft_rows, seed=1)
    return missing_steps(row, drafts, has_tool_errors(item))


def _flatten(row: Mapping[str, Any]) -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in row.items():
        if key in ("item_id", "labeled_at", "relabeled_at", "history", "labels_version"):
            continue
        if key == "drafts" and isinstance(value, dict):
            for drafter, entry in value.items():
                for k, v in (entry or {}).items():
                    if k == "claims":
                        # Numbered from 1, like the `A.c1` steps and the `grounded.1` answers.
                        for i, claim in enumerate(v or [], 1):
                            for claim_key, claim_value in claim.items():
                                flat[f"drafts.{drafter}.claims.{i}.{claim_key}"] = claim_value
                    elif k not in ("slot", "unknown"):
                        flat[f"drafts.{drafter}.{k}"] = v
            continue
        flat[key] = value
    return flat


def _unclear(row: Mapping[str, Any]) -> set[str]:
    """The flattened keys the reviewer answered Unclear. Such an answer is stored as null, as
    an unanswered question is, and told apart by `incomplete` or the draft's `unknown` list."""
    keys: set[str] = set()
    if "failed" in row and row["failed"] is None and not row.get("incomplete"):
        keys.add("failed")
    drafts = row.get("drafts")
    if isinstance(drafts, dict):
        for drafter, entry in drafts.items():
            keys.update(f"drafts.{drafter}.{key}" for key in (entry or {}).get("unknown") or [])
    return keys


def changed_values(previous: Mapping[str, Any], new: Mapping[str, Any]) -> dict[str, Any]:
    """Keys whose answer moved, with the old value. A key that is new, or was unanswered
    before, is not a change. Unclear is an answer, so moving off it is a change, recorded with
    the null it was stored as."""
    before = _flatten(previous)
    after = _flatten(new)
    unclear = _unclear(previous)
    return {
        k: v
        for k, v in before.items()
        if (v is not None or k in unclear) and k in after and after[k] != v
    }
