"""Python owns the labeling workflow. Frontends see only explicitly blinded display fields."""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from .label_questions import GUIDE, INTRO, NAMES, Option, display_safe, question
from .labeling import (
    LABELS_VERSION,
    SLOTS,
    DraftView,
    LabelSummary,
    assemble,
    changed_values,
    cited_entries,
    has_tool_errors,
    missing_steps,
    order_drafts,
    raw_answers,
    steps_for,
    validate_draft_count,
)
from .questions import QUESTIONS_VERSION
from .store import JudgeError, RunStore, ensure_experiment_home, utc_now_iso
from .triggers import closed_by_of


@dataclass(frozen=True)
class Action:
    action: str
    value: str = ""


@dataclass(frozen=True)
class DisplayDraft:
    slot: str
    lesson: str
    cited_evidence: str
    focus: str = ""


@dataclass
class Screen:
    item: int
    total: int
    subject: str
    key: str
    metadata: str
    evidence: str
    projection: str
    question: str
    options: list[Option]
    selected: str = ""
    note: str = ""
    can_back: bool = False
    can_redo: bool = False
    notice: str = ""
    guide: str = GUIDE
    drafts: list[DisplayDraft] = field(default_factory=list)

    def __post_init__(self) -> None:
        validate_draft_count(len(self.drafts))
        slots = [draft.slot for draft in self.drafts]
        if len(set(slots)) != len(slots) or any(
            len(slot) != 1 or slot not in SLOTS for slot in slots
        ):
            raise JudgeError("Invalid labeling draft slots.")
        if self.subject not in {
            "Turn",
            "Compare drafts",
            "Notes",
            "Review",
            "Summary",
            *(f"Draft {slot}" for slot in SLOTS),
        }:
            raise JudgeError("Invalid labeling subject.")

    def payload(self) -> dict[str, Any]:
        return asdict(self)


class LabelController:
    def __init__(
        self,
        store: RunStore,
        *,
        limit: int | None = None,
        seed: int = 1,
        item_ids: Sequence[str] = (),
        review: bool = False,
    ) -> None:
        ensure_experiment_home(store.cfg)
        self.store, self.seed, self.review = store, seed, review
        version = store.read_run().get("questions_version")
        if version != QUESTIONS_VERSION:
            raise JudgeError(
                f"This run uses {version} and is read-only. Label a current-version run instead."
            )
        self.items = store.items()
        self.draft_rows = store.drafts_by_item()
        self.existing = store.labels()
        by_id = {item["item_id"]: item for item in self.items}
        self.backfill = not item_ids and not review
        if item_ids:
            if any(item_id not in by_id for item_id in item_ids):
                raise JudgeError("no such item in this run; check the item ids")
            self.todo = [by_id[item_id] for item_id in dict.fromkeys(item_ids)]
        elif review:
            self.todo = sorted(
                (item for item in self.items if item["item_id"] in self.existing),
                key=lambda item: item["item_id"],
            )
            random.Random(seed).shuffle(self.todo)
        else:
            backfill = sorted(
                (
                    item
                    for item in self.items
                    if item["item_id"] in self.existing
                    and missing_steps(
                        self.existing[item["item_id"]], self.views(item), self.has_errors(item)
                    )
                ),
                key=lambda item: item["item_id"],
            )
            fresh = sorted(
                (item for item in self.items if item["item_id"] not in self.existing),
                key=lambda item: item["item_id"],
            )
            random.Random(seed).shuffle(fresh)
            self.todo = backfill + fresh
        if limit is not None:
            self.todo = self.todo[:limit]
        # Refuse unsupported items before either frontend opens or any labels are saved.
        for item in self.todo:
            validate_draft_count(len(self.draft_rows.get(item["item_id"], [])))
        self.index = self.labeled = self.skipped = self.changed = 0
        self.stopped = False
        self.notice = INTRO
        self.answers: dict[str, str] = {}
        self.only: list[str] | None = None
        self.position = 0
        self.mode = ""
        self.drafts: list[DraftView] = []
        self.projection = ""
        self._start()

    @staticmethod
    def has_errors(item: Mapping[str, Any]) -> bool:
        return has_tool_errors(item)

    def views(self, item: Mapping[str, Any]) -> list[DraftView]:
        return order_drafts(item["item_id"], self.draft_rows.get(item["item_id"], []), self.seed)

    @property
    def done(self) -> bool:
        return self.stopped or self.index >= len(self.todo)

    @property
    def item(self) -> dict[str, Any]:
        return self.todo[self.index]

    @property
    def plan(self) -> list[str]:
        steps = steps_for(self.answers, self.drafts, self.has_errors(self.item))
        return steps if self.only is None else [step for step in steps if step in self.only]

    @property
    def step(self) -> str:
        if self.mode == "review":
            return "review"
        return self.plan[self.position] if self.position < len(self.plan) else "summary"

    def _start(self) -> None:
        left_out = 0
        while not self.done:
            self.drafts = self.views(self.item)
            previous = self.existing.get(self.item["item_id"])
            self.only = (
                missing_steps(previous, self.drafts, self.has_errors(self.item))
                if previous is not None and self.backfill
                else None
            )
            if self.only == []:
                # Nothing is left to ask: another session has finished this item since this
                # one opened, and its label came back with this session's last save.
                self.index += 1
                left_out += 1
                continue
            self.projection = self.store.read_projection(self.item["item_id"])
            self.answers = raw_answers(previous, self.drafts) if previous is not None else {}
            self.position = 0
            self.mode = "review" if self.review and previous is not None else ""
            break
        if left_out:
            items = "1 item" if left_out == 1 else f"{left_out} items"
            self.notice = f"{self.notice} Left out {items} another session labeled.".strip()

    def _advance(self) -> None:
        self.index += 1
        self._start()

    def display_draft(self, draft: DraftView, claim_index: int | None = None) -> DisplayDraft:
        numbers = draft.cited if claim_index is None else draft.claims[claim_index].cited
        entries = cited_entries(self.projection, numbers)
        cited = "\n\n".join(body for _, body in entries)
        absent = sorted(set(numbers) - {number for number, _ in entries})
        if absent:
            cited += "\nMissing citations: " + ", ".join(f"[{number}]" for number in absent)
            cited += ". Treat these as no evidence."
        text, focus = draft.lesson or "(no lesson)", ""
        if claim_index is not None:
            focus = f"Claim {claim_index + 1} of {len(draft.claims)}"
            lines = text.splitlines()
            marked = 3 + claim_index  # after render_lesson's Rule, Kind, and Claims: lines
            if marked < len(lines):
                lines[marked] = "▶ " + lines[marked]
            text = "\n".join(lines)
        return DisplayDraft(
            draft.slot,
            display_safe(text),
            display_safe(cited or "No cited evidence."),
            focus,
        )

    def summary_text(self) -> str:
        lines = []
        row = assemble(self.answers, self.drafts, self.has_errors(self.item))
        for step in steps_for(self.answers, self.drafts, self.has_errors(self.item)):
            if step == "note":
                continue
            _, options = question(step, [draft.slot for draft in self.drafts])
            key = step.rsplit(".", 1)[-1]
            title = {
                "failed": "Failed",
                "category": "Cause",
                "worth_remembering": "Worth remembering",
                "lesson_kind": "Lesson kind",
                "error_intended": "Expected and used error",
                "grounded": "Grounded",
                "faithful": "Faithful",
                "rule_supported": "Exact rule supported",
                "store_exact": "Store this exact draft",
                "generalizable": "Useful on",
                "enforced_by": "Already enforced by",
                "preferred": "Keep",
            }[key]
            selected = next(
                (option.label for option in options if option.id == self.answers.get(step)), "—"
            )
            subject = ""
            if "." in step:
                prefix = step.rsplit(".", 1)[0]
                slot, _, claim = prefix.partition(".c")
                subject = f"Draft {slot}" + (f" claim {claim}" if claim else "") + ": "
            lines.append(f"{subject}{title}: {selected}")
        for draft in self.drafts:
            if draft.declined:
                lines.append(f"Draft {draft.slot}: no lesson")
        if row.get("note"):
            lines.append(f"Note: {display_safe(str(row['note']))}")
        return "\n".join(lines)

    def screen(self) -> Screen:
        if self.done:
            raise JudgeError("labeling session has ended")
        step = self.step
        title, options = question(step, [draft.slot for draft in self.drafts])
        subject, evidence, drafts = "Turn", self.projection, []
        claim_index = None
        if "." in step:
            prefix = step.rsplit(".", 1)[0]
            slot, _, claim = prefix.partition(".c")
            if claim:
                claim_index = int(claim) - 1
            subject, evidence = f"Draft {slot}", ""
            drafts = [draft for draft in self.drafts if draft.slot == slot]
        elif step == "preferred":
            subject, evidence, drafts = "Compare drafts", "", self.drafts
        elif step in ("review", "summary", "note"):
            subject = {"review": "Review", "summary": "Summary", "note": "Notes"}[step]
            evidence = self.summary_text()
            drafts = self.drafts
        facts = self.item.get("facts") or {}
        started = str(facts.get("caller") or self.item.get("caller") or "human")
        closed = NAMES.get(closed_by_of(self.item), closed_by_of(self.item).replace("_", " "))
        metadata = (
            f"Project: {self.item.get('project') or 'unknown'} · Started by: {started} · "
            f"Closed by: {closed} · Outcome: {self.item.get('outcome_reason') or 'none'} · "
            f"Tools: {facts.get('tool_count', 0)} ({facts.get('tool_error_count', 0)} errors)"
        )
        if self.only:
            metadata += " · Asking only missing answers"
        selected = self.answers.get(step, "")
        if step == "note":
            selected = "continue" if "note" in self.answers else "edit"
        return Screen(
            item=self.index + 1,
            total=len(self.todo),
            subject=subject,
            key=f"{self.index + 1}:{step}",
            metadata=display_safe(metadata),
            evidence=display_safe(evidence),
            projection=display_safe(self.projection),
            question=title,
            options=options,
            selected=selected or options[0].id,
            # The Go helper refuses a screen whose note holds a tab, and a note saved before
            # tabs were refused can hold one. One space each, so the note cannot outgrow
            # the limit.
            note=display_safe(self.answers.get("note", "")).replace("\t", " "),
            can_back=self.mode != "review" and self.position > 0,
            can_redo=step in ("review", "summary"),
            notice=self.notice,
            drafts=[self.display_draft(draft, claim_index) for draft in drafts],
        )

    def apply(self, action: Action) -> None:
        if self.done:
            raise JudgeError("action after labeling ended")
        screen, step = self.screen(), self.step
        if action.action not in ("answer", "note") and action.value:
            raise JudgeError("navigation action must not contain a value")
        self.notice = ""
        if action.action == "quit":
            self.stopped = True
        elif action.action == "skip":
            self.skipped += 1
            self.notice = "Skipped. This item remains available next session."
            self._advance()
        elif action.action == "back" and screen.can_back:
            self.position -= 1  # Keep answers for restored highlights.
        elif action.action == "redo" and screen.can_redo:
            self._redo()
        elif action.action == "note" and step == "note":
            if (
                any(ch in "\r\n\t" or unicodedata_control(ch) for ch in action.value)
                or len(action.value.encode("utf-8")) > 4096
            ):
                raise JudgeError("note must be one line of text, at most 4096 bytes")
            self.answers["note"] = action.value
        elif action.action == "answer" and action.value in {o.id for o in screen.options}:
            value = action.value
            if step == "review":
                if value == "keep":
                    self._advance()
                else:
                    self._redo()
            elif step == "summary":
                if value == "save":
                    self._save()
                else:
                    self._redo()
            elif step == "note":
                if value != "continue":
                    raise JudgeError("edit notes through the note input")
                self.answers.setdefault("note", "")
                self.position += 1
            else:
                self.answers[step] = value
                valid = set(steps_for(self.answers, self.drafts, self.has_errors(self.item)))
                self.answers = {key: val for key, val in self.answers.items() if key in valid}
                self.position = self.plan.index(step) + 1
        else:
            raise JudgeError("action is not available on this screen")

    def _redo(self) -> None:
        self.mode = ""
        self.position = 0
        self.answers = raw_answers(self.existing.get(self.item["item_id"], {}), self.drafts)
        self.notice = "Review each answer, then save."

    def _save(self) -> None:
        item_id, stamp = self.item["item_id"], utc_now_iso()
        current = order_drafts(item_id, self.store.drafts_by_item().get(item_id, []), self.seed)
        if current != self.drafts:
            raise JudgeError(
                "Drafts changed during labeling. Quit and reopen to label the new drafts."
            )
        row = assemble(self.answers, self.drafts, self.has_errors(self.item))
        previous = self.existing.get(item_id)
        moved = changed_values(previous, row) if previous is not None else {}
        if previous is not None:
            history = list(previous.get("history") or [])
            if moved:
                history.append({"at": stamp, "changed": moved})
            new = {**previous, **row, "relabeled_at": stamp, "history": history}
        else:
            new = {"item_id": item_id, **row, "labeled_at": stamp}
        new["labels_version"] = LABELS_VERSION
        try:
            # What other sessions saved meanwhile comes back merged with this label.
            self.existing = self.store.save_label(new, seen=previous)
        except OSError:
            self.notice = "Save failed. Retry Save or quit. Earlier saved items are safe."
            return
        self.labeled += 1
        self.changed += bool(moved)
        self.notice = "Saved."
        self._advance()

    def summary(self) -> LabelSummary:
        return LabelSummary(
            self.labeled, self.skipped, len(self.items) - len(self.existing), self.changed
        )

    def result_text(self) -> str:
        summary = self.summary()
        prefix = "Stopped. " if self.stopped else ""
        return (
            f"{prefix}Labeled {summary.labeled}, skipped {summary.skipped}, "
            f"{summary.changed} changed, {summary.remaining} still unlabeled."
        )


def unicodedata_control(ch: str) -> bool:
    # Same policy as the display sanitizer, but refuse controls in newly entered notes.
    return display_safe(ch) != ch
