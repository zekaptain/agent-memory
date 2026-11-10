"""Blinded human admission review using the judge's display-only frontends."""

from __future__ import annotations

import hashlib
import random
import sys
import textwrap
from datetime import UTC, datetime
from typing import Any

from ..capture.pipeline import make_redactor
from ..config import Config
from ..judge.label_build import frontend_binary
from ..judge.label_controller import Action, DisplayDraft, Screen
from ..judge.label_frontend import GoFrontend, PlainFrontend, automatic_plain
from ..judge.label_questions import Option, display_safe
from ..judge.labeling import cited_entries
from ..judge.store import JudgeError, RunStore
from ..lifecycle import stable_id
from . import ledger
from .examples import example_for
from .store import AdmissionStore, write_jsonl

GUIDE = textwrap.fill(
    "Judge the exact lesson, without repairing it. Yes approves it for recall; No retires it; "
    "Unclear leaves its status alone. Review the rule's scope as well as its claims. "
    "The approver's verdict and model identities are hidden. You may add a note, go back, "
    "redo, skip, or quit without losing saved reviews.",
    width=58,
)


def review_split(memory_id: str) -> str:
    return (
        "calibration"
        if int(hashlib.sha256(memory_id.encode()).hexdigest(), 16) % 10 < 6
        else "held_out"
    )


class ReviewController:
    """Own admission review, without instantiating or changing LabelController."""

    def __init__(
        self,
        cfg: Config,
        day: str,
        *,
        discards: int = 30,
        seed: int = 1,
        limit: int | None = None,
        again: bool = False,
    ) -> None:
        self.cfg, self.store = cfg, AdmissionStore(cfg)
        self.day = self.store.day(day)
        self.run = self.day.read_run()
        self.items = {r["item_id"]: r for r in RunStore.read_jsonl(self.day.items_path)}
        self.drafts = {r["item_id"]: r for r in RunStore.read_jsonl(self.day.drafts_path)}
        self.verdicts = {
            r["item_id"]: r
            for r in RunStore.read_jsonl(self.day.verdicts_path(self.run["approver"]["backend"]))
        }
        self.admitted = {
            r["item_id"]: r for r in RunStore.read_jsonl(self.day.root / "admitted.jsonl")
        }
        self.path = self.day.root / "reviews.jsonl"
        rows = RunStore.read_jsonl(self.path)
        for row in rows:
            if row.get("status") == "completed" and row.get("memory_id"):
                iid = row["item_id"]
                self.admitted[iid] = {**self.admitted[iid], "memory_id": row["memory_id"]}
        latest = {r["action_id"]: r for r in rows}
        # A durable save intent can complete after a ledger/file interruption, without
        # asking the human again or silently dropping the training example.
        for row in latest.values():
            if row.get("status") == "pending":
                self._persist(row)
        self.previous = {r["item_id"]: r for r in RunStore.read_jsonl(self.path)}
        available = [
            r
            for iid, r in self.admitted.items()
            if self.drafts[iid].get("rule") and (again or iid not in self.previous)
        ]
        kept = sorted(
            (r for r in available if r["action"] != "discard"), key=lambda r: r["item_id"]
        )
        negatives = sorted(
            (r for r in available if r["action"] == "discard"), key=lambda r: r["item_id"]
        )
        rng = random.Random(seed)
        rng.shuffle(negatives)
        self.todo = kept + negatives[:discards]
        rng.shuffle(self.todo)
        if limit is not None:
            self.todo = self.todo[:limit]
        self.index = self.position = self.saved = self.skipped = 0
        self.stopped = False
        self.answers: dict[str, str] = {}
        self.notice = GUIDE
        self._start()

    def _start(self) -> None:
        if not self.done:
            previous = self.previous.get(self.todo[self.index]["item_id"])
            if previous:
                self.answers = {
                    "store": {True: "y", False: "n", None: "u"}[previous["label"]],
                    "note": previous.get("note", ""),
                }

    @property
    def done(self) -> bool:
        return self.stopped or self.index >= len(self.todo)

    def screen(self) -> Screen:
        result = self.todo[self.index]
        iid = result["item_id"]
        item, draft = self.items[iid], self.drafts[iid]
        projection = self.day.read_projection(iid)
        cited = sorted({n for c in draft["claims"] for n in c["cited"]})
        entries = cited_entries(projection, cited)
        excerpts = "\n\n".join(body for _, body in entries)
        absent = set(cited) - {n for n, _ in entries}
        if absent:
            excerpts += "\nMissing citations: " + ", ".join(f"[{n}]" for n in sorted(absent))
        step = ("store", "note", "summary")[self.position]
        title, options = {
            "store": (
                "Store this exact lesson?",
                [Option("y", "Yes"), Option("n", "No"), Option("u", "Unclear")],
            ),
            "note": (
                "Add a note?",
                [Option("edit", "Add/edit note"), Option("continue", "Continue")],
            ),
            "summary": ("Save this review?", [Option("save", "Save"), Option("redo", "Redo")]),
        }[step]
        evidence = ""
        if step == "summary":
            evidence = (
                "Store this exact lesson: "
                + {
                    "y": "Yes",
                    "n": "No",
                    "u": "Unclear",
                }[self.answers["store"]]
            )
        if step in ("note", "summary") and self.answers.get("note"):
            evidence += "\nNote: " + display_safe(self.answers["note"])
        return Screen(
            item=self.index + 1,
            total=len(self.todo),
            subject={"store": "Draft A", "note": "Notes", "summary": "Summary"}[step],
            key=f"{self.index + 1}:{step}",
            metadata=display_safe(f"Project: {item.get('project') or 'unknown'}"),
            evidence=evidence,
            projection=display_safe(projection),
            question=title,
            options=options,
            selected=(
                "continue"
                if step == "note" and "note" in self.answers
                else self.answers.get(step, options[0].id)
            ),
            note=display_safe(self.answers.get("note", "")),
            can_back=self.position > 0,
            can_redo=step == "summary",
            notice=self.notice,
            guide=GUIDE,
            drafts=[
                DisplayDraft(
                    "A",
                    display_safe(draft["lesson"]),
                    display_safe(excerpts or "No cited evidence."),
                )
            ],
        )

    def apply(self, action: Action) -> None:
        if self.done:
            raise JudgeError("review session has ended")
        screen = self.screen()
        step = ("store", "note", "summary")[self.position]
        self.notice = ""
        if action.action not in ("answer", "note") and action.value:
            raise JudgeError("navigation action must not contain a value")
        if action.action == "quit":
            self.stopped = True
        elif action.action == "skip":
            self.skipped += 1
            self._advance()
        elif action.action == "back" and screen.can_back:
            self.position -= 1
        elif action.action == "redo" and screen.can_redo:
            self.position = 0
        elif action.action == "note" and step == "note":
            if (
                any(ch in "\r\n" or display_safe(ch) != ch for ch in action.value)
                or len(action.value.encode("utf-8")) > 4096
            ):
                raise JudgeError("note must be one line of plain text, at most 4096 bytes")
            self.answers["note"] = action.value
        elif action.action == "answer" and action.value in {o.id for o in screen.options}:
            if step == "summary":
                if action.value == "redo":
                    self.position = 0
                else:
                    self._save()
            elif step == "note":
                if action.value != "continue":
                    raise JudgeError("edit notes through the note input")
                self.position += 1
            else:
                self.answers["store"] = action.value
                self.position += 1
        else:
            raise JudgeError("action is not available on this screen")

    def _advance(self) -> None:
        self.index += 1
        self.position = 0
        self.answers = {}
        self._start()

    def _save(self) -> None:
        result = self.todo[self.index]
        iid = result["item_id"]
        current = self.day.read_drafts().get(iid, [])
        if current != [self.drafts[iid]]:
            raise JudgeError("draft changed during review; quit and reopen the review")
        previous = self.previous.get(iid)
        revision = int(previous.get("revision", 0)) + 1 if previous else 1
        row = {
            "item_id": iid,
            "draft_hash": result["draft_hash"],
            "action_id": stable_id(f"{self.day.run_id}/{iid}/{revision}", "human-review"),
            "revision": revision,
            "label": {"y": True, "n": False, "u": None}[self.answers["store"]],
            "note": self.answers.get("note", ""),
            "reviewed_at": datetime.now(UTC).isoformat(),
            "status": "pending",
        }
        row, _ = make_redactor(self.cfg).value(row)
        write_jsonl(self.path, [row], append=True)
        self._persist(row)
        self.previous[iid] = row
        self.saved += 1
        self.notice = "Saved."
        self._advance()

    def _persist(self, row: dict[str, Any]) -> None:
        iid = row["item_id"]
        result = self.admitted[iid]
        memory_id = result.get("memory_id")
        label = row["label"]
        action = {
            "action_id": row["action_id"],
            "action": "approve" if label else "retire",
            "reason": row["note"] or "human admission review",
        }
        if label is not None:
            terminal = bool(memory_id and label and ledger.terminal_memory(self.cfg, memory_id))
            if memory_id and not terminal:
                memory_id = ledger.review_memory(self.cfg, memory_id, action)
            elif label:
                admitted = ledger.admit(
                    self.cfg,
                    self.day.run_id,
                    self.run["created_at"],
                    self.items[iid],
                    self.drafts[iid],
                    self.verdicts[iid],
                    result,
                    human_action=action,
                    replace_memory_id=memory_id if terminal else None,
                )
                memory_id = admitted["memory_id"]
                self.admitted[iid] = {**result, "memory_id": memory_id}
        # Preserve the group even when a retired memory requires a fresh identity.
        previous = [
            r
            for r in RunStore.read_jsonl(self.path)
            if r["item_id"] == iid and r.get("example_memory_id")
        ]
        identity = (
            previous[-1]["example_memory_id"]
            if previous
            else memory_id or stable_id(f"{self.day.run_id}/{iid}", "discard")
        )
        effective = self.store.read_examples()
        revokes_label = any(
            r.get("memory_id") == identity and r.get("label") is not None for r in effective
        )
        if label is not None or revokes_label:
            example = example_for(
                self.drafts[iid],
                self.items[iid],
                self.verdicts[iid],
                result["embedding"],
                source=row["action_id"],
                label=label,
                split=review_split(identity),
            )
            example.update(
                memory_id=identity,
                action_id=row["action_id"],
                embedder=self.run["approver"]["embedder"],
                reviewed_at=row["reviewed_at"],
                source_type="review",
                ledger_memory_id=memory_id,
                revision=row["revision"],
            )
            existing = {r["source"] for r in RunStore.read_jsonl(self.store.examples_path)}
            if row["action_id"] not in existing:
                self.store.write_state(mode="shadow", approver_version=None)
                write_jsonl(self.store.examples_path, [example], append=True)
        completed = {
            **row,
            "status": "completed",
            "memory_id": memory_id,
            "example_memory_id": identity,
        }
        write_jsonl(self.path, [completed], append=True)

    def result_text(self) -> str:
        return f"Reviewed {self.saved}, skipped {self.skipped}."


def review_day(
    cfg: Config,
    day: str,
    *,
    discards: int = 30,
    seed: int = 1,
    limit: int | None = None,
    plain: bool = False,
    again: bool = False,
) -> dict[str, int]:
    controller = ReviewController(cfg, day, discards=discards, seed=seed, limit=limit, again=again)
    frontend: GoFrontend | PlainFrontend | None = None
    try:
        if not controller.done:
            frontend = (
                PlainFrontend(sys.stdin, sys.stdout)
                if plain or automatic_plain()
                else GoFrontend(frontend_binary())
            )
            while not controller.done:
                controller.apply(frontend.exchange(controller.screen()))
            if isinstance(frontend, GoFrontend):
                frontend.finish(controller.result_text())
    except (KeyboardInterrupt, EOFError):
        controller.stopped = True
    finally:
        if isinstance(frontend, GoFrontend):
            frontend.close()
    return {"reviewed": controller.saved, "skipped": controller.skipped}
