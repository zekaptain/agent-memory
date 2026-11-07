"""Interactive labeling: the human sees the projection and every draft, never a verdict.

Labels are appended to labels.jsonl so a session can stop and resume. Order is a seeded
shuffle of the unlabeled items, so stratum and backend output cannot bias the reviewer. The
drafts of an item are shown together as A, B, ... in a seeded order with the drafter hidden;
the row records which drafter sat in each slot. A changed label keeps its previous value in
the row's `history`, so the file stays append-only in spirit and the report can count how
many labels moved.
"""

from __future__ import annotations

import random
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.prompt import InvalidResponse, Prompt
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from .drafter import draft_hash
from .questions import CATEGORIES, ENFORCED
from .store import JudgeError, RunStore, utc_now_iso
from .triggers import closed_by_of

Ask = Callable[[str, Sequence[str] | None], str]

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
_YN = ("y", "n")
_FAILED = ("y", "n", "u", "i")
_LEVELS = ("0", "1", "2")
SAVE_PROMPT = "save? (Enter/y save, r redo, s skip, q quit)"
REVIEW_PROMPT = "keep these labels? (k keep, r redo, s skip, q quit)"


class StopLabeling(Exception):
    """The reviewer typed q."""


class SkipItem(Exception):
    """The reviewer typed s."""


class GoBack(Exception):
    """The reviewer typed b."""


RUBRIC = """\
[bold]who is speaking[/bold]  The header says who started the session. In a session an agent
  started, the REQUEST and the NEXT MESSAGE were written by that parent agent: treat a
  parent's "no, redo it" as a reviewer's opinion, not as the requester's verdict.
[bold]closed by[/bold]  How the turn ended: at its [bold]outcome[/bold], cut off by the
  [bold]next_message[/bold], or at the [bold]session_end[/bold]. A cut-off turn may hold an
  attempt, not a result.
[bold]failed[/bold]  Did the agent fail to do what was asked in this turn? Judge from tool
  results, the recorded outcome, and the next message, not from what the agent claimed.
  [bold]u[/bold] = the turn finished but the evidence does not say. [bold]i[/bold] = incomplete:
  the turn was cut off before the agent finished, so success or failure cannot be read from it.
[bold]category[/bold]  Asked when failed. The main cause.
[bold]worth remembering[/bold]  Would a lesson from this turn help a future session on this
  project, or any project? [bold]n[/bold] for one-off slips, outages, and user redirects.
[bold]lesson kind[/bold]  Asked when worth remembering. What sort of thing the lesson is.
[bold]error intended[/bold]  Asked when a tool call errored. Was the error expected and used
  (probing for a missing file, a test meant to fail, grep with no match)? [bold]y[/bold] then.
[bold]per draft[/bold]  Drafts are shown as A, B in a random order; the drafter is hidden.
  "(no lesson)" means the drafter declined and asks nothing. A rule before each run of
  questions says what they are about: the turn itself, draft A, draft B, or both.
  [bold]grounded[/bold]: can every claim in the draft be traced to something in the projection,
  in particular the lines it cites? [bold]n[/bold] if it asserts a cause or rule the evidence
  does not show, or cites nothing relevant.
  [bold]faithful[/bold]: does the draft read the evidence right? [bold]n[/bold] if it cites the
  right lines but misstates the cause, the error, or the fact they show.
  [bold]generalizable[/bold]: 0 = only this exact request, 1 = future sessions on this project,
  2 = any project.
  [bold]enforced by[/bold]: is the lesson already enforced somewhere the agent will meet anyway?
  [bold]harness[/bold] for Mission orchestration and subagent constraints, [bold]project_docs[/bold]
  for AGENTS.md and the like, [bold]tooling[/bold] for lint/CI/hooks, [bold]none[/bold] when memory
  would be the only place. Judge the lesson, not the turn.
[bold]preferred[/bold]  Which draft would you rather keep: a, b, same, or neither.
[bold]note[/bold]  Optional free text; Enter for none.
[dim]b goes back one question, s skips the item (comes back next session), q stops; a summary
is shown before saving and r redoes the item. Progress is saved per item. Items labeled before
a question or a drafter existed come back and ask only what is missing.[/dim]"""

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


def rich_ask(console: Console) -> Ask:
    def ask(question: str, choices: Sequence[str] | None) -> str:
        if choices is None:
            return Prompt.ask(question, console=console, default="", show_default=False)
        if question == SAVE_PROMPT:
            return Prompt.ask(
                question, choices=list(choices), console=console, default="y", show_default=False
            )
        return Prompt.ask(question, choices=list(choices), console=console)

    return ask


def _answer(ask: Ask, question: str, choices: Sequence[str], *, back: bool = True) -> str:
    """One answer from the allowed set; `b` goes back, `s` skips the item, `q` stops."""
    allowed = [*choices, "s", "q", *(["b"] if back else [])]
    while True:
        try:
            value = ask(question, allowed).strip().lower()
        except InvalidResponse:
            continue
        if value == "q":
            raise StopLabeling
        if value == "s":
            raise SkipItem
        if value == "b" and back:
            raise GoBack
        if value in choices:
            return value


def _numbers(options: Sequence[str]) -> list[str]:
    return [str(i) for i in range(1, len(options) + 1)]


def _choice_table(choices: Mapping[str, str]) -> Table:
    table = Table(show_header=False, box=None, padding=(0, 1))
    for i, (key, desc) in enumerate(choices.items(), start=1):
        table.add_row(f"[bold]{i}[/bold]", key, Text(desc, style="dim"))
    return table


def render_item(
    console: Console,
    index: int,
    total: int,
    item: Mapping[str, Any],
    projection: str,
    drafts: Sequence[DraftView],
    previous: Mapping[str, Any] | None = None,
) -> None:
    console.print(Rule(f"item {index}/{total}  [dim]{item['item_id']}[/dim]"))
    meta = Table.grid(padding=(0, 2))
    meta.add_row("[bold]project[/bold]", str(item.get("project") or "unknown"))
    started_by = item.get("facts", {}).get("caller") or item.get("caller") or "human"
    meta.add_row("[bold]session started by[/bold]", str(started_by))
    meta.add_row("[bold]closed by[/bold]", closed_by_of(item))
    meta.add_row("[bold]outcome[/bold]", str(item.get("outcome_reason") or "none"))
    meta.add_row("[bold]tool calls[/bold]", str(item.get("facts", {}).get("tool_count", 0)))
    console.print(meta)
    # Text() keeps bracketed evidence numbers and code from being read as Rich markup.
    console.print(Panel(Text(projection), title="projection", border_style="cyan"))
    for draft in drafts:
        if draft.declined:
            console.print(
                Panel(
                    Text("(no lesson)", style="dim"),
                    title=f"draft {draft.slot}",
                    border_style="magenta",
                )
            )
            continue
        cites = " ".join(f"[{n}]" for n in draft.cited) or "nothing"
        console.print(
            Panel(
                Text(draft.lesson or ""),
                title=f"draft {draft.slot}  (cites {cites})",
                border_style="magenta",
            )
        )
        entries = cited_entries(projection, draft.cited)
        if entries:
            body = Text()
            for k, (_, excerpt) in enumerate(entries):
                if k:
                    body.append("\n\n")
                body.append(excerpt)
            console.print(
                Panel(body, title=f"draft {draft.slot}: cited evidence", border_style="magenta")
            )
        elif draft.cited:
            console.print(
                f"[yellow]draft {draft.slot}: the cited line numbers are not in the "
                "projection[/yellow]"
            )
    if previous is not None:
        console.print(
            Panel(
                Text(summary_line(previous, drafts)), title="current labels", border_style="green"
            )
        )


# Steps: the questions an item is asked, in order, given the answers so far. Raw answers are the
# typed strings; `assemble` turns them into the stored row.


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


def _section(step: str) -> str:
    """What a step is about: the turn itself, one draft (its slot), or the drafts side by side."""
    if "." in step:
        return step.partition(".")[0]
    if step == "preferred":
        return "preferred"
    if step == "note":
        return "note"
    return "turn"


def _announce(console: Console, section: str, drafts: Sequence[DraftView]) -> None:
    """A cue before a run of questions saying what they are about. A draft is shown again
    here because its panel has usually scrolled away behind the cited evidence by now."""
    if section == "turn":
        console.print(Rule("about the turn itself (not the drafts)", style="cyan"))
    elif section == "preferred":
        console.print(Rule("the drafts side by side", style="magenta"))
    elif section != "note":
        draft = next((d for d in drafts if d.slot == section), None)
        console.print(Rule(f"about draft {section}", style="magenta"))
        if draft is not None:
            console.print(
                Panel(Text(draft.lesson or ""), title=f"draft {section}", border_style="magenta")
            )


def _ask_step(step: str, ask: Ask, console: Console, drafts: Sequence[DraftView]) -> str:
    if step == "failed":
        return _answer(ask, "failed? (y/n, u unclear, i incomplete)", _FAILED)
    if step == "category":
        console.print(_choice_table(CATEGORIES))
        return _answer(ask, "category (number)", _numbers(CATEGORY_ORDER))
    if step == "worth_remembering":
        return _answer(ask, "worth remembering for future sessions? (y/n)", _YN)
    if step == "lesson_kind":
        console.print(_choice_table(LESSON_KINDS))
        return _answer(ask, "lesson kind (number)", _numbers(LESSON_ORDER))
    if step == "error_intended":
        return _answer(ask, "was the tool error intended and used? (y/n)", _YN)
    if step == "preferred":
        slots = [d.slot.lower() for d in drafts]
        return _answer(
            ask, f"preferred draft? ({'/'.join(slots)}/same/neither)", (*slots, *PREFERRED_OTHER)
        )
    if step == "note":
        value = ask("note (optional, Enter for none)", None).strip()
        if value.lower() == "q":
            raise StopLabeling
        if value.lower() == "b":
            raise GoBack
        return value
    slot, _, key = step.partition(".")
    if key == "grounded":
        return _answer(
            ask, f"draft {slot}: every claim traceable to the cited evidence? (y/n)", _YN
        )
    if key == "faithful":
        return _answer(
            ask, f"draft {slot}: reads the evidence right (cause, error, fact)? (y/n)", _YN
        )
    if key == "generalizable":
        return _answer(
            ask, f"draft {slot}: generalizable? (0 one-off, 1 project, 2 general)", _LEVELS
        )
    if key == "enforced_by":
        console.print(_choice_table(ENFORCED_BY))
        return _answer(
            ask, f"draft {slot}: lesson already enforced by (number)", _numbers(ENFORCED_ORDER)
        )
    raise JudgeError(f"unknown labeling step {step!r}")


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


def summary_line(row: Mapping[str, Any], drafts: Sequence[DraftView]) -> str:
    def yn(value: Any) -> str:
        return "-" if value is None else ("y" if value else "n")

    failed = "i" if row.get("incomplete") else yn(row.get("failed"))
    if failed == "-" and "failed" in row and not row.get("incomplete"):
        failed = "u"
    parts = [
        f"failed={failed}",
        f"category={row.get('category') or '-'}",
        f"worth={yn(row.get('worth_remembering'))}",
        f"kind={row.get('lesson_kind') or '-'}",
    ]
    if row.get("error_intended") is not None:
        parts.append(f"error_intended={yn(row['error_intended'])}")
    entries = row.get("drafts") or {}
    for draft in drafts:
        entry = entries.get(draft.drafter) or {}
        if draft.declined:
            parts.append(f"{draft.slot}: (no lesson)")
            continue
        gen = entry.get("generalizable")
        parts.append(
            f"{draft.slot}: grounded={yn(entry.get('grounded'))} "
            f"faithful={yn(entry.get('faithful'))} gen={'-' if gen is None else gen} "
            f"enforced={entry.get('enforced_by') or '-'}"
        )
    preferred = row.get("preferred")
    if preferred in PREFERRED_OTHER:
        parts.append(f"preferred={preferred}")
    elif preferred:
        slot = next((d.slot for d in drafts if d.drafter == preferred), "?")
        parts.append(f"preferred={slot}")
    if row.get("note"):
        parts.append(f"note={row['note']}")
    return " | ".join(parts)


def label_one(
    ask: Ask,
    console: Console,
    drafts: Sequence[DraftView],
    has_tool_errors: bool = False,
    *,
    only: Sequence[str] | None = None,
    previous: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Collect one item's labels, or with `only` just those steps on top of `previous`. The
    reviewer can step back with b, redo from the summary with r, and skip with s (None)."""
    base = raw_answers(previous, drafts) if previous is not None and only is not None else {}
    try:
        while True:
            answers = {k: v for k, v in base.items() if k not in (only or ())}
            plan = _plan(answers, drafts, has_tool_errors, only)
            i = 0
            section = None
            while i < len(plan):
                if _section(plan[i]) != section:
                    section = _section(plan[i])
                    _announce(console, section, drafts)
                try:
                    value = _ask_step(plan[i], ask, console, drafts)
                except GoBack:
                    if i > 0:
                        i -= 1
                        answers.pop(plan[i], None)
                        plan = _plan(answers, drafts, has_tool_errors, only)
                    continue
                answers[plan[i]] = value
                plan = _plan(answers, drafts, has_tool_errors, only)
                i += 1
            row = assemble(answers, drafts, has_tool_errors)
            console.print(Text(summary_line(row, drafts), style="bold"))
            if _answer(ask, SAVE_PROMPT, ("y", "r"), back=False) == "y":
                return row
    except SkipItem:
        return None


def _plan(
    answers: Mapping[str, str],
    drafts: Sequence[DraftView],
    has_tool_errors: bool,
    only: Sequence[str] | None,
) -> list[str]:
    steps = steps_for(answers, drafts, has_tool_errors)
    return steps if only is None else [s for s in steps if s in only]


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


def label_items(
    store: RunStore,
    *,
    console: Console | None = None,
    ask: Ask | None = None,
    limit: int | None = None,
    seed: int = 1,
    item_ids: Sequence[str] = (),
    review: bool = False,
) -> LabelSummary:
    console = console or Console()
    ask = ask or rich_ask(console)
    run = store.read_run()
    if run.get("questions_version") == "questions_v1":
        raise JudgeError(
            f"run {store.run_id!r} was labeled under questions_v1; those labels stay as they are. "
            "Label a run built with the current version instead."
        )
    items = store.read_jsonl(store.items_path)
    by_id = {i["item_id"]: i for i in items}
    drafts = store.read_drafts()
    existing = {row["item_id"]: row for row in store.read_jsonl(store.labels_path)}

    def errors_in(item: Mapping[str, Any]) -> bool:
        return bool(item.get("facts", {}).get("tool_error_count"))

    def views(item: Mapping[str, Any]) -> list[DraftView]:
        return order_drafts(item["item_id"], drafts.get(item["item_id"]) or [], seed)

    backfill: list[dict[str, Any]] = []
    if item_ids:
        unknown = [i for i in item_ids if i not in by_id]
        if unknown:
            raise JudgeError(f"no such item(s) in run {store.run_id!r}: {', '.join(unknown)}")
        todo = [by_id[i] for i in item_ids]
    elif review:
        todo = sorted((i for i in items if i["item_id"] in existing), key=lambda i: i["item_id"])
        random.Random(seed).shuffle(todo)
    else:
        # Labeled items missing a question or a draft added since come first, asking only that.
        backfill = sorted(
            (
                i
                for i in items
                if i["item_id"] in existing
                and missing_steps(existing[i["item_id"]], views(i), errors_in(i))
            ),
            key=lambda i: i["item_id"],
        )
        fresh = sorted(
            (i for i in items if i["item_id"] not in existing), key=lambda i: i["item_id"]
        )
        random.Random(seed).shuffle(fresh)
        todo = backfill + fresh
    if limit is not None:
        todo = todo[:limit]
    labeled = skipped = changed = 0
    if todo:
        console.print(Panel(RUBRIC, title="how to label", border_style="green"))
        if backfill:
            console.print(
                f"[green]{len(backfill)} labeled item(s) come first to answer questions added "
                "since they were labeled[/green]"
            )
    try:
        for index, item in enumerate(todo, start=1):
            item_id = item["item_id"]
            item_drafts = views(item)
            has_errors = errors_in(item)
            previous = existing.get(item_id)
            only: list[str] | None = None
            if previous is not None and not item_ids and not review:
                only = missing_steps(previous, item_drafts, has_errors)
            render_item(
                console,
                index,
                len(todo),
                item,
                store.read_projection(item_id),
                item_drafts,
                previous=previous if only is None else None,
            )
            if review:
                try:
                    decision = _answer(ask, REVIEW_PROMPT, ("k", "r"), back=False)
                except SkipItem:
                    skipped += 1
                    continue
                if decision == "k":
                    continue
            if only:
                console.print(f"[green]only asking: {', '.join(only)}[/green]")
            answers = label_one(ask, console, item_drafts, has_errors, only=only, previous=previous)
            if answers is None:
                skipped += 1
                continue
            stamp = utc_now_iso()
            if previous is not None:
                moved = changed_values(previous, answers)
                history = list(previous.get("history") or [])
                if moved:
                    history.append({"at": stamp, "changed": moved})
                    changed += 1
                existing[item_id] = {
                    **previous,
                    **answers,
                    "labels_version": LABELS_VERSION,
                    "relabeled_at": stamp,
                    "history": history,
                }
                store.write_jsonl(store.labels_path, list(existing.values()))
            else:
                new = {
                    "item_id": item_id,
                    **answers,
                    "labels_version": LABELS_VERSION,
                    "labeled_at": stamp,
                }
                existing[item_id] = new
                store.write_jsonl(store.labels_path, [new], append=True)
            labeled += 1
    except StopLabeling:
        console.print("[yellow]stopped; progress is saved[/yellow]")
    remaining = len(items) - len(existing)
    console.print(
        f"labeled {labeled}, skipped {skipped}, {changed} changed, {remaining} still unlabeled"
    )
    return LabelSummary(labeled=labeled, skipped=skipped, remaining=remaining, changed=changed)
