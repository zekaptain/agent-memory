"""Interactive labeling: the human sees the projection and the draft, never a verdict.

Labels are appended to labels.jsonl so a session can stop and resume. Order is a seeded
shuffle of the unlabeled items, so stratum and backend output cannot bias the reviewer.
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

from .questions import CATEGORIES
from .store import RunStore, utc_now_iso

Ask = Callable[[str, Sequence[str] | None], str]

LABEL_KEYS = (
    "failed",
    "category",
    "worth_remembering",
    "supported",
    "generalizable",
    "enforced_by",
    "error_intended",
)
CATEGORY_ORDER = tuple(CATEGORIES)
ENFORCED_BY = {
    "none": "nothing else makes the agent behave this way; memory would be the only place",
    "harness": "the harness already enforces it (Mission prompts, system prompt, subagent limits)",
    "project_docs": "the repository already says it (AGENTS.md, contributing guide, conventions)",
    "tooling": "a tool would catch the slip anyway (lint, type check, CI, hook)",
}
ENFORCED_ORDER = tuple(ENFORCED_BY)
_YN = ("y", "n")
_YNU = ("y", "n", "u")
_LEVELS = ("0", "1", "2")
_CONTROLS = ("s", "q")


class StopLabeling(Exception):
    """The reviewer typed q."""


RUBRIC = """\
[bold]who is speaking[/bold]  The header says who started the session. In a session an agent
  started, the REQUEST and the NEXT MESSAGE were written by that parent agent: treat a
  parent's "no, redo it" as a reviewer's opinion, not as the requester's verdict.
[bold]failed[/bold]  Did the agent fail to do what was asked in this turn? Judge from tool
  results, the recorded outcome, and the next message, not from what the agent claimed.
  [bold]u[/bold] = the evidence does not say.
[bold]category[/bold]  What went wrong, or [bold]none[/bold] if nothing did. Pick the main cause.
[bold]worth remembering[/bold]  Would a lesson from this turn help a future session on this
  project, or any project? [bold]n[/bold] for one-off slips, outages, and user redirects.
[bold]supported[/bold]  Shown with the draft only. Can every claim in the draft lesson be traced to
  something in the projection, in particular the lines it cites? [bold]n[/bold] if it asserts a
  cause or rule the evidence does not show, or cites nothing relevant.
[bold]generalizable[/bold]  Shown with the draft only. 0 = only this exact request, 1 = future
  sessions on this project, 2 = any project.
[bold]enforced by[/bold]  Shown with the draft only. Is the lesson already enforced somewhere the
  agent will meet anyway? [bold]harness[/bold] for Mission orchestration and subagent constraints,
  [bold]project_docs[/bold] for AGENTS.md and the like, [bold]tooling[/bold] for lint/CI/hooks,
  [bold]none[/bold] when memory would be the only place. Judge the lesson, not the turn.
[bold]error intended[/bold]  Shown when a tool call errored. Was the error expected and used
  (probing for a missing file, a test meant to fail, grep with no match)? [bold]y[/bold] then.
[dim]s skips the item (comes back next session), q stops; progress is saved per item.
Items labeled before a question existed come back and ask only that question.[/dim]"""

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


def rich_ask(console: Console) -> Ask:
    def ask(question: str, choices: Sequence[str] | None) -> str:
        return Prompt.ask(question, choices=list(choices) if choices else None, console=console)

    return ask


def _answer(ask: Ask, question: str, choices: Sequence[str]) -> str:
    """One answer from the allowed set; `s` skips the item, `q` stops the session."""
    allowed = [*choices, *_CONTROLS]
    while True:
        try:
            value = ask(question, allowed).strip().lower()
        except InvalidResponse:
            continue
        if value == "q":
            raise StopLabeling
        if value in allowed:
            return value


def render_item(
    console: Console,
    index: int,
    total: int,
    item: dict[str, Any],
    projection: str,
    draft: str | None,
    cited: Sequence[int] = (),
) -> None:
    console.print(Rule(f"item {index}/{total}  [dim]{item['item_id']}[/dim]"))
    meta = Table.grid(padding=(0, 2))
    meta.add_row("[bold]project[/bold]", str(item.get("project") or "unknown"))
    started_by = item.get("facts", {}).get("caller") or item.get("caller") or "human"
    meta.add_row("[bold]session started by[/bold]", str(started_by))
    meta.add_row("[bold]outcome[/bold]", str(item.get("outcome_reason") or "none"))
    meta.add_row("[bold]tool calls[/bold]", str(item.get("facts", {}).get("tool_count", 0)))
    console.print(meta)
    # Text() keeps bracketed evidence numbers and code from being read as Rich markup.
    console.print(Panel(Text(projection), title="projection", border_style="cyan"))
    if draft:
        cites = " ".join(f"[{n}]" for n in cited) or "nothing"
        console.print(
            Panel(Text(draft), title=f"draft lesson  (cites {cites})", border_style="magenta")
        )
        entries = cited_entries(projection, cited)
        if entries:
            body = Text()
            for k, (_, excerpt) in enumerate(entries):
                if k:
                    body.append("\n\n")
                body.append(excerpt)
            console.print(Panel(body, title="cited evidence", border_style="magenta"))
        elif cited:
            console.print("[yellow]the cited line numbers are not in the projection[/yellow]")


def _choice_table(choices: dict[str, str]) -> Table:
    table = Table(show_header=False, box=None, padding=(0, 1))
    for i, (key, desc) in enumerate(choices.items(), start=1):
        table.add_row(f"[bold]{i}[/bold]", key, Text(desc, style="dim"))
    return table


def _category_table() -> Table:
    return _choice_table(CATEGORIES)


def applicable_keys(draft: str | None, has_tool_errors: bool) -> tuple[str, ...]:
    """The label keys this item is asked: the draft and tool-error questions are conditional."""
    keys = ["failed", "category", "worth_remembering"]
    if draft:
        keys += ["supported", "generalizable", "enforced_by"]
    if has_tool_errors:
        keys.append("error_intended")
    return tuple(keys)


def missing_keys(label: Mapping[str, Any], draft: str | None, has_tool_errors: bool) -> list[str]:
    return [k for k in applicable_keys(draft, has_tool_errors) if k not in label]


def label_one(
    ask: Ask,
    console: Console,
    draft: str | None,
    has_tool_errors: bool = False,
    only: Sequence[str] | None = None,
) -> dict[str, Any] | None:
    """Collect one item's labels, or with `only` just those keys. Keys a question does not apply
    to are stored as None. Returns None when the reviewer skips."""
    keys = applicable_keys(draft, has_tool_errors)
    wanted = [k for k in keys if only is None or k in only]
    row: dict[str, Any] = dict.fromkeys(LABEL_KEYS)
    for key in wanted:
        value = ""
        if key == "failed":
            value = _answer(ask, "failed? (y/n/u, s=skip, q=quit)", _YNU)
            row[key] = {"y": True, "n": False, "u": None, "s": None}[value]
        elif key == "category":
            console.print(_category_table())
            value = _answer(ask, "category (number)", _numbers(CATEGORY_ORDER))
            row[key] = None if value == "s" else CATEGORY_ORDER[int(value) - 1]
        elif key == "worth_remembering":
            value = _answer(ask, "worth remembering for future sessions? (y/n)", _YN)
            row[key] = value == "y"
        elif key == "supported":
            value = _answer(ask, "every claim in the draft traceable to the evidence? (y/n)", _YN)
            row[key] = value == "y"
        elif key == "generalizable":
            value = _answer(ask, "generalizable? (0 one-off, 1 project, 2 general)", _LEVELS)
            row[key] = None if value == "s" else int(value)
        elif key == "enforced_by":
            console.print(_choice_table(ENFORCED_BY))
            value = _answer(ask, "lesson already enforced by (number)", _numbers(ENFORCED_ORDER))
            row[key] = None if value == "s" else ENFORCED_ORDER[int(value) - 1]
        elif key == "error_intended":
            value = _answer(ask, "was the tool error intended and used? (y/n)", _YN)
            row[key] = value == "y"
        if value == "s":
            return None
    return {k: row[k] for k in LABEL_KEYS if only is None or k in wanted}


def _numbers(options: Sequence[str]) -> list[str]:
    return [str(i) for i in range(1, len(options) + 1)]


def label_items(
    store: RunStore,
    *,
    console: Console | None = None,
    ask: Ask | None = None,
    limit: int | None = None,
    seed: int = 1,
) -> LabelSummary:
    console = console or Console()
    ask = ask or rich_ask(console)
    items = store.read_jsonl(store.items_path)
    drafts = {row["item_id"]: row for row in store.read_jsonl(store.drafts_path)}
    existing = {row["item_id"]: row for row in store.read_jsonl(store.labels_path)}

    def errors_in(item: Mapping[str, Any]) -> bool:
        return bool(item.get("facts", {}).get("tool_error_count"))

    # Labeled items missing a question added since come first, asking only that question.
    backfill = sorted(
        (
            i
            for i in items
            if i["item_id"] in existing
            and missing_keys(
                existing[i["item_id"]], (drafts.get(i["item_id"]) or {}).get("lesson"), errors_in(i)
            )
        ),
        key=lambda i: i["item_id"],
    )
    fresh = sorted((i for i in items if i["item_id"] not in existing), key=lambda i: i["item_id"])
    random.Random(seed).shuffle(fresh)
    todo = backfill + fresh
    if limit is not None:
        todo = todo[:limit]
    labeled = skipped = 0
    if todo:
        console.print(Panel(RUBRIC, title="how to label", border_style="green"))
        if backfill:
            console.print(
                f"[green]{len(backfill)} labeled item(s) come first to answer questions added "
                "since they were labeled[/green]"
            )
    try:
        for index, item in enumerate(todo, start=1):
            row = drafts.get(item["item_id"]) or {}
            draft = row.get("lesson")
            has_errors = errors_in(item)
            previous = existing.get(item["item_id"])
            only = missing_keys(previous, draft, has_errors) if previous is not None else None
            render_item(
                console,
                index,
                len(todo),
                item,
                store.read_projection(item["item_id"]),
                draft,
                cited=[int(n) for n in row.get("cited") or []],
            )
            if only:
                console.print(f"[green]only asking: {', '.join(only)}[/green]")
            answers = label_one(ask, console, draft, has_errors, only)
            if answers is None:
                skipped += 1
                continue
            if previous is not None:
                existing[item["item_id"]] = {**previous, **answers, "relabeled_at": utc_now_iso()}
                store.write_jsonl(store.labels_path, list(existing.values()))
            else:
                new = {"item_id": item["item_id"], **answers, "labeled_at": utc_now_iso()}
                existing[item["item_id"]] = new
                store.write_jsonl(store.labels_path, [new], append=True)
            labeled += 1
    except StopLabeling:
        console.print("[yellow]stopped; progress is saved[/yellow]")
    remaining = len(items) - len(existing)
    console.print(f"labeled {labeled}, skipped {skipped}, {remaining} still unlabeled")
    return LabelSummary(labeled=labeled, skipped=skipped, remaining=remaining)
