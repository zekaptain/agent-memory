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

from rich.console import Console, Group
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


# Rendered as a term/description grid so each entry wraps to the terminal on its own.
RUBRIC = (
    (
        "who is speaking",
        "The header says who started the session. In a session an agent started, the REQUEST "
        "and the NEXT MESSAGE were written by that parent agent: treat a parent's \"no, redo "
        "it\" as a reviewer's opinion, not as the requester's verdict.",
    ),
    (
        "closed by",
        "How the turn ended: at its [bold]outcome[/bold], cut off by the "
        "[bold]next_message[/bold], or at the [bold]session_end[/bold]. A cut-off turn may hold "
        "an attempt, not a result.",
    ),
    (
        "failed",
        "Did the agent fail to do what was asked in this turn? Judge from tool results, the "
        "recorded outcome, and the next message, not from what the agent claimed. "
        "[bold]u[/bold] = the turn finished but the evidence does not say. [bold]i[/bold] = "
        "incomplete: the turn was cut off before the agent finished, so success or failure "
        "cannot be read from it.",
    ),
    ("category", "Asked when failed. The main cause."),
    (
        "worth remembering",
        "Would a lesson from this turn help a future session on this project, or any project? "
        "[bold]n[/bold] for one-off slips, outages, and user redirects.",
    ),
    ("lesson kind", "Asked when worth remembering. What sort of thing the lesson is."),
    (
        "error intended",
        "Asked when a tool call errored. Was the error expected and used (probing for a "
        "missing file, a test meant to fail, grep with no match)? [bold]y[/bold] then.",
    ),
    (
        "drafts",
        'Shown as A, B in a random order; the drafter is hidden. "(no lesson)" means the '
        "drafter declined and asks nothing. Each run of questions starts on a fresh screen that "
        "says what it is about and shows just that: the turn and its projection, one draft "
        "with the evidence it cites, or the drafts side by side.",
    ),
    (
        "grounded",
        "Can every claim in the draft be traced to something in the projection, in particular "
        "the lines it cites? [bold]n[/bold] if it asserts a cause or rule the evidence does not "
        "show, or cites nothing relevant.",
    ),
    (
        "faithful",
        "Does the draft read the evidence right? [bold]n[/bold] if it cites the right lines but "
        "misstates the cause, the error, or the fact they show.",
    ),
    (
        "generalizable",
        "0 = only this exact request, 1 = future sessions on this project, 2 = any project.",
    ),
    (
        "enforced by",
        "Is the lesson already enforced somewhere the agent will meet anyway? "
        "[bold]harness[/bold] for Mission orchestration and subagent constraints, "
        "[bold]project_docs[/bold] for AGENTS.md and the like, [bold]tooling[/bold] for "
        "lint/CI/hooks, [bold]none[/bold] when memory would be the only place. Judge the "
        "lesson, not the turn.",
    ),
    ("preferred", "Which draft would you rather keep: a, b, same, or neither."),
    ("note", "Optional free text; Enter for none."),
)
KEYS = (
    "b back one question, p page the full projection, ? page this rubric, s skip this item, q quit"
)
RUBRIC_KEYS = (
    "At any question: b goes back one question, p pages the full projection, ? pages this "
    "rubric (q leaves the pager), s skips the item (it comes back next session), q stops. The "
    "turn screen shows the projection condensed: a tool call and its result on one row, blank "
    "lines dropped; p has the text as stored. A summary is shown before saving; r redoes the "
    "item. Progress is saved per item. Items labeled before a question or a drafter existed "
    "come back and ask only what is missing."
)
START_PROMPT = "Enter to start (q quits)"
_STEP_NAMES = {
    "worth_remembering": "worth remembering",
    "lesson_kind": "lesson kind",
    "error_intended": "error intended",
    "enforced_by": "enforced by",
}


def rubric_panel() -> Panel:
    grid = Table.grid(padding=(0, 2))
    grid.add_column(no_wrap=True, style="bold")
    grid.add_column()
    for term, description in RUBRIC:
        grid.add_row(term, description)
    return Panel(Group(grid, Text(f"\n{RUBRIC_KEYS}", style="dim")), title="how to label")


def describe_steps(steps: Sequence[str]) -> str:
    """`B.grounded, B.faithful, preferred` as `draft B (grounded, faithful), preferred`."""
    order: list[str] = []
    per_draft: dict[str, list[str]] = {}
    for step in steps:
        slot, dot, key = step.partition(".")
        if dot:
            if slot not in per_draft:
                per_draft[slot] = []
                order.append(f"draft {slot}")
            per_draft[slot].append(_STEP_NAMES.get(key, key))
        else:
            order.append(_STEP_NAMES.get(step, step))
    return ", ".join(
        f"{name} ({', '.join(per_draft[name[6:]])})" if name.startswith("draft ") else name
        for name in order
    )


_ENTRY = re.compile(r"^\[(\d+)\] ", re.MULTILINE)
EXCERPT_CHARS = 320
# The entry kinds `projection.py` writes; anything else is shown as it is.
_KIND = re.compile(
    r"^(REQUEST \([^)]*\)|TOOL CALL \S+|TOOL RESULT \((?:ok|error)\)|ASSISTANT \(final\)|OUTCOME"
    r"|NEXT MESSAGE(?: \([^)]*\))?): ?(.*)\Z",
    re.DOTALL,
)
_MARKER = re.compile(r"^\.\.\. (?:\d+ tool call\(s\) omitted \.\.\.|\[projection cut\])$")
_HEADER = "Coding-agent turn."
# Per row on the turn screen; the stored text keeps 200 and 120.
TOOL_ARGS_CHARS = 72
TOOL_OK_CHARS = 100


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


@dataclass(frozen=True)
class Entry:
    """One projection entry. `number` is None for the header and the omission markers, which
    carry no citation number; `kind` is empty when the entry is not one `projection.py` writes."""

    number: int | None
    kind: str
    body: str


def parse_entries(projection: str) -> list[Entry]:
    out: list[Entry] = []
    starts = [(int(m.group(1)), m.start(), m.end()) for m in _ENTRY.finditer(projection)]
    head = projection[: starts[0][1]] if starts else projection
    out.extend(Entry(None, "", line) for line in head.splitlines() if line.strip())
    for i, (number, _start, end) in enumerate(starts):
        stop = starts[i + 1][1] if i + 1 < len(starts) else len(projection)
        lines = projection[end:stop].rstrip("\n").splitlines()
        # An omission marker has no number of its own and sits after the entry before it.
        markers: list[str] = []
        while lines and _MARKER.match(lines[-1]):
            markers.insert(0, lines.pop())
        chunk = "\n".join(lines)
        match = _KIND.match(chunk)
        if match:
            out.append(Entry(number, match.group(1), match.group(2)))
        else:
            out.append(Entry(number, "", chunk))
        out.extend(Entry(None, "", marker) for marker in markers)
    return out


def _prose(body: str) -> str:
    return "\n".join(line.rstrip() for line in body.splitlines() if line.strip())


def _clip_end(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _clip_middle(text: str, limit: int) -> str:
    """Keeps both ends: for tool arguments the key is at the front and the file name at the
    back, and both say more than the directory between them."""
    if len(text) <= limit:
        return text
    head = (limit - 1) // 2
    return text[:head].rstrip() + "…" + text[-(limit - 1 - head) :].lstrip()


def projection_grid(projection: str) -> Table:
    """The projection condensed for the turn screen. The header line is left out (the meta line
    has it), a tool call and its result share one row with the arguments shortened, blank
    lines are dropped, and each entry opens with its kind in bold so the request, the errors,
    and the next message can be found at a glance. The numbers are the ones a draft cites; the
    stored text, which `p` pages, is the evidence of record."""
    grid = Table.grid(padding=(0, 1))
    grid.add_column(justify="right", no_wrap=True, style="dim")
    grid.add_column(overflow="fold")
    entries = parse_entries(projection)
    i = 0
    while i < len(entries):
        entry = entries[i]
        i += 1
        if entry.number is None:
            if not entry.body.startswith(_HEADER):
                grid.add_row("", Text(entry.body, style="dim", overflow="fold"))
            continue
        if entry.kind.startswith("TOOL CALL "):
            name = entry.kind.removeprefix("TOOL CALL ")
            result = entries[i] if i < len(entries) else None
            if result is not None and result.number and result.kind.startswith("TOOL RESULT "):
                i += 1
                ok = result.kind == "TOOL RESULT (ok)"
                body = Text(overflow="fold")
                body.append(f"{name} ", style="bold" if ok else "bold red")
                body.append(_clip_middle(entry.body, TOOL_ARGS_CHARS))
                body.append("  │  ", style="dim")
                if ok:
                    body.append("ok: " + _clip_end(result.body, TOOL_OK_CHARS))
                else:
                    body.append("error: " + result.body, style="red")
                grid.add_row(f"[{entry.number}]→[{result.number}]", body)
            else:
                body = Text(overflow="fold")
                body.append(f"{name} ", style="bold")
                body.append(_clip_middle(entry.body, TOOL_ARGS_CHARS))
                body.append("  │  (no result)", style="dim")
                grid.add_row(f"[{entry.number}]", body)
            continue
        if entry.kind.startswith("TOOL RESULT "):  # a result whose call was omitted
            ok = entry.kind == "TOOL RESULT (ok)"
            body = Text(overflow="fold")
            body.append("tool result ", style="bold" if ok else "bold red")
            body.append(("ok: " if ok else "error: ") + entry.body, style="" if ok else "red")
            grid.add_row(f"[{entry.number}]", body)
            continue
        body = Text(overflow="fold")
        if entry.kind:
            body.append(f"{entry.kind}: ", style="bold")
            body.append(_prose(entry.body))
        else:
            body.append(entry.body)
        grid.add_row(f"[{entry.number}]", body)
    return grid


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
    # The question already names its choices; Rich's own list would repeat them, with the
    # navigation keys the screen header shows, and push narrow prompts onto two lines.
    def ask(question: str, choices: Sequence[str] | None) -> str:
        if choices is None:
            return Prompt.ask(question, console=console, default="", show_default=False)
        if question == SAVE_PROMPT:
            return Prompt.ask(
                question,
                choices=list(choices),
                console=console,
                show_choices=False,
                default="y",
                show_default=False,
            )
        return Prompt.ask(question, choices=list(choices), console=console, show_choices=False)

    return ask


def _answer(
    ask: Ask,
    question: str,
    choices: Sequence[str],
    *,
    back: bool = True,
    page: Callable[[], None] | None = None,
    rubric: Callable[[], None] | None = None,
) -> str:
    """One answer from the allowed set; `b` goes back, `p` pages the projection and `?` the
    rubric, then the question is asked again; `s` skips the item, `q` stops. A real choice wins
    over a key, so `b` at `preferred` picks draft B."""
    keys = [
        k
        for k in (
            *(["b"] if back else []),
            *(["p"] if page else []),
            *(["?"] if rubric else []),
            "s",
            "q",
        )
        if k not in choices
    ]
    allowed = [*choices, *keys]
    while True:
        try:
            value = ask(question, allowed).strip().lower()
        except InvalidResponse:
            continue
        if value in choices:
            return value
        if value == "q":
            raise StopLabeling
        if value == "s":
            raise SkipItem
        if value == "p" and page is not None:
            page()
        elif value == "?" and rubric is not None:
            rubric()
        elif value == "b" and back:
            raise GoBack


def _numbers(options: Sequence[str]) -> list[str]:
    return [str(i) for i in range(1, len(options) + 1)]


def _choice_table(choices: Mapping[str, str]) -> Table:
    table = Table(show_header=False, box=None, padding=(0, 1))
    for i, (key, desc) in enumerate(choices.items(), start=1):
        table.add_row(f"[bold]{i}[/bold]", key, Text(desc, style="dim"))
    return table


def _meta(item: Mapping[str, Any]) -> Text:
    """The header's facts on one line, so the projection starts near the top of the screen."""
    facts = item.get("facts", {})
    started_by = facts.get("caller") or item.get("caller") or "human"
    calls, errors = facts.get("tool_count", 0), facts.get("tool_error_count", 0)
    pairs = (
        ("project", str(item.get("project") or "unknown")),
        ("session started by", str(started_by)),
        ("closed by", closed_by_of(item)),
        ("outcome", str(item.get("outcome_reason") or "none")),
        (
            "tool calls",
            f"{calls} ({errors} error{'' if errors == 1 else 's'})" if errors else str(calls),
        ),
    )
    line = Text(overflow="fold")
    for k, (label, value) in enumerate(pairs):
        if k:
            line.append("  ·  ", style="dim")
        line.append(f"{label} ", style="bold")
        line.append(value)
    return line


def _projection_panel(projection: str) -> Panel:
    # Text() keeps bracketed evidence numbers and code from being read as Rich markup.
    return Panel(Text(projection), title="projection", border_style="cyan")


def _condensed_panel(projection: str) -> Panel:
    return Panel(
        projection_grid(projection),
        title="projection, condensed (p pages the full text)",
        border_style="cyan",
    )


def _draft_panel(draft: DraftView) -> Panel:
    if draft.declined:
        return Panel(
            Text("(no lesson)", style="dim"), title=f"draft {draft.slot}", border_style="magenta"
        )
    cites = " ".join(f"[{n}]" for n in draft.cited) or "nothing"
    return Panel(
        Text(draft.lesson or ""),
        title=f"draft {draft.slot}  (cites {cites})",
        border_style="magenta",
    )


def _print_draft(console: Console, projection: str, draft: DraftView) -> None:
    """A draft's panel, then excerpts of the projection entries it cites."""
    console.print(_draft_panel(draft))
    if draft.declined:
        return
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
        cites = " ".join(f"[{n}]" for n in draft.cited)
        console.print(
            Text(
                f"draft {draft.slot} cites {cites}, which are not in the projection; judge it as "
                "if it cited nothing",
                style="yellow",
            )
        )


def _side_by_side(drafts: Sequence[DraftView]) -> Table:
    grid = Table.grid(expand=True, padding=(0, 1))
    for _ in drafts:
        grid.add_column(ratio=1)
    grid.add_row(*(_draft_panel(d) for d in drafts))
    return grid


def _labels_panel(previous: Mapping[str, Any], drafts: Sequence[DraftView]) -> Panel:
    return Panel(Text(summary_line(previous, drafts)), title="current labels", border_style="green")


@dataclass
class ItemScreen:
    """One item on the terminal. Each run of questions clears the viewport and draws only what
    it is about (the turn, one draft, the drafts side by side), so the reviewer never scrolls
    back to find it; the terminal's scrollback keeps what was cleared, and `p` at any question
    pages the full projection, `?` the rubric."""

    console: Console
    index: int
    total: int
    item: Mapping[str, Any]
    projection: str
    drafts: Sequence[DraftView]
    previous: Mapping[str, Any] | None = None
    only: Sequence[str] | None = None

    def _fresh(self) -> None:
        if self.console.is_terminal:  # a pipe or log gets the screens one after another
            self.console.clear()
        self.console.print(
            Rule(f"item {self.index}/{self.total}  [dim]{self.item['item_id']}[/dim]")
        )
        self.console.print(Text(KEYS, style="dim"))
        if self.only:
            self.console.print(Text(f"only asking: {describe_steps(self.only)}", style="dim"))

    def show_all(self) -> None:
        """Everything at once, for the keep-or-redo decision in review mode."""
        self._fresh()
        self.console.print(_meta(self.item))
        self.console.print(_condensed_panel(self.projection))
        for draft in self.drafts:
            _print_draft(self.console, self.projection, draft)
        if self.previous is not None:
            self.console.print(_labels_panel(self.previous, self.drafts))

    def show(self, section: str) -> None:
        """A fresh screen saying what the next run of questions is about, and showing it."""
        if section == "note":
            return  # stays under the drafts it comments on
        self._fresh()
        if section == "turn":
            self.console.print(Rule("about the turn itself (not the drafts)", style="cyan"))
            self.console.print(_meta(self.item))
            self.console.print(_condensed_panel(self.projection))
            if self.previous is not None and self.only is None:
                self.console.print(_labels_panel(self.previous, self.drafts))
        elif section == "preferred":
            self.console.print(Rule("the drafts side by side", style="magenta"))
            self.console.print(_side_by_side(self.drafts))
        else:
            self.console.print(Rule(f"about draft {section}", style="magenta"))
            draft = next((d for d in self.drafts if d.slot == section), None)
            if draft is not None:
                _print_draft(self.console, self.projection, draft)

    def page_projection(self) -> None:
        """The full projection as stored, in the system pager."""
        self._page(_projection_panel(self.projection))

    def page_rubric(self) -> None:
        self._page(rubric_panel())

    def _page(self, panel: Panel) -> None:
        if not self.console.is_terminal:  # nothing to page on; print inline
            self.console.print(panel)
            return
        with self.console.pager():
            self.console.print(panel)


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


def _ask_step(step: str, ask: Ask, screen: ItemScreen) -> str:
    console, drafts = screen.console, screen.drafts

    def answer(question: str, choices: Sequence[str]) -> str:
        return _answer(
            ask, question, choices, page=screen.page_projection, rubric=screen.page_rubric
        )

    if step == "failed":
        return answer("failed? (y/n, u unclear, i incomplete)", _FAILED)
    if step == "category":
        console.print(_choice_table(CATEGORIES))
        return answer("category (number)", _numbers(CATEGORY_ORDER))
    if step == "worth_remembering":
        return answer("worth remembering for future sessions? (y/n)", _YN)
    if step == "lesson_kind":
        console.print(_choice_table(LESSON_KINDS))
        return answer("lesson kind (number)", _numbers(LESSON_ORDER))
    if step == "error_intended":
        return answer("was the tool error intended and used? (y/n)", _YN)
    if step == "preferred":
        slots = [d.slot.lower() for d in drafts]
        return answer(
            f"preferred draft? ({'/'.join(slots)}/same/neither)", (*slots, *PREFERRED_OTHER)
        )
    if step == "note":
        while True:
            value = ask("note (optional, Enter for none)", None).strip()
            if value.lower() == "q":
                raise StopLabeling
            if value.lower() == "b":
                raise GoBack
            if value.lower() == "p":
                screen.page_projection()
            elif value == "?":
                screen.page_rubric()
            else:
                return value
    slot, _, key = step.partition(".")
    if key == "grounded":
        return answer(f"draft {slot}: every claim traceable to the cited evidence? (y/n)", _YN)
    if key == "faithful":
        return answer(f"draft {slot}: reads the evidence right (cause, error, fact)? (y/n)", _YN)
    if key == "generalizable":
        return answer(f"draft {slot}: generalizable? (0 one-off, 1 project, 2 general)", _LEVELS)
    if key == "enforced_by":
        console.print(_choice_table(ENFORCED_BY))
        return answer(
            f"draft {slot}: lesson already enforced by (number)", _numbers(ENFORCED_ORDER)
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


def label_one(ask: Ask, screen: ItemScreen, has_tool_errors: bool = False) -> dict[str, Any] | None:
    """Collect one item's labels, or with `screen.only` just those steps on top of the previous
    row. Each run of questions gets a fresh screen showing what it is about. The reviewer can
    step back with b, page the projection with p, redo from the summary with r, and skip with
    s (None)."""
    console, drafts, only = screen.console, screen.drafts, screen.only
    previous = screen.previous
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
                    screen.show(section)
                try:
                    value = _ask_step(plan[i], ask, screen)
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
            save = _answer(
                ask,
                SAVE_PROMPT,
                ("y", "r"),
                back=False,
                page=screen.page_projection,
                rubric=screen.page_rubric,
            )
            if save == "y":
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
    try:
        if todo:
            console.print(rubric_panel())
            if backfill:
                console.print(
                    Text(
                        f"{len(backfill)} labeled item(s) come first to answer questions added "
                        "since they were labeled",
                        style="dim",
                    )
                )
            # The first item clears the screen, so the rubric waits to be read first.
            if ask(START_PROMPT, None).strip().lower() == "q":
                raise StopLabeling
        for index, item in enumerate(todo, start=1):
            item_id = item["item_id"]
            item_drafts = views(item)
            has_errors = errors_in(item)
            previous = existing.get(item_id)
            only: list[str] | None = None
            if previous is not None and not item_ids and not review:
                only = missing_steps(previous, item_drafts, has_errors)
            screen = ItemScreen(
                console,
                index,
                len(todo),
                item,
                store.read_projection(item_id),
                item_drafts,
                previous=previous,
                only=only,
            )
            if review:
                screen.show_all()
                try:
                    decision = _answer(
                        ask,
                        REVIEW_PROMPT,
                        ("k", "r"),
                        back=False,
                        page=screen.page_projection,
                        rubric=screen.page_rubric,
                    )
                except SkipItem:
                    skipped += 1
                    continue
                if decision == "k":
                    continue
            answers = label_one(ask, screen, has_errors)
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
