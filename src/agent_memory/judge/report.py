"""Scores verdicts against labels, freezes thresholds on the calibration split, and writes
report.md. The held-out split is scored only after thresholds are frozen.

Turn questions are scored once per item; draft questions and the write decision once per
(item, draft), per drafter. Runs built under `questions_v1` (one draft per item, `supported`
instead of `grounded`, the v1 write rule) are read through the same code: the aliases live in
`Verdict` and `draft_label`.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from .drafter import draft_hash
from .labeling import ENFORCED_ORDER, LESSON_ORDER, questions_left
from .lesson import entry_numbers, lesson_from_row
from .metrics import (
    BinaryMetrics,
    CascadeResult,
    accuracy,
    best_threshold,
    binary_metrics,
    cascade,
    choose_gate,
    fmt,
    paired_discrimination,
)
from .questions import DEFAULT_THRESHOLDS, QUESTIONS_VERSION, thresholds_from_run, write_gates
from .store import JudgeError, RunStore, utc_now_iso
from .triggers import CLOSED_BY, closed_by_of
from .verdict import Verdict

MARGIN_PTS = 0.03
CASCADE_SLACK = 0.01
CONFIDENCE_AUROC_TARGET = 0.75
WRITE_PRECISION_TARGET = 0.7
MISS_RATE_TARGET = 0.5
FAITHFUL_TARGET = 0.9


def draft_label(
    label: Mapping[str, Any], drafter: str | None, lesson_hash: str | None
) -> dict[str, Any]:
    """The draft-question labels that apply to one drafter's draft. v2 rows hold them per drafter
    and only count when the draft text is the one labeled; v1 rows hold them per item."""
    drafts = label.get("drafts")
    if isinstance(drafts, dict):
        entry = drafts.get(str(drafter or "")) or {}
        labeled_hash = entry.get("draft_hash")
        if entry and labeled_hash != lesson_hash:
            return {"declined": entry.get("declined")}
        return {
            "grounded": entry.get("grounded"),
            "faithful": entry.get("faithful"),
            "rule_supported": entry.get("rule_supported"),
            "store_exact": entry.get("store_exact"),
            "generalizable": entry.get("generalizable"),
            "enforced_by": entry.get("enforced_by"),
            "declined": entry.get("declined"),
            "claims": entry.get("claims"),
        }
    return {
        "grounded": label.get("supported"),
        "faithful": None,
        "generalizable": label.get("generalizable"),
        "enforced_by": label.get("enforced_by"),
        "declined": None,
    }


@dataclass
class Scored:
    """A verdict beside what it is scored against: its item, the item's label, and the labels
    of the draft it judged."""

    item: dict[str, Any]
    label: dict[str, Any]
    verdict: Verdict
    draft: dict[str, Any] = field(default_factory=dict)
    # Set on the row that stands for a whole item: what all of the item's verdicts paid.
    paid_ms: float | None = None
    paid_usd: float | None = None

    @property
    def latency_ms(self) -> float:
        return self.verdict.latency_ms if self.paid_ms is None else self.paid_ms

    @property
    def cost_usd(self) -> float:
        return self.verdict.cost_usd if self.paid_usd is None else self.paid_usd


def gate_truth(gate: str, row: Scored) -> bool | None:
    """The label a gate is scored against: the item's for failed and incomplete, the draft's
    for the rest. generalizable clears at level 1; enforced clears on none."""
    if gate in ("failed", "incomplete"):
        value = row.label.get(gate)
    elif gate in ("grounded", "supported", "faithful", "rule_supported"):
        value = row.draft.get("grounded" if gate == "supported" else gate)
    elif gate == "generalizable":
        level = row.draft.get("generalizable")
        value = None if level is None else int(level) >= 1
    elif gate == "enforced":
        enforced_by = row.draft.get("enforced_by")
        value = None if enforced_by is None else enforced_by == "none"
    else:
        return None
    if gate == "rule_supported" and type(value) is not bool:
        return None
    return None if value is None else bool(value)


def should_write(
    label: Mapping[str, Any],
    draft: Mapping[str, Any] | None = None,
    *,
    questions_version: str | None = QUESTIONS_VERSION,
) -> bool | None:
    """v4 admission is the exact-draft human answer, never turn-level opportunity.

    Historical runs retain worth-and-unenforced, including their missing-value semantics.
    """
    if questions_version == QUESTIONS_VERSION:
        value = (draft or {}).get("store_exact")
        if (draft or {}).get("declined") or type(value) is not bool:
            return None
        return value
    enforced_by = (draft if draft is not None else label).get("enforced_by")
    return bool(label.get("worth_remembering")) and enforced_by in (None, "none")


def enforced(label: Mapping[str, Any], draft: Mapping[str, Any] | None = None) -> bool:
    return (draft if draft is not None else label).get("enforced_by") not in (None, "none")


Key = tuple[str, str | None]


def load_scored(store: RunStore) -> tuple[list[dict[str, Any]], dict[str, dict[Key, Scored]]]:
    """Items, and per backend every labeled verdict row keyed by (item, drafter). Every
    verdict must have been judged with the run's question set."""
    items = store.items()
    labels = store.labels()
    version = store.read_run().get("questions_version")
    current = version == QUESTIONS_VERSION
    hashes = (
        {
            (item_id, row.get("drafter")): draft_hash(row.get("lesson"))
            for item_id, rows in store.drafts_by_item().items()
            for row in rows
        }
        if current
        else {}
    )
    by_item = {i["item_id"]: i for i in items}
    scored: dict[str, dict[Key, Scored]] = {}
    for name, rows in store.verdicts_by_file().items():
        for row in rows:
            verdict = Verdict(row)
            if verdict.questions_version != version:
                # Each question set has its own write gates and labels.
                raise JudgeError(
                    f"{name}.jsonl holds a verdict judged with "
                    f"{verdict.questions_version or 'no question set'}, but run "
                    f"{store.run_id!r} was built with {version or 'no question set'}; "
                    "a report cannot score two question sets as one"
                )
            item = by_item.get(verdict.item_id)
            label = labels.get(verdict.item_id)
            if item is None or label is None:
                continue
            if verdict.projection_hash != item.get("projection_hash"):
                continue  # judged a projection the run no longer has
            key = (item["item_id"], verdict.drafter)
            draft = draft_label(label, verdict.drafter, verdict.draft_hash)
            if current and (key not in hashes or hashes[key] != verdict.draft_hash):
                draft = {}  # An old matching label/verdict is not the current draft.
            scored.setdefault(verdict.backend, {})[key] = Scored(item, label, verdict, draft)
    return items, scored


def per_item(rows: Mapping[Key, Scored]) -> dict[str, Scored]:
    """One row per item for the turn answers, which every row of an item carries alike; the latency
    attributed to the item is what all its rows paid together."""
    out: dict[str, Scored] = {}
    paid: dict[str, float] = {}
    cost: dict[str, float] = {}
    for (item_id, _drafter), row in sorted(rows.items(), key=lambda kv: (kv[0][0], kv[0][1] or "")):
        paid[item_id] = paid.get(item_id, 0.0) + float(row.verdict.latency_ms)
        cost[item_id] = cost.get(item_id, 0.0) + float(row.verdict.cost_usd)
        out.setdefault(item_id, row)
    for item_id, row in out.items():
        # A run before questions_v4 reports an item's cost as its first row's own.
        current = row.verdict.questions_version == QUESTIONS_VERSION
        out[item_id] = replace(
            row, paid_ms=paid[item_id], paid_usd=cost[item_id] if current else None
        )
    return out


def per_drafter(rows: Mapping[Key, Scored]) -> dict[str | None, list[Scored]]:
    out: dict[str | None, list[Scored]] = {}
    for (_item_id, drafter), row in sorted(rows.items(), key=lambda kv: (kv[0][1] or "", kv[0][0])):
        out.setdefault(drafter, []).append(row)
    return out


@dataclass
class TurnEval:
    name: str
    n: int
    failed: BinaryMetrics | None
    incomplete: BinaryMetrics | None
    category_accuracy: float | None
    paired: float | None
    mean_latency_ms: float
    total_cost_usd: float
    majority_failed: float | None


@dataclass
class DraftEval:
    name: str
    drafter: str | None
    n: int
    grounded: BinaryMetrics | None
    faithful: BinaryMetrics | None
    generalizable_accuracy: float | None
    enforced_accuracy: float | None
    mean_latency_ms: float
    total_cost_usd: float
    write_precision: float | None
    miss_rate: float | None
    writes_enforced: int = 0
    actions: dict[str, int] = field(default_factory=dict)
    thresholds: dict[str, float] = field(default_factory=dict)
    gates: tuple[str, ...] = ()
    rule_supported: BinaryMetrics | None = None
    admission_scored: int = 0
    writes_scored: int = 0
    unscored: int = 0
    # Verdicts that are a review because the judge left a write gate unanswered.
    unanswered: int = 0


def _binary(rows: Sequence[Scored], gate: str, threshold: float) -> BinaryMetrics | None:
    probs, confs, labels = [], [], []
    for r in rows:
        p = r.verdict.p(gate)
        y = gate_truth(gate, r)
        if p is None or y is None:
            continue
        probs.append(p)
        confs.append(r.verdict.confidence(gate))
        labels.append(y)
    return binary_metrics(probs, confs, labels, threshold) if probs else None


def _pairs(rows: Sequence[Scored]) -> Iterator[tuple[float, float]]:
    """Every (failed, clean) pair of labeled turns from the same project."""
    by_project: dict[str, tuple[list[float], list[float]]] = {}
    for r in rows:
        p = r.verdict.p("failed")
        y = r.label.get("failed")
        if p is None or y is None:
            continue
        slot = by_project.setdefault(str(r.item.get("project")), ([], []))
        slot[0 if y else 1].append(p)
    return ((f, c) for failed, clean in by_project.values() for f in failed for c in clean)


def evaluate_turns(
    name: str, rows: Sequence[Scored], thresholds: Mapping[str, float], *, category_on_failed: bool
) -> TurnEval:
    cats = [
        (r.verdict.value("category"), r.label["category"])
        for r in rows
        if r.verdict.value("category")
        and r.label.get("category") is not None
        and (not category_on_failed or r.label.get("failed") is True)
    ]
    failed_labels = [bool(r.label["failed"]) for r in rows if r.label.get("failed") is not None]
    majority = (
        max(sum(failed_labels), len(failed_labels) - sum(failed_labels)) / len(failed_labels)
        if failed_labels
        else None
    )
    return TurnEval(
        name=name,
        n=len(rows),
        failed=_binary(rows, "failed", thresholds["failed"]),
        incomplete=_binary(rows, "incomplete", 0.5),
        category_accuracy=accuracy([c for c, _ in cats], [t for _, t in cats]),
        paired=paired_discrimination(_pairs(rows)),
        mean_latency_ms=sum(r.latency_ms for r in rows) / len(rows) if rows else 0.0,
        total_cost_usd=sum(r.cost_usd for r in rows),
        majority_failed=majority,
    )


def evaluate_drafts(
    name: str,
    drafter: str | None,
    rows: Sequence[Scored],
    thresholds: Mapping[str, float],
    gates: Sequence[str],
) -> DraftEval:
    gens = [
        (r.verdict.value("generalizable"), str(r.draft["generalizable"]))
        for r in rows
        if r.verdict.value("generalizable") is not None and r.draft.get("generalizable") is not None
    ]
    enf = [
        (r.verdict.value("enforced"), str(r.draft["enforced_by"]))
        for r in rows
        if r.verdict.value("enforced") is not None and r.draft.get("enforced_by") is not None
    ]
    actions: dict[str, int] = {"write_lesson": 0, "review": 0, "discard": 0}
    current = "rule_supported" in gates
    writes_worth = writes = worth = missed = writes_enforced = admission_scored = unscored = 0
    left_unanswered = 0
    for r in rows:
        # A declined draft is a discard whatever its answers, so no gate holds it.
        left_unanswered += not r.verdict.declined and None in r.verdict.gates().values()
        action = r.verdict.action_at(thresholds).action
        actions[action] += 1
        is_worth = should_write(
            r.label, r.draft, questions_version=QUESTIONS_VERSION if current else None
        )
        if action == "write_lesson":
            writes_enforced += enforced(r.label, r.draft)
        if is_worth is None:
            unscored += 1
            continue
        admission_scored += 1
        worth += is_worth
        if action == "write_lesson":
            writes += 1
            writes_worth += is_worth
        elif is_worth:
            missed += 1
    return DraftEval(
        name=name,
        drafter=drafter,
        n=len(rows),
        # A questions_v1 run froze this cut under its old name.
        grounded=_binary(
            rows, "grounded", thresholds["supported" if "supported" in gates else "grounded"]
        ),
        faithful=_binary(rows, "faithful", thresholds["faithful"]),
        generalizable_accuracy=accuracy([g for g, _ in gens], [t for _, t in gens]),
        enforced_accuracy=accuracy([e for e, _ in enf], [t for _, t in enf]),
        mean_latency_ms=sum(r.verdict.draft_latency_ms for r in rows) / len(rows) if rows else 0.0,
        total_cost_usd=sum(
            r.verdict.draft_cost_usd if current else r.verdict.cost_usd for r in rows
        ),
        write_precision=writes_worth / writes if writes else None,
        miss_rate=missed / worth if worth else None,
        writes_enforced=writes_enforced,
        actions=actions,
        thresholds=dict(thresholds),
        gates=tuple(gates),
        rule_supported=_binary(rows, "rule_supported", thresholds["rule_supported"])
        if current
        else None,
        admission_scored=admission_scored,
        writes_scored=writes,
        unscored=unscored,
        unanswered=left_unanswered,
    )


def reference_backend(scored: Mapping[str, Mapping[Key, Scored]]) -> str | None:
    llms = [name for name in scored if name.startswith("llm:")]
    return max(llms, key=lambda n: len(scored[n])) if llms else None


def cascade_against(
    cheap: Mapping[str, Scored],
    expensive: Mapping[str, Scored],
    threshold: float,
    gate: float | None,
) -> CascadeResult | None:
    shared = [i for i in cheap if i in expensive and cheap[i].label.get("failed") is not None]
    if not shared:
        return None
    confs, cheap_ok, exp_ok, cheap_cost, exp_cost = [], [], [], [], []
    for i in shared:
        y = bool(cheap[i].label["failed"])
        pc, pe = cheap[i].verdict.p("failed"), expensive[i].verdict.p("failed")
        if pc is None or pe is None:
            continue
        confs.append(cheap[i].verdict.confidence("failed"))
        cheap_ok.append((pc >= threshold) == y)
        exp_ok.append((pe >= 0.5) == y)
        cheap_cost.append(cheap[i].latency_ms)
        exp_cost.append(expensive[i].latency_ms)
    if not confs:
        return None
    if gate is None:
        target = sum(exp_ok) / len(exp_ok) - CASCADE_SLACK
        return choose_gate(confs, cheap_ok, exp_ok, cheap_cost, exp_cost, target)
    return cascade(confs, cheap_ok, exp_ok, cheap_cost, exp_cost, gate)


def frozen_gates(questions_version: str | None) -> tuple[str, ...]:
    """The thresholds a freeze chooses: failed, plus every write gate with a label."""
    rest = (g for g in write_gates(questions_version) if g not in ("failed", "recurrence"))
    return ("failed", *rest)


def freeze_thresholds(
    scored: Mapping[str, Mapping[Key, Scored]],
    calibration_ids: set[str],
    questions_version: str | None,
) -> dict[str, Any]:
    """Per backend: the accuracy-maximising cut on every labeled gate, plus the cascade gate
    against the reference judge, all from calibration items only."""
    reference = reference_backend(scored)
    backends: dict[str, Any] = {}
    for name, rows_by_key in scored.items():
        rows = [r for (i, _), r in rows_by_key.items() if i in calibration_ids]
        unique = [r for i, r in per_item(rows_by_key).items() if i in calibration_ids]
        gates = frozen_gates(questions_version)
        thresholds = {k: v for k, v in DEFAULT_THRESHOLDS.items() if k == "band" or k in gates}
        for gate in gates:
            pool = unique if gate == "failed" else rows
            pairs = [
                (p, y)
                for r in pool
                if (p := r.verdict.p(gate)) is not None and (y := gate_truth(gate, r)) is not None
            ]
            if pairs:
                thresholds[gate] = round(
                    best_threshold([p for p, _ in pairs], [y for _, y in pairs])[0], 4
                )
        gate = None
        if reference and name != reference:
            cheap = {i: r for i, r in per_item(rows_by_key).items() if i in calibration_ids}
            expensive = {
                i: r for i, r in per_item(scored[reference]).items() if i in calibration_ids
            }
            result = cascade_against(cheap, expensive, thresholds["failed"], None)
            gate = result.gate if result else None
        backends[name] = {"thresholds": thresholds, "gate": gate}
    return {"frozen_at": utc_now_iso(), "reference": reference, "backends": backends}


# Markdown


def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return out


_TURN_HEADER = [
    "backend",
    "n",
    "failed acc",
    "failed AUROC",
    "Brier",
    "ECE",
    "conf→err AUROC",
    "incomplete acc",
    "incomplete AUROC",
    "category acc",
    "paired",
    "mean s/item",
    "USD",
]


def _turn_rows(evals: Sequence[TurnEval]) -> list[list[str]]:
    rows = []
    for e in evals:
        f, inc = e.failed, e.incomplete
        rows.append(
            [
                e.name,
                str(e.n),
                fmt(f.accuracy if f else None),
                fmt(f.auroc if f else None),
                fmt(f.brier if f else None),
                fmt(f.ece if f else None),
                fmt(f.confidence_error_auroc if f else None),
                fmt(inc.accuracy if inc else None),
                fmt(inc.auroc if inc else None),
                fmt(e.category_accuracy),
                fmt(e.paired),
                f"{e.mean_latency_ms / 1000:.2f}",
                f"{e.total_cost_usd:.4f}",
            ]
        )
    return rows


_DRAFT_HEADER = [
    "backend",
    "drafter",
    "n",
    "grounded acc",
    "grounded AUROC",
    "conf→err AUROC (B)",
    "faithful acc",
    "faithful AUROC",
    "generalizable acc",
    "enforced acc",
    "mean s/draft",
    "USD",
]


def _draft_rows(evals: Sequence[DraftEval]) -> list[list[str]]:
    rows = []
    for e in evals:
        g, f = e.grounded, e.faithful
        rows.append(
            [
                e.name,
                e.drafter or "—",
                str(e.n),
                fmt(g.accuracy if g else None),
                fmt(g.auroc if g else None),
                fmt(g.confidence_error_auroc if g else None),
                fmt(f.accuracy if f else None),
                fmt(f.auroc if f else None),
                fmt(e.generalizable_accuracy),
                fmt(e.enforced_accuracy),
                f"{e.mean_latency_ms / 1000:.2f}",
                f"{e.total_cost_usd:.4f}",
            ]
        )
    return rows


def _write_rows(evals: Sequence[DraftEval]) -> list[list[str]]:
    return [
        [
            e.name,
            e.drafter or "—",
            ", ".join(f"{g} {fmt(e.thresholds.get(g), 2)}" for g in e.gates if g in e.thresholds),
            str(e.actions.get("write_lesson", 0)),
            str(e.actions.get("review", 0)),
            str(e.actions.get("discard", 0)),
            fmt(e.write_precision),
            fmt(e.miss_rate),
            str(e.writes_enforced),
        ]
        for e in evals
    ]


def _unanswered_note(evals: Sequence[DraftEval]) -> list[str]:
    """Per backend, how many verdicts are a review because the judge left a write gate
    unanswered. Nothing when there are none, so a run without any reads as it did before."""
    counts: dict[str, list[int]] = {}
    for e in evals:
        slot = counts.setdefault(e.name, [0, 0])
        slot[0] += e.unanswered
        slot[1] += e.n
    shown = [f"`{name}` {count} of {n}" for name, (count, n) in counts.items() if count]
    if not shown:
        return []
    return [
        "",
        f"Verdicts with an unanswered gate: {', '.join(shown)}. The judge gave no usable "
        "answer to at least one write gate, so each is a review. An unanswered question "
        "counts toward no metric and no fitted threshold.",
    ]


def _rate(count: int, total: int) -> str:
    return f"{count} ({count / total:.0%})" if total else "0"


def label_notes(
    items: Sequence[dict[str, Any]],
    labels: Mapping[str, dict[str, Any]],
    drafts: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    questions_version: str | None = None,
) -> list[str]:
    """What the human-only labels say on their own: how turns closed, what moved, what the
    drafters produced and how the reviewer rated them, and how often a tool error was the point."""
    lines: list[str] = []
    v2 = any(isinstance(label.get("drafts"), dict) for label in labels.values())
    labeled = [i for i in items if i["item_id"] in labels]

    # How turns closed, against the item-level labels.
    by_closed: dict[str, dict[str, int]] = {}
    for item in labeled:
        label = labels[item["item_id"]]
        slot = by_closed.setdefault(
            closed_by_of(item), dict.fromkeys(("n", "failed", "decided", "incomplete", "worth"), 0)
        )
        slot["n"] += 1
        if label.get("failed") is not None:
            slot["decided"] += 1
            slot["failed"] += bool(label["failed"])
        slot["incomplete"] += bool(label.get("incomplete"))
        slot["worth"] += bool(label.get("worth_remembering"))
    if by_closed:
        lines.extend(["### How the turns closed (`closed_by`)", ""])
        rows = [
            [
                key,
                str(c["n"]),
                f"{c['failed']}/{c['decided']}"
                + (f" ({c['failed'] / c['decided']:.0%})" if c["decided"] else ""),
                _rate(c["incomplete"], c["n"]),
                _rate(c["worth"], c["n"]),
            ]
            for key in CLOSED_BY
            if (c := by_closed.get(key))
        ]
        lines.extend(
            _table(
                ["closed by", "labeled", "failed (of decided)", "incomplete", "worth remembering"],
                rows,
            )
        )
        lines.append("")

    if v2:
        kinds = dict.fromkeys(LESSON_ORDER, 0)
        for label in labels.values():
            if label.get("lesson_kind") in kinds:
                kinds[label["lesson_kind"]] += 1
        worth = sum(1 for label in labels.values() if label.get("worth_remembering"))
        if worth:
            lines.extend(
                [
                    "### Lesson kinds (`lesson_kind`, items worth remembering)",
                    "",
                    *_table(["kind", "items"], [[k, _rate(v, worth)] for k, v in kinds.items()]),
                    "",
                ]
            )
        moved_items = [label for label in labels.values() if label.get("history")]
        moved_values = sum(
            len(h.get("changed", {})) for label in moved_items for h in label["history"]
        )
        lines.extend(
            [
                "### Label changes (`history`)",
                "",
                f"{moved_values} value(s) changed on {len(moved_items)} item(s) after their first "
                f"labeling, out of {len(labels)} labeled items.",
                "",
            ]
        )
        lines.extend(drafter_notes(labels, drafts, questions_version=questions_version))
    else:
        by_caller: dict[str, dict[str, int]] = {}
        for item in labeled:
            value = labels[item["item_id"]].get("enforced_by")
            if value is not None:
                kind = str(item.get("caller") or "human")
                counts = by_caller.setdefault(kind, dict.fromkeys(ENFORCED_ORDER, 0))
                counts[str(value)] = counts.get(str(value), 0) + 1
        if by_caller:
            lines.extend(["### Drafted lessons already enforced elsewhere (`enforced_by`)", ""])
            rows = []
            for kind, counts in sorted(by_caller.items()):
                n = sum(counts.values())
                rows.append([kind, str(n)] + [_rate(counts.get(k, 0), n) for k in ENFORCED_ORDER])
            lines.extend(_table(["session started by", "drafts labeled", *ENFORCED_ORDER], rows))
            lines.append("")

    err_total = err_intended = err_failed = 0
    for item in labeled:
        label = labels[item["item_id"]]
        if (
            item.get("facts", {}).get("tool_error_count")
            and label.get("error_intended") is not None
        ):
            err_total += 1
            err_intended += bool(label["error_intended"])
            err_failed += bool(label.get("failed"))
    if err_total:
        lines.extend(
            [
                "### Tool errors (`error_intended`)",
                "",
                f"{err_total} labeled items had a tool error; in {err_intended} "
                f"({err_intended / err_total:.0%}) the error was intended and used, and "
                f"{err_failed} ({err_failed / err_total:.0%}) were labeled failed. A tool-error "
                "trigger (T3) alone would be right about failure "
                f"{err_failed / err_total:.0%} of the time.",
                "",
            ]
        )
    return lines


def drafter_notes(
    labels: Mapping[str, dict[str, Any]],
    drafts: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    questions_version: str | None = None,
) -> list[str]:
    """Per drafter: how often it declined and whether it was right to, how the reviewer rated its
    lessons, which arm was preferred, and what it cost. Criterion 5 is read from this table."""
    names = sorted({str(d.get("drafter") or "") for rows in drafts.values() for d in rows})
    if not names:
        return []
    current = questions_version == QUESTIONS_VERSION
    rating_keys = (
        "grounded",
        "faithful",
        "reusable",
        "unenforced",
        "rule_supported",
        "store_exact",
    )
    stats: dict[str, dict[str, float]] = {
        n: dict.fromkeys(
            (
                "drafts",
                "declined",
                "declined_right",
                "labeled",
                "grounded",
                "faithful",
                "reusable",
                "unenforced",
                "preferred",
                "latency",
                "cost",
                "rule_supported",
                "store_exact",
                "declined_scored",
                *(f"{key}_n" for key in rating_keys),
            ),
            0.0,
        )
        for n in names
    }
    same = neither = compared = 0
    for item_id, rows in drafts.items():
        label = labels.get(item_id)
        labeled_drafts = (label or {}).get("drafts")
        current_hashes = {
            str(row.get("drafter") or ""): draft_hash(row.get("lesson")) for row in rows
        }
        for row in rows:
            drafter = str(row.get("drafter") or "")
            s = stats[drafter]
            s["drafts"] += 1
            s["latency"] += float(row.get("latency_ms") or 0.0)
            s["cost"] += float(row.get("cost_usd") or 0.0)
            declined = row.get("lesson") is None
            s["declined"] += declined
            if label is None:
                continue
            if declined:
                if current and type(label.get("worth_remembering")) is not bool:
                    continue
                s["declined_scored"] += 1
                s["declined_right"] += not label.get("worth_remembering")
                continue
            if not isinstance(labeled_drafts, dict):
                continue  # Historical item-level labels did not rate separate drafter arms.
            entry = draft_label(label, drafter, current_hashes[drafter])
            if current:
                if (labeled_drafts.get(drafter) or {}).get("draft_hash") != current_hashes[drafter]:
                    continue
                s["labeled"] += 1
                values = {
                    "grounded": entry.get("grounded"),
                    "faithful": entry.get("faithful"),
                    "rule_supported": entry.get("rule_supported"),
                    "store_exact": entry.get("store_exact"),
                    "reusable": None
                    if entry.get("generalizable") is None
                    else entry["generalizable"] >= 1,
                    "unenforced": None
                    if entry.get("enforced_by") is None
                    else entry["enforced_by"] == "none",
                }
                for key, value in values.items():
                    if type(value) is bool:
                        s[f"{key}_n"] += 1
                        s[key] += value
                continue
            if entry.get("grounded") is None:
                continue
            s["labeled"] += 1
            s["grounded"] += bool(entry.get("grounded"))
            s["faithful"] += bool(entry.get("faithful"))
            s["reusable"] += int(entry.get("generalizable") or 0) >= 1
            s["unenforced"] += entry.get("enforced_by") == "none"
        if (
            label is not None
            and len(rows) >= 2
            and label.get("preferred") is not None
            and isinstance(labeled_drafts, dict)
            and current_hashes
            == {name: entry.get("draft_hash") for name, entry in labeled_drafts.items()}
        ):
            compared += 1
            preferred = label["preferred"]
            if preferred == "same":
                same += 1
            elif preferred == "neither":
                neither += 1
            elif preferred in stats:
                stats[preferred]["preferred"] += 1
    lines = [
        "### Drafters",
        "",
        (
            (
                "Ratings show yes/scored for matching non-declined drafts; missing or unclear "
                "answers are unscored. `declined right` uses only declines with a decided "
                "turn-worth label (opportunity, not exact-draft admission). "
            )
            if current
            else (
                "`declined right` is the share of declined drafts on items labeled not worth "
                "remembering. Rates in the middle columns are over labeled, non-declined drafts. "
            )
        )
        + f"`preferred` is over the {compared} item(s) where both drafts were compared"
        + (f" ({same} same, {neither} neither)" if compared else "")
        + ".",
        "",
    ]
    rows = []
    for name in names:
        s = stats[name]
        drafts_n, declined, labeled = int(s["drafts"]), int(s["declined"]), int(s["labeled"])
        declined_denominator = s["declined_scored"] if current else declined
        rows.append(
            [
                name,
                str(drafts_n),
                _rate(declined, drafts_n),
                fmt(
                    s["declined_right"] / declined_denominator if declined_denominator else None, 2
                ),
                str(labeled),
                *(
                    [
                        f"{int(s[key])}/{int(s[f'{key}_n'])}" if s[f"{key}_n"] else "n/a (0 scored)"
                        for key in rating_keys
                    ]
                    if current
                    else [fmt(s[key] / labeled if labeled else None, 2) for key in rating_keys[:4]]
                ),
                _rate(int(s["preferred"]), compared),
                f"{s['latency'] / drafts_n / 1000:.2f}" if drafts_n else "n/a",
                f"{s['cost']:.4f}",
            ]
        )
    lines.extend(
        _table(
            [
                "drafter",
                "drafts",
                "declined",
                "declined right",
                "labeled",
                "grounded",
                "faithful",
                "generalizable ≥1",
                "enforced none",
                *(["rule supported", "store exact"] if current else []),
                "preferred",
                "mean s/draft",
                "USD",
            ],
            rows,
        )
    )
    lines.append("")
    if len(names) >= 2 and compared:
        lines.append(
            f"Criterion 5 (faithful ≥ {FAITHFUL_TARGET:.0%} and preferred at least as often as "
            "the other arm):"
        )
        for name in names:
            s = stats[name]
            labeled = int(s["faithful_n"] if current else s["labeled"])
            faithful = s["faithful"] / labeled if labeled else 0.0
            others = max((stats[o]["preferred"] for o in names if o != name), default=0.0)
            ok = labeled > 0 and faithful >= FAITHFUL_TARGET and s["preferred"] >= others
            if not s["drafts"]:
                lines.append(f"- `{name}`: no drafts")
                continue
            lines.append(
                f"- `{name}`: {'yes' if ok else 'no'} (faithful {faithful:.2f}, preferred "
                f"{int(s['preferred'])} vs {int(others)}, "
                f"{s['cost'] / int(s['drafts']):.4f} USD/draft)"
            )
        lines.append("")
    return lines


def claim_notes(
    store: RunStore,
    labels: Mapping[str, dict[str, Any]],
    drafts: Mapping[str, Sequence[Mapping[str, Any]]],
    scored: Mapping[str, Mapping[Key, Scored]],
    *,
    held_out: bool,
) -> list[str]:
    """Claim diagnostics leave draft-level gates and historical reports untouched."""
    by_drafter: dict[str, list[Mapping[str, Any]]] = {}
    for rows in drafts.values():
        for row in rows:
            if "claims" in row:
                by_drafter.setdefault(str(row.get("drafter") or ""), []).append(row)
    if not by_drafter:
        return []
    lines = [
        "## Claims",
        "",
        "Mean claims/draft includes declines (zero claims). Uncited is over claims; absent "
        "citations is over citation references checked against stored projection entries. "
        "Grounded and faithful rates use labeled claims; kind agreement uses items with a "
        "lesson-kind label and a matching labeled draft.",
        "",
    ]
    table = []
    unchecked = 0
    for name, rows in sorted(by_drafter.items()):
        claims_n = uncited = absent = citations_n = kind_n = kind_right = 0
        truth: dict[str, list[bool]] = {"grounded": [], "faithful": []}
        for row in rows:
            lesson = lesson_from_row(row)
            if lesson is None:
                continue
            claims_n += len(lesson.claims)
            uncited += sum(not claim.cited for claim in lesson.claims)
            citations = [n for claim in lesson.claims for n in claim.cited]
            try:
                numbers = entry_numbers(store.read_projection(row["item_id"]))
            except JudgeError:
                unchecked += len(citations)
            else:
                citations_n += len(citations)
                absent += sum(n not in numbers for n in citations)
            label = labels.get(row["item_id"]) or {}
            entry = (label.get("drafts") or {}).get(name) or {}
            if entry.get("draft_hash") != draft_hash(row.get("lesson")):
                continue
            for claim_label in (entry.get("claims") or [])[: len(lesson.claims)]:
                for key in truth:
                    if claim_label.get(key) is not None:
                        truth[key].append(bool(claim_label[key]))
            if label.get("lesson_kind") is not None:
                kind_n += 1
                kind_right += lesson.kind == label["lesson_kind"]
        table.append(
            [
                name,
                str(len(rows)),
                f"{claims_n / len(rows):.2f}",
                str(claims_n),
                fmt(uncited / claims_n if claims_n else None),
                fmt(absent / citations_n if citations_n else None),
                str(len(truth["grounded"])),
                fmt(sum(truth["grounded"]) / len(truth["grounded"]) if truth["grounded"] else None),
                str(len(truth["faithful"])),
                fmt(sum(truth["faithful"]) / len(truth["faithful"]) if truth["faithful"] else None),
                str(kind_n),
                fmt(kind_right / kind_n if kind_n else None),
            ]
        )
    lines.extend(
        _table(
            [
                "drafter",
                "drafts",
                "mean claims/draft",
                "claims",
                "uncited share",
                "absent citation share",
                "grounded n",
                "grounded rate",
                "faithful n",
                "faithful rate",
                "kind n",
                "kind agreement",
            ],
            table,
        )
    )
    if unchecked:
        lines.extend(
            [
                "",
                f"{unchecked} citation reference(s) could not be checked: "
                "their stored projections are unavailable.",
            ]
        )
    lines.extend(
        [
            "",
            "### Claim decisions",
            "",
            "Accuracy, precision, recall and F1 use per-claim yes/no values, not the "
            "derived draft gates. Held-out decisions stay locked until freeze.",
            "",
        ]
    )
    table = []
    for split in ("calibration", "held_out") if held_out else ("calibration",):
        for backend, rows in sorted(scored.items()):
            for key in ("grounded", "faithful"):
                pairs = []
                for row in rows.values():
                    if row.item.get("split") != split:
                        continue
                    for i, claim_label in enumerate(row.draft.get("claims") or [], 1):
                        value = row.verdict.value(f"{key}.{i}")
                        label = claim_label.get(key)
                        if value in ("yes", "no") and label is not None:
                            pairs.append((value == "yes", bool(label)))
                if not pairs:
                    continue
                tp = sum(predicted and label for predicted, label in pairs)
                fp = sum(predicted and not label for predicted, label in pairs)
                fn = sum(not predicted and label for predicted, label in pairs)
                table.append(
                    [
                        backend,
                        split,
                        key,
                        str(len(pairs)),
                        fmt(sum(p == y for p, y in pairs) / len(pairs)),
                        fmt(tp / (tp + fp) if tp + fp else None),
                        fmt(tp / (tp + fn) if tp + fn else None),
                        fmt(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None),
                    ]
                )
    lines.extend(
        _table(
            ["backend", "split", "question", "n", "accuracy", "precision", "recall", "F1"], table
        )
    )
    return [*lines, ""]


def _slices(
    scored: Mapping[str, Mapping[Key, Scored]],
    ids: set[str],
    run: Mapping[str, Any],
    use_frozen: bool,
) -> list[str]:
    """Turn-question accuracy by how the turn closed and by the incomplete label."""
    names = sorted(scored)
    if not names:
        return []

    def table(title: str, key: Callable[[Scored], str], order: Sequence[str]) -> list[str]:
        rows = []
        for value in order:
            cells = [value]
            n_any = None
            for name in names:
                unique = [
                    r for i, r in per_item(scored[name]).items() if i in ids and key(r) == value
                ]
                if n_any is None:
                    n_any = len(unique)
                thresholds = (
                    thresholds_from_run(run, name) if use_frozen else dict(DEFAULT_THRESHOLDS)
                )
                metrics = _binary(unique, "failed", thresholds["failed"])
                inc = _binary(unique, "incomplete", 0.5)
                cells.append(
                    f"{fmt(metrics.accuracy if metrics else None)} / "
                    f"{fmt(inc.accuracy if inc else None)}"
                )
            if n_any:
                rows.append([cells[0], str(n_any), *cells[1:]])
        if not rows:
            return []
        return [
            f"### {title}",
            "",
            *_table(
                [
                    title.split(" ")[-1].strip("`"),
                    "items",
                    *(f"{n} failed / incomplete acc" for n in names),
                ],
                rows,
            ),
            "",
        ]

    lines = table("By `closed_by`", lambda r: closed_by_of(r.item), CLOSED_BY)
    has_incomplete = any("incomplete" in r.label for name in names for r in scored[name].values())
    if has_incomplete:
        lines += table(
            "By the `incomplete` label",
            lambda r: "incomplete" if r.label.get("incomplete") else "finished",
            ("finished", "incomplete"),
        )
    return lines


def _criteria(
    evals: Sequence[TurnEval],
    draft_evals: Sequence[DraftEval],
    reference: str | None,
    cascades: Mapping[str, CascadeResult | None],
    *,
    held_out: bool,
    questions_version: str | None,
) -> list[str]:
    lines = []
    ref = next((e for e in evals if e.name == reference), None)
    if ref is None or ref.failed is None:
        return ["No reference `llm:` backend with labeled verdicts; criteria need one."]
    ref_acc = ref.failed.accuracy or 0.0
    majority = ref.majority_failed or 0.0
    ref_b = [e for e in draft_evals if e.name == reference]
    ref_grounded = (
        max((e.grounded.accuracy or 0.0) for e in ref_b if e.grounded)
        if any(e.grounded for e in ref_b)
        else 0.0
    )
    for e in evals:
        if e.name == reference or e.failed is None:
            continue
        acc = e.failed.accuracy or 0.0
        mine = [b for b in draft_evals if b.name == e.name]
        grounded = (
            max((b.grounded.accuracy or 0.0) for b in mine if b.grounded)
            if any(b.grounded for b in mine)
            else 0.0
        )
        cheap = e.mean_latency_ms <= 0.1 * ref.mean_latency_ms if ref.mean_latency_ms else False
        c1 = acc >= ref_acc - MARGIN_PTS and grounded >= ref_grounded - MARGIN_PTS and cheap
        beats_majority = acc > majority
        c2 = (e.failed.confidence_error_auroc or 0.0) >= CONFIDENCE_AUROC_TARGET
        casc = cascades.get(e.name)
        c3 = bool(
            casc
            and casc.accuracy >= ref_acc - CASCADE_SLACK
            and casc.cost <= 0.5 * ref.mean_latency_ms * e.n
        )
        lines.append(
            f"- `{e.name}`: (1) within {MARGIN_PTS * 100:.0f} pts at ≤10% latency: "
            f"{'yes' if c1 else 'no'} (failed {acc:.3f} vs {ref_acc:.3f}, majority class "
            f"{majority:.3f}{' not beaten' if not beats_majority else ''}; grounded {grounded:.3f} "
            f"vs {ref_grounded:.3f}; {e.mean_latency_ms / 1000:.2f}s vs "
            f"{ref.mean_latency_ms / 1000:.2f}s); "
            f"(2) confidence AUROC ≥ {CONFIDENCE_AUROC_TARGET}: {'yes' if c2 else 'no'} "
            f"({fmt(e.failed.confidence_error_auroc)})"
            + (
                f"; (3) cascade ≥ ref-1 pt at ≤50% cost: {'yes' if c3 else 'no'}"
                f" (gate {casc.gate:.2f}, acc {casc.accuracy:.3f}, escalated "
                f"{casc.escalated_share:.0%})"
                if casc
                else ""
            )
        )
        if e.name == "rules":
            c4 = acc >= ref_acc - MARGIN_PTS
            lines.append(
                f"- `rules` within {MARGIN_PTS * 100:.0f} pts on failed (criterion 4): "
                f"{'yes' if c4 else 'no'} (majority class {majority:.3f}"
                f"{' not beaten' if not beats_majority else ' beaten'})"
            )
    if questions_version != "questions_v1" and held_out:
        for b in ref_b:
            ok = (
                b.write_precision is not None
                and b.miss_rate is not None
                and b.write_precision >= WRITE_PRECISION_TARGET
                and b.miss_rate <= MISS_RATE_TARGET
            )
            lines.append(
                f"- {'v4' if questions_version == QUESTIONS_VERSION else 'v2'} write rule "
                f"with `{reference}` on `{b.drafter or '—'}` (criterion 6: "
                f"precision ≥ {WRITE_PRECISION_TARGET}, miss rate ≤ {MISS_RATE_TARGET}): "
                f"{'yes' if ok else 'no'} (precision {fmt(b.write_precision)}, miss rate "
                f"{fmt(b.miss_rate)}, {b.actions.get('write_lesson', 0)} writes)"
            )
    return lines


def unlabeled_calibration(
    store: RunStore, items: Sequence[Mapping[str, Any]], *, current: bool
) -> tuple[int, int]:
    """The calibration items a freeze would go short of: how many have no label, and how many
    have a label with questions left. The second is only known for the current question set,
    the one a labeling session still asks."""
    labels = store.labels()
    drafts = store.drafts_by_item() if current else {}
    without = left = 0
    for item in items:
        if item.get("split") != "calibration":
            continue
        row = labels.get(item["item_id"])
        if row is None:
            without += 1
        elif current:
            try:
                left += bool(questions_left(item, row, drafts.get(item["item_id"], [])))
            except JudgeError:
                left += 1  # More drafts than a session can show: they cannot all be labeled.
    return without, left


def _freeze(
    store: RunStore,
    run: Mapping[str, Any],
    items: Sequence[Mapping[str, Any]],
    scored: Mapping[str, Mapping[Key, Scored]],
    calibration: set[str],
    *,
    partial: bool,
) -> dict[str, Any]:
    """Choose the thresholds and lock them in the run. They are chosen once, so a freeze while
    calibration items are not fully labeled has to be asked for, and the run keeps the count."""
    version = run.get("questions_version")
    if run.get("frozen"):
        raise JudgeError("thresholds are already frozen for this run")
    current = version == QUESTIONS_VERSION
    without, left = unlabeled_calibration(store, items, current=current)
    short = without + left
    if short and not partial:
        allow = "pass --partial to freeze on the labels there are"
        if current:
            raise JudgeError(
                f"{short} of {len(calibration)} calibration items are not fully labeled "
                f"({without} without a label, {left} with questions left). Finish them with "
                f"`mem judge label`, or {allow}."
            )
        raise JudgeError(
            f"{short} of {len(calibration)} calibration items have no label, and a run built "
            f"with {version or 'no question set'} can no longer be labeled; {allow}."
        )
    chosen = freeze_thresholds(scored, calibration, version)
    if short:
        chosen["partial"] = {"calibration": len(calibration), "unlabeled": short}
    return store.update_run(frozen=chosen)


def build_report(store: RunStore, *, freeze: bool = False, partial: bool = False) -> str:
    """Score the run and write report.md. `freeze` chooses the thresholds first; `partial`
    lets it do so while calibration items are not fully labeled."""
    run = store.read_run()
    version = run.get("questions_version")
    current = version == QUESTIONS_VERSION
    gates = write_gates(version)
    items, scored = load_scored(store)
    if not scored:
        raise JudgeError("no labeled verdicts yet; run `mem judge decide` and `mem judge label`")
    calibration = {i["item_id"] for i in items if i.get("split") == "calibration"}
    held_out = {i["item_id"] for i in items if i.get("split") == "held_out"}
    if freeze:
        run = _freeze(store, run, items, scored, calibration, partial=partial)
    frozen = run.get("frozen")
    reference = (frozen or {}).get("reference") or reference_backend(scored)
    labeled = {i for rows in scored.values() for (i, _) in rows}
    drafters = sorted(run.get("drafters") or {}) or ([run["drafter"]] if run.get("drafter") else [])

    lines = [
        f"# Judge report: {store.run_id}",
        "",
        f"Generated {utc_now_iso()}. Versions: {run.get('triggers_version') or 'unknown'}, "
        f"{run.get('projection_version') or 'unknown'}, "
        f"{run.get('questions_version') or 'unknown'}, "
        f"drafters {', '.join(f'`{d}`' for d in drafters) or 'none'}. Items {len(items)} "
        f"(calibration {len(calibration)}, held-out {len(held_out)}); labeled items with "
        f"verdicts {len(labeled)}. Reference judge: `{reference or 'none'}`.",
        "",
    ]

    def section(title: str, ids: set[str], use_frozen: bool, *, is_held_out: bool = False) -> None:
        turn_evals: list[TurnEval] = []
        draft_evals: list[DraftEval] = []
        cascades: dict[str, CascadeResult | None] = {}
        for name, rows_by_key in sorted(scored.items()):
            unique = {i: r for i, r in per_item(rows_by_key).items() if i in ids}
            if not unique:
                continue
            thresholds = thresholds_from_run(run, name) if use_frozen else dict(DEFAULT_THRESHOLDS)
            turn_evals.append(
                evaluate_turns(
                    name,
                    list(unique.values()),
                    thresholds,
                    category_on_failed=version != "questions_v1",
                )
            )
            for drafter, rows in per_drafter(rows_by_key).items():
                in_split = [r for r in rows if r.item["item_id"] in ids]
                if in_split:
                    draft_evals.append(evaluate_drafts(name, drafter, in_split, thresholds, gates))
            if reference and name != reference and reference in scored:
                gate = (
                    ((frozen or {}).get("backends", {}).get(name, {}).get("gate"))
                    if use_frozen
                    else None
                )
                if use_frozen and gate is None:
                    continue  # Missing calibration evidence is not permission to fit held-out.
                expensive = {i: r for i, r in per_item(scored[reference]).items() if i in ids}
                result = cascade_against(unique, expensive, thresholds["failed"], gate)
                if result is not None:
                    cascades[name] = result
        lines.extend([f"## {title} ({len(ids)} items)", ""])
        if not turn_evals:
            lines.extend(["No labeled verdicts in this split.", ""])
            return
        lines.extend(["### Stage A (one row per item)", ""])
        lines.extend(_table(_TURN_HEADER, _turn_rows(turn_evals)))
        lines.extend(["", "### Stage B (one row per draft)", ""])
        lines.extend(_table(_DRAFT_HEADER, _draft_rows(draft_evals)))
        if current:
            lines.extend(
                [
                    "",
                    "Costs in Stage A are total paid per item across all its draft verdicts; "
                    "Stage B shows only its draft-question costs, a subset, not an additional "
                    "charge. Rule support is a paid question; aggregated claim gates and reused "
                    "Stage A answers add no cost. Unscored admission labels do not remove costs.",
                    "",
                    "### Exact-rule support",
                    "",
                    *_table(
                        ["backend", "drafter", "scored", "accuracy", "AUROC", "Brier", "ECE"],
                        [
                            [
                                b.name,
                                b.drafter or "—",
                                str(b.rule_supported.n if b.rule_supported else 0),
                                fmt(b.rule_supported.accuracy if b.rule_supported else None),
                                fmt(b.rule_supported.auroc if b.rule_supported else None),
                                fmt(b.rule_supported.brier if b.rule_supported else None),
                                fmt(b.rule_supported.ece if b.rule_supported else None),
                            ]
                            for b in draft_evals
                        ],
                    ),
                ]
            )
        lines.extend(
            [
                "",
                "### Write decisions vs `store_exact`"
                if current
                else "### Write decisions vs `worth_remembering` and not `enforced_by`",
                "",
                (
                    "Admission ground truth is the blinded answer to 'Would you store this "
                    "exact draft?' Turn-level `worth_remembering` measures opportunity only. "
                    "Missing, unclear, declined, or stale-hash draft labels are unscored, "
                    "not acceptable drafts or negatives. Precision uses only scored writes; "
                    "miss rate uses only drafts explicitly labeled store-worthy. Action counts "
                    "include unscored rows. A declined draft is always a discard."
                )
                if current
                else (
                    "A write counts as correct when the item is worth remembering and nothing else "
                    "(harness, project docs, tooling) already enforces that draft's lesson. "
                    "`writes enforced` counts writes that landed on an already-enforced lesson. "
                    "A declined draft is always a discard."
                ),
                "",
            ]
        )
        lines.extend(
            _table(
                [
                    "backend",
                    "drafter",
                    "thresholds",
                    "write",
                    "review",
                    "discard",
                    "write precision",
                    "miss rate",
                    "writes enforced",
                    *(["admission scored", "writes scored", "unscored"] if current else []),
                ],
                [
                    row
                    + (
                        [str(b.admission_scored), str(b.writes_scored), str(b.unscored)]
                        if current
                        else []
                    )
                    for row, b in zip(_write_rows(draft_evals), draft_evals, strict=True)
                ],
            )
        )
        lines.extend(_unanswered_note(draft_evals))
        if cascades:
            lines.extend(["", f"### Cascade to `{reference}` on `failed`", ""])
            rows = []
            for name, c in cascades.items():
                if c is None:
                    continue
                ref_eval = next((e for e in turn_evals if e.name == reference), None)
                ref_cost = ref_eval.mean_latency_ms * ref_eval.n if ref_eval else 0.0
                rows.append(
                    [
                        name,
                        f"{c.gate:.2f}",
                        fmt(c.accuracy),
                        f"{c.escalated_share:.0%}",
                        f"{c.cost / ref_cost:.0%}" if ref_cost else "n/a",
                    ]
                )
            lines.extend(
                _table(["backend", "gate", "accuracy", "escalated", "cost vs reference"], rows)
            )
        lines.append("")
        lines.extend(_slices(scored, ids, run, use_frozen))
        lines.extend(["### Success criteria", ""])
        lines.extend(
            _criteria(
                turn_evals,
                draft_evals,
                reference,
                cascades,
                held_out=is_held_out,
                questions_version=version,
            )
        )
        lines.append("")

    labels = store.labels()
    drafts = store.drafts_by_item()
    notes = label_notes(items, labels, drafts, questions_version=version)
    if notes:
        lines.extend(["## Labels on their own (all labeled items)", "", *notes])
    lines.extend(claim_notes(store, labels, drafts, scored, held_out=bool(frozen)))
    section("Calibration split", calibration, use_frozen=bool(frozen))
    by_caller: dict[str, set[str]] = {}
    for i in items:
        by_caller.setdefault(str(i.get("caller") or "human"), set()).add(i["item_id"])
    if len(by_caller) > 1:
        for caller, ids in sorted(by_caller.items()):
            section(
                f"Calibration split, sessions started by {caller}",
                ids & calibration,
                use_frozen=bool(frozen),
            )
    if frozen:
        section("Held-out split", held_out, use_frozen=True, is_held_out=True)
        when = f"Frozen at {frozen['frozen_at']}."
        part = frozen.get("partial")
        if part:
            when += (
                f" {part['unlabeled']} of {part['calibration']} calibration items were not "
                "fully labeled then (`--partial`)."
            )
        lines.extend(["## Frozen thresholds", "", when, ""])
        for name, spec in sorted(frozen.get("backends", {}).items()):
            t = spec["thresholds"]
            gate = spec.get("gate")
            shown = ", ".join(f"{g} ≥ {t[g]:.2f}" for g in frozen_gates(version) if g in t)
            lines.append(
                f"- `{name}`: {shown}, band {t.get('band', DEFAULT_THRESHOLDS['band']):.2f}"
                + (f", cascade gate {gate:.2f}" if gate is not None else "")
            )
        lines.append("")
    else:
        lines.extend(
            [
                "## Held-out split",
                "",
                "Locked. Run `mem judge report --freeze` once labeling is complete; thresholds "
                "are chosen on the calibration split and the held-out split is scored once.",
                "",
            ]
        )
    text = "\n".join(lines)
    store.write_report(text)
    return text
