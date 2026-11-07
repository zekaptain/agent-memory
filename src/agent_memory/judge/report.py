"""Scores verdicts against labels, freezes thresholds on the calibration split, and writes
report.md. The held-out split is scored only after thresholds are frozen.

Stage A questions are scored once per item; Stage B questions and the write decision once per
(item, draft), per drafter. Runs built under `questions_v1` (one draft per item, `supported`
instead of `grounded`, the v1 write rule) are read through the same code: the aliases live in
`Scored` and `draft_label`.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .labeling import ENFORCED_ORDER, LESSON_ORDER
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
from .questions import (
    DEFAULT_THRESHOLDS,
    STAGE_A,
    derive_action,
    gate_probability,
    thresholds_from_run,
    write_gates,
)
from .store import JudgeError, RunStore, utc_now_iso
from .triggers import CLOSED_BY, closed_by_of

MARGIN_PTS = 0.03
CASCADE_SLACK = 0.01
CONFIDENCE_AUROC_TARGET = 0.75
WRITE_PRECISION_TARGET = 0.7
MISS_RATE_TARGET = 0.5
FAITHFUL_TARGET = 0.9
_ALIAS = {"grounded": "supported", "supported": "grounded"}
STAGE_A_IDS = {q.id for q in STAGE_A}


def draft_label(label: Mapping[str, Any], verdict: Mapping[str, Any]) -> dict[str, Any]:
    """The Stage B labels that apply to this verdict's draft. v2 rows hold them per drafter and
    only count when the draft text is the one labeled; v1 rows hold them per item."""
    drafts = label.get("drafts")
    if isinstance(drafts, dict):
        entry = drafts.get(str(verdict.get("drafter") or "")) or {}
        labeled_hash = entry.get("draft_hash")
        if entry and labeled_hash != verdict.get("draft_hash"):
            return {"declined": entry.get("declined")}
        return {
            "grounded": entry.get("grounded"),
            "faithful": entry.get("faithful"),
            "generalizable": entry.get("generalizable"),
            "enforced_by": entry.get("enforced_by"),
            "declined": entry.get("declined"),
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
    item: dict[str, Any]
    label: dict[str, Any]
    verdict: dict[str, Any]
    draft: dict[str, Any] = field(default_factory=dict)

    @property
    def drafter(self) -> str | None:
        return self.verdict.get("drafter")

    def answer(self, qid: str) -> dict[str, Any] | None:
        answers = self.verdict["answers"]
        return answers.get(qid) or answers.get(_ALIAS.get(qid, ""))

    def p(self, qid: str) -> float | None:
        answer = self.answer(qid)
        return answer["probabilities"].get("yes") if answer else None

    def gate_p(self, gate: str) -> float | None:
        answer = self.answer(gate)
        return gate_probability(gate, answer["probabilities"]) if answer else None

    def confidence(self, qid: str) -> float:
        answer = self.answer(qid)
        return float(answer["confidence"]) if answer else 0.0

    def value(self, qid: str) -> str | None:
        answer = self.answer(qid)
        if not answer:
            return None
        value = answer["value"]
        return str(round(value)) if isinstance(value, float) else str(value)

    def gate_probabilities(self, gates: Sequence[str]) -> dict[str, float]:
        return {g: p for g in gates if (p := self.gate_p(g)) is not None}

    def stage_b_latency(self) -> float:
        return sum(
            float(a.get("latency_ms", 0.0))
            for qid, a in self.verdict["answers"].items()
            if qid not in STAGE_A_IDS
        )


def gate_truth(gate: str, row: Scored) -> bool | None:
    """The label a gate is scored against: the item's for failed and incomplete, the draft's
    for the rest. generalizable clears at level 1; enforced clears on none."""
    if gate in ("failed", "incomplete"):
        value = row.label.get(gate)
    elif gate in ("grounded", "supported", "faithful"):
        value = row.draft.get("grounded" if gate == "supported" else gate)
    elif gate == "generalizable":
        level = row.draft.get("generalizable")
        value = None if level is None else int(level) >= 1
    elif gate == "enforced":
        enforced_by = row.draft.get("enforced_by")
        value = None if enforced_by is None else enforced_by == "none"
    else:
        return None
    return None if value is None else bool(value)


def should_write(label: Mapping[str, Any], draft: Mapping[str, Any] | None = None) -> bool:
    """The reference answer for a write: worth remembering, and not already enforced by the
    harness, the project's docs, or tooling. Missing `enforced_by` counts as not enforced."""
    enforced_by = (draft if draft is not None else label).get("enforced_by")
    return bool(label.get("worth_remembering")) and enforced_by in (None, "none")


def enforced(label: Mapping[str, Any], draft: Mapping[str, Any] | None = None) -> bool:
    return (draft if draft is not None else label).get("enforced_by") not in (None, "none")


Key = tuple[str, str | None]


def load_scored(store: RunStore) -> tuple[list[dict[str, Any]], dict[str, dict[Key, Scored]]]:
    """Items, and per backend every labeled verdict row keyed by (item, drafter)."""
    items = store.read_jsonl(store.items_path)
    labels = {row["item_id"]: row for row in store.read_jsonl(store.labels_path)}
    by_item = {i["item_id"]: i for i in items}
    scored: dict[str, dict[Key, Scored]] = {}
    if store.verdicts_dir.exists():
        for path in sorted(store.verdicts_dir.glob("*.jsonl")):
            for verdict in store.read_jsonl(path):
                item = by_item.get(verdict["item_id"])
                label = labels.get(verdict["item_id"])
                if item is None or label is None:
                    continue
                if verdict.get("projection_hash") != item.get("projection_hash"):
                    continue  # judged a projection the run no longer has
                key = (item["item_id"], verdict.get("drafter"))
                scored.setdefault(verdict["backend"], {})[key] = Scored(
                    item, label, verdict, draft_label(label, verdict)
                )
    return items, scored


def per_item(rows: Mapping[Key, Scored]) -> dict[str, Scored]:
    """One row per item for Stage A, which every row of an item carries alike; the latency
    attributed to the item is what all its rows paid together."""
    out: dict[str, Scored] = {}
    paid: dict[str, float] = {}
    for (item_id, _drafter), row in sorted(rows.items(), key=lambda kv: (kv[0][0], kv[0][1] or "")):
        paid[item_id] = paid.get(item_id, 0.0) + float(row.verdict.get("latency_ms", 0.0))
        out.setdefault(item_id, row)
    for item_id, row in out.items():
        out[item_id] = Scored(
            row.item, row.label, {**row.verdict, "latency_ms": paid[item_id]}, row.draft
        )
    return out


def per_drafter(rows: Mapping[Key, Scored]) -> dict[str | None, list[Scored]]:
    out: dict[str | None, list[Scored]] = {}
    for (_item_id, drafter), row in sorted(rows.items(), key=lambda kv: (kv[0][1] or "", kv[0][0])):
        out.setdefault(drafter, []).append(row)
    return out


@dataclass
class StageAEval:
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
class StageBEval:
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


def _binary(rows: Sequence[Scored], gate: str, threshold: float) -> BinaryMetrics | None:
    probs, confs, labels = [], [], []
    for r in rows:
        p = r.gate_p(gate)
        y = gate_truth(gate, r)
        if p is None or y is None:
            continue
        probs.append(p)
        confs.append(r.confidence(gate))
        labels.append(y)
    return binary_metrics(probs, confs, labels, threshold) if probs else None


def _pairs(rows: Sequence[Scored]) -> list[tuple[float, float]]:
    """Every (failed, clean) pair of labeled turns from the same project."""
    by_project: dict[str, tuple[list[float], list[float]]] = {}
    for r in rows:
        p = r.p("failed")
        y = r.label.get("failed")
        if p is None or y is None:
            continue
        slot = by_project.setdefault(str(r.item.get("project")), ([], []))
        slot[0 if y else 1].append(p)
    return [(f, c) for failed, clean in by_project.values() for f in failed for c in clean]


def evaluate_stage_a(
    name: str, rows: Sequence[Scored], thresholds: Mapping[str, float], *, category_on_failed: bool
) -> StageAEval:
    cats = [
        (r.value("category"), r.label["category"])
        for r in rows
        if r.value("category")
        and r.label.get("category") is not None
        and (not category_on_failed or r.label.get("failed") is True)
    ]
    failed_labels = [bool(r.label["failed"]) for r in rows if r.label.get("failed") is not None]
    majority = (
        max(sum(failed_labels), len(failed_labels) - sum(failed_labels)) / len(failed_labels)
        if failed_labels
        else None
    )
    return StageAEval(
        name=name,
        n=len(rows),
        failed=_binary(rows, "failed", thresholds["failed"]),
        incomplete=_binary(rows, "incomplete", 0.5),
        category_accuracy=accuracy([c for c, _ in cats], [t for _, t in cats]),
        paired=paired_discrimination(_pairs(rows)),
        mean_latency_ms=sum(r.verdict["latency_ms"] for r in rows) / len(rows) if rows else 0.0,
        total_cost_usd=sum(r.verdict.get("cost_usd", 0.0) for r in rows),
        majority_failed=majority,
    )


def evaluate_stage_b(
    name: str,
    drafter: str | None,
    rows: Sequence[Scored],
    thresholds: Mapping[str, float],
    gates: Sequence[str],
) -> StageBEval:
    gens = [
        (r.value("generalizable"), str(r.draft["generalizable"]))
        for r in rows
        if r.value("generalizable") is not None and r.draft.get("generalizable") is not None
    ]
    enf = [
        (r.value("enforced"), str(r.draft["enforced_by"]))
        for r in rows
        if r.value("enforced") is not None and r.draft.get("enforced_by") is not None
    ]
    actions: dict[str, int] = {"write_lesson": 0, "review": 0, "discard": 0}
    writes_worth = writes = worth = missed = writes_enforced = 0
    for r in rows:
        action = derive_action(
            r.gate_probabilities(gates),
            thresholds,
            gates=gates,
            declined=bool(r.verdict.get("declined")),
        ).action
        actions[action] += 1
        is_worth = should_write(r.label, r.draft)
        worth += is_worth
        if action == "write_lesson":
            writes += 1
            writes_worth += is_worth
            writes_enforced += enforced(r.label, r.draft)
        elif is_worth:
            missed += 1
    return StageBEval(
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
        mean_latency_ms=sum(r.stage_b_latency() for r in rows) / len(rows) if rows else 0.0,
        total_cost_usd=sum(r.verdict.get("cost_usd", 0.0) for r in rows),
        write_precision=writes_worth / writes if writes else None,
        miss_rate=missed / worth if worth else None,
        writes_enforced=writes_enforced,
        actions=actions,
        thresholds=dict(thresholds),
        gates=tuple(gates),
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
        pc, pe = cheap[i].p("failed"), expensive[i].p("failed")
        if pc is None or pe is None:
            continue
        confs.append(cheap[i].confidence("failed"))
        cheap_ok.append((pc >= threshold) == y)
        exp_ok.append((pe >= 0.5) == y)
        cheap_cost.append(cheap[i].verdict["latency_ms"])
        exp_cost.append(expensive[i].verdict["latency_ms"])
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
                if (p := r.gate_p(gate)) is not None and (y := gate_truth(gate, r)) is not None
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


_STAGE_A_HEADER = [
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


def _stage_a_rows(evals: Sequence[StageAEval]) -> list[list[str]]:
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


_STAGE_B_HEADER = [
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


def _stage_b_rows(evals: Sequence[StageBEval]) -> list[list[str]]:
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


def _write_rows(evals: Sequence[StageBEval]) -> list[list[str]]:
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


def _rate(count: int, total: int) -> str:
    return f"{count} ({count / total:.0%})" if total else "0"


def label_notes(
    items: Sequence[dict[str, Any]],
    labels: Mapping[str, dict[str, Any]],
    drafts: Mapping[str, Sequence[Mapping[str, Any]]],
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
        lines.extend(drafter_notes(labels, drafts))
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
    labels: Mapping[str, dict[str, Any]], drafts: Mapping[str, Sequence[Mapping[str, Any]]]
) -> list[str]:
    """Per drafter: how often it declined and whether it was right to, how the reviewer rated its
    lessons, which arm was preferred, and what it cost. Criterion 5 is read from this table."""
    names = sorted({str(d.get("drafter") or "") for rows in drafts.values() for d in rows})
    if not names:
        return []
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
            ),
            0.0,
        )
        for n in names
    }
    same = neither = compared = 0
    for item_id, rows in drafts.items():
        label = labels.get(item_id)
        for row in rows:
            s = stats[str(row.get("drafter") or "")]
            s["drafts"] += 1
            s["latency"] += float(row.get("latency_ms") or 0.0)
            s["cost"] += float(row.get("cost_usd") or 0.0)
            declined = row.get("lesson") is None
            s["declined"] += declined
            if label is None:
                continue
            if declined:
                s["declined_right"] += not label.get("worth_remembering")
                continue
            entry = (label.get("drafts") or {}).get(str(row.get("drafter") or "")) or {}
            if entry.get("grounded") is None:
                continue
            s["labeled"] += 1
            s["grounded"] += bool(entry.get("grounded"))
            s["faithful"] += bool(entry.get("faithful"))
            s["reusable"] += int(entry.get("generalizable") or 0) >= 1
            s["unenforced"] += entry.get("enforced_by") == "none"
        if label is not None and len(rows) >= 2 and label.get("preferred") is not None:
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
        "`declined right` is the share of declined drafts on items labeled not worth "
        "remembering. Rates in the middle columns are over labeled, non-declined drafts. "
        f"`preferred` is over the {compared} item(s) where both drafts were compared"
        + (f" ({same} same, {neither} neither)" if compared else "")
        + ".",
        "",
    ]
    rows = []
    for name in names:
        s = stats[name]
        drafts_n, declined, labeled = int(s["drafts"]), int(s["declined"]), int(s["labeled"])
        rows.append(
            [
                name,
                str(drafts_n),
                _rate(declined, drafts_n),
                fmt(s["declined_right"] / declined if declined else None, 2),
                str(labeled),
                fmt(s["grounded"] / labeled if labeled else None, 2),
                fmt(s["faithful"] / labeled if labeled else None, 2),
                fmt(s["reusable"] / labeled if labeled else None, 2),
                fmt(s["unenforced"] / labeled if labeled else None, 2),
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
            labeled = int(s["labeled"])
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


def _slices(
    scored: Mapping[str, Mapping[Key, Scored]],
    ids: set[str],
    run: Mapping[str, Any],
    use_frozen: bool,
) -> list[str]:
    """Stage A accuracy by how the turn closed and by the incomplete label."""
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
    evals: Sequence[StageAEval],
    stage_b: Sequence[StageBEval],
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
    ref_b = [e for e in stage_b if e.name == reference]
    ref_grounded = (
        max((e.grounded.accuracy or 0.0) for e in ref_b if e.grounded)
        if any(e.grounded for e in ref_b)
        else 0.0
    )
    for e in evals:
        if e.name == reference or e.failed is None:
            continue
        acc = e.failed.accuracy or 0.0
        mine = [b for b in stage_b if b.name == e.name]
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
            f"({fmt(e.failed.confidence_error_auroc)}); "
            f"(3) cascade ≥ ref-1 pt at ≤50% cost: {'yes' if c3 else 'no'}"
            + (
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
                f"- v2 write rule with `{reference}` on `{b.drafter or '—'}` (criterion 6: "
                f"precision ≥ {WRITE_PRECISION_TARGET}, miss rate ≤ {MISS_RATE_TARGET}): "
                f"{'yes' if ok else 'no'} (precision {fmt(b.write_precision)}, miss rate "
                f"{fmt(b.miss_rate)}, {b.actions.get('write_lesson', 0)} writes)"
            )
    return lines


def build_report(store: RunStore, *, freeze: bool = False) -> str:
    run = store.read_run()
    version = run.get("questions_version")
    gates = write_gates(version)
    items, scored = load_scored(store)
    if not scored:
        raise JudgeError("no labeled verdicts yet; run `mem judge decide` and `mem judge label`")
    calibration = {i["item_id"] for i in items if i.get("split") == "calibration"}
    held_out = {i["item_id"] for i in items if i.get("split") == "held_out"}
    if freeze:
        if run.get("frozen"):
            raise JudgeError("thresholds are already frozen for this run")
        run = store.update_run(frozen=freeze_thresholds(scored, calibration, version))
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
        stage_a: list[StageAEval] = []
        stage_b: list[StageBEval] = []
        cascades: dict[str, CascadeResult | None] = {}
        for name, rows_by_key in sorted(scored.items()):
            unique = {i: r for i, r in per_item(rows_by_key).items() if i in ids}
            if not unique:
                continue
            thresholds = thresholds_from_run(run, name) if use_frozen else dict(DEFAULT_THRESHOLDS)
            stage_a.append(
                evaluate_stage_a(
                    name,
                    list(unique.values()),
                    thresholds,
                    category_on_failed=version != "questions_v1",
                )
            )
            for drafter, rows in per_drafter(rows_by_key).items():
                in_split = [r for r in rows if r.item["item_id"] in ids]
                if in_split:
                    stage_b.append(evaluate_stage_b(name, drafter, in_split, thresholds, gates))
            if reference and name != reference and reference in scored:
                gate = (
                    ((frozen or {}).get("backends", {}).get(name, {}).get("gate"))
                    if use_frozen
                    else None
                )
                expensive = {i: r for i, r in per_item(scored[reference]).items() if i in ids}
                cascades[name] = cascade_against(unique, expensive, thresholds["failed"], gate)
        lines.extend([f"## {title} ({len(ids)} items)", ""])
        if not stage_a:
            lines.extend(["No labeled verdicts in this split.", ""])
            return
        lines.extend(["### Stage A (one row per item)", ""])
        lines.extend(_table(_STAGE_A_HEADER, _stage_a_rows(stage_a)))
        lines.extend(["", "### Stage B (one row per draft)", ""])
        lines.extend(_table(_STAGE_B_HEADER, _stage_b_rows(stage_b)))
        lines.extend(
            [
                "",
                "### Write decisions vs `worth_remembering` and not `enforced_by`",
                "",
                "A write counts as correct when the item is worth remembering and nothing else "
                "(harness, project docs, tooling) already enforces that draft's lesson. "
                "`writes enforced` counts writes that landed on an already-enforced lesson. "
                "A declined draft is always a discard.",
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
                ],
                _write_rows(stage_b),
            )
        )
        if cascades:
            lines.extend(["", f"### Cascade to `{reference}` on `failed`", ""])
            rows = []
            for name, c in cascades.items():
                if c is None:
                    continue
                ref_eval = next((e for e in stage_a if e.name == reference), None)
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
                stage_a,
                stage_b,
                reference,
                cascades,
                held_out=is_held_out,
                questions_version=version,
            )
        )
        lines.append("")

    labels = {row["item_id"]: row for row in store.read_jsonl(store.labels_path)}
    notes = label_notes(items, labels, store.read_drafts())
    if notes:
        lines.extend(["## Labels on their own (all labeled items)", "", *notes])
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
        lines.extend(["## Frozen thresholds", "", f"Frozen at {frozen['frozen_at']}.", ""])
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
    store.report_path.write_text(text, encoding="utf-8")
    return text
