"""Scores verdicts against labels, freezes thresholds on the calibration split, and writes
report.md. The held-out split is scored only after thresholds are frozen."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from .labeling import ENFORCED_ORDER
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
from .questions import DEFAULT_THRESHOLDS, derive_action, thresholds_from_run
from .store import JudgeError, RunStore, utc_now_iso

MARGIN_PTS = 0.03
CASCADE_SLACK = 0.01
CONFIDENCE_AUROC_TARGET = 0.75


@dataclass
class Scored:
    item: dict[str, Any]
    label: dict[str, Any]
    verdict: dict[str, Any]

    def p(self, qid: str) -> float | None:
        answer = self.verdict["answers"].get(qid)
        if not answer:
            return None
        return answer["probabilities"].get("yes")

    def confidence(self, qid: str) -> float:
        answer = self.verdict["answers"].get(qid)
        return float(answer["confidence"]) if answer else 0.0

    def value(self, qid: str) -> str | None:
        answer = self.verdict["answers"].get(qid)
        if not answer:
            return None
        value = answer["value"]
        return str(round(value)) if isinstance(value, float) else str(value)


@dataclass
class BackendEval:
    name: str
    n: int
    failed: BinaryMetrics | None
    supported: BinaryMetrics | None
    category_accuracy: float | None
    generalizable_accuracy: float | None
    paired: float | None
    mean_latency_ms: float
    total_cost_usd: float
    write_precision: float | None
    miss_rate: float | None
    writes_enforced: int = 0
    actions: dict[str, int] = field(default_factory=dict)
    thresholds: dict[str, float] = field(default_factory=dict)


def should_write(label: dict[str, Any]) -> bool:
    """The reference answer for a write: worth remembering, and not already enforced by the
    harness, the project's docs, or tooling. Labels from before `enforced_by` existed count as
    not enforced."""
    return bool(label.get("worth_remembering")) and label.get("enforced_by") in (None, "none")


def enforced(label: dict[str, Any]) -> bool:
    return label.get("enforced_by") not in (None, "none")


def load_scored(store: RunStore) -> tuple[list[dict[str, Any]], dict[str, dict[str, Scored]]]:
    items = store.read_jsonl(store.items_path)
    labels = {row["item_id"]: row for row in store.read_jsonl(store.labels_path)}
    by_item = {i["item_id"]: i for i in items}
    scored: dict[str, dict[str, Scored]] = {}
    if store.verdicts_dir.exists():
        for path in sorted(store.verdicts_dir.glob("*.jsonl")):
            for verdict in store.read_jsonl(path):
                item = by_item.get(verdict["item_id"])
                label = labels.get(verdict["item_id"])
                if item is None or label is None:
                    continue
                if verdict.get("projection_hash") != item.get("projection_hash"):
                    continue  # judged a projection the run no longer has
                scored.setdefault(verdict["backend"], {})[item["item_id"]] = Scored(
                    item, label, verdict
                )
    return items, scored


def _binary(rows: Sequence[Scored], qid: str, label_key: str, threshold: float):
    probs, confs, labels = [], [], []
    for r in rows:
        p = r.p(qid)
        y = r.label.get(label_key)
        if p is None or y is None:
            continue
        probs.append(p)
        confs.append(r.confidence(qid))
        labels.append(bool(y))
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


def evaluate_backend(
    name: str, rows: Sequence[Scored], thresholds: dict[str, float]
) -> BackendEval:
    cats = [(r.value("category"), r.label["category"]) for r in rows if r.value("category")]
    gens = [
        (r.value("generalizable"), str(r.label["generalizable"]))
        for r in rows
        if r.value("generalizable") is not None and r.label.get("generalizable") is not None
    ]
    actions: dict[str, int] = {"write_lesson": 0, "review": 0, "discard": 0}
    writes_worth = writes = worth = missed = writes_enforced = 0
    for r in rows:
        p_true = {k: v for k in ("failed", "recurrence", "supported") if (v := r.p(k)) is not None}
        action = derive_action(p_true, thresholds).action
        actions[action] += 1
        is_worth = should_write(r.label)
        worth += is_worth
        if action == "write_lesson":
            writes += 1
            writes_worth += is_worth
            writes_enforced += enforced(r.label)
        elif is_worth:
            missed += 1
    return BackendEval(
        name=name,
        n=len(rows),
        failed=_binary(rows, "failed", "failed", thresholds["failed"]),
        supported=_binary(rows, "supported", "supported", thresholds["supported"]),
        category_accuracy=accuracy([c for c, _ in cats], [t for _, t in cats]),
        generalizable_accuracy=accuracy([g for g, _ in gens], [t for _, t in gens]),
        paired=paired_discrimination(_pairs(rows)),
        mean_latency_ms=sum(r.verdict["latency_ms"] for r in rows) / len(rows) if rows else 0.0,
        total_cost_usd=sum(r.verdict.get("cost_usd", 0.0) for r in rows),
        write_precision=writes_worth / writes if writes else None,
        miss_rate=missed / worth if worth else None,
        writes_enforced=writes_enforced,
        actions=actions,
        thresholds=dict(thresholds),
    )


def reference_backend(scored: dict[str, dict[str, Scored]]) -> str | None:
    llms = [name for name in scored if name.startswith("llm:")]
    return max(llms, key=lambda n: len(scored[n])) if llms else None


def cascade_against(
    cheap: dict[str, Scored], expensive: dict[str, Scored], threshold: float, gate: float | None
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


def freeze_thresholds(
    scored: dict[str, dict[str, Scored]], calibration_ids: set[str]
) -> dict[str, Any]:
    """Per backend: the accuracy-maximising cut on failed and supported, plus the cascade gate
    against the reference judge, all from calibration items only."""
    reference = reference_backend(scored)
    backends: dict[str, Any] = {}
    for name, rows_by_id in scored.items():
        rows = [r for i, r in rows_by_id.items() if i in calibration_ids]
        thresholds = dict(DEFAULT_THRESHOLDS)
        for qid in ("failed", "supported"):
            probs = [
                p for r in rows if (p := r.p(qid)) is not None and r.label.get(qid) is not None
            ]
            labels = [
                bool(r.label[qid])
                for r in rows
                if r.p(qid) is not None and r.label.get(qid) is not None
            ]
            if probs:
                thresholds[qid] = round(best_threshold(probs, labels)[0], 4)
        gate = None
        if reference and name != reference:
            cheap = {i: r for i, r in rows_by_id.items() if i in calibration_ids}
            expensive = {i: r for i, r in scored[reference].items() if i in calibration_ids}
            result = cascade_against(cheap, expensive, thresholds["failed"], None)
            gate = result.gate if result else None
        backends[name] = {"thresholds": thresholds, "gate": gate}
    return {"frozen_at": utc_now_iso(), "reference": reference, "backends": backends}


# Markdown


def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return out


def _metrics_rows(evals: Sequence[BackendEval]) -> list[list[str]]:
    rows = []
    for e in evals:
        f, s = e.failed, e.supported
        rows.append(
            [
                e.name,
                str(e.n),
                fmt(f.accuracy if f else None),
                fmt(f.auroc if f else None),
                fmt(f.brier if f else None),
                fmt(f.ece if f else None),
                fmt(f.confidence_error_auroc if f else None),
                fmt(s.accuracy if s else None),
                fmt(s.auroc if s else None),
                fmt(s.confidence_error_auroc if s else None),
                fmt(e.category_accuracy),
                fmt(e.generalizable_accuracy),
                fmt(e.paired),
                f"{e.mean_latency_ms / 1000:.2f}",
                f"{e.total_cost_usd:.4f}",
            ]
        )
    return rows


_HEADER = [
    "backend",
    "n",
    "failed acc",
    "failed AUROC",
    "Brier",
    "ECE",
    "conf→err AUROC",
    "supported acc",
    "supported AUROC",
    "conf→err AUROC (B)",
    "category acc",
    "generalizable acc",
    "paired",
    "mean s/item",
    "USD",
]


def _write_rows(evals: Sequence[BackendEval]) -> list[list[str]]:
    return [
        [
            e.name,
            fmt(e.thresholds.get("failed"), 2),
            fmt(e.thresholds.get("supported"), 2),
            str(e.actions.get("write_lesson", 0)),
            str(e.actions.get("review", 0)),
            str(e.actions.get("discard", 0)),
            fmt(e.write_precision),
            fmt(e.miss_rate),
            str(e.writes_enforced),
        ]
        for e in evals
    ]


def label_notes(items: Sequence[dict[str, Any]], labels: dict[str, dict[str, Any]]) -> list[str]:
    """What the human-only labels say on their own: how often a drafted lesson is already
    enforced elsewhere (by who started the session), and how often a tool error was the point."""
    lines: list[str] = []
    by_caller: dict[str, dict[str, int]] = {}
    err_total = err_intended = err_failed = 0
    for item in items:
        label = labels.get(item["item_id"])
        if label is None:
            continue
        kind = str(item.get("caller") or "human")
        value = label.get("enforced_by")
        if value is not None:
            counts = by_caller.setdefault(kind, dict.fromkeys(ENFORCED_ORDER, 0))
            counts[str(value)] = counts.get(str(value), 0) + 1
        if (
            item.get("facts", {}).get("tool_error_count")
            and label.get("error_intended") is not None
        ):
            err_total += 1
            err_intended += bool(label["error_intended"])
            err_failed += bool(label.get("failed"))
    if by_caller:
        lines.extend(["### Drafted lessons already enforced elsewhere (`enforced_by`)", ""])
        rows = []
        for kind, counts in sorted(by_caller.items()):
            n = sum(counts.values())
            rows.append(
                [kind, str(n)]
                + [f"{counts.get(k, 0)} ({counts.get(k, 0) / n:.0%})" for k in ENFORCED_ORDER]
            )
        lines.extend(_table(["session started by", "drafts labeled", *ENFORCED_ORDER], rows))
        lines.append("")
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


def _criteria(
    evals: Sequence[BackendEval], reference: str | None, cascades: dict[str, CascadeResult | None]
) -> list[str]:
    lines = []
    ref = next((e for e in evals if e.name == reference), None)
    if ref is None or ref.failed is None:
        return ["No reference `llm:` backend with labeled verdicts; criteria need one."]
    ref_acc = ref.failed.accuracy or 0.0
    ref_sup = (ref.supported.accuracy if ref.supported else None) or 0.0
    for e in evals:
        if e.name == reference or e.failed is None:
            continue
        acc = e.failed.accuracy or 0.0
        sup = (e.supported.accuracy if e.supported else None) or 0.0
        cheap = e.mean_latency_ms <= 0.1 * ref.mean_latency_ms if ref.mean_latency_ms else False
        c1 = acc >= ref_acc - MARGIN_PTS and sup >= ref_sup - MARGIN_PTS and cheap
        c2 = (e.failed.confidence_error_auroc or 0.0) >= CONFIDENCE_AUROC_TARGET
        casc = cascades.get(e.name)
        c3 = bool(
            casc
            and casc.accuracy >= ref_acc - CASCADE_SLACK
            and casc.cost <= 0.5 * ref.mean_latency_ms * e.n
        )
        lines.append(
            f"- `{e.name}`: (1) within {MARGIN_PTS * 100:.0f} pts at ≤10% latency: "
            f"{'yes' if c1 else 'no'} (failed {acc:.3f} vs {ref_acc:.3f}, supported {sup:.3f} "
            f"vs {ref_sup:.3f}, {e.mean_latency_ms / 1000:.2f}s vs "
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
                f"{'yes' if c4 else 'no'}"
            )
    return lines


def build_report(store: RunStore, *, freeze: bool = False) -> str:
    run = store.read_run()
    items, scored = load_scored(store)
    if not scored:
        raise JudgeError("no labeled verdicts yet; run `mem judge decide` and `mem judge label`")
    calibration = {i["item_id"] for i in items if i.get("split") == "calibration"}
    held_out = {i["item_id"] for i in items if i.get("split") == "held_out"}
    if freeze:
        if run.get("frozen"):
            raise JudgeError("thresholds are already frozen for this run")
        run = store.update_run(frozen=freeze_thresholds(scored, calibration))
    frozen = run.get("frozen")
    reference = (frozen or {}).get("reference") or reference_backend(scored)
    labeled = {i for rows in scored.values() for i in rows}

    lines = [
        f"# Judge report: {store.run_id}",
        "",
        f"Generated {utc_now_iso()}. Versions: {run.get('triggers_version')}, "
        f"{run.get('projection_version')}, {run.get('questions_version')}, "
        f"drafter {run.get('drafter', 'none')}. Items {len(items)} "
        f"(calibration {len(calibration)}, held-out {len(held_out)}); labeled items with "
        f"verdicts {len(labeled)}. Reference judge: `{reference or 'none'}`.",
        "",
    ]

    def section(title: str, ids: set[str], use_frozen: bool) -> None:
        evals = []
        cascades: dict[str, CascadeResult | None] = {}
        for name, rows_by_id in sorted(scored.items()):
            rows = [r for i, r in rows_by_id.items() if i in ids]
            if not rows:
                continue
            thresholds = thresholds_from_run(run, name) if use_frozen else dict(DEFAULT_THRESHOLDS)
            evals.append(evaluate_backend(name, rows, thresholds))
            if reference and name != reference and reference in scored:
                gate = (
                    ((frozen or {}).get("backends", {}).get(name, {}).get("gate"))
                    if use_frozen
                    else None
                )
                expensive = {i: r for i, r in scored[reference].items() if i in ids}
                cascades[name] = cascade_against(
                    {i: r for i, r in rows_by_id.items() if i in ids},
                    expensive,
                    thresholds["failed"],
                    gate,
                )
        lines.extend([f"## {title} ({len(ids)} items)", ""])
        if not evals:
            lines.extend(["No labeled verdicts in this split.", ""])
            return
        lines.extend(_table(_HEADER, _metrics_rows(evals)))
        lines.extend(
            [
                "",
                "### Write decisions vs `worth_remembering` and not `enforced_by`",
                "",
                "A write counts as correct when the label says worth remembering and nothing else "
                "(harness, project docs, tooling) already enforces the lesson. "
                "`writes enforced` counts writes that landed on an already-enforced lesson.",
                "",
            ]
        )
        lines.extend(
            _table(
                [
                    "backend",
                    "τ failed",
                    "τ supported",
                    "write",
                    "review",
                    "discard",
                    "write precision",
                    "miss rate",
                    "writes enforced",
                ],
                _write_rows(evals),
            )
        )
        if cascades:
            lines.extend(["", f"### Cascade to `{reference}` on `failed`", ""])
            rows = []
            for name, c in cascades.items():
                if c is None:
                    continue
                ref_eval = next((e for e in evals if e.name == reference), None)
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
        lines.extend(["", "### Success criteria", ""])
        lines.extend(_criteria(evals, reference, cascades))
        lines.append("")

    labels = {row["item_id"]: row for row in store.read_jsonl(store.labels_path)}
    notes = label_notes(items, labels)
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
        section("Held-out split", held_out, use_frozen=True)
        lines.extend(["## Frozen thresholds", "", f"Frozen at {frozen['frozen_at']}.", ""])
        for name, spec in sorted(frozen.get("backends", {}).items()):
            t = spec["thresholds"]
            gate = spec.get("gate")
            lines.append(
                f"- `{name}`: failed ≥ {t['failed']:.2f}, supported ≥ {t['supported']:.2f}, "
                f"recurrence ≥ {t['recurrence']:.2f}, band {t['band']:.2f}"
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
