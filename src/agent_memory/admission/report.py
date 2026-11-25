"""Read-only admission reports: daily exact-value checks and human retirement rates."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..config import Config
from ..judge.drafter import draft_hash
from ..judge.store import JudgeError, RunStore
from . import AdmissionError
from .exact_values import check_draft
from .store import AdmissionStore, read_jsonl


def _valid_exact(value: Any) -> dict[str, Any] | None:
    if isinstance(value, Mapping) and type(value.get("ok")) is bool:
        return dict(value)
    return None


def _exact_for_day_draft(day: RunStore, draft: Mapping[str, Any]) -> dict[str, Any] | None:
    existing = _valid_exact(draft.get("exact_values"))
    if existing is not None:
        return existing
    iid = draft.get("item_id")
    if not isinstance(iid, str) or not iid:
        return None
    try:
        projection = day.read_projection(iid)
    except JudgeError:
        return None
    return check_draft(draft, projection)


def day_report(day: RunStore) -> dict[str, Any]:
    """Summarize one admission day, including lessons whose rule carries missing values."""
    run = day.read_run()
    items = read_jsonl(day.items_path)
    drafts = {row["item_id"]: row for row in read_jsonl(day.drafts_path)}
    lessons = declined = checked = passed = failed = redacted_cited = 0
    for draft in drafts.values():
        if draft.get("declined") or not draft.get("rule"):
            declined += 1
            continue
        lessons += 1
        exact = _exact_for_day_draft(day, draft)
        if exact is None:
            continue
        checked += 1
        if exact["ok"]:
            passed += 1
        else:
            failed += 1
        if exact.get("redacted_cited"):
            redacted_cited += 1
    return {
        "day": day.run_id,
        "mode": run.get("mode"),
        "items": len(items),
        "lessons": lessons,
        "declined": declined,
        "exact_values": {
            "checked": checked,
            "passed": passed,
            "failed": failed,
            "redacted_cited": redacted_cited,
            "missing_rate": failed / checked if checked else None,
        },
    }


def _day_exact_index(store: AdmissionStore) -> dict[tuple[str, str], dict[str, Any]]:
    index: dict[tuple[str, str], dict[str, Any]] = {}
    days_dir = store.root / "days"
    if not days_dir.exists():
        return index
    for path in sorted(days_dir.glob("*/run.json")):
        day = store.day(path.parent.name)
        for draft in read_jsonl(day.drafts_path):
            if draft.get("declined") or not draft.get("rule"):
                continue
            iid = draft.get("item_id")
            if not isinstance(iid, str) or not iid:
                continue
            exact = _exact_for_day_draft(day, draft)
            if exact is None:
                continue
            digest = draft_hash(draft.get("lesson")) or ""
            index[(iid, digest)] = exact
            index.setdefault((iid, ""), exact)
    return index


def _bucket() -> dict[str, Any]:
    return {"n": 0, "retired": 0, "approved": 0, "retired_rate": None}


def _finalize_bucket(bucket: dict[str, Any]) -> dict[str, Any]:
    n = bucket["n"]
    return {**bucket, "retired_rate": bucket["retired"] / n if n else None}


def examples_report(store: AdmissionStore) -> dict[str, Any]:
    """Over existing human-labeled examples, report how often failed-check lessons were retired."""
    effective = store.read_examples()
    reviewed_items = {
        str(row["item_id"]) for row in effective if row.get("memory_id") and row.get("item_id")
    }
    unique: dict[tuple[str, str], dict[str, Any]] = {}
    for row in effective:
        if not row.get("memory_id") and str(row.get("item_id") or "") in reviewed_items:
            continue
        key = (
            ("memory", str(row["memory_id"]))
            if row.get("memory_id")
            else ("exp", str(row.get("source") or row.get("id")))
        )
        previous = unique.get(key)
        if previous is None or (
            _valid_exact(previous.get("exact_values")) is None
            and _valid_exact(row.get("exact_values")) is not None
        ):
            unique[key] = row

    day_index: dict[tuple[str, str], dict[str, Any]] | None = None
    total = checked = unchecked = redacted_cited = 0
    failed = _bucket()
    passed = _bucket()
    for key in sorted(unique):
        row = unique[key]
        label = row.get("label")
        if type(label) is not bool:
            continue
        total += 1
        exact = _valid_exact(row.get("exact_values"))
        if exact is None and row.get("item_id"):
            if day_index is None:
                day_index = _day_exact_index(store)
            iid = str(row["item_id"])
            draft_hash = str(row.get("draft_hash") or "")
            exact = day_index.get((iid, draft_hash)) or day_index.get((iid, ""))
        if exact is None:
            unchecked += 1
            continue
        checked += 1
        if exact.get("redacted_cited"):
            redacted_cited += 1
        bucket = passed if exact["ok"] else failed
        bucket["n"] += 1
        if label:
            bucket["approved"] += 1
        else:
            bucket["retired"] += 1

    return {
        "total": total,
        "checked": checked,
        "unchecked": unchecked,
        "redacted_cited": redacted_cited,
        "failed": _finalize_bucket(failed),
        "passed": _finalize_bucket(passed),
    }


def admission_report(cfg: Config, *, day: str | None = None) -> dict[str, Any]:
    """Build the read-only admission report for one day or all days, plus example rates."""
    store = AdmissionStore(cfg)
    if day is not None:
        target = store.day(day)
        if not target.exists():
            raise AdmissionError(f"no admission day {day}")
        days = [day_report(target)]
    else:
        days_dir = store.root / "days"
        paths = sorted(days_dir.glob("*/run.json")) if days_dir.exists() else []
        days = [day_report(store.day(path.parent.name)) for path in paths]

    items = sum(d["items"] for d in days)
    lessons = sum(d["lessons"] for d in days)
    declined = sum(d["declined"] for d in days)
    checked = sum(d["exact_values"]["checked"] for d in days)
    passed = sum(d["exact_values"]["passed"] for d in days)
    failed = sum(d["exact_values"]["failed"] for d in days)
    redacted_cited = sum(d["exact_values"]["redacted_cited"] for d in days)
    return {
        "days": days,
        "totals": {
            "items": items,
            "lessons": lessons,
            "declined": declined,
            "exact_values": {
                "checked": checked,
                "passed": passed,
                "failed": failed,
                "redacted_cited": redacted_cited,
                "missing_rate": failed / checked if checked else None,
            },
        },
        "examples": examples_report(store),
    }
