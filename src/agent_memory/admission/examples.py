"""Hash-matched exact-draft human labels, never turn-opportunity proxy labels."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ..config import Config
from ..judge.drafter import draft_hash
from ..judge.lesson import lesson_from_row, render_lesson
from ..judge.questions import (
    DEFAULT_THRESHOLDS,
    QUESTIONS_VERSION,
    derive_action,
    gate_probability,
    unanswered,
)
from . import ADMISSION_VERSION, AdmissionError
from .embed import DEFAULT_EMBEDDER, lesson_fields, make_embedder, validate_vector
from .store import AdmissionStore, latest_examples, read_jsonl, redactor_for_config, write_jsonl

GATES = ("grounded", "faithful", "rule_supported", "generalizable", "enforced")


def validated_thresholds(value: Any, *, complete: bool = False) -> dict[str, float]:
    if not isinstance(value, Mapping):
        raise AdmissionError("frozen judge thresholds must be an object")
    if complete and any(key not in value for key in (*GATES, "band")):
        raise AdmissionError("frozen judge thresholds are incomplete")
    out = {}
    for key in DEFAULT_THRESHOLDS:
        number = value.get(key, DEFAULT_THRESHOLDS[key])
        if (
            isinstance(number, bool)
            or not isinstance(number, (int, float))
            or not math.isfinite(number)
            or not 0 <= number <= (1 if key == "band" else 1.01)
        ):
            raise AdmissionError("invalid frozen judge threshold")
        out[key] = float(number)
    return out


def verdict_gates(verdict: Mapping[str, Any]) -> dict[str, float | None]:
    """The judge's probability for each gate of one draft. A gate the judge left unanswered
    is kept as None, so the example still says it was asked."""
    answers = verdict.get("answers", {})
    if not isinstance(answers, Mapping):
        raise AdmissionError("judge answers must be an object")
    out: dict[str, float | None] = {}
    for key in GATES:
        answer = answers.get(key)
        if answer is None:
            continue
        if not isinstance(answer, Mapping) or not isinstance(answer.get("probabilities"), Mapping):
            raise AdmissionError("invalid judge gate answer")
        if unanswered(str(answer.get("method", ""))):
            out[key] = None
            continue
        try:
            p = gate_probability(key, answer["probabilities"])
        except (ValueError, TypeError):
            raise AdmissionError("invalid judge gate probability") from None
        if p is None:
            continue
        if isinstance(p, bool) or not isinstance(p, (int, float)) or not math.isfinite(p):
            raise AdmissionError("invalid judge gate probability")
        if not 0 <= p <= 1:
            raise AdmissionError("judge gate probability must lie in [0, 1]")
        out[key] = float(p)
    return out


def example_for(
    draft: Mapping[str, Any],
    item: Mapping[str, Any],
    verdict: Mapping[str, Any],
    embedding: list[float],
    *,
    source: Any,
    label: bool | None,
    split: str,
) -> dict[str, Any]:
    """Build a narrow example or scoring row. No arbitrary item/verdict text is copied."""
    if label is not None and type(label) is not bool:
        raise AdmissionError("admission labels must be boolean or null")
    if split not in ("calibration", "held_out", "production"):
        raise AdmissionError("invalid admission example split")
    backend = verdict.get("backend")
    if not isinstance(backend, str) or not backend:
        raise AdmissionError("admission example requires its judge backend")
    fields = lesson_fields(draft)
    probabilities = verdict_gates(verdict)
    thresholds = validated_thresholds(verdict.get("thresholds") or {})
    memory_id = item.get("memory_id")
    item_id = item.get("item_id") or draft.get("item_id") or memory_id
    run_id = verdict.get("run_id") or item.get("run_id") or "production"
    group = f"memory:{memory_id}" if memory_id else f"exp:{run_id}:{item_id}"
    identity = hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest()
    result: dict[str, Any] = {
        "version": ADMISSION_VERSION,
        "id": identity,
        "source": source,
        "item_id": item_id,
        "group_id": group,
        **fields,
        "embedding": validate_vector(embedding),
        "embedder": draft.get("embedder") or item.get("embedder") or DEFAULT_EMBEDDER,
        "label": label,
        "split": split,
        "gates": {backend: probabilities},
        "caller": item.get("caller") or (item.get("facts") or {}).get("caller") or "unknown",
        "judge_thresholds": {backend: thresholds},
        "judge_action": derive_action(
            probabilities, thresholds, declined=bool(draft.get("declined"))
        ).action,
        "draft_hash": verdict.get("draft_hash"),
        "projection_hash": verdict.get("projection_hash"),
    }
    if memory_id:
        result["memory_id"] = str(memory_id)
    return result


def _run_metadata(path: Path) -> dict[str, Any]:
    try:
        run = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, UnicodeError, FileNotFoundError):
        raise AdmissionError("source must contain a valid judge run.json") from None
    if not isinstance(run, dict) or run.get("questions_version") != QUESTIONS_VERSION:
        raise AdmissionError("only questions_v4 runs can supply admission labels")
    if not isinstance(run.get("run_id"), str) or not run["run_id"]:
        raise AdmissionError("judge source is missing its run identity")
    return run


def _item_id(row: Mapping[str, Any]) -> str:
    value = row.get("item_id")
    if not isinstance(value, str) or not value:
        raise AdmissionError("judge source row is missing its item identity")
    return value


def _import_identity(source: str, backend: str, embedder: str) -> str:
    # Different model arms share the human label, not their embedding or gate features.
    value = {"source": source, "backend": backend, "embedder": embedder}
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def import_labels(cfg: Config, source: Path, *, backend: str, embedder: str) -> int:
    """Import current frozen v4 exact-draft labels, without touching an experiment ledger."""
    root = source.parent if source.name in ("run.json", "labels.jsonl") else source
    run = _run_metadata(root / "run.json")
    frozen = run.get("frozen")
    if not isinstance(frozen, Mapping):
        raise AdmissionError("admission labels require a frozen judge run")
    backends = frozen.get("backends")
    if not isinstance(backends, Mapping):
        raise AdmissionError("frozen judge backends must be an object")
    spec = backends.get(backend)
    if not isinstance(spec, Mapping):
        raise AdmissionError("chosen backend has no frozen thresholds in this run")
    thresholds = validated_thresholds(spec.get("thresholds"), complete=True)
    items = {_item_id(row): row for row in read_jsonl(root / "items.jsonl")}
    labels = {_item_id(row): row for row in read_jsonl(root / "labels.jsonl")}
    drafts = {
        (_item_id(row), str(row.get("drafter") or "")): row
        for row in read_jsonl(root / "drafts.jsonl")
    }
    verdicts = {}
    for path in sorted((root / "verdicts").glob("*.jsonl")):
        for row in read_jsonl(path):
            if row.get("backend") == backend:
                verdicts[_item_id(row), str(row.get("drafter") or "")] = row
    pending = []
    for key, draft in sorted(drafts.items()):
        item_id, drafter = key
        item, label, verdict = items.get(item_id), labels.get(item_id), verdicts.get(key)
        if item is None or label is None or verdict is None:
            continue
        labeled_drafts = label.get("drafts")
        if not isinstance(labeled_drafts, Mapping):
            continue
        entry = labeled_drafts.get(drafter)
        if label.get("labels_version") != "labels_v4" or not isinstance(entry, Mapping):
            continue
        lesson_text = draft.get("lesson")
        if lesson_text is not None and not isinstance(lesson_text, str):
            raise AdmissionError("judge draft lesson must be text or null")
        current_hash = draft_hash(lesson_text)
        if (
            not current_hash
            or draft.get("declined")
            or entry.get("declined")
            or verdict.get("declined")
            or type(entry.get("store_exact")) is not bool
            or entry.get("draft_hash") != current_hash
            or verdict.get("draft_hash") != current_hash
            or verdict.get("questions_version") != QUESTIONS_VERSION
            or not item.get("projection_hash")
            or verdict.get("projection_hash") != item["projection_hash"]
        ):
            continue
        structured = lesson_from_row(draft)
        if structured is not None and render_lesson(structured) != draft["lesson"]:
            raise AdmissionError("structured draft fields no longer match the labeled draft")
        if item.get("split") not in ("calibration", "held_out"):
            raise AdmissionError("imported admission items require a frozen split")
        pending.append((draft, item, verdict, entry))
    store = AdmissionStore(cfg)
    redactor = redactor_for_config(cfg)
    existing = store.read_examples()
    by_id = {row["id"]: row for row in existing}
    embed = None
    imported = []
    active_ids = set()
    for draft, item, verdict, entry in pending:
        clean, _ = redactor.value(lesson_fields(draft))
        source_id = f"exp/{run['run_id']}/{item['item_id']}/{draft.get('drafter')}"
        identity = _import_identity(source_id, backend, embedder)
        active_ids.add(identity)
        previous = by_id.get(identity, {})
        if previous.get("embedder") == embedder and all(
            previous.get(key) == value for key, value in clean.items()
        ):
            embedding = validate_vector(previous.get("embedding"))
        else:
            if embed is None:
                embed = make_embedder(embedder)
            embedding = embed.embed(clean)
        example = example_for(
            {**clean, "embedder": embedder},
            item,
            {**verdict, "thresholds": thresholds},
            embedding,
            source=source_id,
            label=entry["store_exact"],
            split=item["split"],
        )
        example["id"] = identity
        example["human_labels"] = {
            key: entry.get(key)
            for key in ("grounded", "faithful", "rule_supported", "generalizable", "enforced_by")
        }
        example["rule_supported"] = entry.get("rule_supported")
        example["frozen_run"] = str(run["run_id"])
        example["frozen_at"] = frozen.get("frozen_at")
        example, _ = redactor.value(example)
        if by_id.get(example["id"]) != example:
            imported.append(example)
    obsolete = {
        row["id"]
        for row in existing
        if row.get("frozen_run") == run["run_id"]
        and row.get("embedder") == embedder
        and backend in row.get("gates", {})
        and row["id"] not in active_ids
    }
    if imported or obsolete:
        # New or corrected labels invalidate an enabled model before the data changes.
        store.write_state(mode="shadow", approver_version=None)
        retained = [row for row in existing if row["id"] not in obsolete]
        write_jsonl(store.examples_path, latest_examples([*retained, *imported]))
    if not pending:
        raise AdmissionError("no matching boolean exact-draft labels in the frozen v4 run")
    return len(imported)
