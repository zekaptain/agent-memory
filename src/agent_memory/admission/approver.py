"""Deterministic neighbor features and guarded logistic admission scoring."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

from ..judge.questions import derive_action
from . import ADMISSION_VERSION, AdmissionError
from .embed import validate_vector
from .examples import GATES, validated_thresholds
from .store import latest_examples

NEIGHBORS = 8
_HASH_FIELDS = (
    "id",
    "source",
    "group_id",
    "memory_id",
    "item_id",
    "rule",
    "kind",
    "claims",
    "embedding",
    "label",
    "split",
    "gates",
    "caller",
    "judge_action",
    "judge_thresholds",
    "embedder",
    "draft_hash",
    "projection_hash",
    "rule_supported",
    "human_labels",
    "frozen_run",
    "frozen_at",
)


def digest(value: Any) -> str:
    try:
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (ValueError, TypeError):
        raise AdmissionError("admission inputs must contain finite JSON values") from None
    return hashlib.sha256(encoded.encode()).hexdigest()


def select_examples(
    examples: Sequence[Mapping[str, Any]], backend: str, embedder: str
) -> list[dict[str, Any]]:
    effective = []
    for row in latest_examples(examples):
        gates = row.get("gates", {})
        if not isinstance(gates, Mapping):
            raise AdmissionError("admission example gates must be an object")
        if row.get("embedder") != embedder or backend not in gates:
            continue
        effective.append(row)
    # Production can review the same captured item previously labeled in an
    # experiment. Its latest feedback, including withdrawal, replaces that sample.
    reviewed = {row["item_id"] for row in effective if row.get("memory_id") and row.get("item_id")}
    rows = []
    dimensions = set()
    groups: dict[str, str] = {}
    for row in effective:
        if not row.get("memory_id") and row.get("item_id") in reviewed:
            continue
        label = row.get("label")
        if label is None:
            continue
        if type(label) is not bool:
            raise AdmissionError("human admission labels must be boolean or null")
        split = row.get("split")
        if split not in ("calibration", "held_out"):
            raise AdmissionError("labeled admission examples require a calibration/held_out split")
        vector = validate_vector(row.get("embedding"))
        dimensions.add(len(vector))
        group = group_id(row)
        if group in groups and groups[group] != split:
            raise AdmissionError("drafts of the same item cannot span calibration and held-out")
        groups[group] = split
        gate_values(row, backend)
        rows.append(row)
    if len(dimensions) > 1:
        raise AdmissionError("admission embedding dimensions do not match")
    return sorted(rows, key=lambda row: str(row["id"]))


def examples_hash(rows: Sequence[Mapping[str, Any]]) -> str:
    return digest([{key: row.get(key) for key in _HASH_FIELDS} for row in rows])


def group_id(example: Mapping[str, Any]) -> str:
    if example.get("memory_id"):
        return f"memory:{example['memory_id']}"
    if example.get("item_id"):
        return str(example["item_id"])
    if example.get("group_id"):
        return str(example["group_id"])
    raise AdmissionError("admission examples require an item grouping identity")


def gate_values(example: Mapping[str, Any], backend: str) -> dict[str, float | None]:
    """The example's gates as `derive_action` reads them: None for a gate the judge left
    unanswered, no entry for one the example does not hold."""
    backends = example.get("gates")
    if not isinstance(backends, Mapping):
        raise AdmissionError("admission example gates must be an object")
    gates = backends.get(backend)
    if not isinstance(gates, Mapping):
        raise AdmissionError("admission example does not match the fitted judge backend")
    values: dict[str, float | None] = {}
    for key in GATES:
        if key not in gates:
            continue
        value = gates[key]
        if value is None:
            values[key] = None
            continue
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not 0 <= value <= 1
        ):
            raise AdmissionError("admission gate probabilities must be finite and in [0, 1]")
        values[key] = float(value)
    return values


def neighbor_features(
    example: Mapping[str, Any], calibration: Sequence[Mapping[str, Any]]
) -> tuple[float, float]:
    """Similarity-weighted k=8 vote, excluding self and every other draft of its item."""
    vector = validate_vector(example.get("embedding"))
    norm = math.hypot(*vector)
    neighbors = []
    own_group = group_id(example)
    for row in calibration:
        if (
            row.get("split") != "calibration"
            or type(row.get("label")) is not bool
            or row.get("id") == example.get("id")
            or group_id(row) == own_group
            or (example.get("item_id") and row.get("item_id") == example["item_id"])
        ):
            continue
        other = validate_vector(row.get("embedding"))
        if len(vector) != len(other):
            raise AdmissionError("admission embedding dimensions do not match")
        other_norm = math.hypot(*other)
        similarity = (
            sum((x / norm) * (y / other_norm) for x, y in zip(vector, other, strict=True))
            if norm and other_norm
            else 0.0
        )
        similarity = max(0.0, min(1.0, similarity))
        neighbors.append((similarity, str(row["id"]), row["label"]))
    neighbors.sort(key=lambda row: (-row[0], row[1]))
    nearest = neighbors[:NEIGHBORS]
    total = sum(sim for sim, _, _ in nearest)
    vote = sum(sim * float(label) for sim, _, label in nearest) / total if total else 0.5
    return vote, nearest[0][0] if nearest else 0.0


def feature_names(calibration: Sequence[Mapping[str, Any]]) -> list[str]:
    """Only calibration rows define the categorical vocabulary."""
    names = [f"judge:{gate}" for gate in GATES]
    for field in ("kind", "caller"):
        values = sorted({str(row.get(field) or "unknown") for row in calibration} - {"<other>"})
        names.extend(f"{field}:{value}" for value in [*values, "<other>"])
    return [*names, "neighbor_vote", "max_similarity"]


def features(
    example: Mapping[str, Any],
    calibration: Sequence[Mapping[str, Any]],
    backend: str,
    names: Sequence[str],
) -> tuple[list[float], float, float]:
    gates = gate_values(example, backend)
    vote, similarity = neighbor_features(example, calibration)
    values: dict[str, float] = {f"judge:{key}": gates.get(key) or 0.0 for key in GATES}
    values.update(neighbor_vote=vote, max_similarity=similarity)
    for field in ("kind", "caller"):
        name = f"{field}:{example.get(field) or 'unknown'}"
        values[name if name in names else f"{field}:<other>"] = 1.0
    return [values.get(name, 0.0) for name in names], vote, similarity


def sigmoid(value: float) -> float:
    if value >= 0:
        return 1 / (1 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1 + exponential)


def probability(model: Mapping[str, Any], values: Sequence[float]) -> float:
    standardized = [
        (value - mean) / scale
        for value, mean, scale in zip(values, model["means"], model["scales"], strict=True)
    ]
    return sigmoid(
        model["intercept"]
        + sum(weight * value for weight, value in zip(model["weights"], standardized, strict=True))
    )


def judge_action(model: Mapping[str, Any], example: Mapping[str, Any]) -> str:
    return derive_action(gate_values(example, model["backend"]), model["thresholds"]).action


def decide(
    model: Mapping[str, Any], example: Mapping[str, Any], value: float, similarity: float
) -> tuple[str, list[str]]:
    if judge_action(model, example) == "discard" or example.get("judge_action") == "discard":
        return "discard", ["frozen judge discarded this draft"]
    gates = gate_values(example, model["backend"])
    if any(gates.get(gate) is None for gate in GATES):
        # The score was fitted on complete answers; without them a person decides.
        return "review", ["judge gate answers are incomplete"]
    if value >= model["p_hi"]:
        if similarity < model["similarity_floor"]:
            return "review", ["similarity below the fitted safety floor"]
        return "write", ["admission score clears the calibrated write cut"]
    if value < model["p_lo"]:
        return "discard", ["admission score below the calibrated review cut"]
    return "review", ["admission score is inside the review band"]


def seal(model: dict[str, Any]) -> dict[str, Any]:
    value = digest({key: value for key, value in model.items() if key not in ("version", "hash")})
    return {**model, "version": f"{ADMISSION_VERSION}:{value[:16]}", "hash": value}


def validate_model(model: Mapping[str, Any]) -> None:
    if not model or model.get("schema_version") != ADMISSION_VERSION:
        raise AdmissionError("no compatible fitted admission approver; import labels and fit first")
    if seal(dict(model)) != model:
        raise AdmissionError("approver hash mismatch; refit before using it")
    try:
        names, weights, means, scales = (
            model[key] for key in ("features", "weights", "means", "scales")
        )
        if not names or len({len(names), len(weights), len(means), len(scales)}) != 1:
            raise ValueError
        values = [*weights, *means, *scales, model["intercept"], model["p_hi"], model["p_lo"]]
        if any(
            isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
            for v in values
        ):
            raise ValueError
        if any(scale <= 0 for scale in scales) or not 0 <= model["p_lo"] <= model["p_hi"] <= 1.01:
            raise ValueError
        if not 0 <= model["similarity_floor"] <= 1:
            raise ValueError
        validated_thresholds(model["thresholds"], complete=True)
    except (KeyError, TypeError, ValueError):
        raise AdmissionError("invalid admission approver coefficients or thresholds") from None


def validate_examples(
    model: Mapping[str, Any], examples: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    validate_model(model)
    selected = select_examples(examples, model["backend"], model["embedder"])
    if examples_hash(selected) != model["examples_hash"]:
        raise AdmissionError("admission examples changed; refit before scoring or enabling auto")
    return selected


def score(
    approver: dict[str, Any], example: dict[str, Any], examples: list[dict[str, Any]]
) -> dict[str, Any]:
    selected = validate_examples(approver, examples)
    if example.get("embedder", approver["embedder"]) != approver["embedder"]:
        raise AdmissionError("scoring embedder does not match the fitted approver")
    if len(validate_vector(example.get("embedding"))) != approver["embedding_dimensions"]:
        raise AdmissionError("scoring embedding dimensions do not match the fitted approver")
    calibration = [row for row in selected if row["split"] == "calibration"]
    values, vote, similarity = features(
        example, calibration, approver["backend"], approver["features"]
    )
    value = probability(approver, values)
    action, reasons = decide(approver, example, value, similarity)
    return {
        "action": action,
        "score": value,
        "reasons": reasons,
        "max_similarity": similarity,
        "neighbor_vote": vote,
    }


def can_auto(approver: dict[str, Any], min_writes: int = 20) -> bool:
    """Only independently scored held-out writes can authorize automatic admission."""
    try:
        validate_model(approver)
        held_out = approver["metrics"]["held_out"]
        precision = held_out.get("precision")
        writes = held_out.get("scored_writes", 0)
        return (
            type(writes) is int
            and type(min_writes) is int
            and min_writes >= 1
            and writes >= min_writes
            and isinstance(precision, (int, float))
            and not isinstance(precision, bool)
            and math.isfinite(precision)
            and 0.8 <= precision <= 1
        )
    except (AdmissionError, KeyError, TypeError, ValueError):
        return False
