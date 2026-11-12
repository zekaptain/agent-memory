"""Standard-library L2 logistic fitting and calibration-only choice of admission thresholds."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from ..config import Config
from ..judge import metrics
from . import ADMISSION_VERSION, AdmissionError
from . import approver as scoring
from .examples import validated_thresholds
from .store import AdmissionStore

STEPS = 1200
STEP_SIZE = 0.1
L2 = 0.1
PRECISION_TARGET = 0.85


def _logistic(matrix: Sequence[Sequence[float]], labels: Sequence[bool]) -> dict[str, Any]:
    n, width = len(matrix), len(matrix[0])
    means = [sum(row[j] for row in matrix) / n for j in range(width)]
    scales = [
        math.sqrt(sum((row[j] - means[j]) ** 2 for row in matrix) / n) or 1.0 for j in range(width)
    ]
    standardized = [
        [(value - mean) / scale for value, mean, scale in zip(row, means, scales, strict=True)]
        for row in matrix
    ]
    weights, intercept = [0.0] * width, 0.0
    for _ in range(STEPS):
        errors = [
            scoring.sigmoid(intercept + sum(w * x for w, x in zip(weights, row, strict=True)))
            - float(label)
            for row, label in zip(standardized, labels, strict=True)
        ]
        intercept -= STEP_SIZE * sum(errors) / n
        weights = [
            weight
            - STEP_SIZE
            * (
                sum(error * row[j] for error, row in zip(errors, standardized, strict=True)) / n
                + L2 * weight
            )
            for j, weight in enumerate(weights)
        ]
    return {"weights": weights, "intercept": intercept, "means": means, "scales": scales}


def _thresholds(rows: Sequence[Mapping[str, Any]], backend: str) -> dict[str, float]:
    thresholds = [
        validated_thresholds(row.get("judge_thresholds", {}).get(backend), complete=True)
        for row in rows
        if row["split"] == "calibration"
    ]
    if not thresholds:
        raise AdmissionError("calibration examples require frozen judge thresholds")
    if any(value != thresholds[0] for value in thresholds):
        raise AdmissionError("calibration examples have conflicting frozen judge thresholds")
    return thresholds[0]


def _cuts(
    model: dict[str, Any],
    rows: Sequence[dict[str, Any]],
    values: Sequence[float],
    similarities: Sequence[float],
) -> tuple[float, float, float | None]:
    eligible = [
        scoring.judge_action(model, row) != "discard"
        and row.get("judge_action") != "discard"
        and all(
            scoring.gate_values(row, model["backend"]).get(gate) is not None
            for gate in scoring.GATES
        )
        and similarity >= model["similarity_floor"]
        for row, similarity in zip(rows, similarities, strict=True)
    ]
    p_hi = 1.01
    for cut in sorted({0.0, *values}):
        written = [
            row["label"]
            for row, value, ok in zip(rows, values, eligible, strict=True)
            if ok and value >= cut
        ]
        if written and sum(written) / len(written) >= PRECISION_TARGET:
            p_hi = cut
            break
    positives = sum(row["label"] for row in rows)
    judge_missed = sum(
        row["label"] and scoring.judge_action(model, row) != "write_lesson" for row in rows
    )
    judge_miss = judge_missed / positives if positives else None
    # A review is retained for a human, not a false discard. Track write misses separately
    # in the report; p_lo bounds false discards against the frozen judge's write miss budget.
    p_lo = 0.0
    for cut in sorted({0.0, p_hi, *(value for value in values if value <= p_hi)}):
        missed = sum(
            row["label"]
            and (
                scoring.judge_action(model, row) == "discard"
                or row.get("judge_action") == "discard"
                or value < cut
            )
            for row, value in zip(rows, values, strict=True)
        )
        if positives and missed <= judge_missed:
            p_lo = cut
    return p_hi, p_lo, judge_miss


def _evaluate(
    model: Mapping[str, Any],
    rows: Sequence[dict[str, Any]],
    probabilities: Sequence[float],
    similarities: Sequence[float],
) -> dict[str, Any]:
    labels = [row["label"] for row in rows]
    actions = [
        scoring.decide(model, row, value, similarity)[0]
        for row, value, similarity in zip(rows, probabilities, similarities, strict=True)
    ]
    written = [label for label, action in zip(labels, actions, strict=True) if action == "write"]
    positives = sum(labels)
    discarded = sum(
        label and action == "discard" for label, action in zip(labels, actions, strict=True)
    )
    write_misses = sum(
        label and action != "write" for label, action in zip(labels, actions, strict=True)
    )
    precision = sum(written) / len(written) if written else None
    miss = discarded / positives if positives else None
    return {
        "n": len(rows),
        "scored_writes": len(written),
        "writes_scored": len(written),
        "precision": precision,
        "write_precision": precision,
        "miss": miss,
        "miss_rate": miss,
        "write_miss": write_misses / positives if positives else None,
        "coverage": sum(action != "review" for action in actions) / len(rows) if rows else None,
        "write_coverage": len(written) / len(rows) if rows else None,
        "brier": metrics.brier(probabilities, labels),
        "auroc": metrics.auroc(probabilities, labels),
        "actions": {key: actions.count(key) for key in ("write", "review", "discard")},
    }


def fit_approver(
    cfg: Config, *, backend: str, embedder: str, similarity_floor: float = 0.35
) -> dict[str, Any]:
    store = AdmissionStore(cfg)
    # Even a failed refit must not leave a previously enabled, now unverified model active.
    store.write_state(mode="shadow", approver_version=None)
    if not math.isfinite(similarity_floor) or not 0 <= similarity_floor <= 1:
        raise AdmissionError("similarity floor must be finite and in [0, 1]")
    rows = scoring.select_examples(store.read_examples(), backend, embedder)
    calibration = [row for row in rows if row["split"] == "calibration"]
    if not calibration:
        raise AdmissionError("no human-labeled calibration examples; import labels before fitting")
    names = scoring.feature_names(calibration)
    cal_features = [scoring.features(row, calibration, backend, names) for row in calibration]
    model: dict[str, Any] = {
        "schema_version": ADMISSION_VERSION,
        "algorithm": "standardized_l2_logistic_fixed_step_v1",
        "backend": backend,
        "embedder": embedder,
        "embedding_dimensions": len(calibration[0]["embedding"]),
        "thresholds": _thresholds(calibration, backend),
        "features": names,
        "neighbors": scoring.NEIGHBORS,
        "similarity_floor": similarity_floor,
        "optimizer": {"steps": STEPS, "step_size": STEP_SIZE, "l2": L2},
        "examples_hash": scoring.examples_hash(rows),
        "calibration_hash": scoring.examples_hash(calibration),
        "calibration_ids": [row["id"] for row in calibration],
        "counts": {
            "calibration": len(calibration),
            "held_out": len(rows) - len(calibration),
            "positive_calibration": sum(row["label"] for row in calibration),
        },
        **_logistic([value for value, _, _ in cal_features], [row["label"] for row in calibration]),
    }
    values = [scoring.probability(model, value) for value, _, _ in cal_features]
    similarities = [similarity for _, _, similarity in cal_features]
    model["p_hi"], model["p_lo"], model["frozen_judge_miss"] = _cuts(
        model, calibration, values, similarities
    )
    model["precision_target"] = PRECISION_TARGET
    model["metrics"] = {
        "calibration": _evaluate(model, calibration, values, similarities),
    }
    held_out = [row for row in rows if row["split"] == "held_out"]
    held_features = [scoring.features(row, calibration, backend, names) for row in held_out]
    model["metrics"]["held_out"] = _evaluate(
        model,
        held_out,
        [scoring.probability(model, value) for value, _, _ in held_features],
        [similarity for _, _, similarity in held_features],
    )
    model = scoring.seal(model)
    scoring.validate_model(model)
    store.write_approver(model)
    store.write_state(mode="shadow", approver_version=model["version"])
    return model
