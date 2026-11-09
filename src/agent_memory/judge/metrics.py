"""Scoring primitives, standard library only, over plain sequences and iterables."""

from __future__ import annotations

import itertools
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass


def accuracy(predicted: Sequence[object], truth: Sequence[object]) -> float | None:
    if not predicted:
        return None
    return sum(p == t for p, t in zip(predicted, truth, strict=True)) / len(predicted)


def auroc(scores: Sequence[float], labels: Sequence[bool]) -> float | None:
    """Mann-Whitney AUROC with ties counted as half. None when only one class is present."""
    pos = [s for s, y in zip(scores, labels, strict=True) if y]
    neg = [s for s, y in zip(scores, labels, strict=True) if not y]
    if not pos or not neg:
        return None
    wins = 0.0
    for p in pos:
        for n in neg:
            wins += 1.0 if p > n else 0.5 if p == n else 0.0
    return wins / (len(pos) * len(neg))


def brier(probs: Sequence[float], labels: Sequence[bool]) -> float | None:
    if not probs:
        return None
    return sum((p - float(y)) ** 2 for p, y in zip(probs, labels, strict=True)) / len(probs)


def ece(probs: Sequence[float], labels: Sequence[bool], bins: int = 10) -> float | None:
    """Expected calibration error over equal-width probability bins."""
    if not probs:
        return None
    buckets: list[list[tuple[float, bool]]] = [[] for _ in range(bins)]
    for p, y in zip(probs, labels, strict=True):
        index = min(bins - 1, int(p * bins))
        buckets[index].append((p, y))
    total = len(probs)
    error = 0.0
    for bucket in buckets:
        if not bucket:
            continue
        mean_p = sum(p for p, _ in bucket) / len(bucket)
        mean_y = sum(float(y) for _, y in bucket) / len(bucket)
        error += len(bucket) / total * abs(mean_p - mean_y)
    return error


def best_threshold(probs: Sequence[float], labels: Sequence[bool]) -> tuple[float, float]:
    """The cut on p(yes) that maximises accuracy; ties resolve toward 0.5."""
    if not probs:
        return 0.5, 0.0
    points = sorted(set(probs))
    cuts = [0.5, *[(a + b) / 2 for a, b in itertools.pairwise(points)]]
    # Include constant predictions. Like the cascade's escalate-all gate, 1.01 also
    # excludes p=1 and survives the four-decimal rounding used when freezing cuts.
    cuts.extend((0.0, 1.0, 1.01))
    best = (0.5, -1.0)
    for cut in cuts:
        acc = sum((p >= cut) == y for p, y in zip(probs, labels, strict=True)) / len(probs)
        better = acc > best[1] or (acc == best[1] and abs(cut - 0.5) < abs(best[0] - 0.5))
        if better:
            best = (cut, acc)
    return best


def paired_discrimination(pairs: Iterable[tuple[float, float]]) -> float | None:
    """Share of (failed, clean) pairs where the failed turn scored higher; ties count half."""
    wins = 0.0
    count = 0
    for f, c in pairs:
        wins += 1.0 if f > c else 0.5 if f == c else 0.0
        count += 1
    return wins / count if count else None


@dataclass(frozen=True)
class BinaryMetrics:
    n: int
    accuracy: float | None
    auroc: float | None
    brier: float | None
    ece: float | None
    confidence_error_auroc: float | None


def binary_metrics(
    probs: Sequence[float],
    confidences: Sequence[float],
    labels: Sequence[bool],
    threshold: float = 0.5,
) -> BinaryMetrics:
    """Accuracy at `threshold`, ranking and calibration of p(yes), and how well the backend's
    confidence separates its own right answers from its wrong ones."""
    predicted = [p >= threshold for p in probs]
    correct = [p == y for p, y in zip(predicted, labels, strict=True)]
    return BinaryMetrics(
        n=len(probs),
        accuracy=accuracy(predicted, labels),
        auroc=auroc(probs, labels),
        brier=brier(probs, labels),
        ece=ece(probs, labels),
        confidence_error_auroc=auroc(confidences, correct),
    )


@dataclass(frozen=True)
class CascadeResult:
    gate: float
    accuracy: float
    cost: float
    escalated_share: float


def cascade(
    confidences: Sequence[float],
    cheap_correct: Sequence[bool],
    expensive_correct: Sequence[bool],
    cheap_cost: Sequence[float],
    expensive_cost: Sequence[float],
    gate: float,
) -> CascadeResult:
    """Accept the cheap answer when its confidence reaches `gate`; otherwise pay for the
    expensive one. Every item pays the cheap cost, since the cheap judge always runs first."""
    n = len(confidences)
    if n == 0:
        return CascadeResult(gate, 0.0, 0.0, 0.0)
    correct = 0
    cost = 0.0
    escalated = 0
    for i in range(n):
        cost += cheap_cost[i]
        if confidences[i] >= gate:
            correct += cheap_correct[i]
        else:
            escalated += 1
            cost += expensive_cost[i]
            correct += expensive_correct[i]
    return CascadeResult(gate, correct / n, cost, escalated / n)


def choose_gate(
    confidences: Sequence[float],
    cheap_correct: Sequence[bool],
    expensive_correct: Sequence[bool],
    cheap_cost: Sequence[float],
    expensive_cost: Sequence[float],
    target_accuracy: float,
) -> CascadeResult:
    """The cheapest gate whose cascade accuracy meets the target; gate 1.01 (escalate all)
    is always a candidate, so the result never falls below the expensive judge alone."""
    candidates = sorted({*confidences, 1.01})
    best: CascadeResult | None = None
    for gate in candidates:
        result = cascade(
            confidences, cheap_correct, expensive_correct, cheap_cost, expensive_cost, gate
        )
        if result.accuracy + 1e-9 < target_accuracy:
            continue
        if best is None or result.cost < best.cost:
            best = result
    if best is None:
        best = cascade(
            confidences, cheap_correct, expensive_correct, cheap_cost, expensive_cost, 1.01
        )
    return best


def fmt(value: float | None, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    return f"{value:.{digits}f}"
