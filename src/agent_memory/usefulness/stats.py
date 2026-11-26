"""Standard-library statistics for the usefulness experiment.

Pure numerical functions with no file I/O and no external dependencies:

- Paired success-rate difference with a task-family cluster bootstrap interval
  that resamples whole families and keeps pairs and repeated trials together.
- Exact paired tests for small samples: exact two-sided McNemar's test on
  discordant pairs, and an exact two-sided sign test over task families.
- Task-family count, one-way random-effects ANOVA intra-class correlation,
  unequal-cluster design effect, and effective sample size.
- Paired differences, paired ratios (ratio of means and per-pair ratios),
  medians, and tail quantiles for tokens and wall time, with family-clustered
  bootstrap intervals.
- Clustered rate estimates and Wilson score intervals for secondary measures.

Missing values are represented as ``None``, excluded from estimates, and
counted explicitly as missing; they are never read as zero.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from statistics import NormalDist

_STANDARD_NORMAL = NormalDist(0.0, 1.0)
_DEFAULT_TAIL_QUANTILES: tuple[float, ...] = (0.90, 0.95)


@dataclass(frozen=True)
class BinaryPair:
    """One paired binary task-success outcome belonging to a task family."""

    family: str
    memory: bool | None
    no_memory: bool | None
    task_id: str = ""


@dataclass(frozen=True)
class NumericPair:
    """One paired numeric measure (such as tokens or wall time) in a task family."""

    family: str
    memory: float | None
    no_memory: float | None
    task_id: str = ""


@dataclass(frozen=True)
class BinaryObservation:
    """One binary observation (for a secondary rate) belonging to a task family."""

    family: str
    outcome: bool | None
    task_id: str = ""


@dataclass(frozen=True)
class Interval:
    """A two-sided confidence interval."""

    low: float
    high: float
    confidence: float


@dataclass(frozen=True)
class EstimateWithCI:
    """A point estimate paired with a bootstrap confidence interval."""

    estimate: float | None
    interval: Interval | None
    n_resamples: int


@dataclass(frozen=True)
class ClusterSummary:
    """Task-family count, intra-class correlation, design effect, and effective sample size."""

    n_observations: int
    n_families: int
    icc: float | None
    design_effect: float | None
    effective_sample_size: float


@dataclass(frozen=True)
class ExactMcNemarResult:
    """Exact two-sided (and one-sided) McNemar test on paired binary outcomes."""

    both_pass: int
    memory_only: int
    no_memory_only: int
    both_fail: int
    n_pairs: int
    n_missing_pairs: int
    discordant: int
    p_value: float
    p_value_memory_better: float


@dataclass(frozen=True)
class ExactFamilySignResult:
    """Exact paired sign test over task-family net differences.

    Reduces identically to exact McNemar's test when each family has one pair,
    and respects family-level independence when families have repeated trials or
    multiple tasks.
    """

    families_memory_better: int
    families_no_memory_better: int
    families_tied: int
    n_families: int
    nonzero_families: int
    p_value: float
    p_value_memory_better: float


@dataclass(frozen=True)
class PairedSuccessSummary:
    """Primary paired task-success estimate with family-clustered interval and exact tests."""

    n_total_pairs: int
    n_pairs: int
    n_missing_pairs: int
    n_missing_memory: int
    n_missing_no_memory: int
    memory_successes: int
    no_memory_successes: int
    memory_rate: float | None
    no_memory_rate: float | None
    difference: EstimateWithCI
    clustering: ClusterSummary
    exact_mcnemar: ExactMcNemarResult
    exact_family_sign: ExactFamilySignResult
    seed: int
    confidence: float


@dataclass(frozen=True)
class DistributionSummary:
    """Mean, median, and tail quantiles of one arm with family-clustered bootstrap intervals."""

    n_total: int
    n_observed: int
    n_missing: int
    n_families: int
    mean: EstimateWithCI
    median: EstimateWithCI
    quantiles: dict[float, EstimateWithCI]


@dataclass(frozen=True)
class PairedNumericSummary:
    """Paired differences, ratios, medians, and tail quantiles for tokens or wall time."""

    n_total_pairs: int
    n_pairs: int
    n_missing_pairs: int
    n_missing_memory: int
    n_missing_no_memory: int
    clustering: ClusterSummary
    memory_arm: DistributionSummary
    no_memory_arm: DistributionSummary
    difference_mean: EstimateWithCI
    difference_median: EstimateWithCI
    difference_quantiles: dict[float, EstimateWithCI]
    ratio_of_means: EstimateWithCI
    n_ratio_pairs: int
    n_excluded_ratio_pairs: int
    ratio_median: EstimateWithCI
    ratio_quantiles: dict[float, EstimateWithCI]
    seed: int
    confidence: float


@dataclass(frozen=True)
class RateSummary:
    """Secondary proportion estimate with denominator, missing count, and intervals."""

    numerator: int
    denominator: int
    n_missing: int
    rate: EstimateWithCI
    wilson: Interval | None
    clustering: ClusterSummary
    seed: int
    confidence: float


def _validate_family(family: str) -> str:
    if not isinstance(family, str) or not family.strip():
        raise ValueError("family must be a non-empty string")
    return family


def _validate_bool_or_none(value: bool | None, name: str) -> bool | None:
    if value is not None and not isinstance(value, bool):
        raise TypeError(f"{name} must be bool or None, got {type(value).__name__}")
    return value


def _validate_number_or_none(value: float | None, name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number or None, got {type(value).__name__}")
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"{name} must be finite; use None for missing values")
    return out


def _validate_bootstrap_args(n_resamples: int, confidence: float) -> None:
    if isinstance(n_resamples, bool) or not isinstance(n_resamples, int) or n_resamples < 1:
        raise ValueError("n_resamples must be a positive integer")
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not math.isfinite(confidence)
        or not (0.0 < float(confidence) < 1.0)
    ):
        raise ValueError("confidence must be strictly between 0 and 1")


def _validate_quantiles(quantiles: Sequence[float]) -> tuple[float, ...]:
    validated: list[float] = []
    for q in quantiles:
        if (
            isinstance(q, bool)
            or not isinstance(q, (int, float))
            or not math.isfinite(q)
            or not (0.0 <= float(q) <= 1.0)
        ):
            raise ValueError(f"quantile must be in [0, 1], got {q!r}")
        validated.append(float(q))
    return tuple(dict.fromkeys(validated))


def _quantile_sorted(sorted_values: Sequence[float], q: float) -> float:
    """Linear-interpolation quantile (Hyndman & Fan type 7) on non-empty sorted data."""
    n = len(sorted_values)
    if n == 0:
        raise ValueError("quantile requires at least one value")
    pos = (n - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return sorted_values[lo]
    frac = pos - lo
    return sorted_values[lo] + frac * (sorted_values[hi] - sorted_values[lo])


def quantile(values: Sequence[float], q: float) -> float:
    """Compute the linear-interpolation quantile (R type 7 / NumPy linear) of ``values``."""
    if (
        isinstance(q, bool)
        or not isinstance(q, (int, float))
        or not math.isfinite(q)
        or not (0.0 <= float(q) <= 1.0)
    ):
        raise ValueError(f"q must be in [0, 1], got {q!r}")
    cleaned = [_validate_number_or_none(v, "value") for v in values]
    if any(v is None for v in cleaned):
        raise ValueError("quantile does not accept None; filter missing values first")
    finite = [v for v in cleaned if v is not None]
    if not finite:
        raise ValueError("quantile requires at least one value")
    return _quantile_sorted(sorted(finite), float(q))


def _percentile_interval(
    samples: Sequence[float],
    confidence: float,
) -> Interval | None:
    if not samples:
        return None
    ordered = sorted(samples)
    alpha = 1.0 - confidence
    return Interval(
        low=_quantile_sorted(ordered, alpha / 2.0),
        high=_quantile_sorted(ordered, 1.0 - alpha / 2.0),
        confidence=confidence,
    )


def _group_by_family[T](items: Sequence[tuple[str, T]]) -> list[tuple[str, tuple[T, ...]]]:
    grouped: dict[str, list[T]] = {}
    for family, value in items:
        grouped.setdefault(_validate_family(family), []).append(value)
    return [(family, tuple(grouped[family])) for family in sorted(grouped)]


def exact_binomial_upper_p(successes: int, trials: int) -> float:
    """Exact one-sided p-value P(X >= successes) for X ~ Binomial(trials, 0.5)."""
    if (
        isinstance(successes, bool)
        or isinstance(trials, bool)
        or not isinstance(successes, int)
        or not isinstance(trials, int)
    ):
        raise TypeError("successes and trials must be integers")
    if trials < 0 or successes < 0 or successes > trials:
        raise ValueError("require 0 <= successes <= trials")
    if trials == 0:
        return 1.0
    numerator = sum(math.comb(trials, k) for k in range(successes, trials + 1))
    return min(1.0, numerator / (1 << trials))


def exact_binomial_two_sided_p(successes: int, trials: int) -> float:
    """Exact two-sided p-value for X ~ Binomial(trials, 0.5)."""
    if (
        isinstance(successes, bool)
        or isinstance(trials, bool)
        or not isinstance(successes, int)
        or not isinstance(trials, int)
    ):
        raise TypeError("successes and trials must be integers")
    if trials < 0 or successes < 0 or successes > trials:
        raise ValueError("require 0 <= successes <= trials")
    if trials == 0:
        return 1.0
    tail = min(successes, trials - successes)
    numerator = sum(math.comb(trials, k) for k in range(tail + 1))
    return min(1.0, (2 * numerator) / (1 << trials))


def cluster_summary(family_values: Sequence[tuple[str, float]]) -> ClusterSummary:
    """Compute family count, ANOVA ICC, unequal-cluster design effect, and effective sample size.

    Uses the one-way random-effects ANOVA intra-class correlation (Donner & Klar /
    Fleiss) and Kish's unequal-cluster design effect:

        m_w = sum(m_g^2) / N
        DEFF = 1 + (m_w - 1) * rho_plus
        N_eff = N / DEFF

    Negative sample ICC is clamped to 0 so clustering never inflates N_eff above N.
    When every value is identical across repeated observations (MSB = MSW = 0) or
    only one family is present, rho_plus defaults conservatively to 1 so identical
    repeats are not counted as independent tasks.
    """
    cleaned: list[tuple[str, float]] = []
    for family, value in family_values:
        num = _validate_number_or_none(value, "value")
        if num is None:
            raise ValueError("cluster_summary does not accept None; filter missing values first")
        cleaned.append((_validate_family(family), num))

    n = len(cleaned)
    if n == 0:
        return ClusterSummary(
            n_observations=0,
            n_families=0,
            icc=None,
            design_effect=None,
            effective_sample_size=0.0,
        )

    groups = _group_by_family(cleaned)
    g = len(groups)
    sizes = [len(vals) for _, vals in groups]
    sum_sq_sizes = sum(m * m for m in sizes)
    m_w = sum_sq_sizes / n

    if n == g:
        return ClusterSummary(
            n_observations=n,
            n_families=g,
            icc=None,
            design_effect=1.0,
            effective_sample_size=float(n),
        )

    if g == 1:
        return ClusterSummary(
            n_observations=n,
            n_families=1,
            icc=None,
            design_effect=float(n),
            effective_sample_size=1.0,
        )

    grand_mean = sum(val for _, val in cleaned) / n
    ssb = 0.0
    ssw = 0.0
    for _, vals in groups:
        m = len(vals)
        mean_g = sum(vals) / m
        ssb += m * (mean_g - grand_mean) ** 2
        ssw += sum((v - mean_g) ** 2 for v in vals)

    msb = ssb / (g - 1)
    msw = ssw / (n - g)
    m0 = (n - m_w) / (g - 1)
    denom = msb + (m0 - 1.0) * msw

    if denom == 0.0:
        deff = m_w
        return ClusterSummary(
            n_observations=n,
            n_families=g,
            icc=None,
            design_effect=deff,
            effective_sample_size=n / deff,
        )

    icc = (msb - msw) / denom
    rho_plus = max(0.0, min(1.0, icc))
    deff = 1.0 + (m_w - 1.0) * rho_plus
    return ClusterSummary(
        n_observations=n,
        n_families=g,
        icc=icc,
        design_effect=deff,
        effective_sample_size=n / deff,
    )


def family_bootstrap[T](
    family_groups: Sequence[tuple[str, Sequence[T]]],
    stat_fn: Callable[[Sequence[T]], float | None],
    *,
    seed: int,
    n_resamples: int = 10_000,
    confidence: float = 0.95,
) -> EstimateWithCI:
    """Cluster bootstrap resampling whole task families with replacement.

    Each resample draws ``G`` families with replacement from the ``G`` distinct
    families (keeping pairs and repeated trials inside a family together) and
    evaluates ``stat_fn`` on the concatenated observations.
    """
    _validate_bootstrap_args(n_resamples, confidence)
    normalized: list[tuple[str, tuple[T, ...]]] = []
    seen_families: set[str] = set()
    for family, group in sorted(family_groups, key=lambda item: _validate_family(item[0])):
        fam = _validate_family(family)
        if fam in seen_families:
            raise ValueError(f"duplicate family {fam!r} in family_groups")
        seen_families.add(fam)
        items = tuple(group)
        if items:
            normalized.append((fam, items))

    if not normalized:
        return EstimateWithCI(estimate=None, interval=None, n_resamples=0)

    all_items: list[T] = []
    for _, items in normalized:
        all_items.extend(items)
    estimate = stat_fn(all_items)
    if estimate is None:
        return EstimateWithCI(estimate=None, interval=None, n_resamples=0)

    g = len(normalized)
    blocks = [items for _, items in normalized]
    rng = random.Random(seed)
    boot_values: list[float] = []
    for _ in range(n_resamples):
        drawn: list[T] = []
        for _ in range(g):
            drawn.extend(blocks[rng.randrange(g)])
        val = stat_fn(drawn)
        if val is not None:
            boot_values.append(val)

    return EstimateWithCI(
        estimate=estimate,
        interval=_percentile_interval(boot_values, confidence),
        n_resamples=len(boot_values),
    )


def exact_mcnemar_test(pairs: Sequence[BinaryPair]) -> ExactMcNemarResult:
    """Exact two-sided (and one-sided) McNemar test on complete binary pairs."""
    both_pass = 0
    memory_only = 0
    no_memory_only = 0
    both_fail = 0
    n_missing_pairs = 0

    for pair in pairs:
        _validate_family(pair.family)
        mem = _validate_bool_or_none(pair.memory, "memory")
        nomem = _validate_bool_or_none(pair.no_memory, "no_memory")
        if mem is None or nomem is None:
            n_missing_pairs += 1
            continue
        if mem and nomem:
            both_pass += 1
        elif mem and not nomem:
            memory_only += 1
        elif not mem and nomem:
            no_memory_only += 1
        else:
            both_fail += 1

    discordant = memory_only + no_memory_only
    n_pairs = both_pass + memory_only + no_memory_only + both_fail
    return ExactMcNemarResult(
        both_pass=both_pass,
        memory_only=memory_only,
        no_memory_only=no_memory_only,
        both_fail=both_fail,
        n_pairs=n_pairs,
        n_missing_pairs=n_missing_pairs,
        discordant=discordant,
        p_value=exact_binomial_two_sided_p(memory_only, discordant),
        p_value_memory_better=exact_binomial_upper_p(memory_only, discordant),
    )


def exact_family_sign_test(pairs: Sequence[BinaryPair]) -> ExactFamilySignResult:
    """Exact two-sided (and one-sided) sign test over task-family net paired differences."""
    family_net: dict[str, int] = {}
    for pair in pairs:
        fam = _validate_family(pair.family)
        mem = _validate_bool_or_none(pair.memory, "memory")
        nomem = _validate_bool_or_none(pair.no_memory, "no_memory")
        if mem is None or nomem is None:
            continue
        diff = int(mem) - int(nomem)
        family_net[fam] = family_net.get(fam, 0) + diff

    pos = sum(1 for net in family_net.values() if net > 0)
    neg = sum(1 for net in family_net.values() if net < 0)
    tied = sum(1 for net in family_net.values() if net == 0)
    nonzero = pos + neg
    return ExactFamilySignResult(
        families_memory_better=pos,
        families_no_memory_better=neg,
        families_tied=tied,
        n_families=len(family_net),
        nonzero_families=nonzero,
        p_value=exact_binomial_two_sided_p(pos, nonzero),
        p_value_memory_better=exact_binomial_upper_p(pos, nonzero),
    )


def paired_success_summary(
    pairs: Sequence[BinaryPair],
    *,
    seed: int,
    n_resamples: int = 10_000,
    confidence: float = 0.95,
) -> PairedSuccessSummary:
    """Paired task-success difference with family-clustered bootstrap CI and exact tests."""
    _validate_bootstrap_args(n_resamples, confidence)

    n_total = len(pairs)
    n_missing_memory = 0
    n_missing_no_memory = 0
    complete: list[tuple[str, int, int]] = []

    for pair in pairs:
        fam = _validate_family(pair.family)
        mem = _validate_bool_or_none(pair.memory, "memory")
        nomem = _validate_bool_or_none(pair.no_memory, "no_memory")
        if mem is None:
            n_missing_memory += 1
        if nomem is None:
            n_missing_no_memory += 1
        if mem is not None and nomem is not None:
            complete.append((fam, int(mem), int(nomem)))

    n_pairs = len(complete)
    n_missing_pairs = n_total - n_pairs
    mcnemar = exact_mcnemar_test(pairs)
    family_sign = exact_family_sign_test(pairs)

    if n_pairs == 0:
        return PairedSuccessSummary(
            n_total_pairs=n_total,
            n_pairs=0,
            n_missing_pairs=n_missing_pairs,
            n_missing_memory=n_missing_memory,
            n_missing_no_memory=n_missing_no_memory,
            memory_successes=0,
            no_memory_successes=0,
            memory_rate=None,
            no_memory_rate=None,
            difference=EstimateWithCI(estimate=None, interval=None, n_resamples=0),
            clustering=cluster_summary([]),
            exact_mcnemar=mcnemar,
            exact_family_sign=family_sign,
            seed=seed,
            confidence=confidence,
        )

    mem_wins = sum(m for _, m, _ in complete)
    nomem_wins = sum(c for _, _, c in complete)
    mem_rate = mem_wins / n_pairs
    nomem_rate = nomem_wins / n_pairs
    point_diff = (mem_wins - nomem_wins) / n_pairs

    diff_by_family: list[tuple[str, float]] = [(fam, float(m - c)) for fam, m, c in complete]
    clustering = cluster_summary(diff_by_family)

    # Pre-aggregate (sum_diff, count) per family so each resample is O(G).
    grouped = _group_by_family([(fam, m - c) for fam, m, c in complete])
    fam_totals: list[tuple[int, int]] = [(sum(diffs), len(diffs)) for _, diffs in grouped]
    g = len(fam_totals)
    rng = random.Random(seed)
    boot_diffs: list[float] = []
    for _ in range(n_resamples):
        total_diff = 0
        total_n = 0
        for _ in range(g):
            d_sum, d_n = fam_totals[rng.randrange(g)]
            total_diff += d_sum
            total_n += d_n
        boot_diffs.append(total_diff / total_n)

    return PairedSuccessSummary(
        n_total_pairs=n_total,
        n_pairs=n_pairs,
        n_missing_pairs=n_missing_pairs,
        n_missing_memory=n_missing_memory,
        n_missing_no_memory=n_missing_no_memory,
        memory_successes=mem_wins,
        no_memory_successes=nomem_wins,
        memory_rate=mem_rate,
        no_memory_rate=nomem_rate,
        difference=EstimateWithCI(
            estimate=point_diff,
            interval=_percentile_interval(boot_diffs, confidence),
            n_resamples=len(boot_diffs),
        ),
        clustering=clustering,
        exact_mcnemar=mcnemar,
        exact_family_sign=family_sign,
        seed=seed,
        confidence=confidence,
    )


def distribution_summary(
    observations: Sequence[tuple[str, float | None]],
    *,
    seed: int,
    n_resamples: int = 10_000,
    confidence: float = 0.95,
    tail_quantiles: Sequence[float] = _DEFAULT_TAIL_QUANTILES,
) -> DistributionSummary:
    """Mean, median, and tail quantiles with family-clustered bootstrap intervals."""
    _validate_bootstrap_args(n_resamples, confidence)
    qs = _validate_quantiles(tail_quantiles)

    n_total = len(observations)
    observed: list[tuple[str, float]] = []
    for family, value in observations:
        fam = _validate_family(family)
        num = _validate_number_or_none(value, "value")
        if num is not None:
            observed.append((fam, num))

    n_observed = len(observed)
    n_missing = n_total - n_observed
    if n_observed == 0:
        empty = EstimateWithCI(estimate=None, interval=None, n_resamples=0)
        return DistributionSummary(
            n_total=n_total,
            n_observed=0,
            n_missing=n_missing,
            n_families=0,
            mean=empty,
            median=empty,
            quantiles={q: empty for q in qs},
        )

    groups = _group_by_family(observed)
    blocks = [vals for _, vals in groups]
    g = len(blocks)

    all_sorted = sorted(val for _, val in observed)
    mean_est = math.fsum(all_sorted) / n_observed
    median_est = _quantile_sorted(all_sorted, 0.5)
    q_ests = {q: _quantile_sorted(all_sorted, q) for q in qs}

    rng = random.Random(seed)
    boot_means: list[float] = []
    boot_medians: list[float] = []
    boot_qs: dict[float, list[float]] = {q: [] for q in qs}

    for _ in range(n_resamples):
        sample: list[float] = []
        for _ in range(g):
            sample.extend(blocks[rng.randrange(g)])
        sample.sort()
        boot_means.append(math.fsum(sample) / len(sample))
        boot_medians.append(_quantile_sorted(sample, 0.5))
        for q in qs:
            boot_qs[q].append(_quantile_sorted(sample, q))

    return DistributionSummary(
        n_total=n_total,
        n_observed=n_observed,
        n_missing=n_missing,
        n_families=g,
        mean=EstimateWithCI(
            estimate=mean_est,
            interval=_percentile_interval(boot_means, confidence),
            n_resamples=n_resamples,
        ),
        median=EstimateWithCI(
            estimate=median_est,
            interval=_percentile_interval(boot_medians, confidence),
            n_resamples=n_resamples,
        ),
        quantiles={
            q: EstimateWithCI(
                estimate=q_ests[q],
                interval=_percentile_interval(boot_qs[q], confidence),
                n_resamples=n_resamples,
            )
            for q in qs
        },
    )


def paired_numeric_summary(
    pairs: Sequence[NumericPair],
    *,
    seed: int,
    n_resamples: int = 10_000,
    confidence: float = 0.95,
    tail_quantiles: Sequence[float] = _DEFAULT_TAIL_QUANTILES,
) -> PairedNumericSummary:
    """Paired differences, ratios, medians, and tail quantiles with family-clustered CIs.

    Missing values (``None``) are excluded and counted explicitly, never treated as
    zero. Per-pair ratios (``memory / no_memory``) exclude pairs whose ``no_memory``
    denominator is ``<= 0`` and report that excluded count separately.
    """
    _validate_bootstrap_args(n_resamples, confidence)
    qs = _validate_quantiles(tail_quantiles)

    n_total = len(pairs)
    n_missing_memory = 0
    n_missing_no_memory = 0
    memory_obs: list[tuple[str, float | None]] = []
    no_memory_obs: list[tuple[str, float | None]] = []
    complete: list[tuple[str, float, float]] = []

    for pair in pairs:
        fam = _validate_family(pair.family)
        mem = _validate_number_or_none(pair.memory, "memory")
        nomem = _validate_number_or_none(pair.no_memory, "no_memory")
        memory_obs.append((fam, mem))
        no_memory_obs.append((fam, nomem))
        if mem is None:
            n_missing_memory += 1
        if nomem is None:
            n_missing_no_memory += 1
        if mem is not None and nomem is not None:
            complete.append((fam, mem, nomem))

    memory_arm = distribution_summary(
        memory_obs,
        seed=seed,
        n_resamples=n_resamples,
        confidence=confidence,
        tail_quantiles=qs,
    )
    no_memory_arm = distribution_summary(
        no_memory_obs,
        seed=seed,
        n_resamples=n_resamples,
        confidence=confidence,
        tail_quantiles=qs,
    )

    n_pairs = len(complete)
    n_missing_pairs = n_total - n_pairs
    empty = EstimateWithCI(estimate=None, interval=None, n_resamples=0)

    if n_pairs == 0:
        return PairedNumericSummary(
            n_total_pairs=n_total,
            n_pairs=0,
            n_missing_pairs=n_missing_pairs,
            n_missing_memory=n_missing_memory,
            n_missing_no_memory=n_missing_no_memory,
            clustering=cluster_summary([]),
            memory_arm=memory_arm,
            no_memory_arm=no_memory_arm,
            difference_mean=empty,
            difference_median=empty,
            difference_quantiles={q: empty for q in qs},
            ratio_of_means=empty,
            n_ratio_pairs=0,
            n_excluded_ratio_pairs=0,
            ratio_median=empty,
            ratio_quantiles={q: empty for q in qs},
            seed=seed,
            confidence=confidence,
        )

    diff_items = [(fam, m - c) for fam, m, c in complete]
    clustering = cluster_summary(diff_items)

    # Complete-pair bootstrap for paired differences and ratio of means.
    grouped_pairs = _group_by_family([(fam, (m, c, m - c)) for fam, m, c in complete])
    pair_blocks = [vals for _, vals in grouped_pairs]
    g_pairs = len(pair_blocks)

    all_diffs = sorted(d for _, d in diff_items)
    diff_mean_est = math.fsum(all_diffs) / n_pairs
    diff_median_est = _quantile_sorted(all_diffs, 0.5)
    diff_q_ests = {q: _quantile_sorted(all_diffs, q) for q in qs}

    total_mem = math.fsum(m for _, m, _ in complete)
    total_nomem = math.fsum(c for _, _, c in complete)
    ratio_of_means_est: float | None = (total_mem / total_nomem) if total_nomem > 0.0 else None

    rng = random.Random(seed)
    boot_diff_means: list[float] = []
    boot_diff_medians: list[float] = []
    boot_diff_qs: dict[float, list[float]] = {q: [] for q in qs}
    boot_rom: list[float] = []

    for _ in range(n_resamples):
        drawn_diffs: list[float] = []
        drawn_mem = 0.0
        drawn_nomem = 0.0
        for _ in range(g_pairs):
            for m, c, d in pair_blocks[rng.randrange(g_pairs)]:
                drawn_diffs.append(d)
                drawn_mem += m
                drawn_nomem += c
        drawn_diffs.sort()
        boot_diff_means.append(math.fsum(drawn_diffs) / len(drawn_diffs))
        boot_diff_medians.append(_quantile_sorted(drawn_diffs, 0.5))
        for q in qs:
            boot_diff_qs[q].append(_quantile_sorted(drawn_diffs, q))
        if ratio_of_means_est is not None and drawn_nomem > 0.0:
            boot_rom.append(drawn_mem / drawn_nomem)

    ratio_of_means = (
        EstimateWithCI(
            estimate=ratio_of_means_est,
            interval=_percentile_interval(boot_rom, confidence),
            n_resamples=len(boot_rom),
        )
        if ratio_of_means_est is not None
        else empty
    )

    # Per-pair ratios over pairs with strictly positive no_memory denominator.
    ratio_obs: list[tuple[str, float | None]] = [(fam, m / c) for fam, m, c in complete if c > 0.0]
    n_ratio_pairs = len(ratio_obs)
    n_excluded_ratio_pairs = n_pairs - n_ratio_pairs
    if n_ratio_pairs == 0:
        ratio_median = empty
        ratio_quantiles = {q: empty for q in qs}
    else:
        ratio_dist = distribution_summary(
            ratio_obs,
            seed=seed,
            n_resamples=n_resamples,
            confidence=confidence,
            tail_quantiles=qs,
        )
        ratio_median = ratio_dist.median
        ratio_quantiles = ratio_dist.quantiles

    return PairedNumericSummary(
        n_total_pairs=n_total,
        n_pairs=n_pairs,
        n_missing_pairs=n_missing_pairs,
        n_missing_memory=n_missing_memory,
        n_missing_no_memory=n_missing_no_memory,
        clustering=clustering,
        memory_arm=memory_arm,
        no_memory_arm=no_memory_arm,
        difference_mean=EstimateWithCI(
            estimate=diff_mean_est,
            interval=_percentile_interval(boot_diff_means, confidence),
            n_resamples=len(boot_diff_means),
        ),
        difference_median=EstimateWithCI(
            estimate=diff_median_est,
            interval=_percentile_interval(boot_diff_medians, confidence),
            n_resamples=len(boot_diff_medians),
        ),
        difference_quantiles={
            q: EstimateWithCI(
                estimate=diff_q_ests[q],
                interval=_percentile_interval(boot_diff_qs[q], confidence),
                n_resamples=len(boot_diff_qs[q]),
            )
            for q in qs
        },
        ratio_of_means=ratio_of_means,
        n_ratio_pairs=n_ratio_pairs,
        n_excluded_ratio_pairs=n_excluded_ratio_pairs,
        ratio_median=ratio_median,
        ratio_quantiles=ratio_quantiles,
        seed=seed,
        confidence=confidence,
    )


def wilson_interval(
    successes: int,
    total: int,
    *,
    confidence: float = 0.95,
) -> Interval | None:
    """Two-sided Wilson score confidence interval for a binomial proportion."""
    if (
        isinstance(successes, bool)
        or isinstance(total, bool)
        or not isinstance(successes, int)
        or not isinstance(total, int)
    ):
        raise TypeError("successes and total must be integers")
    if total < 0 or successes < 0 or successes > total:
        raise ValueError("require 0 <= successes <= total")
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not math.isfinite(confidence)
        or not (0.0 < float(confidence) < 1.0)
    ):
        raise ValueError("confidence must be strictly between 0 and 1")
    if total == 0:
        return None
    z = _STANDARD_NORMAL.inv_cdf(1.0 - (1.0 - float(confidence)) / 2.0)
    z2 = z * z
    p_hat = successes / total
    denom = 1.0 + z2 / total
    center = (p_hat + z2 / (2.0 * total)) / denom
    half = (z / denom) * math.sqrt((p_hat * (1.0 - p_hat) + z2 / (4.0 * total)) / total)
    return Interval(
        low=max(0.0, center - half),
        high=min(1.0, center + half),
        confidence=float(confidence),
    )


def clustered_rate_summary(
    observations: Sequence[BinaryObservation],
    *,
    seed: int,
    n_resamples: int = 10_000,
    confidence: float = 0.95,
) -> RateSummary:
    """Proportion with numerator, denominator, missing count, and family-clustered CI."""
    _validate_bootstrap_args(n_resamples, confidence)

    observed: list[tuple[str, int]] = []
    n_missing = 0
    for obs in observations:
        fam = _validate_family(obs.family)
        val = _validate_bool_or_none(obs.outcome, "outcome")
        if val is None:
            n_missing += 1
        else:
            observed.append((fam, int(val)))

    denom = len(observed)
    if denom == 0:
        return RateSummary(
            numerator=0,
            denominator=0,
            n_missing=n_missing,
            rate=EstimateWithCI(estimate=None, interval=None, n_resamples=0),
            wilson=None,
            clustering=cluster_summary([]),
            seed=seed,
            confidence=confidence,
        )

    num = sum(v for _, v in observed)
    rate_est = num / denom
    clustering = cluster_summary([(fam, float(v)) for fam, v in observed])

    grouped = _group_by_family(observed)
    fam_totals: list[tuple[int, int]] = [(sum(vals), len(vals)) for _, vals in grouped]
    g = len(fam_totals)
    rng = random.Random(seed)
    boot_rates: list[float] = []
    for _ in range(n_resamples):
        tot_num = 0
        tot_den = 0
        for _ in range(g):
            s, c = fam_totals[rng.randrange(g)]
            tot_num += s
            tot_den += c
        boot_rates.append(tot_num / tot_den)

    return RateSummary(
        numerator=num,
        denominator=denom,
        n_missing=n_missing,
        rate=EstimateWithCI(
            estimate=rate_est,
            interval=_percentile_interval(boot_rates, confidence),
            n_resamples=len(boot_rates),
        ),
        wilson=wilson_interval(num, denom, confidence=confidence),
        clustering=clustering,
        seed=seed,
        confidence=confidence,
    )
