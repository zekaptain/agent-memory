"""Tests for usefulness experiment statistics (`agent_memory.usefulness.stats`).

Every statistic is checked against an independent hand-worked or closed-form
example, including whole-family cluster bootstrap distributions, exact paired
tests, ANOVA ICC / design effect / effective sample size, paired numeric
differences and ratios with tail quantiles, Wilson intervals, and missing-value
accounting.
"""

from __future__ import annotations

import ast
import math
import sys
from pathlib import Path

import pytest

from agent_memory.usefulness import (
    BinaryObservation,
    BinaryPair,
    NumericPair,
    cluster_summary,
    clustered_rate_summary,
    distribution_summary,
    exact_binomial_two_sided_p,
    exact_binomial_upper_p,
    exact_family_sign_test,
    exact_mcnemar_test,
    family_bootstrap,
    paired_numeric_summary,
    paired_success_summary,
    quantile,
    wilson_interval,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Hygiene: stdlib-only and package isolation
# ---------------------------------------------------------------------------


def test_stats_module_uses_stdlib_only() -> None:
    stats_path = REPO_ROOT / "src" / "agent_memory" / "usefulness" / "stats.py"
    tree = ast.parse(stats_path.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, "stats.py should not need internal imports"
            assert node.module is not None
            imported.add(node.module.split(".")[0])
    non_stdlib = imported - sys.stdlib_module_names
    assert non_stdlib == set(), f"unexpected non-stdlib imports: {sorted(non_stdlib)}"


def test_judge_and_admission_do_not_import_usefulness() -> None:
    for pkg in ("judge", "admission"):
        pkg_dir = REPO_ROOT / "src" / "agent_memory" / pkg
        for path in pkg_dir.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        assert "usefulness" not in alias.name.split("."), str(path)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    assert "usefulness" not in node.module.split("."), str(path)


# ---------------------------------------------------------------------------
# Quantiles (Hyndman & Fan type 7 / R default)
# ---------------------------------------------------------------------------


def test_quantile_worked_examples() -> None:
    # Odd n = 5: [10, 20, 30, 40, 50]; pos = (5 - 1) * q = 4 * q.
    odd = [50.0, 10.0, 40.0, 20.0, 30.0]
    assert quantile(odd, 0.0) == pytest.approx(10.0)
    assert quantile(odd, 0.25) == pytest.approx(20.0)
    assert quantile(odd, 0.5) == pytest.approx(30.0)
    assert quantile(odd, 0.9) == pytest.approx(46.0)  # 40 + 0.6 * (50 - 40)
    assert quantile(odd, 0.95) == pytest.approx(48.0)  # 40 + 0.8 * (50 - 40)
    assert quantile(odd, 1.0) == pytest.approx(50.0)

    # Even n = 4: [1, 2, 4, 10]; pos = 3 * q.
    even = [10.0, 1.0, 4.0, 2.0]
    assert quantile(even, 0.5) == pytest.approx(3.0)  # 2 + 0.5 * (4 - 2)
    assert quantile(even, 0.9) == pytest.approx(8.2)  # 4 + 0.7 * (10 - 4)
    assert quantile([42.0], 0.95) == pytest.approx(42.0)


# ---------------------------------------------------------------------------
# Exact paired tests (small samples)
# ---------------------------------------------------------------------------


def test_exact_binomial_and_mcnemar_worked_examples() -> None:
    # Classic small-sample McNemar table: n11=3, n10=7, n01=1, n00=2 (k=8 discordant).
    # Exact P(X <= 1 | Bin(8, 0.5)) = (C(8,0) + C(8,1)) / 256 = 9 / 256 = 0.03515625.
    # Exact two-sided p = 18 / 256 = 9 / 128 = 0.0703125.
    pairs = (
        [BinaryPair(f"both_{i}", True, True) for i in range(3)]
        + [BinaryPair(f"mem_{i}", True, False) for i in range(7)]
        + [BinaryPair("nomem_0", False, True)]
        + [BinaryPair(f"fail_{i}", False, False) for i in range(2)]
        + [BinaryPair("miss_0", True, None), BinaryPair("miss_1", None, False)]
    )
    res = exact_mcnemar_test(pairs)
    assert res.both_pass == 3
    assert res.memory_only == 7
    assert res.no_memory_only == 1
    assert res.both_fail == 2
    assert res.n_pairs == 13
    assert res.n_missing_pairs == 2
    assert res.discordant == 8
    assert res.p_value == pytest.approx(9 / 128)
    assert res.p_value_memory_better == pytest.approx(9 / 256)

    # All 6 discordant pairs favor memory: two-sided p = 2 / 64 = 1 / 32, one-sided = 1 / 64.
    assert exact_binomial_two_sided_p(6, 6) == pytest.approx(1 / 32)
    assert exact_binomial_upper_p(6, 6) == pytest.approx(1 / 64)

    # Equal split (3 vs 3): two-sided capped at 1.0; one-sided = (20+15+6+1)/64 = 21/32.
    assert exact_binomial_two_sided_p(3, 6) == pytest.approx(1.0)
    assert exact_binomial_upper_p(3, 6) == pytest.approx(21 / 32)

    # Zero discordant pairs: p = 1.0.
    zero = exact_mcnemar_test([BinaryPair("a", True, True), BinaryPair("b", False, False)])
    assert zero.discordant == 0
    assert zero.p_value == 1.0
    assert zero.p_value_memory_better == 1.0

    # Large k does not overflow floating point.
    assert exact_binomial_two_sided_p(750, 1500) == pytest.approx(1.0)


def test_exact_family_sign_test_respects_families_vs_repeats() -> None:
    # Two families repeated 5 times identically: naive McNemar treats 10 pairs as
    # independent (p = 2/1024 = 1/512), while family sign test sees 2 families (p = 2/4 = 0.5).
    repeated = [BinaryPair("fam_a", True, False) for _ in range(5)] + [
        BinaryPair("fam_b", True, False) for _ in range(5)
    ]
    mcnemar = exact_mcnemar_test(repeated)
    fam_sign = exact_family_sign_test(repeated)
    assert mcnemar.p_value == pytest.approx(1 / 512)
    assert fam_sign.n_families == 2
    assert fam_sign.families_memory_better == 2
    assert fam_sign.families_no_memory_better == 0
    assert fam_sign.families_tied == 0
    assert fam_sign.p_value == pytest.approx(0.5)
    assert fam_sign.p_value_memory_better == pytest.approx(0.25)

    # With one pair per family, family sign test matches exact McNemar identically.
    singleton = [
        BinaryPair("f1", True, False),
        BinaryPair("f2", True, False),
        BinaryPair("f3", True, False),
        BinaryPair("f4", True, False),
        BinaryPair("f5", False, True),
        BinaryPair("f6", True, True),
    ]
    assert exact_family_sign_test(singleton).p_value == pytest.approx(
        exact_mcnemar_test(singleton).p_value
    )


# ---------------------------------------------------------------------------
# Family count, ICC, design effect, and effective sample size
# ---------------------------------------------------------------------------


def test_cluster_summary_worked_examples() -> None:
    # 1. Empty input.
    empty = cluster_summary([])
    assert empty.n_observations == 0
    assert empty.n_families == 0
    assert empty.icc is None
    assert empty.design_effect is None
    assert empty.effective_sample_size == 0.0

    # 2. One pair per family (N = G = 5): DEFF = 1, N_eff = 5.
    singles = cluster_summary([(f"f{i}", float(i % 2)) for i in range(5)])
    assert singles.n_families == 5
    assert singles.icc is None
    assert singles.design_effect == pytest.approx(1.0)
    assert singles.effective_sample_size == pytest.approx(5.0)

    # 3. Perfect clustering: 4 families x 3 identical repeats.
    #    MSB = 1.0, MSW = 0.0, ICC = 1.0, DEFF = 3.0, N_eff = 12 / 3 = 4.0 = G.
    perfect = cluster_summary(
        [("a", 1.0)] * 3 + [("b", 0.0)] * 3 + [("c", 1.0)] * 3 + [("d", 0.0)] * 3
    )
    assert perfect.n_observations == 12
    assert perfect.n_families == 4
    assert perfect.icc == pytest.approx(1.0)
    assert perfect.design_effect == pytest.approx(3.0)
    assert perfect.effective_sample_size == pytest.approx(4.0)

    # 4. Partial clustering: 3 families x 2: [1, 1], [1, 0], [0, 0].
    #    grand_mean = 0.5, family means = [1, 0.5, 0].
    #    SSB = 2(0.25) + 0 + 2(0.25) = 1.0 -> MSB = 0.5.
    #    SSW = 0 + (0.25 + 0.25) + 0 = 0.5 -> MSW = 1/6.
    #    m0 = 2, ICC = (1/2 - 1/6) / (1/2 + 1/6) = 0.5.
    #    DEFF = 1 + (2 - 1) * 0.5 = 1.5, N_eff = 6 / 1.5 = 4.0.
    partial = cluster_summary(
        [("a", 1.0), ("a", 1.0), ("b", 1.0), ("b", 0.0), ("c", 0.0), ("c", 0.0)]
    )
    assert partial.n_families == 3
    assert partial.icc == pytest.approx(0.5)
    assert partial.design_effect == pytest.approx(1.5)
    assert partial.effective_sample_size == pytest.approx(4.0)

    # 5. Unequal cluster sizes: A=[1,1,1] (3), B=[0,0] (2), C=[1] (1).
    #    N=6, G=3, sum(m_g^2)=14, m_w=7/3, MSW=0 -> ICC=1.0, DEFF=7/3, N_eff=18/7.
    unequal = cluster_summary([("a", 1.0)] * 3 + [("b", 0.0)] * 2 + [("c", 1.0)])
    assert unequal.n_families == 3
    assert unequal.icc == pytest.approx(1.0)
    assert unequal.design_effect == pytest.approx(7 / 3)
    assert unequal.effective_sample_size == pytest.approx(18 / 7)

    # 6. Negative sample ICC clamped to 0 so N_eff never exceeds N.
    #    4 families x 3 with identical family mean 1/3 -> MSB=0, MSW=1/3, ICC=-0.5.
    neg_icc = cluster_summary(
        [
            ("a", 1.0),
            ("a", 0.0),
            ("a", 0.0),
            ("b", 0.0),
            ("b", 1.0),
            ("b", 0.0),
            ("c", 0.0),
            ("c", 0.0),
            ("c", 1.0),
            ("d", 1.0),
            ("d", 0.0),
            ("d", 0.0),
        ]
    )
    assert neg_icc.icc == pytest.approx(-0.5)
    assert neg_icc.design_effect == pytest.approx(1.0)
    assert neg_icc.effective_sample_size == pytest.approx(12.0)

    # 7. Constant values across repeated trials (MSB = MSW = 0):
    #    conservatively counts families, not repeats.
    constant = cluster_summary([("a", 0.0)] * 2 + [("b", 0.0)] * 2 + [("c", 0.0)] * 2)
    assert constant.icc is None
    assert constant.design_effect == pytest.approx(2.0)
    assert constant.effective_sample_size == pytest.approx(3.0)

    # 8. Single family with repeats: N_eff = 1.0.
    one_fam = cluster_summary([("only", 1.0), ("only", 0.0), ("only", 1.0)])
    assert one_fam.n_families == 1
    assert one_fam.design_effect == pytest.approx(3.0)
    assert one_fam.effective_sample_size == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Paired success summary and whole-family bootstrap verification
# ---------------------------------------------------------------------------


def test_paired_success_exact_finite_bootstrap_distribution() -> None:
    # Two families with unequal sizes:
    #   A: 3 pairs of (True, False) -> diff sum = +3, count = 3
    #   B: 1 pair of (False, True)  -> diff sum = -1, count = 1
    # Point estimate: (3 - 1) / 4 = +0.5.
    # Exact whole-family bootstrap draws (each prob 0.25):
    #   (A, A) -> +6 / 6 = +1.0
    #   (A, B) -> +2 / 4 = +0.5
    #   (B, A) -> +2 / 4 = +0.5
    #   (B, B) -> -2 / 2 = -1.0
    # Exact 95% percentile CI is [-1.0, +1.0].
    pairs = [
        BinaryPair("A", True, False, task_id="t1"),
        BinaryPair("A", True, False, task_id="t1"),
        BinaryPair("A", True, False, task_id="t1"),
        BinaryPair("B", False, True, task_id="t2"),
    ]
    res = paired_success_summary(pairs, seed=12345, n_resamples=10_000, confidence=0.95)
    assert res.n_pairs == 4
    assert res.n_missing_pairs == 0
    assert res.memory_rate == pytest.approx(0.75)
    assert res.no_memory_rate == pytest.approx(0.25)
    assert res.difference.estimate == pytest.approx(0.5)
    assert res.difference.interval is not None
    assert res.difference.interval.low == pytest.approx(-1.0)
    assert res.difference.interval.high == pytest.approx(1.0)
    assert res.clustering.n_families == 2
    # sum(m^2) = 9 + 1 = 10 -> m_w = 2.5 -> DEFF = 2.5 -> N_eff = 4 / 2.5 = 1.6.
    assert res.clustering.effective_sample_size == pytest.approx(1.6)


def test_paired_success_bootstrap_keeps_repeated_trials_in_family() -> None:
    # 8 families: 4 families with 4 identical (True, False) repeats (diff=+1) and
    # 4 families with 4 identical (True, True) repeats (diff=0). N=32, G=8.
    # Whole-family bootstrap distribution is (1/8) * Binomial(8, 0.5):
    #   P(X = 0) = 1/256 = 0.0039 < 0.025
    #   P(X <= 1) = 9/256 = 0.0352 > 0.025
    # So exact 95% percentile CI is [1/8, 7/8] = [0.125, 0.875].
    # (An unclustered pair-level bootstrap would falsely narrow it to [0.3125, 0.6875].)
    pairs: list[BinaryPair] = []
    for i in range(4):
        pairs.extend(BinaryPair(f"win_{i}", True, False) for _ in range(4))
    for i in range(4):
        pairs.extend(BinaryPair(f"tie_{i}", True, True) for _ in range(4))

    res = paired_success_summary(pairs, seed=2026, n_resamples=20_000, confidence=0.95)
    assert res.n_pairs == 32
    assert res.clustering.n_families == 8
    assert res.clustering.effective_sample_size == pytest.approx(8.0)
    assert res.difference.estimate == pytest.approx(0.5)
    assert res.difference.interval is not None
    assert res.difference.interval.low == pytest.approx(0.125)
    assert res.difference.interval.high == pytest.approx(0.875)


def test_paired_success_missing_values_excluded_and_counted() -> None:
    pairs = [
        BinaryPair("f1", True, False),
        BinaryPair("f2", False, False),
        BinaryPair("f3", True, None),  # missing no_memory -> must not be read as False
        BinaryPair("f4", None, True),  # missing memory -> must not be read as False
        BinaryPair("f5", None, None),
    ]
    res = paired_success_summary(pairs, seed=7, n_resamples=2_000)
    assert res.n_total_pairs == 5
    assert res.n_pairs == 2
    assert res.n_missing_pairs == 3
    assert res.n_missing_memory == 2
    assert res.n_missing_no_memory == 2
    assert res.memory_successes == 1
    assert res.no_memory_successes == 0
    assert res.memory_rate == pytest.approx(0.5)
    assert res.no_memory_rate == pytest.approx(0.0)
    assert res.difference.estimate == pytest.approx(0.5)
    assert res.clustering.n_families == 2

    # All missing -> None estimates, never 0.0.
    all_missing = paired_success_summary(
        [BinaryPair("f1", None, True), BinaryPair("f2", True, None)],
        seed=7,
        n_resamples=100,
    )
    assert all_missing.n_pairs == 0
    assert all_missing.n_missing_pairs == 2
    assert all_missing.memory_rate is None
    assert all_missing.no_memory_rate is None
    assert all_missing.difference.estimate is None
    assert all_missing.difference.interval is None
    assert all_missing.clustering.effective_sample_size == 0.0


def test_paired_success_seed_determinism_and_order_invariance() -> None:
    pairs = [
        BinaryPair("f1", True, False),
        BinaryPair("f2", True, True),
        BinaryPair("f3", False, True),
        BinaryPair("f4", True, False),
        BinaryPair("f5", False, False),
        BinaryPair("f6", True, False),
    ]
    a = paired_success_summary(pairs, seed=99, n_resamples=4_000)
    b = paired_success_summary(list(reversed(pairs)), seed=99, n_resamples=4_000)
    assert a == b


# ---------------------------------------------------------------------------
# Paired numeric summary (tokens and wall time)
# ---------------------------------------------------------------------------


def test_paired_numeric_worked_example_with_missing_values() -> None:
    # Complete pairs (n = 4):
    #   f1: (120, 100) -> diff = +20,  ratio = 1.2
    #   f2: (150, 100) -> diff = +50,  ratio = 1.5
    #   f3: (180, 200) -> diff = -20,  ratio = 0.9
    #   f4: (250, 100) -> diff = +150, ratio = 2.5
    # Incomplete pairs (missing values must NOT be treated as 0):
    #   f5: (None, 500)
    #   f6: (300, None)
    pairs = [
        NumericPair("f1", 120.0, 100.0),
        NumericPair("f2", 150.0, 100.0),
        NumericPair("f3", 180.0, 200.0),
        NumericPair("f4", 250.0, 100.0),
        NumericPair("f5", None, 500.0),
        NumericPair("f6", 300.0, None),
    ]
    res = paired_numeric_summary(
        pairs,
        seed=42,
        n_resamples=5_000,
        confidence=0.95,
        tail_quantiles=(0.90, 0.95),
    )

    assert res.n_total_pairs == 6
    assert res.n_pairs == 4
    assert res.n_missing_pairs == 2
    assert res.n_missing_memory == 1
    assert res.n_missing_no_memory == 1
    assert res.clustering.n_families == 4
    assert res.clustering.effective_sample_size == pytest.approx(4.0)

    # Paired differences sorted: [-20, 20, 50, 150].
    assert res.difference_mean.estimate == pytest.approx(50.0)
    assert res.difference_median.estimate == pytest.approx(35.0)
    assert res.difference_quantiles[0.90].estimate == pytest.approx(120.0)
    assert res.difference_quantiles[0.95].estimate == pytest.approx(135.0)

    # Ratio of means: (120 + 150 + 180 + 250) / (100 + 100 + 200 + 100) = 700 / 500 = 1.4.
    assert res.ratio_of_means.estimate == pytest.approx(1.4)
    assert res.n_ratio_pairs == 4
    assert res.n_excluded_ratio_pairs == 0

    # Per-pair ratios sorted: [0.9, 1.2, 1.5, 2.5].
    assert res.ratio_median.estimate == pytest.approx(1.35)
    assert res.ratio_quantiles[0.90].estimate == pytest.approx(2.2)
    assert res.ratio_quantiles[0.95].estimate == pytest.approx(2.35)

    # Per-arm distributions keep all observed values in each arm (5 observed, 1 missing).
    # memory_arm: [120, 150, 180, 250, 300] -> mean=200, median=180, p90=280, p95=290.
    assert res.memory_arm.n_total == 6
    assert res.memory_arm.n_observed == 5
    assert res.memory_arm.n_missing == 1
    assert res.memory_arm.n_families == 5
    assert res.memory_arm.mean.estimate == pytest.approx(200.0)
    assert res.memory_arm.median.estimate == pytest.approx(180.0)
    assert res.memory_arm.quantiles[0.90].estimate == pytest.approx(280.0)
    assert res.memory_arm.quantiles[0.95].estimate == pytest.approx(290.0)

    # no_memory_arm: [100, 100, 100, 200, 500] -> mean=200, median=100, p90=380, p95=440.
    assert res.no_memory_arm.n_observed == 5
    assert res.no_memory_arm.n_missing == 1
    assert res.no_memory_arm.mean.estimate == pytest.approx(200.0)
    assert res.no_memory_arm.median.estimate == pytest.approx(100.0)
    assert res.no_memory_arm.quantiles[0.90].estimate == pytest.approx(380.0)
    assert res.no_memory_arm.quantiles[0.95].estimate == pytest.approx(440.0)

    # Every interval brackets its point estimate and has 5,000 resamples.
    for est in (
        res.difference_mean,
        res.difference_median,
        res.difference_quantiles[0.90],
        res.difference_quantiles[0.95],
        res.ratio_of_means,
        res.ratio_median,
        res.ratio_quantiles[0.90],
        res.ratio_quantiles[0.95],
        res.memory_arm.mean,
        res.memory_arm.median,
        res.no_memory_arm.mean,
        res.no_memory_arm.median,
    ):
        assert est.estimate is not None
        assert est.interval is not None
        assert est.n_resamples == 5_000
        assert est.interval.low <= est.estimate <= est.interval.high

    # Order invariance.
    rev = paired_numeric_summary(
        list(reversed(pairs)),
        seed=42,
        n_resamples=5_000,
        confidence=0.95,
        tail_quantiles=(0.90, 0.95),
    )
    assert res == rev


def test_paired_numeric_cluster_bootstrap_and_zero_denominator() -> None:
    # Two families with exact 4-outcome cluster bootstrap for ratio_of_means:
    #   A: (200, 100) x 2 -> sum_m=400, sum_c=200, diff=+100, ratio=2.0
    #   B: (100, 200) x 2 -> sum_m=200, sum_c=400, diff=-100, ratio=0.5
    # Draws: (A,A) -> ratio 2.0, diff +100; (A,B)/(B,A) -> ratio 1.0, diff 0;
    # (B,B) -> ratio 0.5, diff -100.
    pairs = [
        NumericPair("A", 200.0, 100.0),
        NumericPair("A", 200.0, 100.0),
        NumericPair("B", 100.0, 200.0),
        NumericPair("B", 100.0, 200.0),
    ]
    res = paired_numeric_summary(pairs, seed=11, n_resamples=8_000)
    assert res.difference_mean.estimate == pytest.approx(0.0)
    assert res.difference_mean.interval is not None
    assert res.difference_mean.interval.low == pytest.approx(-100.0)
    assert res.difference_mean.interval.high == pytest.approx(100.0)
    assert res.ratio_of_means.estimate == pytest.approx(1.0)
    assert res.ratio_of_means.interval is not None
    assert res.ratio_of_means.interval.low == pytest.approx(0.5)
    assert res.ratio_of_means.interval.high == pytest.approx(2.0)

    # Zero denominator in no_memory is excluded from per-pair ratios without error.
    with_zero = paired_numeric_summary(
        [
            NumericPair("f1", 20.0, 10.0),
            NumericPair("f2", 15.0, 0.0),
        ],
        seed=5,
        n_resamples=500,
    )
    assert with_zero.n_pairs == 2
    assert with_zero.difference_mean.estimate == pytest.approx(12.5)
    assert with_zero.n_ratio_pairs == 1
    assert with_zero.n_excluded_ratio_pairs == 1
    assert with_zero.ratio_median.estimate == pytest.approx(2.0)
    assert with_zero.ratio_of_means.estimate == pytest.approx(3.5)

    # All-missing pairs return None estimates, never 0.
    all_missing = paired_numeric_summary(
        [NumericPair("f1", None, 10.0), NumericPair("f2", 20.0, None)],
        seed=5,
        n_resamples=200,
    )
    assert all_missing.n_pairs == 0
    assert all_missing.n_missing_pairs == 2
    assert all_missing.difference_mean.estimate is None
    assert all_missing.ratio_of_means.estimate is None
    assert all_missing.ratio_median.estimate is None
    assert all_missing.memory_arm.n_observed == 1
    assert all_missing.memory_arm.mean.estimate == pytest.approx(20.0)
    assert all_missing.no_memory_arm.n_observed == 1
    assert all_missing.no_memory_arm.mean.estimate == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# Secondary rate intervals (Wilson score and family-clustered bootstrap)
# ---------------------------------------------------------------------------


def test_wilson_interval_and_clustered_rate_worked_examples() -> None:
    # Closed-form Wilson 95% CI for 3 / 10 (matches R prop.test(3, 10, correct=FALSE)).
    iv = wilson_interval(3, 10, confidence=0.95)
    assert iv is not None
    assert iv.low == pytest.approx(0.10779127, abs=2e-4)
    assert iv.high == pytest.approx(0.60322186, abs=2e-4)

    # Exact closed-form at x = 0: high = z^2 / (n + z^2).
    iv_zero = wilson_interval(0, 10, confidence=0.95)
    assert iv_zero is not None
    assert iv_zero.low == 0.0
    assert iv_zero.high == pytest.approx(0.2775328, abs=1e-6)

    iv_one = wilson_interval(10, 10, confidence=0.95)
    assert iv_one is not None
    assert iv_one.high == 1.0
    assert iv_one.low == pytest.approx(1.0 - 0.2775328, abs=1e-6)

    assert wilson_interval(0, 0) is None

    # Clustered rate summary excludes None and resamples whole families.
    obs = [
        BinaryObservation("A", True),
        BinaryObservation("A", True),
        BinaryObservation("B", False),
        BinaryObservation("B", False),
        BinaryObservation("C", None),
    ]
    rate = clustered_rate_summary(obs, seed=17, n_resamples=8_000, confidence=0.95)
    assert rate.numerator == 2
    assert rate.denominator == 4
    assert rate.n_missing == 1
    assert rate.rate.estimate == pytest.approx(0.5)
    assert rate.rate.interval is not None
    # Exact 2-family draws: (A,A)->1.0, (A,B)/(B,A)->0.5, (B,B)->0.0.
    assert rate.rate.interval.low == pytest.approx(0.0)
    assert rate.rate.interval.high == pytest.approx(1.0)
    assert rate.clustering.n_families == 2
    assert rate.clustering.effective_sample_size == pytest.approx(2.0)


def test_family_bootstrap_generic_and_validation() -> None:
    res = family_bootstrap(
        [("b", [2.0, 2.0]), ("a", [10.0, 10.0])],
        lambda vals: sum(vals) / len(vals),
        seed=3,
        n_resamples=4_000,
    )
    assert res.estimate == pytest.approx(6.0)
    assert res.interval is not None
    assert res.interval.low == pytest.approx(2.0)
    assert res.interval.high == pytest.approx(10.0)

    # Empty groups.
    empty = family_bootstrap([], lambda vals: 0.0, seed=3, n_resamples=100)
    assert empty.estimate is None
    assert empty.interval is None

    # Input validation guards.
    with pytest.raises(ValueError, match="finite"):
        paired_numeric_summary([NumericPair("f1", math.nan, 1.0)], seed=1, n_resamples=10)
    with pytest.raises(ValueError, match="finite"):
        paired_numeric_summary([NumericPair("f1", math.inf, 1.0)], seed=1, n_resamples=10)
    with pytest.raises(TypeError, match="real number"):
        paired_numeric_summary([NumericPair("f1", True, 1.0)], seed=1, n_resamples=10)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="bool or None"):
        paired_success_summary([BinaryPair("f1", 1, False)], seed=1, n_resamples=10)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="family"):
        paired_success_summary([BinaryPair("  ", True, False)], seed=1, n_resamples=10)
    with pytest.raises(ValueError, match="n_resamples"):
        paired_success_summary([BinaryPair("f1", True, False)], seed=1, n_resamples=0)
    with pytest.raises(ValueError, match="confidence"):
        paired_success_summary([BinaryPair("f1", True, False)], seed=1, confidence=1.0)
    with pytest.raises(ValueError, match="quantile"):
        distribution_summary([("f1", 1.0)], seed=1, tail_quantiles=(1.5,))
    with pytest.raises(ValueError, match="duplicate family"):
        family_bootstrap([("a", [1.0]), ("a", [2.0])], lambda v: v[0], seed=1, n_resamples=10)
