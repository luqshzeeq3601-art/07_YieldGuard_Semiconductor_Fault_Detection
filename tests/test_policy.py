"""Tests for alarm-budget threshold calibration, strict > boundary, ties, and undefined ratios."""

from __future__ import annotations

import numpy as np
import pytest

from yieldguard.policy import (
    compute_alarm_budget_threshold,
    compute_multi_budget_thresholds,
    evaluate_at_threshold,
)


def test_budget_threshold_and_strict_greater_than_boundary() -> None:
    """Verify 1 - B/100 linear quantile threshold and strict `>` alarm rule."""
    # 101 calibration scores: 0, 1, 2, ..., 100
    # 95th percentile (budget 5 per 100) with linear method is 95.0
    # Strict `score > 95.0` flags {96, 97, 98, 99, 100} = 5 runs
    scores = np.arange(101, dtype=np.float64)
    labels = np.zeros(101, dtype=np.int64)
    labels[[96, 99, 50]] = 1  # 3 fails total, 2 above threshold

    tau = compute_alarm_budget_threshold(scores, budget_per_100=5.0)
    assert tau == pytest.approx(95.0, rel=1e-12)

    op = evaluate_at_threshold(labels, scores, tau, budget_per_100=5.0)
    assert op.alarms == 5
    assert op.true_alarms == 2
    assert op.false_alarms == 3
    assert op.precision == pytest.approx(2.0 / 5.0, rel=1e-12)
    assert op.recall == pytest.approx(2.0 / 3.0, rel=1e-12)
    assert op.undefined_reasons == {}


def test_budget_threshold_with_ties_at_boundary() -> None:
    """When multiple runs tie at the threshold, strict `>` excludes tied boundary runs."""
    # 20 runs: 15 zeros, 4 ones (tied at 1.0), 1 two (2.0)
    # At budget 10 per 100 (90th quantile), index = 0.9 * 19 = 17.1 -> falls inside the 1.0 ties -> tau = 1.0
    # Strict `score > 1.0` flags only the single 2.0 run (never exceeding budget due to ties)
    scores = np.array([0.0] * 15 + [1.0] * 4 + [2.0], dtype=np.float64)
    labels = np.array([0] * 19 + [1], dtype=np.int64)

    tau = compute_alarm_budget_threshold(scores, budget_per_100=10.0)
    assert tau == pytest.approx(1.0, rel=1e-12)

    op = evaluate_at_threshold(labels, scores, tau, budget_per_100=10.0)
    assert op.alarms == 1
    assert op.true_alarms == 1
    assert op.false_alarms == 0


def test_zero_alarms_returns_null_precision_and_lift_with_reason() -> None:
    """Constant scores produce 0 alarms under strict `>`, returning None precision/lift with reason."""
    scores = np.ones(50, dtype=np.float64)
    labels = np.array([0] * 45 + [1] * 5, dtype=np.int64)

    tau = compute_alarm_budget_threshold(scores, budget_per_100=5.0)
    assert tau == pytest.approx(1.0, rel=1e-12)

    op = evaluate_at_threshold(labels, scores, tau, budget_per_100=5.0)
    assert op.alarms == 0
    assert op.precision is None
    assert op.lift_over_prevalence is None
    assert op.undefined_reasons["precision"] == "zero_alarms"
    assert op.undefined_reasons["lift_over_prevalence"] == "zero_alarms"
    assert op.recall == 0.0


def test_multi_budget_thresholds_keys() -> None:
    """Verify budgets 2, 5, 10 return expected dictionary keys and monotonic thresholds."""
    scores = np.linspace(0.0, 10.0, 235)
    t_dict = compute_multi_budget_thresholds(scores, budgets_per_100=(2.0, 5.0, 10.0))
    assert set(t_dict.keys()) == {"budget_2", "budget_5", "budget_10"}
    assert t_dict["budget_2"] > t_dict["budget_5"] > t_dict["budget_10"]
