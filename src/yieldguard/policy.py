"""Alarm-budget threshold calibration and operating-point diagnostics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass
class BudgetOperatingMetrics:
    """Operating metrics at a fixed alarm budget threshold."""

    budget_per_100: float
    threshold: float
    n_runs: int
    n_fails: int
    n_passing: int
    alarms: int
    true_alarms: int
    false_alarms: int
    realised_alarms_per_100: float | None
    false_alarms_per_100_passing: float | None
    precision: float | None
    recall: float | None
    prevalence: float | None
    lift_over_prevalence: float | None
    undefined_reasons: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compute_alarm_budget_threshold(
    calibration_scores: np.ndarray | list[float],
    budget_per_100: float,
) -> float:
    """Compute alarm threshold for `budget_per_100` alarms per 100 runs.

    Per docs/05_EXPERIMENT_PLAN.md section 5.3:
    Threshold is the `1 - B/100` quantile (method `linear`) of the calibration scores.
    Alarm fires when `score > threshold`.
    """
    scores = np.asarray(calibration_scores, dtype=np.float64)
    if len(scores) == 0:
        raise ValueError("Cannot compute alarm budget threshold on empty calibration scores.")
    if np.any(np.isnan(scores)) or np.any(np.isinf(scores)):
        raise ValueError("Calibration scores must be finite.")
    if not (0.0 < budget_per_100 < 100.0):
        raise ValueError(f"budget_per_100 must be in (0, 100), got {budget_per_100}.")

    q = 1.0 - (budget_per_100 / 100.0)
    return float(np.quantile(scores, q, method="linear"))


def compute_multi_budget_thresholds(
    calibration_scores: np.ndarray | list[float],
    budgets_per_100: list[float] | tuple[float, ...] = (2.0, 5.0, 10.0),
) -> dict[str, float]:
    """Compute thresholds for multiple alarm budgets (e.g. 2, 5, 10 per 100 runs)."""
    out: dict[str, float] = {}
    for b in budgets_per_100:
        key = f"budget_{int(b) if float(b).is_integer() else b}"
        out[key] = compute_alarm_budget_threshold(calibration_scores, float(b))
    return out


def apply_alarm_threshold(
    scores: np.ndarray | list[float],
    threshold: float,
) -> np.ndarray:
    """Apply strict `score > threshold` alarm rule."""
    arr = np.asarray(scores, dtype=np.float64)
    return arr > float(threshold)


def evaluate_at_threshold(
    y_true: np.ndarray | list[int],
    scores: np.ndarray | list[float],
    threshold: float,
    budget_per_100: float = 5.0,
) -> BudgetOperatingMetrics:
    """Compute counts and ratios at a frozen threshold using strict `score > threshold`.

    Every ratio is accompanied by raw counts. Undefined ratios return `None` with an
    explicit reason in `undefined_reasons`.
    """
    y = np.asarray(y_true, dtype=np.int64)
    s = np.asarray(scores, dtype=np.float64)
    if len(y) != len(s):
        raise ValueError(f"Length mismatch: y_true ({len(y)}) vs scores ({len(s)}).")

    n_runs = int(len(y))
    n_fails = int(np.sum(y == 1))
    n_passing = int(np.sum(y == 0))

    flagged = apply_alarm_threshold(s, threshold)
    alarms = int(np.sum(flagged))
    true_alarms = int(np.sum(flagged & (y == 1)))
    false_alarms = int(np.sum(flagged & (y == 0)))

    undefined_reasons: dict[str, str] = {}

    if n_runs > 0:
        realised_rate: float | None = float(100.0 * alarms / n_runs)
        prevalence: float | None = float(n_fails / n_runs)
    else:
        realised_rate = None
        prevalence = None
        undefined_reasons["realised_alarms_per_100"] = "zero_runs"
        undefined_reasons["prevalence"] = "zero_runs"

    if n_passing > 0:
        far_per_100: float | None = float(100.0 * false_alarms / n_passing)
    else:
        far_per_100 = None
        undefined_reasons["false_alarms_per_100_passing"] = "zero_passing_runs"

    if alarms > 0:
        precision: float | None = float(true_alarms / alarms)
    else:
        precision = None
        undefined_reasons["precision"] = "zero_alarms"

    if n_fails > 0:
        recall: float | None = float(true_alarms / n_fails)
    else:
        recall = None
        undefined_reasons["recall"] = "zero_failing_runs"

    if precision is not None and prevalence is not None and prevalence > 0.0:
        lift: float | None = float(precision / prevalence)
    else:
        lift = None
        if alarms == 0:
            undefined_reasons["lift_over_prevalence"] = "zero_alarms"
        elif n_fails == 0:
            undefined_reasons["lift_over_prevalence"] = "zero_failing_runs"
        else:
            undefined_reasons["lift_over_prevalence"] = "zero_prevalence"

    return BudgetOperatingMetrics(
        budget_per_100=float(budget_per_100),
        threshold=float(threshold),
        n_runs=n_runs,
        n_fails=n_fails,
        n_passing=n_passing,
        alarms=alarms,
        true_alarms=true_alarms,
        false_alarms=false_alarms,
        realised_alarms_per_100=realised_rate,
        false_alarms_per_100_passing=far_per_100,
        precision=precision,
        recall=recall,
        prevalence=prevalence,
        lift_over_prevalence=lift,
        undefined_reasons=undefined_reasons,
    )
