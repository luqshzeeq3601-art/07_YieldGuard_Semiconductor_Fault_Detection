"""Tests for YieldGuard probability calibration, ECE computation, and odds correction."""

import numpy as np
import pytest

from yieldguard.calibration import (
    ProbabilityCalibrator,
    adjust_probability_for_imbalance,
    brier_score_loss,
    compute_ece,
)


def test_brier_score_loss_exact_values() -> None:
    y_true = np.array([0, 1, 1, 0])
    y_prob = np.array([0.0, 1.0, 1.0, 0.0])
    assert brier_score_loss(y_true, y_prob) == 0.0

    y_prob_inv = np.array([1.0, 0.0, 0.0, 1.0])
    assert brier_score_loss(y_true, y_prob_inv) == 1.0


def test_compute_ece_perfect_calibration() -> None:
    # 10 samples with predicted prob 0.1, exactly 1 positive -> empirical acc = 0.1 -> error = 0.0
    # 10 samples with predicted prob 0.9, exactly 9 positive -> empirical acc = 0.9 -> error = 0.0
    y_true = np.array([1] + [0] * 9 + [1] * 9 + [0] * 1)
    y_prob = np.array([0.1] * 10 + [0.9] * 10)
    ece, bin_df = compute_ece(y_true, y_prob, n_bins=10)
    assert pytest.approx(ece, abs=1e-4) == 0.0
    assert len(bin_df) == 10


def test_adjust_probability_for_imbalance() -> None:
    # Model trained with balanced weights (train prior = 0.5) predicts 0.5
    # Target true prior = 0.05
    # Adjusted probability should equal target true prior 0.05
    probs = np.array([0.5])
    adjusted = adjust_probability_for_imbalance(probs, train_prevalence=0.5, target_prevalence=0.05)
    assert pytest.approx(adjusted[0], abs=1e-4) == 0.05

    # If predicted prob is near 1.0, adjusted remains high
    high_prob = np.array([0.99])
    adj_high = adjust_probability_for_imbalance(high_prob, train_prevalence=0.5, target_prevalence=0.05)
    assert adj_high[0] > 0.8


def test_probability_calibrator_sigmoid_and_isotonic() -> None:
    rng = np.random.default_rng(42)
    # Generate scores correlated with label
    scores = rng.normal(loc=0.0, scale=1.0, size=200)
    probs = 1.0 / (1.0 + np.exp(-scores))
    y = (rng.uniform(size=200) < probs).astype(int)

    # Sigmoid (Platt)
    cal_sig = ProbabilityCalibrator(method="sigmoid").fit(scores, y)
    preds_sig = cal_sig.predict_proba(scores)
    assert preds_sig.min() >= 0.0
    assert preds_sig.max() <= 1.0
    assert cal_sig.brier_score_ < 0.25
    assert cal_sig.ece_ >= 0.0

    # Test monotonic order preserved roughly
    sorted_idx = np.argsort(scores)
    sorted_preds_sig = preds_sig[sorted_idx]
    # Platt scaling is strictly monotonic
    assert np.all(np.diff(sorted_preds_sig) >= -1e-6)

    # Isotonic
    cal_iso = ProbabilityCalibrator(method="isotonic").fit(scores, y)
    preds_iso = cal_iso.predict_proba(scores)
    assert preds_iso.min() >= 0.0
    assert preds_iso.max() <= 1.0
    # Isotonic predictions must be monotonically non-decreasing with sorted scores
    sorted_preds_iso = preds_iso[sorted_idx]
    assert np.all(np.diff(sorted_preds_iso) >= 0.0)
