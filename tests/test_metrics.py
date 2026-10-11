"""Tests for PR-AUC, recall at fixed false-alarm rate, bootstrap reproducibility, and undefined ratios."""

from __future__ import annotations

import numpy as np
import pytest

from yieldguard.config import load_config
from yieldguard.evaluate import (
    compute_pr_auc,
    compute_recall_at_fixed_far,
    compute_roc_auc,
    moving_block_bootstrap_ci,
    paired_stratified_bootstrap_diff,
    select_primary_detector,
    stratified_bootstrap_ci,
)


def test_hand_computed_pr_auc_and_recall_at_fixed_far() -> None:
    """PR-AUC and recall at fixed FAR match exact hand-computed values."""
    # 4 runs sorted descending by score: [0.9, 0.8, 0.7, 0.6], labels: [1, 0, 1, 0]
    # Precision at rank 1 (y=1): 1/1 = 1.0; Precision at rank 3 (y=1): 2/3
    # Average precision = (1.0 + 2/3) / 2 = 5/6
    y_small = [1, 0, 1, 0]
    s_small = [0.9, 0.8, 0.7, 0.6]
    ap, reason = compute_pr_auc(y_small, s_small)
    assert reason is None
    assert ap == pytest.approx(5.0 / 6.0, rel=1e-12)

    # 10 passing runs with scores 1..10, 4 failing runs with scores [4.5, 8.5, 9.5, 11.0]
    # At FAR = 20 per 100 passing runs:
    # 0.80 linear quantile of [1..10] is 1 + 0.8 * 9 = 8.2
    # Passing runs > 8.2: {9, 10} -> 2 / 10 = 20.0 per 100
    # Failing runs > 8.2: {8.5, 9.5, 11.0} -> 3 / 4 = 0.75 recall
    y_far = [0] * 10 + [1] * 4
    s_far = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 4.5, 8.5, 9.5, 11.0]
    res = compute_recall_at_fixed_far(y_far, s_far, far_per_100=20.0)
    assert res["undefined_reason"] is None
    assert res["threshold"] == pytest.approx(8.2, rel=1e-12)
    assert res["false_alarms"] == 2
    assert res["realised_far_per_100"] == pytest.approx(20.0, rel=1e-12)
    assert res["true_alarms"] == 3
    assert res["recall"] == pytest.approx(0.75, rel=1e-12)


def test_seeded_bootstrap_is_reproducible() -> None:
    """Stratified, paired, and moving-block bootstraps are deterministic under a fixed seed."""
    rng = np.random.default_rng(7)
    y = np.array([0] * 80 + [1] * 20, dtype=np.int64)
    s1 = rng.normal(0.0, 1.0, size=100) + y * 1.2
    s2 = rng.normal(0.0, 1.0, size=100) + y * 0.5

    ci1 = stratified_bootstrap_ci(y, s1, n_bootstraps=500, seed=42)
    ci2 = stratified_bootstrap_ci(y, s1, n_bootstraps=500, seed=42)
    assert ci1["pr_auc_95_ci"] == ci2["pr_auc_95_ci"]
    assert ci1["recall_at_far_95_ci"] == ci2["recall_at_far_95_ci"]

    mb1 = moving_block_bootstrap_ci(y, s1, block_length=10, n_bootstraps=300, seed=42)
    mb2 = moving_block_bootstrap_ci(y, s1, block_length=10, n_bootstraps=300, seed=42)
    assert mb1["pr_auc_95_ci"] == mb2["pr_auc_95_ci"]

    diff1 = paired_stratified_bootstrap_diff(y, s1, s2, n_bootstraps=500, seed=42)
    diff2 = paired_stratified_bootstrap_diff(y, s1, s2, n_bootstraps=500, seed=42)
    assert diff1 == diff2


def test_undefined_ratios_return_null_with_reason() -> None:
    """When failing or passing runs are zero, metrics return None with an explicit reason."""
    y_no_fails = [0, 0, 0, 0]
    s = [0.1, 0.2, 0.3, 0.4]

    ap, ap_reason = compute_pr_auc(y_no_fails, s)
    assert ap is None
    assert ap_reason == "zero_failing_runs"

    roc, roc_reason = compute_roc_auc(y_no_fails, s)
    assert roc is None
    assert roc_reason == "zero_failing_runs"

    far_res = compute_recall_at_fixed_far(y_no_fails, s, far_per_100=5.0)
    assert far_res["recall"] is None
    assert far_res["undefined_reason"] == "zero_failing_runs"


def test_primary_selection_rule_prefers_spc_when_ci_contains_zero() -> None:
    """Rule 4 selects best SPC when ML advantage 95% CI contains zero."""
    summaries = {
        "E04": {"mean_fold_pr_auc": 0.20},
        "E06": {"mean_fold_pr_auc": 0.25},
        "E07": {"mean_fold_pr_auc": 0.22},
        "E08": {"mean_fold_pr_auc": 0.27},
    }
    # Construct bootstrap draws where E08 - E06 spans [-0.03, +0.07] (contains 0)
    boot_means = {
        "E04": np.full(100, 0.20),
        "E06": np.full(100, 0.25),
        "E07": np.full(100, 0.22),
        "E08": np.linspace(0.22, 0.32, 100),
    }
    sel_noise = select_primary_detector(summaries, boot_means)
    assert sel_noise["ci_contains_zero"] is True
    assert sel_noise["primary_detector"] == "E06"

    # Now construct draws where E08 - E06 is strictly positive [+0.02, +0.08]
    boot_means["E08"] = np.linspace(0.27, 0.33, 100)
    sel_clear = select_primary_detector(summaries, boot_means)
    assert sel_clear["ci_contains_zero"] is False
    assert sel_clear["primary_detector"] == "E08"


def test_targets_o1_to_o7_present_in_config() -> None:
    """Verify targets O1-O7 are predeclared in configs/default.yaml."""
    cfg = load_config("configs/default.yaml")
    assert cfg.targets.o1_pr_auc_min_prevalence_multiple == 2.0
    assert cfg.targets.o1_require_ci_lower_above_prevalence is True
    assert cfg.targets.o2_min_recall_at_5_far == 0.20
    assert cfg.targets.o3_alarm_rate_min_per_100 == 2.0
    assert cfg.targets.o3_alarm_rate_max_per_100 == 8.0
    assert cfg.targets.o4_paired_bootstrap_reported is True
    assert cfg.targets.o5_synthetic_top3_hit_rate_min == 0.90
    assert cfg.targets.o5_shap_additivity_tol == 1e-4
    assert cfg.targets.o6_reproducibility_tol == 1e-6
    assert cfg.targets.o7_cli_api_parity_tol == 1e-9
    assert cfg.targets.o7_warm_score_p95_max_ms == 200.0
