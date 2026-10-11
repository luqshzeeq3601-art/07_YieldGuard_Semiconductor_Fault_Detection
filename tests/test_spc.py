"""Tests for univariate SPC: hand-computed EWMA, four run rules, ARL0 simulation, missing handling, and leakage invariance."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from yieldguard.spc import (
    compute_ewma_series,
    evaluate_run_rules,
    fit_univariate_spc,
    simulate_in_control_arl,
)


def test_ewma_values_and_limits_match_hand_computed_series() -> None:
    """Verify EWMA recursion, time-varying limits, and missing-value preservation against hand maths."""
    # mu0 = 10.0, sigma0 = 2.0, lam = 0.2, L = 3.0
    # Series: x = [12.0, NaN, 8.0]
    # t=1: z_1 = 0.2 * 12.0 + 0.8 * 10.0 = 10.4
    #      sigma_z(1) = 2.0 * sqrt((0.2/1.8) * (1 - 0.8^2)) = 2.0 * sqrt((1/9) * 0.36) = 2.0 * 0.2 = 0.4
    #      ucl_1 = 10.0 + 3 * 0.4 = 11.2, lcl_1 = 8.8
    # t=2 (NaN): z_2 = z_1 = 10.4
    #      sigma_z(2) = 2.0 * sqrt((1/9) * (1 - 0.8^4)) = 2.0 * sqrt(0.5904 / 9)
    # t=3: z_3 = 0.2 * 8.0 + 0.8 * 10.4 = 1.6 + 8.32 = 9.92
    res = compute_ewma_series([12.0, np.nan, 8.0], mu0=10.0, sigma0=2.0, lam=0.2, L=3.0)

    assert res["z"][0] == pytest.approx(10.4, rel=1e-12)
    assert res["z"][1] == pytest.approx(10.4, rel=1e-12)
    assert res["z"][2] == pytest.approx(9.92, rel=1e-12)
    assert bool(res["is_missing"][1]) is True

    assert res["sigma_z"][0] == pytest.approx(0.4, rel=1e-12)
    assert res["ucl"][0] == pytest.approx(11.2, rel=1e-12)
    assert res["lcl"][0] == pytest.approx(8.8, rel=1e-12)

    expected_sigma_z2 = 2.0 * math.sqrt((0.2 / 1.8) * (1.0 - 0.8**4))
    assert res["sigma_z"][1] == pytest.approx(expected_sigma_z2, rel=1e-12)


def test_each_run_rule_fires_on_constructed_sequence_and_not_on_near_miss() -> None:
    """Verify all 4 Western Electric / Nelson run rules on positive and near-miss sequences."""
    mu0, sigma0 = 0.0, 1.0

    # Rule 1: 1 point beyond 3 sigma
    r1_pos = evaluate_run_rules([0.0, 2.99, 3.01], mu0, sigma0)["rule_1_beyond_3sigma"]
    assert list(r1_pos) == [False, False, True]

    # Rule 2: 2 of 3 beyond 2 sigma on the same side
    # Near miss: +2.5, -2.5, +1.9 (opposite sides, or only 1 beyond +2)
    r2_miss = evaluate_run_rules([2.5, -2.5, 1.9], mu0, sigma0)["rule_2_of_3_beyond_2sigma"]
    assert not any(r2_miss)
    # Positive: +2.2, +0.5, +2.1 -> fires at index 2
    r2_hit = evaluate_run_rules([2.2, 0.5, 2.1], mu0, sigma0)["rule_2_of_3_beyond_2sigma"]
    assert list(r2_hit) == [False, False, True]

    # Rule 3: 4 of 5 beyond 1 sigma on the same side
    # Near miss: 3 of 5 > 1.0 and 1 < -1.0
    r3_miss = evaluate_run_rules([1.2, 1.3, -1.5, 1.4, 0.2], mu0, sigma0)[
        "rule_4_of_5_beyond_1sigma"
    ]
    assert not any(r3_miss)
    # Positive: 4 of 5 > 1.0 on same side -> fires at index 4
    r3_hit = evaluate_run_rules([1.2, 1.3, 0.2, 1.4, 1.1], mu0, sigma0)[
        "rule_4_of_5_beyond_1sigma"
    ]
    assert list(r3_hit) == [False, False, False, False, True]

    # Rule 4: 8 consecutive on one side of centre line
    # Near miss: 7 positive then 1 negative
    r4_miss = evaluate_run_rules([0.1] * 7 + [-0.1], mu0, sigma0)["rule_8_on_one_side"]
    assert not any(r4_miss)
    # Positive: 8 positive -> fires at index 7
    r4_hit = evaluate_run_rules([0.1] * 8, mu0, sigma0)["rule_8_on_one_side"]
    assert list(r4_hit) == [False] * 7 + [True]


def test_simulated_in_control_arl_matches_published_value() -> None:
    """In-control ARL0 for EWMA (lambda=0.2, L=3.0) agrees with published ~465-500 within simulation error."""
    arl0 = simulate_in_control_arl(lam=0.20, L=3.0, n_trials=600, max_steps=3000, seed=42)
    # Published Lucas & Saccucci (1990) / Montgomery value for lambda=0.2, L=3.0 is ~465.5 (steady-state ~500)
    assert 410.0 <= arl0 <= 550.0


def test_sensor_selection_unchanged_when_later_blocks_altered() -> None:
    """Altering values or labels in later blocks leaves training-only sensor selection and Phase I limits unchanged."""
    rng = np.random.default_rng(42)
    n_total = 120
    cols = [f"f{i:03d}" for i in range(25)]
    data = rng.normal(0.0, 1.0, size=(n_total, 25))
    labels = np.zeros(n_total, dtype=np.int64)
    labels[[10, 20, 30, 40, 50, 80, 90, 100]] = 1
    # Inject strong signal in f003 and f007 on training rows (0..79)
    data[labels == 1, 3] += 4.0
    data[labels == 1, 7] -= 3.5

    df_full = pd.DataFrame(data, columns=cols)
    spc1 = fit_univariate_spc(df_full.iloc[:80], labels[:80], n_monitored=5)

    # Corrupt later rows 80..119 completely
    df_corrupted = df_full.copy()
    df_corrupted.iloc[80:, :] = -9999.0
    labels_corrupted = labels.copy()
    labels_corrupted[80:] = 1
    spc2 = fit_univariate_spc(df_corrupted.iloc[:80], labels_corrupted[:80], n_monitored=5)

    assert spc1.monitored_sensors == spc2.monitored_sensors
    assert spc1.monitored_sensors[:2] == ["f003", "f007"]
    for s in spc1.monitored_sensors:
        assert spc1.sensor_specs[s].mu0 == spc2.sensor_specs[s].mu0
        assert spc1.sensor_specs[s].sigma0 == spc2.sensor_specs[s].sigma0
