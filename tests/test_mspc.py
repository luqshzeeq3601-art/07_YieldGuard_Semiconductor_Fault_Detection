"""Tests for multivariate SPC: direct linear-algebra match, contribution additivity, single-sensor fault ranking, and passing-run isolation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from yieldguard.mspc import (
    compute_t2_and_spe,
    fit_multivariate_spc,
    score_multivariate_spc,
)


def test_t2_and_spe_match_direct_linear_algebra_and_contributions_sum() -> None:
    """Verify T-squared and SPE against direct SVD/matrix linear algebra and check contribution sums."""
    rng = np.random.default_rng(42)
    x_pass = rng.normal(0.0, 1.0, size=(50, 6))
    cols = [f"f{i:03d}" for i in range(6)]
    df_pass = pd.DataFrame(x_pass, columns=cols)
    labels = np.zeros(50, dtype=np.int64)

    model = fit_multivariate_spc(
        df_pass,
        labels,
        enable_trimming=False,
        n_components_fixed=2,
    )

    # Direct linear algebra via numpy SVD on centered x_pass
    mu_direct = np.mean(x_pass, axis=0)
    xc = x_pass - mu_direct
    _, s_vals, vt = np.linalg.svd(xc, full_matrices=False)
    p_direct = vt[:2].T  # (6, 2)
    eig_direct = (s_vals[:2] ** 2) / (len(x_pass) - 1)

    t_direct = xc @ p_direct
    t2_direct = np.sum((t_direct**2) / eig_direct[None, :], axis=1)
    res_direct = xc - (t_direct @ p_direct.T)
    spe_direct = np.sum(res_direct**2, axis=1)

    scored = score_multivariate_spc(model, df_pass, include_contributions=True)
    np.testing.assert_allclose(scored["t2"], t2_direct, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(scored["spe"], spe_direct, rtol=1e-10, atol=1e-10)

    # Verify SPE contributions sum to SPE and T2 contributions sum to T2
    np.testing.assert_allclose(
        np.sum(scored["spe_contributions"], axis=1),
        scored["spe"],
        rtol=1e-10,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        np.sum(scored["t2_contributions"], axis=1),
        scored["t2"],
        rtol=1e-10,
        atol=1e-10,
    )


def test_synthetic_single_sensor_fault_raises_spe_or_t2_and_ranks_sensor_first() -> None:
    """Injecting a single-sensor fault raises SPE and T-squared and ranks that sensor #1."""
    rng = np.random.default_rng(101)
    n_runs, n_cols = 120, 15
    z = rng.normal(0.0, 1.0, size=(n_runs, 3))
    w = rng.normal(0.0, 0.7, size=(3, n_cols))
    x = z @ w + rng.normal(0.0, 0.2, size=(n_runs, n_cols))
    cols = [f"f{i:03d}" for i in range(n_cols)]
    df_train = pd.DataFrame(x, columns=cols)
    labels = np.zeros(n_runs, dtype=np.int64)

    model = fit_multivariate_spc(df_train, labels, variance_explained_target=0.80)

    # Create a clean run at the Phase I mean and a faulty run with a 5-sigma shift on sensor f008
    clean_run = pd.DataFrame([model.mean_vector], columns=cols)
    fault_run = clean_run.copy()
    fault_run["f008"] = fault_run["f008"] + 6.0

    clean_res = score_multivariate_spc(model, clean_run, include_contributions=True)
    fault_res = score_multivariate_spc(model, fault_run, include_contributions=True)

    assert fault_res["spe"][0] > clean_res["spe"][0]
    assert fault_res["t2"][0] > clean_res["t2"][0]
    assert fault_res["mspc_index"][0] > clean_res["mspc_index"][0]

    top_spe_idx = int(np.argmax(fault_res["spe_contributions"][0]))
    top_t2_idx = int(np.argmax(fault_res["t2_contributions"][0]))
    assert cols[top_spe_idx] == "f008"
    assert cols[top_t2_idx] == "f008"


def test_pca_fitted_on_passing_training_runs_only() -> None:
    """Altering failing runs in training data leaves the fitted PCA model and limits identical."""
    rng = np.random.default_rng(55)
    cols = [f"f{i:03d}" for i in range(10)]
    x = rng.normal(0.0, 1.0, size=(100, 10))
    labels = np.zeros(100, dtype=np.int64)
    labels[80:] = 1  # 20 failing runs

    df1 = pd.DataFrame(x, columns=cols)
    m1 = fit_multivariate_spc(df1, labels)

    df2 = df1.copy()
    df2.iloc[80:, :] = 12345.0  # Massive corruption of failing runs
    m2 = fit_multivariate_spc(df2, labels)

    assert m1.n_components == m2.n_components
    assert np.isclose(m1.t2_limit_99, m2.t2_limit_99, rtol=0.0, atol=1e-13)
    assert np.isclose(m1.spe_limit_99, m2.spe_limit_99, rtol=0.0, atol=1e-13)
    np.testing.assert_allclose(m1.mean_vector, m2.mean_vector, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(m1.loadings, m2.loadings, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(m1.eigenvalues, m2.eigenvalues, rtol=0.0, atol=1e-12)
    assert compute_t2_and_spe(x[:5], m1.mean_vector, m1.loadings, m1.eigenvalues)[0] is not None

