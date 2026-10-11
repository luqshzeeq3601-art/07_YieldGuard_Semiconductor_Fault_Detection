"""Tests for Isolation Forest (E07): passing-run fitting, synthetic outlier ranking, and determinism."""

from __future__ import annotations

import numpy as np
import pandas as pd

from yieldguard.anomaly import fit_isolation_forest, score_isolation_forest


def test_isolation_forest_fitted_on_passing_runs_only() -> None:
    """Altering failing runs in training data does not alter the fitted IsolationForest."""
    rng = np.random.default_rng(42)
    cols = [f"f{i:03d}" for i in range(12)]
    x = rng.normal(0.0, 1.0, size=(100, 12))
    labels = np.zeros(100, dtype=np.int64)
    labels[85:] = 1

    df1 = pd.DataFrame(x, columns=cols)
    df2 = df1.copy()
    df2.iloc[85:, :] = 9999.0  # Massive corruption of failing runs

    m1 = fit_isolation_forest(df1, labels, n_estimators=200, seed=42, fit_on_passing_only=True)
    m2 = fit_isolation_forest(df2, labels, n_estimators=200, seed=42, fit_on_passing_only=True)

    assert m1.n_training_runs_used == 85
    assert m2.n_training_runs_used == 85

    s1 = score_isolation_forest(m1, df1.iloc[:20])
    s2 = score_isolation_forest(m2, df1.iloc[:20])
    np.testing.assert_allclose(s1, s2, rtol=0.0, atol=1e-12)


def test_isolation_forest_higher_score_for_synthetic_outlier_and_deterministic() -> None:
    """A synthetic outlier receives a strictly higher score than in-control runs and is deterministic."""
    rng = np.random.default_rng(7)
    cols = [f"f{i:03d}" for i in range(10)]
    x = rng.normal(0.0, 1.0, size=(120, 10))
    labels = np.zeros(120, dtype=np.int64)
    df_train = pd.DataFrame(x, columns=cols)

    m_a = fit_isolation_forest(df_train, labels, n_estimators=200, seed=42)
    m_b = fit_isolation_forest(df_train, labels, n_estimators=200, seed=42)

    test_df = pd.DataFrame(
        [
            np.zeros(10),  # normal center run
            np.full(10, 6.0),  # strong synthetic outlier
        ],
        columns=cols,
    )

    scores_a = score_isolation_forest(m_a, test_df)
    scores_b = score_isolation_forest(m_b, test_df)

    np.testing.assert_array_equal(scores_a, scores_b)
    assert scores_a[1] > scores_a[0]
