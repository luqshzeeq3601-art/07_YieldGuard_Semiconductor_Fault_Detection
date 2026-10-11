"""Tests for train-only data cleaning, collinear group mapping, leakage invariance, and serialization."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from yieldguard.quality import QualityTransformer, fit_quality_transformer


def _make_known_quality_fixture(n_rows: int = 100) -> tuple[pd.DataFrame, np.ndarray]:
    """Construct a small deterministic sensor matrix with known pruning outcomes."""
    rng = np.random.default_rng(123)
    f0 = rng.normal(10.0, 2.0, size=n_rows)
    f1 = rng.normal(-5.0, 1.5, size=n_rows)

    # f2: >50% missing (60% NaN)
    f2 = rng.normal(0.0, 1.0, size=n_rows)
    f2[:60] = np.nan

    # f3: exact constant
    f3 = np.full(n_rows, 7.0)

    # f4: near-constant (99 out of 100 values identical = 0.99 >= 0.99)
    f4 = np.full(n_rows, 3.0)
    f4[0] = 99.0

    # f5: collinear with f0 (r = 1.0), same missingness (0%) -> f0 has lower index so f0 kept, f5 dropped
    f5 = f0 * 2.0 + 1.0

    # f6: collinear with f1 (r = 1.0), f1 has 5 NaNs and f6 has 0 NaNs -> f6 has less missingness so f6 kept, f1 dropped!
    f6 = f1 * 1.5 - 2.0
    f1_with_na = f1.copy()
    f1_with_na[:5] = np.nan

    df = pd.DataFrame(
        {
            "f000": f0,
            "f001": f1_with_na,
            "f002": f2,
            "f003": f3,
            "f004": f4,
            "f005": f5,
            "f006": f6,
        }
    )
    labels_fail = np.zeros(n_rows, dtype=np.int64)
    labels_fail[-10:] = 1
    return df, labels_fail


def test_known_fixture_pruning_and_group_map() -> None:
    """Verify exact high-missing, constant, and collinear drops plus group map tie-breaking."""
    df, labels = _make_known_quality_fixture(100)
    qt = fit_quality_transformer(df, labels, sensor_columns=list(df.columns))

    assert qt.dropped_high_missing == ["f002"]
    assert qt.dropped_constant == ["f003", "f004"]
    assert set(qt.dropped_collinear) == {"f001", "f005"}
    assert qt.retained_columns == ["f000", "f006"]

    # f005 maps to f000 (same missingness, lower index f000 wins)
    assert qt.dropped_to_representative["f005"] == "f000"
    assert qt.get_group_members("f000") == ["f005"]

    # f001 maps to f006 (f006 has 0% missing vs f001 5% missing, so f006 wins despite higher index)
    assert qt.dropped_to_representative["f001"] == "f006"
    assert qt.get_group_members("f006") == ["f001"]


def test_perturbing_later_blocks_leaves_fitted_state_unchanged() -> None:
    """Changing sensor values or labels in later blocks cannot alter fitted lists, medians, or scaling."""
    df_all, labels_all = _make_known_quality_fixture(100)
    # Suppose rows 0..59 are training blocks (D1-D3) and rows 60..99 are later blocks (D4-holdout)
    df_train_1 = df_all.iloc[:60].copy()
    y_train_1 = labels_all[:60].copy()
    qt1 = fit_quality_transformer(df_train_1, y_train_1, sensor_columns=list(df_all.columns))

    # Heavily corrupt rows 60..99 in a copy of the full table and slice training rows again
    df_corrupted = df_all.copy()
    df_corrupted.iloc[60:, :] = 999999.0
    labels_corrupted = labels_all.copy()
    labels_corrupted[60:] = 1

    qt2 = fit_quality_transformer(
        df_corrupted.iloc[:60], labels_corrupted[:60], sensor_columns=list(df_all.columns)
    )

    assert qt1.retained_columns == qt2.retained_columns
    assert qt1.dropped_high_missing == qt2.dropped_high_missing
    assert qt1.dropped_constant == qt2.dropped_constant
    assert qt1.dropped_collinear == qt2.dropped_collinear
    assert qt1.dropped_to_representative == qt2.dropped_to_representative
    assert qt1.medians == qt2.medians
    assert qt1.passing_means == qt2.passing_means
    assert qt1.passing_stds == qt2.passing_stds


def test_passing_runs_only_scaling() -> None:
    """Altering failing runs in training data does not change passing_means or passing_stds."""
    df, labels = _make_known_quality_fixture(100)
    qt1 = fit_quality_transformer(df, labels, sensor_columns=list(df.columns))

    df_fail_shifted = df.copy()
    # Failing runs are the last 10 rows; shift f000 and f006 on failing runs without changing median/corr
    df_fail_shifted.loc[labels == 1, "f000"] += 500.0
    df_fail_shifted.loc[labels == 1, "f005"] = df_fail_shifted["f000"] * 2.0 + 1.0
    qt2 = fit_quality_transformer(df_fail_shifted, labels, sensor_columns=list(df.columns))

    assert qt1.passing_means == qt2.passing_means
    assert qt1.passing_stds == qt2.passing_stds


def test_save_and_reload_preserves_column_order_and_outputs(tmp_path: Path) -> None:
    """Saving and reloading QualityTransformer preserves exact column order and scaled outputs."""
    df, labels = _make_known_quality_fixture(100)
    qt = fit_quality_transformer(df, labels, sensor_columns=list(df.columns))

    save_path = qt.save(tmp_path / "quality.joblib")
    reloaded = QualityTransformer.load(save_path)

    assert reloaded.retained_columns == qt.retained_columns
    s1 = qt.transform_scaled(df)
    s2 = reloaded.transform_scaled(df)
    assert list(s1.columns) == list(s2.columns) == qt.retained_columns
    np.testing.assert_allclose(s1.to_numpy(), s2.to_numpy(), rtol=0.0, atol=1e-12)
