"""Tests for sensor ranking (T11): SHAP additivity within 1e-4, O5 synthetic top-3 hit rate >= 90%, group members, and raw sensor names."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from yieldguard.data import SENSOR_COLUMNS
from yieldguard.explain import (
    compute_xgb_shap_raw_margin,
    evaluate_o5_synthetic_single_sensor_faults,
    rank_sensors_for_run,
)
from yieldguard.mspc import fit_multivariate_spc
from yieldguard.quality import fit_quality_transformer


def test_shap_additivity_within_1e4_with_missing_values() -> None:
    """Verify SHAP TreeExplainer raw-margin attributions sum to model raw margin within 1e-4."""
    rng = np.random.default_rng(42)
    cols = list(SENSOR_COLUMNS[:20])
    x = rng.normal(0.0, 1.0, size=(120, 20))
    x[rng.random(size=(120, 20)) < 0.08] = np.nan
    y = ((np.nan_to_num(x[:, 0]) + np.nan_to_num(x[:, 1])) > 1.0).astype(np.int64)

    df = pd.DataFrame(x, columns=cols)
    clf = xgb.XGBClassifier(
        max_depth=2,
        learning_rate=0.05,
        n_estimators=100,
        subsample=0.8,
        colsample_bytree=0.6,
        random_state=42,
        n_jobs=1,
    )
    clf.fit(df.iloc[:90], y[:90])

    res = compute_xgb_shap_raw_margin(clf, df.iloc[90:], background_df=df.iloc[:90])
    assert float(res["max_additivity_error"]) < 1e-4


def test_o5_synthetic_single_sensor_5sigma_fault_in_top3_ge_90pct(tmp_path: Path) -> None:
    """Verify synthetic 5-sigma single-sensor faults rank in top-3 MSPC contributions >= 90% of the time."""
    out_json = tmp_path / "o5_synthetic_verification.json"
    rep = evaluate_o5_synthetic_single_sensor_faults(
        n_train_runs=200,
        n_test_trials=100,
        shift_sigma=5.0,
        seed=42,
        output_path=out_json,
    )
    assert out_json.exists()
    assert rep["top3_mspc_hit_rate"] >= 0.90
    assert rep["max_shap_additivity_error"] <= 1e-4
    assert rep["verdict"] == "PASS"


def test_rank_sensors_attaches_group_members_and_maps_to_raw_sensor_names() -> None:
    """Verify output sensor names map to f000..f589 and collinear group members are attached."""
    rng = np.random.default_rng(123)
    cols = list(SENSOR_COLUMNS[:15])
    x = rng.normal(0.0, 1.0, size=(100, 15))
    # Make f010 a near-exact collinear copy of f000 with slightly more missingness
    x[:, 10] = x[:, 0] * 1.0001
    x[0, 10] = np.nan

    y = np.zeros(100, dtype=np.int64)
    y[-15:] = 1
    x[-15:, 0] += 5.0
    x[-15:, 10] += 5.0

    df = pd.DataFrame(x, columns=cols)
    df.insert(0, "run_id", np.arange(100))

    qt = fit_quality_transformer(df[cols], y, sensor_columns=cols)
    assert "f000" in qt.retained_columns
    assert "f010" in qt.get_group_members("f000")

    mspc = fit_multivariate_spc(qt.transform_scaled(df), y, variance_explained_target=0.80)
    clf = xgb.XGBClassifier(max_depth=2, learning_rate=0.05, n_estimators=60, random_state=42, n_jobs=1)
    clf.fit(qt.transform_retained(df), y)

    fault_row = df.iloc[-1].copy()
    fault_row["f000"] = 8.0
    fault_row["f010"] = 8.0

    ranking = rank_sensors_for_run(
        fault_row,
        quality=qt,
        mspc_model=mspc,
        xgb_model_or_booster=clf,
        top_k=5,
    )

    sensor_set = set(SENSOR_COLUMNS)
    for item in ranking["mspc_top_sensors"]:
        assert item["sensor"] in sensor_set
        assert isinstance(item["group_members"], list)
    for item in ranking["xgb_shap_top_sensors"]:
        assert item["sensor"] in sensor_set
        assert isinstance(item["group_members"], list)

    assert ranking["mspc_top_sensors"][0]["sensor"] == "f000"
    assert "f010" in ranking["mspc_top_sensors"][0]["group_members"]
    assert ranking["xgb_summary"]["shap_additivity_error"] < 1e-4
