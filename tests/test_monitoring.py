"""Tests for data drift detection, missingness monitoring, and drift controls."""

import json
from pathlib import Path

import pandas as pd

from yieldguard.artifacts import load_model_bundle
from yieldguard.monitoring import (
    compute_missing_rate_table,
    run_drift_analysis,
    run_drift_controls,
)


def test_compute_missing_rate_table() -> None:
    ref_df = pd.DataFrame(
        {
            "s1": [1.0, 2.0, None, 4.0],
            "s2": [None, None, 1.0, 2.0],
        }
    )
    curr_df = pd.DataFrame(
        {
            "s1": [1.0, 2.0, 3.0, 4.0],
            "s2": [None, None, None, None],
        }
    )
    table = compute_missing_rate_table(ref_df, curr_df, ["s1", "s2"])
    assert len(table) == 2

    # s1: ref missing 1/4 = 0.25, curr missing 0/4 = 0.0 -> diff = -0.25
    s1_row = table[table["column"] == "s1"].iloc[0]
    assert s1_row["ref_missing_rate"] == 0.25
    assert s1_row["current_missing_rate"] == 0.0
    assert s1_row["missing_rate_difference"] == -0.25

    # s2: ref missing 2/4 = 0.50, curr missing 4/4 = 1.0 -> diff = +0.50
    s2_row = table[table["column"] == "s2"].iloc[0]
    assert s2_row["ref_missing_rate"] == 0.50
    assert s2_row["current_missing_rate"] == 1.0
    assert s2_row["missing_rate_difference"] == 0.50


def test_run_drift_controls_all_pass(tmp_path: Path) -> None:
    bundle = load_model_bundle("models/v0.1.0")
    controls_dir = tmp_path / "drift_controls"

    res = run_drift_controls(bundle, output_dir=controls_dir)

    assert res["all_controls_passed"] is True
    assert res["controls"]["no_change_control"]["passed"] is True
    assert res["controls"]["shifted_sensor_control"]["passed"] is True
    assert res["controls"]["insufficient_data_control"]["passed"] is True

    # Output JSON written
    out_json = controls_dir / "control_results.json"
    assert out_json.exists()
    saved = json.loads(out_json.read_text(encoding="utf-8"))
    assert saved["all_controls_passed"] is True


def test_insufficient_data_returns_early(tmp_path: Path) -> None:
    bundle = load_model_bundle("models/v0.1.0")
    small_df = pd.DataFrame({"f000": [1.0] * 20})  # 20 runs < 100

    summary = run_drift_analysis(bundle, small_df, output_dir=tmp_path, min_runs_required=100)
    assert summary.status == "insufficient_data"
    assert summary.current_count == 20
    assert summary.dataset_drift is False
    assert (tmp_path / "drift_summary.json").exists()
