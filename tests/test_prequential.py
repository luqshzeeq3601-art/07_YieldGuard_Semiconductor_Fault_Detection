"""Unit tests for YieldGuard v0.2 prequential drift-adaptive protocol engine."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from yieldguard.prequential import (
    RobustPreprocessor,
    compute_composite_top3_score,
    partition_prequential_blocks,
    run_prequential_evaluation,
)


def _make_synthetic_drifting_df(n_runs: int = 240, n_sensors: int = 12, seed: int = 42) -> tuple[pd.DataFrame, list[str]]:
    rng = np.random.default_rng(seed)
    sensor_cols = [f"s{j:03d}" for j in range(n_sensors)]
    t_idx = np.arange(n_runs, dtype=np.float64)
    # Linear drift + fault spikes
    labels = np.where(rng.uniform(0, 1, size=n_runs) < 0.12, 1, -1)
    # Ensure every 60-run block has at least 3 fails
    for b_start in range(0, n_runs, 60):
        labels[b_start : b_start + 3] = 1

    X = rng.normal(0.0, 1.0, size=(n_runs, n_sensors))
    # Add drift over time to all sensors
    X += (t_idx[:, None] / 60.0) * 1.5
    # Inject strong fault signal in first 3 sensors
    fail_mask = labels == 1
    X[fail_mask, 0] += 3.5
    X[fail_mask, 1] -= 3.0
    X[fail_mask, 2] += 2.8

    # Make sensor s010 constant and s011 collinear with s000
    X[:, 10] = 5.0
    X[:, 11] = X[:, 0] * 1.001 + rng.normal(0.0, 0.001, size=n_runs)

    df = pd.DataFrame(X, columns=sensor_cols)
    df.insert(0, "run_id", [f"syn_{i:04d}" for i in range(n_runs)])
    df.insert(
        1,
        "timestamp",
        pd.date_range("2025-01-01", periods=n_runs, freq="h").strftime("%Y-%m-%d %H:%M:%S"),
    )
    df["label"] = labels
    return df, sensor_cols


def test_robust_preprocessor_median_mad_and_pruning() -> None:
    df, sensor_cols = _make_synthetic_drifting_df(n_runs=120, n_sensors=12, seed=42)
    pre = RobustPreprocessor().fit(df, sensor_cols)
    assert "s010" in pre.dropped_constant
    assert "s011" in pre.dropped_collinear or "s000" in pre.dropped_collinear
    scaled = pre.transform_scaled(df)
    assert list(scaled.columns) == pre.retained_sensors
    # Check medians of transformed features are near 0
    for col in pre.retained_sensors:
        assert abs(float(scaled[col].median())) < 1e-6


def test_composite_top3_score_breaks_ties_deterministically() -> None:
    # Two runs share identical max |Z| = 4.0, but Run 1 has higher 2nd and 3rd largest |Z|
    abs_z = np.array(
        [
            [4.0, 0.5, 0.2, 0.1],
            [4.0, 2.5, 1.8, 0.1],
            [3.9, 3.8, 3.7, 3.6],
        ],
        dtype=np.float64,
    )
    scores = compute_composite_top3_score(abs_z)
    assert scores[1] > scores[0]
    # Primary order by max |Z| is preserved when max |Z| differs by >= 0.1
    assert scores[0] > scores[2]


def test_prequential_temporal_causality_and_one_shot_lock(tmp_path: Path) -> None:
    df, sensor_cols = _make_synthetic_drifting_df(n_runs=240, n_sensors=12, seed=42)
    blocks = partition_prequential_blocks(df, n_blocks=4)
    assert len(blocks) == 4

    pred_p1 = tmp_path / "preds1.csv"
    out_p1 = tmp_path / "metrics1.json"

    rep1 = run_prequential_evaluation(
        df=df,
        sensor_cols=sensor_cols,
        dataset_id="SYNTHETIC",
        n_blocks=4,
        rolling_window_runs=120,
        calibration_window_runs=60,
        bootstrap_iterations=50,
        seed=42,
        predictions_csv_path=pred_p1,
        output_json_path=out_p1,
    )
    assert out_p1.exists()
    assert pred_p1.exists()
    assert rep1["predictions_sha256"] is not None
    assert "E09_v2" in rep1["detectors"]

    # O5_v2 input: calibration-window alarm rate is recorded per block and matches its count
    for block in rep1["per_block_summaries"]:
        for det_metrics in block["detectors"].values():
            expected = 100.0 * det_metrics["calibration_alarms_b5"] / block["n_calibration_runs"]
            assert det_metrics["calibration_alarms_per_100_b5"] == pytest.approx(expected)

    # Second call with overwrite=False must refuse to overwrite
    with pytest.raises(FileExistsError):
        run_prequential_evaluation(
            df=df,
            sensor_cols=sensor_cols,
            dataset_id="SYNTHETIC",
            n_blocks=4,
            rolling_window_runs=120,
            calibration_window_runs=60,
            bootstrap_iterations=50,
            seed=42,
            output_json_path=out_p1,
            overwrite=False,
        )

    # Perturb the final block (B4, indices 180..239) and verify earlier blocks (B2, B3) are identical
    df_perturbed = df.copy()
    df_perturbed.loc[180:, "s000"] += 50.0
    pred_p2 = tmp_path / "preds2.csv"
    run_prequential_evaluation(
        df=df_perturbed,
        sensor_cols=sensor_cols,
        dataset_id="SYNTHETIC",
        n_blocks=4,
        rolling_window_runs=120,
        calibration_window_runs=60,
        bootstrap_iterations=50,
        seed=42,
        predictions_csv_path=pred_p2,
    )
    p1_df = pd.read_csv(pred_p1)
    p2_df = pd.read_csv(pred_p2)
    early1 = p1_df[p1_df["prequential_block"].isin(["B2", "B3"])]
    early2 = p2_df[p2_df["prequential_block"].isin(["B2", "B3"])]
    pd.testing.assert_frame_equal(early1, early2)
