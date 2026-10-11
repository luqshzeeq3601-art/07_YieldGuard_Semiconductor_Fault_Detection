"""Unit tests for Phase 2 exploratory diagnostics on synthetic data."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from yieldguard.data import SENSOR_COLUMNS
from yieldguard.diagnostics import (
    compute_wald_wolfowitz_runs_test,
    run_secom_phase2_diagnostics,
)
from yieldguard.quality import fit_quality_transformer
from yieldguard.spc import fit_univariate_spc
from yieldguard.synthetic import generate_synthetic_secom


def _make_synthetic_runs_and_labels(n_runs: int = 180, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    matrix, labels_raw, timestamps, _ = generate_synthetic_secom(n_runs=n_runs, seed=seed)
    runs_df = pd.DataFrame(matrix, columns=SENSOR_COLUMNS)
    runs_df.insert(0, "run_id", list(range(n_runs)))
    runs_df.insert(1, "timestamp", timestamps)
    blocks = ["D1", "D2", "D3", "D4", "D5", "holdout"]
    block_size = n_runs // len(blocks)
    runs_df["block"] = [blocks[min(i // block_size, len(blocks) - 1)] for i in range(n_runs)]
    labels_df = pd.DataFrame(
        {
            "run_id": runs_df["run_id"],
            "label_fail": [1 if y == 1 else 0 for y in labels_raw],
        }
    )
    return runs_df, labels_df


def test_wald_wolfowitz_runs_test_clustered_vs_alternating() -> None:
    clustered = np.array([1] * 15 + [0] * 85, dtype=np.int64)
    res_clust = compute_wald_wolfowitz_runs_test(clustered)
    assert res_clust["observed_runs"] == 2
    assert res_clust["z_statistic"] is not None and res_clust["z_statistic"] < -2.0
    assert res_clust["p_value"] is not None and res_clust["p_value"] < 0.05


def test_phase2_diagnostics_end_to_end_on_synthetic(tmp_path: Path) -> None:
    runs_df, labels_df = _make_synthetic_runs_and_labels(n_runs=180, seed=42)
    # Introduce quantized ties on f000 to verify tie decomposition
    runs_df["f000"] = np.where(runs_df.index % 3 == 0, 100.0, 5.0)

    train_mask = runs_df["block"].isin(["D1", "D2", "D3", "D4"])
    qt = fit_quality_transformer(
        runs_df.loc[train_mask, SENSOR_COLUMNS],
        labels_df.loc[train_mask, "label_fail"],
    )
    spc_model = fit_univariate_spc(
        qt.transform_retained(runs_df.loc[train_mask]),
        labels_df.loc[train_mask, "label_fail"].to_numpy(),
        group_map=qt.representative_to_members,
    )

    out_json = tmp_path / "diag.json"
    report = run_secom_phase2_diagnostics(
        runs_df=runs_df,
        labels_df=labels_df,
        retained_columns=qt.retained_columns[:30],
        spc_model=spc_model,
        thresholds_e02={"budget_2": 5.0, "budget_5": 3.5, "budget_10": 2.5},
        output_path=out_json,
        n_bootstraps=100,
        seed=42,
    )

    assert out_json.exists()
    assert report["evaluation_type"] == "exploratory_post_holdout"
    assert "D1" in report["step1_drift_and_fail_clustering"]["per_block_sensor_drift"]
    assert "holdout" in report["step1_drift_and_fail_clustering"]["per_block_sensor_drift"]

    oracle = report["step2_oracle_upper_bound"]
    assert oracle["best_oracle_model"] in oracle["models"]
    assert isinstance(oracle["oracle_beats_chance"], bool)

    ties = report["step3_e02_score_ties"]
    assert ties["all_runs"]["n_runs"] == 180
    assert "winning_sensor_counts" in ties["all_runs"]
