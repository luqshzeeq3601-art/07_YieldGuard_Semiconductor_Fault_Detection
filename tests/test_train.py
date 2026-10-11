"""Tests for XGBoost search budget, per-fold scale_pos_weight, forward-only folds, and MLflow logging."""

from __future__ import annotations

import json
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

from yieldguard.config import Config
from yieldguard.data import load_raw_secom
from yieldguard.splits import FORWARD_FOLDS, compute_chronological_split
from yieldguard.synthetic import generate_synthetic_secom
from yieldguard.train import (
    PREDECLARED_XGB_CONFIGS,
    XgbCandidateConfig,
    compute_scale_pos_weight,
    fit_xgb_candidate,
    run_training_experiments,
    validate_xgb_search_space,
)


def test_predeclared_xgb_search_space_within_bounds() -> None:
    """Verify <= 20 predeclared XGBoost configurations and all hyperparameter bounds."""
    assert 1 <= len(PREDECLARED_XGB_CONFIGS) <= 20
    validate_xgb_search_space(PREDECLARED_XGB_CONFIGS)


def test_scale_pos_weight_computed_per_training_fold() -> None:
    """Verify `scale_pos_weight = n_neg / n_pos` is computed dynamically per fold."""
    y_fold1 = np.array([0] * 90 + [1] * 10, dtype=np.int64)
    y_fold2 = np.array([0] * 140 + [1] * 20, dtype=np.int64)
    assert compute_scale_pos_weight(y_fold1) == 9.0
    assert compute_scale_pos_weight(y_fold2) == 7.0

    rng = np.random.default_rng(42)
    x_df = pd.DataFrame(rng.normal(size=(100, 8)), columns=[f"f{i:03d}" for i in range(8)])
    cand = PREDECLARED_XGB_CONFIGS[0]
    model, spw = fit_xgb_candidate(x_df, y_fold1, cand)
    assert spw == 9.0
    assert float(model.get_params()["scale_pos_weight"]) == 9.0


def test_forward_only_folds_and_mlflow_tracking_on_synthetic(tmp_path: Path) -> None:
    """Verify forward-only fold ordering, model_comparison.csv output, and MLflow hash tags on synthetic data."""
    # Verify forward-only structure of FORWARD_FOLDS
    block_order = {"D1": 1, "D2": 2, "D3": 3, "D4": 4, "D5": 5}
    for f in FORWARD_FOLDS:
        max_train = max(block_order[b] for b in f["train_blocks"])
        eval_idx = block_order[f["eval_block"]]
        assert max_train < eval_idx

    syn_dir = tmp_path / "raw_syn"
    generate_synthetic_secom(output_dir=syn_dir, n_runs=160, fail_rate=0.15, seed=42)
    df = load_raw_secom(syn_dir, verify_uci_counts=False).runs_df

    blocks, split_manifest = compute_chronological_split(df[["run_id", "timestamp"]])
    df_runs = df.drop(columns=["label_fail", "label_raw"]).copy()
    df_runs.insert(3, "block", blocks)

    proc_dir = tmp_path / "processed"
    proc_dir.mkdir(parents=True, exist_ok=True)
    runs_pq = proc_dir / "runs.parquet"
    labels_dev_pq = proc_dir / "labels_dev.parquet"
    split_json = proc_dir / "split_manifest.json"
    source_json = tmp_path / "source_manifest.json"

    df_runs.to_parquet(runs_pq, index=False)
    labels_df = pd.DataFrame(
        {
            "run_id": df["run_id"].values,
            "timestamp": df["timestamp"].values,
            "block": blocks,
            "label_fail": df["label_fail"].values,
        }
    )
    labels_df[labels_df["block"] != "holdout"].to_parquet(labels_dev_pq, index=False)
    split_json.write_text(json.dumps(split_manifest, indent=2), encoding="utf-8")
    source_json.write_text(json.dumps({"archive_sha256": "syn_hash_123"}), encoding="utf-8")

    staging_dir = tmp_path / "staging"
    reports_dir = tmp_path / "reports"
    db_path = tmp_path / "mlruns.db"

    cfg = Config()
    cfg.data.runs_parquet = str(runs_pq)
    cfg.data.labels_dev_parquet = str(labels_dev_pq)
    cfg.data.split_manifest = str(split_json)
    cfg.data.source_manifest = str(source_json)
    cfg.models.staging_dir = str(staging_dir)
    cfg.models.reports_dir = str(reports_dir)
    cfg.models.tracking_uri = f"sqlite:///{db_path.as_posix()}"
    cfg.anomaly.n_estimators = 25

    cfg_path = tmp_path / "config.yaml"
    import yaml

    cfg_path.write_text(yaml.safe_dump(cfg.model_dump()), encoding="utf-8")

    small_configs = (
        XgbCandidateConfig("xgb_c01", max_depth=1, learning_rate=0.05, n_estimators=100, subsample=0.8, colsample_bytree=0.5, min_child_weight=2, seed=42),
        XgbCandidateConfig("xgb_c02", max_depth=2, learning_rate=0.05, n_estimators=120, subsample=0.8, colsample_bytree=0.5, min_child_weight=2, seed=42),
    )

    summary = run_training_experiments(cfg_path, experiment="all", configs=small_configs)

    assert (staging_dir / "predeclared_xgb_configs.json").exists()
    assert (staging_dir / "dev_oof_scores.parquet").exists()
    assert (reports_dir / "model_comparison.csv").exists()

    comp_df = pd.read_csv(reports_dir / "model_comparison.csv")
    for det_id in ("E01", "E02", "E03", "E04", "E05", "E06", "E07", "E08"):
        assert len(comp_df[comp_df["detector_id"] == det_id]) == 3

    # Verify MLflow run carries data, split, and config hashes
    mlflow.set_tracking_uri(cfg.models.tracking_uri)
    run = mlflow.get_run(summary["mlflow_run_id"])
    assert run.data.tags["data_sha256"] == "syn_hash_123"
    assert run.data.tags["split_sha256"] == split_manifest["manifest_sha256"]
    assert len(run.data.tags["config_sha256"]) == 64
