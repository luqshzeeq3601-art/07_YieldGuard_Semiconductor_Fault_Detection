"""Tests proving D5 and holdout cannot affect XGBoost configuration choice and verifying primary detector selection."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import yaml

from yieldguard.config import Config
from yieldguard.data import load_raw_secom
from yieldguard.evaluate import run_development_evaluation
from yieldguard.splits import compute_chronological_split
from yieldguard.synthetic import generate_synthetic_secom
from yieldguard.train import (
    XgbCandidateConfig,
    run_training_experiments,
    select_xgb_config_from_inner_folds,
    select_xgb_config_on_dev_blocks,
)


def test_one_se_rule_prefers_shallower_fewer_trees() -> None:
    """Within 1 standard error of the highest mean inner-fold PR-AUC, prefer shallower depth and fewer trees."""
    configs = (
        XgbCandidateConfig("c_deep", max_depth=3, learning_rate=0.05, n_estimators=300, subsample=0.8, colsample_bytree=0.5, min_child_weight=3, seed=42),
        XgbCandidateConfig("c_shallow_more_trees", max_depth=1, learning_rate=0.05, n_estimators=250, subsample=0.8, colsample_bytree=0.5, min_child_weight=3, seed=42),
        XgbCandidateConfig("c_shallow_fewer_trees", max_depth=1, learning_rate=0.05, n_estimators=120, subsample=0.8, colsample_bytree=0.5, min_child_weight=5, seed=42),
    )
    # c_deep has mean 0.30, SE = std([0.26, 0.34], ddof=1)/sqrt(2) = 0.04 -> 1-SE threshold = 0.26
    # Both shallow configs have mean 0.27 >= 0.26, so c_shallow_fewer_trees wins!
    inner_pr_aucs = {
        "c_deep": [0.26, 0.34],
        "c_shallow_more_trees": [0.27, 0.27],
        "c_shallow_fewer_trees": [0.27, 0.27],
    }
    chosen, report = select_xgb_config_from_inner_folds(inner_pr_aucs, configs=configs)
    assert chosen.config_id == "c_shallow_fewer_trees"
    assert report["best_mean_config_id"] == "c_deep"
    assert report["selected_config_id"] == "c_shallow_fewer_trees"


def test_d5_and_holdout_cannot_affect_xgb_configuration_choice(tmp_path) -> None:
    """Altering D5 and holdout sensor values and labels leaves XGBoost configuration selection identical."""
    syn_dir = tmp_path / "syn"
    generate_synthetic_secom(output_dir=syn_dir, n_runs=160, fail_rate=0.15, seed=99)
    df = load_raw_secom(syn_dir, verify_uci_counts=False).runs_df
    blocks, _ = compute_chronological_split(df[["run_id", "timestamp"]])

    runs_1 = df.drop(columns=["label_fail", "label_raw"]).copy()
    runs_1.insert(3, "block", blocks)
    labels_1 = pd.DataFrame(
        {
            "run_id": df["run_id"].values,
            "timestamp": df["timestamp"].values,
            "block": blocks,
            "label_fail": df["label_fail"].values,
        }
    )
    labels_dev_1 = labels_1[labels_1["block"] != "holdout"].reset_index(drop=True)

    # Create a heavily corrupted copy where D5 and holdout sensors and labels are inverted/corrupted
    runs_2 = runs_1.copy()
    sensor_cols = [c for c in runs_2.columns if c.startswith("f")]
    d5_or_holdout = runs_2["block"].isin(["D5", "holdout"])
    runs_2.loc[d5_or_holdout, sensor_cols] = -99999.0

    labels_dev_2 = labels_dev_1.copy()
    labels_dev_2.loc[labels_dev_2["block"] == "D5", "label_fail"] = (
        1 - labels_dev_2.loc[labels_dev_2["block"] == "D5", "label_fail"]
    )

    configs = (
        XgbCandidateConfig("xgb_c01", max_depth=1, learning_rate=0.05, n_estimators=100, subsample=0.8, colsample_bytree=0.5, min_child_weight=2, seed=42),
        XgbCandidateConfig("xgb_c02", max_depth=2, learning_rate=0.08, n_estimators=150, subsample=0.8, colsample_bytree=0.5, min_child_weight=3, seed=42),
        XgbCandidateConfig("xgb_c03", max_depth=3, learning_rate=0.05, n_estimators=200, subsample=0.9, colsample_bytree=0.7, min_child_weight=5, seed=42),
    )

    cfg = Config()
    chosen_1, rep_1 = select_xgb_config_on_dev_blocks(runs_1, labels_dev_1, cfg, configs=configs)
    chosen_2, rep_2 = select_xgb_config_on_dev_blocks(runs_2, labels_dev_2, cfg, configs=configs)

    assert chosen_1 == chosen_2
    assert rep_1["candidates"] == rep_2["candidates"]
    for c1, c2 in zip(rep_1["candidates"], rep_2["candidates"], strict=True):
        np.testing.assert_allclose(c1["inner_fold_pr_aucs"], c2["inner_fold_pr_aucs"], rtol=0.0, atol=1e-12)


def test_development_evaluation_never_opens_holdout_labels(tmp_path, monkeypatch) -> None:
    """File-access guard test proving training and dev evaluation never open labels_holdout.parquet."""
    syn_dir = tmp_path / "raw_syn"
    generate_synthetic_secom(output_dir=syn_dir, n_runs=160, fail_rate=0.15, seed=42)
    df = load_raw_secom(syn_dir, verify_uci_counts=False).runs_df
    blocks, split_manifest = compute_chronological_split(df[["run_id", "timestamp"]])

    proc_dir = tmp_path / "processed"
    proc_dir.mkdir(parents=True, exist_ok=True)
    runs_pq = proc_dir / "runs.parquet"
    labels_dev_pq = proc_dir / "labels_dev.parquet"
    labels_holdout_pq = proc_dir / "labels_holdout.parquet"
    split_json = proc_dir / "split_manifest.json"
    source_json = tmp_path / "source_manifest.json"

    df_runs = df.drop(columns=["label_fail", "label_raw"]).copy()
    df_runs.insert(3, "block", blocks)
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
    labels_df[labels_df["block"] == "holdout"].to_parquet(labels_holdout_pq, index=False)
    split_json.write_text(json.dumps(split_manifest, indent=2), encoding="utf-8")
    source_json.write_text(json.dumps({"archive_sha256": "syn_hash"}), encoding="utf-8")

    cfg = Config()
    cfg.data.runs_parquet = str(runs_pq)
    cfg.data.labels_dev_parquet = str(labels_dev_pq)
    cfg.data.labels_holdout_parquet = str(labels_holdout_pq)
    cfg.data.split_manifest = str(split_json)
    cfg.data.source_manifest = str(source_json)
    cfg.models.staging_dir = str(tmp_path / "staging")
    cfg.models.reports_dir = str(tmp_path / "reports")
    cfg.models.tracking_uri = f"sqlite:///{(tmp_path / 'mlruns.db').as_posix()}"
    cfg.anomaly.n_estimators = 20
    cfg.policy.n_bootstraps = 200

    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg.model_dump()), encoding="utf-8")

    orig_read_parquet = pd.read_parquet
    accessed_paths: list[str] = []

    def _guarded_read_parquet(path, *args, **kwargs):
        p_str = str(path)
        accessed_paths.append(p_str)
        if "labels_holdout" in p_str.lower():
            raise AssertionError(f"Forbidden access to holdout labels: {p_str}")
        return orig_read_parquet(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", _guarded_read_parquet)

    small_configs = (
        XgbCandidateConfig("xgb_c01", max_depth=1, learning_rate=0.05, n_estimators=100, subsample=0.8, colsample_bytree=0.5, min_child_weight=2, seed=42),
    )
    run_training_experiments(cfg_path, experiment="all", configs=small_configs)
    val_rep = run_development_evaluation(cfg_path)

    assert val_rep["holdout_labels_opened"] is False
    assert not any("labels_holdout" in p.lower() for p in accessed_paths)

