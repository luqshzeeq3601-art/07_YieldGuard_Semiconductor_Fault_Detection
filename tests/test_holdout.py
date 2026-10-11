"""Tests for holdout evaluation gate, one-shot protection, and prediction hashing."""

from pathlib import Path

import pandas as pd
import pytest

from yieldguard.artifacts import freeze_model_bundle
from yieldguard.evaluate import run_holdout_evaluation


def test_holdout_evaluation_permission_gate(tmp_path: Path) -> None:
    # Must raise PermissionError when allow_heldout_evaluation is False
    with pytest.raises(PermissionError, match="Holdout evaluation is sealed"):
        run_holdout_evaluation(allow_heldout_evaluation=False)


def test_holdout_evaluation_hashes_predictions_before_labels_open_and_refuses_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project_root = Path(__file__).resolve().parent.parent

    # Freeze bundle to temporary models dir
    bundle_dir = tmp_path / "models" / "v0.1.0"
    monkeypatch.chdir(tmp_path)

    # Copy staging and data
    tmp_stg = tmp_path / "models" / "staging"
    tmp_stg.mkdir(parents=True, exist_ok=True)
    for f in (project_root / "models" / "staging").iterdir():
        if f.is_file():
            (tmp_stg / f.name).write_bytes(f.read_bytes())

    tmp_rep = tmp_path / "reports"
    tmp_rep.mkdir(parents=True, exist_ok=True)
    if (project_root / "reports" / "validation_metrics.json").exists():
        (tmp_rep / "validation_metrics.json").write_bytes(
            (project_root / "reports" / "validation_metrics.json").read_bytes()
        )

    tmp_data = tmp_path / "data" / "processed"
    tmp_data.mkdir(parents=True, exist_ok=True)
    for f in (project_root / "data" / "processed").iterdir():
        if f.is_file():
            (tmp_data / f.name).write_bytes(f.read_bytes())

    runs_p = (tmp_data / "runs.parquet").as_posix()
    dev_p = (tmp_data / "labels_dev.parquet").as_posix()
    hold_p = (tmp_data / "labels_holdout.parquet").as_posix()
    split_p = (tmp_data / "split_manifest.json").as_posix()
    source_p = (project_root / "data" / "source_manifest.json").as_posix()
    stg_p = tmp_stg.as_posix()
    rep_p = tmp_rep.as_posix()

    cfg_path = tmp_path / "config.yaml"
    cfg_content = f"""
project:
  name: YieldGuard
  seed: 42
data:
  dataset_id: SECOM
  runs_parquet: {runs_p}
  labels_dev_parquet: {dev_p}
  labels_holdout_parquet: {hold_p}
  split_manifest: {split_p}
  source_manifest: {source_p}
models:
  staging_dir: {stg_p}
  reports_dir: {rep_p}
policy:
  alarm_budgets_per_100: [2.0, 5.0, 10.0]
  false_alarm_rates_per_100: [2.0, 5.0, 10.0]
  n_bootstraps: 100
  block_bootstrap_length: 20
targets:
  o1_pr_auc_min_prevalence_multiple: 2.0
  o2_min_recall_at_5_far: 0.20
  o3_alarm_rate_min_per_100: 2.0
  o3_alarm_rate_max_per_100: 8.0
"""
    cfg_path.write_text(cfg_content, encoding="utf-8")

    freeze_model_bundle(config_path=cfg_path, version="v0.1.0")

    preds_csv = tmp_rep / "holdout_predictions.csv"
    orig_read_parquet = pd.read_parquet

    def _verified_read_parquet(path, *args, **kwargs):
        p_str = str(path)
        if "labels_holdout" in p_str.lower():
            # Verify predictions CSV already exists and is non-empty before labels are read
            assert preds_csv.exists(), "holdout_predictions.csv MUST exist before opening holdout labels"
            assert preds_csv.stat().st_size > 0, "holdout_predictions.csv MUST be populated before opening labels"
        return orig_read_parquet(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", _verified_read_parquet)

    # First run: should succeed
    rep = run_holdout_evaluation(
        config_path=cfg_path,
        bundle_dir=bundle_dir,
        allow_heldout_evaluation=True,
    )
    assert rep is not None
    assert "predictions_sha256_before_labels_opened" in rep
    assert (tmp_rep / "holdout_metrics.json").exists()
    assert "O1_ranking_pr_auc" in rep["objectives"]
    assert "O2_recall_at_5_far" in rep["objectives"]
    assert "O3_alarm_budget_stability" in rep["objectives"]
    assert "O4_spc_vs_ml_comparison" in rep["objectives"]

    # Second run: must strictly refuse to overwrite
    with pytest.raises(FileExistsError, match="strictly refuses to overwrite an existing evaluation record"):
        run_holdout_evaluation(
            config_path=cfg_path,
            bundle_dir=bundle_dir,
            allow_heldout_evaluation=True,
        )
