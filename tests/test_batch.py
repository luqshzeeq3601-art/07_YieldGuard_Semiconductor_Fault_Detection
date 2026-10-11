"""Tests for CLI batch scoring and explanation pipeline."""

from pathlib import Path

import pandas as pd
import pytest

from yieldguard.cli import main
from yieldguard.data import SENSOR_COLUMNS
from yieldguard.synthetic import generate_synthetic_secom


def test_batch_scoring_and_explain_pipeline(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parent.parent
    bundle_dir = (project_root / "models" / "v0.1.0").as_posix()

    # Generate synthetic input batch of 10 runs
    matrix, _, timestamps, _ = generate_synthetic_secom(n_runs=10, seed=123)
    syn_df = pd.DataFrame(matrix, columns=SENSOR_COLUMNS)
    syn_df.insert(0, "run_id", list(range(10)))
    syn_df.insert(1, "timestamp", timestamps)
    # Shuffle order so we can verify chronological ordering in output
    syn_df = syn_df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    input_csv = tmp_path / "batch_input.csv"
    output_csv = tmp_path / "batch_scored.csv"
    syn_df.to_csv(input_csv, index=False)

    # Run CLI score command
    exit_code = main(["score", "--bundle", bundle_dir, "--input", str(input_csv), "--output", str(output_csv)])
    assert exit_code == 0
    assert output_csv.exists()

    scored = pd.read_csv(output_csv)
    assert len(scored) == 10
    # Verify chronological ordering by timestamp
    assert scored["timestamp"].is_monotonic_increasing

    # Verify score and flag columns
    expected_cols = [
        "e01_score", "e02_score", "e03_score", "t2", "spe", "mspc_index",
        "iforest_score", "xgb_score", "primary_detector", "alarm",
        "alarm_e02_b5", "alarm_budget_per_100", "n_missing",
    ]
    for col in expected_cols:
        assert col in scored.columns

    # Test explain command on input batch
    explain_json = tmp_path / "rankings.json"
    exp_exit = main([
        "explain",
        "--bundle", bundle_dir,
        "--input", str(input_csv),
        "--output", str(explain_json),
    ])
    assert exp_exit == 0
    assert explain_json.exists()


def test_batch_scoring_fails_on_invalid_run(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parent.parent
    bundle_dir = (project_root / "models" / "v0.1.0").as_posix()

    matrix, _, timestamps, _ = generate_synthetic_secom(n_runs=5, seed=456)
    syn_df = pd.DataFrame(matrix, columns=SENSOR_COLUMNS)
    syn_df["run_id"] = list(range(5))
    syn_df["timestamp"] = timestamps
    # Corrupt one sensor value with a string
    syn_df.loc[2, "f050"] = "corrupted_sensor_reading"

    bad_csv = tmp_path / "bad_batch.csv"
    out_csv = tmp_path / "out.csv"
    syn_df.to_csv(bad_csv, index=False)

    with pytest.raises(ValueError, match="Invalid non-numeric value"):
        from yieldguard.predict import score_csv_file
        score_csv_file(bundle_dir, bad_csv, out_csv)


def test_batch_scoring_fails_on_missing_sensors(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parent.parent
    bundle_dir = (project_root / "models" / "v0.1.0").as_posix()

    df = pd.DataFrame({"run_id": [1, 2], "timestamp": ["2026-01-01", "2026-01-02"]})
    bad_csv = tmp_path / "missing_sensors.csv"
    out_csv = tmp_path / "out.csv"
    df.to_csv(bad_csv, index=False)

    with pytest.raises(ValueError, match="missing .* required sensor columns"):
        from yieldguard.predict import score_csv_file
        score_csv_file(bundle_dir, bad_csv, out_csv)
