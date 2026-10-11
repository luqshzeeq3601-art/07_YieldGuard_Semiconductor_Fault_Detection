"""Tests for chronological splitting, tie-boundary protection, and holdout label isolation."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from yieldguard.splits import (
    DEV_BLOCKS,
    compute_chronological_split,
    load_development_labels,
    prepare_dataset,
)
from yieldguard.synthetic import generate_synthetic_secom


def test_split_manifest_identical_under_label_permutation() -> None:
    """Permuting labels leaves the chronological split manifest and its SHA-256 identical."""
    rng = np.random.default_rng(42)
    n_runs = 200
    base_ts = datetime(2008, 7, 19, 12, 0, 0)
    timestamps = [base_ts + timedelta(minutes=15 * i) for i in range(n_runs)]
    df1 = pd.DataFrame(
        {
            "run_id": np.arange(n_runs),
            "timestamp": timestamps,
            "label_fail": rng.integers(0, 2, size=n_runs),
        }
    )
    df2 = df1.copy()
    df2["label_fail"] = rng.permutation(df1["label_fail"].values)

    blocks1, manifest1 = compute_chronological_split(df1)
    blocks2, manifest2 = compute_chronological_split(df2)

    assert blocks1 == blocks2
    assert manifest1 == manifest2
    assert manifest1["manifest_sha256"] == manifest2["manifest_sha256"]


def test_no_timestamp_straddles_boundary() -> None:
    """When identical timestamps cross a nominal boundary, the boundary shifts later."""
    n_runs = 160
    # Nominal dev = 120, blocks of 24: boundaries at 24, 48, 72, 96, 120
    base_ts = datetime(2008, 7, 19, 12, 0, 0)
    timestamps = [base_ts + timedelta(minutes=10 * i) for i in range(n_runs)]
    # Force a tie straddling nominal boundary 24 (indices 23, 24, 25 share timestamp)
    timestamps[24] = timestamps[23]
    timestamps[25] = timestamps[23]
    # Force a tie straddling nominal holdout boundary 120 (indices 119, 120 share timestamp)
    timestamps[120] = timestamps[119]

    df = pd.DataFrame({"run_id": np.arange(n_runs), "timestamp": timestamps})
    block_labels, manifest = compute_chronological_split(df, dev_fraction=0.75, n_dev_blocks=5)

    assert len(block_labels) == n_runs
    # Verify D1 shifted by +2 (ends at 26 instead of 24)
    assert manifest["blocks"]["D1"]["end_index_exclusive"] == 26
    assert manifest["blocks"]["D1"]["tie_shift"] == 2
    # Verify D5 shifted by +1 (ends at 121 instead of 120)
    assert manifest["blocks"]["D5"]["end_index_exclusive"] == 121
    assert manifest["blocks"]["D5"]["tie_shift"] == 1

    # Verify no timestamp straddles any boundary
    all_blocks = [*DEV_BLOCKS, "holdout"]
    for i in range(len(all_blocks) - 1):
        left_b = manifest["blocks"][all_blocks[i]]
        right_b = manifest["blocks"][all_blocks[i + 1]]
        assert left_b["end_index_exclusive"] == right_b["start_index"]
        assert left_b["max_timestamp"] < right_b["min_timestamp"]


def test_training_loader_raises_on_labels_holdout(tmp_path: Path) -> None:
    """Training/development label loader raises PermissionError on holdout labels."""
    holdout_path = tmp_path / "labels_holdout.parquet"
    df_hold = pd.DataFrame(
        {"run_id": [1, 2], "timestamp": pd.to_datetime(["2008-08-01", "2008-08-02"]), "block": ["holdout", "holdout"], "label_fail": [0, 1]}
    )
    df_hold.to_parquet(holdout_path, index=False)

    with pytest.raises(PermissionError, match="Label isolation violation"):
        load_development_labels(holdout_path)

    # Also test sneaky filename containing holdout block rows
    sneaky_path = tmp_path / "labels_dev.parquet"
    df_hold.to_parquet(sneaky_path, index=False)
    with pytest.raises(PermissionError, match="Label isolation violation"):
        load_development_labels(sneaky_path)


def test_prepare_dataset_on_synthetic(tmp_path: Path) -> None:
    """End-to-end prepare on synthetic data writes unlabeled runs.parquet and sealed labels."""
    raw_dir = tmp_path / "raw"
    proc_dir = tmp_path / "processed"
    rep_dir = tmp_path / "reports"
    generate_synthetic_secom(output_dir=raw_dir, n_runs=200, fail_rate=0.10, seed=42)

    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(
        f"""
data:
  dataset_id: "SYNTHETIC"
  raw_dir: "{raw_dir.as_posix()}"
  processed_dir: "{proc_dir.as_posix()}"
  quality_report: "{(rep_dir / 'data_quality.json').as_posix()}"
  split_manifest: "{(proc_dir / 'split_manifest.json').as_posix()}"
  runs_parquet: "{(proc_dir / 'runs.parquet').as_posix()}"
  labels_dev_parquet: "{(proc_dir / 'labels_dev.parquet').as_posix()}"
  labels_holdout_parquet: "{(proc_dir / 'labels_holdout.parquet').as_posix()}"
  verify_uci_counts: false
""",
        encoding="utf-8",
    )

    manifest = prepare_dataset(cfg_file)
    assert manifest["n_total_runs"] == 200
    runs_df = pd.read_parquet(proc_dir / "runs.parquet")
    assert "label_fail" not in runs_df.columns
    assert "label_raw" not in runs_df.columns

    dev_labels = load_development_labels(proc_dir / "labels_dev.parquet")
    assert set(dev_labels["block"].unique()).issubset(set(DEV_BLOCKS))


@pytest.mark.integration
def test_real_secom_chronological_split() -> None:
    """Verify real SECOM split has 1,175 dev runs (5 x 235) and 392 holdout runs with no straddling timestamps."""
    manifest_path = Path("data/processed/split_manifest.json")
    if not manifest_path.exists():
        pytest.skip("Real SECOM split_manifest.json not yet generated.")

    import json

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["n_total_runs"] == 1567
    assert manifest["n_holdout_runs"] == 392
    assert manifest["n_dev_runs"] == 1175
    for b in DEV_BLOCKS:
        assert manifest["blocks"][b]["n_runs"] == 235

    all_blocks = [*DEV_BLOCKS, "holdout"]
    for i in range(len(all_blocks) - 1):
        left_b = manifest["blocks"][all_blocks[i]]
        right_b = manifest["blocks"][all_blocks[i + 1]]
        assert left_b["end_index_exclusive"] == right_b["start_index"]
        assert left_b["max_timestamp"] < right_b["min_timestamp"]
