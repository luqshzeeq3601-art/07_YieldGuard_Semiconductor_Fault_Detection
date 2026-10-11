"""Tests for synthetic SECOM-like generator and loader compatibility."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from yieldguard.data import N_SENSORS, SENSOR_COLUMNS, load_raw_secom
from yieldguard.synthetic import generate_single_sensor_fault_matrix, generate_synthetic_secom


def test_synthetic_secom_loads_through_real_loader(tmp_path: Path) -> None:
    """Synthetic two-file generator writes valid files that load through `load_raw_secom`."""
    matrix, labels, timestamps, meta = generate_synthetic_secom(
        output_dir=tmp_path, n_runs=120, fail_rate=0.10, seed=42
    )
    assert matrix.shape == (120, N_SENSORS)
    assert len(labels) == 120
    assert len(timestamps) == 120
    assert len(meta.faulty_sensors) == 5
    assert len(meta.high_missing_columns) == 30
    assert len(meta.constant_columns) == 30
    assert len(meta.collinear_map) == 30

    loaded = load_raw_secom(tmp_path, verify_uci_counts=False)
    assert len(loaded.runs_df) == 120
    assert int(loaded.runs_df["label_fail"].sum()) == meta.n_fails
    assert all(col in loaded.runs_df.columns for col in SENSOR_COLUMNS)
    # Timestamp tie group was included and preserved
    assert loaded.quality_summary["timestamp_findings"]["duplicate_timestamp_runs"] >= 1


def test_synthetic_determinism() -> None:
    """Same seed produces identical synthetic sensor matrix and labels."""
    m1, l1, ts1, _ = generate_synthetic_secom(output_dir=None, n_runs=60, seed=99)
    m2, l2, ts2, _ = generate_synthetic_secom(output_dir=None, n_runs=60, seed=99)
    np.testing.assert_allclose(m1, m2, equal_nan=True)
    assert l1 == l2
    assert ts1 == ts2


def test_single_sensor_fault_generator() -> None:
    """Verify single-sensor 5-sigma fault injection helper."""
    mean = np.zeros(20)
    std = np.ones(20)
    runs, true_idx = generate_single_sensor_fault_matrix(
        mean, std, n_cases=40, shift_sigma=5.0, seed=42
    )
    assert runs.shape == (40, 20)
    assert len(true_idx) == 40
    for i, s_idx in enumerate(true_idx):
        assert abs(runs[i, s_idx]) == 5.0
