"""Synthetic SECOM-like dataset generator for unit tests, CI, dashboard demo, and O5 verification."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from yieldguard.data import DEFAULT_TIMESTAMP_FORMAT, N_SENSORS, SENSOR_COLUMNS


@dataclass
class SyntheticSecomMetadata:
    """Ground-truth structure of a generated synthetic SECOM-like dataset."""

    n_runs: int
    n_fails: int
    seed: int
    faulty_sensors: list[str]
    high_missing_columns: list[str]
    constant_columns: list[str]
    collinear_map: dict[str, str]
    drift_columns: list[str]
    single_sensor_fault_runs: dict[int, str]


def generate_synthetic_secom(
    output_dir: str | Path | None = None,
    n_runs: int = 240,
    fail_rate: float = 0.10,
    seed: int = 42,
    include_timestamp_ties: bool = True,
) -> tuple[np.ndarray, list[int], list[str], SyntheticSecomMetadata]:
    """Generate a synthetic 590-sensor SECOM-like dataset and optionally write raw files.

    Structure of the 590 columns (`f000`-`f589`):
    - `f000`-`f004`: known faulty sensors shifted on failing runs.
    - `f020`-`f024`: slow linear drift columns across run index.
    - `f000`-`f499`: latent-factor correlated normal sensors (10 latent factors) with ~1% missingness.
    - `f500`-`f529`: high-missingness columns (~75% missing, > 0.50 pruning threshold).
    - `f530`-`f559`: constant columns (100% identical value on non-missing entries).
    - `f560`-`f589`: collinear duplicates of `f000`-`f029` (r > 0.999, higher missingness so
      `f000`-`f029` are kept as representatives).
    """
    rng = np.random.default_rng(seed)

    faulty_indices = [0, 1, 2, 3, 4]
    faulty_sensors = [f"f{i:03d}" for i in faulty_indices]
    drift_indices = [20, 21, 22, 23, 24]
    drift_columns = [f"f{i:03d}" for i in drift_indices]
    high_missing_indices = list(range(500, 530))
    high_missing_columns = [f"f{i:03d}" for i in high_missing_indices]
    constant_indices = list(range(530, 560))
    constant_columns = [f"f{i:03d}" for i in constant_indices]
    collinear_map = {f"f{560 + k:03d}": f"f{k:03d}" for k in range(30)}

    # 1. Base latent-factor model for f000..f499
    n_latent = 10
    n_base = 500
    z = rng.normal(0.0, 1.0, size=(n_runs, n_latent))
    loadings = rng.normal(0.0, 0.4, size=(n_latent, n_base))
    noise = rng.normal(0.0, 0.8, size=(n_runs, n_base))
    base_data = z @ loadings + noise

    # Make f000..f029 well-separated (low cross-correlation < 0.5) so collinear map & O5 are unambiguous
    base_data[:, :30] = rng.normal(10.0, 2.0, size=(n_runs, 30))

    # 2. Add slow drift on f020..f024
    t_norm = np.linspace(0.0, 1.5, n_runs)[:, None]
    base_data[:, drift_indices] += t_norm * 2.0

    # 3. Assign failing runs across time so every chronological block has fails
    labels_raw = [-1] * n_runs
    single_sensor_fault_runs: dict[int, str] = {}
    if n_runs <= 2:
        fail_positions = np.array([], dtype=int)
    else:
        n_fails = max(1, min(n_runs - 1, round(n_runs * fail_rate)))
        start_pos = 1 if n_runs < 10 else 5
        end_pos = max(start_pos, n_runs - 1)
        fail_positions = np.linspace(start_pos, end_pos, n_fails, dtype=int)

    for k, r_idx in enumerate(fail_positions):
        labels_raw[int(r_idx)] = 1
        target_sensor_idx = faulty_indices[k % len(faulty_indices)]
        # Inject strong 5-sigma shift on the primary faulty sensor and moderate shift on others
        base_data[int(r_idx), target_sensor_idx] += 5.0 * 2.0
        for other_idx in faulty_indices:
            if other_idx != target_sensor_idx:
                base_data[int(r_idx), other_idx] += 2.5 * 2.0
        single_sensor_fault_runs[int(r_idx)] = f"f{target_sensor_idx:03d}"

    # Assemble full 590 matrix
    matrix = np.empty((n_runs, N_SENSORS), dtype=np.float64)
    matrix[:, :500] = base_data

    # Low random missingness (~1%) on f030..f499 (keep f000..f029 0% missing so representatives are clean)
    low_miss_mask = rng.random((n_runs, 470)) < 0.01
    matrix[:, 30:500][low_miss_mask] = np.nan

    # 4. High-missingness columns f500..f529 (~75% missing)
    high_miss_vals = rng.normal(0.0, 1.0, size=(n_runs, 30))
    high_miss_mask = rng.random((n_runs, 30)) < 0.75
    high_miss_vals[high_miss_mask] = np.nan
    matrix[:, 500:530] = high_miss_vals

    # 5. Constant columns f530..f559
    for k, col_idx in enumerate(constant_indices):
        matrix[:, col_idx] = float(k + 1)

    # 6. Collinear columns f560..f589 (near-exact copies of f000..f029 with 3% missingness)
    for k in range(30):
        col_copy = matrix[:, k].copy() * 1.0001 + 0.001
        # Mark ~3% missing so f000..f029 (0% missing) always win the representative tie-break
        miss_rows = rng.choice(n_runs, size=max(1, n_runs // 30), replace=False)
        col_copy[miss_rows] = np.nan
        matrix[:, 560 + k] = col_copy

    # 7. Generate timestamps (with optional tie groups)
    start_dt = datetime(2008, 7, 19, 12, 0, 0)
    timestamps: list[str] = []
    curr_dt = start_dt
    for i in range(n_runs):
        if include_timestamp_ties and i in (25, 90):
            # Same timestamp as previous run to test tie-handling
            pass
        else:
            curr_dt = curr_dt + timedelta(minutes=30)
        timestamps.append(curr_dt.strftime(DEFAULT_TIMESTAMP_FORMAT))

    meta = SyntheticSecomMetadata(
        n_runs=n_runs,
        n_fails=int(sum(1 for x in labels_raw if x == 1)),
        seed=seed,
        faulty_sensors=faulty_sensors,
        high_missing_columns=high_missing_columns,
        constant_columns=constant_columns,
        collinear_map=collinear_map,
        drift_columns=drift_columns,
        single_sensor_fault_runs=single_sensor_fault_runs,
    )

    if output_dir is not None:
        write_synthetic_secom_files(Path(output_dir), matrix, labels_raw, timestamps)

    return matrix, labels_raw, timestamps, meta


def write_synthetic_secom_files(
    output_dir: Path,
    matrix: np.ndarray,
    labels_raw: list[int],
    timestamps: list[str],
) -> tuple[Path, Path]:
    """Write `secom.data` and `secom_labels.data` in the exact UCI two-file format."""
    output_dir.mkdir(parents=True, exist_ok=True)
    data_path = output_dir / "secom.data"
    labels_path = output_dir / "secom_labels.data"

    data_lines: list[str] = []
    for row in matrix:
        tokens = ["NaN" if np.isnan(v) else f"{v:.6f}" for v in row]
        data_lines.append(" ".join(tokens))
    data_path.write_text("\n".join(data_lines) + "\n", encoding="utf-8")

    label_lines = [f'{lbl} "{ts}"' for lbl, ts in zip(labels_raw, timestamps, strict=True)]
    labels_path.write_text("\n".join(label_lines) + "\n", encoding="utf-8")

    return data_path, labels_path


def generate_single_sensor_fault_matrix(
    reference_mean: np.ndarray,
    reference_std: np.ndarray,
    n_cases: int = 50,
    shift_sigma: float = 5.0,
    seed: int = 42,
) -> tuple[np.ndarray, list[int]]:
    """Generate synthetic runs each having a single-sensor `shift_sigma` fault for O5 verification.

    Returns `(runs_matrix, true_sensor_indices)` in the retained sensor space.
    """
    rng = np.random.default_rng(seed)
    n_features = len(reference_mean)
    base = rng.normal(loc=reference_mean, scale=reference_std * 0.25, size=(n_cases, n_features))
    true_indices: list[int] = []
    for i in range(n_cases):
        s_idx = int(i % n_features)
        sign = 1.0 if (i % 2 == 0) else -1.0
        base[i, s_idx] = reference_mean[s_idx] + sign * shift_sigma * reference_std[s_idx]
        true_indices.append(s_idx)
    return base, true_indices


__all__ = [
    "SENSOR_COLUMNS",
    "SyntheticSecomMetadata",
    "generate_single_sensor_fault_matrix",
    "generate_synthetic_secom",
    "write_synthetic_secom_files",
]
