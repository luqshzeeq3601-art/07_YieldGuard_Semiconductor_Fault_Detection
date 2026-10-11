"""Raw SECOM data validation, chronological loader, and CSV batch input validation."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

N_SENSORS = 590
SENSOR_COLUMNS: list[str] = [f"f{i:03d}" for i in range(N_SENSORS)]
DEFAULT_TIMESTAMP_FORMAT = "%d/%m/%Y %H:%M:%S"
EXPECTED_SECOM_ROWS = 1567
EXPECTED_SECOM_FAILS = 104

_LABEL_LINE_RE = re.compile(r'^\s*(-?\d+)\s+"([^"]+)"\s*$')


@dataclass
class LoadedSecomDataset:
    """Validated SECOM dataset with ordered runs, labels, and raw quality summary."""

    runs_df: pd.DataFrame
    quality_summary: dict[str, Any]


def parse_secom_labels_file(
    labels_path: str | Path,
    timestamp_format: str = DEFAULT_TIMESTAMP_FORMAT,
) -> tuple[list[int], list[str], list[datetime]]:
    """Parse and validate `secom_labels.data` lines.

    Each line must have `<label> "<timestamp>"` where label is in {-1, 1}
    and timestamp parses with `timestamp_format`.
    """
    path = Path(labels_path)
    if not path.exists():
        raise FileNotFoundError(f"Labels file not found: {path}")

    raw_lines = path.read_text(encoding="utf-8").splitlines()
    if not raw_lines:
        raise ValueError(f"Labels file is empty: {path}")

    labels_raw: list[int] = []
    ts_strings: list[str] = []
    ts_parsed: list[datetime] = []

    for line_idx, line in enumerate(raw_lines):
        m = _LABEL_LINE_RE.match(line)
        if not m:
            raise ValueError(
                f"Malformed label line at row {line_idx} in {path.name}: {line!r}. "
                'Expected `<label> "<timestamp>"`.'
            )
        label_str, ts_str = m.group(1), m.group(2)
        try:
            label_val = int(label_str)
        except ValueError as e:
            raise ValueError(f"Invalid integer label at row {line_idx}: {label_str!r}") from e

        if label_val not in (-1, 1):
            raise ValueError(
                f"Invalid label value at row {line_idx}: {label_val}. Must be in {{-1, 1}}."
            )

        try:
            dt = datetime.strptime(ts_str, timestamp_format)
        except ValueError as e:
            raise ValueError(
                f"Unparseable timestamp at row {line_idx}: {ts_str!r} "
                f"(expected format {timestamp_format!r})."
            ) from e

        labels_raw.append(label_val)
        ts_strings.append(ts_str)
        ts_parsed.append(dt)

    return labels_raw, ts_strings, ts_parsed


def parse_secom_data_file(data_path: str | Path) -> np.ndarray:
    """Parse and validate `secom.data` lines.

    Each line must have exactly 590 space-separated tokens, each finite float or NaN.
    Infinities and non-numeric strings raise ValueError.
    """
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Sensor data file not found: {path}")

    raw_lines = path.read_text(encoding="utf-8").splitlines()
    if not raw_lines:
        raise ValueError(f"Sensor data file is empty: {path}")

    n_rows = len(raw_lines)
    matrix = np.empty((n_rows, N_SENSORS), dtype=np.float64)

    for row_idx, line in enumerate(raw_lines):
        tokens = line.strip().split()
        if len(tokens) != N_SENSORS:
            raise ValueError(
                f"Wrong field count at row {row_idx} in {path.name}: "
                f"expected {N_SENSORS} fields, got {len(tokens)}."
            )
        for col_idx, tok in enumerate(tokens):
            if tok == "NaN" or tok == "nan":
                matrix[row_idx, col_idx] = np.nan
            else:
                try:
                    val = float(tok)
                except ValueError as e:
                    raise ValueError(
                        f"Non-numeric sensor token at row {row_idx}, col f{col_idx:03d}: {tok!r}"
                    ) from e
                if math.isinf(val):
                    raise ValueError(
                        f"Infinite sensor value not allowed at row {row_idx}, col f{col_idx:03d}: {tok!r}"
                    )
                matrix[row_idx, col_idx] = val

    return matrix


def load_raw_secom(
    raw_dir: str | Path = "data/raw/secom",
    timestamp_format: str = DEFAULT_TIMESTAMP_FORMAT,
    verify_uci_counts: bool = True,
    sort_chronologically: bool = True,
) -> LoadedSecomDataset:
    """Load, validate, and chronologically order SECOM raw files from `raw_dir`.

    Ordering is by `(timestamp, run_id)` where `run_id` is the 0-based file row index.
    """
    raw_path = Path(raw_dir)
    data_file = raw_path / "secom.data"
    labels_file = raw_path / "secom_labels.data"

    labels_raw, ts_strings, ts_parsed = parse_secom_labels_file(
        labels_file, timestamp_format=timestamp_format
    )
    sensor_matrix = parse_secom_data_file(data_file)

    if len(labels_raw) != sensor_matrix.shape[0]:
        raise ValueError(
            f"Row-count mismatch between secom.data ({sensor_matrix.shape[0]} rows) "
            f"and secom_labels.data ({len(labels_raw)} rows)."
        )

    n_runs = len(labels_raw)
    labels_arr = np.asarray(labels_raw, dtype=np.int64)
    label_fail = (labels_arr == 1).astype(np.int64)
    n_fails = int(np.sum(label_fail))

    if verify_uci_counts and (n_runs != EXPECTED_SECOM_ROWS or n_fails != EXPECTED_SECOM_FAILS):
        raise ValueError(
            f"UCI SECOM count mismatch: expected {EXPECTED_SECOM_ROWS} runs and "
            f"{EXPECTED_SECOM_FAILS} fails, got {n_runs} runs and {n_fails} fails."
        )

    # Check timestamp properties in original file order
    inversions = sum(1 for i in range(1, n_runs) if ts_parsed[i] < ts_parsed[i - 1])
    is_file_sorted = inversions == 0
    unique_ts_count = len(set(ts_parsed))
    duplicate_run_count = int(n_runs - unique_ts_count)

    ts_series_orig = pd.Series(ts_parsed)
    ts_counts = ts_series_orig.value_counts()
    duplicate_timestamp_groups = int((ts_counts > 1).sum())
    max_tie_group_size = int(ts_counts.max()) if n_runs > 0 else 0

    df_sensors = pd.DataFrame(sensor_matrix, columns=SENSOR_COLUMNS)
    meta_df = pd.DataFrame(
        {
            "run_id": np.arange(n_runs, dtype=np.int64),
            "timestamp": pd.to_datetime(ts_parsed),
            "timestamp_str": ts_strings,
            "label_raw": labels_arr,
            "label_fail": label_fail,
        }
    )
    full_df = pd.concat([meta_df, df_sensors], axis=1)

    if sort_chronologically:
        full_df = full_df.sort_values(by=["timestamp", "run_id"], kind="mergesort").reset_index(
            drop=True
        )

    missing_frac_per_col = full_df[SENSOR_COLUMNS].isna().mean()
    n_constant_raw = 0
    for col in SENSOR_COLUMNS:
        non_na = full_df[col].dropna()
        if len(non_na) <= 1 or non_na.nunique() <= 1:
            n_constant_raw += 1

    quality_summary: dict[str, Any] = {
        "dataset_id": "SECOM" if verify_uci_counts else "SYNTHETIC_SECOM",
        "n_runs": int(n_runs),
        "n_sensors": int(N_SENSORS),
        "n_total_fails_published": int(n_fails),
        "timestamp_findings": {
            "declared_format": timestamp_format,
            "min_timestamp": min(ts_parsed).isoformat() if ts_parsed else None,
            "max_timestamp": max(ts_parsed).isoformat() if ts_parsed else None,
            "is_file_already_sorted": bool(is_file_sorted),
            "file_order_inversions": int(inversions),
            "unique_timestamps": int(unique_ts_count),
            "duplicate_timestamp_runs": int(duplicate_run_count),
            "duplicate_timestamp_groups": int(duplicate_timestamp_groups),
            "max_tie_group_size": int(max_tie_group_size),
        },
        "raw_missingness_summary": {
            "overall_missing_fraction": float(full_df[SENSOR_COLUMNS].isna().values.mean()),
            "columns_with_any_missing": int((missing_frac_per_col > 0.0).sum()),
            "columns_gt_50pct_missing": int((missing_frac_per_col > 0.50).sum()),
            "columns_le_5pct_missing": int((missing_frac_per_col <= 0.05).sum()),
            "raw_single_value_columns": int(n_constant_raw),
        },
    }

    return LoadedSecomDataset(runs_df=full_df, quality_summary=quality_summary)


def write_raw_quality_report(
    quality_summary: dict[str, Any],
    report_path: str | Path = "reports/data_quality.json",
) -> Path:
    """Write or update `reports/data_quality.json` with raw quality findings."""
    path = Path(report_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing: dict[str, Any] = {}
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing = {}
    existing.update(quality_summary)
    path.write_text(json.dumps(existing, indent=2) + "\n", encoding="utf-8")
    return path


def validate_single_run_sensors(sensors: list[Any]) -> np.ndarray:
    """Validate a single run's 590 sensor values (reject wrong length, strings, bools, inf)."""
    if not isinstance(sensors, (list, tuple)):
        raise ValueError("Sensors must be a list of 590 numeric values or nulls.")
    if len(sensors) != N_SENSORS:
        raise ValueError(f"Expected exactly {N_SENSORS} sensor values, got {len(sensors)}.")

    out = np.empty(N_SENSORS, dtype=np.float64)
    for idx, val in enumerate(sensors):
        if val is None:
            out[idx] = np.nan
        elif isinstance(val, bool) or not isinstance(val, (int, float, np.number)):
            raise ValueError(
                f"Invalid type at sensor f{idx:03d}: {type(val).__name__} ({val!r}). "
                "Booleans and strings are not permitted."
            )
        else:
            fval = float(val)
            if math.isinf(fval):
                raise ValueError(f"Infinite value not permitted at sensor f{idx:03d}.")
            out[idx] = fval
    return out


def load_and_validate_runs_csv(
    csv_path: str | Path,
    timestamp_format: str = DEFAULT_TIMESTAMP_FORMAT,
) -> pd.DataFrame:
    """Load and strictly validate a batch CSV of runs for scoring or explanation.

    If any row has missing required columns, non-numeric strings, booleans, or infinities,
    raises ValueError naming the offending run or column so the whole file fails cleanly.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Input CSV not found: {path}")

    # Read as object first so strings/booleans/infs are caught explicitly
    raw_df = pd.read_csv(path, dtype=object, keep_default_na=True)
    if len(raw_df) == 0:
        raise ValueError(f"Input CSV is empty: {path}")

    missing_cols = [c for c in SENSOR_COLUMNS if c not in raw_df.columns]
    if missing_cols:
        raise ValueError(
            f"Input CSV missing {len(missing_cols)} required sensor columns "
            f"(e.g. {missing_cols[:5]})."
        )

    extra_sensor_like = [
        c
        for c in raw_df.columns
        if c not in SENSOR_COLUMNS
        and c not in ("run_id", "timestamp", "timestamp_str", "block", "label_fail", "label_raw")
    ]
    if extra_sensor_like:
        raise ValueError(f"Unknown columns in input CSV: {extra_sensor_like[:5]}")

    n_rows = len(raw_df)
    sensor_mat = np.empty((n_rows, N_SENSORS), dtype=np.float64)

    for row_idx in range(n_rows):
        run_label = raw_df["run_id"].iloc[row_idx] if "run_id" in raw_df.columns else row_idx
        for col_idx, col_name in enumerate(SENSOR_COLUMNS):
            cell = raw_df[col_name].iloc[row_idx]
            if cell is None or (isinstance(cell, float) and math.isnan(cell)) or pd.isna(cell):
                sensor_mat[row_idx, col_idx] = np.nan
                continue
            cell_str = str(cell).strip()
            if cell_str == "" or cell_str.lower() == "nan":
                sensor_mat[row_idx, col_idx] = np.nan
                continue
            if cell_str.lower() in ("true", "false"):
                raise ValueError(
                    f"Invalid boolean value in run {run_label}, sensor {col_name}: {cell!r}"
                )
            try:
                fval = float(cell_str)
            except ValueError as e:
                raise ValueError(
                    f"Invalid non-numeric value in run {run_label}, sensor {col_name}: {cell!r}"
                ) from e
            if math.isinf(fval):
                raise ValueError(
                    f"Infinite value in run {run_label}, sensor {col_name}: {cell!r}"
                )
            sensor_mat[row_idx, col_idx] = fval

    out_df = pd.DataFrame(sensor_mat, columns=SENSOR_COLUMNS)
    if "run_id" in raw_df.columns:
        out_df.insert(0, "run_id", raw_df["run_id"].astype(str).values)
    else:
        out_df.insert(0, "run_id", [str(i) for i in range(n_rows)])

    if "timestamp" in raw_df.columns:
        ts_list = []
        for row_idx, val in enumerate(raw_df["timestamp"]):
            run_label = out_df["run_id"].iloc[row_idx]
            if pd.isna(val) or str(val).strip() == "":
                raise ValueError(f"Missing timestamp in run {run_label} at row {row_idx}")
            ts_str = str(val).strip()
            parsed_dt = None
            for fmt in (timestamp_format, "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
                try:
                    parsed_dt = datetime.strptime(ts_str, fmt)
                    break
                except ValueError:
                    continue
            if parsed_dt is None:
                try:
                    parsed_dt = pd.to_datetime(ts_str).to_pydatetime()
                except Exception as e:
                    raise ValueError(
                        f"Unparseable timestamp in run {run_label} at row {row_idx}: {ts_str!r}"
                    ) from e
            ts_list.append(parsed_dt)
        out_df.insert(1, "timestamp", pd.to_datetime(ts_list))
        out_df.insert(2, "timestamp_str", [str(v).strip() for v in raw_df["timestamp"]])
        out_df["_orig_order"] = np.arange(n_rows)
        out_df = (
            out_df.sort_values(by=["timestamp", "_orig_order"], kind="mergesort")
            .drop(columns=["_orig_order"])
            .reset_index(drop=True)
        )

    return out_df
