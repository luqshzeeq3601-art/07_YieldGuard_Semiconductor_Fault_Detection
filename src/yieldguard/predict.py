"""Inference and scoring pipeline using loaded model bundle."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb

from yieldguard.anomaly import score_isolation_forest
from yieldguard.artifacts import LoadedBundle
from yieldguard.data import SENSOR_COLUMNS
from yieldguard.explain import rank_sensors_for_run
from yieldguard.mspc import score_multivariate_spc
from yieldguard.spc import score_univariate_spc


def validate_sensors_input(
    df: pd.DataFrame,
    expected_sensor_columns: list[str] | None = None,
) -> None:
    """Validate that input dataframe contains expected sensor columns."""
    exp = expected_sensor_columns or SENSOR_COLUMNS
    missing = [c for c in exp if c not in df.columns]
    if missing:
        raise ValueError(f"Input data missing {len(missing)} expected sensor columns: {missing[:5]}...")


def score_runs_dataframe(
    bundle: LoadedBundle,
    runs_df: pd.DataFrame,
) -> pd.DataFrame:
    """Score a DataFrame of runs containing raw sensor columns (f000..f589).

    Produces out-of-sample scores for all 8 detectors (E01-E08), alarms at budgets 2, 5, 10,
    and primary detector alarm flag at budget 5 per docs/04_TECHNICAL_DESIGN.md section 6.
    """
    validate_sensors_input(runs_df, bundle.feature_schema.get("raw_sensor_columns", SENSOR_COLUMNS))

    n_runs = len(runs_df)
    sensor_df = runs_df[SENSOR_COLUMNS].astype(np.float64).copy()

    # Per-run missing sensor counts
    n_missing = sensor_df.isna().sum(axis=1).to_numpy(dtype=np.int64)

    # 1. Cleaning and feature transforms
    retained_df = bundle.quality.transform_retained(sensor_df)
    scaled_df = bundle.quality.transform_scaled(sensor_df)

    # 2. Compute scores for all detectors
    # E01: Chance baseline (zeros)
    e01_scores = np.zeros(n_runs, dtype=np.float64)

    # E02 & E03: Univariate SPC (Individuals & EWMA)
    spc_res = score_univariate_spc(bundle.spc_model, retained_df)
    e02_scores = spc_res["e02_scores"]
    e03_scores = spc_res["e03_scores"]

    # E04, E05, E06: Multivariate SPC (T2, SPE, MSPC index)
    mspc_res = score_multivariate_spc(bundle.mspc_model, scaled_df)
    t2_scores = mspc_res["t2"]
    spe_scores = mspc_res["spe"]
    mspc_index_scores = mspc_res["mspc_index"]

    # E07: Isolation Forest
    iforest_scores = score_isolation_forest(bundle.iforest_model, scaled_df)

    # E08: XGBoost
    dmat = xgb.DMatrix(retained_df)
    xgb_scores = bundle.xgb_model.predict(dmat).astype(np.float64)

    scores_by_det = {
        "E01": e01_scores,
        "E02": e02_scores,
        "E03": e03_scores,
        "E04": t2_scores,
        "E05": spe_scores,
        "E06": mspc_index_scores,
        "E07": iforest_scores,
        "E08": xgb_scores,
    }

    # 3. Evaluate threshold alarms per detector
    thresholds = bundle.policy.get("thresholds", {})
    primary_id = bundle.policy.get("primary_detector", "E02")

    alarm_cols: dict[str, np.ndarray] = {}
    for det_id in ("E01", "E02", "E03", "E04", "E05", "E06", "E07", "E08"):
        s = scores_by_det[det_id]
        det_thresh = thresholds.get(det_id, {})
        for b in (2, 5, 10):
            tau = det_thresh.get(f"budget_{b}")
            if tau is not None:
                alarm_cols[f"alarm_{det_id.lower()}_b{b}"] = s > float(tau)
            else:
                alarm_cols[f"alarm_{det_id.lower()}_b{b}"] = np.zeros(n_runs, dtype=bool)

    primary_alarm = alarm_cols.get(f"alarm_{primary_id.lower()}_b5", np.zeros(n_runs, dtype=bool))

    # 4. Construct output DataFrame
    out_dict: dict[str, Any] = {}
    if "run_id" in runs_df.columns:
        out_dict["run_id"] = runs_df["run_id"].values
    if "timestamp" in runs_df.columns:
        out_dict["timestamp"] = runs_df["timestamp"].values

    out_dict.update(
        {
            "e01_score": e01_scores,
            "e02_score": e02_scores,
            "e03_score": e03_scores,
            "t2": t2_scores,
            "spe": spe_scores,
            "mspc_index": mspc_index_scores,
            "iforest_score": iforest_scores,
            "xgb_score": xgb_scores,
            "primary_detector": [primary_id] * n_runs,
            "alarm_budget_per_100": [5.0] * n_runs,
            "alarm": primary_alarm,
            "n_missing": n_missing,
            **alarm_cols,
        }
    )

    return pd.DataFrame(out_dict)


def score_single_run(
    bundle: LoadedBundle,
    run_dict: dict[str, Any],
    explain: bool = False,
    top_k: int = 5,
) -> dict[str, Any]:
    """Score a single run dictionary and optionally return sensor rankings."""
    run_id = str(run_dict.get("run_id", "run_unknown"))
    timestamp = run_dict.get("timestamp")
    sensors = run_dict.get("sensors")

    if sensors is not None and isinstance(sensors, (list, tuple)):
        if len(sensors) != len(SENSOR_COLUMNS):
            raise ValueError(f"Expected {len(SENSOR_COLUMNS)} sensor values, got {len(sensors)}.")
        row_dict = {col: val for col, val in zip(SENSOR_COLUMNS, sensors)}
    else:
        row_dict = {col: run_dict.get(col) for col in SENSOR_COLUMNS}

    df_single = pd.DataFrame([row_dict])
    if "run_id" in run_dict:
        df_single["run_id"] = run_id
    if timestamp is not None:
        df_single["timestamp"] = timestamp

    scored_df = score_runs_dataframe(bundle, df_single)
    row = scored_df.iloc[0]

    primary_id = bundle.policy.get("primary_detector", "E02")
    is_alarm = bool(row["alarm"])

    scores = {
        "e01_score": float(row["e01_score"]),
        "e02_score": float(row["e02_score"]),
        "e03_score": float(row["e03_score"]),
        "t2": float(row["t2"]),
        "spe": float(row["spe"]),
        "mspc_index": float(row["mspc_index"]),
        "iforest_score": float(row["iforest_score"]),
        "xgb_score": float(row["xgb_score"]),
    }

    alarms = {
        col: bool(row[col]) for col in scored_df.columns if col.startswith("alarm_")
    }

    top_sensors = None
    if explain:
        top_sensors = rank_sensors_for_run(
            df_single.iloc[0],
            quality=bundle.quality,
            mspc_model=bundle.mspc_model,
            xgb_model_or_booster=bundle.xgb_model,
            top_k=top_k,
        )

    return {
        "run_id": run_id,
        "timestamp": timestamp,
        "scores": scores,
        "alarms": alarms,
        "primary_detector": primary_id,
        "alarm": is_alarm,
        "alarm_budget_per_100": 5.0,
        "n_missing": int(row["n_missing"]),
        "quality_warnings": [] if int(row["n_missing"]) < 50 else ["High missing sensor count"],
        "top_sensors": top_sensors,
        "bundle_version": bundle.version,
    }


def score_csv_file(
    bundle_dir: str | Path,
    input_csv: str | Path,
    output_csv: str | Path,
) -> Path:
    """Score a CSV file of runs and write results in chronological order.

    Validates:
    - Input file exists and is non-empty.
    - All 590 raw sensor columns (f000..f589) are present.
    - Sensor values are numeric or nulls (fails whole file on string/invalid values).
    - Orders rows chronologically by timestamp if present.
    """
    input_p = Path(input_csv)
    if not input_p.exists():
        raise FileNotFoundError(f"Input CSV not found at {input_p}")

    df = pd.read_csv(input_p)
    if len(df) == 0:
        raise ValueError("Input CSV is empty.")

    # Validate presence of sensor columns
    missing_cols = [c for c in SENSOR_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Input CSV missing {len(missing_cols)} required sensor columns: {missing_cols[:5]}..."
        )

    # Validate sensor values are strictly numeric or NaN
    for col in SENSOR_COLUMNS:
        orig = df[col]
        converted = pd.to_numeric(orig, errors="coerce")
        bad_mask = orig.notna() & converted.isna()
        if bad_mask.any():
            bad_row = int(df.index[bad_mask][0])
            bad_val = orig.iloc[bad_row]
            raise ValueError(
                f"Invalid non-numeric value '{bad_val}' in column '{col}' at row {bad_row}."
            )
        df[col] = converted

    # Chronological sort if timestamp is provided
    if "timestamp" in df.columns:
        sort_cols = ["timestamp", "run_id"] if "run_id" in df.columns else ["timestamp"]
        df = df.sort_values(sort_cols).reset_index(drop=True)

    from yieldguard.artifacts import load_model_bundle

    bundle = load_model_bundle(bundle_dir)
    scored_df = score_runs_dataframe(bundle, df)

    out_p = Path(output_csv)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    scored_df.to_csv(out_p, index=False)
    return out_p

