"""Data drift detection, Evidently integration, missing-rate monitoring, and drift controls."""

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from evidently.legacy.metric_preset import DataDriftPreset
from evidently.legacy.report import Report

from yieldguard.artifacts import LoadedBundle, load_model_bundle

logger = logging.getLogger(__name__)


@dataclass
class DriftSummary:
    status: str  # "ok", "drift_detected", "insufficient_data", "error"
    reference_count: int
    current_count: int
    number_of_columns: int
    number_of_drifted_columns: int
    share_of_drifted_columns: float
    dataset_drift: bool
    drifted_features: list[str]
    method: str
    dataset_drift_threshold: float
    missing_rate_summary: dict[str, Any]
    details: dict[str, Any]


def compute_missing_rate_table(
    ref_df: pd.DataFrame,
    curr_df: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    """Compute per-column missingness counts and rates for reference and current cohorts."""
    n_ref = len(ref_df)
    n_curr = len(curr_df)

    ref_miss_cnt = [int(ref_df[c].isna().sum()) if c in ref_df.columns else n_ref for c in columns]
    curr_miss_cnt = [
        int(curr_df[c].isna().sum()) if c in curr_df.columns else n_curr for c in columns
    ]

    ref_miss_rate = [float(c / n_ref) if n_ref > 0 else 0.0 for c in ref_miss_cnt]
    curr_miss_rate = [float(c / n_curr) if n_curr > 0 else 0.0 for c in curr_miss_cnt]
    diff = [float(c - r) for c, r in zip(curr_miss_rate, ref_miss_rate)]

    return pd.DataFrame(
        {
            "column": columns,
            "ref_missing_count": ref_miss_cnt,
            "ref_missing_rate": ref_miss_rate,
            "current_missing_count": curr_miss_cnt,
            "current_missing_rate": curr_miss_rate,
            "missing_rate_difference": diff,
        }
    )


def run_drift_analysis(
    bundle: LoadedBundle,
    current_df: pd.DataFrame,
    output_dir: str | Path | None = None,
    min_runs_required: int = 100,
) -> DriftSummary:
    """Execute offline Evidently data drift check comparing current runs against frozen reference runs.

    Rules (docs/05_EXPERIMENT_PLAN.md section 9):
    - Reference cohort loaded from bundle.bundle_dir / "monitoring_reference.parquet" (D1-D4 training runs).
    - Current cohort: any batch of runs. If len(current_df) < min_runs_required (100):
      returns status="insufficient_data" without false alarms.
    - Dataset drift threshold is set to 50% (drift_share=0.50) of active retained features.
    - Calculates per-column missing-rate changes between reference and current cohorts.
    - Saves drift_report.html, drift_summary.json, and missing_rate_change.csv if output_dir is given.
    """
    ref_path = bundle.bundle_dir / "monitoring_reference.parquet"
    if not ref_path.exists():
        raise FileNotFoundError(f"Monitoring reference parquet missing in {bundle.bundle_dir}")

    ref_df = pd.read_parquet(ref_path)
    feature_cols = [
        c for c in bundle.feature_schema.get("retained_columns", []) if c in ref_df.columns
    ]
    if not feature_cols:
        feature_cols = [c for c in ref_df.columns if c in current_df.columns]

    n_ref = len(ref_df)
    n_curr = len(current_df)

    if n_curr < min_runs_required:
        summary = DriftSummary(
            status="insufficient_data",
            reference_count=n_ref,
            current_count=n_curr,
            number_of_columns=len(feature_cols),
            number_of_drifted_columns=0,
            share_of_drifted_columns=0.0,
            dataset_drift=False,
            drifted_features=[],
            method="evidently_data_drift_preset",
            dataset_drift_threshold=0.50,
            missing_rate_summary={
                "message": f"Sample size {n_curr} < minimum required {min_runs_required}.",
            },
            details={
                "message": (
                    f"Current sample size ({n_curr} runs) is below minimum statistical batch threshold "
                    f"({min_runs_required} runs). Drift analysis skipped to prevent small-sample false alarms."
                ),
            },
        )
        if output_dir:
            out_p = Path(output_dir)
            out_p.mkdir(parents=True, exist_ok=True)
            (out_p / "drift_summary.json").write_text(
                json.dumps(asdict(summary), indent=2), encoding="utf-8"
            )
        return summary

    # Filter to common retained features
    ref_sub = ref_df[feature_cols].copy()
    curr_sub = current_df[[c for c in feature_cols if c in current_df.columns]].copy()

    # Missing rate table
    missing_table = compute_missing_rate_table(ref_sub, curr_sub, feature_cols)
    large_shifts = missing_table[missing_table["missing_rate_difference"].abs() > 0.05]
    missing_rate_summary = {
        "columns_evaluated": len(feature_cols),
        "columns_with_missingness_change_gt_5pct": int(len(large_shifts)),
        "max_positive_missing_rate_change": float(missing_table["missing_rate_difference"].max()),
        "max_negative_missing_rate_change": float(missing_table["missing_rate_difference"].min()),
        "top_5_shifted_columns": large_shifts.sort_values(
            by="missing_rate_difference", key=abs, ascending=False
        )
        .head(5)[["column", "ref_missing_rate", "current_missing_rate", "missing_rate_difference"]]
        .to_dict(orient="records"),
    }

    # Run Evidently DataDriftPreset with 50% dataset drift threshold
    report = Report(metrics=[DataDriftPreset(drift_share=0.50)])
    report.run(reference_data=ref_sub, current_data=curr_sub)
    rep_dict = report.as_dict()

    drift_res: dict[str, Any] = {}
    drift_by_col: dict[str, Any] = {}
    for m in rep_dict.get("metrics", []):
        res = m.get("result", {})
        if "drift_by_columns" in res:
            drift_by_col = res["drift_by_columns"]
            if not drift_res:
                drift_res = res
        elif "dataset_drift" in res:
            drift_res = res

    n_cols = int(drift_res.get("number_of_columns", len(feature_cols)))
    n_drifted = int(drift_res.get("number_of_drifted_columns", 0))
    share_drifted = float(
        drift_res.get(
            "share_of_drifted_columns", float(n_drifted) / float(n_cols) if n_cols > 0 else 0.0
        )
    )
    is_dataset_drift = bool(drift_res.get("dataset_drift", share_drifted >= 0.50))

    drifted_features: list[str] = []
    feature_drift_details: dict[str, Any] = {}
    for col_name, c_info in drift_by_col.items():
        drifted = bool(c_info.get("drift_detected", False))
        score = float(c_info.get("drift_score", 0.0))
        stat_test = str(c_info.get("stattest_name", c_info.get("stat_test", "unknown")))
        feature_drift_details[col_name] = {
            "drift_detected": drifted,
            "drift_score": score,
            "stat_test": stat_test,
        }
        if drifted:
            drifted_features.append(col_name)

    status = "drift_detected" if (is_dataset_drift or n_drifted > 0) else "ok"

    summary = DriftSummary(
        status=status,
        reference_count=n_ref,
        current_count=n_curr,
        number_of_columns=n_cols,
        number_of_drifted_columns=n_drifted,
        share_of_drifted_columns=share_drifted,
        dataset_drift=is_dataset_drift,
        drifted_features=drifted_features,
        method="evidently_data_drift_preset",
        dataset_drift_threshold=0.50,
        missing_rate_summary=missing_rate_summary,
        details={
            "feature_drift_sample": {
                k: feature_drift_details[k] for k in list(feature_drift_details.keys())[:10]
            },
        },
    )

    if output_dir:
        out_p = Path(output_dir)
        out_p.mkdir(parents=True, exist_ok=True)
        (out_p / "drift_summary.json").write_text(
            json.dumps(asdict(summary), indent=2), encoding="utf-8"
        )
        missing_table.to_csv(out_p / "missing_rate_change.csv", index=False)
        try:
            report.save_html(str(out_p / "drift_report.html"))
        except (OSError, ValueError, RuntimeError) as e:
            logger.warning(f"Could not write Evidently HTML report: {e}")

    return summary


def run_drift_controls(
    bundle: LoadedBundle,
    output_dir: str | Path = "reports/drift/controls",
) -> dict[str, Any]:
    """Execute three controlled drift verification scenarios (docs/05_EXPERIMENT_PLAN.md section 9).

    Controls:
    1. No-change control: reference vs identical reference copy.
       Expected: status='ok', dataset_drift=False, 0 drifted columns.
    2. Shifted-sensor control: reference with 5 sensors shifted by +2 reference standard deviations.
       Expected: status='drift_detected', all 5 perturbed sensors flagged as drifted.
    3. Insufficient-data control: sub-sample of 50 runs (< 100).
       Expected: status='insufficient_data', no false alarms.
    """
    ref_path = bundle.bundle_dir / "monitoring_reference.parquet"
    ref_df = pd.read_parquet(ref_path)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    feature_cols = [
        c for c in bundle.feature_schema.get("retained_columns", []) if c in ref_df.columns
    ]
    if not feature_cols:
        feature_cols = list(ref_df.columns)

    # 1. No-change Control
    no_change_summary = run_drift_analysis(bundle, ref_df, min_runs_required=100)

    # 2. Shifted-sensor Control (+2 sigma shift on 5 retained sensors)
    shifted_df = ref_df.copy()
    perturbed_sensors = feature_cols[:5]
    for col in perturbed_sensors:
        std_val = float(shifted_df[col].std(skipna=True))
        if np.isnan(std_val) or std_val == 0.0:
            std_val = 1.0
        shifted_df[col] = shifted_df[col] + 2.0 * std_val

    shifted_summary = run_drift_analysis(bundle, shifted_df, min_runs_required=100)

    # 3. Insufficient-data Control (50 runs < 100)
    small_df = ref_df.iloc[:50].copy()
    insufficient_summary = run_drift_analysis(bundle, small_df, min_runs_required=100)

    control_results = {
        "bundle_version": bundle.version,
        "controls": {
            "no_change_control": {
                "expected_status": "ok",
                "actual_status": no_change_summary.status,
                "dataset_drift": no_change_summary.dataset_drift,
                "drifted_features_count": no_change_summary.number_of_drifted_columns,
                "passed": bool(
                    no_change_summary.status == "ok"
                    and not no_change_summary.dataset_drift
                    and no_change_summary.number_of_drifted_columns == 0
                ),
            },
            "shifted_sensor_control": {
                "perturbed_columns": perturbed_sensors,
                "expected_status": "drift_detected",
                "actual_status": shifted_summary.status,
                "dataset_drift": shifted_summary.dataset_drift,
                "drifted_features_detected": shifted_summary.drifted_features,
                "passed": bool(
                    shifted_summary.status == "drift_detected"
                    and all(c in shifted_summary.drifted_features for c in perturbed_sensors)
                ),
            },
            "insufficient_data_control": {
                "sample_size": len(small_df),
                "threshold_required": 100,
                "expected_status": "insufficient_data",
                "actual_status": insufficient_summary.status,
                "passed": bool(insufficient_summary.status == "insufficient_data"),
            },
        },
        "all_controls_passed": bool(
            (
                no_change_summary.status == "ok"
                and not no_change_summary.dataset_drift
                and no_change_summary.number_of_drifted_columns == 0
            )
            and (
                shifted_summary.status == "drift_detected"
                and all(c in shifted_summary.drifted_features for c in perturbed_sensors)
            )
            and (insufficient_summary.status == "insufficient_data")
        ),
    }

    (out_path / "control_results.json").write_text(
        json.dumps(control_results, indent=2), encoding="utf-8"
    )
    logger.info(f"Saved drift control results to {out_path / 'control_results.json'}")
    return control_results


def run_cli_drift(
    bundle_dir: str,
    current_path: str,
    output_dir: str,
    split_filter: str | None = None,
) -> dict[str, Any]:
    """CLI entrypoint for running drift analysis."""
    bundle = load_model_bundle(bundle_dir)
    curr_p = Path(current_path)

    if curr_p.suffix == ".parquet":
        curr_df = pd.read_parquet(curr_p)
    else:
        curr_df = pd.read_csv(curr_p)

    if split_filter and "block" in curr_df.columns:
        curr_df = curr_df[curr_df["block"] == split_filter].copy()

    summary = run_drift_analysis(bundle, curr_df, output_dir=output_dir)
    return asdict(summary)


def run_cli_drift_controls(
    bundle_dir: str,
    output_dir: str,
) -> dict[str, Any]:
    """CLI entrypoint for running drift verification controls."""
    bundle = load_model_bundle(bundle_dir)
    return run_drift_controls(bundle, output_dir=output_dir)
