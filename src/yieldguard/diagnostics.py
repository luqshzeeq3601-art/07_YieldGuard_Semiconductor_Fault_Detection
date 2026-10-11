"""Phase 2 exploratory diagnostics for SECOM: per-block drift, fail clustering, oracle upper bound, and E02 tie analysis.

All outputs produced by this module are explicitly labelled `exploratory_post_holdout` because
the v0.1.0 holdout set has already been evaluated once.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import ks_2samp, norm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from yieldguard.evaluate import (
    compute_pr_auc,
    compute_recall_at_fixed_far,
    compute_roc_auc,
    stratified_bootstrap_ci,
)
from yieldguard.spc import FittedSpcModel


def compute_wald_wolfowitz_runs_test(binary_seq: np.ndarray) -> dict[str, float | int | None]:
    """Compute two-sided Wald-Wolfowitz runs test on a binary (0/1) sequence for temporal clustering."""
    y = np.asarray(binary_seq, dtype=np.int64)
    n = len(y)
    n1 = int(np.sum(y == 1))
    n0 = int(np.sum(y == 0))
    if n < 2 or n1 == 0 or n0 == 0:
        return {
            "n_total": n,
            "n_fails": n1,
            "n_passing": n0,
            "observed_runs": 1 if n > 0 else 0,
            "expected_runs": None,
            "z_statistic": None,
            "p_value": None,
        }

    # Number of contiguous runs of identical values
    observed_runs = int(1 + np.sum(y[1:] != y[:-1]))
    expected_runs = float(1.0 + (2.0 * n1 * n0) / n)
    var_runs = float((2.0 * n1 * n0 * (2.0 * n1 * n0 - n)) / ((n**2) * (n - 1)))
    if var_runs <= 1e-12:
        return {
            "n_total": n,
            "n_fails": n1,
            "n_passing": n0,
            "observed_runs": observed_runs,
            "expected_runs": expected_runs,
            "z_statistic": 0.0,
            "p_value": 1.0,
        }

    z_stat = float((observed_runs - expected_runs) / np.sqrt(var_runs))
    p_val = float(2.0 * (1.0 - norm.cdf(abs(z_stat))))
    return {
        "n_total": n,
        "n_fails": n1,
        "n_passing": n0,
        "observed_runs": observed_runs,
        "expected_runs": expected_runs,
        "z_statistic": z_stat,
        "p_value": p_val,
    }


def analyze_block_drift_and_fails(
    runs_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    retained_columns: list[str],
    monitored_sensors: list[str],
) -> dict[str, Any]:
    """Quantify per-block fail rates, temporal fail clustering, and per-block sensor drift relative to D1."""
    merged = (
        runs_df.merge(labels_df[["run_id", "label_fail"]], on="run_id", how="inner")
        .sort_values("run_id")
        .reset_index(drop=True)
    )

    block_order = [b for b in ["D1", "D2", "D3", "D4", "D5", "holdout"] if b in set(merged["block"])]
    if not block_order:
        block_order = list(merged["block"].unique())

    # 1. Temporal fail clustering across full timeline and per block
    y_all = merged["label_fail"].to_numpy(dtype=np.int64)
    fail_indices = np.flatnonzero(y_all == 1)
    inter_fail_gaps = np.diff(fail_indices).astype(np.int64) if len(fail_indices) > 1 else np.array([], dtype=np.int64)

    lag1_autocorr = (
        float(np.corrcoef(y_all[:-1], y_all[1:])[0, 1])
        if len(y_all) > 2 and np.std(y_all) > 0
        else 0.0
    )
    runs_test_all = compute_wald_wolfowitz_runs_test(y_all)

    per_block_fails: dict[str, Any] = {}
    for blk in block_order:
        sub = merged[merged["block"] == blk]
        yb = sub["label_fail"].to_numpy(dtype=np.int64)
        per_block_fails[blk] = {
            "n_runs": int(len(sub)),
            "n_fails": int(np.sum(yb == 1)),
            "fail_prevalence": float(np.mean(yb == 1)) if len(yb) > 0 else 0.0,
            "start_timestamp": str(sub["timestamp"].iloc[0]) if "timestamp" in sub.columns and len(sub) > 0 else None,
            "end_timestamp": str(sub["timestamp"].iloc[-1]) if "timestamp" in sub.columns and len(sub) > 0 else None,
        }

    # 2. Per-block sensor drift against D1 (first block)
    ref_block = block_order[0]
    ref_df = merged[merged["block"] == ref_block]

    ref_medians: dict[str, float] = {}
    ref_mads: dict[str, float] = {}
    for col in retained_columns:
        v = ref_df[col].dropna().to_numpy(dtype=np.float64)
        med = float(np.median(v)) if len(v) > 0 else 0.0
        mad = float(np.median(np.abs(v - med)) * 1.4826) if len(v) > 0 else 1.0
        if mad < 1e-9:
            sd = float(np.std(v, ddof=1)) if len(v) > 1 else 1.0
            mad = sd if sd >= 1e-9 else 1.0
        ref_medians[col] = med
        ref_mads[col] = mad

    per_block_drift: dict[str, Any] = {}
    monitored_sensor_timeline: dict[str, dict[str, Any]] = {s: {} for s in monitored_sensors}

    for blk in block_order:
        cur_df = merged[merged["block"] == blk]
        drifted_cols: list[str] = []
        ks_stats: list[tuple[str, float, float]] = []

        for col in retained_columns:
            x_ref = ref_df[col].dropna().to_numpy(dtype=np.float64)
            x_cur = cur_df[col].dropna().to_numpy(dtype=np.float64)
            if len(x_ref) < 2 or len(x_cur) < 2:
                continue
            res = ks_2samp(x_ref, x_cur)
            p_val = float(res.pvalue)
            d_stat = float(res.statistic)
            if p_val < 0.05 and blk != ref_block:
                drifted_cols.append(col)
            ks_stats.append((col, d_stat, p_val))

        ks_stats.sort(key=lambda item: (-item[1], item[2]))
        top_5_drifted = [
            {"sensor": c, "ks_statistic": d, "p_value": p}
            for c, d, p in ks_stats[:5]
        ]

        per_block_drift[blk] = {
            "reference_block": ref_block,
            "n_retained_evaluated": len(retained_columns),
            "n_drifted_ks_p005": len(drifted_cols),
            "share_drifted_ks_p005": float(len(drifted_cols) / max(len(retained_columns), 1)),
            "top_5_drifted_sensors": top_5_drifted,
        }

        for s in monitored_sensors:
            if s not in cur_df.columns:
                continue
            vals = cur_df[s].dropna().to_numpy(dtype=np.float64)
            cur_med = float(np.median(vals)) if len(vals) > 0 else np.nan
            std_med_shift = (
                float(abs(cur_med - ref_medians.get(s, 0.0)) / ref_mads.get(s, 1.0))
                if not np.isnan(cur_med)
                else 0.0
            )
            x_ref_s = ref_df[s].dropna().to_numpy(dtype=np.float64)
            ks_p = float(ks_2samp(x_ref_s, vals).pvalue) if len(x_ref_s) >= 2 and len(vals) >= 2 else 1.0
            monitored_sensor_timeline[s][blk] = {
                "median": cur_med,
                "std_median_shift_vs_ref": std_med_shift,
                "ks_p_value_vs_ref": ks_p,
                "missing_fraction": float(cur_df[s].isna().mean()),
            }

    return {
        "temporal_fail_clustering": {
            "n_total_runs": int(len(merged)),
            "n_total_fails": int(np.sum(y_all == 1)),
            "overall_prevalence": float(np.mean(y_all == 1)),
            "lag1_fail_autocorrelation": lag1_autocorr,
            "wald_wolfowitz_runs_test": runs_test_all,
            "inter_fail_gap_summary": {
                "min_gap": int(np.min(inter_fail_gaps)) if len(inter_fail_gaps) > 0 else None,
                "median_gap": float(np.median(inter_fail_gaps)) if len(inter_fail_gaps) > 0 else None,
                "mean_gap": float(np.mean(inter_fail_gaps)) if len(inter_fail_gaps) > 0 else None,
                "max_gap": int(np.max(inter_fail_gaps)) if len(inter_fail_gaps) > 0 else None,
            },
            "per_block_fails": per_block_fails,
        },
        "per_block_sensor_drift": per_block_drift,
        "monitored_sensor_drift_timeline": monitored_sensor_timeline,
    }


def _impute_and_robust_scale_fold(
    x_train: np.ndarray,
    x_val: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fit median imputation and MAD scaling on x_train and apply to x_train and x_val."""
    medians = np.nanmedian(x_train, axis=0)
    medians = np.where(np.isnan(medians), 0.0, medians)

    x_tr_imp = np.where(np.isnan(x_train), medians[None, :], x_train)
    x_va_imp = np.where(np.isnan(x_val), medians[None, :], x_val)

    mads = np.median(np.abs(x_tr_imp - medians[None, :]), axis=0) * 1.4826
    stds = np.std(x_tr_imp, axis=0, ddof=1)
    scales = np.where(mads >= 1e-9, mads, np.where(stds >= 1e-9, stds, 1.0))

    x_tr_scaled = np.clip((x_tr_imp - medians[None, :]) / scales[None, :], -25.0, 25.0)
    x_va_scaled = np.clip((x_va_imp - medians[None, :]) / scales[None, :], -25.0, 25.0)
    return x_tr_imp, x_va_imp, x_tr_scaled, x_va_scaled


def evaluate_oracle_upper_bound(
    runs_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    retained_columns: list[str],
    n_splits: int = 5,
    seed: int = 42,
    n_bootstraps: int = 2000,
) -> dict[str, Any]:
    """Evaluate a 5-fold stratified random-split oracle upper bound across all runs.

    This diagnostic intentionally removes temporal drift by shuffling runs across time
    to test whether the sensor channels contain intrinsic predictive signal for `label_fail`.
    """
    merged = (
        runs_df.merge(labels_df[["run_id", "label_fail"]], on="run_id", how="inner")
        .sort_values("run_id")
        .reset_index(drop=True)
    )
    x_all = merged[retained_columns].to_numpy(dtype=np.float64)
    y_all = merged["label_fail"].to_numpy(dtype=np.int64)
    n_runs = len(y_all)
    n_fails = int(np.sum(y_all == 1))
    prevalence = float(n_fails / n_runs) if n_runs > 0 else 0.0

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    model_names = [
        "oracle_xgb_shallow",
        "oracle_xgb_depth3",
        "oracle_logreg_l2",
        "oracle_rf_balanced",
    ]
    oof_scores: dict[str, np.ndarray] = {m: np.zeros(n_runs, dtype=np.float64) for m in model_names}
    fold_pr_aucs: dict[str, list[float]] = {m: [] for m in model_names}

    for train_idx, val_idx in skf.split(x_all, y_all):
        x_tr, x_va = x_all[train_idx], x_all[val_idx]
        y_tr, y_va = y_all[train_idx], y_all[val_idx]

        n_pos = max(int(np.sum(y_tr == 1)), 1)
        n_neg = int(np.sum(y_tr == 0))
        spw = float(n_neg / n_pos)

        # 1. Shallow XGBoost (xgb_c01)
        clf_xgb1 = xgb.XGBClassifier(
            max_depth=1,
            learning_rate=0.03,
            n_estimators=100,
            subsample=0.8,
            colsample_bytree=0.5,
            min_child_weight=5,
            scale_pos_weight=spw,
            objective="binary:logistic",
            eval_metric="aucpr",
            tree_method="hist",
            random_state=seed,
            n_jobs=1,
        )
        clf_xgb1.fit(x_tr, y_tr)
        s_xgb1 = clf_xgb1.predict_proba(x_va)[:, 1].astype(np.float64)
        oof_scores["oracle_xgb_shallow"][val_idx] = s_xgb1
        ap1, _ = compute_pr_auc(y_va, s_xgb1)
        fold_pr_aucs["oracle_xgb_shallow"].append(float(ap1 or 0.0))

        # 2. Depth-3 XGBoost
        clf_xgb3 = xgb.XGBClassifier(
            max_depth=3,
            learning_rate=0.05,
            n_estimators=150,
            subsample=0.8,
            colsample_bytree=0.5,
            min_child_weight=3,
            scale_pos_weight=spw,
            objective="binary:logistic",
            eval_metric="aucpr",
            tree_method="hist",
            random_state=seed,
            n_jobs=1,
        )
        clf_xgb3.fit(x_tr, y_tr)
        s_xgb3 = clf_xgb3.predict_proba(x_va)[:, 1].astype(np.float64)
        oof_scores["oracle_xgb_depth3"][val_idx] = s_xgb3
        ap3, _ = compute_pr_auc(y_va, s_xgb3)
        fold_pr_aucs["oracle_xgb_depth3"].append(float(ap3 or 0.0))

        # 3. Regularised Logistic Regression (L2, C=0.1) on robust-scaled features
        x_tr_imp, x_va_imp, x_tr_sc, x_va_sc = _impute_and_robust_scale_fold(x_tr, x_va)
        clf_lr = LogisticRegression(
            C=0.1,
            class_weight="balanced",
            max_iter=1000,
            random_state=seed,
        )
        clf_lr.fit(x_tr_sc, y_tr)
        s_lr = clf_lr.predict_proba(x_va_sc)[:, 1].astype(np.float64)
        oof_scores["oracle_logreg_l2"][val_idx] = s_lr
        ap_lr, _ = compute_pr_auc(y_va, s_lr)
        fold_pr_aucs["oracle_logreg_l2"].append(float(ap_lr or 0.0))

        # 4. Random Forest on median-imputed features
        clf_rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=5,
            class_weight="balanced",
            random_state=seed,
            n_jobs=1,
        )
        clf_rf.fit(x_tr_imp, y_tr)
        s_rf = clf_rf.predict_proba(x_va_imp)[:, 1].astype(np.float64)
        oof_scores["oracle_rf_balanced"][val_idx] = s_rf
        ap_rf, _ = compute_pr_auc(y_va, s_rf)
        fold_pr_aucs["oracle_rf_balanced"].append(float(ap_rf or 0.0))

    models_summary: dict[str, Any] = {}
    best_model_id = model_names[0]
    best_mean_pr_auc = -1.0

    for m in model_names:
        scores = oof_scores[m]
        f_aps = fold_pr_aucs[m]
        mean_fold_ap = float(np.mean(f_aps))
        se_fold_ap = float(np.std(f_aps, ddof=1) / np.sqrt(len(f_aps))) if len(f_aps) > 1 else 0.0
        pooled_ap, _ = compute_pr_auc(y_all, scores)
        pooled_roc, _ = compute_roc_auc(y_all, scores)
        rec_5 = compute_recall_at_fixed_far(y_all, scores, far_per_100=5.0)
        boot = stratified_bootstrap_ci(
            y_all,
            scores,
            n_bootstraps=n_bootstraps,
            seed=seed,
            far_target_per_100=5.0,
        )
        models_summary[m] = {
            "fold_pr_aucs": [float(v) for v in f_aps],
            "mean_fold_pr_auc": mean_fold_ap,
            "se_fold_pr_auc": se_fold_ap,
            "pooled_oof_pr_auc": pooled_ap,
            "pooled_oof_roc_auc": pooled_roc,
            "pooled_recall_at_5_far": rec_5["recall"],
            "stratified_95_ci_pr_auc": boot["pr_auc_95_ci"],
            "stratified_95_ci_recall_at_5_far": boot["recall_at_far_95_ci"],
            "lift_over_prevalence": float(mean_fold_ap / prevalence) if prevalence > 0 else None,
        }
        if mean_fold_ap > best_mean_pr_auc:
            best_mean_pr_auc = mean_fold_ap
            best_model_id = m

    best_ci_lo = float(models_summary[best_model_id]["stratified_95_ci_pr_auc"][0])
    # Gate rule: best oracle model's 95% CI lower bound > prevalence AND mean fold PR-AUC >= 1.5x prevalence
    min_meaningful_pr_auc = 1.5 * prevalence
    oracle_beats_chance = bool((best_ci_lo > prevalence) and (best_mean_pr_auc >= min_meaningful_pr_auc))

    return {
        "protocol_note": "Exploratory 5-fold stratified random-split CV across all runs (intentionally includes temporal leakage to test intrinsic sensor signal).",
        "n_runs": n_runs,
        "n_fails": n_fails,
        "prevalence": prevalence,
        "min_meaningful_pr_auc_1_5x": min_meaningful_pr_auc,
        "target_2x_prevalence": 2.0 * prevalence,
        "best_oracle_model": best_model_id,
        "best_oracle_mean_fold_pr_auc": best_mean_pr_auc,
        "best_oracle_95_ci_pr_auc": models_summary[best_model_id]["stratified_95_ci_pr_auc"],
        "oracle_beats_chance": oracle_beats_chance,
        "gate_decision": (
            "PROCEED_SECOM_AND_WAFER"
            if oracle_beats_chance
            else "STOP_SECOM_PROCEED_WAFER_ONLY"
        ),
        "models": models_summary,
    }


def analyze_e02_score_ties(
    spc_model: FittedSpcModel,
    runs_df: pd.DataFrame,
    thresholds_e02: dict[str, float],
) -> dict[str, Any]:
    """Decompose E02 max(|z|) scores by winning sensor and quantify quantization/tie clusters."""
    monitored = spc_model.monitored_sensors
    sensor_quantization: dict[str, Any] = {}

    for s in monitored:
        spec = spc_model.sensor_specs[s]
        vals = runs_df[s].to_numpy(dtype=np.float64)
        non_na = vals[~np.isnan(vals)]
        val_counts = pd.Series(non_na).value_counts()
        top_val = float(val_counts.index[0]) if len(val_counts) > 0 else None
        top_count = int(val_counts.iloc[0]) if len(val_counts) > 0 else 0
        top_z = float(abs(top_val - spec.mu0) / spec.sigma0) if top_val is not None else 0.0
        sensor_quantization[s] = {
            "mu0": spec.mu0,
            "sigma0": spec.sigma0,
            "n_non_missing": int(len(non_na)),
            "n_unique_values": int(len(val_counts)),
            "unique_to_sample_ratio": float(len(val_counts) / max(len(non_na), 1)),
            "most_frequent_raw_value": top_val,
            "most_frequent_count": top_count,
            "most_frequent_abs_z": top_z,
        }

    def _inspect_subset(sub_df: pd.DataFrame) -> dict[str, Any]:
        n = len(sub_df)
        n_mon = len(monitored)
        abs_z_mat = np.zeros((n, n_mon), dtype=np.float64)
        for j, s in enumerate(monitored):
            spec = spc_model.sensor_specs[s]
            v = sub_df[s].to_numpy(dtype=np.float64)
            u = np.abs((v - spec.mu0) / spec.sigma0)
            abs_z_mat[:, j] = np.where(np.isnan(u), 0.0, u)

        e02 = np.max(abs_z_mat, axis=1)
        win_idx = np.argmax(abs_z_mat, axis=1)
        win_sensors = [monitored[idx] for idx in win_idx]
        win_counts = pd.Series(win_sensors).value_counts().to_dict()

        score_counts = pd.Series(np.round(e02, 10)).value_counts()
        tied_groups = score_counts[score_counts > 1]
        top_tied_values = []
        for val, cnt in tied_groups.head(10).items():
            mask = np.isclose(e02, float(val), atol=1e-9)
            dominant_sensor = pd.Series(np.array(win_sensors)[mask]).value_counts().index[0]
            top_tied_values.append(
                {
                    "e02_score": float(val),
                    "count": int(cnt),
                    "dominant_winning_sensor": str(dominant_sensor),
                }
            )

        tau_b2 = float(thresholds_e02.get("budget_2", 7.244515))
        tau_b5 = float(thresholds_e02.get("budget_5", 6.282155))
        tau_b10 = float(thresholds_e02.get("budget_10", 6.036828))

        return {
            "n_runs": int(n),
            "n_unique_e02_scores": int(len(score_counts)),
            "n_runs_in_ties": int(tied_groups.sum()),
            "tie_fraction": float(tied_groups.sum() / max(n, 1)),
            "winning_sensor_counts": {str(k): int(v) for k, v in win_counts.items()},
            "top_tied_scores": top_tied_values,
            "alarms_above_budget_2_threshold": int(np.sum(e02 > tau_b2)),
            "alarms_at_or_above_budget_2_threshold": int(np.sum(e02 >= tau_b2 - 1e-12)),
            "alarms_above_budget_5_threshold": int(np.sum(e02 > tau_b5)),
            "alarms_at_or_above_budget_5_threshold": int(np.sum(e02 >= tau_b5 - 1e-12)),
            "alarms_above_budget_10_threshold": int(np.sum(e02 > tau_b10)),
            "alarms_at_or_above_budget_10_threshold": int(np.sum(e02 >= tau_b10 - 1e-12)),
            "runs_Strictly_between_b10_and_b5": int(np.sum((e02 > tau_b10) & (e02 <= tau_b5))),
        }

    d5_df = runs_df[runs_df["block"] == "D5"].reset_index(drop=True)
    hold_df = runs_df[runs_df["block"] == "holdout"].reset_index(drop=True)

    return {
        "monitored_sensor_quantization": sensor_quantization,
        "calibration_block_d5": _inspect_subset(d5_df) if len(d5_df) > 0 else {},
        "holdout_block": _inspect_subset(hold_df) if len(hold_df) > 0 else {},
        "all_runs": _inspect_subset(runs_df),
    }


def run_secom_phase2_diagnostics(
    runs_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    retained_columns: list[str],
    spc_model: FittedSpcModel,
    thresholds_e02: dict[str, float],
    output_path: str | Path | None = "reports/exploratory/secom_phase2_diagnostics.json",
    n_bootstraps: int = 2000,
    seed: int = 42,
) -> dict[str, Any]:
    """Run all three Phase 2 exploratory diagnostics on SECOM and persist JSON report."""
    drift_and_fails = analyze_block_drift_and_fails(
        runs_df=runs_df,
        labels_df=labels_df,
        retained_columns=retained_columns,
        monitored_sensors=spc_model.monitored_sensors,
    )
    oracle = evaluate_oracle_upper_bound(
        runs_df=runs_df,
        labels_df=labels_df,
        retained_columns=retained_columns,
        n_splits=5,
        seed=seed,
        n_bootstraps=n_bootstraps,
    )
    e02_ties = analyze_e02_score_ties(
        spc_model=spc_model,
        runs_df=runs_df,
        thresholds_e02=thresholds_e02,
    )

    report = {
        "dataset_id": "SECOM",
        "evaluation_type": "exploratory_post_holdout",
        "disclaimer": (
            "EXPLORATORY DIAGNOSTIC REPORT: The v0.1.0 SECOM holdout was already evaluated once. "
            "All analyses in this file are post-holdout root-cause diagnostics and are NOT confirmatory claims."
        ),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "step1_drift_and_fail_clustering": drift_and_fails,
        "step2_oracle_upper_bound": oracle,
        "step3_e02_score_ties": e02_ties,
    }

    if output_path is not None:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    return report
