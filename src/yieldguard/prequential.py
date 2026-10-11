"""YieldGuard v0.2 prequential (rolling-origin) drift-adaptive evaluation engine.

Implements:
- Chronological prequential block partitioning (B_1 .. B_K)
- Train-only cleaning with robust Median/MAD scaling (RobustPreprocessor)
- Adaptive rolling-window Univariate SPC with top-3 RMS tie-breaking (E02_v2, E03_v2)
- Adaptive rolling-window Multivariate PCA SPC (E04_v2, E05_v2, E06_v2)
- Adaptive rolling-window Isolation Forest (E07_v2)
- Expanding-window Class-Weighted XGBoost (E08_v2)
- Expanding-window L2-Regularised Logistic Regression baseline (E09_v2)
- Unlabelled recent-window alarm budget calibration (budgets 2, 5, 10 per 100 runs)
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

from yieldguard.evaluate import (
    compute_pr_auc,
    compute_recall_at_fixed_far,
    compute_roc_auc,
    moving_block_bootstrap_ci,
    paired_stratified_bootstrap_diff,
    stratified_bootstrap_ci,
)
from yieldguard.mspc import fit_multivariate_spc, score_multivariate_spc
from yieldguard.policy import compute_multi_budget_thresholds, evaluate_at_threshold


@dataclass
class RobustPreprocessor:
    """Train-only sensor cleaning, collinearity pruning, median imputation, and Median/MAD scaling."""

    missing_threshold: float = 0.50
    dominant_value_threshold: float = 0.99
    collinearity_threshold: float = 0.95
    clip_z: float = 15.0

    retained_sensors: list[str] = field(default_factory=list)
    dropped_high_missing: list[str] = field(default_factory=list)
    dropped_constant: list[str] = field(default_factory=list)
    dropped_collinear: list[str] = field(default_factory=list)
    group_map: dict[str, list[str]] = field(default_factory=dict)
    medians: dict[str, float] = field(default_factory=dict)
    mad_scales: dict[str, float] = field(default_factory=dict)
    is_fitted: bool = False

    def fit(self, df: pd.DataFrame, sensor_cols: list[str]) -> RobustPreprocessor:
        X = df[sensor_cols].copy()
        miss_rates = X.isna().mean()
        self.dropped_high_missing = [c for c in sensor_cols if miss_rates[c] > self.missing_threshold]
        after_miss = [c for c in sensor_cols if c not in set(self.dropped_high_missing)]

        self.dropped_constant = []
        after_const = []
        for c in after_miss:
            s = X[c].dropna()
            if len(s) <= 1 or float(s.std(ddof=1)) == 0.0:
                self.dropped_constant.append(c)
                continue
            top_frac = float(s.value_counts(normalize=True).iloc[0])
            if top_frac >= self.dominant_value_threshold:
                self.dropped_constant.append(c)
                continue
            after_const.append(c)

        # Impute medians temporarily for correlation calculation
        temp_medians = X[after_const].median().fillna(0.0)
        X_imp = X[after_const].fillna(temp_medians)

        # Sort by missingness ascending, then sensor name for deterministic greedy collinearity pruning
        ordered = sorted(after_const, key=lambda c: (float(miss_rates[c]), c))
        corr_mat = X_imp[ordered].corr().abs().fillna(0.0).to_numpy(dtype=np.float64)

        retained: list[str] = []
        retained_idx: list[int] = []
        dropped_col: list[str] = []
        group_map: dict[str, list[str]] = {}

        for idx, col in enumerate(ordered):
            matched_rep: str | None = None
            for r_idx, r_col in zip(retained_idx, retained):
                if corr_mat[idx, r_idx] > self.collinearity_threshold:
                    matched_rep = r_col
                    break
            if matched_rep is None:
                retained.append(col)
                retained_idx.append(idx)
                group_map[col] = [col]
            else:
                dropped_col.append(col)
                group_map[matched_rep].append(col)

        self.retained_sensors = retained
        self.dropped_collinear = dropped_col
        self.group_map = group_map

        self.medians = {}
        self.mad_scales = {}
        for col in self.retained_sensors:
            vals = X[col].dropna().to_numpy(dtype=np.float64)
            med = float(np.median(vals)) if len(vals) > 0 else 0.0
            mad = float(np.median(np.abs(vals - med))) if len(vals) > 0 else 0.0
            iqr_scale = (
                float(np.percentile(vals, 75) - np.percentile(vals, 25)) / 1.349
                if len(vals) > 1
                else 0.0
            )
            std_scale = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            mad_scaled = 1.4826 * mad
            if mad_scaled > 1e-6:
                scale = mad_scaled
            elif iqr_scale > 1e-6:
                scale = iqr_scale
            elif std_scale > 1e-6:
                scale = std_scale
            else:
                scale = 1.0
            self.medians[col] = med
            self.mad_scales[col] = scale

        self.is_fitted = True
        return self

    def transform_imputed(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            raise RuntimeError("RobustPreprocessor must be fitted before transform.")
        out = df[self.retained_sensors].copy()
        for col in self.retained_sensors:
            out[col] = out[col].fillna(self.medians[col]).astype(np.float64)
        return out

    def transform_scaled(self, df: pd.DataFrame) -> pd.DataFrame:
        imp = self.transform_imputed(df)
        med_s = pd.Series(self.medians)
        scale_s = pd.Series(self.mad_scales)
        scaled = (imp - med_s) / scale_s
        return scaled.clip(lower=-self.clip_z, upper=self.clip_z)


def partition_prequential_blocks(
    df: pd.DataFrame,
    n_blocks: int = 8,
    timestamp_col: str | None = "timestamp",
) -> list[pd.DataFrame]:
    """Partition chronologically sorted DataFrame into B contiguous blocks, preserving timestamp ties."""
    if n_blocks < 2:
        raise ValueError("n_blocks must be >= 2 for prequential evaluation.")
    n_total = len(df)
    if n_total < n_blocks * 5:
        raise ValueError(f"Dataset too small ({n_total} rows) for {n_blocks} prequential blocks.")

    df_sorted = df.reset_index(drop=True)
    raw_bounds = [int(round(i * n_total / n_blocks)) for i in range(n_blocks + 1)]
    raw_bounds[0] = 0
    raw_bounds[-1] = n_total

    if timestamp_col is not None and timestamp_col in df_sorted.columns:
        ts = df_sorted[timestamp_col].astype(str).tolist()
        for b_idx in range(1, n_blocks):
            pos = raw_bounds[b_idx]
            while pos < raw_bounds[b_idx + 1] - 1 and ts[pos - 1] == ts[pos]:
                pos += 1
            raw_bounds[b_idx] = pos

    blocks: list[pd.DataFrame] = []
    for b_idx in range(n_blocks):
        start, end = raw_bounds[b_idx], raw_bounds[b_idx + 1]
        block = df_sorted.iloc[start:end].copy()
        block["prequential_block"] = f"B{b_idx + 1}"
        blocks.append(block)
    return blocks


def compute_composite_top3_score(abs_z_matrix: np.ndarray) -> np.ndarray:
    """Compute max |Z| with deterministic top-3 RMS and all-sensor RMS tie-breaking.

    S(i) = |z_{i,(1)}| + 1e-3 * sqrt(mean(z_{i,(1..3)}^2)) + 1e-6 * sqrt(mean(z_{i,(1..M)}^2))
    """
    if abs_z_matrix.ndim != 2 or abs_z_matrix.shape[1] == 0:
        raise ValueError("abs_z_matrix must be 2D with at least 1 column.")
    sorted_desc = np.sort(np.abs(abs_z_matrix), axis=1)[:, ::-1]
    z_max = sorted_desc[:, 0]
    k_top = min(3, sorted_desc.shape[1])
    top3_rms = np.sqrt(np.mean(sorted_desc[:, :k_top] ** 2, axis=1))
    all_rms = np.sqrt(np.mean(sorted_desc**2, axis=1))
    return z_max + 1e-3 * top3_rms + 1e-6 * all_rms


@dataclass
class AdaptiveUnivariateSPC:
    """Adaptive rolling-window Univariate SPC with Median/MAD Phase I parameters and tie-broken scores."""

    top_m: int = 10
    max_missing_rate: float = 0.05
    ewma_lambda: float = 0.2
    monitored_sensors: list[str] = field(default_factory=list)
    medians: dict[str, float] = field(default_factory=dict)
    scales: dict[str, float] = field(default_factory=dict)

    def fit(self, window_df: pd.DataFrame, retained_sensors: list[str], y_window: np.ndarray) -> AdaptiveUnivariateSPC:
        miss = window_df[retained_sensors].isna().mean()
        eligible = [c for c in retained_sensors if float(miss[c]) <= self.max_missing_rate]
        if not eligible:
            eligible = list(retained_sensors)

        passing_mask = y_window == 0
        pass_df = window_df.loc[passing_mask, eligible] if np.any(passing_mask) else window_df[eligible]

        scored_sensors: list[tuple[float, str]] = []
        for col in eligible:
            s_pass = pass_df[col].dropna()
            if len(s_pass) < 3 or float(s_pass.std(ddof=1)) == 0.0:
                continue
            vals = window_df[col].fillna(float(s_pass.median())).to_numpy(dtype=np.float64)
            if np.std(vals) == 0.0 or np.std(y_window) == 0.0:
                corr = 0.0
            else:
                r, _ = stats.pearsonr(vals, y_window)
                corr = abs(float(r)) if np.isfinite(r) else 0.0
            scored_sensors.append((corr, col))

        scored_sensors.sort(key=lambda item: (-item[0], item[1]))
        self.monitored_sensors = [col for _, col in scored_sensors[: self.top_m]]
        if not self.monitored_sensors:
            self.monitored_sensors = eligible[: min(self.top_m, len(eligible))]

        self.medians = {}
        self.scales = {}
        for col in self.monitored_sensors:
            vals = pass_df[col].dropna().to_numpy(dtype=np.float64)
            med = float(np.median(vals)) if len(vals) > 0 else 0.0
            mad = float(np.median(np.abs(vals - med))) if len(vals) > 0 else 0.0
            iqr_s = (
                float(np.percentile(vals, 75) - np.percentile(vals, 25)) / 1.349
                if len(vals) > 1
                else 0.0
            )
            std_s = float(np.std(vals, ddof=1)) if len(vals) > 1 else 1.0
            scale = 1.4826 * mad if 1.4826 * mad > 1e-6 else (iqr_s if iqr_s > 1e-6 else max(std_s, 1e-6))
            self.medians[col] = med
            self.scales[col] = scale
        return self

    def score_e02_and_e03(
        self,
        eval_df: pd.DataFrame,
        warmup_tail_df: pd.DataFrame | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (e02_v2_scores, e03_v2_scores) for eval_df."""
        n_eval = len(eval_df)
        m = len(self.monitored_sensors)
        if m == 0:
            return np.zeros(n_eval, dtype=np.float64), np.zeros(n_eval, dtype=np.float64)

        if warmup_tail_df is not None and len(warmup_tail_df) > 0:
            combined = pd.concat([warmup_tail_df[self.monitored_sensors], eval_df[self.monitored_sensors]], ignore_index=True)
            offset = len(warmup_tail_df)
        else:
            combined = eval_df[self.monitored_sensors].reset_index(drop=True)
            offset = 0

        n_comb = len(combined)
        abs_z_eval = np.zeros((n_eval, m), dtype=np.float64)
        abs_ewma_eval = np.zeros((n_eval, m), dtype=np.float64)
        lam = self.ewma_lambda

        for j, col in enumerate(self.monitored_sensors):
            med = self.medians[col]
            scale = self.scales[col]
            raw = combined[col].to_numpy(dtype=np.float64)
            nan_mask = np.isnan(raw)
            z_series = np.where(nan_mask, 0.0, (raw - med) / scale)
            abs_z_eval[:, j] = np.abs(z_series[offset:])

            ewma_norm = np.zeros(n_comb, dtype=np.float64)
            z_cur = 0.0
            t_step = 0
            for idx in range(n_comb):
                if not nan_mask[idx]:
                    t_step += 1
                    z_cur = lam * z_series[idx] + (1.0 - lam) * z_cur
                if t_step > 0:
                    sigma_z = np.sqrt((lam / (2.0 - lam)) * (1.0 - (1.0 - lam) ** (2 * t_step)))
                    ewma_norm[idx] = abs(z_cur) / max(sigma_z, 1e-9)
                else:
                    ewma_norm[idx] = 0.0
            abs_ewma_eval[:, j] = ewma_norm[offset:]

        e02_v2 = compute_composite_top3_score(abs_z_eval)
        e03_v2 = compute_composite_top3_score(abs_ewma_eval)
        return e02_v2, e03_v2


def _compute_supervised_oof_scores(
    X_scaled: np.ndarray,
    y: np.ndarray,
    seed: int = 42,
    xgb_params: dict[str, Any] | None = None,
    logreg_c: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute out-of-fold predicted probabilities on H_{k-1} for unlabelled threshold calibration."""
    n = len(y)
    n_fails = int(np.sum(y == 1))
    n_splits = min(3, max(2, n_fails))
    oof_xgb = np.zeros(n, dtype=np.float64)
    oof_lr = np.zeros(n, dtype=np.float64)

    if n_fails < 2 or n_fails >= n:
        return oof_xgb, oof_lr

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    params = xgb_params or {
        "max_depth": 2,
        "n_estimators": 100,
        "learning_rate": 0.05,
        "min_child_weight": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 5.0,
    }

    for tr_idx, val_idx in skf.split(X_scaled, y):
        y_tr = y[tr_idx]
        pos = int(np.sum(y_tr == 1))
        neg = int(np.sum(y_tr == 0))
        spw = float(neg / max(pos, 1))

        xgb = XGBClassifier(
            **params,
            scale_pos_weight=spw,
            eval_metric="aucpr",
            random_state=seed,
            n_jobs=1,
        )
        xgb.fit(X_scaled[tr_idx], y_tr)
        oof_xgb[val_idx] = xgb.predict_proba(X_scaled[val_idx])[:, 1]

        lr = LogisticRegression(
            C=logreg_c,
            class_weight="balanced",
            solver="lbfgs",
            max_iter=1000,
            random_state=seed,
        )
        lr.fit(X_scaled[tr_idx], y_tr)
        oof_lr[val_idx] = lr.predict_proba(X_scaled[val_idx])[:, 1]

    return oof_xgb, oof_lr


def run_prequential_evaluation(
    df: pd.DataFrame,
    sensor_cols: list[str],
    dataset_id: str,
    evaluation_type: str = "confirmatory_v0_2",
    n_blocks: int = 8,
    rolling_window_runs: int = 470,
    calibration_window_runs: int = 235,
    pca_variance: float = 0.80,
    pca_max_components: int = 25,
    logreg_c: float = 0.1,
    xgb_params: dict[str, Any] | None = None,
    bootstrap_iterations: int = 2000,
    bootstrap_block_length: int = 20,
    seed: int = 42,
    predictions_csv_path: str | Path | None = None,
    output_json_path: str | Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Execute the frozen v0.2 prequential (rolling-origin) evaluation across blocks B_2 .. B_K."""
    if output_json_path is not None:
        out_path = Path(output_json_path)
        if out_path.exists() and not overwrite:
            raise FileExistsError(
                f"Output report {out_path} already exists and overwrite=False. "
                "Confirmatory v0.2 evaluation is one-shot."
            )

    blocks = partition_prequential_blocks(df, n_blocks=n_blocks)
    detectors = [
        "E01_v2",
        "E02_v2",
        "E03_v2",
        "E04_v2",
        "E05_v2",
        "E06_v2",
        "E07_v2",
        "E08_v2",
        "E09_v2",
    ]
    budgets = [2.0, 5.0, 10.0]

    xgb_cfg = xgb_params or {
        "max_depth": 2,
        "n_estimators": 100,
        "learning_rate": 0.05,
        "min_child_weight": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 5.0,
    }

    scored_blocks: list[pd.DataFrame] = []
    block_summaries: list[dict[str, Any]] = []

    for k in range(1, len(blocks)):
        block_name = f"B{k + 1}"
        history_df = pd.concat(blocks[:k], ignore_index=True)
        eval_df = blocks[k].copy().reset_index(drop=True)

        y_hist = (history_df["label"].to_numpy(dtype=np.int64) == 1).astype(np.int64)
        y_eval = (eval_df["label"].to_numpy(dtype=np.int64) == 1).astype(np.int64)

        w_len = min(len(history_df), rolling_window_runs)
        window_df = history_df.iloc[-w_len:].copy().reset_index(drop=True)
        y_win = y_hist[-w_len:]

        c_len = min(len(window_df), calibration_window_runs)
        cal_offset = len(window_df) - c_len

        # 1. Fit train-only RobustPreprocessor on history_df
        preproc = RobustPreprocessor().fit(history_df, sensor_cols)
        win_imp = preproc.transform_imputed(window_df)
        eval_imp = preproc.transform_imputed(eval_df)

        X_hist_scaled = preproc.transform_scaled(history_df).to_numpy(dtype=np.float64)
        X_win_scaled_df = preproc.transform_scaled(window_df)
        X_eval_scaled_df = preproc.transform_scaled(eval_df)
        X_win_scaled = X_win_scaled_df.to_numpy(dtype=np.float64)
        X_eval_scaled = X_eval_scaled_df.to_numpy(dtype=np.float64)

        # 2. E01_v2: Historical fail prevalence
        hist_prev = float(np.mean(y_hist))
        cal_scores: dict[str, np.ndarray] = {
            "E01_v2": np.full(c_len, hist_prev, dtype=np.float64),
        }
        eval_scores: dict[str, np.ndarray] = {
            "E01_v2": np.full(len(eval_df), hist_prev, dtype=np.float64),
        }

        # 3. E02_v2 & E03_v2: Adaptive Univariate SPC on rolling window W_{k-1}
        spc = AdaptiveUnivariateSPC(top_m=10).fit(win_imp, preproc.retained_sensors, y_win)
        win_e02, win_e03 = spc.score_e02_and_e03(win_imp, warmup_tail_df=None)
        eval_e02, eval_e03 = spc.score_e02_and_e03(
            eval_imp,
            warmup_tail_df=win_imp.iloc[-min(30, len(win_imp)) :],
        )
        cal_scores["E02_v2"] = win_e02[cal_offset:]
        cal_scores["E03_v2"] = win_e03[cal_offset:]
        eval_scores["E02_v2"] = eval_e02
        eval_scores["E03_v2"] = eval_e03

        # 4. E04_v2, E05_v2, E06_v2: Adaptive Multivariate PCA SPC on passing runs in W_{k-1}
        mspc_model = fit_multivariate_spc(
            train_scaled_df=X_win_scaled_df,
            train_labels_fail=y_win,
            variance_explained_target=pca_variance,
            max_components=pca_max_components,
        )
        win_mspc = score_multivariate_spc(mspc_model, X_win_scaled_df)
        eval_mspc = score_multivariate_spc(mspc_model, X_eval_scaled_df)

        cal_scores["E04_v2"] = win_mspc["t2"][cal_offset:]
        cal_scores["E05_v2"] = win_mspc["spe"][cal_offset:]
        cal_scores["E06_v2"] = win_mspc["mspc_index"][cal_offset:]
        eval_scores["E04_v2"] = eval_mspc["t2"]
        eval_scores["E05_v2"] = eval_mspc["spe"]
        eval_scores["E06_v2"] = eval_mspc["mspc_index"]

        # 5. E07_v2: Adaptive Isolation Forest on passing runs in W_{k-1}
        pass_win_arr = X_win_scaled[y_win == 0]
        if len(pass_win_arr) < 5:
            pass_win_arr = X_win_scaled
        iforest = IsolationForest(
            n_estimators=200,
            contamination="auto",
            random_state=seed,
            n_jobs=1,
        )
        iforest.fit(pass_win_arr)
        win_if = -iforest.score_samples(X_win_scaled)
        eval_if = -iforest.score_samples(X_eval_scaled)
        cal_scores["E07_v2"] = win_if[cal_offset:]
        eval_scores["E07_v2"] = eval_if

        # 6. E08_v2 (Class-Weighted XGBoost) & E09_v2 (L2-Regularised Logistic Regression)
        oof_xgb, oof_lr = _compute_supervised_oof_scores(
            X_hist_scaled,
            y_hist,
            seed=seed + k,
            xgb_params=xgb_cfg,
            logreg_c=logreg_c,
        )
        cal_scores["E08_v2"] = oof_xgb[-c_len:]
        cal_scores["E09_v2"] = oof_lr[-c_len:]

        pos_hist = int(np.sum(y_hist == 1))
        neg_hist = int(np.sum(y_hist == 0))
        spw = float(neg_hist / max(pos_hist, 1))

        xgb_model = XGBClassifier(
            **xgb_cfg,
            scale_pos_weight=spw,
            eval_metric="aucpr",
            random_state=seed,
            n_jobs=1,
        )
        xgb_model.fit(X_hist_scaled, y_hist)
        eval_scores["E08_v2"] = xgb_model.predict_proba(X_eval_scaled)[:, 1].astype(np.float64)

        lr_model = LogisticRegression(
            C=logreg_c,
            class_weight="balanced",
            solver="lbfgs",
            max_iter=1000,
            random_state=seed,
        )
        lr_model.fit(X_hist_scaled, y_hist)
        eval_scores["E09_v2"] = lr_model.predict_proba(X_eval_scaled)[:, 1].astype(np.float64)

        # 7. Calibrate unlabelled alarm thresholds on C_{k-1} and flag B_k runs
        block_out = pd.DataFrame(
            {
                "run_id": eval_df["run_id"].astype(str) if "run_id" in eval_df.columns else [f"run_{i}" for i in range(len(eval_df))],
                "timestamp": eval_df["timestamp"].astype(str) if "timestamp" in eval_df.columns else [f"t_{i}" for i in range(len(eval_df))],
                "prequential_block": block_name,
                "label": y_eval,
            }
        )

        block_det_metrics: dict[str, Any] = {}
        for det in detectors:
            s_cal = cal_scores[det]
            s_ev = eval_scores[det]
            block_out[f"{det}_score"] = s_ev

            if det == "E01_v2":
                thresholds = {f"budget_{int(b)}": float(hist_prev + 1.0) for b in budgets}
            else:
                thresholds = compute_multi_budget_thresholds(s_cal, budgets_per_100=budgets)

            for b in budgets:
                b_key = f"budget_{int(b)}"
                thr = float(thresholds[b_key])
                block_out[f"{det}_alarm_b{int(b)}"] = s_ev > thr

            b5_eval = evaluate_at_threshold(
                y_true=y_eval,
                scores=s_ev,
                threshold=float(thresholds["budget_5"]),
                budget_per_100=5.0,
            )
            pr_val, _ = compute_pr_auc(y_eval, s_ev)
            block_det_metrics[det] = {
                "pr_auc": pr_val,
                "threshold_budget_5": float(thresholds["budget_5"]),
                "alarms_budget_5": int(b5_eval.alarms),
                "realised_alarms_per_100_b5": float(b5_eval.realised_alarms_per_100 or 0.0),
                "true_alarms_b5": int(b5_eval.true_alarms),
                "false_alarms_b5": int(b5_eval.false_alarms),
            }

        scored_blocks.append(block_out)
        block_summaries.append(
            {
                "block": block_name,
                "n_history_runs": int(len(history_df)),
                "n_window_runs": int(len(window_df)),
                "n_calibration_runs": int(c_len),
                "n_eval_runs": int(len(eval_df)),
                "n_eval_fails": int(np.sum(y_eval == 1)),
                "eval_prevalence": float(np.mean(y_eval)),
                "n_retained_sensors": int(len(preproc.retained_sensors)),
                "monitored_spc_sensors": list(spc.monitored_sensors),
                "detectors": block_det_metrics,
            }
        )

    pooled_df = pd.concat(scored_blocks, ignore_index=True)

    predictions_sha256: str | None = None
    if predictions_csv_path is not None:
        pred_path = Path(predictions_csv_path)
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        pred_cols = [c for c in pooled_df.columns if c != "label"]
        pooled_df[pred_cols].to_csv(pred_path, index=False)
        predictions_sha256 = hashlib.sha256(pred_path.read_bytes()).hexdigest()

    y_pooled = pooled_df["label"].to_numpy(dtype=np.int64)
    n_pooled = int(len(y_pooled))
    n_fails_pooled = int(np.sum(y_pooled == 1))
    prevalence_pooled = float(np.mean(y_pooled))

    detector_results: dict[str, Any] = {}
    for det in detectors:
        is_chance = det == "E01_v2"
        s_pool = pooled_df[f"{det}_score"].to_numpy(dtype=np.float64)
        pr_val, _ = compute_pr_auc(y_pooled, s_pool)
        roc_val, _ = compute_roc_auc(y_pooled, s_pool)
        pr_auc = float(pr_val or 0.0)
        roc_auc = float(roc_val or 0.5)

        rec_at_2 = float(
            compute_recall_at_fixed_far(y_pooled, s_pool, far_per_100=2.0, is_chance_baseline=is_chance)["recall"] or 0.0
        )
        rec_at_5 = float(
            compute_recall_at_fixed_far(y_pooled, s_pool, far_per_100=5.0, is_chance_baseline=is_chance)["recall"] or 0.0
        )
        rec_at_10 = float(
            compute_recall_at_fixed_far(y_pooled, s_pool, far_per_100=10.0, is_chance_baseline=is_chance)["recall"] or 0.0
        )

        strat_ci = stratified_bootstrap_ci(
            y_pooled,
            s_pool,
            n_bootstraps=bootstrap_iterations,
            seed=seed,
            far_target_per_100=5.0,
            is_chance_baseline=is_chance,
        )
        mbb_ci = moving_block_bootstrap_ci(
            y_pooled,
            s_pool,
            block_length=bootstrap_block_length,
            n_bootstraps=bootstrap_iterations,
            seed=seed,
            far_target_per_100=5.0,
            is_chance_baseline=is_chance,
        )

        strat_pr_ci = [float(strat_ci["pr_auc_95_ci"][0]), float(strat_ci["pr_auc_95_ci"][1])]
        mbb_pr_ci = [float(mbb_ci["pr_auc_95_ci"][0]), float(mbb_ci["pr_auc_95_ci"][1])]
        wider_pr = [
            float(min(strat_pr_ci[0], mbb_pr_ci[0])),
            float(max(strat_pr_ci[1], mbb_pr_ci[1])),
        ]

        strat_rec5_ci = [float(strat_ci["recall_at_far_95_ci"][0]), float(strat_ci["recall_at_far_95_ci"][1])]
        mbb_rec5_ci = [float(mbb_ci["recall_at_far_95_ci"][0]), float(mbb_ci["recall_at_far_95_ci"][1])]
        wider_rec5 = [
            float(min(strat_rec5_ci[0], mbb_rec5_ci[0])),
            float(max(strat_rec5_ci[1], mbb_rec5_ci[1])),
        ]

        budget_evals: dict[str, Any] = {}
        for b in budgets:
            b_int = int(b)
            alarm_flags = pooled_df[f"{det}_alarm_b{b_int}"].to_numpy(dtype=bool)
            n_alarms = int(np.sum(alarm_flags))
            tp = int(np.sum(alarm_flags & (y_pooled == 1)))
            fp = int(np.sum(alarm_flags & (y_pooled == 0)))
            fn = int(np.sum((~alarm_flags) & (y_pooled == 1)))
            realised_rate = float(100.0 * n_alarms / max(n_pooled, 1))
            prec = float(tp / n_alarms) if n_alarms > 0 else 0.0
            rec = float(tp / max(tp + fn, 1))
            budget_evals[f"budget_{b_int}"] = {
                "target_budget_per_100": float(b),
                "alarms": n_alarms,
                "realised_alarms_per_100": realised_rate,
                "true_alarms": tp,
                "false_alarms": fp,
                "precision": prec,
                "recall": rec,
                "within_2_to_8_band": bool(2.0 <= realised_rate <= 8.0) if b_int == 5 else None,
            }

        block_pr_aucs = [
            bs["detectors"][det]["pr_auc"]
            for bs in block_summaries
            if bs["detectors"][det]["pr_auc"] is not None
        ]
        mean_block_pr_auc = float(np.mean(block_pr_aucs)) if block_pr_aucs else pr_auc

        detector_results[det] = {
            "pooled_pr_auc": pr_auc,
            "mean_block_pr_auc": mean_block_pr_auc,
            "pooled_roc_auc": roc_auc,
            "lift_over_prevalence": float(pr_auc / max(prevalence_pooled, 1e-9)),
            "stratified_95_ci_pr_auc": strat_pr_ci,
            "moving_block_95_ci_pr_auc": mbb_pr_ci,
            "wider_95_ci_pr_auc": wider_pr,
            "recall_at_2_far": rec_at_2,
            "recall_at_5_far": rec_at_5,
            "recall_at_10_far": rec_at_10,
            "stratified_95_ci_recall_at_5_far": strat_rec5_ci,
            "moving_block_95_ci_recall_at_5_far": mbb_rec5_ci,
            "wider_95_ci_recall_at_5_far": wider_rec5,
            "alarm_budgets": budget_evals,
        }

    spc_candidates = ["E02_v2", "E03_v2", "E04_v2", "E05_v2", "E06_v2"]
    ml_candidates = ["E07_v2", "E08_v2", "E09_v2"]

    best_spc = max(spc_candidates, key=lambda d: detector_results[d]["pooled_pr_auc"])
    best_ml = max(ml_candidates, key=lambda d: detector_results[d]["pooled_pr_auc"])

    ml_gain = detector_results[best_ml]["pooled_pr_auc"] - detector_results[best_spc]["pooled_pr_auc"]
    ml_b5_ok = bool(detector_results[best_ml]["alarm_budgets"]["budget_5"]["within_2_to_8_band"])
    primary_detector = best_ml if (ml_gain >= 0.02 and ml_b5_ok) else best_spc

    prim_res = detector_results[primary_detector]
    best_overall = max(detectors[1:], key=lambda d: detector_results[d]["pooled_pr_auc"])
    best_res = detector_results[best_overall]

    paired_diff = paired_stratified_bootstrap_diff(
        y_pooled,
        pooled_df[f"{best_ml}_score"].to_numpy(dtype=np.float64),
        pooled_df[f"{best_spc}_score"].to_numpy(dtype=np.float64),
        n_bootstraps=bootstrap_iterations,
        seed=seed,
    )
    diff_ci = [float(paired_diff["diff_95_ci"][0]), float(paired_diff["diff_95_ci"][1])]

    o1_pass = bool(
        prim_res["pooled_pr_auc"] >= 2.0 * prevalence_pooled
        and prim_res["wider_95_ci_pr_auc"][0] > prevalence_pooled
    )
    o2_pass = bool(
        prim_res["recall_at_5_far"] >= 0.20
        and prim_res["wider_95_ci_recall_at_5_far"][0] > 0.05
    )
    prim_b5 = prim_res["alarm_budgets"]["budget_5"]
    o3_pass = bool(
        2.0 <= prim_b5["realised_alarms_per_100"] <= 8.0
        and prim_b5["true_alarms"] >= 1
    )

    report: dict[str, Any] = {
        "protocol_version": "0.2.0",
        "dataset_id": dataset_id,
        "evaluation_type": evaluation_type,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "predictions_sha256": predictions_sha256,
        "prequential_summary": {
            "n_total_blocks": len(blocks),
            "n_evaluated_blocks": len(blocks) - 1,
            "n_evaluated_runs": n_pooled,
            "n_evaluated_fails": n_fails_pooled,
            "evaluated_prevalence": prevalence_pooled,
            "rolling_window_runs": rolling_window_runs,
            "calibration_window_runs": calibration_window_runs,
        },
        "selection": {
            "primary_detector": primary_detector,
            "best_overall_detector": best_overall,
            "best_spc_detector": best_spc,
            "best_ml_detector": best_ml,
            "diff_best_ml_minus_best_spc": float(ml_gain),
            "diff_95_ci": diff_ci,
        },
        "target_outcomes_v0_2": {
            "O1_v2_prequential_pr_auc": {
                "verdict": "PASS" if o1_pass else "MISS",
                "primary_detector": primary_detector,
                "actual_pr_auc": prim_res["pooled_pr_auc"],
                "best_detector": best_overall,
                "best_detector_pr_auc": best_res["pooled_pr_auc"],
                "wider_95_ci": prim_res["wider_95_ci_pr_auc"],
                "target_min_pr_auc_2x_prevalence": float(2.0 * prevalence_pooled),
                "chance_prevalence": prevalence_pooled,
                "n_fails": n_fails_pooled,
            },
            "O2_v2_recall_at_5_far": {
                "verdict": "PASS" if o2_pass else "MISS",
                "primary_detector": primary_detector,
                "actual_recall_at_5_far": prim_res["recall_at_5_far"],
                "best_detector": best_overall,
                "best_detector_recall_at_5_far": best_res["recall_at_5_far"],
                "wider_95_ci": prim_res["wider_95_ci_recall_at_5_far"],
                "target_min_recall": 0.20,
                "chance_recall": 0.05,
                "n_fails": n_fails_pooled,
            },
            "O3_v2_adaptive_alarm_budget": {
                "verdict": "PASS" if o3_pass else "MISS",
                "primary_detector": primary_detector,
                "target_budget_per_100": 5.0,
                "allowed_range_per_100": [2.0, 8.0],
                "realised_alarms_per_100": prim_b5["realised_alarms_per_100"],
                "alarms": prim_b5["alarms"],
                "true_alarms": prim_b5["true_alarms"],
                "false_alarms": prim_b5["false_alarms"],
                "precision": prim_b5["precision"],
                "recall": prim_b5["recall"],
                "n_runs": n_pooled,
            },
            "O4_v2_spc_vs_ml_comparison": {
                "verdict": "REPORTED",
                "best_ml_detector": best_ml,
                "best_spc_detector": best_spc,
                "best_ml_pr_auc": detector_results[best_ml]["pooled_pr_auc"],
                "best_spc_pr_auc": detector_results[best_spc]["pooled_pr_auc"],
                "e09_logreg_pr_auc": detector_results["E09_v2"]["pooled_pr_auc"],
                "diff_ml_minus_spc": float(ml_gain),
                "stratified_95_ci_diff": diff_ci,
                "n_fails": n_fails_pooled,
            },
        },
        "detectors": detector_results,
        "per_block_summaries": block_summaries,
    }

    if output_json_path is not None:
        out_path = Path(output_json_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    return report
