"""Model training, bounded XGBoost search (E08), forward-fold evaluation (E01-E08), and MLflow tracking."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
import xgboost as xgb

from yieldguard.anomaly import fit_isolation_forest, score_isolation_forest
from yieldguard.config import Config, load_config
from yieldguard.data import SENSOR_COLUMNS
from yieldguard.evaluate import compute_pr_auc, compute_recall_at_fixed_far, compute_roc_auc
from yieldguard.mspc import fit_multivariate_spc, score_multivariate_spc
from yieldguard.policy import evaluate_at_threshold
from yieldguard.quality import fit_quality_transformer
from yieldguard.spc import fit_univariate_spc, score_univariate_spc
from yieldguard.splits import FORWARD_FOLDS, load_development_labels


@dataclass(frozen=True)
class XgbCandidateConfig:
    """Single predeclared XGBoost hyperparameter configuration within the Section 3 search budget."""

    config_id: str
    max_depth: int
    learning_rate: float
    n_estimators: int
    subsample: float
    colsample_bytree: float
    min_child_weight: int
    seed: int = 42


# Predeclared list of 16 bounded XGBoost configurations (<= 20 limit in docs/05_EXPERIMENT_PLAN.md Section 3).
# Bounds: max_depth in [1, 3], learning_rate in [0.02, 0.10], n_estimators in [100, 400],
# subsample in [0.7, 1.0], colsample_bytree in [0.3, 1.0], min_child_weight in [1, 10].
PREDECLARED_XGB_CONFIGS: tuple[XgbCandidateConfig, ...] = (
    XgbCandidateConfig("xgb_c01", max_depth=1, learning_rate=0.03, n_estimators=100, subsample=0.8, colsample_bytree=0.5, min_child_weight=5, seed=42),
    XgbCandidateConfig("xgb_c02", max_depth=1, learning_rate=0.05, n_estimators=150, subsample=0.8, colsample_bytree=0.5, min_child_weight=3, seed=42),
    XgbCandidateConfig("xgb_c03", max_depth=1, learning_rate=0.05, n_estimators=200, subsample=0.9, colsample_bytree=0.7, min_child_weight=5, seed=42),
    XgbCandidateConfig("xgb_c04", max_depth=1, learning_rate=0.08, n_estimators=250, subsample=0.8, colsample_bytree=0.3, min_child_weight=8, seed=42),
    XgbCandidateConfig("xgb_c05", max_depth=1, learning_rate=0.10, n_estimators=300, subsample=0.7, colsample_bytree=0.5, min_child_weight=10, seed=42),
    XgbCandidateConfig("xgb_c06", max_depth=2, learning_rate=0.02, n_estimators=150, subsample=0.8, colsample_bytree=0.5, min_child_weight=5, seed=42),
    XgbCandidateConfig("xgb_c07", max_depth=2, learning_rate=0.03, n_estimators=200, subsample=0.8, colsample_bytree=0.5, min_child_weight=3, seed=42),
    XgbCandidateConfig("xgb_c08", max_depth=2, learning_rate=0.05, n_estimators=150, subsample=0.9, colsample_bytree=0.7, min_child_weight=5, seed=42),
    XgbCandidateConfig("xgb_c09", max_depth=2, learning_rate=0.05, n_estimators=250, subsample=0.8, colsample_bytree=0.3, min_child_weight=3, seed=42),
    XgbCandidateConfig("xgb_c10", max_depth=2, learning_rate=0.08, n_estimators=200, subsample=0.7, colsample_bytree=0.5, min_child_weight=8, seed=42),
    XgbCandidateConfig("xgb_c11", max_depth=2, learning_rate=0.10, n_estimators=300, subsample=0.8, colsample_bytree=0.5, min_child_weight=10, seed=42),
    XgbCandidateConfig("xgb_c12", max_depth=3, learning_rate=0.02, n_estimators=200, subsample=0.8, colsample_bytree=0.5, min_child_weight=5, seed=42),
    XgbCandidateConfig("xgb_c13", max_depth=3, learning_rate=0.03, n_estimators=250, subsample=0.8, colsample_bytree=0.3, min_child_weight=5, seed=42),
    XgbCandidateConfig("xgb_c14", max_depth=3, learning_rate=0.05, n_estimators=200, subsample=0.9, colsample_bytree=0.5, min_child_weight=3, seed=42),
    XgbCandidateConfig("xgb_c15", max_depth=3, learning_rate=0.05, n_estimators=350, subsample=0.8, colsample_bytree=0.7, min_child_weight=8, seed=42),
    XgbCandidateConfig("xgb_c16", max_depth=3, learning_rate=0.08, n_estimators=400, subsample=0.7, colsample_bytree=0.3, min_child_weight=10, seed=42),
)


def validate_xgb_search_space(configs: tuple[XgbCandidateConfig, ...] = PREDECLARED_XGB_CONFIGS) -> None:
    """Verify that the predeclared XGBoost search space obeys all Section 3 bounds."""
    if len(configs) == 0 or len(configs) > 20:
        raise ValueError(f"Expected 1..20 XGBoost configs, got {len(configs)}")
    for cfg in configs:
        if not (1 <= cfg.max_depth <= 3):
            raise ValueError(f"{cfg.config_id}: max_depth={cfg.max_depth} outside [1, 3]")
        if not (0.02 <= cfg.learning_rate <= 0.10):
            raise ValueError(f"{cfg.config_id}: learning_rate={cfg.learning_rate} outside [0.02, 0.10]")
        if not (100 <= cfg.n_estimators <= 400):
            raise ValueError(f"{cfg.config_id}: n_estimators={cfg.n_estimators} outside [100, 400]")
        if not (0.7 <= cfg.subsample <= 1.0):
            raise ValueError(f"{cfg.config_id}: subsample={cfg.subsample} outside [0.7, 1.0]")
        if not (0.3 <= cfg.colsample_bytree <= 1.0):
            raise ValueError(f"{cfg.config_id}: colsample_bytree={cfg.colsample_bytree} outside [0.3, 1.0]")
        if not (1 <= cfg.min_child_weight <= 10):
            raise ValueError(f"{cfg.config_id}: min_child_weight={cfg.min_child_weight} outside [1, 10]")


def compute_scale_pos_weight(y_train: np.ndarray | list[int]) -> float:
    """Compute training-fold negative-to-positive ratio (`n_neg / n_pos`)."""
    y = np.asarray(y_train, dtype=np.int64)
    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))
    if n_pos <= 0:
        raise ValueError("Training fold has zero positive (fail) labels; cannot compute scale_pos_weight.")
    return float(n_neg / n_pos)


def fit_xgb_candidate(
    x_train_df: pd.DataFrame,
    y_train: np.ndarray | list[int],
    cfg: XgbCandidateConfig,
) -> tuple[xgb.XGBClassifier, float]:
    """Fit a single predeclared XGBoost candidate with fold-specific `scale_pos_weight`."""
    y = np.asarray(y_train, dtype=np.int64)
    spw = compute_scale_pos_weight(y)
    clf = xgb.XGBClassifier(
        max_depth=cfg.max_depth,
        learning_rate=cfg.learning_rate,
        n_estimators=cfg.n_estimators,
        subsample=cfg.subsample,
        colsample_bytree=cfg.colsample_bytree,
        min_child_weight=cfg.min_child_weight,
        scale_pos_weight=spw,
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        random_state=cfg.seed,
        n_jobs=1,
    )
    clf.fit(x_train_df, y)
    return clf, spw


def score_xgb_candidate(model: xgb.XGBClassifier, x_df: pd.DataFrame) -> np.ndarray:
    """Return predicted failure probability (`P(label_fail == 1)`) for each run."""
    probs = model.predict_proba(x_df)[:, 1]
    return np.asarray(probs, dtype=np.float64)


def select_xgb_config_from_inner_folds(
    inner_fold_pr_aucs: dict[str, list[float]],
    configs: tuple[XgbCandidateConfig, ...] = PREDECLARED_XGB_CONFIGS,
) -> tuple[XgbCandidateConfig, dict[str, Any]]:
    """Select the E08 configuration using inner forward folds (`fold_1`, `fold_2` only).

    Per docs/05_EXPERIMENT_PLAN.md Section 4 Rule 3 and tasks/todo.md T09:
    - Highest mean fold PR-AUC across inner tuning folds (`fold_1` on D3, `fold_2` on D4).
    - Within one standard error of the best mean PR-AUC, prefer the shallower model (`max_depth`)
      with fewer trees (`n_estimators`), then higher `min_child_weight`.
    - Because only inner folds (`D1-D4`) are used, `D5` and `holdout` cannot affect the choice.
    """
    cfg_map = {c.config_id: c for c in configs}
    summary_rows: list[dict[str, Any]] = []

    for cfg in configs:
        vals = inner_fold_pr_aucs.get(cfg.config_id, [])
        if len(vals) == 0:
            raise ValueError(f"Missing inner fold PR-AUC values for {cfg.config_id}")
        arr = np.asarray(vals, dtype=np.float64)
        mean_val = float(np.mean(arr))
        se_val = float(np.std(arr, ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0
        summary_rows.append(
            {
                "config_id": cfg.config_id,
                "max_depth": cfg.max_depth,
                "n_estimators": cfg.n_estimators,
                "learning_rate": cfg.learning_rate,
                "subsample": cfg.subsample,
                "colsample_bytree": cfg.colsample_bytree,
                "min_child_weight": cfg.min_child_weight,
                "inner_fold_pr_aucs": [float(v) for v in vals],
                "mean_inner_pr_auc": mean_val,
                "se_inner_pr_auc": se_val,
            }
        )

    best_row = max(summary_rows, key=lambda r: (r["mean_inner_pr_auc"], -r["max_depth"], -r["n_estimators"]))
    one_se_threshold = float(best_row["mean_inner_pr_auc"] - best_row["se_inner_pr_auc"])

    eligible = [r for r in summary_rows if r["mean_inner_pr_auc"] >= one_se_threshold - 1e-12]
    eligible.sort(
        key=lambda r: (
            r["max_depth"],
            r["n_estimators"],
            -r["min_child_weight"],
            -r["mean_inner_pr_auc"],
            r["config_id"],
        )
    )
    chosen_row = eligible[0]
    chosen_cfg = cfg_map[chosen_row["config_id"]]

    selection_report = {
        "best_mean_config_id": best_row["config_id"],
        "best_mean_inner_pr_auc": best_row["mean_inner_pr_auc"],
        "best_se_inner_pr_auc": best_row["se_inner_pr_auc"],
        "one_se_threshold": one_se_threshold,
        "eligible_config_ids": [r["config_id"] for r in eligible],
        "selected_config_id": chosen_cfg.config_id,
        "selected_config": asdict(chosen_cfg),
        "candidates": summary_rows,
    }
    return chosen_cfg, selection_report


def select_xgb_config_on_dev_blocks(
    runs_df: pd.DataFrame,
    labels_dev_df: pd.DataFrame,
    cfg: Config,
    configs: tuple[XgbCandidateConfig, ...] = PREDECLARED_XGB_CONFIGS,
) -> tuple[XgbCandidateConfig, dict[str, Any]]:
    """Run inner folds (`fold_1`: D1-D2 -> D3, `fold_2`: D1-D3 -> D4) to select the E08 config.

    Explicitly ignores `D5` (`fold_3`) and `holdout` so neither can influence hyperparameter selection.
    """
    validate_xgb_search_space(configs)
    sensor_cols = [c for c in SENSOR_COLUMNS if c in runs_df.columns]
    dev_merged = runs_df.merge(
        labels_dev_df[["run_id", "label_fail"]],
        on="run_id",
        how="inner",
    ).sort_values("run_id").reset_index(drop=True)

    inner_folds = [f for f in FORWARD_FOLDS if f["fold_id"] in ("fold_1", "fold_2")]
    inner_pr_aucs: dict[str, list[float]] = {c.config_id: [] for c in configs}

    for fold in inner_folds:
        train_blocks = set(fold["train_blocks"])
        eval_block = fold["eval_block"]
        assert "D5" not in train_blocks and eval_block != "D5"

        train_df = dev_merged[dev_merged["block"].isin(train_blocks)].reset_index(drop=True)
        val_df = dev_merged[dev_merged["block"] == eval_block].reset_index(drop=True)

        y_train = train_df["label_fail"].to_numpy(dtype=np.int64)
        y_val = val_df["label_fail"].to_numpy(dtype=np.int64)

        qt = fit_quality_transformer(
            train_df[sensor_cols],
            y_train,
            max_missing_fraction=cfg.quality.max_missing_fraction,
            max_constant_dominant_fraction=cfg.quality.max_constant_dominant_fraction,
            collinearity_threshold=cfg.quality.collinearity_threshold,
            sensor_columns=sensor_cols,
        )
        x_train_xgb = qt.transform_retained(train_df)
        x_val_xgb = qt.transform_retained(val_df)

        for cand in configs:
            model, _ = fit_xgb_candidate(x_train_xgb, y_train, cand)
            scores = score_xgb_candidate(model, x_val_xgb)
            ap, _ = compute_pr_auc(y_val, scores)
            inner_pr_aucs[cand.config_id].append(float(ap or 0.0))

    return select_xgb_config_from_inner_folds(inner_pr_aucs, configs=configs)


def _compute_fold_row(
    fold_id: str,
    eval_block: str,
    detector_id: str,
    detector_name: str,
    family: str,
    y_true: np.ndarray,
    scores: np.ndarray,
    eligible_for_primary: bool = True,
    is_diagnostic: bool = False,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    n_runs = int(len(y_true))
    n_fails = int(np.sum(y_true == 1))
    n_pass = int(np.sum(y_true == 0))
    prevalence = float(n_fails / n_runs)
    is_chance = detector_id == "E01"
    pr_auc, _ = compute_pr_auc(y_true, scores)
    roc_auc, _ = compute_roc_auc(y_true, scores)
    rec_far_2 = compute_recall_at_fixed_far(y_true, scores, far_per_100=2.0, is_chance_baseline=is_chance)["recall"]
    rec_far_5 = compute_recall_at_fixed_far(y_true, scores, far_per_100=5.0, is_chance_baseline=is_chance)["recall"]
    rec_far_10 = compute_recall_at_fixed_far(y_true, scores, far_per_100=10.0, is_chance_baseline=is_chance)["recall"]

    row: dict[str, Any] = {
        "fold_id": fold_id,
        "eval_block": eval_block,
        "detector_id": detector_id,
        "detector_name": detector_name,
        "family": family,
        "eligible_for_primary": eligible_for_primary,
        "is_diagnostic": is_diagnostic,
        "n_runs": n_runs,
        "n_fails": n_fails,
        "n_passing": n_pass,
        "prevalence": prevalence,
        "pr_auc": pr_auc,
        "pr_auc_over_prevalence": float(pr_auc / prevalence) if (pr_auc is not None and prevalence > 0) else None,
        "roc_auc": roc_auc,
        "recall_at_far_2pct": rec_far_2,
        "recall_at_far_5pct": rec_far_5,
        "recall_at_far_10pct": rec_far_10,
    }
    if extra:
        row.update(extra)
    return row


def run_training_experiments(
    config_path: str | Path = "configs/default.yaml",
    experiment: str = "all",
    configs: tuple[XgbCandidateConfig, ...] = PREDECLARED_XGB_CONFIGS,
) -> dict[str, Any]:
    """Execute forward-fold evaluation for E01-E08 and declared diagnostics, log to MLflow, and stage fold-3 models."""
    cfg = load_config(config_path)
    validate_xgb_search_space(configs)

    runs_path = Path(cfg.data.runs_parquet)
    labels_dev_path = Path(cfg.data.labels_dev_parquet)
    split_manifest_path = Path(cfg.data.split_manifest)
    source_manifest_path = Path(cfg.data.source_manifest)

    runs_df = pd.read_parquet(runs_path)
    labels_dev_df = load_development_labels(labels_dev_path)
    split_manifest = json.loads(split_manifest_path.read_text(encoding="utf-8"))

    data_hash = "synthetic_or_unregistered"
    if source_manifest_path.exists():
        sm = json.loads(source_manifest_path.read_text(encoding="utf-8"))
        data_hash = sm.get("archive_sha256", data_hash)
    split_hash = split_manifest.get("manifest_sha256", "unknown")
    config_hash = hashlib.sha256(
        json.dumps(cfg.model_dump(), sort_keys=True).encode("utf-8")
    ).hexdigest()

    staging_dir = Path(cfg.models.staging_dir)
    staging_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = Path(cfg.models.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Save predeclared XGBoost configurations before fitting
    predeclared_path = staging_dir / "predeclared_xgb_configs.json"
    predeclared_path.write_text(
        json.dumps([asdict(c) for c in configs], indent=2) + "\n",
        encoding="utf-8",
    )

    sensor_cols = [c for c in SENSOR_COLUMNS if c in runs_df.columns]
    dev_merged = runs_df.merge(
        labels_dev_df[["run_id", "label_fail"]],
        on="run_id",
        how="inner",
    ).sort_values("run_id").reset_index(drop=True)

    fold_rows: list[dict[str, Any]] = []
    oof_records: list[pd.DataFrame] = []
    inner_xgb_pr_aucs: dict[str, list[float]] = {c.config_id: [] for c in configs}
    xgb_fold_scores_by_config: dict[str, dict[str, np.ndarray]] = {c.config_id: {} for c in configs}
    xgb_fold_models_by_config: dict[str, dict[str, tuple[xgb.XGBClassifier, float]]] = {
        c.config_id: {} for c in configs
    }

    fold3_artifacts: dict[str, Any] = {}

    for fold in split_manifest["forward_folds"]:
        fold_id = fold["fold_id"]
        train_blocks = set(fold["train_blocks"])
        eval_block = fold["eval_block"]

        train_df = dev_merged[dev_merged["block"].isin(train_blocks)].reset_index(drop=True)
        val_df = dev_merged[dev_merged["block"] == eval_block].reset_index(drop=True)

        y_train = train_df["label_fail"].to_numpy(dtype=np.int64)
        y_val = val_df["label_fail"].to_numpy(dtype=np.int64)

        # 1. Fit QualityTransformer on training fold only
        qt = fit_quality_transformer(
            train_df[sensor_cols],
            y_train,
            max_missing_fraction=cfg.quality.max_missing_fraction,
            max_constant_dominant_fraction=cfg.quality.max_constant_dominant_fraction,
            collinearity_threshold=cfg.quality.collinearity_threshold,
            sensor_columns=sensor_cols,
        )

        x_train_retained = qt.transform_retained(train_df)
        x_val_retained = qt.transform_retained(val_df)

        x_train_scaled = qt.transform_scaled(train_df)
        x_val_scaled = qt.transform_scaled(val_df)

        # E01: Prevalence baseline (constant score equal to training fold fail rate)
        train_prev = float(np.mean(y_train == 1))
        s_e01 = np.full(len(val_df), train_prev, dtype=np.float64)
        fold_rows.append(
            _compute_fold_row(fold_id, eval_block, "E01", "Prevalence baseline", "baseline", y_val, s_e01)
        )

        # E02 & E03: Univariate SPC (Individuals & EWMA)
        spc_model = fit_univariate_spc(
            x_train_retained,
            y_train,
            group_map=qt.representative_to_members,
            n_monitored=cfg.spc.n_monitored_sensors,
            max_sensor_missing_fraction=cfg.spc.max_sensor_missing_fraction,
            ewma_lambda=cfg.spc.ewma_lambda,
            ewma_l=cfg.spc.ewma_l,
            mr_d2=cfg.spc.mr_d2,
        )
        spc_val_scores = score_univariate_spc(spc_model, x_val_retained)
        s_e02 = spc_val_scores["e02_scores"]
        s_e03 = spc_val_scores["e03_scores"]
        fold_rows.append(
            _compute_fold_row(fold_id, eval_block, "E02", "Individuals chart (max |z|)", "SPC", y_val, s_e02)
        )
        fold_rows.append(
            _compute_fold_row(fold_id, eval_block, "E03", "EWMA (max standardised distance)", "SPC", y_val, s_e03)
        )

        # E04, E05, E06: Multivariate SPC (T2, SPE, Combined MSPC index)
        mspc_model = fit_multivariate_spc(
            x_train_scaled,
            y_train,
            variance_explained_target=cfg.mspc.variance_explained_target,
            max_components=cfg.mspc.max_components,
            phase1_trim_alpha=cfg.mspc.phase1_trim_alpha,
            limit_alpha=cfg.mspc.limit_alpha,
        )
        mspc_val_scores = score_multivariate_spc(mspc_model, x_val_scaled, include_contributions=False)
        s_e04 = mspc_val_scores["t2"]
        s_e05 = mspc_val_scores["spe"]
        s_e06 = mspc_val_scores["mspc_index"]
        fold_rows.append(
            _compute_fold_row(fold_id, eval_block, "E04", "Hotelling T-squared", "SPC", y_val, s_e04)
        )
        fold_rows.append(
            _compute_fold_row(fold_id, eval_block, "E05", "SPE / Q residual", "SPC", y_val, s_e05)
        )
        fold_rows.append(
            _compute_fold_row(fold_id, eval_block, "E06", "Combined MSPC index", "SPC", y_val, s_e06)
        )

        # E07: Isolation Forest on passing training runs
        iforest_model = fit_isolation_forest(
            x_train_scaled,
            y_train,
            n_estimators=cfg.anomaly.n_estimators,
            max_samples=cfg.anomaly.max_samples,
            seed=cfg.anomaly.seed,
            fit_on_passing_only=True,
        )
        s_e07 = score_isolation_forest(iforest_model, x_val_scaled)
        fold_rows.append(
            _compute_fold_row(fold_id, eval_block, "E07", "Isolation Forest (passing runs)", "ML", y_val, s_e07)
        )

        # Declared diagnostics (never eligible as primary)
        iforest_all = fit_isolation_forest(
            x_train_scaled,
            y_train,
            n_estimators=cfg.anomaly.n_estimators,
            max_samples=cfg.anomaly.max_samples,
            seed=cfg.anomaly.seed,
            fit_on_passing_only=False,
        )
        s_d_if_all = score_isolation_forest(iforest_all, x_val_scaled)
        fold_rows.append(
            _compute_fold_row(
                fold_id,
                eval_block,
                "D_IF_ALL",
                "Diagnostic: Isolation Forest (all training runs)",
                "diagnostic",
                y_val,
                s_d_if_all,
                eligible_for_primary=False,
                is_diagnostic=True,
            )
        )

        for lam, diag_id in ((0.1, "D_EWMA_L01"), (0.3, "D_EWMA_L03")):
            spc_lam = fit_univariate_spc(
                x_train_retained,
                y_train,
                group_map=qt.representative_to_members,
                n_monitored=cfg.spc.n_monitored_sensors,
                max_sensor_missing_fraction=cfg.spc.max_sensor_missing_fraction,
                ewma_lambda=lam,
                ewma_l=cfg.spc.ewma_l,
                mr_d2=cfg.spc.mr_d2,
            )
            s_lam = score_univariate_spc(spc_lam, x_val_retained)["e03_scores"]
            fold_rows.append(
                _compute_fold_row(
                    fold_id,
                    eval_block,
                    diag_id,
                    f"Diagnostic: EWMA (lambda={lam})",
                    "diagnostic",
                    y_val,
                    s_lam,
                    eligible_for_primary=False,
                    is_diagnostic=True,
                )
            )

        for var_rule, diag_id in ((0.70, "D_PCA_V70"), (0.90, "D_PCA_V90")):
            mspc_var = fit_multivariate_spc(
                x_train_scaled,
                y_train,
                variance_explained_target=var_rule,
                max_components=cfg.mspc.max_components,
                phase1_trim_alpha=cfg.mspc.phase1_trim_alpha,
                limit_alpha=cfg.mspc.limit_alpha,
            )
            s_var = score_multivariate_spc(mspc_var, x_val_scaled)["mspc_index"]
            act_pct = int(round(mspc_var.cumulative_variance_explained * 100))
            fold_rows.append(
                _compute_fold_row(
                    fold_id,
                    eval_block,
                    diag_id,
                    f"Diagnostic: MSPC index (PCA rule={int(var_rule * 100)}%, actual={act_pct}%, A={mspc_var.n_components})",
                    "diagnostic",
                    y_val,
                    s_var,
                    eligible_for_primary=False,
                    is_diagnostic=True,
                    extra={
                        "target_variance": var_rule,
                        "actual_cumulative_variance": mspc_var.cumulative_variance_explained,
                        "n_components": mspc_var.n_components,
                        "max_components_cap_hit": bool(mspc_var.n_components == cfg.mspc.max_components),
                    },
                )
            )

        # Fit all 16 predeclared XGBoost candidates on this fold
        for cand in configs:
            xgb_clf, spw = fit_xgb_candidate(x_train_retained, y_train, cand)
            s_cand = score_xgb_candidate(xgb_clf, x_val_retained)
            xgb_fold_scores_by_config[cand.config_id][fold_id] = s_cand
            xgb_fold_models_by_config[cand.config_id][fold_id] = (xgb_clf, spw)

            cand_pr_auc, _ = compute_pr_auc(y_val, s_cand)
            if fold_id in ("fold_1", "fold_2"):
                inner_xgb_pr_aucs[cand.config_id].append(float(cand_pr_auc or 0.0))

            fold_rows.append(
                _compute_fold_row(
                    fold_id,
                    eval_block,
                    cand.config_id,
                    f"XGBoost candidate {cand.config_id} (d={cand.max_depth}, n={cand.n_estimators}, lr={cand.learning_rate})",
                    "xgb_candidate",
                    y_val,
                    s_cand,
                    eligible_for_primary=False,
                    is_diagnostic=False,
                    extra={
                        "scale_pos_weight": spw,
                        "max_depth": cand.max_depth,
                        "n_estimators": cand.n_estimators,
                        "learning_rate": cand.learning_rate,
                    },
                )
            )

        oof_df = pd.DataFrame(
            {
                "run_id": val_df["run_id"].to_numpy(dtype=np.int64),
                "timestamp": val_df["timestamp"].astype(str).to_numpy(),
                "block": val_df["block"].astype(str).to_numpy(),
                "fold_id": fold_id,
                "label_fail": y_val,
                "E01": s_e01,
                "E02": s_e02,
                "E03": s_e03,
                "E04": s_e04,
                "E05": s_e05,
                "E06": s_e06,
                "E07": s_e07,
            }
        )
        oof_records.append(oof_df)

        if fold_id == "fold_3":
            # Compute textbook SPC limit diagnostics on D5 (fold_3 evaluation block)
            textbook_d5 = {
                "E02_individuals_3sigma": evaluate_at_threshold(
                    y_val, s_e02, 3.0, budget_per_100=0.27
                ).to_dict(),
                "E03_ewma_L3": evaluate_at_threshold(
                    y_val, s_e03, spc_model.ewma_l, budget_per_100=0.27
                ).to_dict(),
                "E04_t2_99pct": evaluate_at_threshold(
                    y_val, s_e04, mspc_model.t2_limit_99, budget_per_100=1.0
                ).to_dict(),
                "E05_spe_99pct": evaluate_at_threshold(
                    y_val, s_e05, mspc_model.spe_limit_99, budget_per_100=1.0
                ).to_dict(),
            }
            fold3_artifacts = {
                "quality": qt,
                "spc": spc_model,
                "mspc": mspc_model,
                "iforest": iforest_model,
                "x_train_retained_columns": list(x_train_retained.columns),
                "x_train_retained": x_train_retained,
                "textbook_d5": textbook_d5,
                "train_fail_prevalence": train_prev,
            }

    # Step 2: Select E08 configuration strictly from fold_1 and fold_2 (inner folds, D1-D4 only)
    selected_xgb_cfg, xgb_selection_report = select_xgb_config_from_inner_folds(
        inner_xgb_pr_aucs,
        configs=configs,
    )

    # Step 3: Populate E08 fold rows and attach E08 OOF scores
    for i, fold in enumerate(split_manifest["forward_folds"]):
        fold_id = fold["fold_id"]
        eval_block = fold["eval_block"]
        s_e08 = xgb_fold_scores_by_config[selected_xgb_cfg.config_id][fold_id]
        _, spw = xgb_fold_models_by_config[selected_xgb_cfg.config_id][fold_id]
        y_val = oof_records[i]["label_fail"].to_numpy(dtype=np.int64)
        oof_records[i]["E08"] = s_e08

        fold_rows.append(
            _compute_fold_row(
                fold_id,
                eval_block,
                "E08",
                f"XGBoost with scale_pos_weight ({selected_xgb_cfg.config_id})",
                "ML",
                y_val,
                s_e08,
                eligible_for_primary=True,
                is_diagnostic=False,
                extra={
                    "selected_config_id": selected_xgb_cfg.config_id,
                    "scale_pos_weight": spw,
                    "max_depth": selected_xgb_cfg.max_depth,
                    "n_estimators": selected_xgb_cfg.n_estimators,
                    "learning_rate": selected_xgb_cfg.learning_rate,
                },
            )
        )

    oof_all_df = pd.concat(oof_records, ignore_index=True).sort_values("run_id").reset_index(drop=True)
    oof_path = staging_dir / "dev_oof_scores.parquet"
    oof_all_df.to_parquet(oof_path, index=False)

    comparison_df = pd.DataFrame(fold_rows)
    comparison_csv_path = reports_dir / "model_comparison.csv"
    comparison_df.to_csv(comparison_csv_path, index=False)

    # Step 4: Save fold-3 models (trained on D1-D4) to models/staging/
    fold3_artifacts["quality"].save(staging_dir / "quality.joblib")
    fold3_artifacts["spc"].save(staging_dir / "spc.json")
    fold3_artifacts["mspc"].save(staging_dir / "mspc.joblib")
    fold3_artifacts["iforest"].save(staging_dir / "iforest.joblib")

    (staging_dir / "spc_textbook_diagnostics.json").write_text(
        json.dumps(fold3_artifacts["textbook_d5"], indent=2) + "\n",
        encoding="utf-8",
    )

    fold3_xgb_model, fold3_spw = xgb_fold_models_by_config[selected_xgb_cfg.config_id]["fold_3"]
    xgb_model_path = staging_dir / "xgb_model.json"
    fold3_xgb_model.get_booster().save_model(str(xgb_model_path))

    xgb_meta_path = staging_dir / "xgb_metadata.json"
    xgb_meta_path.write_text(
        json.dumps(
            {
                "selected_config": asdict(selected_xgb_cfg),
                "fold3_scale_pos_weight": fold3_spw,
                "train_fail_prevalence": fold3_artifacts["train_fail_prevalence"],
                "feature_names": fold3_artifacts["x_train_retained_columns"],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    # Save reference monitoring table (retained sensor columns of D1-D4 with original NaNs for missing-rate comparison)
    ref_parquet_path = staging_dir / "monitoring_reference.parquet"
    fold3_artifacts["x_train_retained"].to_parquet(ref_parquet_path, index=False)

    # Step 5: Log to local MLflow (SQLite backend)
    mlflow.set_tracking_uri(cfg.models.tracking_uri)
    mlflow.set_experiment(cfg.project.name)

    with mlflow.start_run(run_name=f"dev_forward_folds_{experiment}") as active_run:
        mlflow_run_id = active_run.info.run_id
        mlflow.set_tags(
            {
                "data_sha256": data_hash,
                "split_sha256": split_hash,
                "config_sha256": config_hash,
                "seed": str(cfg.project.seed),
                "selected_xgb_config": selected_xgb_cfg.config_id,
            }
        )
        mlflow.log_params(
            {
                "seed": cfg.project.seed,
                "xgb_selected_config_id": selected_xgb_cfg.config_id,
                "xgb_max_depth": selected_xgb_cfg.max_depth,
                "xgb_n_estimators": selected_xgb_cfg.n_estimators,
                "xgb_learning_rate": selected_xgb_cfg.learning_rate,
                "xgb_subsample": selected_xgb_cfg.subsample,
                "xgb_colsample_bytree": selected_xgb_cfg.colsample_bytree,
                "xgb_min_child_weight": selected_xgb_cfg.min_child_weight,
                "fold3_scale_pos_weight": round(fold3_spw, 6),
            }
        )
        for det_id in ("E01", "E02", "E03", "E04", "E05", "E06", "E07", "E08"):
            sub = comparison_df[comparison_df["detector_id"] == det_id]
            mean_pr = float(sub["pr_auc"].mean())
            mlflow.log_metric(f"{det_id}_mean_pr_auc", mean_pr)
            for _, r in sub.iterrows():
                mlflow.log_metric(f"{det_id}_{r['fold_id']}_pr_auc", float(r["pr_auc"]))

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "mlflow_run_id": mlflow_run_id,
        "data_sha256": data_hash,
        "split_sha256": split_hash,
        "config_sha256": config_hash,
        "seed": cfg.project.seed,
        "n_predeclared_xgb_configs": len(configs),
        "xgb_selection": xgb_selection_report,
        "fold3_scale_pos_weight": fold3_spw,
        "oof_scores_path": str(oof_path.as_posix()),
        "model_comparison_csv": str(comparison_csv_path.as_posix()),
    }
    summary_path = staging_dir / "training_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary
