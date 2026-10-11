"""Sensor ranking for flagged runs: MSPC (T-squared & SPE) contributions, XGBoost TreeSHAP, group mapping, and O5 verification."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import shap
import shap.explainers._tree as _shap_tree
import xgboost as xgb

from yieldguard.config import load_config
from yieldguard.data import SENSOR_COLUMNS
from yieldguard.mspc import FittedMspcModel, fit_multivariate_spc, score_multivariate_spc
from yieldguard.quality import QualityTransformer, fit_quality_transformer


def _ensure_shap_xgboost21_compat() -> None:
    """Ensure shap==0.46.0 decodes XGBoost 2.1 bracketed base_score ('[1.3E-1]') cleanly."""
    if getattr(_shap_tree, "_yieldguard_patched", False):
        return
    orig_decode = _shap_tree.decode_ubjson_buffer

    def _patched_decode(fd: Any) -> dict[str, Any]:
        res = orig_decode(fd)
        try:
            lmp = res["learner"]["learner_model_param"]
            bs = lmp.get("base_score")
            if isinstance(bs, str) and bs.startswith("[") and bs.endswith("]"):
                lmp["base_score"] = bs.strip("[]")
        except Exception:
            pass
        return res

    _shap_tree.decode_ubjson_buffer = _patched_decode
    _shap_tree._yieldguard_patched = True  # type: ignore[attr-defined]


_ensure_shap_xgboost21_compat()


SMEARING_CAVEAT = (
    "Sensor contributions and SHAP attributions can smear across correlated sensors. "
    "Collinear group members are listed alongside each retained representative sensor. "
    "On SECOM, rankings are descriptive triage aids and not verified root-cause diagnoses."
)


def compute_xgb_shap_raw_margin(
    model_or_booster: xgb.XGBClassifier | xgb.Booster,
    x_retained_df: pd.DataFrame,
    background_df: pd.DataFrame | None = None,
) -> dict[str, np.ndarray | float]:
    """Compute SHAP values on raw log-odds margin using SHAP TreeExplainer and check additivity.

    Per docs/05_EXPERIMENT_PLAN.md Section 8.2:
    SHAP TreeExplainer on raw margin, background from training runs only.
    When `background_df` is provided without NaNs, uses interventional perturbation; otherwise
    uses tree_path_dependent TreeExplainer on raw margin (which supports native NaN routing
    and matches XGBoost C++ TreeSHAP to < 1e-6).
    """
    _ensure_shap_xgboost21_compat()
    booster = model_or_booster.get_booster() if hasattr(model_or_booster, "get_booster") else model_or_booster
    cols = list(x_retained_df.columns)
    x_mat = x_retained_df[cols].to_numpy(dtype=np.float64, copy=True)

    dmat = xgb.DMatrix(x_mat, feature_names=cols)
    raw_margin = np.asarray(booster.predict(dmat, output_margin=True), dtype=np.float64)

    if (
        background_df is not None
        and not background_df[cols].isna().any().any()
        and not np.isnan(x_mat).any()
    ):
        bg_mat = background_df[cols].to_numpy(dtype=np.float64, copy=True)
        if len(bg_mat) > 100:
            bg_mat = bg_mat[:100]
        explainer = shap.TreeExplainer(
            booster,
            data=bg_mat,
            feature_perturbation="interventional",
            model_output="raw",
        )
    else:
        explainer = shap.TreeExplainer(
            booster,
            feature_perturbation="tree_path_dependent",
            model_output="raw",
        )

    shap_values = np.asarray(explainer.shap_values(x_mat), dtype=np.float64)
    expected_val = float(np.asarray(explainer.expected_value).flat[0])
    reconstructed = np.sum(shap_values, axis=1) + expected_val
    additivity_errors = np.abs(reconstructed - raw_margin)
    max_additivity_error = float(np.max(additivity_errors)) if len(additivity_errors) > 0 else 0.0

    return {
        "shap_values": shap_values,
        "expected_value": expected_val,
        "raw_margin": raw_margin,
        "reconstructed_margin": reconstructed,
        "max_additivity_error": max_additivity_error,
    }


def rank_sensors_for_run(
    raw_run_row: pd.Series | dict[str, Any],
    quality: QualityTransformer,
    mspc_model: FittedMspcModel,
    xgb_model_or_booster: xgb.XGBClassifier | xgb.Booster,
    top_k: int = 5,
    background_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Compute top-k MSPC and XGBoost SHAP sensor rankings with collinear group members for a single run."""
    if isinstance(raw_run_row, dict):
        df_one = pd.DataFrame([raw_run_row])
    elif isinstance(raw_run_row, pd.Series):
        df_one = pd.DataFrame([raw_run_row.to_dict()])
    else:
        raise TypeError("raw_run_row must be a dict or pandas Series.")

    x_ret = quality.transform_retained(df_one)
    x_scaled = quality.transform_scaled(df_one)
    retained_cols = list(quality.retained_columns)

    # Verify all retained column names map to raw sensor names f000..f589
    sensor_set = set(SENSOR_COLUMNS)
    for col in retained_cols:
        if col not in sensor_set:
            raise ValueError(f"Retained column {col} does not map to a raw SECOM sensor name.")

    mspc_res = score_multivariate_spc(mspc_model, x_scaled, include_contributions=True)
    t2_val = float(mspc_res["t2"][0])
    spe_val = float(mspc_res["spe"][0])
    mspc_val = float(mspc_res["mspc_index"][0])
    t2_c = mspc_res["t2_contributions"][0]
    spe_c = mspc_res["spe_contributions"][0]
    mspc_c = mspc_res["mspc_contributions"][0]

    shap_res = compute_xgb_shap_raw_margin(xgb_model_or_booster, x_ret, background_df=background_df)
    shap_row = np.asarray(shap_res["shap_values"])[0]
    raw_margin = float( np.asarray(shap_res["raw_margin"])[0] )
    max_add_err = float(shap_res["max_additivity_error"])

    k = min(int(top_k), len(retained_cols))

    def _build_mspc_list(contrib_vec: np.ndarray, kind: str) -> list[dict[str, Any]]:
        order = np.argsort(np.abs(contrib_vec))[::-1][:k]
        items: list[dict[str, Any]] = []
        for rank_idx, col_idx in enumerate(order, start=1):
            s_name = retained_cols[int(col_idx)]
            raw_v = df_one.iloc[0].get(s_name, np.nan)
            items.append(
                {
                    "rank": rank_idx,
                    "sensor": s_name,
                    "method": kind,
                    "contribution": float(contrib_vec[int(col_idx)]),
                    "t2_contribution": float(t2_c[int(col_idx)]),
                    "spe_contribution": float(spe_c[int(col_idx)]),
                    "mspc_contribution": float(mspc_c[int(col_idx)]),
                    "sensor_value": None if pd.isna(raw_v) else float(raw_v),
                    "group_members": quality.get_group_members(s_name),
                }
            )
        return items

    mspc_top = _build_mspc_list(mspc_c, "mspc_combined")
    spe_top = _build_mspc_list(spe_c, "spe")
    t2_top = _build_mspc_list(t2_c, "t2")

    shap_order = np.argsort(np.abs(shap_row))[::-1][:k]
    xgb_top: list[dict[str, Any]] = []
    for rank_idx, col_idx in enumerate(shap_order, start=1):
        s_name = retained_cols[int(col_idx)]
        raw_v = df_one.iloc[0].get(s_name, np.nan)
        sv = float(shap_row[int(col_idx)])
        xgb_top.append(
            {
                "rank": rank_idx,
                "sensor": s_name,
                "method": "xgboost_shap_raw_margin",
                "shap_value": sv,
                "abs_shap_value": abs(sv),
                "direction": "increases_fail_risk" if sv > 0 else "decreases_fail_risk",
                "sensor_value": None if pd.isna(raw_v) else float(raw_v),
                "group_members": quality.get_group_members(s_name),
            }
        )

    mspc_sensor_set = {item["sensor"] for item in mspc_top}
    xgb_sensor_set = {item["sensor"] for item in xgb_top}
    direct_overlap = sorted(mspc_sensor_set & xgb_sensor_set)

    return {
        "run_id": int(df_one.iloc[0]["run_id"]) if "run_id" in df_one.columns else 0,
        "top_k": k,
        "mspc_summary": {
            "t2": t2_val,
            "t2_limit_99": mspc_model.t2_limit_99,
            "spe": spe_val,
            "spe_limit_99": mspc_model.spe_limit_99,
            "mspc_index": mspc_val,
        },
        "xgb_summary": {
            "raw_margin": raw_margin,
            "expected_value": float(shap_res["expected_value"]),
            "shap_additivity_error": max_add_err,
        },
        "mspc_top_sensors": mspc_top,
        "spe_top_sensors": spe_top,
        "t2_top_sensors": t2_top,
        "xgb_shap_top_sensors": xgb_top,
        "overlap": {
            "top_k_common_sensors": direct_overlap,
            "overlap_count": len(direct_overlap),
        },
        "caveat": SMEARING_CAVEAT,
    }


def evaluate_o5_synthetic_single_sensor_faults(
    n_train_runs: int = 200,
    n_test_trials: int = 100,
    shift_sigma: float = 5.0,
    seed: int = 42,
    output_path: str | Path | None = "reports/explanations/o5_synthetic_verification.json",
) -> dict[str, Any]:
    """Run quantitative synthetic verification for Target O5 (docs/01_PROBLEM_AND_OBJECTIVES.md).

    Target O5 requirements:
    1. On synthetic single-sensor 5-sigma faults, the injected sensor appears in the top 3
       MSPC contributions in >= 90% of cases.
    2. SHAP additivity holds within 1e-4 (`|sum(shap) + base_value - margin| <= 1e-4`).
    """
    rng = np.random.default_rng(seed)
    cols = list(SENSOR_COLUMNS[:60])  # 60 sensors with latent structure
    n_cols = len(cols)

    # Generate correlated in-control training runs (5 latent factors + sensor noise)
    z_train = rng.normal(0.0, 1.0, size=(n_train_runs, 5))
    loadings_true = rng.normal(0.0, 0.45, size=(5, n_cols))
    x_train = z_train @ loadings_true + rng.normal(0.0, 0.85, size=(n_train_runs, n_cols))
    train_df = pd.DataFrame(x_train, columns=cols)
    y_train = np.zeros(n_train_runs, dtype=np.int64)
    # Add 15 synthetic failing runs for quality/XGBoost fitting
    y_train[-15:] = 1
    train_df.iloc[-15:, 0] += shift_sigma

    qt = fit_quality_transformer(train_df, y_train, sensor_columns=cols)
    x_train_scaled = qt.transform_scaled(train_df)
    mspc_model = fit_multivariate_spc(
        x_train_scaled,
        y_train,
        variance_explained_target=0.80,
        max_components=20,
    )

    retained = list(qt.retained_columns)
    passing_stds = np.array([qt.passing_stds[c] for c in retained], dtype=np.float64)
    passing_means = np.array([qt.passing_means[c] for c in retained], dtype=np.float64)

    # Generate `n_test_trials` in-control test runs and inject a 5-sigma single-sensor fault in each
    z_test = rng.normal(0.0, 1.0, size=(n_test_trials, 5))
    x_test_base = z_test @ loadings_true + rng.normal(0.0, 0.85, size=(n_test_trials, n_cols))
    test_df = pd.DataFrame(x_test_base, columns=cols)[retained].copy()

    injected_sensors: list[str] = []
    for i in range(n_test_trials):
        target_idx = int(i % len(retained))
        target_col = retained[target_idx]
        sign = 1.0 if (i % 2 == 0) else -1.0
        # Inject a 5-sigma shift on the single target sensor relative to its Phase I std
        test_df.iloc[i, target_idx] = passing_means[target_idx] + sign * shift_sigma * passing_stds[target_idx]
        injected_sensors.append(target_col)

    x_test_scaled = qt.transform_scaled(test_df)
    scored = score_multivariate_spc(mspc_model, x_test_scaled, include_contributions=True)
    mspc_contribs = scored["mspc_contributions"]
    spe_contribs = scored["spe_contributions"]

    top1_hits = 0
    top3_mspc_hits = 0
    top3_spe_hits = 0
    ranks_recorded: list[int] = []

    for i, target_col in enumerate(injected_sensors):
        order_mspc = np.argsort(np.abs(mspc_contribs[i]))[::-1]
        ranked_cols_mspc = [retained[int(idx)] for idx in order_mspc]
        rank_1based = ranked_cols_mspc.index(target_col) + 1
        ranks_recorded.append(rank_1based)
        if rank_1based == 1:
            top1_hits += 1
        if rank_1based <= 3:
            top3_mspc_hits += 1

        order_spe = np.argsort(np.abs(spe_contribs[i]))[::-1]
        ranked_cols_spe = [retained[int(idx)] for idx in order_spe]
        if (ranked_cols_spe.index(target_col) + 1) <= 3:
            top3_spe_hits += 1

    top3_hit_rate = float(top3_mspc_hits / n_test_trials)
    top3_spe_hit_rate = float(top3_spe_hits / n_test_trials)
    top1_hit_rate = float(top1_hits / n_test_trials)

    # Also verify SHAP additivity on an XGBoost model fitted on the same synthetic data
    xgb_clf = xgb.XGBClassifier(
        max_depth=2,
        learning_rate=0.05,
        n_estimators=100,
        subsample=0.8,
        colsample_bytree=0.5,
        random_state=seed,
        n_jobs=1,
    )
    x_train_ret = qt.transform_retained(train_df)
    xgb_clf.fit(x_train_ret, y_train)
    shap_check = compute_xgb_shap_raw_margin(xgb_clf, test_df.iloc[:25], background_df=x_train_ret)
    max_shap_err = float(shap_check["max_additivity_error"])

    o5_pass = bool(top3_hit_rate >= 0.90 and max_shap_err <= 1e-4)
    report = {
        "target_id": "O5",
        "verdict": "PASS" if o5_pass else "MISS",
        "n_test_trials": int(n_test_trials),
        "shift_sigma": float(shift_sigma),
        "top1_mspc_hit_rate": top1_hit_rate,
        "top3_mspc_hit_rate": top3_hit_rate,
        "top3_spe_hit_rate": top3_spe_hit_rate,
        "target_min_top3_hit_rate": 0.90,
        "max_shap_additivity_error": max_shap_err,
        "target_max_shap_additivity_error": 1e-4,
        "mean_injected_sensor_rank": float(np.mean(ranks_recorded)),
    }

    if output_path is not None:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    return report


def run_cli_explain(
    config_path: str | Path = "configs/default.yaml",
    bundle_dir: str | Path | None = None,
    input_csv: str | Path | None = None,
    output_path: str | Path = "reports/explanations/sensor_rankings.json",
    verify_synthetic: bool = False,
    top_k: int = 5,
) -> dict[str, Any]:
    """Run synthetic O5 verification and/or explain flagged runs from staging or a frozen bundle."""
    cfg = load_config(config_path)
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    o5_report = evaluate_o5_synthetic_single_sensor_faults(
        seed=cfg.project.seed,
        output_path=out_file.parent / "o5_synthetic_verification.json",
    )
    if verify_synthetic and input_csv is None and bundle_dir is None:
        return o5_report

    model_dir = Path(bundle_dir) if bundle_dir is not None else Path(cfg.models.staging_dir)
    quality = QualityTransformer.load(model_dir / "quality.joblib")
    mspc_model = FittedMspcModel.load(model_dir / "mspc.joblib")
    booster = xgb.Booster()
    booster.load_model(str(model_dir / "xgb_model.json"))

    ref_path = model_dir / "monitoring_reference.parquet"
    bg_df = pd.read_parquet(ref_path) if ref_path.exists() else None

    if input_csv is not None:
        runs_df = pd.read_csv(input_csv)
    else:
        runs_df = pd.read_parquet(cfg.data.runs_parquet)
        runs_df = runs_df[runs_df["block"] == "D5"].reset_index(drop=True)

    # Identify flagged runs by primary detector or MSPC/XGBoost budget_5 threshold if policy.json exists
    policy_path = model_dir / "policy.json"
    flagged_indices: list[int] = []
    if policy_path.exists():
        policy = json.loads(policy_path.read_text(encoding="utf-8"))
        t2_tau = float(policy["thresholds"]["E04"]["budget_5"])
        spe_tau = float(policy["thresholds"]["E05"]["budget_5"])
        mspc_tau = float(policy["thresholds"]["E06"]["budget_5"])
        x_scaled = quality.transform_scaled(runs_df)
        m_scores = score_multivariate_spc(mspc_model, x_scaled, include_contributions=False)
        mask = (
            (m_scores["t2"] > t2_tau)
            | (m_scores["spe"] > spe_tau)
            | (m_scores["mspc_index"] > mspc_tau)
        )
        flagged_indices = [int(i) for i in np.flatnonzero(mask)]

    if not flagged_indices:
        flagged_indices = list(range(min(5, len(runs_df))))

    explanations: list[dict[str, Any]] = []
    for idx in flagged_indices[:25]:
        row = runs_df.iloc[idx]
        explanations.append(
            rank_sensors_for_run(
                row,
                quality=quality,
                mspc_model=mspc_model,
                xgb_model_or_booster=booster,
                top_k=top_k,
                background_df=bg_df,
            )
        )

    payload = {
        "model_dir": str(model_dir.as_posix()),
        "n_flagged_runs_explained": len(explanations),
        "o5_synthetic_verification": o5_report,
        "caveat": SMEARING_CAVEAT,
        "flagged_run_explanations": explanations,
    }
    out_file.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload
