"""Evaluation metrics, recall at fixed false-alarm rate, bootstrap CIs, and primary detector selection."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from yieldguard.config import load_config
from yieldguard.policy import (
    compute_alarm_budget_threshold,
    evaluate_at_threshold,
)


def compute_pr_auc(
    y_true: np.ndarray | list[int],
    scores: np.ndarray | list[float],
) -> tuple[float | None, str | None]:
    """Compute PR-AUC (average precision) with explicit undefined-ratio handling."""
    y = np.asarray(y_true, dtype=np.int64)
    s = np.asarray(scores, dtype=np.float64)
    if len(y) != len(s):
        raise ValueError(f"Length mismatch: y_true ({len(y)}) vs scores ({len(s)}).")
    if len(y) == 0:
        return None, "zero_runs"
    n_pos = int(np.sum(y == 1))
    if n_pos == 0:
        return None, "zero_failing_runs"
    return float(average_precision_score(y, s)), None


def compute_roc_auc(
    y_true: np.ndarray | list[int],
    scores: np.ndarray | list[float],
) -> tuple[float | None, str | None]:
    """Compute secondary ROC-AUC metric with explicit undefined handling."""
    y = np.asarray(y_true, dtype=np.int64)
    s = np.asarray(scores, dtype=np.float64)
    if len(y) != len(s):
        raise ValueError(f"Length mismatch: y_true ({len(y)}) vs scores ({len(s)}).")
    if len(y) == 0:
        return None, "zero_runs"
    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))
    if n_pos == 0:
        return None, "zero_failing_runs"
    if n_neg == 0:
        return None, "zero_passing_runs"
    return float(roc_auc_score(y, s)), None


def compute_recall_at_fixed_far(
    y_true: np.ndarray | list[int],
    scores: np.ndarray | list[float],
    far_per_100: float = 5.0,
    is_chance_baseline: bool = False,
) -> dict[str, Any]:
    """Compute recall at `far_per_100` false alarms per 100 passing runs.

    Threshold is set at the `1 - far_per_100/100` linear quantile of passing-run scores,
    and alarms fire for `score > threshold`.
    If `is_chance_baseline=True` (E01 prevalence baseline), returns the theoretical chance
    recall `far_per_100 / 100.0` alongside empirical constant-score counts.
    """
    y = np.asarray(y_true, dtype=np.int64)
    s = np.asarray(scores, dtype=np.float64)
    if len(y) != len(s):
        raise ValueError(f"Length mismatch: y_true ({len(y)}) vs scores ({len(s)}).")

    n_runs = int(len(y))
    n_fails = int(np.sum(y == 1))
    n_passing = int(np.sum(y == 0))

    if n_runs == 0:
        return {
            "far_target_per_100": float(far_per_100),
            "threshold": None,
            "n_runs": 0,
            "n_fails": 0,
            "n_passing": 0,
            "true_alarms": 0,
            "false_alarms": 0,
            "realised_far_per_100": None,
            "recall": None,
            "undefined_reason": "zero_runs",
        }
    if n_passing == 0:
        return {
            "far_target_per_100": float(far_per_100),
            "threshold": None,
            "n_runs": n_runs,
            "n_fails": n_fails,
            "n_passing": 0,
            "true_alarms": 0,
            "false_alarms": 0,
            "realised_far_per_100": None,
            "recall": None,
            "undefined_reason": "zero_passing_runs",
        }
    if n_fails == 0:
        return {
            "far_target_per_100": float(far_per_100),
            "threshold": None,
            "n_runs": n_runs,
            "n_fails": 0,
            "n_passing": n_passing,
            "true_alarms": 0,
            "false_alarms": 0,
            "realised_far_per_100": 0.0,
            "recall": None,
            "undefined_reason": "zero_failing_runs",
        }

    s_pass = s[y == 0]
    s_fail = s[y == 1]
    q = 1.0 - (far_per_100 / 100.0)
    threshold = float(np.quantile(s_pass, q, method="linear"))

    if is_chance_baseline:
        return {
            "far_target_per_100": float(far_per_100),
            "threshold": threshold,
            "n_runs": n_runs,
            "n_fails": n_fails,
            "n_passing": n_passing,
            "true_alarms": 0,
            "false_alarms": 0,
            "realised_far_per_100": float(far_per_100),
            "recall": float(far_per_100 / 100.0),
            "undefined_reason": None,
            "note": "theoretical_chance_level",
        }

    tp = int(np.sum(s_fail > threshold))
    fp = int(np.sum(s_pass > threshold))
    return {
        "far_target_per_100": float(far_per_100),
        "threshold": threshold,
        "n_runs": n_runs,
        "n_fails": n_fails,
        "n_passing": n_passing,
        "true_alarms": tp,
        "false_alarms": fp,
        "realised_far_per_100": float(100.0 * fp / n_passing),
        "recall": float(tp / n_fails),
        "undefined_reason": None,
    }


def compute_detector_metrics_summary(
    y_true: np.ndarray | list[int],
    scores: np.ndarray | list[float],
    far_targets: list[float] | tuple[float, ...] = (2.0, 5.0, 10.0),
    is_chance_baseline: bool = False,
) -> dict[str, Any]:
    """Compute complete single-split metrics (PR-AUC, ROC-AUC, recall at FAR 2/5/10, counts)."""
    y = np.asarray(y_true, dtype=np.int64)
    s = np.asarray(scores, dtype=np.float64)
    n_runs = int(len(y))
    n_fails = int(np.sum(y == 1))
    n_passing = int(np.sum(y == 0))
    prevalence = float(n_fails / n_runs) if n_runs > 0 else None

    pr_auc, pr_reason = compute_pr_auc(y, s)
    roc_auc, roc_reason = compute_roc_auc(y, s)

    recall_at_far: dict[str, Any] = {}
    for f_rate in far_targets:
        key = f"far_{int(f_rate) if float(f_rate).is_integer() else f_rate}"
        recall_at_far[key] = compute_recall_at_fixed_far(
            y, s, far_per_100=float(f_rate), is_chance_baseline=is_chance_baseline
        )

    undefined_reasons: dict[str, str] = {}
    if pr_reason:
        undefined_reasons["pr_auc"] = pr_reason
    if roc_reason:
        undefined_reasons["roc_auc"] = roc_reason

    return {
        "n_runs": n_runs,
        "n_fails": n_fails,
        "n_passing": n_passing,
        "prevalence": prevalence,
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "recall_at_far": recall_at_far,
        "undefined_reasons": undefined_reasons,
    }


def stratified_bootstrap_ci(
    y_true: np.ndarray | list[int],
    scores: np.ndarray | list[float],
    n_bootstraps: int = 2000,
    seed: int = 42,
    far_target_per_100: float = 5.0,
    is_chance_baseline: bool = False,
) -> dict[str, Any]:
    """Compute 2,000 seeded class-stratified bootstrap 95% CIs for PR-AUC and recall@FAR."""
    y = np.asarray(y_true, dtype=np.int64)
    s = np.asarray(scores, dtype=np.float64)
    pos_idx = np.flatnonzero(y == 1)
    neg_idx = np.flatnonzero(y == 0)
    n_pos = len(pos_idx)
    n_neg = len(neg_idx)
    if n_pos == 0 or n_neg == 0:
        return {
            "pr_auc_95_ci": [None, None],
            "recall_at_far_95_ci": [None, None],
            "undefined_reason": "zero_passing_or_failing_runs",
        }

    rng = np.random.default_rng(seed)
    pr_draws = np.empty(n_bootstraps, dtype=np.float64)
    rec_draws = np.empty(n_bootstraps, dtype=np.float64)
    y_boot = np.concatenate([np.ones(n_pos, dtype=np.int64), np.zeros(n_neg, dtype=np.int64)])
    q = 1.0 - (far_target_per_100 / 100.0)

    for b in range(n_bootstraps):
        p_draw = pos_idx[rng.integers(0, n_pos, size=n_pos)]
        n_draw = neg_idx[rng.integers(0, n_neg, size=n_neg)]
        s_pos_b = s[p_draw]
        s_neg_b = s[n_draw]
        s_boot = np.concatenate([s_pos_b, s_neg_b])
        pr_draws[b] = float(average_precision_score(y_boot, s_boot))
        if is_chance_baseline:
            rec_draws[b] = float(far_target_per_100 / 100.0)
        else:
            tau = float(np.quantile(s_neg_b, q, method="linear"))
            rec_draws[b] = float(np.mean(s_pos_b > tau))

    return {
        "n_bootstraps": int(n_bootstraps),
        "seed": int(seed),
        "pr_auc_95_ci": [
            float(np.percentile(pr_draws, 2.5)),
            float(np.percentile(pr_draws, 97.5)),
        ],
        "recall_at_far_95_ci": [
            float(np.percentile(rec_draws, 2.5)),
            float(np.percentile(rec_draws, 97.5)),
        ],
    }


def moving_block_bootstrap_ci(
    y_true: np.ndarray | list[int],
    scores: np.ndarray | list[float],
    block_length: int = 20,
    n_bootstraps: int = 2000,
    seed: int = 42,
    far_target_per_100: float = 5.0,
    is_chance_baseline: bool = False,
) -> dict[str, Any]:
    """Compute moving-block bootstrap 95% CIs with contiguous blocks of `block_length` runs."""
    y = np.asarray(y_true, dtype=np.int64)
    s = np.asarray(scores, dtype=np.float64)
    n = len(y)
    if n == 0 or np.sum(y == 1) == 0 or np.sum(y == 0) == 0:
        return {
            "pr_auc_95_ci": [None, None],
            "recall_at_far_95_ci": [None, None],
            "undefined_reason": "zero_passing_or_failing_runs",
        }

    b_len = min(block_length, n)
    n_blocks_needed = int(np.ceil(n / b_len))
    max_start = max(1, n - b_len + 1)
    rng = np.random.default_rng(seed)

    pr_draws: list[float] = []
    rec_draws: list[float] = []
    q = 1.0 - (far_target_per_100 / 100.0)

    attempts = 0
    max_attempts = n_bootstraps * 10
    while len(pr_draws) < n_bootstraps and attempts < max_attempts:
        attempts += 1
        starts = rng.integers(0, max_start, size=n_blocks_needed)
        idx = np.concatenate([np.arange(st, st + b_len) for st in starts])[:n]
        y_b = y[idx]
        if np.sum(y_b == 1) == 0 or np.sum(y_b == 0) == 0:
            continue
        s_b = s[idx]
        pr_draws.append(float(average_precision_score(y_b, s_b)))
        if is_chance_baseline:
            rec_draws.append(float(far_target_per_100 / 100.0))
        else:
            tau = float(np.quantile(s_b[y_b == 0], q, method="linear"))
            rec_draws.append(float(np.mean(s_b[y_b == 1] > tau)))

    return {
        "n_bootstraps": len(pr_draws),
        "block_length": int(b_len),
        "seed": int(seed),
        "pr_auc_95_ci": [
            float(np.percentile(pr_draws, 2.5)),
            float(np.percentile(pr_draws, 97.5)),
        ],
        "recall_at_far_95_ci": [
            float(np.percentile(rec_draws, 2.5)),
            float(np.percentile(rec_draws, 97.5)),
        ],
    }


def paired_stratified_bootstrap_diff(
    y_true: np.ndarray | list[int],
    scores_a: np.ndarray | list[float],
    scores_b: np.ndarray | list[float],
    n_bootstraps: int = 2000,
    seed: int = 42,
) -> dict[str, Any]:
    """Compute paired class-stratified bootstrap PR-AUC difference (A - B) using identical draws."""
    y = np.asarray(y_true, dtype=np.int64)
    sa = np.asarray(scores_a, dtype=np.float64)
    sb = np.asarray(scores_b, dtype=np.float64)
    pos_idx = np.flatnonzero(y == 1)
    neg_idx = np.flatnonzero(y == 0)
    n_pos = len(pos_idx)
    n_neg = len(neg_idx)
    if n_pos == 0 or n_neg == 0:
        return {"diff_point": None, "diff_95_ci": [None, None], "contains_zero": True}

    point_a = float(average_precision_score(y, sa))
    point_b = float(average_precision_score(y, sb))
    point_diff = float(point_a - point_b)

    rng = np.random.default_rng(seed)
    diffs = np.empty(n_bootstraps, dtype=np.float64)
    y_boot = np.concatenate([np.ones(n_pos, dtype=np.int64), np.zeros(n_neg, dtype=np.int64)])

    for b in range(n_bootstraps):
        p_draw = pos_idx[rng.integers(0, n_pos, size=n_pos)]
        n_draw = neg_idx[rng.integers(0, n_neg, size=n_neg)]
        idx = np.concatenate([p_draw, n_draw])
        ap_a = float(average_precision_score(y_boot, sa[idx]))
        ap_b = float(average_precision_score(y_boot, sb[idx]))
        diffs[b] = ap_a - ap_b

    low = float(np.percentile(diffs, 2.5))
    high = float(np.percentile(diffs, 97.5))
    return {
        "diff_point": point_diff,
        "diff_95_ci": [low, high],
        "contains_zero": bool(low <= 0.0 <= high),
    }


def paired_moving_block_bootstrap_diff(
    y_true: np.ndarray | list[int],
    scores_a: np.ndarray | list[float],
    scores_b: np.ndarray | list[float],
    block_length: int = 20,
    n_bootstraps: int = 2000,
    seed: int = 42,
) -> dict[str, Any]:
    """Compute paired moving-block bootstrap PR-AUC difference (A - B) using identical block draws."""
    y = np.asarray(y_true, dtype=np.int64)
    sa = np.asarray(scores_a, dtype=np.float64)
    sb = np.asarray(scores_b, dtype=np.float64)
    n = len(y)
    if n == 0 or np.sum(y == 1) == 0 or np.sum(y == 0) == 0:
        return {"diff_point": None, "diff_95_ci": [None, None], "contains_zero": True}

    point_diff = float(average_precision_score(y, sa) - average_precision_score(y, sb))
    b_len = min(block_length, n)
    n_blocks_needed = int(np.ceil(n / b_len))
    max_start = max(1, n - b_len + 1)
    rng = np.random.default_rng(seed)

    diffs: list[float] = []
    attempts = 0
    max_attempts = n_bootstraps * 10
    while len(diffs) < n_bootstraps and attempts < max_attempts:
        attempts += 1
        starts = rng.integers(0, max_start, size=n_blocks_needed)
        idx = np.concatenate([np.arange(st, st + b_len) for st in starts])[:n]
        y_b = y[idx]
        if np.sum(y_b == 1) == 0 or np.sum(y_b == 0) == 0:
            continue
        diffs.append(float(average_precision_score(y_b, sa[idx]) - average_precision_score(y_b, sb[idx])))

    low = float(np.percentile(diffs, 2.5))
    high = float(np.percentile(diffs, 97.5))
    return {
        "diff_point": point_diff,
        "diff_95_ci": [low, high],
        "contains_zero": bool(low <= 0.0 <= high),
    }


def wider_confidence_interval(
    ci_a: list[float | None],
    ci_b: list[float | None],
) -> list[float | None]:
    """Return the outer (wider) interval combining stratified and moving-block bootstrap CIs."""
    if ci_a[0] is None or ci_a[1] is None:
        return ci_b
    if ci_b[0] is None or ci_b[1] is None:
        return ci_a
    return [float(min(ci_a[0], ci_b[0])), float(max(ci_a[1], ci_b[1]))]


def fold_mean_pr_auc_bootstrap(
    fold_y: dict[str, np.ndarray],
    fold_scores_by_detector: dict[str, dict[str, np.ndarray]],
    n_bootstraps: int = 2000,
    seed: int = 42,
) -> dict[str, Any]:
    """Compute within-fold mean PR-AUC and paired bootstrap CIs across forward folds D3-D5.

    Per docs/05_EXPERIMENT_PLAN.md section 4.2 & 7:
    - PR-AUC is computed within each fold and averaged across folds so score scales never mix.
    - All detectors share identical class-stratified bootstrap draws within each fold.
    """
    fold_ids = sorted(fold_y.keys())
    rng = np.random.default_rng(seed)

    # Pre-generate stratified index draws per fold: shape (n_bootstraps, n_fold)
    fold_boot_indices: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for f_id in fold_ids:
        yf = np.asarray(fold_y[f_id], dtype=np.int64)
        pos_idx = np.flatnonzero(yf == 1)
        neg_idx = np.flatnonzero(yf == 0)
        n_pos = len(pos_idx)
        n_neg = len(neg_idx)
        p_draws = pos_idx[rng.integers(0, n_pos, size=(n_bootstraps, n_pos))]
        n_draws = neg_idx[rng.integers(0, n_neg, size=(n_bootstraps, n_neg))]
        idx_draws = np.concatenate([p_draws, n_draws], axis=1)
        y_boot = np.concatenate([np.ones(n_pos, dtype=np.int64), np.zeros(n_neg, dtype=np.int64)])
        fold_boot_indices[f_id] = (idx_draws, y_boot)

    det_summaries: dict[str, Any] = {}
    det_boot_means: dict[str, np.ndarray] = {}

    for det_id, f_scores in fold_scores_by_detector.items():
        per_fold_ap: dict[str, float] = {}
        per_fold_fails: dict[str, int] = {}
        boot_mat = np.empty((len(fold_ids), n_bootstraps), dtype=np.float64)

        for f_idx, f_id in enumerate(fold_ids):
            yf = np.asarray(fold_y[f_id], dtype=np.int64)
            sf = np.asarray(f_scores[f_id], dtype=np.float64)
            per_fold_ap[f_id] = float(average_precision_score(yf, sf))
            per_fold_fails[f_id] = int(np.sum(yf == 1))
            idx_draws, y_boot = fold_boot_indices[f_id]
            for b in range(n_bootstraps):
                boot_mat[f_idx, b] = float(average_precision_score(y_boot, sf[idx_draws[b]]))

        boot_mean = np.mean(boot_mat, axis=0)
        det_boot_means[det_id] = boot_mean
        fold_vals = list(per_fold_ap.values())
        mean_ap = float(np.mean(fold_vals))
        se_ap = float(np.std(fold_vals, ddof=1) / np.sqrt(len(fold_vals))) if len(fold_vals) > 1 else 0.0

        det_summaries[det_id] = {
            "per_fold_pr_auc": per_fold_ap,
            "per_fold_fails": per_fold_fails,
            "total_eval_fails": int(sum(per_fold_fails.values())),
            "mean_fold_pr_auc": mean_ap,
            "fold_standard_error": se_ap,
            "bootstrap_95_ci": [
                float(np.percentile(boot_mean, 2.5)),
                float(np.percentile(boot_mean, 97.5)),
            ],
        }

    return {
        "detector_summaries": det_summaries,
        "boot_means": det_boot_means,
    }


def select_primary_detector(
    det_summaries: dict[str, Any],
    det_boot_means: dict[str, np.ndarray],
    spc_ids: list[str] | tuple[str, ...] = ("E02", "E03", "E04", "E05", "E06"),
    ml_ids: list[str] | tuple[str, ...] = ("E07", "E08"),
) -> dict[str, Any]:
    """Select the primary detector according to docs/05_EXPERIMENT_PLAN.md section 4.4:

    Highest mean fold PR-AUC, except that if the best ML detector's paired bootstrap
    advantage over the best SPC detector has a 95% interval containing zero,
    the SPC detector is primary.
    """
    avail_spc = [d for d in spc_ids if d in det_summaries]
    avail_ml = [d for d in ml_ids if d in det_summaries]
    if not avail_spc or not avail_ml:
        raise ValueError("Both SPC (E02-E06) and ML (E07-E08) detectors must be present for selection.")

    best_spc = max(avail_spc, key=lambda d: det_summaries[d]["mean_fold_pr_auc"])
    best_ml = max(avail_ml, key=lambda d: det_summaries[d]["mean_fold_pr_auc"])

    spc_mean = float(det_summaries[best_spc]["mean_fold_pr_auc"])
    ml_mean = float(det_summaries[best_ml]["mean_fold_pr_auc"])
    diff_point = float(ml_mean - spc_mean)

    diff_draws = det_boot_means[best_ml] - det_boot_means[best_spc]
    ci_low = float(np.percentile(diff_draws, 2.5))
    ci_high = float(np.percentile(diff_draws, 97.5))
    ci_contains_zero = bool(ci_low <= 0.0 <= ci_high)

    # Rule 4: ML is primary only if ml_mean > spc_mean AND 95% paired bootstrap CI does not contain 0
    if ml_mean > spc_mean and not ci_contains_zero:
        primary_id = best_ml
        reason = (
            f"Best ML detector ({best_ml}, mean fold PR-AUC={ml_mean:.6f}) beats best SPC detector "
            f"({best_spc}, mean fold PR-AUC={spc_mean:.6f}) and the paired bootstrap 95% CI "
            f"[{ci_low:.6f}, {ci_high:.6f}] excludes zero."
        )
    else:
        primary_id = best_spc
        reason = (
            f"Selected best SPC detector ({best_spc}, mean fold PR-AUC={spc_mean:.6f}) over best ML detector "
            f"({best_ml}, mean fold PR-AUC={ml_mean:.6f}): paired bootstrap difference (ML - SPC) = "
            f"{diff_point:+.6f} with 95% CI [{ci_low:.6f}, {ci_high:.6f}] "
            f"({'contains zero' if ci_contains_zero else 'ML not superior'})."
        )

    return {
        "primary_detector": primary_id,
        "best_spc_detector": best_spc,
        "best_spc_mean_fold_pr_auc": spc_mean,
        "best_ml_detector": best_ml,
        "best_ml_mean_fold_pr_auc": ml_mean,
        "paired_diff_ml_minus_spc": diff_point,
        "paired_diff_95_ci": [ci_low, ci_high],
        "ci_contains_zero": ci_contains_zero,
        "selection_rule_reason": reason,
    }


def run_development_evaluation(config_path: str | Path = "configs/default.yaml") -> dict[str, Any]:
    """Aggregate E01-E08 out-of-fold scores on D3-D5, choose primary detector, freeze D5 thresholds."""
    cfg = load_config(config_path)
    staging_dir = Path(cfg.models.staging_dir)
    oof_path = staging_dir / "dev_oof_scores.parquet"
    if not oof_path.exists():
        raise FileNotFoundError(
            f"Missing out-of-fold development scores at {oof_path}. Run `yieldguard train` first."
        )

    oof_df = pd.read_parquet(oof_path)
    if "block" in oof_df.columns and (oof_df["block"] == "holdout").any():
        raise PermissionError("Holdout rows detected in development OOF scores!")

    detectors = ["E01", "E02", "E03", "E04", "E05", "E06", "E07", "E08"]
    block_to_fold = {"D3": "fold_1", "D4": "fold_2", "D5": "fold_3"}

    fold_y: dict[str, np.ndarray] = {}
    fold_scores_by_det: dict[str, dict[str, np.ndarray]] = {d: {} for d in detectors}

    for b_name, f_id in block_to_fold.items():
        sub = oof_df[oof_df["block"] == b_name]
        fold_y[f_id] = sub["label_fail"].to_numpy(dtype=np.int64)
        for d in detectors:
            fold_scores_by_det[d][f_id] = sub[d].to_numpy(dtype=np.float64)

    boot_res = fold_mean_pr_auc_bootstrap(
        fold_y,
        fold_scores_by_det,
        n_bootstraps=cfg.policy.n_bootstraps,
        seed=cfg.project.seed,
    )
    det_summaries = boot_res["detector_summaries"]
    det_boot_means = boot_res["boot_means"]

    # Pre-generate within-fold moving-block bootstrap draws
    b_len = cfg.policy.block_bootstrap_length
    fold_ids = list(fold_y.keys())
    rng = np.random.default_rng(cfg.project.seed)
    fold_mb_draws: dict[str, list[np.ndarray]] = {}
    for f_id in fold_ids:
        n_f = len(fold_y[f_id])
        cur_b_len = min(b_len, n_f)
        n_blocks_needed = int(np.ceil(n_f / cur_b_len))
        max_st = max(1, n_f - cur_b_len + 1)
        f_draws: list[np.ndarray] = []
        attempts = 0
        while len(f_draws) < cfg.policy.n_bootstraps and attempts < cfg.policy.n_bootstraps * 20:
            attempts += 1
            starts = rng.integers(0, max_st, size=n_blocks_needed)
            idx_mb = np.concatenate([np.arange(st, st + cur_b_len) for st in starts])[:n_f]
            y_sample = fold_y[f_id][idx_mb]
            if np.sum(y_sample == 1) > 0 and np.sum(y_sample == 0) > 0:
                f_draws.append(idx_mb)
        fold_mb_draws[f_id] = f_draws

    det_mb_means: dict[str, np.ndarray] = {}
    for d in detectors:
        mb_mat = np.empty((len(fold_ids), cfg.policy.n_bootstraps), dtype=np.float64)
        for f_idx, f_id in enumerate(fold_ids):
            yf = fold_y[f_id]
            sf = fold_scores_by_det[d][f_id]
            draws_f = fold_mb_draws[f_id]
            for b in range(cfg.policy.n_bootstraps):
                idx_b = draws_f[b]
                mb_mat[f_idx, b] = float(average_precision_score(yf[idx_b], sf[idx_b]))
        det_mb_means[d] = np.mean(mb_mat, axis=0)

    # Compute per-fold recall at FAR 2, 5, 10 and average across folds, plus within-fold moving-block CI
    y_all = oof_df["label_fail"].to_numpy(dtype=np.int64)
    n_eval_runs = int(len(y_all))
    n_eval_fails = int(np.sum(y_all == 1))
    eval_prevalence = float(n_eval_fails / n_eval_runs)

    for d in detectors:
        is_chance = d == "E01"
        far_means: dict[str, float] = {}
        roc_vals: list[float] = []
        for f_id in ["fold_1", "fold_2", "fold_3"]:
            yf = fold_y[f_id]
            sf = fold_scores_by_det[d][f_id]
            roc_val, _ = compute_roc_auc(yf, sf)
            if roc_val is not None:
                roc_vals.append(roc_val)
        for far in cfg.policy.false_alarm_rates_per_100:
            key = f"far_{int(far) if float(far).is_integer() else far}"
            r_folds = [
                compute_recall_at_fixed_far(
                    fold_y[f_id],
                    fold_scores_by_det[d][f_id],
                    far_per_100=float(far),
                    is_chance_baseline=is_chance,
                )["recall"]
                for f_id in ["fold_1", "fold_2", "fold_3"]
            ]
            far_means[key] = float(np.mean([r for r in r_folds if r is not None]))

        mb_draws_d = det_mb_means[d]
        mb_ci_range = [
            float(np.percentile(mb_draws_d, 2.5)),
            float(np.percentile(mb_draws_d, 97.5)),
        ]
        det_summaries[d]["mean_fold_roc_auc"] = float(np.mean(roc_vals)) if roc_vals else None
        det_summaries[d]["mean_fold_recall_at_far"] = far_means
        det_summaries[d]["moving_block_pr_auc_95_ci"] = mb_ci_range
        det_summaries[d]["wider_pr_auc_95_ci"] = wider_confidence_interval(
            det_summaries[d]["bootstrap_95_ci"], mb_ci_range
        )

    selection = select_primary_detector(det_summaries, det_boot_means)
    # Add paired moving-block difference between best ML and best SPC
    best_ml_id = selection["best_ml_detector"]
    best_spc_id = selection["best_spc_detector"]
    diff_mb = det_mb_means[best_ml_id] - det_mb_means[best_spc_id]
    mb_diff_ci = [
        float(np.percentile(diff_mb, 2.5)),
        float(np.percentile(diff_mb, 97.5)),
    ]
    selection["paired_diff_moving_block_95_ci"] = mb_diff_ci
    selection["paired_diff_wider_95_ci"] = wider_confidence_interval(
        selection["paired_diff_95_ci"], mb_diff_ci
    )
    primary_id = selection["primary_detector"]

    # Freeze alarm thresholds on calibration block D5 (fold_3 evaluation block) for budgets 2, 5, 10
    d5_df = oof_df[oof_df["block"] == "D5"]
    d5_y = d5_df["label_fail"].to_numpy(dtype=np.int64)
    thresholds_by_detector: dict[str, dict[str, float]] = {}
    d5_calibration_diagnostics: dict[str, dict[str, Any]] = {}

    for d in detectors:
        d5_scores = d5_df[d].to_numpy(dtype=np.float64)
        thresholds_by_detector[d] = {}
        d5_calibration_diagnostics[d] = {}
        for b in cfg.policy.alarm_budgets_per_100:
            b_key = f"budget_{int(b) if float(b).is_integer() else b}"
            tau = compute_alarm_budget_threshold(d5_scores, float(b))
            thresholds_by_detector[d][b_key] = tau
            op_metrics = evaluate_at_threshold(d5_y, d5_scores, tau, budget_per_100=float(b))
            d5_calibration_diagnostics[d][b_key] = op_metrics.to_dict()

    # Load textbook SPC limit diagnostics on D5 if saved
    textbook_spc_diag = {}
    spc_diag_path = staging_dir / "spc_textbook_diagnostics.json"
    if spc_diag_path.exists():
        textbook_spc_diag = json.loads(spc_diag_path.read_text(encoding="utf-8"))

    policy_artifact = {
        "policy_version": "1.0",
        "primary_detector": primary_id,
        "primary_alarm_budget_per_100": cfg.policy.primary_alarm_budget_per_100,
        "alarm_budgets_per_100": cfg.policy.alarm_budgets_per_100,
        "score_direction": "higher_is_more_anomalous",
        "alarm_rule": "score > threshold",
        "calibration_block": "D5",
        "n_calibration_runs": int(len(d5_df)),
        "thresholds": thresholds_by_detector,
        "selection": selection,
    }
    (staging_dir / "policy.json").write_text(
        json.dumps(policy_artifact, indent=2) + "\n", encoding="utf-8"
    )

    val_report = {
        "dataset_id": cfg.data.dataset_id,
        "protocol_version": "1.0",
        "seed": cfg.project.seed,
        "eval_blocks": ["D3", "D4", "D5"],
        "n_eval_runs": n_eval_runs,
        "n_eval_fails": n_eval_fails,
        "eval_prevalence": eval_prevalence,
        "primary_detector": primary_id,
        "selection": selection,
        "detectors": det_summaries,
        "calibration_block_d5": {
            "n_runs": int(len(d5_df)),
            "n_fails": int(np.sum(d5_y == 1)),
            "thresholds": thresholds_by_detector,
            "operating_diagnostics": d5_calibration_diagnostics,
        },
        "textbook_spc_limits_d5": textbook_spc_diag,
        "holdout_labels_opened": False,
    }

    report_path = Path(cfg.models.reports_dir) / "validation_metrics.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(val_report, indent=2) + "\n", encoding="utf-8")
    return val_report


def run_holdout_evaluation(
    config_path: str | Path = "configs/default.yaml",
    bundle_dir: str | Path = "models/v0.1.0",
    release_id: str = "v0.1.0",
    allow_heldout_evaluation: bool = False,
) -> dict[str, Any]:
    """One-shot holdout evaluation (T12).

    1. Requires `allow_heldout_evaluation=True` and a verified frozen bundle.
    2. Strictly refuses to overwrite an existing `reports/holdout_metrics.json` per §10.4.
    3. Scores all holdout runs without labels, writes `reports/holdout_predictions.csv`, and
       computes its SHA-256 hash BEFORE `labels_holdout.parquet` is opened.
    4. Opens `labels_holdout.parquet` once, computes metrics, stratified & moving-block CIs,
       and predeclared target verdicts O1-O4.
    """
    if not allow_heldout_evaluation:
        raise PermissionError(
            "Holdout evaluation is sealed. Pass --allow-heldout-evaluation after C3 approval."
        )

    cfg = load_config(config_path)
    reports_dir = Path(cfg.models.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = reports_dir / "holdout_metrics.json"
    preds_path = reports_dir / "holdout_predictions.csv"

    if metrics_path.exists():
        raise FileExistsError(
            f"Holdout evaluation record already exists at {metrics_path}. "
            "Per protocol §10.4, the command strictly refuses to overwrite an existing evaluation record."
        )

    from yieldguard.artifacts import load_model_bundle
    from yieldguard.predict import score_runs_dataframe

    bundle = load_model_bundle(bundle_dir)
    runs_df = pd.read_parquet(cfg.data.runs_parquet)
    holdout_runs = runs_df[runs_df["block"] == "holdout"].reset_index(drop=True)
    if "label_fail" in holdout_runs.columns or "label_raw" in holdout_runs.columns:
        raise ValueError("Unlabeled holdout runs table must not contain labels.")

    # Step 1: Score holdout runs and hash predictions BEFORE opening labels_holdout.parquet
    scored_df = score_runs_dataframe(bundle, holdout_runs)
    scored_df.to_csv(preds_path, index=False)
    preds_sha256 = hashlib.sha256(preds_path.read_bytes()).hexdigest()

    # Step 2: Open labels_holdout.parquet for the first and only time
    labels_holdout = pd.read_parquet(cfg.data.labels_holdout_parquet)
    merged = pd.merge(
        scored_df,
        labels_holdout[["run_id", "label_fail"]],
        on="run_id",
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(holdout_runs):
        raise ValueError("Holdout run_id alignment mismatch.")

    y_true = merged["label_fail"].to_numpy(dtype=np.int64)
    n_runs = int(len(y_true))
    n_fails = int(np.sum(y_true == 1))
    n_passing = int(np.sum(y_true == 0))
    prevalence = float(n_fails / n_runs)

    det_col_map = {
        "E01": "e01_score",
        "E02": "e02_score",
        "E03": "e03_score",
        "E04": "t2",
        "E05": "spe",
        "E06": "mspc_index",
        "E07": "iforest_score",
        "E08": "xgb_score",
    }

    detectors_holdout: dict[str, Any] = {}
    for det_id, col_name in det_col_map.items():
        s = merged[col_name].to_numpy(dtype=np.float64)
        is_chance = det_id == "E01"
        base_metrics = compute_detector_metrics_summary(
            y_true,
            s,
            far_targets=cfg.policy.false_alarm_rates_per_100,
            is_chance_baseline=is_chance,
        )
        strat_ci = stratified_bootstrap_ci(
            y_true,
            s,
            n_bootstraps=cfg.policy.n_bootstraps,
            seed=cfg.project.seed,
            far_target_per_100=5.0,
            is_chance_baseline=is_chance,
        )
        block_ci = moving_block_bootstrap_ci(
            y_true,
            s,
            block_length=cfg.policy.block_bootstrap_length,
            n_bootstraps=cfg.policy.n_bootstraps,
            seed=cfg.project.seed,
            far_target_per_100=5.0,
            is_chance_baseline=is_chance,
        )
        wider_pr_ci = wider_confidence_interval(
            strat_ci["pr_auc_95_ci"], block_ci["pr_auc_95_ci"]
        )
        wider_rec_ci = wider_confidence_interval(
            strat_ci["recall_at_far_95_ci"], block_ci["recall_at_far_95_ci"]
        )

        budget_results: dict[str, Any] = {}
        det_thresholds = bundle.policy["thresholds"].get(det_id, {})
        for b in cfg.policy.alarm_budgets_per_100:
            b_key = f"budget_{int(b) if float(b).is_integer() else b}"
            if b_key in det_thresholds:
                tau = float(det_thresholds[b_key])
                budget_results[b_key] = evaluate_at_threshold(
                    y_true, s, tau, budget_per_100=float(b)
                ).to_dict()

        detectors_holdout[det_id] = {
            **base_metrics,
            "stratified_bootstrap": strat_ci,
            "moving_block_bootstrap": block_ci,
            "wider_pr_auc_95_ci": wider_pr_ci,
            "wider_recall_at_5_far_95_ci": wider_rec_ci,
            "alarm_budgets": budget_results,
        }

    # Textbook SPC limits on holdout
    t2_lim_99 = float(bundle.mspc_model["t2_limit_99"])
    spe_lim_99 = float(bundle.mspc_model["spe_limit_99"])
    ewma_l = float(bundle.spc_model["ewma_l"])
    textbook_holdout = {
        "E02_individuals_3sigma": evaluate_at_threshold(
            y_true, merged["e02_score"].to_numpy(), 3.0, budget_per_100=0.27
        ).to_dict(),
        "E03_ewma_L3": evaluate_at_threshold(
            y_true, merged["e03_score"].to_numpy(), ewma_l, budget_per_100=0.27
        ).to_dict(),
        "E04_t2_99pct": evaluate_at_threshold(
            y_true, merged["t2"].to_numpy(), t2_lim_99, budget_per_100=1.0
        ).to_dict(),
        "E05_spe_99pct": evaluate_at_threshold(
            y_true, merged["spe"].to_numpy(), spe_lim_99, budget_per_100=1.0
        ).to_dict(),
    }

    primary_id = bundle.policy["primary_detector"]
    prim_res = detectors_holdout[primary_id]
    prim_pr_auc = prim_res["pr_auc"]
    prim_pr_ci_low = prim_res["wider_pr_auc_95_ci"][0]
    prim_rec_5 = prim_res["recall_at_far"]["far_5"]["recall"]
    prim_b5 = prim_res["alarm_budgets"]["budget_5"]
    prim_realised_alarm_rate = prim_b5["realised_alarms_per_100"]

    # O1: Primary detector holdout PR-AUC >= 2x holdout fail prevalence AND 95% bootstrap lower bound > prevalence (using quoted wider CI per §7.3)
    o1_target_val = float(cfg.targets.o1_pr_auc_min_prevalence_multiple * prevalence)
    o1_pass = bool(
        prim_pr_auc is not None
        and prim_pr_ci_low is not None
        and prim_pr_auc >= o1_target_val
        and prim_pr_ci_low > prevalence
    )

    # O2: Holdout recall >= 0.20 at 5 false alarms per 100 passing runs
    o2_pass = bool(prim_rec_5 is not None and prim_rec_5 >= cfg.targets.o2_min_recall_at_5_far)

    # O3: Threshold frozen for 5 alarms per 100 runs; realised holdout rate within 2-8 per 100
    o3_pass = bool(
        prim_realised_alarm_rate is not None
        and cfg.targets.o3_alarm_rate_min_per_100
        <= prim_realised_alarm_rate
        <= cfg.targets.o3_alarm_rate_max_per_100
    )

    # O4: Paired bootstrap PR-AUC difference between best ML and best SPC detector
    best_spc_holdout = max(
        ["E02", "E03", "E04", "E05", "E06"], key=lambda d: detectors_holdout[d]["pr_auc"] or -1.0
    )
    best_ml_holdout = max(["E07", "E08"], key=lambda d: detectors_holdout[d]["pr_auc"] or -1.0)
    dev_best_spc = bundle.policy["selection"]["best_spc_detector"]
    dev_best_ml = bundle.policy["selection"]["best_ml_detector"]

    o4_dev_pair_strat = paired_stratified_bootstrap_diff(
        y_true,
        merged[det_col_map[dev_best_ml]].to_numpy(),
        merged[det_col_map[dev_best_spc]].to_numpy(),
        n_bootstraps=cfg.policy.n_bootstraps,
        seed=cfg.project.seed,
    )
    o4_dev_pair_block = paired_moving_block_bootstrap_diff(
        y_true,
        merged[det_col_map[dev_best_ml]].to_numpy(),
        merged[det_col_map[dev_best_spc]].to_numpy(),
        block_length=cfg.policy.block_bootstrap_length,
        n_bootstraps=cfg.policy.n_bootstraps,
        seed=cfg.project.seed,
    )
    o4_holdout_best_pair = paired_stratified_bootstrap_diff(
        y_true,
        merged[det_col_map[best_ml_holdout]].to_numpy(),
        merged[det_col_map[best_spc_holdout]].to_numpy(),
        n_bootstraps=cfg.policy.n_bootstraps,
        seed=cfg.project.seed,
    )

    holdout_report = {
        "dataset_id": cfg.data.dataset_id,
        "release_id": release_id,
        "bundle_version": bundle.version,
        "predictions_csv_path": str(preds_path.as_posix()),
        "predictions_sha256_before_labels_opened": preds_sha256,
        "n_holdout_runs": n_runs,
        "n_holdout_fails": n_fails,
        "n_holdout_passing": n_passing,
        "holdout_fail_prevalence": prevalence,
        "primary_detector": primary_id,
        "detectors": detectors_holdout,
        "textbook_spc_limits_holdout": textbook_holdout,
        "objectives": {
            "O1_ranking_pr_auc": {
                "verdict": "PASS" if o1_pass else "MISS",
                "primary_detector": primary_id,
                "actual_pr_auc": prim_pr_auc,
                "stratified_95_ci": prim_res["stratified_bootstrap"]["pr_auc_95_ci"],
                "moving_block_95_ci": prim_res["moving_block_bootstrap"]["pr_auc_95_ci"],
                "wider_95_ci": prim_res["wider_pr_auc_95_ci"],
                "target_min_pr_auc_2x_prevalence": o1_target_val,
                "chance_prevalence": prevalence,
                "n_fails": n_fails,
            },
            "O2_recall_at_5_far": {
                "verdict": "PASS" if o2_pass else "MISS",
                "primary_detector": primary_id,
                "actual_recall_at_5_far": prim_rec_5,
                "stratified_95_ci": prim_res["stratified_bootstrap"]["recall_at_far_95_ci"],
                "moving_block_95_ci": prim_res["moving_block_bootstrap"]["recall_at_far_95_ci"],
                "wider_95_ci": prim_res["wider_recall_at_5_far_95_ci"],
                "target_min_recall": cfg.targets.o2_min_recall_at_5_far,
                "chance_recall": 0.05,
                "recall_at_2_far": prim_res["recall_at_far"]["far_2"]["recall"],
                "recall_at_10_far": prim_res["recall_at_far"]["far_10"]["recall"],
                "n_fails": n_fails,
            },
            "O3_alarm_budget_stability": {
                "verdict": "PASS" if o3_pass else "MISS",
                "primary_detector": primary_id,
                "target_budget_per_100": 5.0,
                "allowed_range_per_100": [
                    cfg.targets.o3_alarm_rate_min_per_100,
                    cfg.targets.o3_alarm_rate_max_per_100,
                ],
                "realised_alarms_per_100": prim_realised_alarm_rate,
                "alarms": prim_b5["alarms"],
                "true_alarms": prim_b5["true_alarms"],
                "false_alarms": prim_b5["false_alarms"],
                "n_runs": n_runs,
            },
            "O4_spc_vs_ml_comparison": {
                "verdict": "REPORTED",
                "dev_selected_ml": dev_best_ml,
                "dev_selected_spc": dev_best_spc,
                "diff_ml_minus_spc_dev_pair": o4_dev_pair_strat["diff_point"],
                "stratified_95_ci_dev_pair": o4_dev_pair_strat["diff_95_ci"],
                "moving_block_95_ci_dev_pair": o4_dev_pair_block["diff_95_ci"],
                "wider_95_ci_dev_pair": wider_confidence_interval(
                    o4_dev_pair_strat["diff_95_ci"], o4_dev_pair_block["diff_95_ci"]
                ),
                "holdout_best_ml": best_ml_holdout,
                "holdout_best_spc": best_spc_holdout,
                "diff_ml_minus_spc_holdout_best": o4_holdout_best_pair["diff_point"],
                "stratified_95_ci_holdout_best": o4_holdout_best_pair["diff_95_ci"],
                "n_fails": n_fails,
            },
        },
        "claims_boundary": (
            "Offline benchmark results on the public UCI SECOM dataset. "
            "Not measured yield improvement, scrap reduction, or cost savings."
        ),
    }

    metrics_path.write_text(json.dumps(holdout_report, indent=2) + "\n", encoding="utf-8")
    return holdout_report
