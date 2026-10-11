"""Univariate Phase I / Phase II SPC: sensor selection, EWMA charts, and Western Electric run rules."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import false_discovery_control, mannwhitneyu


@dataclass
class MonitoredSensorSpec:
    """Phase I parameters and selection statistics for a single monitored sensor."""

    sensor: str
    mu0: float
    sigma0: float
    training_missing_fraction: float
    rank_biserial: float
    abs_rank_biserial: float
    p_value: float
    bh_adjusted_p_value: float
    group_members: list[str]


@dataclass
class FittedSpcModel:
    """Fitted Phase I univariate SPC parameters across all monitored sensors."""

    monitored_sensors: list[str]
    sensor_specs: dict[str, MonitoredSensorSpec]
    ewma_lambda: float
    ewma_l: float
    mr_d2: float
    n_screened_columns: int
    n_train_runs: int
    n_passing_train_runs: int
    n_failing_train_runs: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "monitored_sensors": self.monitored_sensors,
            "sensor_specs": {k: asdict(v) for k, v in self.sensor_specs.items()},
            "ewma_lambda": self.ewma_lambda,
            "ewma_l": self.ewma_l,
            "mr_d2": self.mr_d2,
            "n_screened_columns": self.n_screened_columns,
            "n_train_runs": self.n_train_runs,
            "n_passing_train_runs": self.n_passing_train_runs,
            "n_failing_train_runs": self.n_failing_train_runs,
        }

    def save(self, path: str | Path) -> Path:
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
        return out_path

    @staticmethod
    def load(path: str | Path) -> FittedSpcModel:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        specs = {
            k: MonitoredSensorSpec(**v) for k, v in raw["sensor_specs"].items()
        }
        return FittedSpcModel(
            monitored_sensors=list(raw["monitored_sensors"]),
            sensor_specs=specs,
            ewma_lambda=float(raw["ewma_lambda"]),
            ewma_l=float(raw["ewma_l"]),
            mr_d2=float(raw["mr_d2"]),
            n_screened_columns=int(raw["n_screened_columns"]),
            n_train_runs=int(raw["n_train_runs"]),
            n_passing_train_runs=int(raw["n_passing_train_runs"]),
            n_failing_train_runs=int(raw["n_failing_train_runs"]),
        )


def compute_phase1_center_and_mr_sigma(
    passing_values: np.ndarray,
    mr_d2: float = 1.128,
) -> tuple[float, float]:
    """Compute Phase I centre line (mean) and sigma = average moving range / 1.128."""
    clean = passing_values[~np.isnan(passing_values)]
    if len(clean) == 0:
        return 0.0, 1.0
    mu0 = float(np.mean(clean))
    if len(clean) < 2:
        return mu0, 1.0
    mr = np.abs(np.diff(clean))
    amr = float(np.mean(mr))
    sigma0 = float(amr / mr_d2)
    if sigma0 < 1e-9 or np.isnan(sigma0):
        std_fallback = float(np.std(clean, ddof=1))
        sigma0 = std_fallback if std_fallback >= 1e-9 else 1.0
    return mu0, sigma0


def fit_univariate_spc(
    train_retained_df: pd.DataFrame,
    train_labels_fail: np.ndarray | list[int],
    group_map: dict[str, list[str]] | None = None,
    n_monitored: int = 10,
    max_sensor_missing_fraction: float = 0.05,
    ewma_lambda: float = 0.20,
    ewma_l: float = 3.0,
    mr_d2: float = 1.128,
) -> FittedSpcModel:
    """Select top `n_monitored` sensors on training runs only and fit Phase I SPC parameters.

    Selection criteria (docs/04_TECHNICAL_DESIGN.md section 5.3):
    - Candidate columns: retained columns with training missingness <= `max_sensor_missing_fraction` (0.05).
    - Ranked by largest absolute rank-biserial correlation between failing and passing training runs.
    - Benjamini-Hochberg adjusted p-values are recorded beside the selection.
    - Phase I centre line and moving-range sigma are fitted on passing training runs in time order.
    """
    y = np.asarray(train_labels_fail, dtype=np.int64)
    if len(train_retained_df) != len(y):
        raise ValueError("Length mismatch between train_retained_df and train_labels_fail.")

    pass_mask = y == 0
    fail_mask = y == 1
    if not np.any(pass_mask) or not np.any(fail_mask):
        raise ValueError("Training data must contain both passing and failing runs for sensor screening.")

    cols = list(train_retained_df.columns)
    eligible_cols: list[str] = []
    miss_fracs: dict[str, float] = {}
    for c in cols:
        m_frac = float(train_retained_df[c].isna().mean())
        miss_fracs[c] = m_frac
        if m_frac <= max_sensor_missing_fraction:
            eligible_cols.append(c)

    if not eligible_cols:
        # Fallback if all columns exceed 5% missing in a tiny synthetic test
        eligible_cols = cols

    r_rb_list: list[float] = []
    p_val_list: list[float] = []

    for c in eligible_cols:
        col_vals = train_retained_df[c].to_numpy(dtype=np.float64)
        x1 = col_vals[fail_mask]
        x0 = col_vals[pass_mask]
        x1 = x1[~np.isnan(x1)]
        x0 = x0[~np.isnan(x0)]
        if len(x1) == 0 or len(x0) == 0:
            r_rb_list.append(0.0)
            p_val_list.append(1.0)
            continue
        res = mannwhitneyu(x1, x0, alternative="two-sided")
        u_stat = float(res.statistic)
        r_rb = (2.0 * u_stat) / (len(x1) * len(x0)) - 1.0
        r_rb_list.append(float(r_rb))
        p_val_list.append(float(res.pvalue))

    p_arr = np.clip(np.asarray(p_val_list, dtype=np.float64), 0.0, 1.0)
    bh_adj = false_discovery_control(p_arr, method="bh") if len(p_arr) > 0 else np.array([])

    # Rank by (-abs(r_rb), column_position)
    scored_candidates = []
    for idx, c in enumerate(eligible_cols):
        r_val = r_rb_list[idx]
        scored_candidates.append(
            (
                -abs(r_val),
                idx,
                c,
                r_val,
                abs(r_val),
                float(p_arr[idx]),
                float(bh_adj[idx]),
            )
        )
    scored_candidates.sort(key=lambda item: (item[0], item[1]))
    selected = scored_candidates[: min(n_monitored, len(scored_candidates))]

    monitored_sensors: list[str] = []
    sensor_specs: dict[str, MonitoredSensorSpec] = {}
    g_map = group_map or {}

    for _, _, c, r_val, abs_r, p_val, bh_p in selected:
        pass_vals = train_retained_df.loc[pass_mask, c].to_numpy(dtype=np.float64)
        mu0, sigma0 = compute_phase1_center_and_mr_sigma(pass_vals, mr_d2=mr_d2)
        monitored_sensors.append(c)
        sensor_specs[c] = MonitoredSensorSpec(
            sensor=c,
            mu0=mu0,
            sigma0=sigma0,
            training_missing_fraction=miss_fracs[c],
            rank_biserial=r_val,
            abs_rank_biserial=abs_r,
            p_value=p_val,
            bh_adjusted_p_value=bh_p,
            group_members=list(g_map.get(c, [])),
        )

    return FittedSpcModel(
        monitored_sensors=monitored_sensors,
        sensor_specs=sensor_specs,
        ewma_lambda=ewma_lambda,
        ewma_l=ewma_l,
        mr_d2=mr_d2,
        n_screened_columns=len(eligible_cols),
        n_train_runs=int(len(y)),
        n_passing_train_runs=int(np.sum(pass_mask)),
        n_failing_train_runs=int(np.sum(fail_mask)),
    )


def compute_ewma_series(
    values: np.ndarray | list[float],
    mu0: float,
    sigma0: float,
    lam: float = 0.20,
    L: float = 3.0,
) -> dict[str, np.ndarray]:
    """Compute EWMA series, time-varying control limits, and standardised EWMA distance.

    Formula (docs/04_TECHNICAL_DESIGN.md section 5.1):
    - `z_0 = mu0`
    - `z_t = lam * x_t + (1 - lam) * z_{t-1}` if `x_t` is not NaN, else `z_{t-1}`
    - `sigma_z(t) = sigma0 * sqrt((lam / (2 - lam)) * (1 - (1 - lam)**(2*t)))` for t = 1..T
    - `ucl_t = mu0 + L * sigma_z(t)`, `lcl_t = mu0 - L * sigma_z(t)`
    - `std_ewma_distance_t = |z_t - mu0| / sigma_z(t)` (exceeds `L` iff `z_t` is outside `[lcl_t, ucl_t]`)
    """
    x = np.asarray(values, dtype=np.float64)
    n = len(x)
    z = np.empty(n, dtype=np.float64)
    is_missing = np.isnan(x)

    prev_z = float(mu0)
    for t_idx in range(n):
        if is_missing[t_idx]:
            z[t_idx] = prev_z
        else:
            prev_z = lam * float(x[t_idx]) + (1.0 - lam) * prev_z
            z[t_idx] = prev_z

    t_steps = np.arange(1, n + 1, dtype=np.float64)
    factor = np.sqrt((lam / (2.0 - lam)) * (1.0 - np.power(1.0 - lam, 2.0 * t_steps)))
    sigma_z = sigma0 * factor
    ucl = mu0 + L * sigma_z
    lcl = mu0 - L * sigma_z
    std_dist = np.abs(z - mu0) / sigma_z
    alarm = std_dist > L

    return {
        "x": x,
        "z": z,
        "ucl": ucl,
        "lcl": lcl,
        "sigma_z": sigma_z,
        "std_ewma_distance": std_dist,
        "ewma_alarm": alarm,
        "is_missing": is_missing,
    }


def evaluate_run_rules(
    values: np.ndarray | list[float],
    mu0: float,
    sigma0: float,
) -> dict[str, np.ndarray]:
    """Evaluate the four Western Electric / Nelson run rules on standardised individuals.

    Rules (docs/04_TECHNICAL_DESIGN.md section 5.2):
    1. `rule_1_beyond_3sigma`: 1 point beyond 3 sigma (`|u_t| > 3.0`).
    2. `rule_2_of_3_beyond_2sigma`: 2 of 3 consecutive points beyond 2 sigma on the same side.
    3. `rule_4_of_5_beyond_1sigma`: 4 of 5 consecutive points beyond 1 sigma on the same side.
    4. `rule_8_on_one_side`: 8 consecutive points on the same side of the centre line.
    """
    x = np.asarray(values, dtype=np.float64)
    u = (x - mu0) / sigma0
    n = len(u)

    r1 = np.zeros(n, dtype=bool)
    r2 = np.zeros(n, dtype=bool)
    r3 = np.zeros(n, dtype=bool)
    r4 = np.zeros(n, dtype=bool)

    pos_3 = u > 3.0
    neg_3 = u < -3.0
    pos_2 = u > 2.0
    neg_2 = u < -2.0
    pos_1 = u > 1.0
    neg_1 = u < -1.0
    pos_0 = u > 0.0
    neg_0 = u < 0.0

    for t in range(n):
        if pos_3[t] or neg_3[t]:
            r1[t] = True

        if t >= 2:
            if np.sum(pos_2[t - 2 : t + 1]) >= 2 or np.sum(neg_2[t - 2 : t + 1]) >= 2:
                r2[t] = True

        if t >= 4:
            if np.sum(pos_1[t - 4 : t + 1]) >= 4 or np.sum(neg_1[t - 4 : t + 1]) >= 4:
                r3[t] = True

        if t >= 7:
            if np.all(pos_0[t - 7 : t + 1]) or np.all(neg_0[t - 7 : t + 1]):
                r4[t] = True

    any_rule = r1 | r2 | r3 | r4
    return {
        "u_standardised": u,
        "rule_1_beyond_3sigma": r1,
        "rule_2_of_3_beyond_2sigma": r2,
        "rule_4_of_5_beyond_1sigma": r3,
        "rule_8_on_one_side": r4,
        "any_rule_violation": any_rule,
    }


def score_univariate_spc(
    spc_model: FittedSpcModel,
    runs_df: pd.DataFrame,
    lam_override: float | None = None,
) -> dict[str, Any]:
    """Score an ordered sequence of runs on all monitored sensors.

    Returns:
    - `e02_scores`: per-run maximum absolute standardised individual value over monitored sensors.
    - `e03_scores`: per-run maximum standardised EWMA distance over monitored sensors.
    - `charts`: per-sensor dictionary with EWMA series, limits, and four run-rule flags.
    """
    lam = float(lam_override) if lam_override is not None else spc_model.ewma_lambda
    L = spc_model.ewma_l
    n_runs = len(runs_df)
    n_mon = len(spc_model.monitored_sensors)

    ind_abs_mat = np.zeros((n_runs, n_mon), dtype=np.float64)
    ewma_dist_mat = np.zeros((n_runs, n_mon), dtype=np.float64)
    charts: dict[str, dict[str, Any]] = {}

    for j, s_name in enumerate(spc_model.monitored_sensors):
        spec = spc_model.sensor_specs[s_name]
        vals = runs_df[s_name].to_numpy(dtype=np.float64)
        ewma_res = compute_ewma_series(vals, spec.mu0, spec.sigma0, lam=lam, L=L)
        rules_res = evaluate_run_rules(vals, spec.mu0, spec.sigma0)

        u_abs = np.abs(rules_res["u_standardised"])
        u_abs = np.where(np.isnan(u_abs), 0.0, u_abs)
        ind_abs_mat[:, j] = u_abs
        ewma_dist_mat[:, j] = ewma_res["std_ewma_distance"]

        charts[s_name] = {
            "mu0": spec.mu0,
            "sigma0": spec.sigma0,
            "ewma_lambda": lam,
            "ewma_l": L,
            **ewma_res,
            **rules_res,
        }

    e02_scores = np.max(ind_abs_mat, axis=1) if n_mon > 0 else np.zeros(n_runs)
    e03_scores = np.max(ewma_dist_mat, axis=1) if n_mon > 0 else np.zeros(n_runs)

    return {
        "e02_scores": e02_scores,
        "e03_scores": e03_scores,
        "charts": charts,
    }


def simulate_in_control_arl(
    lam: float = 0.20,
    L: float = 3.0,
    n_trials: int = 1500,
    max_steps: int = 4000,
    seed: int = 42,
) -> float:
    """Monte Carlo estimate of in-control average run length (ARL0) for normal(0, 1) EWMA chart."""
    rng = np.random.default_rng(seed)
    t_steps = np.arange(1, max_steps + 1, dtype=np.float64)
    sigma_z = np.sqrt((lam / (2.0 - lam)) * (1.0 - np.power(1.0 - lam, 2.0 * t_steps)))
    limit = L * sigma_z

    run_lengths = np.empty(n_trials, dtype=np.float64)
    for trial in range(n_trials):
        x = rng.normal(0.0, 1.0, size=max_steps)
        z = 0.0
        rl = max_steps
        for t in range(max_steps):
            z = lam * x[t] + (1.0 - lam) * z
            if abs(z) > limit[t]:
                rl = t + 1
                break
        run_lengths[trial] = float(rl)

    return float(np.mean(run_lengths))
