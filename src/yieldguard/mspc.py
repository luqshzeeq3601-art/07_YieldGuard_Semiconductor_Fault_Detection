"""Multivariate Phase I / Phase II SPC: PCA model, Hotelling T-squared, SPE/Q, limits, and contributions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.stats import chi2
from scipy.stats import f as f_dist
from sklearn.decomposition import PCA


@dataclass
class FittedMspcModel:
    """Fitted Phase I PCA model and theoretical control limits."""

    retained_columns: list[str]
    n_components: int
    variance_explained_target: float
    cumulative_variance_explained: float
    mean_vector: np.ndarray
    loadings: np.ndarray  # shape (p, A), orthonormal columns
    eigenvalues: np.ndarray  # shape (A,)
    t2_limit_99: float
    spe_limit_99: float
    t2_limit_999_initial: float
    spe_limit_999_initial: float
    n_initial_passing_runs: int
    n_trimmed_runs: int
    n_final_passing_runs: int
    spe_phase1_mean: float
    spe_phase1_var: float

    def to_metadata_dict(self) -> dict[str, Any]:
        return {
            "n_features": len(self.retained_columns),
            "n_components": self.n_components,
            "variance_explained_target": self.variance_explained_target,
            "cumulative_variance_explained": self.cumulative_variance_explained,
            "t2_limit_99": self.t2_limit_99,
            "spe_limit_99": self.spe_limit_99,
            "t2_limit_999_initial": self.t2_limit_999_initial,
            "spe_limit_999_initial": self.spe_limit_999_initial,
            "n_initial_passing_runs": self.n_initial_passing_runs,
            "n_trimmed_runs": self.n_trimmed_runs,
            "n_final_passing_runs": self.n_final_passing_runs,
            "spe_phase1_mean": self.spe_phase1_mean,
            "spe_phase1_var": self.spe_phase1_var,
        }

    def save(self, path: str | Path) -> Path:
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, out_path)
        return out_path

    @staticmethod
    def load(path: str | Path) -> FittedMspcModel:
        loaded = joblib.load(Path(path))
        if not isinstance(loaded, FittedMspcModel):
            raise TypeError(f"Expected FittedMspcModel, got {type(loaded).__name__}")
        return loaded


def compute_theoretical_t2_limit(n_samples: int, n_components: int, alpha: float = 0.99) -> float:
    """Compute theoretical Hotelling T-squared control limit from the F distribution."""
    a = int(n_components)
    n = int(n_samples)
    if n <= a:
        # Fallback to chi-square approximation if sample size is smaller than components
        return float(chi2.ppf(alpha, df=a))
    f_crit = float(f_dist.ppf(alpha, dfn=a, dfd=n - a))
    return float((a * (n - 1) / (n - a)) * f_crit)


def compute_theoretical_spe_limit(spe_values: np.ndarray, alpha: float = 0.99) -> tuple[float, float, float]:
    """Compute moment-matched scaled chi-square SPE limit (g * chi2_alpha(h)).

    From docs/04_TECHNICAL_DESIGN.md section 5.6:
    `g = var / (2 * mean)`, `h = 2 * mean^2 / var`.
    """
    m = float(np.mean(spe_values))
    v = float(np.var(spe_values, ddof=1)) if len(spe_values) > 1 else 0.0
    if m <= 1e-12 or v <= 1e-12:
        return float(max(m, 1e-6)), m, v
    g = v / (2.0 * m)
    h = (2.0 * (m**2)) / v
    limit = float(g * chi2.ppf(alpha, df=h))
    return limit, m, v


def _fit_pca_pass(
    x_pass: np.ndarray,
    variance_target: float,
    max_components: int,
    n_components_fixed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Fit PCA on passing runs and extract mean, orthonormal loadings P (p x A), and eigenvalues."""
    n_samples, n_features = x_pass.shape
    max_possible = min(n_samples - 1, n_features, max_components)
    if max_possible < 1:
        max_possible = 1

    pca = PCA(svd_solver="full")
    pca.fit(x_pass)

    ratios = pca.explained_variance_ratio_
    cum_ratios = np.cumsum(ratios)

    if n_components_fixed is not None:
        a = min(int(n_components_fixed), len(ratios))
    else:
        hits = np.flatnonzero(cum_ratios >= variance_target)
        a_target = int(hits[0] + 1) if len(hits) > 0 else len(ratios)
        a = min(a_target, max_possible)
        a = max(1, a)

    mean_vec = pca.mean_.astype(np.float64)
    loadings = pca.components_[:a].T.astype(np.float64)  # shape (p, a)
    eigenvalues = np.maximum(pca.explained_variance_[:a].astype(np.float64), 1e-12)
    cum_explained = float(cum_ratios[a - 1])
    return mean_vec, loadings, eigenvalues, cum_explained


def compute_t2_and_spe(
    x_matrix: np.ndarray,
    mean_vec: np.ndarray,
    loadings: np.ndarray,
    eigenvalues: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute T-squared, SPE, T-squared sensor contributions, and SPE sensor contributions.

    For centered matrix `X_c = X - mean_vec` (shape N x p) and orthonormal loadings `P` (shape p x A):
    - Scores `T = X_c @ P` (shape N x A)
    - `T2 = sum_a (T_{i,a}^2 / lambda_a)` (shape N,)
    - Residuals `E = X_c - T @ P.T` (shape N x p)
    - `SPE = sum_j E_{i,j}^2` (shape N,)
    - `SPE contributions = E^2` (shape N x p); row sums equal `SPE`.
    - `T2 direct inner-product contributions = X_c * ((T / lambda) @ P.T)` (shape N x p);
      row sums equal `T2` identically, and any sensor at its Phase I mean has 0 contribution.
    """
    x_c = np.asarray(x_matrix, dtype=np.float64) - mean_vec[None, :]
    t_scores = x_c @ loadings  # (N, A)
    inv_eig = 1.0 / eigenvalues  # (A,)
    t2 = np.sum((t_scores**2) * inv_eig[None, :], axis=1)

    x_hat = t_scores @ loadings.T  # (N, p)
    residuals = x_c - x_hat  # (N, p)
    spe_contribs = residuals**2  # (N, p)
    spe = np.sum(spe_contribs, axis=1)

    t2_direction = (t_scores * inv_eig[None, :]) @ loadings.T  # (N, p)
    t2_contribs = x_c * t2_direction  # (N, p)

    return t2, spe, t2_contribs, spe_contribs


def fit_multivariate_spc(
    train_scaled_df: pd.DataFrame,
    train_labels_fail: np.ndarray | list[int],
    variance_explained_target: float = 0.80,
    max_components: int = 50,
    phase1_trim_alpha: float = 0.999,
    limit_alpha: float = 0.99,
    enable_trimming: bool = True,
    n_components_fixed: int | None = None,
) -> FittedMspcModel:
    """Fit Phase I PCA model strictly on passing (`label_fail == 0`) training runs.

    Per docs/03_DATA_SPEC.md section 6.2:
    One declared trimming pass: remove Phase I runs above the 99.9% theoretical T-squared
    or SPE limit, refit once, and record the number removed.
    """
    y = np.asarray(train_labels_fail, dtype=np.int64)
    if len(train_scaled_df) != len(y):
        raise ValueError("Length mismatch between train_scaled_df and train_labels_fail.")

    pass_mask = y == 0
    if not np.any(pass_mask):
        raise ValueError("Training data contains zero passing runs for PCA fitting.")

    cols = list(train_scaled_df.columns)
    x_pass_init = np.ascontiguousarray(train_scaled_df.loc[pass_mask, cols].to_numpy(dtype=np.float64, copy=True))
    n_init = len(x_pass_init)

    mean_1, load_1, eig_1, cum_1 = _fit_pca_pass(
        x_pass_init,
        variance_target=variance_explained_target,
        max_components=max_components,
        n_components_fixed=n_components_fixed,
    )
    t2_1, spe_1, _, _ = compute_t2_and_spe(x_pass_init, mean_1, load_1, eig_1)
    t2_lim_999 = compute_theoretical_t2_limit(n_init, load_1.shape[1], alpha=phase1_trim_alpha)
    spe_lim_999, _, _ = compute_theoretical_spe_limit(spe_1, alpha=phase1_trim_alpha)

    n_trimmed = 0
    x_pass_final = x_pass_init
    if enable_trimming:
        keep_mask = (t2_1 <= t2_lim_999) & (spe_1 <= spe_lim_999)
        n_keep = int(np.sum(keep_mask))
        if 0 < n_keep < n_init and n_keep > (load_1.shape[1] + 2):
            n_trimmed = int(n_init - n_keep)
            x_pass_final = x_pass_init[keep_mask]

    if n_trimmed > 0:
        mean_f, load_f, eig_f, cum_f = _fit_pca_pass(
            x_pass_final,
            variance_target=variance_explained_target,
            max_components=max_components,
            n_components_fixed=n_components_fixed,
        )
    else:
        mean_f, load_f, eig_f, cum_f = mean_1, load_1, eig_1, cum_1

    n_final = len(x_pass_final)
    n_comp = int(load_f.shape[1])
    _, spe_f, _, _ = compute_t2_and_spe(x_pass_final, mean_f, load_f, eig_f)

    t2_lim_99 = compute_theoretical_t2_limit(n_final, n_comp, alpha=limit_alpha)
    spe_lim_99, spe_m, spe_v = compute_theoretical_spe_limit(spe_f, alpha=limit_alpha)

    return FittedMspcModel(
        retained_columns=cols,
        n_components=n_comp,
        variance_explained_target=variance_explained_target,
        cumulative_variance_explained=cum_f,
        mean_vector=mean_f,
        loadings=load_f,
        eigenvalues=eig_f,
        t2_limit_99=t2_lim_99,
        spe_limit_99=spe_lim_99,
        t2_limit_999_initial=t2_lim_999,
        spe_limit_999_initial=spe_lim_999,
        n_initial_passing_runs=n_init,
        n_trimmed_runs=n_trimmed,
        n_final_passing_runs=n_final,
        spe_phase1_mean=spe_m,
        spe_phase1_var=spe_v,
    )


def score_multivariate_spc(
    mspc_model: FittedMspcModel,
    scaled_df: pd.DataFrame,
    include_contributions: bool = False,
) -> dict[str, np.ndarray]:
    """Score scaled runs with T-squared (E04), SPE (E05), and combined MSPC index (E06)."""
    x = scaled_df[mspc_model.retained_columns].to_numpy(dtype=np.float64)
    t2, spe, t2_contribs, spe_contribs = compute_t2_and_spe(
        x,
        mspc_model.mean_vector,
        mspc_model.loadings,
        mspc_model.eigenvalues,
    )
    mspc_index = (t2 / mspc_model.t2_limit_99) + (spe / mspc_model.spe_limit_99)
    out = {
        "t2": t2,
        "spe": spe,
        "mspc_index": mspc_index,
    }
    if include_contributions:
        out["t2_contributions"] = t2_contribs
        out["spe_contributions"] = spe_contribs
        out["mspc_contributions"] = (t2_contribs / mspc_model.t2_limit_99) + (
            spe_contribs / mspc_model.spe_limit_99
        )
    return out

