"""YieldGuard probability calibration, class-imbalance correction, and reliability assessment."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


def brier_score_loss(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Compute the Brier score loss (mean squared difference between probability and binary label).

    Lower values indicate better calibration and discrimination.
    """
    y_t = np.asarray(y_true, dtype=np.float64)
    y_p = np.asarray(y_prob, dtype=np.float64)
    if len(y_t) == 0:
        return 0.0
    return float(np.mean((y_p - y_t) ** 2))


def compute_ece(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> tuple[float, pd.DataFrame]:
    """Compute Expected Calibration Error (ECE) and bin-level reliability table.

    ECE is the weighted average absolute difference between predicted probability and
    observed empirical failure frequency across uniform bins in [0, 1].

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth binary failure labels (0 or 1).
    y_prob : np.ndarray
        Predicted posterior failure probabilities in [0, 1].
    n_bins : int, default=10
        Number of equal-width bins across the probability interval [0, 1].

    Returns
    -------
    ece : float
        Expected Calibration Error in [0, 1].
    bin_table : pd.DataFrame
        DataFrame with columns ['bin_idx', 'bin_lower', 'bin_upper', 'n_samples', 'mean_confidence', 'empirical_accuracy', 'calibration_error'].
    """
    y_t = np.asarray(y_true, dtype=np.int64)
    y_p = np.clip(np.asarray(y_prob, dtype=np.float64), 0.0, 1.0)
    n_total = len(y_t)

    if n_total == 0:
        return 0.0, pd.DataFrame()

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_records: list[dict[str, Any]] = []
    weighted_ece = 0.0

    for b in range(n_bins):
        low = bin_edges[b]
        high = bin_edges[b + 1]
        if b == n_bins - 1:
            mask = (y_p >= low) & (y_p <= high)
        else:
            mask = (y_p >= low) & (y_p < high)

        count = int(np.sum(mask))
        if count > 0:
            conf = float(np.mean(y_p[mask]))
            acc = float(np.mean(y_t[mask]))
            err = abs(acc - conf)
            weighted_ece += (count / n_total) * err
        else:
            conf = float((low + high) / 2.0)
            acc = 0.0
            err = 0.0

        bin_records.append(
            {
                "bin_idx": b + 1,
                "bin_lower": round(low, 4),
                "bin_upper": round(high, 4),
                "n_samples": count,
                "mean_confidence": round(conf, 4),
                "empirical_accuracy": round(acc, 4),
                "calibration_error": round(err, 4),
            }
        )

    bin_df = pd.DataFrame(bin_records)
    return float(weighted_ece), bin_df


def adjust_probability_for_imbalance(
    probs: np.ndarray,
    train_prevalence: float,
    target_prevalence: float,
    eps: float = 1e-12,
) -> np.ndarray:
    """Apply Bayes odds-ratio correction to probabilities estimated under class-weighted or resampled training sets.

    Parameters
    ----------
    probs : np.ndarray
        Model output probabilities trained with weighted loss or artificial resampling.
    train_prevalence : float
        Effective positive class prior in the training weighting scheme (e.g., 0.5 under balanced weights).
    target_prevalence : float
        True empirical marginal failure prevalence in the production population (e.g., 0.068 for SECOM).
    eps : float, default=1e-12
        Epsilon to prevent division by zero.

    Returns
    -------
    adjusted_probs : np.ndarray
        Unbiased posterior probabilities scaled to the target prevalence.
    """
    p = np.clip(np.asarray(probs, dtype=np.float64), eps, 1.0 - eps)
    t_prev = float(np.clip(target_prevalence, eps, 1.0 - eps))
    tr_prev = float(np.clip(train_prevalence, eps, 1.0 - eps))

    # Odds ratio correction: odds_target = odds_train * (target_prev / (1 - target_prev)) * ((1 - train_prev) / train_prev)
    odds_train = p / (1.0 - p)
    prior_ratio = (t_prev / (1.0 - t_prev)) * ((1.0 - tr_prev) / tr_prev)
    odds_target = odds_train * prior_ratio
    p_adjusted = odds_target / (1.0 + odds_target)
    return np.clip(p_adjusted, 0.0, 1.0)


@dataclass
class ProbabilityCalibrator:
    """Post-hoc probability calibrator supporting parametric Platt scaling (sigmoid) and non-parametric isotonic regression.

    Handles monotone transformation of continuous detector anomaly scores (MSPC, iForest, raw logits)
    or uncalibrated tree probabilities into well-calibrated probabilities.
    """

    method: Literal["sigmoid", "isotonic"] = "sigmoid"
    is_fitted: bool = False
    calibrator: Any = field(default=None, init=False)
    brier_score_: float = field(default=0.0, init=False)
    ece_: float = field(default=0.0, init=False)

    def fit(
        self,
        scores: np.ndarray | list[float],
        y_true: np.ndarray | list[int],
    ) -> ProbabilityCalibrator:
        """Fit calibration mapping from continuous scores to binary labels {0, 1}."""
        s = np.asarray(scores, dtype=np.float64).reshape(-1, 1)
        y = np.asarray(y_true, dtype=np.int64).ravel()

        if len(s) != len(y):
            raise ValueError(f"Length mismatch: {len(s)} scores vs {len(y)} labels.")
        if len(np.unique(y)) < 2:
            # Degenerate single-class case: cannot train logistic model, default to constant
            self.method = "sigmoid"
            self.calibrator = None
            self.is_fitted = True
            self.brier_score_ = 0.0
            self.ece_ = 0.0
            return self

        if self.method == "sigmoid":
            # Platt scaling: 1D Logistic Regression
            clf = LogisticRegression(solver="lbfgs", max_iter=1000)
            clf.fit(s, y)
            self.calibrator = clf
        elif self.method == "isotonic":
            # Isotonic Regression: monotonic piecewise constant mapping
            iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            iso.fit(s.ravel(), y)
            self.calibrator = iso
        else:
            raise ValueError(f"Unknown calibration method: '{self.method}'. Must be 'sigmoid' or 'isotonic'.")

        self.is_fitted = True
        preds = self.predict_proba(s.ravel())
        self.brier_score_ = brier_score_loss(y, preds)
        self.ece_, _ = compute_ece(y, preds)
        return self

    def predict_proba(self, scores: np.ndarray | list[float]) -> np.ndarray:
        """Predict calibrated posterior probability of positive class (fault)."""
        if not self.is_fitted:
            raise RuntimeError("ProbabilityCalibrator must be fitted before predict_proba.")

        s = np.asarray(scores, dtype=np.float64)
        if len(s) == 0:
            return np.empty(0, dtype=np.float64)

        if self.calibrator is None:
            # Degenerate single-class fallback
            return np.zeros(len(s), dtype=np.float64)

        if self.method == "sigmoid":
            return self.calibrator.predict_proba(s.reshape(-1, 1))[:, 1].astype(np.float64)
        else:
            return np.clip(self.calibrator.predict(s.ravel()), 0.0, 1.0).astype(np.float64)
