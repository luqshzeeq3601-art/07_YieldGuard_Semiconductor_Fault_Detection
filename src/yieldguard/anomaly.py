"""Isolation Forest anomaly detector (E07) fitted on passing training runs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


@dataclass
class FittedAnomalyModel:
    """Container for fitted IsolationForest and retained column contract."""

    retained_columns: list[str]
    model: IsolationForest
    n_estimators: int
    max_samples: str | int | float
    seed: int
    fit_on_passing_only: bool
    n_training_runs_used: int

    def to_metadata_dict(self) -> dict[str, Any]:
        return {
            "n_features": len(self.retained_columns),
            "n_estimators": self.n_estimators,
            "max_samples": self.max_samples,
            "seed": self.seed,
            "fit_on_passing_only": self.fit_on_passing_only,
            "n_training_runs_used": self.n_training_runs_used,
        }

    def save(self, path: str | Path) -> Path:
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, out_path)
        return out_path

    @staticmethod
    def load(path: str | Path) -> FittedAnomalyModel:
        loaded = joblib.load(Path(path))
        if not isinstance(loaded, FittedAnomalyModel):
            raise TypeError(f"Expected FittedAnomalyModel, got {type(loaded).__name__}")
        return loaded


def fit_isolation_forest(
    train_scaled_df: pd.DataFrame,
    train_labels_fail: np.ndarray | list[int],
    n_estimators: int = 200,
    max_samples: str | int | float = "auto",
    seed: int = 42,
    fit_on_passing_only: bool = True,
) -> FittedAnomalyModel:
    """Fit an IsolationForest on passing training runs (`label_fail == 0`) by default."""
    y = np.asarray(train_labels_fail, dtype=np.int64)
    if len(train_scaled_df) != len(y):
        raise ValueError("Length mismatch between train_scaled_df and train_labels_fail.")

    cols = list(train_scaled_df.columns)
    if fit_on_passing_only:
        pass_mask = y == 0
        if not np.any(pass_mask):
            raise ValueError("Zero passing runs available for IsolationForest training.")
        x_fit = np.ascontiguousarray(train_scaled_df.loc[pass_mask, cols].to_numpy(dtype=np.float64, copy=True))
    else:
        x_fit = np.ascontiguousarray(train_scaled_df[cols].to_numpy(dtype=np.float64, copy=True))

    clf = IsolationForest(
        n_estimators=n_estimators,
        max_samples=max_samples,
        random_state=seed,
        n_jobs=1,
    )
    clf.fit(x_fit)

    return FittedAnomalyModel(
        retained_columns=cols,
        model=clf,
        n_estimators=n_estimators,
        max_samples=max_samples,
        seed=seed,
        fit_on_passing_only=fit_on_passing_only,
        n_training_runs_used=len(x_fit),
    )


def score_isolation_forest(
    fitted_model: FittedAnomalyModel,
    scaled_df: pd.DataFrame,
) -> np.ndarray:
    """Compute `-score_samples(X)` so higher scores indicate more anomalous runs."""
    x = np.ascontiguousarray(scaled_df[fitted_model.retained_columns].to_numpy(dtype=np.float64, copy=True))
    return -fitted_model.model.score_samples(x).astype(np.float64)
