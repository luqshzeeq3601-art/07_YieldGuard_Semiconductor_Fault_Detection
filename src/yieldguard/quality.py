"""Train-only 5-step data-quality pruning, group mapping, imputation, and passing-run scaling."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from yieldguard.data import SENSOR_COLUMNS


@dataclass
class QualityTransformer:
    """Fitted train-only data cleaning and preprocessing state."""

    input_columns: list[str]
    dropped_high_missing: list[str]
    dropped_constant: list[str]
    dropped_collinear: list[str]
    retained_columns: list[str]
    dropped_to_representative: dict[str, str]
    representative_to_members: dict[str, list[str]]
    training_missing_fractions: dict[str, float]
    medians: dict[str, float]
    passing_means: dict[str, float]
    passing_stds: dict[str, float]
    max_missing_fraction: float
    max_constant_dominant_fraction: float
    collinearity_threshold: float
    n_train_runs: int
    n_passing_train_runs: int

    def summary_counts(self) -> dict[str, Any]:
        """Return summary counts of dropped and retained columns."""
        return {
            "n_input_columns": len(self.input_columns),
            "n_dropped_high_missing": len(self.dropped_high_missing),
            "n_dropped_constant": len(self.dropped_constant),
            "n_dropped_collinear": len(self.dropped_collinear),
            "n_retained_columns": len(self.retained_columns),
            "n_train_runs": self.n_train_runs,
            "n_passing_train_runs": self.n_passing_train_runs,
        }

    def get_group_members(self, sensor: str) -> list[str]:
        """Return collinear group members represented by `sensor`."""
        return list(self.representative_to_members.get(sensor, []))

    def transform_retained(self, df: pd.DataFrame) -> pd.DataFrame:
        """Select retained columns in exact fitted order, leaving NaNs in place (for XGBoost / EWMA)."""
        missing = [c for c in self.retained_columns if c not in df.columns]
        if missing:
            raise ValueError(f"Input DataFrame missing retained columns: {missing[:5]}")
        return df[self.retained_columns].copy()

    def transform_imputed(self, df: pd.DataFrame) -> pd.DataFrame:
        """Select retained columns in exact fitted order and impute NaNs with training medians."""
        out = self.transform_retained(df)
        for col in self.retained_columns:
            if out[col].isna().any():
                out[col] = out[col].fillna(self.medians[col])
        return out

    def transform_scaled(self, df: pd.DataFrame) -> pd.DataFrame:
        """Impute with training medians and scale by passing-training-run mean and std (for PCA / IForest)."""
        out = self.transform_imputed(df)
        arr = out.to_numpy(dtype=np.float64, copy=True)
        means = np.array([self.passing_means[c] for c in self.retained_columns], dtype=np.float64)
        stds = np.array([self.passing_stds[c] for c in self.retained_columns], dtype=np.float64)
        scaled = (arr - means) / stds
        return pd.DataFrame(scaled, index=out.index, columns=self.retained_columns)

    def save(self, path: str | Path) -> Path:
        """Persist fitted QualityTransformer via joblib."""
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, out_path)
        return out_path

    @staticmethod
    def load(path: str | Path) -> QualityTransformer:
        """Load persisted QualityTransformer."""
        loaded = joblib.load(Path(path))
        if not isinstance(loaded, QualityTransformer):
            raise TypeError(f"Expected QualityTransformer, got {type(loaded).__name__}")
        return loaded


def fit_quality_transformer(
    train_sensors_df: pd.DataFrame,
    train_labels_fail: np.ndarray | pd.Series | list[int],
    max_missing_fraction: float = 0.50,
    max_constant_dominant_fraction: float = 0.99,
    collinearity_threshold: float = 0.95,
    sensor_columns: list[str] | None = None,
) -> QualityTransformer:
    """Fit all 5 data-quality steps strictly on the training runs of the current fold.

    Steps:
    1. Missingness pruning: drop columns with training missing fraction > `max_missing_fraction` (0.50).
    2. Constant pruning: drop columns with <= 1 distinct non-missing value, or most-frequent
       non-missing value covering >= `max_constant_dominant_fraction` (0.99) of non-missing values.
    3. Collinearity reduction: pairwise-complete Pearson |r| >= `collinearity_threshold` (0.95);
       keep column with lower missing fraction, then lower column index; record group map.
    4. Imputation: training median per retained column.
    5. Scaling: mean and std of passing (`label_fail == 0`) training runs after median imputation.
    """
    cols = list(sensor_columns) if sensor_columns is not None else [
        c for c in SENSOR_COLUMNS if c in train_sensors_df.columns
    ]
    if not cols:
        raise ValueError("No sensor columns provided to fit_quality_transformer.")

    x_train = train_sensors_df[cols].copy()
    y_fail = np.asarray(train_labels_fail, dtype=np.int64)
    if len(x_train) != len(y_fail):
        raise ValueError(
            f"Length mismatch between train_sensors_df ({len(x_train)}) and train_labels_fail ({len(y_fail)})."
        )

    n_train = len(x_train)
    passing_mask = y_fail == 0
    if not np.any(passing_mask):
        raise ValueError("Training fold contains zero passing runs.")

    # Step 1: Missingness pruning (> 0.50)
    missing_fracs: dict[str, float] = {}
    dropped_high_missing: list[str] = []
    after_step1: list[str] = []

    for col in cols:
        m_frac = float(x_train[col].isna().mean())
        missing_fracs[col] = m_frac
        if m_frac > max_missing_fraction:
            dropped_high_missing.append(col)
        else:
            after_step1.append(col)

    # Step 2: Constant / near-constant pruning (>= 0.99 dominant value among non-missing)
    dropped_constant: list[str] = []
    after_step2: list[str] = []

    for col in after_step1:
        non_na = x_train[col].dropna()
        n_non_na = len(non_na)
        if n_non_na <= 1:
            dropped_constant.append(col)
            continue
        val_counts = non_na.value_counts()
        if len(val_counts) <= 1:
            dropped_constant.append(col)
            continue
        top_share = float(val_counts.iloc[0] / n_non_na)
        if top_share >= max_constant_dominant_fraction:
            dropped_constant.append(col)
        else:
            after_step2.append(col)

    # Step 3: Collinearity reduction (|r| >= 0.95)
    # Order candidates by (missing_fraction, original_col_position) so preferred column always comes first
    col_pos = {c: i for i, c in enumerate(cols)}
    pref_order = sorted(after_step2, key=lambda c: (missing_fracs[c], col_pos[c]))

    corr_df = x_train[pref_order].corr(method="pearson", min_periods=3).abs()
    corr_mat = corr_df.to_numpy(dtype=np.float64)

    dropped_collinear_set: set[str] = set()
    dropped_to_rep: dict[str, str] = {}
    rep_to_members: dict[str, list[str]] = {}

    n_cand = len(pref_order)
    for i in range(n_cand):
        col_i = pref_order[i]
        if col_i in dropped_collinear_set:
            continue
        rep_to_members[col_i] = []
        for j in range(i + 1, n_cand):
            col_j = pref_order[j]
            if col_j in dropped_collinear_set:
                continue
            r_ij = corr_mat[i, j]
            if not np.isnan(r_ij) and r_ij >= collinearity_threshold:
                dropped_collinear_set.add(col_j)
                dropped_to_rep[col_j] = col_i
                rep_to_members[col_i].append(col_j)

    # Preserve canonical column order for retained_columns and dropped_collinear
    dropped_collinear = [c for c in after_step2 if c in dropped_collinear_set]
    retained_columns = [c for c in after_step2 if c not in dropped_collinear_set]
    for rep in rep_to_members:
        rep_to_members[rep] = sorted(rep_to_members[rep], key=lambda c: col_pos[c])

    # Step 4: Training medians per retained column
    medians: dict[str, float] = {}
    for col in retained_columns:
        med = float(np.nanmedian(x_train[col].to_numpy(dtype=np.float64)))
        medians[col] = 0.0 if np.isnan(med) else med

    # Step 5: Passing training runs mean and std (after median imputation)
    x_passing = x_train.loc[passing_mask, retained_columns].copy()
    passing_means: dict[str, float] = {}
    passing_stds: dict[str, float] = {}
    for col in retained_columns:
        vals = x_passing[col].fillna(medians[col]).to_numpy(dtype=np.float64)
        mu = float(np.mean(vals))
        sd = float(np.std(vals, ddof=1)) if len(vals) > 1 else 1.0
        if sd < 1e-9 or np.isnan(sd):
            sd = 1.0
        passing_means[col] = mu
        passing_stds[col] = sd

    return QualityTransformer(
        input_columns=cols,
        dropped_high_missing=dropped_high_missing,
        dropped_constant=dropped_constant,
        dropped_collinear=dropped_collinear,
        retained_columns=retained_columns,
        dropped_to_representative=dropped_to_rep,
        representative_to_members=rep_to_members,
        training_missing_fractions={c: missing_fracs[c] for c in retained_columns},
        medians=medians,
        passing_means=passing_means,
        passing_stds=passing_stds,
        max_missing_fraction=max_missing_fraction,
        max_constant_dominant_fraction=max_constant_dominant_fraction,
        collinearity_threshold=collinearity_threshold,
        n_train_runs=int(n_train),
        n_passing_train_runs=int(np.sum(passing_mask)),
    )
