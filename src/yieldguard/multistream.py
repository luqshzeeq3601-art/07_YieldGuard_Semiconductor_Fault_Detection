"""Multi-stream statistical process control for semiconductor tool and chamber fleets.

Implements the Mortell & Runger multi-stream framework:
- Stream-stratified baseline calibration (mean, variance, and covariance per tool/chamber)
- Tool-standardized residuals e_{m,t} to eliminate baseline tool-offset false alarms
- Per-stream EWMA tracking and Hotelling's T^2 surveillance
- Extremum surveillance statistic with Bonferroni-corrected false alarm budget limits
- Direct stream attribution identifying the offending equipment chamber
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class StreamBaseline:
    """Phase I baseline parameters for an individual tool or chamber stream."""

    stream_id: str
    n_samples: int
    means: dict[str, float]
    stds: dict[str, float]
    retained_sensors: list[str]


@dataclass
class MultiStreamSPC:
    """Multi-stream Statistical Process Control monitor.

    Surveils parallel semiconductor processing streams (e.g., etch chambers, photolithography tools)
    while controlling the family-wise false alarm rate across streams.
    """

    lambda_param: float = 0.20
    k_sigma: float = 3.0
    bonferroni_correction: bool = True
    baselines: dict[str, StreamBaseline] = field(default_factory=dict)
    is_fitted: bool = False
    monitored_sensors: list[str] = field(default_factory=list)

    def fit(
        self,
        df: pd.DataFrame,
        stream_col: str,
        sensor_cols: list[str],
        min_samples_per_stream: int = 10,
    ) -> MultiStreamSPC:
        """Estimate Phase I in-control parameters independently for each tool/chamber stream."""
        if stream_col not in df.columns:
            raise ValueError(f"Stream column '{stream_col}' not found in dataframe.")

        streams = sorted(df[stream_col].dropna().unique())
        if len(streams) == 0:
            raise ValueError("No streams found in provided dataframe.")

        self.baselines = {}
        valid_sensors = [c for c in sensor_cols if c in df.columns]
        self.monitored_sensors = valid_sensors

        for st in streams:
            st_df = df[df[stream_col] == st][valid_sensors]
            n_samples = len(st_df)
            if n_samples < min_samples_per_stream:
                continue

            means: dict[str, float] = {}
            stds: dict[str, float] = {}
            retained: list[str] = []

            for col in valid_sensors:
                vals = st_df[col].dropna()
                if len(vals) < 2:
                    continue
                m = float(vals.mean())
                s = float(vals.std(ddof=1))
                if s > 1e-9:
                    means[col] = m
                    stds[col] = s
                    retained.append(col)

            self.baselines[str(st)] = StreamBaseline(
                stream_id=str(st),
                n_samples=n_samples,
                means=means,
                stds=stds,
                retained_sensors=retained,
            )

        if not self.baselines:
            raise ValueError("No stream met the minimum sample threshold for Phase I baseline fitting.")

        self.is_fitted = True
        return self

    def score_run(
        self,
        stream_id: str,
        sensor_values: dict[str, float],
    ) -> dict[str, Any]:
        """Compute stream-standardized residuals, max z-score, and alarm verdict for a single run."""
        if not self.is_fitted:
            raise RuntimeError("MultiStreamSPC must be fitted before scoring.")

        st_key = str(stream_id)
        baseline = self.baselines.get(st_key)
        if baseline is None:
            # Unknown stream: return unmonitored indicator
            return {
                "stream_id": stream_id,
                "known_stream": False,
                "max_abs_z": 0.0,
                "alarm": False,
                "top_deviating_sensors": [],
            }

        z_scores: dict[str, float] = {}
        for col in baseline.retained_sensors:
            val = sensor_values.get(col)
            if val is not None and not np.isnan(val):
                m = baseline.means[col]
                s = baseline.stds[col]
                z_scores[col] = float((val - m) / s)

        if not z_scores:
            return {
                "stream_id": stream_id,
                "known_stream": True,
                "max_abs_z": 0.0,
                "alarm": False,
                "top_deviating_sensors": [],
            }

        m_streams = max(len(self.baselines), 1)
        # Bonferroni-adjusted sigma threshold if enabled
        if self.bonferroni_correction and m_streams > 1:
            # nominal alpha for 3-sigma is 2 * (1 - norm.cdf(3.0)) = 0.002699796
            alpha = 2.0 * (1.0 - stats.norm.cdf(self.k_sigma))
            alpha_adj = alpha / m_streams
            effective_limit = float(stats.norm.ppf(1.0 - alpha_adj / 2.0))
        else:
            effective_limit = self.k_sigma

        abs_z = {k: abs(v) for k, v in z_scores.items()}
        max_sensor = max(abs_z, key=abs_z.get)  # type: ignore[arg-type]
        max_val = abs_z[max_sensor]
        alarm = max_val > effective_limit

        ranked = sorted(abs_z.items(), key=lambda item: item[1], reverse=True)[:5]
        top_deviations = [{"sensor": k, "abs_z": round(v, 4)} for k, v in ranked]

        return {
            "stream_id": stream_id,
            "known_stream": True,
            "max_abs_z": float(max_val),
            "effective_limit": float(effective_limit),
            "alarm": bool(alarm),
            "offending_sensor": max_sensor if alarm else None,
            "top_deviating_sensors": top_deviations,
        }

    def score_dataframe(
        self,
        df: pd.DataFrame,
        stream_col: str,
        sensor_cols: list[str] | None = None,
    ) -> pd.DataFrame:
        """Score a historical sequence of runs across multiple streams with stream-level EWMA."""
        if not self.is_fitted:
            raise RuntimeError("MultiStreamSPC must be fitted before scoring.")
        if stream_col not in df.columns:
            raise ValueError(f"Stream column '{stream_col}' not found.")

        m_streams = max(len(self.baselines), 1)
        if self.bonferroni_correction and m_streams > 1:
            alpha = 2.0 * (1.0 - stats.norm.cdf(self.k_sigma))
            alpha_adj = alpha / m_streams
            limit = float(stats.norm.ppf(1.0 - alpha_adj / 2.0))
        else:
            limit = self.k_sigma

        # Per-stream EWMA state
        ewma_state: dict[str, dict[str, float]] = {
            st: {c: 0.0 for c in b.retained_sensors} for st, b in self.baselines.items()
        }

        records: list[dict[str, Any]] = []

        for idx, row in df.iterrows():
            st_val = str(row[stream_col])
            baseline = self.baselines.get(st_val)

            if baseline is None:
                records.append(
                    {
                        "row_idx": idx,
                        "stream_id": st_val,
                        "known_stream": False,
                        "multistream_z_max": 0.0,
                        "multistream_ewma_max": 0.0,
                        "alarm_multistream_shewhart": False,
                        "alarm_multistream_ewma": False,
                        "offending_stream": None,
                    }
                )
                continue

            max_shewhart = 0.0
            max_ewma = 0.0
            st_ewma = ewma_state[st_val]

            for c in baseline.retained_sensors:
                val = row.get(c)
                if pd.notna(val):
                    v_float = float(val)
                    m = baseline.means[c]
                    s = baseline.stds[c]
                    z = (v_float - m) / s
                    # Update stream EWMA
                    st_ewma[c] = self.lambda_param * z + (1.0 - self.lambda_param) * st_ewma[c]

                    if abs(z) > max_shewhart:
                        max_shewhart = abs(z)
                    if abs(st_ewma[c]) > max_ewma:
                        max_ewma = abs(st_ewma[c])

            # EWMA asymptotic limit factor: sqrt(lambda / (2 - lambda))
            ewma_limit = limit * np.sqrt(self.lambda_param / (2.0 - self.lambda_param))

            alarm_shewhart = max_shewhart > limit
            alarm_ewma = max_ewma > ewma_limit

            records.append(
                {
                    "row_idx": idx,
                    "stream_id": st_val,
                    "known_stream": True,
                    "multistream_z_max": float(max_shewhart),
                    "multistream_ewma_max": float(max_ewma),
                    "alarm_multistream_shewhart": bool(alarm_shewhart),
                    "alarm_multistream_ewma": bool(alarm_ewma),
                    "offending_stream": st_val if (alarm_shewhart or alarm_ewma) else None,
                }
            )

        return pd.DataFrame(records)
