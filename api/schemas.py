"""Pydantic request and response schemas for YieldGuard REST API."""

from __future__ import annotations

import math
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class HealthResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: str = "healthy"
    bundle_version: str | None = None
    timestamp_utc: str


class ReadyResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: str = "ready"
    bundle_loaded: bool = True
    bundle_version: str


class ModelInfoResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    bundle_version: str
    dataset_id: str
    primary_detector: str
    alarm_budget_per_100: float
    thresholds: dict[str, Any]
    data_scope: str
    target_outcomes: dict[str, Any]
    file_hashes: dict[str, str]


class ScoreRunRequest(BaseModel):
    """Single run prediction request with 590 raw sensor readings or nulls."""

    model_config = ConfigDict(extra="forbid")

    run_id: str = Field(..., min_length=1, max_length=64, description="Unique run identifier (echoed)")
    timestamp: str | None = Field(default=None, description="Run timestamp (echoed, not a predictor)")
    sensors: list[float | None] = Field(..., description="Array of exactly 590 numbers or nulls")
    explain: bool = Field(default=False, description="Whether to compute sensor rankings for flagged runs")
    top_k: int = Field(default=5, ge=1, le=20, description="Top-k sensors to return if explain=True")

    @field_validator("sensors")
    @classmethod
    def validate_sensors(cls, v: list[float | None]) -> list[float | None]:
        if len(v) != 590:
            raise ValueError(f"sensors array must contain exactly 590 elements, got {len(v)}.")
        for idx, val in enumerate(v):
            if val is not None:
                if math.isinf(val):
                    raise ValueError(f"Infinite values are not allowed (found at index {idx}).")
                if math.isnan(val):
                    # Canonicalize NaN to None for consistent null handling
                    v[idx] = None
        return v


class ScoreRunResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    timestamp: str | None = None
    scores: dict[str, float]
    alarms: dict[str, bool]
    primary_detector: str
    alarm: bool
    alarm_budget_per_100: float
    n_missing: int
    quality_warnings: list[str]
    top_sensors: dict[str, Any] | None = None
    bundle_version: str
    policy_version: str = "1.0"
    schema_version: str = "1.0"


class SpcHistoryRun(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str = Field(..., min_length=1, max_length=64)
    timestamp: str | None = None
    sensors: list[float | None] = Field(..., description="Array of exactly 590 numbers or nulls")

    @field_validator("sensors")
    @classmethod
    def validate_sensors(cls, v: list[float | None]) -> list[float | None]:
        if len(v) != 590:
            raise ValueError(f"sensors array must contain exactly 590 elements, got {len(v)}.")
        for idx, val in enumerate(v):
            if val is not None:
                if math.isinf(val):
                    raise ValueError(f"Infinite values are not allowed (found at index {idx}).")
                if math.isnan(val):
                    v[idx] = None
        return v


class SpcHistoryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    runs: list[SpcHistoryRun] = Field(..., min_length=1, max_length=200, description="Ordered batch of 1 to 200 runs")


class SpcHistoryResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    monitored_sensors: list[str]
    charts: dict[str, Any]
    e02_scores: list[float]
    e03_scores: list[float]
    bundle_version: str
