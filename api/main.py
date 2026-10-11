"""FastAPI application for YieldGuard online scoring and Phase II SPC monitoring."""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncGenerator

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from api.schemas import (
    HealthResponse,
    ModelInfoResponse,
    ReadyResponse,
    ScoreRunRequest,
    ScoreRunResponse,
    SpcHistoryRequest,
    SpcHistoryResponse,
)
from yieldguard.artifacts import LoadedBundle, load_model_bundle
from yieldguard.data import SENSOR_COLUMNS
from yieldguard.predict import score_single_run
from yieldguard.spc import score_univariate_spc

logger = logging.getLogger(__name__)

# Configurable bundle directory path
DEFAULT_BUNDLE_DIR = os.getenv("YIELDGUARD_BUNDLE_DIR", "models/v0.1.0")
bundle_instance: LoadedBundle | None = None


def get_loaded_bundle() -> LoadedBundle | None:
    global bundle_instance
    if bundle_instance is not None:
        return bundle_instance
    bundle_path = Path(DEFAULT_BUNDLE_DIR)
    if bundle_path.exists() and (bundle_path / "metadata.json").exists():
        try:
            bundle_instance = load_model_bundle(bundle_path)
            logger.info("Loaded model bundle from %s", bundle_path)
        except Exception as e:
            logger.warning("Failed to load bundle from %s: %s", bundle_path, e)
            bundle_instance = None
    return bundle_instance


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    get_loaded_bundle()
    yield


app = FastAPI(
    title="YieldGuard Semiconductor Fault Detection API",
    version="0.1.0",
    description="Statistical process control and machine learning fault detection for semiconductor manufacturing.",
    lifespan=lifespan,
)


@app.middleware("http")
async def check_payload_size(request: Request, call_next: Any) -> Response:
    content_length = request.headers.get("content-length")
    if content_length:
        length_bytes = int(content_length)
        # Cap 256 KiB for /score, 2 MiB for /spc
        if request.url.path == "/score" and length_bytes > 256 * 1024:
            return JSONResponse(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                content={"detail": "Request body exceeds maximum size of 256 KiB."},
            )
        if request.url.path == "/spc" and length_bytes > 2 * 1024 * 1024:
            return JSONResponse(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                content={"detail": "Request body exceeds maximum size of 2 MiB."},
            )
    return await call_next(request)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    clean_errors = [
        {
            "loc": list(err.get("loc", [])),
            "msg": str(err.get("msg", "")),
            "type": str(err.get("type", "")),
        }
        for err in exc.errors()
    ]
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": clean_errors},
    )


@app.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    bundle = get_loaded_bundle()
    return HealthResponse(
        status="healthy",
        bundle_version=bundle.version if bundle else None,
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
    )


@app.get("/ready", response_model=ReadyResponse)
def readiness_check() -> ReadyResponse:
    bundle = get_loaded_bundle()
    if bundle is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model bundle not loaded or unready.",
        )
    return ReadyResponse(
        status="ready",
        bundle_loaded=True,
        bundle_version=bundle.version,
    )


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
    bundle = get_loaded_bundle()
    if bundle is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model bundle not loaded.",
        )

    # Load targets outcomes if available
    holdout_p = Path("reports/holdout_metrics.json")
    outcomes = {}
    if holdout_p.exists():
        import json
        outcomes = json.loads(holdout_p.read_text(encoding="utf-8")).get("objectives", {})

    return ModelInfoResponse(
        bundle_version=bundle.version,
        dataset_id=bundle.metadata.get("dataset_id", "SECOM"),
        primary_detector=bundle.policy.get("primary_detector", "E02"),
        alarm_budget_per_100=5.0,
        thresholds=bundle.policy.get("thresholds", {}),
        data_scope="UCI SECOM (590 sensor channels, 1567 runs across 2008)",
        target_outcomes=outcomes,
        file_hashes=bundle.metadata.get("file_hashes", {}),
    )


@app.post("/score", response_model=ScoreRunResponse)
def score_run(payload: ScoreRunRequest) -> ScoreRunResponse:
    bundle = get_loaded_bundle()
    if bundle is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model bundle not loaded.",
        )

    res = score_single_run(
        bundle=bundle,
        run_dict={
            "run_id": payload.run_id,
            "timestamp": payload.timestamp,
            "sensors": payload.sensors,
        },
        explain=payload.explain,
        top_k=payload.top_k,
    )

    return ScoreRunResponse(
        run_id=res["run_id"],
        timestamp=res["timestamp"],
        scores=res["scores"],
        alarms=res["alarms"],
        primary_detector=res["primary_detector"],
        alarm=res["alarm"],
        alarm_budget_per_100=res["alarm_budget_per_100"],
        n_missing=res["n_missing"],
        quality_warnings=res["quality_warnings"],
        top_sensors=res["top_sensors"],
        bundle_version=bundle.version,
        policy_version="1.0",
        schema_version="1.0",
    )


@app.post("/spc", response_model=SpcHistoryResponse)
def spc_history(payload: SpcHistoryRequest) -> SpcHistoryResponse:
    bundle = get_loaded_bundle()
    if bundle is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model bundle not loaded.",
        )

    # Convert runs into DataFrame
    rows: list[dict[str, Any]] = []
    for r in payload.runs:
        row_dict = {col: val for col, val in zip(SENSOR_COLUMNS, r.sensors, strict=True)}
        row_dict["run_id"] = r.run_id
        if r.timestamp is not None:
            row_dict["timestamp"] = r.timestamp
        rows.append(row_dict)

    df_runs = pd.DataFrame(rows)
    retained_df = bundle.quality.transform_retained(df_runs[SENSOR_COLUMNS])

    spc_result = score_univariate_spc(bundle.spc_model, retained_df)

    # Convert NumPy arrays in charts to lists
    clean_charts: dict[str, Any] = {}
    for s_name, chart_data in spc_result["charts"].items():
        clean_entry: dict[str, Any] = {}
        for k, v in chart_data.items():
            if isinstance(v, np.ndarray):
                clean_entry[k] = v.tolist()
            elif isinstance(v, (np.floating, np.integer)):
                clean_entry[k] = float(v)
            else:
                clean_entry[k] = v
        clean_charts[s_name] = clean_entry

    return SpcHistoryResponse(
        monitored_sensors=bundle.spc_model.monitored_sensors,
        charts=clean_charts,
        e02_scores=[float(x) for x in spc_result["e02_scores"]],
        e03_scores=[float(x) for x in spc_result["e03_scores"]],
        bundle_version=bundle.version,
    )
