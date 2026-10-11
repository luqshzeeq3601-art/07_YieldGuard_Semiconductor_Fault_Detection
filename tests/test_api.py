"""Unit and integration tests for YieldGuard FastAPI endpoints."""

from typing import Any
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.main import app
from yieldguard.artifacts import load_model_bundle
from yieldguard.data import SENSOR_COLUMNS
from yieldguard.predict import score_runs_dataframe
from yieldguard.spc import score_univariate_spc
from yieldguard.synthetic import generate_synthetic_secom


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture(scope="module")
def sample_sensors() -> list[float | None]:
    matrix, _, _, _ = generate_synthetic_secom(n_runs=1, seed=999)
    return [None if np.isnan(x) else float(x) for x in matrix[0]]


def test_health_endpoint(client: TestClient) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert "timestamp_utc" in data


def test_ready_and_model_info_endpoints(client: TestClient) -> None:
    ready_resp = client.get("/ready")
    assert ready_resp.status_code == 200
    ready_data = ready_resp.json()
    assert ready_data["status"] == "ready"
    assert ready_data["bundle_loaded"] is True

    info_resp = client.get("/model-info")
    assert info_resp.status_code == 200
    info_data = info_resp.json()
    assert info_data["dataset_id"] == "SECOM"
    assert info_data["primary_detector"] == "E02"
    assert "thresholds" in info_data


def test_valid_score_request(client: TestClient, sample_sensors: list[float]) -> None:
    payload = {
        "run_id": "test_run_001",
        "timestamp": "2026-10-11 12:00:00",
        "sensors": sample_sensors,
        "explain": True,
        "top_k": 3,
    }
    resp = client.post("/score", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["run_id"] == "test_run_001"
    assert data["timestamp"] == "2026-10-11 12:00:00"
    assert "e02_score" in data["scores"]
    assert "alarm" in data
    assert "top_sensors" in data and data["top_sensors"] is not None


def test_score_request_with_nulls(client: TestClient, sample_sensors: list[float]) -> None:
    sensors_with_nulls = list(sample_sensors)
    sensors_with_nulls[10] = None
    sensors_with_nulls[20] = None

    payload = {
        "run_id": "run_with_nulls",
        "sensors": sensors_with_nulls,
    }
    resp = client.post("/score", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["n_missing"] >= 2


def test_score_request_invalid_sensor_counts(client: TestClient, sample_sensors: list[float]) -> None:
    # 589 values (too few)
    resp_589 = client.post("/score", json={"run_id": "r1", "sensors": sample_sensors[:589]})
    assert resp_589.status_code == 422

    # 591 values (too many)
    resp_591 = client.post("/score", json={"run_id": "r2", "sensors": sample_sensors + [0.0]})
    assert resp_591.status_code == 422


def test_score_request_string_sensor_value(client: TestClient, sample_sensors: list[float]) -> None:
    bad_sensors = list(sample_sensors)
    bad_sensors[5] = "not_a_number"  # type: ignore[assignment]
    resp = client.post("/score", json={"run_id": "r_str", "sensors": bad_sensors})
    assert resp.status_code == 422


def test_score_request_infinite_sensor_value(client: TestClient, sample_sensors: list[float | None]) -> None:
    import json
    clean_sensors: list[Any] = [0.0 if s is None else s for s in sample_sensors]
    clean_sensors[5] = float("inf")
    body_str = json.dumps({"run_id": "r_inf", "sensors": clean_sensors}, allow_nan=True)
    resp = client.post("/score", content=body_str, headers={"content-type": "application/json"})
    assert resp.status_code == 422


def test_score_request_unknown_field_rejected(client: TestClient, sample_sensors: list[float]) -> None:
    payload = {
        "run_id": "r_extra",
        "sensors": sample_sensors,
        "unexpected_field": "disallowed",
    }
    resp = client.post("/score", json=payload)
    assert resp.status_code == 422


def test_score_request_oversized_body(client: TestClient, sample_sensors: list[float]) -> None:
    # Create body exceeding 256 KiB
    huge_run_id = "X" * (300 * 1024)
    payload = {"run_id": huge_run_id, "sensors": sample_sensors}
    resp = client.post("/score", json=payload)
    assert resp.status_code == 413


def test_missing_bundle_returns_503(client: TestClient, sample_sensors: list[float]) -> None:
    with patch("api.main.get_loaded_bundle", return_value=None):
        ready_resp = client.get("/ready")
        assert ready_resp.status_code == 503

        score_resp = client.post("/score", json={"run_id": "r1", "sensors": sample_sensors})
        assert score_resp.status_code == 503


def test_score_api_matches_cli_prediction_within_tolerance(client: TestClient, sample_sensors: list[float]) -> None:
    bundle = load_model_bundle("models/v0.1.0")

    # API score
    payload = {"run_id": "r_compare", "sensors": sample_sensors}
    api_resp = client.post("/score", json=payload)
    assert api_resp.status_code == 200
    api_data = api_resp.json()

    # CLI package score
    row_dict = {col: val for col, val in zip(SENSOR_COLUMNS, sample_sensors)}
    row_dict["run_id"] = "r_compare"
    df_single = pd.DataFrame([row_dict])
    cli_scored = score_runs_dataframe(bundle, df_single).iloc[0]

    # Verify scores equal within 1e-9
    for score_col in ["e01_score", "e02_score", "e03_score", "t2", "spe", "mspc_index", "iforest_score", "xgb_score"]:
        assert abs(api_data["scores"][score_col] - cli_scored[score_col]) < 1e-9

    assert api_data["alarm"] == bool(cli_scored["alarm"])
    assert api_data["primary_detector"] == cli_scored["primary_detector"]


def test_spc_endpoint_matches_spc_package(client: TestClient) -> None:
    matrix, _, timestamps, _ = generate_synthetic_secom(n_runs=5, seed=777)
    runs_payload = [
        {
            "run_id": f"run_{i}",
            "timestamp": timestamps[i],
            "sensors": [None if np.isnan(x) else float(x) for x in matrix[i]],
        }
        for i in range(5)
    ]

    spc_resp = client.post("/spc", json={"runs": runs_payload})
    assert spc_resp.status_code == 200
    spc_data = spc_resp.json()

    # Compare against direct score_univariate_spc call
    bundle = load_model_bundle("models/v0.1.0")
    df_runs = pd.DataFrame(matrix, columns=SENSOR_COLUMNS)
    retained_df = bundle.quality.transform_retained(df_runs)
    direct_spc = score_univariate_spc(bundle.spc_model, retained_df)

    for i in range(5):
        assert abs(spc_data["e02_scores"][i] - direct_spc["e02_scores"][i]) < 1e-9
        assert abs(spc_data["e03_scores"][i] - direct_spc["e03_scores"][i]) < 1e-9
