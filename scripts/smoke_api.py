"""Exercise an actual YieldGuard API service with synthetic runs, readiness, and scoring."""

import argparse
import json
from urllib.request import Request, urlopen


def smoke(base_url: str) -> None:
    def request(path: str, payload: dict | None = None) -> dict:
        body = json.dumps(payload).encode() if payload is not None else None
        req = Request(
            base_url.rstrip("/") + path,
            data=body,
            headers={"Content-Type": "application/json"},
        )
        with urlopen(req, timeout=30) as response:
            assert response.status == 200, f"Expected 200 from {path}, got {response.status}"
            return json.load(response)

    # 1. Health check
    health = request("/health")
    assert health["status"] == "healthy", f"Health status unexpected: {health}"

    # 2. Ready check
    ready = request("/ready")
    assert ready.get("bundle_loaded") is True, f"Model bundle not loaded: {ready}"
    assert ready["bundle_version"] == "v0.1.0", f"Unexpected bundle version: {ready}"

    # 3. Model info check
    info = request("/model-info")
    assert info["bundle_version"] == "v0.1.0"
    assert info["primary_detector"] == "E02"

    # 4. Score request with 590 sensor floats
    synthetic_sensors = [float(i * 0.1) for i in range(590)]
    score_payload = {
        "run_id": "syn_smoke_001",
        "timestamp": "2026-10-11T12:00:00",
        "sensors": synthetic_sensors,
        "explain": True,
        "top_k": 5,
    }
    score_res = request("/score", score_payload)
    assert score_res["run_id"] == "syn_smoke_001"
    assert score_res["bundle_version"] == "v0.1.0"
    assert score_res["primary_detector"] == "E02"
    assert isinstance(score_res["alarm"], bool)
    assert "e02_score" in score_res["scores"]
    assert "t2" in score_res["scores"]
    assert "xgb_score" in score_res["scores"]
    assert "alarm_e02_b5" in score_res["alarms"]

    # 5. SPC endpoint check
    spc_run1 = {"run_id": "spc_001", "sensors": [float(i * 0.1) for i in range(590)]}
    spc_run2 = {"run_id": "spc_002", "sensors": [float(i * 0.12) for i in range(590)]}
    spc_payload = {"runs": [spc_run1, spc_run2]}
    spc_res = request("/spc", spc_payload)
    assert len(spc_res["monitored_sensors"]) > 0
    assert "charts" in spc_res
    assert len(spc_res["e02_scores"]) == 2

    output_summary = {
        "status": "smoke_passed",
        "health": health,
        "ready": ready,
        "model_info": info,
        "score_sample": {
            "run_id": score_res["run_id"],
            "alarm": score_res["alarm"],
            "e02_score": score_res["scores"]["e02_score"],
            "primary_detector": score_res["primary_detector"],
        },
        "spc_sample": {
            "monitored_sensors": spc_res["monitored_sensors"],
            "n_runs_evaluated": len(spc_res["e02_scores"]),
        },
    }
    print(json.dumps(output_summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Smoke test YieldGuard API service.")
    parser.add_argument("--url", default="http://127.0.0.1:8000", help="Base URL of YieldGuard API")
    args = parser.parse_args()
    smoke(args.url)
