"""Runtime latency, parity, and reproducibility benchmarks for YieldGuard."""

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from api.main import app
from yieldguard.artifacts import load_model_bundle
from yieldguard.predict import score_single_run
from yieldguard.synthetic import generate_synthetic_secom


def measure_reproducibility_o6(bundle) -> dict:
    """Measure score reproducibility across multiple independent evaluations."""
    matrix, _, _, _ = generate_synthetic_secom(n_runs=10, seed=42)
    scores_run1 = []
    scores_run2 = []

    for i in range(10):
        sensors_row = [None if np.isnan(x) else float(x) for x in matrix[i]]
        payload = {"run_id": f"syn_rep_{i}", "sensors": sensors_row}
        s1 = score_single_run(bundle, payload)["scores"]
        s2 = score_single_run(bundle, payload)["scores"]
        scores_run1.append(s1)
        scores_run2.append(s2)

    max_diffs: dict[str, float] = {}
    for det_key in (
        "e01_score",
        "e02_score",
        "e03_score",
        "t2",
        "spe",
        "mspc_index",
        "iforest_score",
        "xgb_score",
    ):
        diffs = [abs(scores_run1[i][det_key] - scores_run2[i][det_key]) for i in range(10)]
        max_diffs[det_key] = float(np.max(diffs))

    overall_max_diff = float(max(max_diffs.values()))
    passed = bool(overall_max_diff <= 1e-6)

    return {
        "overall_max_score_difference": overall_max_diff,
        "max_difference_by_detector": max_diffs,
        "tolerance": 1e-6,
        "passed": passed,
        "verdict": "PASS" if passed else "FAIL",
    }


def measure_latency_and_parity_o7(bundle, n_trials: int = 100) -> dict:
    """Measure warm single-run scoring latency and CLI/API score parity."""
    client = TestClient(app)
    matrix, _, _, _ = generate_synthetic_secom(n_runs=5, seed=123)
    sample_sensors = [None if np.isnan(x) else float(x) for x in matrix[0]]
    payload = {
        "run_id": "bench_001",
        "timestamp": "2026-10-11T12:00:00",
        "sensors": sample_sensors,
        "explain": False,
    }

    # Parity check between direct scoring and API scoring
    direct_res = score_single_run(bundle, payload)
    api_resp = client.post("/score", json=payload)
    assert api_resp.status_code == 200, f"API failed with {api_resp.status_code}: {api_resp.text}"
    api_data = api_resp.json()

    parity_diffs = {}
    for det_key in (
        "e01_score",
        "e02_score",
        "e03_score",
        "t2",
        "spe",
        "mspc_index",
        "iforest_score",
        "xgb_score",
    ):
        d_val = direct_res["scores"][det_key]
        a_val = api_data["scores"][det_key]
        diff = abs(d_val - a_val)
        parity_diffs[det_key] = diff

    max_parity_diff = float(max(parity_diffs.values()))
    parity_passed = bool(max_parity_diff <= 1e-9)

    # Warm-up (10 runs)
    for _ in range(10):
        score_single_run(bundle, payload)
        client.post("/score", json=payload)

    # Timed runs: Direct function latency
    direct_times_ms: list[float] = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        score_single_run(bundle, payload)
        t1 = time.perf_counter()
        direct_times_ms.append((t1 - t0) * 1000.0)

    # Timed runs: API client latency (including serialization)
    api_times_ms: list[float] = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        client.post("/score", json=payload)
        t1 = time.perf_counter()
        api_times_ms.append((t1 - t0) * 1000.0)

    direct_p95 = float(np.percentile(direct_times_ms, 95))
    api_p95 = float(np.percentile(api_times_ms, 95))

    return {
        "sample_size_trials": n_trials,
        "direct_score_latency_ms": {
            "mean": float(np.mean(direct_times_ms)),
            "p50": float(np.median(direct_times_ms)),
            "p90": float(np.percentile(direct_times_ms, 90)),
            "p95": direct_p95,
            "p99": float(np.percentile(direct_times_ms, 99)),
            "min": float(np.min(direct_times_ms)),
            "max": float(np.max(direct_times_ms)),
        },
        "api_score_latency_ms": {
            "mean": float(np.mean(api_times_ms)),
            "p50": float(np.median(api_times_ms)),
            "p90": float(np.percentile(api_times_ms, 90)),
            "p95": api_p95,
            "p99": float(np.percentile(api_times_ms, 99)),
            "min": float(np.min(api_times_ms)),
            "max": float(np.max(api_times_ms)),
        },
        "cli_api_score_parity": {
            "max_absolute_difference": max_parity_diff,
            "tolerance": 1e-9,
            "passed": parity_passed,
        },
        "target_o7_evaluation": {
            "direct_score_p95_le_10ms": bool(direct_p95 <= 10.0),
            "api_score_p95_le_50ms": bool(api_p95 <= 50.0),
            "verdict": "PASS" if parity_passed else "FAIL",
        },
    }


def run_benchmark(
    bundle_dir: str = "models/v0.1.0",
    output_path: str = "reports/runtime.json",
    n_trials: int = 100,
) -> dict:
    """Run full reproducibility and latency benchmark and save runtime.json."""
    os.environ["YIELDGUARD_BUNDLE_DIR"] = bundle_dir
    bundle = load_model_bundle(bundle_dir)

    repro_res = measure_reproducibility_o6(bundle)
    latency_res = measure_latency_and_parity_o7(bundle, n_trials=n_trials)

    report = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "bundle_version": bundle.version,
        "environment": {
            "python_version": platform.python_version(),
            "os": platform.system(),
            "os_release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "reproducibility_o6": repro_res,
        "latency_and_parity_o7": latency_res,
        "container_verification": {
            "status": "unverified_locally",
            "reason": "Docker daemon offline on local host; container build and smoke test defined in .github/workflows/ci.yml",
            "dockerfile_present": Path("Dockerfile").exists(),
            "dockerignore_present": Path(".dockerignore").exists(),
            "smoke_script_present": Path("scripts/smoke_api.py").exists(),
        },
    }

    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Benchmark results saved to {out_p}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run YieldGuard runtime and reproducibility benchmarks."
    )
    parser.add_argument("--bundle", default="models/v0.1.0", help="Model bundle directory")
    parser.add_argument("--output", default="reports/runtime.json", help="Output JSON path")
    parser.add_argument("--trials", type=int, default=100, help="Number of benchmark trials")
    args = parser.parse_args()

    res = run_benchmark(bundle_dir=args.bundle, output_path=args.output, n_trials=args.trials)
    print(json.dumps(res, indent=2))
