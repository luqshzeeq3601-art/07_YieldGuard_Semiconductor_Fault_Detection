"""CLI script measuring the predeclared v0.2 objectives O5_v2 and O6_v2 on the Wafer Etch run.

Reruns the prequential evaluation with the frozen arguments into a temporary directory and
compares it with the stored one-shot confirmatory report, which is never overwritten.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from yieldguard.prequential import run_prequential_evaluation
from yieldguard.wafer_data import load_wafer_runs

O5_RANGE_PER_100 = (4.5, 5.5)
O6_TOLERANCE = 1e-6
# Fields added after the confirmatory run or expected to differ between runs.
IGNORED_KEYS = {"generated_at_utc", "calibration_alarms_b5", "calibration_alarms_per_100_b5"}


def max_numeric_difference(stored: Any, rerun: Any, path: str = "") -> tuple[float, list[str]]:
    """Return the largest absolute numeric difference and any structural mismatches."""
    if isinstance(stored, dict) and isinstance(rerun, dict):
        worst, mismatches = 0.0, []
        for key in sorted((set(stored) | set(rerun)) - IGNORED_KEYS):
            child = f"{path}.{key}" if path else key
            if key not in stored or key not in rerun:
                mismatches.append(f"missing key: {child}")
                continue
            diff, sub = max_numeric_difference(stored[key], rerun[key], child)
            worst, mismatches = max(worst, diff), mismatches + sub
        return worst, mismatches
    if isinstance(stored, list) and isinstance(rerun, list):
        if len(stored) != len(rerun):
            return 0.0, [f"length differs: {path}"]
        worst, mismatches = 0.0, []
        for i, (a, b) in enumerate(zip(stored, rerun)):
            diff, sub = max_numeric_difference(a, b, f"{path}[{i}]")
            worst, mismatches = max(worst, diff), mismatches + sub
        return worst, mismatches
    numeric = (int, float)
    if isinstance(stored, numeric) and isinstance(rerun, numeric) and not isinstance(stored, bool):
        return abs(float(stored) - float(rerun)), []
    return 0.0, ([] if stored == rerun else [f"value differs: {path}"])


def calibration_alarm_rates(report: dict[str, Any]) -> dict[str, dict[str, float]]:
    """Pool budget-5 calibration-window alarms per detector across evaluated blocks."""
    blocks = report["per_block_summaries"]
    out: dict[str, dict[str, float]] = {}
    for det in report["detectors"]:
        rates = [b["detectors"][det]["calibration_alarms_per_100_b5"] for b in blocks]
        alarms = sum(b["detectors"][det]["calibration_alarms_b5"] for b in blocks)
        n_cal = sum(b["n_calibration_runs"] for b in blocks)
        out[det] = {
            "pooled_alarms_per_100": 100.0 * alarms / max(n_cal, 1),
            "min_block_alarms_per_100": min(rates),
            "max_block_alarms_per_100": max(rates),
        }
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Measure v0.2 objectives O5_v2 and O6_v2 on Wafer Etch.")
    parser.add_argument("--stored", type=str, default="reports/v0_2/wafer_confirmatory_metrics.json")
    parser.add_argument("--output", type=str, default="reports/v0_2/wafer_o5_o6_verification.json")
    args = parser.parse_args(argv)

    stored = json.loads(Path(args.stored).read_text(encoding="utf-8"))
    summary = stored["prequential_summary"]
    df, sensor_cols = load_wafer_runs()

    with tempfile.TemporaryDirectory() as tmp:
        rerun = run_prequential_evaluation(
            df=df,
            sensor_cols=sensor_cols,
            dataset_id=stored["dataset_id"],
            evaluation_type=stored["evaluation_type"],
            n_blocks=int(summary["n_total_blocks"]),
            rolling_window_runs=int(summary["rolling_window_runs"]),
            calibration_window_runs=int(summary["calibration_window_runs"]),
            seed=int(stored["seed"]),
            predictions_csv_path=Path(tmp) / "predictions.csv",
        )
    # Round-trip through JSON so both sides hold the same types.
    rerun = json.loads(json.dumps(rerun))

    primary = stored["selection"]["primary_detector"]
    rates = calibration_alarm_rates(rerun)
    lo, hi = O5_RANGE_PER_100
    o5_pass = lo <= rates[primary]["pooled_alarms_per_100"] <= hi

    max_diff, mismatches = max_numeric_difference(stored, rerun)
    o6_pass = max_diff <= O6_TOLERANCE and not mismatches

    result = {
        "protocol_version": "0.2.0",
        "dataset_id": stored["dataset_id"],
        "evaluation_type": "post_confirmatory_verification",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "compared_against": args.stored,
        "O5_v2_calibration_fidelity": {
            "verdict": "PASS" if o5_pass else "MISS",
            "judged_on": "pooled calibration-window alarm rate of the primary detector at budget 5",
            "primary_detector": primary,
            "allowed_range_per_100": [lo, hi],
            "detectors": rates,
        },
        "O6_v2_reproducibility": {
            "verdict": "PASS" if o6_pass else "MISS",
            "seed": int(stored["seed"]),
            "tolerance": O6_TOLERANCE,
            "max_abs_metric_difference": max_diff,
            "structural_mismatches": mismatches,
            "stored_predictions_sha256": stored["predictions_sha256"],
            "rerun_predictions_sha256": rerun["predictions_sha256"],
            "predictions_sha256_match": stored["predictions_sha256"] == rerun["predictions_sha256"],
        },
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
