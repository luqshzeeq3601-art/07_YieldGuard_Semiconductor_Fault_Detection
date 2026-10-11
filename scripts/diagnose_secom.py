"""CLI script for Phase 2 exploratory SECOM diagnostics (per-block drift, oracle upper bound, E02 ties)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from yieldguard.artifacts import load_model_bundle
from yieldguard.config import load_config
from yieldguard.diagnostics import run_secom_phase2_diagnostics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Phase 2 exploratory diagnostics on SECOM.")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to YAML config.")
    parser.add_argument("--bundle", type=str, default="models/v0.1.0", help="Frozen v0.1.0 bundle directory.")
    parser.add_argument(
        "--output",
        type=str,
        default="reports/exploratory/secom_phase2_diagnostics.json",
        help="Output JSON path for exploratory diagnostics.",
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    bundle = load_model_bundle(args.bundle)

    runs_df = pd.read_parquet(cfg.data.runs_parquet)
    labels_dev = pd.read_parquet(cfg.data.labels_dev_parquet)
    labels_hold = pd.read_parquet(cfg.data.labels_holdout_parquet)
    labels_all = pd.concat(
        [labels_dev[["run_id", "label_fail"]], labels_hold[["run_id", "label_fail"]]],
        ignore_index=True,
    )

    thresholds_e02 = bundle.policy.get("thresholds", {}).get("E02", {})
    report = run_secom_phase2_diagnostics(
        runs_df=runs_df,
        labels_df=labels_all,
        retained_columns=bundle.quality.retained_columns,
        spc_model=bundle.spc_model,
        thresholds_e02=thresholds_e02,
        output_path=Path(args.output),
        n_bootstraps=cfg.policy.n_bootstraps,
        seed=cfg.project.seed,
    )

    oracle = report["step2_oracle_upper_bound"]
    ties = report["step3_e02_score_ties"]["holdout_block"]
    print(
        json.dumps(
            {
                "status": "completed",
                "evaluation_type": report["evaluation_type"],
                "output_path": args.output,
                "best_oracle_model": oracle["best_oracle_model"],
                "best_oracle_mean_fold_pr_auc": oracle["best_oracle_mean_fold_pr_auc"],
                "best_oracle_95_ci_pr_auc": oracle["best_oracle_95_ci_pr_auc"],
                "prevalence": oracle["prevalence"],
                "oracle_beats_chance": oracle["oracle_beats_chance"],
                "gate_decision": oracle["gate_decision"],
                "holdout_e02_unique_scores": ties.get("n_unique_e02_scores"),
                "holdout_e02_runs_in_ties": ties.get("n_runs_in_ties"),
                "holdout_runs_between_b10_and_b5": ties.get("runs_Strictly_between_b10_and_b5"),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
