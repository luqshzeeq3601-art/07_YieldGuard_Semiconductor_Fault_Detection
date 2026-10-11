"""Command-line interface for YieldGuard."""

from __future__ import annotations

import argparse
import json
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="yieldguard",
        description="YieldGuard semiconductor fault detection and SPC vs ML CLI.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # prepare
    p_prep = subparsers.add_parser("prepare", help="Validate raw SECOM, split chronologically, seal holdout labels.")
    p_prep.add_argument("--config", type=str, default="configs/default.yaml", help="Path to YAML config.")

    # train
    p_train = subparsers.add_parser("train", help="Fit SPC, MSPC, Isolation Forest, or XGBoost detectors.")
    p_train.add_argument("--config", type=str, default="configs/default.yaml", help="Path to YAML config.")
    p_train.add_argument(
        "--experiment",
        type=str,
        choices=["spc", "mspc", "anomaly", "xgboost", "all"],
        default="all",
        help="Detector family to fit and score across forward folds.",
    )

    # evaluate
    p_eval = subparsers.add_parser("evaluate", help="Evaluate development folds or frozen holdout.")
    p_eval.add_argument("--config", type=str, default="configs/default.yaml", help="Path to YAML config.")
    p_eval.add_argument("--split", type=str, choices=["dev", "holdout"], default="dev")
    p_eval.add_argument("--bundle", type=str, default="models/v0.1.0", help="Frozen bundle directory for holdout.")
    p_eval.add_argument("--release-id", type=str, default="v0.1.0", help="Release identifier.")
    p_eval.add_argument(
        "--allow-heldout-evaluation",
        action="store_true",
        help="Explicit authorization flag required to open labels_holdout.parquet once.",
    )

    # freeze
    p_freeze = subparsers.add_parser("freeze", help="Freeze staging models and policy into immutable version bundle.")
    p_freeze.add_argument("--config", type=str, default="configs/default.yaml", help="Path to YAML config.")
    p_freeze.add_argument("--version", type=str, default="v0.1.0", help="Target bundle version.")

    # score
    p_score = subparsers.add_parser("score", help="Batch score a CSV of runs with a frozen bundle.")
    p_score.add_argument("--bundle", type=str, default="models/v0.1.0", help="Frozen bundle directory.")
    p_score.add_argument("--input", type=str, required=True, help="Input CSV path.")
    p_score.add_argument("--output", type=str, required=True, help="Output scored CSV path.")

    # explain
    p_exp = subparsers.add_parser("explain", help="Generate sensor rankings for flagged runs or verify synthetic O5.")
    p_exp.add_argument("--config", type=str, default="configs/default.yaml", help="Path to YAML config.")
    p_exp.add_argument("--bundle", type=str, default=None, help="Frozen bundle directory (or staging if omitted).")
    p_exp.add_argument("--input", type=str, default=None, help="Input CSV of runs to explain.")
    p_exp.add_argument("--output", type=str, default="reports/explanations/sensor_rankings.json")
    p_exp.add_argument("--verify-synthetic", action="store_true", help="Run O5 synthetic ground-truth verification.")

    # drift
    p_drift = subparsers.add_parser("drift", help="Run offline Evidently sensor drift report against reference.")
    p_drift.add_argument("--bundle", type=str, default="models/v0.1.0", help="Frozen bundle directory.")
    p_drift.add_argument("--current", type=str, required=True, help="Current parquet or CSV path.")
    p_drift.add_argument("--split", type=str, default=None, help="Optional block filter (e.g. holdout).")
    p_drift.add_argument("--output", type=str, default="reports/drift", help="Output directory for HTML and JSON.")

    # drift-controls
    p_dctrl = subparsers.add_parser("drift-controls", help="Run three drift verification controls.")
    p_dctrl.add_argument("--bundle", type=str, default="models/v0.1.0", help="Frozen bundle directory.")
    p_dctrl.add_argument("--output", type=str, default="reports/drift/controls", help="Output directory.")

    args = parser.parse_args(argv)

    if args.command == "prepare":
        from yieldguard.splits import prepare_dataset

        manifest = prepare_dataset(args.config)
        print(
            json.dumps(
                {
                    "status": "prepared",
                    "manifest_sha256": manifest["manifest_sha256"],
                    "n_dev_runs": manifest["n_dev_runs"],
                    "n_holdout_runs": manifest["n_holdout_runs"],
                },
                indent=2,
            )
        )
        return 0

    if args.command == "train":
        from yieldguard.train import run_training_experiments

        res = run_training_experiments(args.config, experiment=args.experiment)
        print(json.dumps(res, indent=2))
        return 0

    if args.command == "evaluate":
        from yieldguard.evaluate import run_development_evaluation, run_holdout_evaluation

        if args.split == "dev":
            res = run_development_evaluation(args.config)
        else:
            res = run_holdout_evaluation(
                config_path=args.config,
                bundle_dir=args.bundle,
                release_id=args.release_id,
                allow_heldout_evaluation=args.allow_heldout_evaluation,
            )
        print(json.dumps({"status": "evaluated", "split": args.split, "primary_detector": res.get("primary_detector")}, indent=2))
        return 0

    if args.command == "freeze":
        from yieldguard.artifacts import freeze_model_bundle

        bundle_dir = freeze_model_bundle(args.config, version=args.version)
        print(json.dumps({"status": "frozen", "bundle_dir": str(bundle_dir)}, indent=2))
        return 0

    if args.command == "score":
        from yieldguard.predict import score_csv_file

        out_path = score_csv_file(args.bundle, args.input, args.output)
        print(json.dumps({"status": "scored", "output": str(out_path)}, indent=2))
        return 0

    if args.command == "explain":
        from yieldguard.explain import run_cli_explain

        res = run_cli_explain(
            config_path=args.config,
            bundle_dir=args.bundle,
            input_csv=args.input,
            output_path=args.output,
            verify_synthetic=args.verify_synthetic,
        )
        print(json.dumps(res, indent=2))
        return 0

    if args.command == "drift":
        from yieldguard.monitoring import run_cli_drift

        res = run_cli_drift(args.bundle, args.current, args.output, split_filter=args.split)
        print(json.dumps(res, indent=2))
        return 0

    if args.command == "drift-controls":
        from yieldguard.monitoring import run_cli_drift_controls

        res = run_cli_drift_controls(args.bundle, args.output)
        print(json.dumps(res, indent=2))
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
