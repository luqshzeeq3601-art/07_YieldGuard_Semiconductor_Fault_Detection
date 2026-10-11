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

    # prequential (v0.2 protocol)
    p_preq = subparsers.add_parser(
        "prequential",
        help="Run v0.2 prequential (rolling-origin) drift-adaptive evaluation.",
    )
    p_preq.add_argument("--config", type=str, default="configs/v0_2.yaml", help="Path to v0.2 YAML config.")
    p_preq.add_argument(
        "--dataset",
        type=str,
        choices=["secom", "wafer"],
        required=True,
        help="Dataset to evaluate (secom=exploratory, wafer=confirmatory).",
    )
    p_preq.add_argument("--output", type=str, required=True, help="Output JSON metrics report path.")
    p_preq.add_argument("--predictions", type=str, default=None, help="Optional output CSV path for scored runs.")

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

    if args.command == "prequential":
        from pathlib import Path

        import pandas as pd
        import yaml

        from yieldguard.prequential import run_prequential_evaluation

        cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
        preq_cfg = cfg.get("prequential", {})
        ds_cfg = preq_cfg.get(args.dataset, {})

        if args.dataset == "secom":
            import numpy as np

            runs_df = pd.read_parquet("data/processed/runs.parquet")
            dev_labels = pd.read_parquet("data/processed/labels_dev.parquet")
            holdout_labels = pd.read_parquet("data/processed/labels_holdout.parquet")
            all_labels = pd.concat([dev_labels, holdout_labels], ignore_index=True)
            df = runs_df.merge(all_labels[["run_id", "label_fail"]], on="run_id", how="inner")
            df = df.sort_values(["timestamp", "run_id"]).reset_index(drop=True)
            df["label"] = np.where(df["label_fail"].to_numpy(dtype=np.int64) == 1, 1, -1)
            sensor_cols = [c for c in df.columns if c.startswith("f") and c[1:].isdigit()]
            dataset_id = "SECOM"
            eval_type = ds_cfg.get("evaluation_type", "exploratory_post_holdout")
            n_blocks = int(ds_cfg.get("n_blocks", 8))
            win_runs = int(ds_cfg.get("rolling_window_runs", 470))
            cal_runs = int(ds_cfg.get("calibration_window_runs", 235))
        else:
            from yieldguard.wafer_data import load_wafer_runs

            df, sensor_cols = load_wafer_runs()
            dataset_id = "WAFER_ETCH"
            eval_type = ds_cfg.get("evaluation_type", "confirmatory_one_shot")
            n_blocks = int(ds_cfg.get("n_blocks", 8))
            win_runs = int(ds_cfg.get("rolling_window_runs", 1800))
            cal_runs = int(ds_cfg.get("calibration_window_runs", 895))

        pred_csv = args.predictions or f"reports/v0_2/{args.dataset}_prequential_predictions.csv"
        rep = run_prequential_evaluation(
            df=df,
            sensor_cols=sensor_cols,
            dataset_id=dataset_id,
            evaluation_type=eval_type,
            n_blocks=n_blocks,
            rolling_window_runs=win_runs,
            calibration_window_runs=cal_runs,
            predictions_csv_path=pred_csv,
            output_json_path=args.output,
            overwrite=(args.dataset == "secom"),
        )
        print(
            json.dumps(
                {
                    "status": "completed",
                    "dataset_id": dataset_id,
                    "evaluation_type": eval_type,
                    "primary_detector": rep["selection"]["primary_detector"],
                    "best_overall_detector": rep["selection"]["best_overall_detector"],
                    "target_outcomes_v0_2": rep["target_outcomes_v0_2"],
                    "output": args.output,
                },
                indent=2,
            )
        )
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
