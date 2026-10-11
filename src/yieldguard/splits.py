"""Chronological split by timestamp, tie-boundary protection, and sealed holdout label isolation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from yieldguard.config import load_config
from yieldguard.data import SENSOR_COLUMNS, load_raw_secom, write_raw_quality_report

DEV_BLOCKS = ["D1", "D2", "D3", "D4", "D5"]
FORWARD_FOLDS = [
    {"fold_id": "fold_1", "train_blocks": ["D1", "D2"], "eval_block": "D3"},
    {"fold_id": "fold_2", "train_blocks": ["D1", "D2", "D3"], "eval_block": "D4"},
    {"fold_id": "fold_3", "train_blocks": ["D1", "D2", "D3", "D4"], "eval_block": "D5"},
]


def compute_chronological_split(
    runs_df: pd.DataFrame,
    dev_fraction: float = 0.75,
    n_dev_blocks: int = 5,
) -> tuple[list[str], dict[str, Any]]:
    """Compute contiguous chronological blocks D1..D5 and holdout from timestamps and run_id only.

    Rules:
    - Order by `(timestamp, run_id)` ascending (ties keep file order).
    - Development is the first `dev_fraction` (75% = 1,175 of 1,567 runs) cut into 5 equal blocks.
    - Holdout is the remaining 25% (392 runs).
    - A boundary never separates runs with identical timestamps: move it later until the
      timestamp changes, and record the shift.
    - Labels are never read or used.
    """
    if "run_id" not in runs_df.columns or "timestamp" not in runs_df.columns:
        raise ValueError("runs_df must contain 'run_id' and 'timestamp' columns.")

    # Extract only run_id and timestamp so no label column can ever influence the split
    meta = runs_df[["run_id", "timestamp"]].copy()
    meta["timestamp"] = pd.to_datetime(meta["timestamp"])

    # Verify ordering by (timestamp, run_id)
    sorted_meta = meta.sort_values(by=["timestamp", "run_id"], kind="mergesort").reset_index(
        drop=True
    )
    if not (
        (meta["run_id"].values == sorted_meta["run_id"].values).all()
        and (meta["timestamp"].values == sorted_meta["timestamp"].values).all()
    ):
        raise ValueError("runs_df must be ordered by (timestamp, run_id) before splitting.")

    n_runs = len(meta)
    if n_runs < n_dev_blocks + 1:
        raise ValueError(f"Not enough runs ({n_runs}) to create {n_dev_blocks} dev blocks + holdout.")

    n_dev_nominal = round(n_runs * dev_fraction)
    nominal_ends = [round(k * n_dev_nominal / n_dev_blocks) for k in range(1, n_dev_blocks + 1)]

    timestamps = list(meta["timestamp"])
    run_ids = [int(r) for r in meta["run_id"]]

    adjusted_ends: list[int] = []
    tie_shifts: list[int] = []
    prev_end = 0
    for nom_end in nominal_ends:
        b = max(nom_end, prev_end)
        while 0 < b < n_runs and timestamps[b] == timestamps[b - 1]:
            b += 1
        adjusted_ends.append(b)
        tie_shifts.append(int(b - nom_end))
        prev_end = b

    block_names = [f"D{k}" for k in range(1, n_dev_blocks + 1)] + ["holdout"]
    starts = [0, *adjusted_ends]
    ends = [*adjusted_ends, n_runs]

    block_labels: list[str] = []
    blocks_info: dict[str, Any] = {}

    for idx, b_name in enumerate(block_names):
        s_idx = starts[idx]
        e_idx = ends[idx]
        count = e_idx - s_idx
        block_labels.extend([b_name] * count)

        nom_end = nominal_ends[idx] if idx < len(nominal_ends) else n_runs
        shift = tie_shifts[idx] if idx < len(tie_shifts) else 0
        blocks_info[b_name] = {
            "start_index": int(s_idx),
            "end_index_exclusive": int(e_idx),
            "n_runs": int(count),
            "nominal_end_index": int(nom_end),
            "tie_shift": int(shift),
            "min_timestamp": timestamps[s_idx].isoformat() if count > 0 else None,
            "max_timestamp": timestamps[e_idx - 1].isoformat() if count > 0 else None,
            "run_ids": run_ids[s_idx:e_idx],
        }

    core_manifest = {
        "n_total_runs": int(n_runs),
        "dev_fraction": float(dev_fraction),
        "n_dev_runs": int(adjusted_ends[-1]),
        "n_holdout_runs": int(n_runs - adjusted_ends[-1]),
        "nominal_dev_runs": int(n_dev_nominal),
        "nominal_holdout_runs": int(n_runs - n_dev_nominal),
        "blocks": blocks_info,
        "forward_folds": FORWARD_FOLDS,
        "final_fit_blocks": ["D1", "D2", "D3", "D4"],
        "calibration_block": "D5",
        "holdout_block": "holdout",
    }

    canonical_bytes = json.dumps(core_manifest, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    manifest_hash = hashlib.sha256(canonical_bytes).hexdigest()
    manifest = {
        "split_version": "1.0",
        "manifest_sha256": manifest_hash,
        **core_manifest,
    }
    return block_labels, manifest


def load_development_labels(labels_path: str | Path) -> pd.DataFrame:
    """Load development labels while strictly forbidding access to `labels_holdout.parquet`."""
    path = Path(labels_path)
    if "holdout" in path.name.lower():
        raise PermissionError(
            f"Label isolation violation: training and development code cannot read holdout labels ({path})."
        )
    if not path.exists():
        raise FileNotFoundError(f"Development labels file not found: {path}")

    df = pd.read_parquet(path)
    if "block" in df.columns and (df["block"] == "holdout").any():
        raise PermissionError(
            f"Label isolation violation: holdout rows detected inside development labels file ({path})."
        )
    return df


def prepare_dataset(config_path: str | Path = "configs/default.yaml") -> dict[str, Any]:
    """Run full data preparation: validate raw SECOM, freeze chronological split, seal holdout labels."""
    cfg = load_config(config_path)
    loaded = load_raw_secom(
        raw_dir=cfg.data.raw_dir,
        timestamp_format=cfg.data.timestamp_format,
        verify_uci_counts=cfg.data.verify_uci_counts,
        sort_chronologically=True,
    )
    full_df = loaded.runs_df

    # 1. Compute split from timestamps and run_id ONLY, before any block label statistics
    block_labels, split_manifest = compute_chronological_split(
        full_df[["run_id", "timestamp"]],
        dev_fraction=cfg.splits.dev_fraction,
        n_dev_blocks=cfg.splits.n_dev_blocks,
    )

    split_manifest_path = Path(cfg.data.split_manifest)
    split_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    split_manifest_path.write_text(json.dumps(split_manifest, indent=2) + "\n", encoding="utf-8")

    # 2. Write unlabeled runs.parquet (run_id, timestamp, timestamp_str, block, 590 sensors)
    runs_out = full_df[["run_id", "timestamp", "timestamp_str", *SENSOR_COLUMNS]].copy()
    runs_out.insert(3, "block", block_labels)
    runs_parquet_path = Path(cfg.data.runs_parquet)
    runs_parquet_path.parent.mkdir(parents=True, exist_ok=True)
    runs_out.to_parquet(runs_parquet_path, index=False)

    # 3. Separate and seal labels_dev.parquet and labels_holdout.parquet
    labels_all = pd.DataFrame(
        {
            "run_id": full_df["run_id"].values,
            "timestamp": full_df["timestamp"].values,
            "block": block_labels,
            "label_fail": full_df["label_fail"].values,
        }
    )
    labels_dev = labels_all[labels_all["block"].isin(DEV_BLOCKS)].reset_index(drop=True)
    labels_holdout = labels_all[labels_all["block"] == "holdout"].reset_index(drop=True)

    labels_dev_path = Path(cfg.data.labels_dev_parquet)
    labels_holdout_path = Path(cfg.data.labels_holdout_parquet)
    labels_dev.to_parquet(labels_dev_path, index=False)
    labels_holdout.to_parquet(labels_holdout_path, index=False)

    # 4. Compute development-only label stats (holdout fail count is never reported before T12)
    dev_block_stats: dict[str, Any] = {}
    for b_name in DEV_BLOCKS:
        b_sub = labels_dev[labels_dev["block"] == b_name]
        n_b = len(b_sub)
        n_f = int(b_sub["label_fail"].sum())
        dev_block_stats[b_name] = {
            "n_runs": int(n_b),
            "n_fails": int(n_f),
            "fail_rate": float(n_f / n_b) if n_b > 0 else 0.0,
        }

    n_dev_total = len(labels_dev)
    n_dev_fails = int(labels_dev["label_fail"].sum())

    # Also compute per-fold train-only cleaning summary if quality module is available
    pruning_by_fold: dict[str, Any] = {}
    try:
        from yieldguard.quality import fit_quality_transformer

        for fold_spec in FORWARD_FOLDS:
            f_id = fold_spec["fold_id"]
            tr_blocks = fold_spec["train_blocks"]
            tr_runs = runs_out[runs_out["block"].isin(tr_blocks)]
            tr_labels = labels_dev[labels_dev["block"].isin(tr_blocks)]
            qt = fit_quality_transformer(
                tr_runs[SENSOR_COLUMNS],
                tr_labels["label_fail"].values,
                max_missing_fraction=cfg.quality.max_missing_fraction,
                max_constant_dominant_fraction=cfg.quality.max_constant_dominant_fraction,
                collinearity_threshold=cfg.quality.collinearity_threshold,
            )
            pruning_by_fold[f_id] = qt.summary_counts()
    except ImportError:
        pass

    quality_update = {
        **loaded.quality_summary,
        "split_summary": {
            "manifest_sha256": split_manifest["manifest_sha256"],
            "n_dev_runs": split_manifest["n_dev_runs"],
            "n_holdout_runs": split_manifest["n_holdout_runs"],
            "tie_shifts": {
                b: split_manifest["blocks"][b]["tie_shift"] for b in [*DEV_BLOCKS, "holdout"]
            },
        },
        "development_label_summary": {
            "n_dev_runs": int(n_dev_total),
            "n_dev_fails": int(n_dev_fails),
            "dev_fail_rate": float(n_dev_fails / n_dev_total) if n_dev_total > 0 else 0.0,
            "by_block": dev_block_stats,
            "holdout_labels_sealed": True,
        },
    }
    if pruning_by_fold:
        quality_update["pruning_counts_by_fold"] = pruning_by_fold

    write_raw_quality_report(quality_update, cfg.data.quality_report)
    return split_manifest
