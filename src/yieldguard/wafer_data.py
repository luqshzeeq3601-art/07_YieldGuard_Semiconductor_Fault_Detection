"""Acquisition, validation, and canonical loading for the Semiconductor Wafer Etch dataset (Wafer).

Dataset: Bobick & Olszewski (2001), CMU / UCR Time Series Classification Archive ('Wafer').
- 7,164 semiconductor wafer inline process sensor traces (1,000 train + 6,164 test in UCR archive)
- 152 inline sensor time-step readings per wafer run (s000 .. s151)
- Raw UCR labels: +1 = normal wafer (89.36%, 6,402 runs), -1 = abnormal/faulty wafer (10.64%, 762 runs)
- YieldGuard canonical label convention: +1 = fail/abnormal (minority), -1 = pass/normal (majority)
"""

from __future__ import annotations

import hashlib
import io
import json
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

WAFER_URLS = [
    "https://www.timeseriesclassification.com/aeon-toolkit/Wafer.zip",
    "https://timeseriesclassification.com/Downloads/Wafer.zip",
]
WAFER_N_RUNS = 7164
WAFER_N_FEATURES = 152


def _safe_extract_zip(zip_bytes: bytes, dest_dir: Path) -> list[Path]:
    """Extract zip archive into dest_dir while preventing Zip-Slip path traversal."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_resolved = dest_dir.resolve()
    extracted: list[Path] = []
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        for member in zf.infolist():
            target_path = (dest_dir / member.filename).resolve()
            if not str(target_path).startswith(str(dest_resolved)):
                raise ValueError(f"Unsafe zip member path traversal blocked: {member.filename}")
            if member.is_dir():
                target_path.mkdir(parents=True, exist_ok=True)
            else:
                target_path.parent.mkdir(parents=True, exist_ok=True)
                target_path.write_bytes(zf.read(member.filename))
                extracted.append(target_path)
    return extracted


def _parse_ucr_txt_or_ts(file_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Parse UCR whitespace/comma-separated .txt or .ts file into (raw_labels, features_matrix)."""
    text = file_path.read_text(encoding="utf-8", errors="replace").splitlines()
    rows_y: list[int] = []
    rows_x: list[list[float]] = []

    is_ts_format = file_path.suffix.lower() == ".ts" or any(line.strip().lower().startswith("@data") for line in text[:100])
    in_data = not is_ts_format

    for line_num, raw_line in enumerate(text, start=1):
        line = raw_line.strip()
        if not line or line.startswith("#") or line.startswith("%"):
            continue
        if is_ts_format and not in_data:
            if line.lower().startswith("@data"):
                in_data = True
            continue

        if ":" in line and is_ts_format:
            # aeon .ts format: v1,v2,...,v152:label
            parts = line.rsplit(":", 1)
            vals_str, lbl_str = parts[0], parts[1]
            vals = [float(x) for x in vals_str.replace(",", " ").split()]
            lbl_float = float(lbl_str.strip())
        else:
            # Standard UCR .txt format: label v1 v2 ... v152
            tokens = line.replace(",", " ").split()
            lbl_float = float(tokens[0])
            vals = [float(x) for x in tokens[1:]]

        if len(vals) != WAFER_N_FEATURES:
            raise ValueError(
                f"File {file_path.name} line {line_num}: expected {WAFER_N_FEATURES} features, got {len(vals)}."
            )
        lbl_int = int(round(lbl_float))
        if lbl_int not in (-1, 1):
            raise ValueError(f"File {file_path.name} line {line_num}: invalid label {lbl_float}.")
        rows_y.append(lbl_int)
        rows_x.append(vals)

    return np.asarray(rows_y, dtype=np.int64), np.asarray(rows_x, dtype=np.float64)


def download_and_prepare_wafer(
    raw_dir: str | Path = "data/raw/wafer",
    processed_path: str | Path = "data/processed/wafer_runs.parquet",
    manifest_path: str | Path = "data/wafer_source_manifest.json",
    urls: list[str] | None = None,
) -> dict[str, Any]:
    """Download UCR Wafer archive, verify integrity, and save canonical DataFrame + manifest."""
    raw_dir_p = Path(raw_dir)
    raw_dir_p.mkdir(parents=True, exist_ok=True)
    zip_path = raw_dir_p / "Wafer.zip"

    source_urls = urls or WAFER_URLS
    used_url = source_urls[0]

    if not zip_path.exists():
        last_err: Exception | None = None
        for candidate_url in source_urls:
            try:
                req = urllib.request.Request(
                    candidate_url,
                    headers={"User-Agent": "YieldGuard-Benchmark/0.2.0"},
                )
                with urllib.request.urlopen(req, timeout=60) as resp:
                    zip_bytes = resp.read()
                zip_path.write_bytes(zip_bytes)
                used_url = candidate_url
                break
            except Exception as exc:  # noqa: BLE001
                last_err = exc
        else:
            raise RuntimeError(f"Failed to download Wafer.zip from all mirrors. Last error: {last_err}")
    else:
        zip_bytes = zip_path.read_bytes()

    zip_sha256 = hashlib.sha256(zip_bytes).hexdigest()
    extracted_files = _safe_extract_zip(zip_bytes, raw_dir_p)

    # Locate Wafer_TRAIN and Wafer_TEST files (.txt preferred, fallback to .ts)
    train_candidates = sorted(
        [p for p in extracted_files if "wafer_train" in p.name.lower() and p.suffix.lower() in (".txt", ".ts")],
        key=lambda p: (0 if p.suffix.lower() == ".txt" else 1, p.name),
    )
    test_candidates = sorted(
        [p for p in extracted_files if "wafer_test" in p.name.lower() and p.suffix.lower() in (".txt", ".ts")],
        key=lambda p: (0 if p.suffix.lower() == ".txt" else 1, p.name),
    )
    if not train_candidates or not test_candidates:
        raise FileNotFoundError(f"Could not find Wafer_TRAIN and Wafer_TEST in {[p.name for p in extracted_files]}")

    train_file = train_candidates[0]
    test_file = test_candidates[0]

    y_tr_raw, x_tr = _parse_ucr_txt_or_ts(train_file)
    y_te_raw, x_te = _parse_ucr_txt_or_ts(test_file)

    y_raw = np.concatenate([y_tr_raw, y_te_raw], axis=0)
    x_all = np.vstack([x_tr, x_te])

    if len(y_raw) != WAFER_N_RUNS or x_all.shape[1] != WAFER_N_FEATURES:
        raise ValueError(
            f"Wafer shape mismatch: expected ({WAFER_N_RUNS}, {WAFER_N_FEATURES}), got {x_all.shape}."
        )

    # In raw UCR Wafer: -1 is abnormal/faulty (~10.64%, 762 wafers), +1 is normal (~89.36%, 6402 wafers).
    # Convert to YieldGuard canonical convention: +1 = fail (abnormal), -1 = pass (normal).
    y_canonical = np.where(y_raw == -1, 1, -1).astype(np.int64)

    sensor_cols = [f"s{j:03d}" for j in range(WAFER_N_FEATURES)]
    df = pd.DataFrame(x_all, columns=sensor_cols)
    df.insert(0, "run_id", [f"wafer_{i:04d}" for i in range(len(df))])
    # Assign sequential run timestamps at 10-minute intervals for chronological tracking
    timestamps = pd.date_range("2001-01-01 00:00:00", periods=len(df), freq="10min").strftime("%Y-%m-%d %H:%M:%S")
    df.insert(1, "timestamp", timestamps)
    df["label"] = y_canonical

    proc_p = Path(processed_path)
    proc_p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(proc_p, index=False)
    parquet_sha256 = hashlib.sha256(proc_p.read_bytes()).hexdigest()

    n_fails = int(np.sum(y_canonical == 1))
    n_pass = int(np.sum(y_canonical == -1))

    manifest = {
        "dataset_id": "WAFER_ETCH",
        "source_url": used_url,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "zip_sha256": zip_sha256,
        "train_file": train_file.name,
        "train_file_sha256": hashlib.sha256(train_file.read_bytes()).hexdigest(),
        "test_file": test_file.name,
        "test_file_sha256": hashlib.sha256(test_file.read_bytes()).hexdigest(),
        "processed_parquet": str(proc_p),
        "processed_parquet_sha256": parquet_sha256,
        "n_total_runs": int(len(df)),
        "n_train_ucr": int(len(y_tr_raw)),
        "n_test_ucr": int(len(y_te_raw)),
        "n_features": WAFER_N_FEATURES,
        "n_fails_abnormal": n_fails,
        "n_passing_normal": n_pass,
        "fail_prevalence": float(n_fails / len(df)),
        "citation": "Olszewski, R. T. (2001). Generalized Feature Extraction for Structural Pattern Recognition in Time-Series Data. PhD Thesis, Carnegie Mellon University.",
        "license": "Public Research Benchmark (UCR Time Series Classification Archive)",
    }

    man_p = Path(manifest_path)
    man_p.parent.mkdir(parents=True, exist_ok=True)
    man_p.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def load_wafer_runs(processed_path: str | Path = "data/processed/wafer_runs.parquet") -> tuple[pd.DataFrame, list[str]]:
    """Load prepared Wafer Etch DataFrame and list of sensor column names."""
    p = Path(processed_path)
    if not p.exists():
        raise FileNotFoundError(
            f"Prepared Wafer dataset not found at {p}. Run `python scripts/download_wafer_data.py` first."
        )
    df = pd.read_parquet(p)
    sensor_cols = [c for c in df.columns if c.startswith("s") and c[1:].isdigit()]
    return df, sensor_cols
