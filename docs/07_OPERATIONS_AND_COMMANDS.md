# 07. Operations and planned commands

## 1. Command status

Run commands from the project root in PowerShell using the project-local Python 3.11 virtual environment (`.venv\Scripts\python.exe`). Do not install packages globally.

## 2. Environment setup: T01

```powershell
Set-Location '07_YieldGuard_Semiconductor_Fault_Detection'
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.lock
.\.venv\Scripts\python.exe -m pip install -e . --no-deps
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m ruff check src tests
.\.venv\Scripts\python.exe -m pytest tests/test_smoke.py -q
```

## 3. Data acquisition and preparation: T02-T04

```powershell
.\.venv\Scripts\python.exe scripts/download_data.py --output data/raw/secom --manifest data/source_manifest.json
.\.venv\Scripts\python.exe -m yieldguard.cli prepare --config configs/default.yaml
```

`download_data.py` fetches `secom.zip` from the UCI archive, checks against Zip Slip, verifies SHA-256 hashes and records CC BY 4.0 licence and DOI metadata. `prepare` validates raw files, orders runs by `(timestamp, run_id)`, freezes `data/processed/split_manifest.json`, separates `labels_dev.parquet` from `labels_holdout.parquet`, and writes `reports/data_quality.json`.

## 4. Development experiments, selection and thresholds: T05-T11

```powershell
.\.venv\Scripts\python.exe -m yieldguard.cli train --config configs/default.yaml --experiment spc
.\.venv\Scripts\python.exe -m yieldguard.cli train --config configs/default.yaml --experiment mspc
.\.venv\Scripts\python.exe -m yieldguard.cli train --config configs/default.yaml --experiment anomaly
.\.venv\Scripts\python.exe -m yieldguard.cli train --config configs/default.yaml --experiment xgboost
.\.venv\Scripts\python.exe -m yieldguard.cli evaluate --config configs/default.yaml --split dev
.\.venv\Scripts\python.exe -m yieldguard.cli explain --config configs/default.yaml --verify-synthetic
```

`evaluate --split dev` compares E01-E08 across forward folds D3-D5, selects the primary detector via the predeclared rule, sets alarm thresholds on calibration block D5 for budgets 2, 5 and 10 per 100 runs, and writes `reports/validation_metrics.json`.

## 5. Freeze and one-shot holdout evaluation: T12

```powershell
.\.venv\Scripts\python.exe -m yieldguard.cli freeze --config configs/default.yaml --version v0.1.0
.\.venv\Scripts\python.exe -m yieldguard.cli evaluate --config configs/default.yaml --split holdout --bundle models/v0.1.0 --release-id v0.1.0 --allow-heldout-evaluation
```

`evaluate --split holdout` requires user approval at Checkpoint C3, a frozen bundle and `--allow-heldout-evaluation`. It writes and hashes `reports/holdout_predictions.csv` before reading `data/processed/labels_holdout.parquet`, writes `reports/holdout_metrics.json`, and refuses to overwrite an existing record.

## 6. Batch scoring, explanations and drift: T13, T16

```powershell
.\.venv\Scripts\python.exe -m yieldguard.cli score --bundle models/v0.1.0 --input data/processed/runs_holdout_unlabeled.csv --output reports/scored_runs.csv
.\.venv\Scripts\python.exe -m yieldguard.cli explain --bundle models/v0.1.0 --input data/processed/runs_holdout_unlabeled.csv --output reports/explanations/flagged_rankings.json
.\.venv\Scripts\python.exe -m yieldguard.cli drift-controls --bundle models/v0.1.0 --output reports/drift/controls
.\.venv\Scripts\python.exe -m yieldguard.cli drift --bundle models/v0.1.0 --current data/processed/runs.parquet --split holdout --output reports/drift
```

## 7. API and Streamlit dashboard: T14, T15

```powershell
$env:YIELDGUARD_BUNDLE_DIR = 'models/v0.1.0'
.\.venv\Scripts\python.exe -m uvicorn api.main:app --host 127.0.0.1 --port 8000
.\.venv\Scripts\python.exe -m streamlit run app/dashboard.py
```

## 8. Runtime benchmark, linting and tests: T17, T18

```powershell
.\.venv\Scripts\python.exe scripts/benchmark_runtime.py --bundle models/v0.1.0 --output reports/runtime.json
.\.venv\Scripts\python.exe -m ruff check src api app scripts tests
.\.venv\Scripts\python.exe -m pytest tests -q
```
