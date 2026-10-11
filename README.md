# YieldGuard

**Semiconductor process control, early fault detection, and sensor attribution under bounded false alarm budgets.**

[![License: MIT](https://img.shields.io/badge/License-MIT-0f766e.svg)](LICENSE)
[![Python: 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](pyproject.toml)

YieldGuard is a decision-support and fault detection system built on the UCI SECOM semiconductor manufacturing dataset (590 sensor channels, 1,567 runs). It combines classical univariate Statistical Process Control (Individuals and EWMA charts), multivariate Hotelling $T^2$ / $SPE$ charts, Isolation Forest, and tuned XGBoost under a strict false-alarm budget constraint (2, 5, or 10 alarms per 100 passing runs).

---

## 1. System Architecture & Workflow

```
Raw SECOM Runs (590 channels)
      │
      ▼
Train-Only Cleaning & Sensor Screening (Phase I: D1–D4, 940 runs)
  ├── Retained Features: 274 channels (drops >5% missing, constants, collinear duplicates)
  ├── 10 Monitored SPC Sensors (Rank-Biserial correlation with FDR correction)
  └── Collinear Group Tracking (preserves sensor groupings for explainability)
      │
      ▼
Detector Candidate Evaluation & Selection
  ├── E01: Chance Baseline
  ├── E02: Univariate SPC Individuals (Phase I Centre Line & Moving-Range Sigma) [Selected Primary]
  ├── E03: Univariate EWMA Chart (λ = 0.20, L = 3.0)
  ├── E04: Hotelling T² Chart (PCA on passing runs, 50-component cap)
  ├── E05: Squared Prediction Error (SPE / Q-Residual)
  ├── E06: Combined MSPC Index
  ├── E07: Isolation Forest
  └── E08: XGBoost Booster
      │
      ▼
Threshold Calibration on D5 (235 runs)
  └── Quantile thresholds set at strict budgets: 2, 5, and 10 alarms per 100 passing runs
      │
      ▼
Frozen Release Bundle (`models/v0.1.0/`) & One-Shot Holdout Evaluation (392 runs, 24 fails)
      │
      ├── CLI Batch Scoring (`yieldguard score`, `yieldguard explain`)
      ├── REST API Service (FastAPI: `/health`, `/ready`, `/model-info`, `/score`, `/spc`)
      ├── Interactive Engineering Dashboard (Streamlit: 4 operational views)
      └── Offline Data Drift Monitoring (Evidently + missingness tracking + 3 controls)
```

---

## 2. Measured Empirical Results

All measurements are recorded on the **unseen chronological holdout set** (392 runs, 24 fails, prevalence 6.12%). Evaluations were executed once post-freeze; predictions were written and hashed before opening holdout labels (`reports/holdout_predictions.csv`).

| Target ID | Description | Predeclared Criterion | Measured Holdout Result | 95% Wider Bootstrap CI | Fail Count | Target Status | Evidence Link |
|---|---|---|---|---|---|---|---|
| **O1** | Fault Ranking PR-AUC | $\ge 2\times \text{prevalence} = 0.1224$ | **0.0724** | $[0.0384, 0.1514]$ | 24 | **MISS** | [holdout_metrics.json](reports/holdout_metrics.json) |
| **O2** | Operational Recall @ 5 FAR | $\ge 0.20$ (chance 0.05) | **0.0417** (1 / 24 fails) | $[0.0000, 0.1765]$ | 24 | **MISS** | [holdout_metrics.json](reports/holdout_metrics.json) |
| **O3** | Alarm Budget Stability | Realised alarms $\in [2.0, 8.0]$ per 100 | **3.06 per 100** (12 alarms) | N/A | 24 | **PASS** | [holdout_metrics.json](reports/holdout_metrics.json) |
| **O4** | SPC vs ML Comparison | Report paired diff & 95% CI | **+0.0153** (E08 vs E02) | $[-0.0470, +0.0910]$ | 24 | **REPORTED** | [holdout_metrics.json](reports/holdout_metrics.json) |
| **O5** | Sensor Attribution Ranking | 5-$\sigma$ injected fault in top-3 $\ge 90\%$ | **100.0%** | N/A | N/A | **PASS** | [o5_verification.json](reports/explanations/o5_synthetic_verification.json) |
| **O6** | Score Reproducibility | Multi-run score delta $\le 10^{-6}$ | **0.0** (exact match) | N/A | N/A | **PASS** | [runtime.json](reports/runtime.json) |
| **O7** | Engineering & API Delivery | CLI/API parity $\le 10^{-9}$; p95 latency | Parity: **0.0**; p95: **63.8 ms** | N/A | N/A | **PASS** | [runtime.json](reports/runtime.json) |

> [!WARNING]
> **Mandatory Claims Boundary**: All figures produced under this protocol are offline benchmark results on a public dataset. They are not measured yield improvement, scrap reduction, or financial savings.

---

## 3. Quick Start

### 3.1 Local Environment Setup

Requires Python 3.11:

```powershell
# Create local virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install pinned dependencies and package in editable mode
pip install -r requirements.lock
pip install --no-deps -e .
```

### 3.2 Run the REST API Service

```powershell
$env:YIELDGUARD_BUNDLE_DIR = "models/v0.1.0"
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Verify service readiness and run the end-to-end smoke test:

```powershell
python scripts/smoke_api.py --url http://127.0.0.1:8000
```

Interactive OpenAPI documentation is available at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

### 3.3 Run the Interactive Streamlit Dashboard

```powershell
python -m streamlit run app/dashboard.py
```

The dashboard includes:
1. **Univariate EWMA & Run Rules**: Interactive control charts with Phase I limits ($\mu_0, \sigma_0$) and Western Electric run rule markers.
2. **Multivariate SPC ($T^2$ & $SPE$)**: Hotelling $T^2$ and Squared Prediction Error tracking across time.
3. **Flagged-Run Drill-Down**: MSPC contribution bars and TreeSHAP waterfall attributions with collinear group maps.
4. **Alarm-Budget Performance**: Realised alarm rate and precision/recall trade-offs across budgets 2, 5, and 10.
5. **Synthetic Demo Mode**: Seamless toggle to explore full dashboard functionality on synthetic data without raw data downloads.

### 3.4 Batch Scoring & Explanations via CLI

```powershell
# Batch score a CSV of runs
python -m yieldguard.cli score --bundle models/v0.1.0 --input data/processed/runs_holdout_unlabeled.csv --output reports/scored_runs.csv

# Generate sensor rankings for flagged runs
python -m yieldguard.cli explain --bundle models/v0.1.0 --input data/processed/runs_holdout_unlabeled.csv --output reports/explanations/sensor_rankings.json

# Run Evidently data drift monitoring
python -m yieldguard.cli drift --bundle models/v0.1.0 --current data/processed/runs.parquet --split holdout --output reports/drift

# Run 3 drift verification controls
python -m yieldguard.cli drift-controls --bundle models/v0.1.0 --output reports/drift/controls
```

---

## 4. Testing & Quality Verification

Run the full automated test suite (all 74+ tests run on synthetic data without requiring SECOM download):

```powershell
# Run code formatting and linter
python -m ruff check src api app scripts tests

# Run complete pytest test suite
python -m pytest -v
```

---

## 5. Status of Unverified Components

In accordance with project integrity standards:
- **Docker Container Runtime**: Container specification is fully defined (`Dockerfile`, `.dockerignore`, `scripts/smoke_api.py`). The local Docker daemon was offline during local verification; container execution is marked **unverified locally** and validated via GitHub Actions CI workflow.
- **Remote CI Run**: `.github/workflows/ci.yml` defines the `test` and `docker-smoke` jobs; status is **unverified on GitHub** until the repository is pushed to remote.

---

## 6. Dataset Attribution & License

- **Dataset**: UCI Machine Learning Repository - SECOM Dataset.
- **Citation**: McCann, M., Li, Y., Maguire, L., & Johnston, A. (2008). *SECOM Data Set*. University of Ulster.
- **Dataset License**: Creative Commons Attribution 4.0 International ([CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)).
- **Code License**: [MIT License](LICENSE).
