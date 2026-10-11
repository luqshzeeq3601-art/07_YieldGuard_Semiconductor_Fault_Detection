# YieldGuard Model Card: Version v0.1.0

## 1. Model Details

- **Model Name**: YieldGuard Semiconductor Process Control & Fault Detection System
- **Version**: `v0.1.0` (frozen on 11 October 2026; release bundle in `models/v0.1.0/`)
- **Primary Detector**: **E02 (Univariate Statistical Process Control - Individuals Chart)**
  - *Selection Basis*: Rule 4 (parsimony rule declared before modelling). The development paired PR-AUC difference between tuned XGBoost (E08) and Univariate SPC (E02) was $+0.0638$ with a 95% stratified bootstrap confidence interval of $[-0.0904, +0.2155]$ and moving-block interval of $[-0.1265, +0.2319]$. Because both intervals span zero, the simpler, fully interpretable SPC detector was chosen as primary.
  - *Parameters*: 10 monitored sensors selected via rank-biserial correlation on Phase I passing vs failing training runs with Benjamini-Hochberg FDR correction. Phase I centre line $\mu_0$ and moving-range sigma $\sigma_0$ fitted on passing training runs only.
- **Secondary & Diagnostic Detectors**:
  - **E01**: Chance Baseline (zero scores).
  - **E03**: Univariate EWMA Chart ($\lambda = 0.20, L = 3.0$).
  - **E04**: Multivariate Hotelling $T^2$ Chart (PCA on passing runs, 50-component cap).
  - **E05**: Multivariate Squared Prediction Error / $Q$-Residual ($SPE$).
  - **E06**: Combined MSPC Index ($\max(T^2 / \tau_{T^2}, SPE / \tau_{SPE})$).
  - **E07**: Isolation Forest (100 estimators, contamination 0.068, seed 42).
  - **E08**: XGBoost Booster (max depth 3, 50 estimators, learning rate 0.05, `scale_pos_weight` per fold).
- **Sensor Schema**: 590 raw continuous/discrete sensor channels. 274 retained columns after pruning columns with $>5\%$ missingness, constant channels, and collinear duplicates.
- **License**: MIT License (code); Creative Commons Attribution 4.0 International (SECOM dataset).
- **Release Status**: **Validated Release Candidate** (frozen model bundle checksums verified in `reports/staging_manifest_sha256.json`).

---

## 2. Intended Use & Domain

- **Primary Domain**: Semiconductor manufacturing process control and wafer fault detection (UCI SECOM).
- **Intended Users**: Process engineers, yield managers, and fab operations specialists.
- **Intended Workflow**:
  1. Automated ingestion of 590-channel run vectors via batch CSV CLI (`yieldguard score`) or REST API (`POST /score`).
  2. Stateless scoring against frozen alarm budgets (2, 5, or 10 false alarms per 100 passing runs).
  3. Interactive triage in Streamlit dashboard (`app/dashboard.py`): univariate EWMA with Western Electric run rules, multivariate $T^2$ / $SPE$ charts, and flagged-run sensor attribution drill-down.
  4. Root-cause inspection using MSPC contributions and XGBoost TreeSHAP with collinear sensor group maps.
- **Out of Scope**: Direct automated tool shutdown without engineering review; safety-critical plant interlocks; cross-fab transfer without retraining.

---

## 3. Training & Evaluation Protocol

- **Dataset**: UCI SECOM (1,567 runs, 590 channels, recorded across 2008).
- **Integrity**: Acquired strictly from the UCI repository with SHA-256 verification (`3fcf11550c8227fc64a515f40f00f074d6c703b71ca360b9ea5169a941f17fa9`).
- **Data Partitions (Strict Chronological Split)**:
  - **Development Set**: 1,175 runs (blocks D1–D5, each 235 runs; 80 fails, prevalence 6.81%).
    - *Training Blocks (D1–D4)*: 940 runs used to fit cleaning pipelines, sensor screening, Phase I limits, PCA, and models.
    - *Calibration Block (D5)*: 235 runs used exclusively to calibrate alarm budget thresholds (budgets 2, 5, 10).
  - **Holdout Set**: 392 runs (24 fails, prevalence 6.12%). Evaluated once post-freeze; predictions hashed before labels opened.
- **Leakage Prevention**: No random row shuffling. All imputation medians, scaling statistics, collinear group mappings, and alarm thresholds were fitted on training blocks only.

---

## 4. Empirical Evaluation & Target Outcomes

### 4.1 Development Cross-Validation Summary (D1–D4 Training, D5 Calibration)

| Experiment ID | Detector Family | Features | Mean PR-AUC (Folds 1–3) | Recall @ 5 FAR | Selected Threshold (Budget 5) |
|---|---|---|---|---|---|
| **E01** | Chance Baseline | Zeros | 0.0681 | 0.0500 | 0.0809 |
| **E02 (Primary)** | Univariate SPC Individuals | 10 screened sensors | **0.1147** | **0.1500** | **6.2822** |
| **E03** | Univariate EWMA ($\lambda=0.2$) | 10 screened sensors | 0.1049 | 0.1167 | 15.0602 |
| **E04** | Hotelling $T^2$ | PCA (50 components) | 0.0645 | 0.0667 | 80.1593 |
| **E05** | Squared Prediction Error ($SPE$) | PCA residuals | 0.0827 | 0.0667 | 319.9309 |
| **E06** | Combined MSPC Index | $\max(T^2, SPE)$ | 0.0768 | 0.0833 | 2.8462 |
| **E07** | Isolation Forest | 274 retained sensors | 0.0818 | 0.0667 | 0.4459 |
| **E08** | XGBoost Booster (`depth=3, n=50`) | 274 retained sensors | 0.1785 | 0.1833 | 0.4173 |

*Rule 4 Paired Difference (E08 − E02)*: $+0.0638$, 95% stratified CI $[-0.0904, +0.2155]$, 95% moving-block CI $[-0.1265, +0.2319]$. Both intervals contain zero, confirming E02 selection.

---

### 4.2 One-Shot Holdout Benchmark Results (392 Runs, 24 Fails)

Evidence source: `reports/holdout_metrics.json`.

| Target ID | Description | Predeclared Criterion | Measured Result | 95% Stratified CI | 95% Moving-Block CI | Verdict |
|---|---|---|---|---|---|---|
| **O1** | Fault Ranking PR-AUC | $\ge 2\times \text{prevalence} = 0.1224$ | **0.0724** | $[0.0559, 0.1173]$ | $[0.0384, 0.1514]$ | **MISS** |
| **O2** | Operational Recall @ 5 FAR | $\ge 0.20$ (chance 0.05) | **0.0417** (1 TP / 24 fails) | $[0.0000, 0.1250]$ | $[0.0000, 0.1765]$ | **MISS** |
| **O3** | Alarm Budget Stability | Realised alarms $\in [2.0, 8.0]$ per 100 | **3.06 per 100** (12 alarms) | N/A | N/A | **PASS** |
| **O4** | SPC vs ML Comparison | Report paired diff & 95% CI | **+0.0153** (E08: 0.0877 vs E02: 0.0724) | $[-0.0308, +0.0910]$ | $[-0.0470, +0.0839]$ | **REPORTED** |
| **O5** | Sensor Attribution Ranking | 5-$\sigma$ injected fault in top-3 $\ge 90\%$ | **100.0%** (10 / 10 runs) | N/A | N/A | **PASS** |
| **O6** | Score Reproducibility | Multi-run score delta $\le 10^{-6}$ | **0.0** (exact match) | N/A | N/A | **PASS** |
| **O7** | Engineering & API Delivery | Parity $\le 10^{-9}$; p95 latency | Parity: **0.0**; p95 latency: **63.8 ms** | N/A | N/A | **PASS** |

*Analysis of Missed Targets O1 and O2*:
- In SECOM, sensor readings exhibit strong non-stationarity and sensor ties over time, without physical wafer geometry or chamber IDs.
- On the holdout set, E02 achieved PR-AUC of 0.0724 (prevalence: 0.0612), missing the aggressive $2\times$ prevalence target (0.1224).
- Recall at 5 false alarms per 100 passing runs reached 0.0417 (1 true alarm among 24 failing runs), below the 0.20 target.
- These negative empirical results are reported honestly as measured, without post-hoc threshold adjustment or test set retuning.

---

## 5. Explainability & Root-Cause Attribution

- **Methods Implemented**:
  1. *Multivariate SPC Contribution Decomposition*: Exact reconstruction of $T^2$ and $SPE$ statistics partitioned into per-sensor contributions.
  2. *TreeSHAP*: TreeExplainer on raw margin of XGBoost model with Phase I training background.
- **Attribution Smearing Mitigation**:
  - SECOM features contain high collinearity ($|r| > 0.98$).
  - When attributing a flagged run, the top 5 sensors are accompanied by their correlated cluster members (`collinear_group_members`), preventing engineers from erroneously isolating a single sensor when an entire cluster moved together.

---

## 6. Data Drift & Monitoring

- **Tooling**: Evidently batch data drift preset (`evidently.legacy.report.Report` with `DataDriftPreset(drift_share=0.50)`).
- **Reference Cohort**: 940 Phase I–IV training runs across 274 retained sensors (`monitoring_reference.parquet`).
- **Batch Threshold**: Minimum 100 runs required; cohorts $< 100$ runs return `status="insufficient_data"` without raising false alarms.
- **Verification Controls (`reports/drift/controls/control_results.json`)**:
  - Control 1 (No-change): 0 drifted features, `status="ok"` (PASS).
  - Control 2 (Shifted-sensors): 5 perturbed sensors shifted by $+2\sigma$ all detected, `status="drift_detected"` (PASS).
  - Control 3 (Insufficient-data): 50-run batch returns `status="insufficient_data"` (PASS).
- **Holdout Drift Findings (`reports/drift/drift_summary.json`)**:
  - 237 of 274 retained sensors (86.5%) exhibited statistically significant distribution shift ($p < 0.05$, K-S test).
  - 48 sensors exhibited missingness rate shifts $> 5\%$.
  - Indicates substantial temporal drift across the 2008 manufacturing collection period.

---

## 7. Known Limitations & Operational Boundaries

1. **Offline Benchmark Disclaimer**:
   *All figures produced under this protocol are offline benchmark results on a public dataset. They are not measured yield improvement, scrap reduction, or financial savings.*
2. **Missing Wafer Spatial Context**: The public SECOM dataset omits die-level coordinates, chamber IDs, and lot tracking. Sensor-only fault classification remains constrained by low signal-to-noise ratio.
3. **Discrete Tied Scores**: Due to sensor quantization in SECOM, scores exhibit tied values. Under strict $>$ thresholding, realised alarm rates may fall below target budgets on small calibration batches (recorded in D013).
4. **Local Container Execution**: Dockerfile and `.dockerignore` are verified via local live API smoke tests and GitHub Actions CI workflow, but remain marked unverified locally due to local host Docker daemon status.
