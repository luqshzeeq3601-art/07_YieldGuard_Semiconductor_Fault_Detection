# YieldGuard Model Card: Version v0.1.0

## 1. Model Details

- **Model Name**: YieldGuard Semiconductor Process Control & Fault Detection System
- **Version**: `v0.1.0` (frozen on 11 October 2026; release bundle in `models/v0.1.0/`)
- **Headline Holdout Verdict (`v0.1.0`)**: **No detector beat chance on the unseen chronological SECOM holdout** (392 runs, 24 fails, prevalence `6.12%`). Every detector's 95% PR-AUC confidence interval includes or falls below the `0.0612` prevalence baseline, and while primary detector `E02` stayed within the `O3` alarm-volume window (`3.06 alarms per 100 runs`, 12 total alarms), **0 of those 12 alarms were true failures (`0 TP, 12 FP`)**.
- **Primary Detector**: **E02 (Univariate Statistical Process Control - Individuals Chart)**
  - *Selection Basis*: Rule 4 (parsimony rule declared before modelling). The development paired PR-AUC difference between tuned XGBoost (E08) and Univariate SPC (E02) was $+0.0638$ with a 95% stratified bootstrap confidence interval of $[-0.0899, +0.2161]$ and moving-block interval of $[-0.1273, +0.2324]$. Because both intervals span zero, the simpler, fully interpretable SPC detector was chosen as primary.
  - *Parameters*: 10 monitored sensors selected via rank-biserial correlation on Phase I passing vs failing training runs (among retained sensors with $\le 5\%$ missingness) with Benjamini-Hochberg FDR correction. Phase I centre line $\mu_0$ and moving-range sigma $\sigma_0 = \overline{MR}/1.128$ fitted on passing training runs only.
- **Secondary & Diagnostic Detectors**:
  - **E01**: Chance Baseline (constant prevalence score).
  - **E03**: Univariate EWMA Chart ($\lambda = 0.20, L = 3.0$).
  - **E04**: Multivariate Hotelling $T^2$ Chart (PCA on passing runs, 50-component cap, 61.97% cumulative variance explained).
  - **E05**: Multivariate Squared Prediction Error / $Q$-Residual ($SPE$).
  - **E06**: Combined MSPC Index ($T^2 / \tau_{T^2} + SPE / \tau_{SPE}$).
  - **E07**: Isolation Forest (`200` estimators, `max_samples='auto'`, seed `42`, fitted strictly on passing training runs).
  - **E08**: XGBoost Booster (`xgb_c01`: `max_depth=1, n_estimators=100, learning_rate=0.03, subsample=0.8, colsample_bytree=0.5, min_child_weight=5`, dynamic per-fold `scale_pos_weight = 11.3684` on fold 3).
- **Sensor Schema**: 590 raw continuous/discrete sensor channels. 274 retained columns on Fold 3 after pruning columns with $>50\%$ missingness (24 dropped), $\ge 99\%$ constant values (122 dropped), and $|r| \ge 0.95$ collinear duplicates (170 dropped with group map preserved).
- **License**: MIT License (code); Creative Commons Attribution 4.0 International (SECOM dataset).
- **Release Status**: **Validated Release Candidate — Recorded Negative Empirical Result** (frozen model bundle checksums verified in `reports/staging_manifest_sha256.json`).

---

## 2. Intended Use & Domain

- **Primary Domain**: Semiconductor manufacturing process control and wafer fault detection (UCI SECOM).
- **Intended Users**: Process engineers, yield managers, and fab operations specialists.
- **Intended Workflow**:
  1. Automated ingestion of 590-channel run vectors via batch CSV CLI (`yieldguard score`) or REST API (`POST /score`).
  2. Stateless scoring against frozen alarm budgets (2, 5, or 10 alarms per 100 scored runs).
  3. Interactive triage in Streamlit dashboard (`app/dashboard.py`): project overview, univariate EWMA with Western Electric run rules, multivariate $T^2$ / $SPE$ charts, and flagged-run sensor attribution drill-down.
  4. Descriptive sensor inspection using MSPC contributions and XGBoost TreeSHAP with collinear sensor group maps.
- **Out of Scope**: Direct automated tool shutdown without engineering review; safety-critical plant interlocks; cross-fab transfer without retraining.

---

## 3. Training & Evaluation Protocol

- **Dataset**: UCI SECOM (1,567 runs, 590 channels, recorded July–October 2008, DOI `10.24432/C54305`).
- **Integrity**: Acquired strictly from the UCI repository with SHA-256 verification (`secom.zip`: `eea568baf3c2229096d7d294cf0b096b5502bd96d92c0b80a65b84714059be8e`, `secom.data`: `20f0e7ee434f7dcbae0eea9ffff009a2b57f42d6b0dc9a5bd4f00782c0a3374c`, `secom_labels.data`: `126884cf453705c9e61a903fe906f0665a3b45ce3639e621edc5c93c89627e03`).
- **Data Partitions (Strict Chronological Split)**:
  - **Development Set**: 1,175 runs (blocks D1–D5, each 235 runs; 80 fails, prevalence 6.81%).
    - *Training Blocks (D1–D4)*: 940 runs (76 fails, 864 passing) used to fit cleaning pipelines, sensor screening, Phase I limits, PCA, and models.
    - *Calibration Block (D5)*: 235 runs (4 fails, 231 passing) used exclusively to calibrate alarm budget thresholds (budgets 2, 5, 10 alarms per 100 scored runs).
  - **Holdout Set**: 392 runs (24 fails, prevalence 6.12%). Evaluated once post-freeze; predictions hashed (`678f6268c4b3952a30d4e17273a039649dafd77229310d539a61b8f2201ee36a`) before labels opened.
- **Leakage Prevention**: No random row shuffling. All imputation medians, scaling statistics, collinear group mappings, and alarm thresholds were fitted on training blocks only.

---

## 4. Empirical Evaluation & Target Outcomes

### 4.1 Development Forward-Fold Summary (`D3`–`D5` Evaluation, 705 Runs, 22 Fails, Prevalence 0.0312)

Evidence source: `reports/validation_metrics.json`.

| Experiment ID | Detector Family | Features | Mean Fold PR-AUC (`D3`–`D5`) | 95% Wider PR-AUC CI | Mean Recall @ 5 FAR | Selected Threshold (Budget 5 on `D5`) |
|---|---|---|---|---|---|---|
| **E01** | Chance Baseline | Prevalence constant | 0.0312 | $[0.0184, 0.0454]$ | 0.0500 | 0.0809 |
| **E02 (Primary)** | Univariate SPC Individuals | 10 screened sensors | **0.1153** | $[0.0471, 0.2602]$ | **0.2083** | **6.2822** |
| **E03** | Univariate EWMA ($\lambda=0.2$) | 10 screened sensors | 0.1046 | $[0.0295, 0.2185]$ | 0.0833 | 15.0602 |
| **E04** | Hotelling $T^2$ | PCA (50 components) | 0.1034 | $[0.0497, 0.2810]$ | 0.2500 | 80.1593 |
| **E05** | Squared Prediction Error ($SPE$) | PCA residuals | 0.0857 | $[0.0470, 0.2045]$ | 0.1667 | 319.9309 |
| **E06** | Combined MSPC Index | $T^2/\tau_{T^2} + SPE/\tau_{SPE}$ | 0.0911 | $[0.0479, 0.2212]$ | 0.1667 | 2.8462 |
| **E07** | Isolation Forest | 274 retained sensors | 0.1383 | $[0.0511, 0.3291]$ | 0.1667 | 0.4459 |
| **E08** | XGBoost Booster (`xgb_c01`, `depth=1, n=100`) | 274 retained sensors | 0.1791 | $[0.0513, 0.3321]$ | 0.1750 | 0.4173 |

*Rule 4 Paired Difference (E08 − E02)*: $+0.0638$, 95% stratified CI $[-0.0899, +0.2161]$, 95% moving-block CI $[-0.1273, +0.2324]$. Both intervals contain zero, confirming `E02` selection.

---

### 4.2 One-Shot Holdout Benchmark Results (392 Runs, 24 Fails, Prevalence 0.0612)

Evidence source: `reports/holdout_metrics.json`.

| Target ID | Description | Predeclared Criterion | Measured Result | 95% Stratified CI | 95% Moving-Block CI | Verdict |
|---|---|---|---|---|---|---|
| **O1** | Fault Ranking PR-AUC | $\ge 2\times \text{prevalence} = 0.1224$ | **0.0724** (chance: 0.0612) | $[0.0559, 0.1173]$ | $[0.0384, 0.1514]$ | **MISS** |
| **O2** | Operational Recall @ 5 FAR | $\ge 0.20$ (chance: 0.05) | **0.0417** (1 TP / 24 fails) | $[0.0000, 0.1250]$ | $[0.0000, 0.1765]$ | **MISS** |
| **O3** | Alarm Budget Stability | Realised alarms $\in [2.0, 8.0]$ per 100 | **3.06 per 100** (12 alarms: **0 TP, 12 FP**) | N/A | N/A | **PASS (0 TP)** |
| **O4** | SPC vs ML Comparison | Report paired diff & 95% CI | **+0.0153** (E08: 0.0877 vs E02: 0.0724) | $[-0.0308, +0.0910]$ | $[-0.0470, +0.0839]$ | **REPORTED** |
| **O5** | Sensor Attribution Ranking | 5-$\sigma$ injected fault in top-3 $\ge 90\%$ | **100.0%** (100 / 100 trials) | N/A | N/A | **PASS** |
| **O6** | Score Reproducibility | Multi-run score delta $\le 10^{-6}$ | **0.0** (exact match) | N/A | N/A | **PASS** |
| **O7** | Engineering & API Delivery | Parity $\le 10^{-9}$; p95 latency | Parity: **0.0**; p95 latency: **63.8 ms** | N/A | N/A | **PASS** |

*Analysis of Missed Targets O1 and O2 (Why No Detector Beat Chance)*:
- Across all eight detectors (`E01`–`E08`), no detector achieved a 95% PR-AUC confidence interval above the 6.12% failure prevalence on the holdout set (`E02`: `0.0724`, `E03`: `0.0547`, `E04`: `0.0573`, `E05`: `0.0594`, `E06`: `0.0583`, `E07`: `0.0609`, `E08`: `0.0877`).
- Although `E02` satisfied `O3` by raising 12 alarms (`3.06 per 100 runs`, inside `[2.0, 8.0]`), all 12 alarms were false positives (`0 TP, 12 FP`, precision `0.0`, recall `0.0` at the frozen D5 threshold `6.2822`).
- Meanwhile, multivariate and ML detectors (`E04`, `E05`, `E06`, `E08`) experienced severe score inflation from temporal sensor drift (86.5% of retained sensors drifted), doubling their realised alarm rates to `9.95–10.46 per 100 runs` at budget 5 while catching at most 1–3 of 24 fails.
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
4. **Container & CI Execution**: Local Docker image build (`docker build -t yieldguard:ci .`) and live container smoke test (`scripts/smoke_api.py`) are verified locally (`status="verified_locally"` in `reports/runtime.json`). Remote GitHub Actions workflow execution (`.github/workflows/ci.yml`) remains unverified until pushed to a remote repository.

---

## 8. Post-`v0.1.0` Diagnostics & Protocol `v0.2.0` Confirmatory Benchmark

### 8.1 Phase 2 Root-Cause Diagnostics on SECOM (`reports/exploratory/secom_phase2_diagnostics.json`)
- **Label**: `exploratory_post_holdout` (not a confirmatory claim, as the SECOM holdout was spent in `v0.1.0`).
- **Temporal Clustering & Drift**: Fails cluster significantly in time (Wald-Wolfowitz runs test $Z = -4.53, p = 5.77 \times 10^{-6}$, lag-1 autocorrelation $r = 0.1143$), while K-S drift vs Phase I block `D1` climbs from `66.4%` (`D2`) to `83.2%` (`holdout`).
- **Oracle Upper Bound (5-Fold Stratified Random-Split CV)**: Removing temporal drift via stratified random splits across all 1,567 runs proves intrinsic predictive signal exists (`oracle_beats_chance = True`, prevalence `0.0664`): `oracle_xgb_depth3` achieves mean fold PR-AUC `0.2055` (`3.10x` prevalence, 95% CI `[0.1316, 0.2353]`), `oracle_xgb_shallow` `0.1957`, `oracle_rf_balanced` `0.1923`, and `oracle_logreg_l2` `0.1920`.
- **E02 Tie Decomposition**: Quantized step sizes in `f103` (`153` unique values), `f122` (`309`), `f127` (`324`), and `f033` (`330`) caused `f122` to dominate `E02` on `54.5%` of `D5` calibration runs (`77.0%` tied runs in `D5`, `49.5%` tied runs on `holdout`), leaving `0` holdout runs strictly between `budget_10` (`6.0368`) and `budget_5` (`6.2822`).

### 8.2 Protocol `v0.2.0` — Prequential Drift-Adaptive Results (`docs/11_PROTOCOL_V02.md`)
- **Exploratory SECOM (`reports/v0_2/secom_exploratory_prequential.json`)**: Across 7 prequential blocks (`1,371` runs, `75` fails, prevalence `0.0547`), `E02_v2` (Median/MAD + top-3 RMS tie-break + rolling unlabelled threshold calibration) stabilized `budget_5` at `6.20 alarms per 100` (`11 TP / 74 FP`, `14.67%` recall vs `0 TP` in `v0.1.0`), and `E08_v2` reached `0.1049` pooled PR-AUC (`1.92x` prevalence, `18 TP / 125 FP` at budget 5), though `O1_v2` (`0.0843` for `E03_v2` vs `0.1094` target) remained a `MISS` under severe block-to-block regime shifts.
- **One-Shot Confirmatory Benchmark — Semiconductor Wafer Etch (`reports/v0_2/wafer_confirmatory_metrics.json`)**:
  - Dataset: Bobick & Olszewski (2001) `Wafer` (`7,164` wafer runs, `152` inline sensor readings, `762` abnormal/faulty wafers = `10.64%`; `6,268` evaluated runs and `676` fails across $B_2 \dots B_8$, prevalence `0.1078`).
  - **O1_v2 (PASS)**: Primary detector `E08_v2` (Class-Weighted XGBoost) achieved pooled PR-AUC **0.9729** (`9.02x` prevalence, wider 95% CI `[0.9574, 0.9854]`, target $\ge 0.2157$). Unsupervised `E07_v2` (Isolation Forest on passing runs) achieved **0.8692** (`[0.8340, 0.9036]`), `E09_v2` (L2 Logistic Regression) **0.5683** (`[0.5286, 0.6105]`), `E05_v2` (Adaptive PCA $SPE$) **0.3014** (`[0.2763, 0.3279]`), and `E02_v2` (Adaptive Robust Individuals) **0.2497** (`[0.2212, 0.2789]`).
  - **O2_v2 (PASS)**: `E08_v2` Recall @ 5% FAR = **0.9970** (`674 / 676` fails, wider 95% CI `[0.9926, 1.0000]`, target $\ge 0.20$).
  - **O3_v2 (PASS)**: `E08_v2` realised **5.55 alarms per 100 runs** at budget 5 (`348` alarms: **344 TP, 4 FP**, precision **98.85%**, recall **50.89%**). All 8 non-trivial detectors (`E02_v2`–`E09_v2`) stayed tightly inside `[4.95, 6.14] alarms per 100 runs`.
  - **O4_v2 (REPORTED)**: Best ML (`E08_v2`: `0.9729`) minus Best SPC (`E05_v2`: `0.3014`) = **+0.6715** (`95% CI [+0.6458, +0.6931]`).

