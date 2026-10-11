# 06. Validation and release gates

## 1. Gate structure

| Gate | Required evidence | Outcome |
|---|---|---|
| G1 Data | UCI SECOM source hashes, licence, citation, schema, timestamp format and frozen split manifest | Pass/fail |
| G2 Leakage | Timestamp-only chronological split, train-only cleaning/selection/scaling, sealed `labels_holdout.parquet` | Pass/fail |
| G3 Detection & policy | Predeclared objectives O1-O4 evaluated on frozen holdout once with 2,000 stratified and block bootstrap CIs | Recorded as measured (pass/miss per target) |
| G4 Explanations | Synthetic single-sensor 5-sigma fault in top-3 MSPC contributions >= 90% (O5); SHAP additivity within 1e-4 | Pass/fail |
| G5 Engineering | Reproducibility within 1e-6 (O6), CLI/API `/score` parity within 1e-9, warm `/score` p95 < 200 ms, Streamlit AppTest, Docker and CI (O7) | Pass/fail/unverified |
| G6 Monitoring | No-change control, shifted-sensor (+2 sigma on 5 sensors) control, and insufficient-data (< 100 runs) control | Pass/fail |
| G7 Evidence | Single holdout evaluation record, reports, model card and honest README with fail counts and CIs | Complete/incomplete |

Engineering completion (G1, G2, G4, G5, G6, G7) is judged separately from whether empirical targets O1-O3 pass on the SECOM holdout. Missed targets and negative results remain visible in all reports, the model card and the README.

## 2. Meaningful tests

| Area | Required cases |
|---|---|
| Acquisition & parsing | Wrong field count, bad label, unparseable timestamp, row-count mismatch, Zip Slip path traversal, synthetic loader parity |
| Chronology & splits | Identical split manifest under label permutation, contiguous ordered blocks, timestamp tie boundary rule, training loader refusal on `labels_holdout.parquet` |
| Train-only cleaning | Known fixture drops expected high-missing, constant and collinear columns; group map tracking; perturbing later blocks leaves fitted lists, medians and scaling unchanged; save/reload preserves column order |
| Metrics & policy | Hand-computed PR-AUC and recall at fixed FAR; budget quantile threshold with ties and strict `>` rule; seeded reproducible bootstrap; null with reason on undefined ratios |
| Univariate SPC | Hand-computed EWMA series and time-varying limits; all 4 run rules fire on positive sequences and not on near-misses; in-control ARL simulation for lambda=0.2, L=3; missing value leaves EWMA unchanged; sensor selection invariant to later blocks |
| Multivariate SPC | Known-matrix T-squared and SPE match direct linear algebra; SPE and T-squared contributions sum to total statistics; synthetic single-sensor fault raises SPE/T-squared and ranks sensor first; PCA fitted on passing training runs only |
| Anomaly & XGBoost | Isolation Forest fitted on passing training runs only; deterministic with seed; XGBoost <= 20 configs declared before fit; `scale_pos_weight` per fold; D5 and holdout cannot affect config selection |
| Explainability | SHAP additivity within 1e-4; synthetic 5-sigma fault in top-3 MSPC contributions >= 90%; collinear group members attached |
| Artifacts & holdout | Bundle round-trip with SHA-256 and schema verification; tampered file refused; predictions written and hashed before holdout labels opened; second holdout run refuses to overwrite |
| Batch & API | CLI `score` and `explain` contracts; FastAPI TestClient cases for valid run, nulls, 589/591 values, string, infinity, unknown field, oversized body (413), missing bundle (503), CLI/API parity within 1e-9, `/spc` parity |
| Dashboard | `streamlit.testing.v1.AppTest` runs all four views on synthetic data without exceptions; chart values match package outputs |
| Monitoring | Three drift controls pass; reference vs holdout HTML and JSON written with sample sizes, method and missing-rate table |

All required pytest tests run on synthetic data without requiring the SECOM download.

## 3. Required final evidence

| Artifact | Content |
|---|---|
| `data/source_manifest.json` | UCI URL, download timestamp, SHA-256 hashes, licence, citation |
| `data/processed/split_manifest.json` | Chronological block boundaries D1-D5 and holdout indices, tie shifts, SHA-256 |
| `reports/data_quality.json` | Schema, counts, timestamp format/range/ties, per-fold pruning counts |
| `reports/model_comparison.csv` | Fold metrics for E01-E08 and all bounded XGBoost configurations |
| `reports/validation_metrics.json` | Development fold results, intervals, primary detector choice and frozen thresholds |
| `reports/holdout_predictions.csv` | Scored holdout runs written and hashed before holdout labels are opened |
| `reports/holdout_metrics.json` | One-shot holdout metrics, stratified and moving-block bootstrap CIs, O1-O4 verdicts |
| `reports/explanations/` | Synthetic O5 verification and example flagged-run sensor rankings with group maps |
| `reports/drift/` | Evidently HTML, drift summary JSON, missing-rate change table and control outcomes |
| `reports/runtime.json` | Reproducibility check (O6), warm `/score` latency p95 (O7) and container verification status |
| `docs/MODEL_CARD.md` | Intended use, dataset attribution, limitations, development and holdout metrics, target verdicts |
| `README.md` | Reproduction commands and measured figures linked to `reports/`, with offline benchmark disclaimer |

## 4. Final release gate outcomes (Checkpoint C5)

| Gate | Criterion | Measured Evidence | Gate Outcome |
|---|---|---|---|
| **G1 Data** | UCI SECOM hashes, licence, schema, split manifest | `data/source_manifest.json`, `data/processed/split_manifest.json` | **PASS** |
| **G2 Leakage** | Chronological split, train-only cleaning, sealed holdout labels | `src/yieldguard/splits.py`, `src/yieldguard/evaluate.py` | **PASS** |
| **G3 Detection & policy** | One-shot holdout evaluation; predeclared targets O1-O4 | `reports/holdout_metrics.json` (O1 MISS, O2 MISS, O3 PASS, O4 REPORTED) | **RECORDED (O1/O2 MISS, O3 PASS)** |
| **G4 Explanations** | Synthetic 5-$\sigma$ fault in top-3 $\ge 90\%$; TreeSHAP additivity | `reports/explanations/o5_synthetic_verification.json` (100.0%) | **PASS** |
| **G5 Engineering** | Reproducibility $\le 10^{-6}$ (O6), parity $\le 10^{-9}$ (O7), warm latency, Docker smoke | `reports/runtime.json` (0.0 diff; direct p95: 63.8 ms; `docker build -t yieldguard:ci .` + `smoke_api.py` PASS) | **PASS** |
| **G6 Monitoring** | 3 drift controls pass; reference vs holdout drift HTML/JSON | `reports/drift/controls/control_results.json`, `reports/drift/drift_summary.json` | **PASS** |
| **G7 Evidence** | Single holdout record, model card, README with fail counts and CIs | `reports/`, `docs/MODEL_CARD.md`, `README.md` | **COMPLETE** |
