# YieldGuard implementation tasks

## 1. Tracker rules

- Execute the first unchecked task whose dependencies are met; one task at a time; record evidence.
- T01 onwards needs an explicit go-ahead to implement. Planning approval covered T00 only.
- Required tests use synthetic data. Benchmark runs use the local SECOM download.

## 2. Planning

### T00: Planning files
- [x] Complete T00 (11 October 2026).
- Dependencies: plan approval.
- Scope: create the project folder with `docs/01`-`05`, `tasks/plan.md`, `tasks/todo.md`.
- Acceptance check: the seven files exist and match the approved drafts; no code, environment or data created.

## 3. Data foundation

### T01: Environment, package and repository
- [x] Complete T01 (11 October 2026).
- Dependencies: T00 and go-ahead to implement.
- Files: `pyproject.toml`, `requirements.lock`, `src/yieldguard/__init__.py`, `tests/test_smoke.py`, `.gitignore`, `AGENTS.md`, `docs/00`, `06`-`10`.
- Acceptance check: `python -m pip check` clean; `pytest tests/test_smoke.py -q` passes; `ruff check` clean; versions recorded; nothing installed globally; siblings untouched.

### T02: Acquisition, raw validation, synthetic generator
- [x] Complete T02 (11 October 2026).
- Dependencies: T01.
- Files: `scripts/download_data.py`, `src/yieldguard/data.py`, `synthetic.py`, `tests/test_data.py`, `tests/test_synthetic.py`.
- Acceptance check: manifest holds URL, hashes, licence, DOI; loader returns 1,567 x 590 and 104 fails; timestamp format, range and ties recorded in `reports/data_quality.json`; tests reject wrong field count, bad label, unparseable timestamp, row-count mismatch and zip path traversal; synthetic files load through the same loader.

### T03: Chronological split and sealed labels
- [x] Complete T03 (11 October 2026).
- Dependencies: T02.
- Files: `src/yieldguard/splits.py`, `cli.py`, `configs/default.yaml`, `tests/test_splits.py`.
- Acceptance check: manifest is identical when labels are permuted; blocks are contiguous and time-ordered; no timestamp straddles a boundary; holdout is the last 392 runs (or the recorded tie-adjusted count); a test proves training loaders raise on `labels_holdout.parquet`.

### T04: Train-only cleaning
- [x] Complete T04 (11 October 2026).
- Dependencies: T03.
- Files: `src/yieldguard/quality.py`, `tests/test_quality.py`.
- Acceptance check: known fixture drops exactly the expected high-missing, constant and collinear columns; group map correct; changing values in later blocks leaves fitted lists, medians and scaling unchanged; save and reload preserves column order.

### Checkpoint C1
- [x] Leakage checks pass; split frozen and hashed (11 October 2026).

## 4. Evaluation and SPC baselines

### T05: Metrics, alarm policy, bootstrap
- [x] Complete T05 (11 October 2026).
- Dependencies: T04.
- Files: `src/yieldguard/evaluate.py`, `policy.py`, `tests/test_metrics.py`, `tests/test_policy.py`.
- Acceptance check: PR-AUC and recall at fixed false-alarm rate match hand-computed arrays; budget threshold yields the expected alarm count including ties and the strict `>` boundary; bootstrap is seeded and reproducible; undefined ratios return null with a reason; targets O1-O7 written into config before any detector exists.

### T06: Univariate SPC
- [x] Complete T06 (11 October 2026).
- Dependencies: T05.
- Files: `src/yieldguard/spc.py`, `tests/test_spc.py`.
- Acceptance check: EWMA values and limits match a hand-computed series; each run rule fires on its constructed sequence and not on its near miss; simulated in-control run length for lambda 0.2, L 3 agrees with the published value within simulation error; missing value leaves EWMA unchanged; sensor selection unchanged when later blocks are altered.

### T07: Multivariate SPC
- [x] Complete T07 (11 October 2026).
- Dependencies: T05.
- Files: `src/yieldguard/mspc.py`, `tests/test_mspc.py`.
- Acceptance check: T-squared and SPE match direct linear-algebra calculation on a small matrix; SPE contributions sum to SPE and T-squared contributions sum to T-squared; a synthetic single-sensor fault raises SPE or T-squared and ranks that sensor first; PCA fitted on passing training runs only (test with altered failing runs).

### Checkpoint C2
- [x] SPC maths verified; development fold scores for E01-E06 written (11 October 2026).

## 5. ML comparison and decision

### T08: Isolation Forest
- [x] Complete T08 (11 October 2026).
- Dependencies: T05.
- Files: `src/yieldguard/anomaly.py`, `tests/test_anomaly.py`.
- Acceptance check: fitted on passing training runs only; higher score for a synthetic outlier; deterministic with seed; fold scores written.

### T09: Class-weighted XGBoost
- [x] Complete T09 (11 October 2026).
- Dependencies: T05.
- Files: `src/yieldguard/train.py`, `tests/test_train.py`, `tests/test_selection.py`.
- Acceptance check: configuration list (<= 20) written before fitting; `scale_pos_weight` computed per training fold; folds are forward-only; all fold metrics kept in `reports/model_comparison.csv`; MLflow runs carry data/config/split hashes; a test proves D5 and holdout cannot affect the choice of configuration.

### T10: Comparison, primary detector, thresholds
- [x] Complete T10 (11 October 2026).
- Dependencies: T06, T07, T08, T09.
- Files: `src/yieldguard/evaluate.py`, `policy.py`, `reports/validation_metrics.json`.
- Acceptance check: table of E01-E08 with fold PR-AUC, intervals and fail counts; primary detector chosen by the written rule with the rule's inputs shown; thresholds for budgets 2, 5, 10 stored; decision logged.

### Checkpoint C3
- [x] Protocol frozen; holdout labels unopened (verified by file-access test and log, 11 October 2026).

## 6. Explanations and frozen evaluation

### T11: Sensor ranking
- [x] Complete T11 (11 October 2026).
- Dependencies: T10.
- Files: `src/yieldguard/explain.py`, `tests/test_explain.py`.
- Acceptance check: SHAP additivity within 1e-4; synthetic single-sensor 5-sigma fault in top 3 MSPC contributions in >= 90% of cases (O5 verdict recorded either way); group members attached; output names map to raw sensor names.

### T12: Bundle, freeze, holdout evaluation
- [ ] Complete T12.
- Dependencies: T11.
- Files: `src/yieldguard/artifacts.py`, `predict.py`, `tests/test_artifacts.py`, `tests/test_holdout.py`.
- Acceptance check: bundle round-trips with checksum and schema checks; tampered file refused; predictions hashed before labels open; `reports/holdout_metrics.json` holds all metrics, intervals and O1-O4 verdicts; second run of the command refuses to overwrite.

### Checkpoint C4
- [ ] One holdout record; all verdicts visible, including misses.

## 7. Inference surfaces

### T13: CLI batch scoring
- [ ] Complete T13.
- Dependencies: T12.
- Files: `src/yieldguard/cli.py`, `tests/test_batch.py`.
- Acceptance check: `score` on a CSV returns one row per run in time order with all scores and flags; invalid run fails the whole file with a named error; `explain` writes rankings for flagged runs.

### T14: FastAPI
- [ ] Complete T14.
- Dependencies: T13.
- Files: `api/main.py`, `api/schemas.py`, `tests/test_api.py`.
- Acceptance check: TestClient cases for valid run, nulls, 589 and 591 values, string value, infinity, unknown field, oversized body and missing bundle (503); `/score` equals CLI within 1e-9 with identical flags; `/spc` equals `spc.py` on the same sequence.

### T15: Streamlit dashboard
- [ ] Complete T15.
- Dependencies: T14.
- Files: `app/dashboard.py`, `tests/test_dashboard.py`.
- Acceptance check: `streamlit.testing.v1.AppTest` runs all four views on synthetic data without exceptions; manual check in the browser on the real bundle; chart values equal package outputs for one sensor and one run; synthetic mode is labelled on screen.

## 8. Operations and handoff

### T16: Drift check
- [ ] Complete T16.
- Dependencies: T13.
- Files: `src/yieldguard/monitoring.py`, `tests/test_monitoring.py`.
- Acceptance check: three controls pass; reference-versus-holdout HTML and JSON written with sample sizes, method and missing-rate table.

### T17: Docker
- [ ] Complete T17.
- Dependencies: T14.
- Files: `Dockerfile`, `.dockerignore`, `scripts/smoke_api.py`.
- Acceptance check: image builds from the lock; runs as nonroot; `/ready` returns 200; smoke script gets a valid `/score` response; no raw data or MLflow store in the image. If Docker is unavailable locally, marked unverified, not passed.

### T18: CI and runtime
- [ ] Complete T18.
- Dependencies: T17.
- Files: `.github/workflows/ci.yml`, `scripts/benchmark_runtime.py`, `reports/runtime.json`.
- Acceptance check: `test` job (ruff, pytest on synthetic data, no download) and `docker-smoke` job (build, readiness, real scoring request) defined; equivalent commands pass locally; latency measured for O7; remote run status stated as unverified until GitHub runs it.

### T19: Model card, README, evidence audit
- [ ] Complete T19.
- Dependencies: T15, T16, T18.
- Files: `docs/MODEL_CARD.md`, `README.md`, `docs/06`, `docs/09`.
- Acceptance check: every number in README and model card traces to a file in `reports/`; fail counts and intervals beside every headline figure; statement that results are offline benchmark results, not yield or cost savings; missed targets listed; dataset attribution and licence present; Markdown links resolve.

### Checkpoint C5
- [ ] Local MVP complete; unverified items marked as such.

## 9. Backlog, outside this MVP

- Regularised logistic baseline.
- Probability calibration.
- Public hosting.
- A second fab dataset.
- Multi-stream charts if tool identifiers become available.
