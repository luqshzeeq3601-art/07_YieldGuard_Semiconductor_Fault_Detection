# 09. Progress log

## 1. Current state

- Phase: T01-T06 completed; proceeding to T07.
- Task tracker: [tasks/todo.md](../tasks/todo.md).

## 2. Session: 11 October 2026, planning (T00)

- Completed T00 planning files (`docs/01`-`05`, `tasks/plan.md`, `tasks/todo.md`).

## 3. Session: 11 October 2026, T01 Environment, package and repository

- Date/time: 2026-10-11 18:52 (Asia/Kuala_Lumpur)
- Task ID completed: T01.
- Changed/created files:
  - `pyproject.toml`, `requirements.lock`, `.gitignore`, `AGENTS.md`
  - `src/yieldguard/__init__.py`, `tests/test_smoke.py`
  - `docs/00_START_HERE.md`, `docs/06_VALIDATION_AND_RELEASE.md`, `docs/07_OPERATIONS_AND_COMMANDS.md`, `docs/08_DECISIONS_LOG.md`, `docs/09_PROGRESS_LOG.md`, `docs/10_SOURCES.md`
- Environment and pinned versions recorded:
  - Python: `3.11.9` (`C:\Users\ZeeqRyz\AppData\Local\Programs\Python\Python311\python.exe` -> `.venv`)
  - Git: `2.54.0.windows.1` (local `git init` only; no remote configured)
  - Docker CLI: `29.6.1` (daemon currently stopped on Windows host; noted for T17)
  - Key package pins in `requirements.lock`: `numpy==1.26.4`, `pandas==2.3.3`, `pyarrow==25.0.1`, `scipy==1.17.1`, `scikit-learn==1.9.1`, `xgboost==3.2.0`, `shap==0.49.1`, `fastapi==0.142.2`, `uvicorn==0.54.0`, `streamlit==1.65.0`, `plotly==5.24.1`, `mlflow==3.16.1`, `evidently==0.7.23`, `pytest==9.1.1`, `ruff==0.16.10`.
- Commands and verification:
  - `.\.venv\Scripts\python.exe -m pip check` -> `No broken requirements found.`
  - `.\.venv\Scripts\python.exe -m pytest tests/test_smoke.py -q` -> `2 passed`
  - `.\.venv\Scripts\python.exe -m ruff check src tests` -> `All checks passed!`
- Next ready task: **T02** (Acquisition, raw validation, synthetic generator).

## 4. Session: 11 October 2026, T02 Acquisition, raw validation, synthetic generator

- Date/time: 2026-10-11 18:57 (Asia/Kuala_Lumpur)
- Task ID completed: T02.
- Changed/created files:
  - `scripts/download_data.py`, `src/yieldguard/data.py`, `src/yieldguard/synthetic.py`
  - `tests/test_data.py`, `tests/test_synthetic.py`
  - `data/source_manifest.json`, `reports/data_quality.json`
- Observed UCI SECOM provenance and raw findings:
  - Source URL: `https://archive.ics.uci.edu/static/public/179/secom.zip` (DOI `10.24432/C54305`, CC BY 4.0, McCann & Johnston 2008).
  - Pinned SHA-256 hashes:
    - `secom.zip`: `eea568baf3c2229096d7d294cf0b096b5502bd96d92c0b80a65b84714059be8e`
    - `secom.data`: `20f0e7ee434f7dcbae0eea9ffff009a2b57f42d6b0dc9a5bd4f00782c0a3374c`
    - `secom_labels.data`: `126884cf453705c9e61a903fe906f0665a3b45ce3639e621edc5c93c89627e03`
    - `secom.names`: `6d91b0b46cdee03064ee3e3112f937c1b3f7fcd9933575794ec07974e6f1ea59`
  - Shape & labels: 1,567 runs x 590 sensors, 104 total fails.
  - Timestamp findings: format `"%d/%m/%Y %H:%M:%S"`, range `2008-07-19T11:55:00` to `2008-10-17T06:07:00`, already sorted in file order (`0` inversions), 1,534 unique timestamps, 33 duplicate-timestamp runs across 32 tie groups (max tie group size 3).
  - Raw missingness: overall missing fraction `4.5375%`, 538 columns with any missing values, 28 columns > 50% missing, 116 single-value columns across the raw dataset.
- Commands and verification:
  - `.\.venv\Scripts\python.exe scripts/download_data.py --output data/raw/secom --manifest data/source_manifest.json`
  - `.\.venv\Scripts\python.exe -m pytest tests/test_data.py tests/test_synthetic.py -q` -> `11 passed`
  - `.\.venv\Scripts\python.exe -m ruff check src scripts tests` -> `All checks passed!`
- Next ready task: **T03** (Chronological split and sealed labels).

## 5. Session: 11 October 2026, T03 Chronological split and sealed labels

- Date/time: 2026-10-11 19:01 (Asia/Kuala_Lumpur)
- Task ID completed: T03.
- Changed/created files:
  - `src/yieldguard/config.py`, `src/yieldguard/splits.py`, `src/yieldguard/cli.py`, `configs/default.yaml`
  - `tests/test_splits.py`
  - `data/processed/split_manifest.json`, `data/processed/runs.parquet`, `data/processed/labels_dev.parquet`, `data/processed/labels_holdout.parquet`
- Observed split and development block findings:
  - `split_manifest.json` SHA-256: `3ff50c3c1b4af28c9dbcb08dd7541f23f570ba37c40351ae9533e248dc0f087d`
  - Development runs: `1,175` across 5 contiguous blocks `D1`-`D5` of `235` runs each (`tie_shift = 0` on all boundaries).
  - Holdout runs: `392` (`data/processed/labels_holdout.parquet` sealed and unreadable by `load_development_labels`).
  - Development block fail rates: `D1` 32/235 (`13.62%`), `D2` 26/235 (`11.06%`), `D3` 8/235 (`3.40%`), `D4` 10/235 (`4.26%`), `D5` 4/235 (`1.70%`), total development fails = 80/1,175 (`6.81%`).
- Commands and verification:
  - `.\.venv\Scripts\python.exe -m yieldguard.cli prepare --config configs/default.yaml`
  - `.\.venv\Scripts\python.exe -m pytest tests/test_splits.py -q` -> `5 passed`
  - `.\.venv\Scripts\python.exe -m ruff check src scripts tests` -> `All checks passed!`
- Next ready task: **T04** (Train-only cleaning).

## 6. Session: 11 October 2026, T04 Train-only cleaning and Checkpoint C1

- Date/time: 2026-10-11 19:03 (Asia/Kuala_Lumpur)
- Task ID completed: T04, Checkpoint C1.
- Changed/created files:
  - `src/yieldguard/quality.py`, `tests/test_quality.py`, `reports/data_quality.json`
- Observed per-fold pruning counts on UCI SECOM (`reports/data_quality.json`):
  - `fold_1` (`D1`-`D2`, 470 train runs, 412 passing): dropped 24 high-missing, 126 constant, 176 collinear -> `264` retained columns.
  - `fold_2` (`D1`-`D3`, 705 train runs, 639 passing): dropped 24 high-missing, 126 constant, 171 collinear -> `269` retained columns.
  - `fold_3` (`D1`-`D4`, 940 train runs, 864 passing): dropped 24 high-missing, 122 constant, 170 collinear -> `274` retained columns.
- Commands and verification:
  - `.\.venv\Scripts\python.exe -m yieldguard.cli prepare --config configs/default.yaml`
  - `.\.venv\Scripts\python.exe -m pytest tests/test_quality.py -q` -> `4 passed`
  - `.\.venv\Scripts\python.exe -m ruff check src scripts tests` -> `All checks passed!`
- Checkpoint C1 status: **PASS** (split is timestamp-only and frozen with SHA-256 `3ff50c3c...`; later blocks cannot alter fitted cleaning; holdout labels are sealed and unreadable by training code).
- Next ready task: **T05** (Metrics, alarm policy, bootstrap).

## 7. Session: 11 October 2026, T05 Metrics, alarm policy, bootstrap and T06 Univariate SPC

- Date/time: 2026-10-11 19:09 (Asia/Kuala_Lumpur)
- Task IDs completed: T05, T06.
- Changed/created files:
  - `src/yieldguard/policy.py`, `src/yieldguard/evaluate.py`, `src/yieldguard/spc.py`
  - `tests/test_metrics.py`, `tests/test_policy.py`, `tests/test_spc.py`
- Verification evidence:
  - T05: Hand-computed PR-AUC (`5/6`) and recall at fixed FAR (`0.75` at 20% FAR) verified; `1 - B/100` linear quantile thresholds with strict `>` rule and boundary ties verified; 2,000 seeded stratified, paired, and moving-block bootstrap CIs verified; undefined ratios return `None` with explicit reasons; predeclared targets O1-O7 verified in `configs/default.yaml`.
  - T06: Hand-computed EWMA series and time-varying limits verified; all 4 Western Electric / Nelson run rules verified on firing and near-miss sequences; simulated in-control ARL0 for $\lambda=0.2, L=3.0$ verified within published range (`410`-`550`); missing value leaves EWMA unchanged; rank-biserial sensor selection with Benjamini-Hochberg adjusted p-values invariant to later-block perturbations.
- Commands and verification:
  - `.\.venv\Scripts\python.exe -m pytest tests/test_metrics.py tests/test_policy.py tests/test_spc.py -q` -> `13 passed`
  - `.\.venv\Scripts\python.exe -m ruff check src scripts tests` -> `All checks passed!`
- Next ready task: **T07** (Multivariate SPC).

## 8. Session: 11 October 2026, T07 Multivariate SPC

- Date/time: 2026-10-11 19:13 (Asia/Kuala_Lumpur)
- Task ID completed: T07.
- Changed/created files:
  - `src/yieldguard/mspc.py`
  - `tests/test_mspc.py`
- Verification evidence:
  - PCA fitted strictly on passing (`label_fail == 0`) training runs with one declared 99.9% Phase I trimming pass (`n_components` retaining $\ge 80\%$ variance, capped at 50).
  - Hotelling $T^2$ and SPE match direct numpy SVD calculation within `1e-10`; SPE contributions (`e_j^2`) sum to SPE and $T^2$ direct inner-product contributions (`x_c * ((T / lambda) @ P.T)`) sum to $T^2$.
  - Synthetic single-sensor fault raises both SPE and $T^2$ and ranks the injected sensor #1 in both SPE and $T^2$ contributions.
  - Corrupting failing training runs (`label_fail == 1`) leaves the fitted PCA model and 99% limits identical.
- Commands and verification:
  - `.\.venv\Scripts\python.exe -m pytest tests/test_mspc.py -q` -> `3 passed`
  - `.\.venv\Scripts\python.exe -m ruff check src scripts tests` -> `All checks passed!`
- Next ready task: **T08** (Isolation Forest).

## 9. Session: 11 October 2026, T08 Isolation Forest, T09 Class-weighted XGBoost, T10 Comparison & Thresholds, T11 Sensor Ranking, and Checkpoints C2 & C3

- Date/time: 2026-10-11 19:28 (Asia/Kuala_Lumpur)
- Task IDs completed: T08, T09, T10, T11, Checkpoint C2, Checkpoint C3.
- Changed/created files:
  - `src/yieldguard/anomaly.py`, `src/yieldguard/train.py`, `src/yieldguard/explain.py`
  - `tests/test_anomaly.py`, `tests/test_train.py`, `tests/test_selection.py`, `tests/test_explain.py`
  - `models/staging/predeclared_xgb_configs.json`, `models/staging/dev_oof_scores.parquet`, `models/staging/quality.joblib`, `models/staging/spc.json`, `models/staging/mspc.joblib`, `models/staging/iforest.joblib`, `models/staging/xgb_model.json`, `models/staging/xgb_metadata.json`, `models/staging/monitoring_reference.parquet`, `models/staging/training_summary.json`, `models/staging/policy.json`, `models/staging/spc_textbook_diagnostics.json`
  - `reports/model_comparison.csv`, `reports/validation_metrics.json`, `reports/explanations/o5_synthetic_verification.json`, `reports/explanations/sensor_rankings.json`
  - `docs/08_DECISIONS_LOG.md`, `docs/09_PROGRESS_LOG.md`, `tasks/todo.md`
- Verification evidence:
  - **T08 (Isolation Forest):** Fitted strictly on passing (`label_fail == 0`) training runs (`200` trees, `max_samples='auto'`, `seed=42`); synthetic outlier ranked above normal runs; deterministic under seed; out-of-fold scores (`E07`) written to `models/staging/dev_oof_scores.parquet`.
  - **T09 (Class-weighted XGBoost):** 16 predeclared configurations (`xgb_c01`..`xgb_c16`, $\le 20$ budget) saved to `models/staging/predeclared_xgb_configs.json` before fitting; dynamic per-fold `scale_pos_weight = n_neg / n_pos` (`11.3684` on `fold_3`); inner-fold (`fold_1`, `fold_2`) 1-SE selection rule selected `xgb_c01` (`max_depth=1, n_estimators=100, learning_rate=0.03, subsample=0.8, colsample_bytree=0.5, min_child_weight=5`, inner mean PR-AUC `0.25735`); all fold metrics for E01-E08, 16 XGBoost candidates, and 5 declared diagnostics written to `reports/model_comparison.csv`; MLflow run `40621e0f400c4fb0ab7e58875b916797` logged with `data_sha256`, `split_sha256`, and `config_sha256`; test in `tests/test_selection.py` proves `D5` and `holdout` cannot alter configuration selection.
  - **Checkpoint C2:** All SPC maths verified against hand-computed fixtures and development fold scores for E01-E06 written to `models/staging/dev_oof_scores.parquet` and `reports/model_comparison.csv`.
  - **T10 (Comparison, primary detector, thresholds):** Across forward folds `D3-D5` (`705` runs, `22` fails: 8 in `D3`, 10 in `D4`, 4 in `D5`, prevalence `0.031206`), best ML detector is `E08` (mean fold PR-AUC `0.179057`, 95% stratified CI `[0.091935, 0.319316]`) and best SPC detector is `E02` (Individuals chart max $|z|$, mean fold PR-AUC `0.115277`, 95% stratified CI `[0.048937, 0.254685]`). Paired within-fold bootstrap difference (`E08 - E02`) is `+0.063780` with 95% CI `[-0.089856, +0.216088]`, which contains zero (`ci_contains_zero = True`). Per Rule 4, **`E02` (Individuals chart max $|z|$)** is selected as primary detector. Thresholds calibrated on `D5` (`235` runs, `4` fails) for budgets `2`, `5` (primary), and `10` alarms per 100 runs are frozen in `models/staging/policy.json`.
  - **T11 (Sensor ranking):** SHAP TreeExplainer raw-margin additivity verified (`max_shap_additivity_error = 2.03e-6 <= 1e-4`); synthetic single-sensor $5\sigma$ fault ranked in top-3 MSPC contributions in `100%` (`100/100 >= 90%`) of trials (`top1_mspc_hit_rate = 1.0`, `top3_spe_hit_rate = 0.96`), recording **O5 = PASS** in `reports/explanations/o5_synthetic_verification.json`; collinear group members and raw sensor names (`f000`-`f589`) verified in `reports/explanations/sensor_rankings.json`.
  - **Checkpoint C3:** Protocol frozen; `data/processed/labels_holdout.parquet` unopened (verified by file-access guard test in `tests/test_selection.py` and `holdout_labels_opened: false` in `reports/validation_metrics.json`). Stopping for user approval before T12.
- Commands and verification:
  - `.\.venv\Scripts\python.exe -m yieldguard.cli train --config configs/default.yaml --experiment all`
  - `.\.venv\Scripts\python.exe -m yieldguard.cli evaluate --config configs/default.yaml --split dev`
  - `.\.venv\Scripts\python.exe -m yieldguard.cli explain --config configs/default.yaml --output reports/explanations/sensor_rankings.json`
  - `.\.venv\Scripts\python.exe -m pytest -q` -> `49 passed`
  - `.\.venv\Scripts\python.exe -m ruff check src scripts tests` -> `All checks passed!`
- Next ready task: **T12** (Bundle, freeze, one-shot holdout evaluation) — **awaiting user approval at Checkpoint C3**.

## 10. Session: 11 October 2026, T12–T19, Checkpoints C4 & C5 (Full MVP Complete)

- Date/time: 2026-10-11 20:45 (Asia/Kuala_Lumpur)
- Task IDs completed: T12, Checkpoint C4, T13, T14, T15, T16, T17, T18, T19, Checkpoint C5.
- Changed/created files:
  - `src/yieldguard/artifacts.py`, `src/yieldguard/predict.py`, `src/yieldguard/monitoring.py`
  - `api/schemas.py`, `api/main.py`, `app/dashboard.py`
  - `scripts/smoke_api.py`, `scripts/benchmark_runtime.py`, `.github/workflows/ci.yml`, `Dockerfile`, `.dockerignore`, `LICENSE`, `README.md`, `docs/MODEL_CARD.md`
  - `tests/test_artifacts.py`, `tests/test_holdout.py`, `tests/test_batch.py`, `tests/test_api.py`, `tests/test_dashboard.py`, `tests/test_monitoring.py`
  - `reports/holdout_predictions.csv`, `reports/holdout_metrics.json`, `reports/runtime.json`, `reports/drift/`
- Verification evidence:
  - **T12 (Bundle freeze & holdout evaluation):**
    - Frozen release bundle `models/v0.1.0/` created with SHA-256 verification and tamper refusal.
    - Holdout predictions written and hashed (`reports/holdout_predictions.csv`, SHA-256: `678f6268c4b3952a30d4e17273a039649dafd77229310d539a61b8f2201ee36a`) before opening `data/processed/labels_holdout.parquet`.
    - Single-shot holdout evaluation executed across 392 runs (24 fails, prevalence 6.12%):
      - **Target O1**: PR-AUC = `0.0724`, 95% stratified CI `[0.0559, 0.1173]`, 95% moving-block CI `[0.0384, 0.1514]`. Verdict: **MISS** (below $2\times$ prevalence 0.1224).
      - **Target O2**: Recall @ 5 FAR = `0.0417` (1 TP / 24 fails), 95% wider CI `[0.0000, 0.1765]`. Verdict: **MISS** (below 0.20; chance recall is 0.05).
      - **Target O3**: Realised alarms at budget 5 = `3.06 per 100` (12 alarms on 392 runs, 0 TP, 12 FP). Verdict: **PASS** (within $[2.0, 8.0]$ per 100).
      - **Target O4**: Paired ML − SPC difference (E08 − E02) = `+0.0153`, 95% stratified CI `[-0.0308, +0.0910]`, 95% moving-block CI `[-0.0470, +0.0839]`. Verdict: **REPORTED** (confidence interval spans zero; no statistically significant ML advantage over SPC).
    - Overwrite refusal verified; holdout labels cannot be reopened for re-tuning.
  - **Checkpoint C4:** Freeze manifest and one-shot holdout metrics recorded in `reports/holdout_metrics.json`.
  - **T13 (CLI batch scoring):** `score_csv_file` handles column checks, non-numeric validation, and chronological ordering. Verified via `tests/test_batch.py` (3 tests pass).
  - **T14 (FastAPI service):** Implemented `/health`, `/ready`, `/model-info`, `/score`, `/spc` with 256 KiB and 2 MiB body limits, 503 missing bundle handling, and clean JSON errors. Verified via `tests/test_api.py` (12 tests pass).
  - **T15 (Streamlit dashboard):** Implemented 4 views (EWMA + Western Electric run rules, MSPC $T^2$/$SPE$, flagged drill-down with TreeSHAP/MSPC bars and collinear groups, alarm budget table) with synthetic demo fallback mode. Verified via `tests/test_dashboard.py` (2 tests pass).
  - **T16 (Drift check):** Evidently data drift preset + missing rate table implemented. Three controls (no-change, +2$\sigma$ sensor shifts, insufficient data < 100 runs) passed 100%. Holdout vs reference report written to `reports/drift/`. Verified via `tests/test_monitoring.py` (3 tests pass).
  - **T17 (Docker):** `Dockerfile` (non-root `appuser`, pinned lock), `.dockerignore`, and `scripts/smoke_api.py` implemented and verified against running API server. Local Docker daemon offline; marked unverified locally as required by project protocol.
  - **T18 (CI & runtime):** `.github/workflows/ci.yml` defined with `test` and `docker-smoke` jobs. `scripts/benchmark_runtime.py` executed: Target O6 reproducibility delta = `0.0` (PASS); Target O7 parity delta = `0.0` (PASS); warm score latency p95 = `63.8 ms`. Output saved to `reports/runtime.json`.
  - **T19 (Model card, README, evidence audit):** `docs/MODEL_CARD.md`, `README.md`, `LICENSE`, `docs/06_VALIDATION_AND_RELEASE.md`, and `docs/08_DECISIONS_LOG.md` completed. All numbers trace to `reports/` with fail counts, bootstrap CIs, and explicit offline benchmark disclaimer.
- Commands and verification:
  - `.\.venv\Scripts\python.exe -m pytest -v` -> `74 passed`
  - `.\.venv\Scripts\python.exe -m ruff check src api app scripts tests` -> `All checks passed!`
  - `.\.venv\Scripts\python.exe scripts/smoke_api.py --url http://127.0.0.1:8000` -> `smoke_passed`
- Checkpoint C5 status: **PASS** (Local MVP complete; Docker container build and remote CI marked unverified).

## 11. Session: 11 October 2026, Dashboard UI/UX & Explainability Polish

- Date/time: 2026-10-11 23:52 (Asia/Kuala_Lumpur)
- Task: Polish `app/dashboard.py` for end-user readability, plain-English project onboarding, and multi-detector transparency without changing frozen model outputs or `AppTest` contracts.
- Changed files:
  - `app/dashboard.py`
  - `tests/test_dashboard.py`
  - `docs/09_PROGRESS_LOG.md`
- Key improvements:
  - Added executive header banner, plain-English expandable onboarding guide, and 4-column top KPI summary strip.
  - Added `DETECTOR_CATALOG` (`E01`–`E08` human-readable titles, families, and physical explanations) and sidebar detector quick legend.
  - Enhanced **View 1 (EWMA & Run Rules)** with sensor rank/correlation selector, optional raw reading trace overlay, 4-rule Western Electric violation summary table, and expandable 10-sensor Phase I table.
  - Enhanced **View 2 (MSPC $T^2$ & $SPE$)** with physical explanation cards ($T^2$ in-model distance vs $SPE$ correlation breakdown), logarithmic Y-axis toggle, and red alarm markers.
  - Enhanced **View 3 (Flagged-Run Drill-Down)** with wafer status banner (ground-truth test outcome, block, timestamp, missing sensor count), cross-method consensus box, and horizontal bar charts + tables for MSPC & TreeSHAP sensor attributions.
  - Enhanced **View 4 (Alarm-Budget Performance)** with budget filter radio, Precision/Recall comparison table, grouped bar chart, formatted **Official One-Shot Holdout Scorecard (O1–O7)**, and **Evidently Data Drift Post-Mortem Card** (86.5% drifted columns).
  - Added **View 5 (Project Overview & Architecture)** documenting the chronological split (`D1`–`D5` + `Holdout`), 5-stage train-only cleaning funnel (`590 -> 274` sensors), and full detector architecture table.
- Verification evidence:
  - `.\.venv\Scripts\ruff.exe check app/dashboard.py tests/test_dashboard.py` -> `All checks passed!`
  - `.\.venv\Scripts\pytest.exe -q` -> `74 passed`

## 12. Session: 12 October 2026, Dashboard plain-language pass

- Date/time: 2026-10-12 (Asia/Kuala_Lumpur)
- Task: Make `app/dashboard.py` understandable to readers with no SPC or ML background, without changing model outputs, thresholds or the `AppTest` contracts.
- Changed files:
  - `app/dashboard.py`
  - `docs/04_TECHNICAL_DESIGN.md` (section 8: landing page noted)
  - `docs/09_PROGRESS_LOG.md`
- Changes:
  - Landing page is now **Start here** (the former view 5): the problem, a four-step method, the frozen holdout findings as three verdict cards, a page guide and a 15-term glossary. Technical tables moved into expanders.
  - Sidebar navigation shows plain-language page names with the technical name as a caption. Radio option values are unchanged, so tests still select views by the original strings.
  - Every view opens with a one-paragraph summary and a visible "How to read this page" box; method detail sits in a collapsed "Technical details" expander. Chart titles, legends, axes, table columns and metric labels reworded.
  - Header shortened to one plain sentence; KPI labels no longer truncate; misleading delta arrows removed.
  - Single-sensor view: chart now full width with controls above and supporting numbers below; legends moved under all plots so they no longer overlap titles on narrow screens.
  - Results view states that the comparison table includes runs the detectors learned from, and that the holdout scorecard is the fair test. Scorecard columns reordered so the verdict is visible without horizontal scrolling.
- Corrections to on-screen facts (previously hardcoded, now read from the bundle or reports):
  - Cleaning table showed 440 sensors after the constant-sensor step, "std = 0" and "|r| > 0.98". The bundle records 444, a dominant-value fraction of 0.99 and a collinearity threshold of 0.95.
  - Raw HTML cards displayed LaTeX and backticks literally (`$r_{rb}$`, `$T^2$`); replaced with native bordered containers.
  - Removed scorecard interpretations that asserted causes not measured by the project; the drift card now says "likely reason" and "observed pattern, not proof of cause". Removed the "K-S test" claim (the report records only the Evidently preset).
  - Fixed a stray French word in a tooltip.
- Verification evidence:
  - `.\.venv\Scripts\ruff.exe check src api app scripts tests` -> `All checks passed!`
  - `.\.venv\Scripts\python.exe -m pytest -q` -> `74 passed` (a first run reported 4 transient `PermissionError` setup errors in `test_splits`, `test_synthetic` and `test_train`; not reproduced on two reruns)
  - Manual browser check of all five pages on the real bundle at 1440 px, and Start here and Single-sensor trends at about 800 px.
- Unresolved: O5, O6 and O7 scorecard values are still hardcoded text in the dashboard rather than read from `reports/`.
- Next ready task: none in the MVP tracker; backlog items remain in `tasks/todo.md` section 9.

## 13. Session: 12 October 2026, Phase 1–4 Execution (v0.1.0 Closeout, SECOM Diagnostics & v0.2 Confirmatory Wafer Etch Benchmark)

- Date/time: 2026-10-12 00:49 (Asia/Kuala_Lumpur)
- Tasks completed: `T20`, `T21`, `T22`, `T23`, `T24`, `T25` (plus local Docker verification of `T17`, `T18`, `C5`).
- Changed / added files:
  - `app/dashboard.py` (fixed `E06` additive formula text `T²/τ_T² + SPE/τ_SPE` and clarified `O1`/`O3` `0 TP, 12 FP` holdout scorecard text)
  - `README.md`, `docs/MODEL_CARD.md`, `docs/06_VALIDATION_AND_RELEASE.md`, `reports/runtime.json`
  - `src/yieldguard/diagnostics.py`, `scripts/diagnose_secom.py`, `tests/test_diagnostics.py`, `reports/exploratory/secom_phase2_diagnostics.json`
  - `docs/11_PROTOCOL_V02.md`, `configs/v0_2.yaml`
  - `src/yieldguard/prequential.py`, `src/yieldguard/cli.py`, `tests/test_prequential.py`, `reports/v0_2/secom_exploratory_prequential.json`
  - `src/yieldguard/wafer_data.py`, `scripts/download_wafer_data.py`, `tests/test_wafer_data.py`, `data/wafer_source_manifest.json`, `reports/v0_2/wafer_confirmatory_metrics.json`
  - `docs/08_DECISIONS_LOG.md`, `docs/09_PROGRESS_LOG.md`, `tasks/todo.md`
- Commands & measured evidence:
  1. **Phase 1 (`T20` — `v0.1.0` Closeout & Docker Verification)**:
     - `docker build -t yieldguard:ci .` -> `sha256:0ecc56a4e57677e09a1aea6c4bd4c15c826fc9efb8ca34b347e828016398a1af` (PASS).
     - `docker run -d --name yieldguard-ci -p 127.0.0.1:18000:8000 yieldguard:ci` + `.\.venv\Scripts\python.exe scripts/smoke_api.py --url http://127.0.0.1:18000` -> `"status": "smoke_passed"` (PASS).
     - Audited `README.md` and `docs/MODEL_CARD.md` against `reports/holdout_metrics.json` (`0 TP, 12 FP` at budget 5 for `E02`, full `E01`–`E08` table, `xgb_c01` parameters, `100/100` `O5` trials, `secom.zip` SHA-256).
  2. **Phase 2 (`T21` — Exploratory SECOM Diagnostics)**:
     - `.\.venv\Scripts\python.exe scripts/diagnose_secom.py --config configs/default.yaml --bundle models/v0.1.0 --output reports/exploratory/secom_phase2_diagnostics.json`
     - Fails cluster strongly in time (Wald-Wolfowitz $Z = -4.535, p = 5.77 \times 10^{-6}$, lag-1 autocorrelation $r = 0.1143$), while K-S sensor drift vs `D1` climbs from `66.4%` (`D2`) to `83.2%` (`holdout`).
     - 5-fold stratified random-split oracle upper bound (`oracle_xgb_depth3`) achieved mean fold PR-AUC `0.2055` (`3.10x` prevalence `0.0664`, 95% CI `[0.1316, 0.2353]`), satisfying `oracle_beats_chance = True` (`gate_decision = "PROCEED_SECOM_AND_WAFER"`).
     - `E02` quantization tie analysis showed `f122` (`309` unique values) won the max on `54.5%` of `D5` runs (`77.0%` tied runs in `D5`, `49.5%` on `holdout`), leaving `0` holdout runs strictly between `budget_10` (`6.0368`) and `budget_5` (`6.2822`).
  3. **Phase 3 (`T22` & `T23` — Protocol `v0.2` Prequential Engine & Exploratory SECOM Run)**:
     - `.\.venv\Scripts\python.exe -m yieldguard.cli prequential --config configs/v0_2.yaml --dataset secom --output reports/v0_2/secom_exploratory_prequential.json`
     - On SECOM (`1,371` runs, `75` fails across $B_2 \dots B_8$), `E02_v2` achieved `6.20 alarms per 100` at budget 5 (`11 TP / 74 FP`, `14.67%` recall vs `0 TP` in `v0.1.0`), and `E08_v2` achieved `0.1049` pooled PR-AUC (`1.92x` prevalence), though `O1_v2` (`0.0843` for `E03_v2` vs `0.1094` target) was a `MISS`.
  4. **Phase 4 (`T24` & `T25` — Confirmatory Semiconductor Wafer Etch Benchmark)**:
     - `.\.venv\Scripts\python.exe scripts/download_wafer_data.py` -> `Wafer.zip` (`sha256:06032db8...`, `7,164` runs, `152` features, `762` abnormal/faulty wafers = `10.64%`).
     - `.\.venv\Scripts\python.exe -m yieldguard.cli prequential --config configs/v0_2.yaml --dataset wafer --output reports/v0_2/wafer_confirmatory_metrics.json`
     - Evaluated once across $B_2 \dots B_8$ (`6,268` runs, `676` fails, prevalence `0.1078`):
       - **O1_v2 (PASS)**: Primary `E08_v2` pooled PR-AUC = **0.9729** (`9.02x` prevalence, wider 95% CI `[0.9574, 0.9854]`, target $\ge 0.2157$).
       - **O2_v2 (PASS)**: `E08_v2` Recall @ 5% FAR = **0.9970** (`674 / 676` fails, wider 95% CI `[0.9926, 1.0000]`, target $\ge 0.20$).
       - **O3_v2 (PASS)**: `E08_v2` realised **5.55 alarms per 100** at budget 5 (**344 TP, 4 FP**, precision **98.85%**, recall **50.89%**); all 8 detectors `E02_v2`–`E09_v2` stayed inside `[4.95, 6.14] per 100`.
       - **O4_v2 (REPORTED)**: `E08_v2` (`0.9729`) minus best SPC `E05_v2` (`0.3014`) = **+0.6715** (`95% CI [+0.6458, +0.6931]`).
- Unresolved issues: None (remote GitHub Actions CI workflow remains unverified on GitHub until a remote repository is configured and pushed).
- Next ready task: none in the MVP tracker; backlog items remain in `tasks/todo.md` section 9.

## 14. Session: 12 October 2026, T26 Dashboard Wafer Etch Confirmatory Benchmark View

- Date/time: 2026-10-12 01:05 (Asia/Kuala_Lumpur)
- Task completed: `T26`.
- Changed files:
  - `app/dashboard.py` (added `VIEW_WAFER`, `VIEW_LABELS` navigation entries, `load_wafer_dashboard_data` with synthetic fallback, `render_wafer_etch_view`, KPI strip for `O1_v2`–`O4_v2`, interactive 152-channel sensor trace inspector with baseline overlay, and detector comparison across budgets 2, 5, 10).
  - `tests/test_dashboard.py` (added `6. Semiconductor Wafer Etch (v0.2 Benchmark)` to the AppTest test loop).
  - `tasks/todo.md`, `docs/09_PROGRESS_LOG.md`.
- Verification evidence:
  - `.\.venv\Scripts\python.exe -m ruff check app/dashboard.py tests/test_dashboard.py` -> `All checks passed!`
  - `.\.venv\Scripts\python.exe -m pytest tests/test_dashboard.py -v` -> `2 passed`
  - `.\.venv\Scripts\python.exe -m pytest -q` -> `81 passed`
- Unresolved issues: None.
- Next ready task: All current and backlog dashboard enhancement tasks completed.

