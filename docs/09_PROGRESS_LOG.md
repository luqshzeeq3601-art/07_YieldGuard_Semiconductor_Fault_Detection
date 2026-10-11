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


