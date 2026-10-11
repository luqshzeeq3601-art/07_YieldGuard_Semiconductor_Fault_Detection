# 08. Decision log

## 1. Initial decisions: 11 October 2026

These defaults come from the planning contract (`docs/01`-`05` and `tasks/plan.md` open questions Q1-Q10).

| ID | Decision | Reason | Revisit when |
|---|---|---|---|
| D001 | Name: YieldGuard; folder `07_YieldGuard_Semiconductor_Fault_Detection` | Matches portfolio structure and semiconductor fault detection scope | User requests another name |
| D002 | UCI SECOM archive (`secom.zip`, DOI 10.24432/C54305, CC BY 4.0) is the sole primary dataset; download on demand with pinned hashes (Q1, Q2) | Real 590-sensor semiconductor run data with clear CC BY 4.0 attribution; avoids Kaggle mirror differences | Dataset unavailable or user requests another source |
| D003 | Required tests and CI run on synthetic SECOM-like data without the UCI download (Q3) | Fast, deterministic, self-contained CI and synthetic ground-truth validation for O5 | Never weaken CI independence |
| D004 | Chronological split by `(timestamp, run_id)`: 75% development (D1-D5 of 235 runs each, forward folds D3-D5) and 25% holdout (392 runs), with tie-shift boundary protection (Q5) | Prevents temporal leakage and preserves ~60+ training fails | New dataset or protocol version |
| D005 | Train-only 5-step cleaning: >50% missingness drop, >=99% constant drop, \|r\| >= 0.95 collinearity drop with group map, median imputation (SPC/PCA/IForest), passing-run scaling (PCA/IForest) | Prevents preprocessing leakage and preserves collinear sensor groups for explanations | New protocol version |
| D006 | Primary alarm budget of 5 alarms per 100 runs (with 2 and 10 also reported), calibrated on block D5 via `1 - B/100` linear quantile and strict `>` rule (Q6) | Reflects fixed engineering review capacity without accuracy/F1 threshold tuning | User specifies different operating capacity |
| D007 | Predeclared targets O1-O7 fixed before modelling (Q7) | Prevents post-hoc target moving; missed targets are reported as findings | Never change after viewing results |
| D008 | SPC detector is chosen as primary if best ML detector's paired bootstrap 95% PR-AUC advantage over best SPC contains zero (Q8) | Prefers simpler, interpretable statistical process control when ML advantage is within small-sample noise | Never change after viewing results |
| D009 | Local Git repository (`git init` only, no remote or push) and project-local Python 3.11 `.venv` (Q9) | Complete project isolation without side effects on host or remotes | User authorizes remote push |
| D010 | Regularised logistic baseline omitted from MVP and kept in backlog (Q10) | Scope control; 8 detectors (E01-E08) already cover univariate SPC, multivariate SPC, unsupervised ML and supervised ML | Post-MVP extension |

## 2. Change entry format

Append rather than erase earlier reasoning:

```text
ID and date:
Question:
Evidence examined:
Decision and alternatives:
Effect on requirements/data/protocol:
Documents changed:
Verification or follow-up task:
```

## 3. Development model selection and Checkpoint C3 freeze: 11 October 2026

- **ID and date:** D011 (11 October 2026)
- **Question:** Which predeclared XGBoost configuration represents `E08` across forward folds without leaking `D5` or `holdout`?
- **Evidence examined:** 16 predeclared configurations (`xgb_c01`..`xgb_c16`) evaluated on inner forward folds `fold_1` (`D1-D2 -> D3`, 8 fails) and `fold_2` (`D1-D3 -> D4`, 10 fails) in `models/staging/training_summary.json` and `reports/model_comparison.csv`.
- **Decision and alternatives:** `xgb_c01` (`max_depth=1, n_estimators=100, learning_rate=0.03, subsample=0.8, colsample_bytree=0.5, min_child_weight=5, seed=42`) achieved the highest mean inner-fold PR-AUC (`0.25735`, SE `0.15732`, 1-SE threshold `0.10003`) and is also the shallowest model with the fewest trees (`depth=1, trees=100`).
- **Effect on requirements/data/protocol:** None; follows predeclared Section 4 Rule 3 of `docs/05_EXPERIMENT_PLAN.md`.
- **Documents changed:** `docs/08_DECISIONS_LOG.md`, `docs/09_PROGRESS_LOG.md`.
- **Verification or follow-up task:** Verified by `tests/test_selection.py` (`D5` and `holdout` cannot alter configuration selection).

- **ID and date:** D012 (11 October 2026)
- **Question:** Which detector is frozen as the primary detector at Checkpoint C3, and what are the frozen alarm-budget thresholds on `D5`?
- **Evidence examined:** `reports/validation_metrics.json` and `models/staging/policy.json` across `D3-D5` (`705` runs, `22` fails: 8 in `D3`, 10 in `D4`, 4 in `D5`). Best ML detector is `E08` (mean fold PR-AUC `0.179057`, 95% stratified CI `[0.091935, 0.319316]`); best SPC detector is `E02` (Individuals chart max $|z|$, mean fold PR-AUC `0.115277`, 95% stratified CI `[0.048937, 0.254685]`, followed closely by `E03` `0.104573` and `E04` Hotelling $T^2$ `0.103357`). Paired within-fold bootstrap difference (`E08 - E02`) is `+0.063780` with 95% CI `[-0.089856, +0.216088]`, which contains zero (`ci_contains_zero = True`).
- **Decision and alternatives:** Per predeclared Selection Rule 4 (`docs/05_EXPERIMENT_PLAN.md` Section 4.4 and Q8), because the 95% paired bootstrap CI of `best_ml - best_spc` contains zero, **`E02` (Individuals chart max $|z|$)** is selected as the primary detector. All 8 detectors (`E01-E08`) and their `D5`-calibrated thresholds for budgets `2`, `5` (primary), and `10` alarms per 100 runs are frozen in `models/staging/policy.json` and will be scored on the holdout in T12 after user approval.
- **Effect on requirements/data/protocol:** Protocol frozen at Checkpoint C3; `data/processed/labels_holdout.parquet` remains unopened.
- **Documents changed:** `docs/08_DECISIONS_LOG.md`, `docs/09_PROGRESS_LOG.md`, `models/staging/policy.json`, `reports/validation_metrics.json`.
- **Verification or follow-up task:** Await user approval at Checkpoint C3 before running T12.

- **ID and date:** D013 (11 October 2026)
- **Question:** How are the 4 pre-T12 audit findings resolved and recorded before opening holdout labels?
- **Evidence examined:** User code review audit notes: (1) repo lacked initial commit and tracked hashes; (2) E02 tied scores on D5 calibration block yield 6 alarms (2.55 per 100 runs) at budget 5 instead of ~12; (3) moving-block bootstrap previously pooled D3-D5 across fold models; (4) PCA 50-component cap binds on training data explaining 61.97% variance.
- **Decision and alternatives:**
  1. Record explicit SHA-256 hashes in git-tracked `reports/staging_manifest_sha256.json`, track `models/staging/policy.json`, and commit to Git before T12.
  2. Keep strict `>` threshold comparison rule as predeclared; record 2.55 alarms per 100 on D5 (and 2.13 per 100 at budget 2) as a known limitation without post-hoc tie alteration.
  3. Recompute moving-block bootstrap strictly within folds: E03 moving-block 95% CI is `[0.0295, 0.2185]` (contains 0.1046 point estimate). Paired within-fold moving-block difference (E08 - E02) is `[-0.1273, +0.2324]`, confirming Rule 4 retains E02 under both bootstrap methods.
  4. Explicitly label PCA diagnostics with actual cumulative variance explained (~62%) and note the 50-component cap.
  5. Enforce strict single-shot evaluation per §10.4 by permanently removing the `allow_overwrite` flag from `run_holdout_evaluation`. Once executed, holdout evaluation output cannot be rewritten.
  6. Enforce strict Target O1 verification per §7.3 by requiring the lower bound of the wider confidence interval (the wider of stratified and moving-block bootstrap) to satisfy the bound, rather than the stratified interval alone. Decided and frozen prior to opening holdout labels.
- **Effect on requirements/data/protocol:** Strengthens protocol tamper-resistance and makes Target O1 evaluation criterion strict and unambiguous before holdout evaluation. Confirms E02 primary selection.
- **Documents changed:** `docs/08_DECISIONS_LOG.md`, `docs/09_PROGRESS_LOG.md`, `docs/05_EXPERIMENT_PLAN.md`, `src/yieldguard/evaluate.py`, `src/yieldguard/train.py`, `reports/staging_manifest_sha256.json`, `.gitignore`.
- **Verification or follow-up task:** T12 holdout evaluation.

## 2026-10-11: D014 - Post-holdout MVP implementation, drift monitoring, and release packaging (T12-T19)

- **Context:** Following user approval at Checkpoint C3, tasks T12-T19 were executed to complete the end-to-end local MVP.
- **Decision:**
  1. Immutable model bundle `models/v0.1.0/` frozen with SHA-256 checksums and schema validation before opening holdout labels.
  2. One-shot holdout evaluation completed and recorded without overwrite: E02 holdout PR-AUC 0.0724 (O1 MISS), recall at 5 FAR 0.0417 (O2 MISS), realised alarm rate at budget 5 is 3.06 per 100 (O3 PASS), paired ML - SPC diff +0.0153 (O4 REPORTED). Missed targets recorded openly without protocol changes.
  3. Batch scoring and explanation CLI implemented for time-ordered CSV processing with collinear group map tracking.
  4. FastAPI service implemented with strict 2 MiB request body limits, 503 missing bundle handling, and TestClient verification.
  5. Streamlit dashboard implemented with 4 views (EWMA + run rules, MSPC T²/SPE, flagged drill-down, alarm budget table) and synthetic demo data fallback.
  6. Evidently data drift monitoring implemented with 100-run batch threshold, missing-rate table, and 3 verification controls (no-change, shifted sensors, insufficient data) passing 100%.
  7. Container runtime (`Dockerfile`, `.dockerignore`, `scripts/smoke_api.py`) and CI workflow (`.github/workflows/ci.yml`) defined; container execution marked unverified locally due to offline host Docker daemon.
  8. Model card (`docs/MODEL_CARD.md`) and `README.md` published with all figures traced to `reports/`, fail counts, bootstrap CIs, and mandatory offline benchmark disclaimer.
- **Effect on requirements/data/protocol:** Satisfies all requirements across PRD and technical design to Checkpoint C5.
- **Documents changed:** `tasks/todo.md`, `docs/06_VALIDATION_AND_RELEASE.md`, `docs/08_DECISIONS_LOG.md`, `docs/09_PROGRESS_LOG.md`, `docs/MODEL_CARD.md`, `README.md`.
- **Verification or follow-up task:** Checkpoint C5 reached.


## 2026-10-12: D015 - Closeout of v0.1.0 Negative Result, Phase 2 Exploratory SECOM Diagnostics, and v0.2 Confirmatory Protocol on Semiconductor Wafer Etch

- **ID and date:** D015 (12 October 2026)
- **Question:** How should YieldGuard address the negative `v0.1.0` holdout result on SECOM (where no detector beat chance and 86.5% of sensors drifted) without violating single-shot holdout integrity?
- **Evidence examined:** `reports/holdout_metrics.json` (all 8 detectors' 95% PR-AUC CIs include or fall below 6.12% prevalence; E02's 12 budget-5 alarms were 0 TP, 12 FP), `reports/drift/drift_summary.json` (237/274 drifted sensors), and user-approved 4-phase plan.
- **Decision and alternatives:**
  1. **Preserve `v0.1.0` as the immutable negative result on SECOM**: Audit `README.md`, `docs/MODEL_CARD.md`, and `app/dashboard.py` so the headline explicitly states *"No detector beat chance on the chronological SECOM holdout"* and reports `0 TP, 12 FP` alongside O3's alarm-rate pass.
  2. **Phase 2 Exploratory Diagnostics on SECOM**: Because `labels_holdout.parquet` has been opened, all subsequent SECOM analyses are labelled `exploratory_post_holdout` (`reports/exploratory/secom_phase2_diagnostics.json`). Quantify per-block sensor drift and fail clustering, decompose E02's tied scores on `D5` and `holdout`, and run a 5-fold stratified random-split **oracle upper bound** across all 1,567 runs.
  3. **Stopping Rule on SECOM**: Per user authorization, if the Phase 2 oracle upper bound on SECOM is near chance even with temporal leakage, stop SECOM modelling after Phase 2 (concluding SECOM sensors lack predictive signal for the end-of-line label) and apply `v0.2` directly to the second dataset. If the oracle beats chance, run `v0.2` on SECOM (labelled exploratory) as well as confirmatory evaluation on the second dataset.
  4. **In-Repo `v0.2` Prequential Drift-Adaptive Protocol**: Keep `v0.2` in this repository (`docs/11_PROTOCOL_V02.md`, `configs/v0_2.yaml`, `models/v0.2.0/`, `reports/v0_2/`) with prequential rolling-origin evaluation, rolling-window SPC/PCA baselines, unlabelled recent-window threshold recalibration, robust median/MAD scaling with deterministic top-3 RMS tie-breaking, and `E09` L2-regularised logistic regression.
  5. **Confirmatory Second Dataset**: Use the **Semiconductor Wafer Etch** dataset (Olszewski 2001 / UCR `Wafer`: 7,164 wafer runs, 152 time-step sensor readings per wafer, ~10.64% abnormal/faulty wafers) for one-shot confirmatory evaluation of the frozen `v0.2` protocol.
- **Effect on requirements/data/protocol:** Leaves `v0.1.0` frozen and unmodified; opens versioned `v0.2` protocol and adds `Wafer` as the user-authorized confirmatory dataset.
- **Documents changed:** `docs/08_DECISIONS_LOG.md`, `tasks/todo.md`, `README.md`, `docs/MODEL_CARD.md`, `docs/11_PROTOCOL_V02.md`, `configs/v0_2.yaml`.
- **Verification or follow-up task:** Execute T20–T25.

## 2026-10-12: D016 - Phase 2 Oracle Gate Outcome and Protocol v0.2 Confirmatory Evaluation Results (T20-T25)

- **ID and date:** D016 (12 October 2026)
- **Context:** Execution of Tasks T20–T25 across Phase 1 (`v0.1.0` closeout and Docker verification), Phase 2 (SECOM exploratory root-cause diagnostics), Phase 3 (`v0.2` prequential engine), and Phase 4 (confirmatory Wafer Etch evaluation).
- **Measured Evidence & Outcomes:**
  1. **Phase 1 Closeout (`T20`)**: Local Docker image `yieldguard:ci` (`sha256:0ecc56a4...`) built and verified via `scripts/smoke_api.py` (`status="verified_locally"` in `reports/runtime.json`). `README.md`, `docs/MODEL_CARD.md`, and `app/dashboard.py` audited with zero discrepancies against `reports/holdout_metrics.json`.
  2. **Phase 2 Exploratory Diagnostics on SECOM (`T21`)**:
     - Per-block K-S drift vs `D1` rises from `66.4%` (`D2`) to `83.2%` (`holdout`), while fails cluster strongly in time (Wald-Wolfowitz $Z = -4.53, p = 5.77 \times 10^{-6}$).
     - 5-fold stratified random-split oracle upper bound (`oracle_xgb_depth3`) achieved mean fold PR-AUC `0.2055` (`3.10x` prevalence `0.0664`, 95% CI `[0.1316, 0.2353]`), satisfying `oracle_beats_chance = True` (`gate_decision = "PROCEED_SECOM_AND_WAFER"`).
     - `E02` tie analysis proved sensor `f122` (`309` unique values) won the max on `54.5%` of `D5` runs (`77.0%` tied runs in `D5`, `49.5%` on `holdout`), leaving `0` holdout runs strictly between `budget_10` (`6.0368`) and `budget_5` (`6.2822`).
  3. **Phase 3 Exploratory `v0.2` on SECOM (`T23`)**: Under prequential rolling adaptation (`1,371` runs, `75` fails across $B_2 \dots B_8$), `E02_v2` (Median/MAD + top-3 RMS tie-break + rolling unlabelled threshold calibration) stabilized `budget_5` at `6.20 per 100` (`11 TP / 74 FP`, `14.67%` recall vs `0 TP` in `v0.1.0`), and `E08_v2` achieved `0.1049` pooled PR-AUC (`1.92x` prevalence), though `O1_v2` (`0.0843` for `E03_v2` vs `0.1094` target) remained a `MISS`.
  4. **Phase 4 One-Shot Confirmatory `v0.2` on Semiconductor Wafer Etch (`T24`)**:
     - Downloaded and verified `Wafer.zip` (`sha256:06032db8...`, `7,164` runs, `152` features, `762` abnormal/faulty wafers = `10.64%`).
     - Evaluated once across $B_2 \dots B_8$ (`6,268` runs, `676` fails, prevalence `0.1078`, predictions SHA-256 `751c522e...`):
       - **O1_v2 (PASS)**: Primary `E08_v2` pooled PR-AUC = **0.9729** (`9.02x` prevalence, wider 95% CI `[0.9574, 0.9854]`, target $\ge 0.2157$).
       - **O2_v2 (PASS)**: `E08_v2` Recall @ 5% FAR = **0.9970** (`674 / 676` fails, wider 95% CI `[0.9926, 1.0000]`, target $\ge 0.20$).
       - **O3_v2 (PASS)**: `E08_v2` realised **5.55 alarms per 100 runs** at budget 5 (`348` alarms: **344 TP, 4 FP**, precision **98.85%**, recall **50.89%**); all 8 detectors `E02_v2`–`E09_v2` stayed within `[4.95, 6.14] per 100`.
       - **O4_v2 (REPORTED)**: `E08_v2` (`0.9729`) minus best SPC `E05_v2` (`0.3014`) = **+0.6715** (`95% CI [+0.6458, +0.6931]`).
- **Documents changed:** `reports/exploratory/secom_phase2_diagnostics.json`, `reports/v0_2/secom_exploratory_prequential.json`, `reports/v0_2/wafer_confirmatory_metrics.json`, `README.md`, `docs/MODEL_CARD.md`, `docs/08_DECISIONS_LOG.md`, `docs/09_PROGRESS_LOG.md`, `tasks/todo.md`.
- **Verification or follow-up task:** All tasks T01–T25 complete.

## 2026-10-12: D017 - Post-T26 Portfolio-Ready Corrections Pass for Wafer Benchmark View (T27)

- **ID and date:** D017 (12 October 2026)
- **Question:** How should View 6 and related project documentation be refined to correct data labeling inaccuracies, eliminate hard-coded fallbacks, and disclose synthetic timestamps?
- **Evidence examined:** User code review audit notes: (1) data is 1 sensor trace of 152 time steps per wafer (not 152 RF/chemical sensor channels); (2) KPI deltas rendered redundant "PASS (PASS)"; (3) scorecard interpretations and verdicts had hard-coded text; (4) synthetic mode mixed sources without explicit wafer banner; (5) bare exception hid read issues; (6) synthetic timestamps and protocol gates O5_v2/O6_v2 required explicit disclosure across docs.
- **Decision:**
  1. Updated `app/dashboard.py` labels to "one sensor trace of 152 sequential time steps" and axis label to "Time Step Index (0 to 151)".
  2. Drove KPI deltas and scorecard table dynamically from loaded JSON metrics without hardcoded numbers, returning `"n/a"` if keys are missing.
  3. Added an explicit `st.info` banner when running synthetic wafer traces.
  4. Narrowed parquet loading exceptions to `(OSError, ValueError)`.
  5. Added explicit assertions in `tests/test_dashboard.py` checking View 6 metrics, deltas, and synthetic mode banner.
  6. Disclosed in `README.md`, `docs/MODEL_CARD.md`, and dashboard technical expander that timestamps are synthesized at 10-minute intervals tracking UCR file order, and clarified that `O5_v2` and `O6_v2` are protocol verification gates.
- **Documents changed:** `app/dashboard.py`, `tests/test_dashboard.py`, `README.md`, `docs/MODEL_CARD.md`, `docs/08_DECISIONS_LOG.md`, `docs/09_PROGRESS_LOG.md`, `tasks/todo.md`.
- **Verification or follow-up task:** T27 verified via `ruff check` and `pytest`.

