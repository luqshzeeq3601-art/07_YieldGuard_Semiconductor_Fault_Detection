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
  1. Record explicit SHA-256 hashes in git-tracked `reports/staging_manifest_sha256.json` and create initial git commit before T12.
  2. Keep strict `>` threshold comparison rule as predeclared; record 2.55 alarms per 100 on D5 as a known limitation without post-hoc tie alteration.
  3. Recompute moving-block bootstrap strictly within folds: E03 moving-block 95% CI is `[0.0295, 0.2185]` (contains 0.1046 point estimate). Paired within-fold moving-block difference (E08 - E02) is `[-0.1273, +0.2324]`, confirming Rule 4 retains E02 under both bootstrap methods.
  4. Explicitly label PCA diagnostics with actual cumulative variance explained (~62%) and note the 50-component cap.
- **Effect on requirements/data/protocol:** Preserves integrity of frozen protocol without post-hoc alterations; confirms E02 selection.
- **Documents changed:** `docs/08_DECISIONS_LOG.md`, `docs/09_PROGRESS_LOG.md`, `docs/05_EXPERIMENT_PLAN.md`, `src/yieldguard/evaluate.py`, `src/yieldguard/train.py`, `reports/staging_manifest_sha256.json`.
- **Verification or follow-up task:** T12 holdout evaluation.


