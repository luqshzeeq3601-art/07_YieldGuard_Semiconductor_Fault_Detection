# 02. Product requirements document

Status: draft, planning v0.1, 11 October 2026. No code or results exist yet.

## 1. Product definition

- **Name:** YieldGuard.
- **Version:** planning v0.1, 11 October 2026.
- **Owner:** user.
- **Problem:** 590 sensors per run cannot be watched by hand, and a 6.6% fail rate defeats accuracy-driven models.
- **Value:** one validated pipeline produces SPC charts, per-run fault scores under an alarm budget and ranked sensors.
- **Delivery environment:** local CPU, Windows development, Linux Docker serving.

## 2. Primary user journey

1. An engineer supplies a time-ordered CSV of runs, or one run through the API.
2. The system validates the 590-value schema and applies the frozen training-derived cleaning.
3. It returns T-squared, SPE, Isolation Forest and gradient-boosting scores, an alarm flag at the frozen budget and, on request, ranked sensors.
4. The dashboard shows EWMA and T-squared/SPE charts with limits, rule violations and a drill-down for a flagged run.
5. The ML engineer compares a new batch with the frozen reference and reads the drift report.

## 3. User stories

| ID | Story | Acceptance |
|---|---|---|
| US1 | As a yield engineer, I want flagged runs limited to my review capacity | Alarm count on the calibration block equals the budget within rounding; realised holdout rate reported |
| US2 | As a process engineer, I want to see which sensors drove a flag | Ranked sensors with method, value and collinear group members |
| US3 | As a process engineer, I want standard charts | EWMA chart with time-varying limits and four run rules per monitored sensor |
| US4 | As an ML engineer, I want to reproduce a reported number | Bundle links data hash, split manifest, config, lock and MLflow run |
| US5 | As an operator, I want bad input rejected | Errors name the field or run; no score is produced for an invalid run |
| US6 | As an ML engineer, I want to notice changed data | Drift report states reference/current definitions, method, sample sizes and limits |

## 4. Functional requirements

| ID | Priority | Requirement | Verification |
|---|---|---|---|
| FR1 | Must | Acquire SECOM from UCI with hashes, licence and citation recorded | Source manifest |
| FR2 | Must | Split chronologically; seal holdout labels | Split manifest; label-isolation tests |
| FR3 | Must | Prune by missingness, constant value and collinearity, fitted on training runs only | Perturbation tests: later blocks cannot change fitted selections |
| FR4 | Must | EWMA charts and four run rules on selected sensors | Hand-computed fixture; simulated in-control run length |
| FR5 | Must | PCA on passing training runs; Hotelling T-squared and SPE/Q with limits | Known-matrix tests; injected-fault tests |
| FR6 | Must | Isolation Forest and class-weighted XGBoost under a bounded search | Comparison report with fold scores |
| FR7 | Must | Thresholds chosen by alarms per 100 runs | Policy tests; no accuracy-based threshold anywhere |
| FR8 | Must | Per-run sensor ranking from MSPC contributions and SHAP | Additivity and synthetic ground-truth tests |
| FR9 | Must | PR-AUC, recall at fixed false-alarm rate and bootstrap intervals | Hand-computed metric tests |
| FR10 | Must | CLI for prepare, train, evaluate, freeze, score, explain, drift | Command checks in tasks |
| FR11 | Must | FastAPI `/health`, `/ready`, `/model-info`, `/score`, `/spc` | Contract tests |
| FR12 | Must | Streamlit SPC dashboard using the shared package | AppTest smoke test; manual check |
| FR13 | Must | Docker image with a runtime smoke test in CI | CI `docker-smoke` job |
| FR14 | Must | Evidently drift check with controls | No-change, shifted and insufficient-data controls |
| FR15 | Must | Model card and README with measured results only | Evidence audit |
| FR16 | Later | Public hosting, other fab datasets, calibrated probabilities | Separate approval |

## 5. Nonfunctional requirements

| ID | Requirement | Acceptance |
|---|---|---|
| NFR1 | Reproducibility | Seed 42; metric tolerance 1e-6 on the same environment |
| NFR2 | Responsiveness | Warm `/score` p95 < 200 ms without explanations, 100 sequential requests; explanation latency reported separately |
| NFR3 | Reliability | Missing or incompatible bundle gives 503 on `/ready`; `/health` still answers |
| NFR4 | Validation | Exactly 590 values; null allowed for missing; reject strings, booleans, infinities, wrong length |
| NFR5 | Resource use | CPU only; <= 20 XGBoost configurations over 3 forward folds; run time recorded |
| NFR6 | Input containment | No file path or URL in requests; only project-generated bundles are loaded |
| NFR7 | CI independence | Required tests pass with no SECOM download |

## 6. Output definitions

- **t2, spe:** Hotelling T-squared and squared prediction error of the run against the PCA model of passing training runs.
- **mspc_index:** `t2 / t2_limit + spe / spe_limit`; a ranking score, not a probability.
- **iforest_score:** negative `score_samples`; higher means more unusual.
- **xgb_score:** class-weighted model output. Class weighting distorts probabilities, so it is **not** a calibrated failure probability.
- **alarm:** primary detector score exceeds the threshold frozen for the alarm budget.
- **top_sensors:** ranked sensors with method (`spe_contribution`, `t2_contribution`, `shap`), value and collinear group members. They show what moved the score, not root cause.

## 7. MVP boundaries

Included: SECOM, offline training, batch CSV, stateless scoring, SPC charts, explanations, dashboard, container, CI, local MLflow, drift report.

Excluded: equipment control, root-cause claims, yield or cost estimates, live MES/SECS-GEM ingestion, streaming, deep learning, automatic retraining, calibrated probabilities, paid cloud resources.

## 8. Release decision

Every Must requirement has evidence or an explicit unresolved status. Engineering delivery and target outcomes are recorded separately. Public release is a later user-directed action.
