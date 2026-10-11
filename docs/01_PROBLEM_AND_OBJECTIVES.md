# 01. Problem and objectives

Status: draft, planning v0.1, 11 October 2026. No code or results exist yet.

## 1. Problem statement

A process engineer sees 590 sensor readings per production run and a pass/fail result from end-of-line test. About 1 run in 15 fails. The practical question is: **which runs should be flagged for review within a fixed alarm capacity, and which sensors should be looked at first for each flagged run?** Accuracy is the wrong measure: predicting "pass" for every run scores about 93% and finds nothing.

YieldGuard compares classical SPC (EWMA charts, run rules, Hotelling T-squared, SPE/Q) with machine-learning detectors (Isolation Forest, class-weighted gradient boosting) under one alarm budget, and ranks sensors per flagged run.

## 2. Evidence and assumptions

1. UCI SECOM is real semiconductor process data: 1,567 runs, 590 anonymised sensor features, 104 fails, a timestamp per run. Sensor names, tools, chambers, products and recipes are not disclosed.
2. The alarm budget, review capacity and personas are project assumptions. There are no interviews, yield records or cost data.
3. All runs are treated as one time-ordered stream. If the data mixes tools or products, charts will be noisier than in a real single-tool deployment. This cannot be checked from the data.
4. The portfolio maps this project to process/yield engineering skills. It does not establish that Intel, Keysight or ViTrox use this dataset or method.
5. Results are **offline benchmark results**. They are not measured yield improvement, scrap reduction or cost savings.

## 3. Users and decisions

| User | Problem | Useful output |
|---|---|---|
| Process engineer | Too many sensors to watch | SPC charts with limits and rule violations on selected sensors |
| Yield engineer | Needs to triage suspect runs | Flagged runs within an alarm budget, with ranked sensors |
| ML engineer | Needs a trustworthy comparison | Frozen bundle, chronological evaluation, intervals, drift report |

## 4. Objectives and predeclared targets

Targets are set before any modelling. They are design goals, not results. Chance levels are stated so a miss or a pass can be read correctly.

| ID | Objective | Measurement and target |
|---|---|---|
| O1 | Rank failing runs above passing runs | Primary detector holdout PR-AUC >= 2x holdout fail prevalence, and 95% bootstrap lower bound > prevalence (chance PR-AUC = prevalence) |
| O2 | Catch fails at a fixed false-alarm rate | Holdout recall >= 0.20 at 5 false alarms per 100 passing runs (chance = 0.05); also report at 2 and 10 |
| O3 | Hold an alarm budget out of sample | Threshold frozen for 5 alarms per 100 runs; realised holdout rate within 2-8 per 100 |
| O4 | Compare SPC with ML honestly | Paired bootstrap PR-AUC difference between best ML and best SPC detector, with interval. No pass/fail target; the direction is reported as found |
| O5 | Rank sensors for a flagged run | On synthetic runs with a known single-sensor 5-sigma fault, the true sensor is in the top 3 MSPC contributions in >= 90% of cases; SHAP contributions sum to the raw model output within 1e-4. SECOM has no sensor ground truth, so no accuracy claim is made there |
| O6 | Reproduce outputs | Same data hash, split manifest, config, lock and seed reproduce every reported metric within 1e-6 on the same environment |
| O7 | Deliver usable engineering | CLI/API score parity within 1e-9; Docker runtime smoke test and CI pass; warm `/score` p95 < 200 ms without explanations |

## 5. Uncertainty statement

1. The holdout is the last 25% of runs: 392 runs. If fails were evenly spread it would hold about 26 fails; the real count is not used for any decision.
2. With about 26 fails, one extra detected fail moves recall by about 4 points, and a 95% interval on recall is roughly +/-0.18 wide on each side. Differences between detectors smaller than that cannot be resolved by this dataset.
3. A detector that "wins" the holdout by a few points has not been shown to be better.

## 6. Success and failure

1. Engineering completion (reproducibility, validation, tests, serving, honest reports) is judged separately from whether targets O1-O3 are met.
2. Missed targets and negative results stay in the README and model card as measured. A likely negative result is declared now: EWMA charts detect sustained mean shifts, while SECOM labels are per-run outcomes, so univariate charts may detect few fails.
3. The holdout is evaluated once. Nothing is retuned afterwards under the same holdout claim. The procedure is in [the experiment plan](05_EXPERIMENT_PLAN.md).

## 7. Portfolio value

Adds Phase I/Phase II SPC, multivariate SPC with contribution analysis, train-only high-dimensional data cleaning, imbalanced evaluation under an alarm budget and small-sample uncertainty reporting. None of these appear in TurbineGuard.
