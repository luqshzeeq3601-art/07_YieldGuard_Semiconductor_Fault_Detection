# 05. Experiment and evaluation protocol

Status: frozen at Checkpoint C3, 11 October 2026. Forward folds D3-D5 evaluated; holdout labels unopened.

## 1. Predeclared protocol

Frozen before any experiment: data hashes, split manifest, cleaning thresholds, SPC parameters, search budget, metrics, targets ([objectives, section 4](01_PROBLEM_AND_OBJECTIVES.md)) and the selection rule. Each run is logged to local MLflow with a JSON run manifest. Seed 42.

The [data specification](03_DATA_SPEC.md) owns the chronological split, the blocks D1-D5 and the cleaning rules.

## 2. Detectors

| ID | Detector | Uses labels to fit | Purpose |
|---|---|---|---|
| E01 | Prevalence baseline | No | Chance level for every metric |
| E02 | Individuals chart, max standardised value | Sensor selection only | Simplest univariate SPC |
| E03 | EWMA, max standardised distance | Sensor selection only | Shift-oriented univariate SPC |
| E04 | Hotelling T-squared | Passing-run reference | Multivariate SPC, model space |
| E05 | SPE/Q | Passing-run reference | Multivariate SPC, residual space |
| E06 | Combined MSPC index | Passing-run reference | Single multivariate score |
| E07 | Isolation Forest on passing training runs | Passing-run reference | ML anomaly detector |
| E08 | XGBoost with `scale_pos_weight` | Yes | Supervised comparison |

Declared diagnostics, never eligible as primary: Isolation Forest fitted on all training runs; EWMA lambda 0.1 and 0.3; PCA variance rule 70% and 90% (note: when the 50-component cap binds on ~270 retained features, PCA variance explained is capped at ~62%, so 70% and 90% rules yield identical components to the 50-component cap).

## 3. Search budget

1. E02-E07 have fixed parameters. Isolation Forest: 200 trees, `max_samples='auto'`, seed 42.
2. E08: at most 20 configurations, listed with their seed before execution. Depth 1-3, learning rate 0.02-0.10, 100-400 trees, subsample 0.7-1.0, column sample 0.3-1.0, minimum child weight 1-10. Fixed tree counts; no early stopping. `scale_pos_weight` is the training-fold negative/positive ratio.
3. No resampling (SMOTE or undersampling). No supervised feature selection outside the fold.

## 4. Selection rule

1. Compare detectors on out-of-fold scores from D3-D5 (705 runs).
2. PR-AUC is computed within each fold and averaged, so score scales from different fold models never mix.
3. E08 configuration: chosen from inner folds 1-2 (`fold_1` on D3, `fold_2` on D4) by highest mean fold PR-AUC; within one standard error, prefer the shallower model with fewer trees. Fold 3 (D5) is strictly excluded from configuration selection so calibration data cannot influence hyperparameter choice.
4. Primary detector: highest mean fold PR-AUC across D3-D5, except that if the best ML detector's paired bootstrap advantage over the best SPC detector has a 95% interval containing zero, the SPC detector is primary. Both stratified and moving-block (block length 20) within-fold bootstraps are evaluated.
5. All eight detectors are scored on the holdout. The headline is the primary detector; the rest is a declared comparison of eight, not eight chances to find a winner.

## 5. Alarm budget

1. **Alarm rate** = alarms per 100 scored runs. This is the operating constraint and needs no labels.
2. **False-alarm rate** = false alarms per 100 passing runs. This is the reporting axis for "recall at fixed false-alarm rate".
3. Thresholds: for budgets B in {2, 5, 10}, the threshold is the `1 - B/100` quantile (method `linear`) of the fold-3 model's scores on D5. Alarm when score > threshold. Primary budget: 5.
4. With 235 calibration runs the 5-per-100 threshold rests on about 12 runs. The realised holdout alarm rate is reported against the budget (target O3); a miss is a finding about drift, not a reason to recalibrate.
5. No threshold is chosen by accuracy, F1 or Youden index.

## 6. Metrics

1. PR-AUC (average precision), with prevalence beside it.
2. Recall at 2, 5 and 10 false alarms per 100 passing runs.
3. At each alarm budget: alarms, true alarms, false alarms, precision, recall, lift over prevalence, realised alarms per 100.
4. Textbook SPC limits (L = 3, 99% T-squared and SPE): realised alarm rate and recall, reported as is.
5. ROC-AUC as a secondary figure only.
6. Counts accompany every ratio. Undefined ratios are null with a reason.

## 7. Uncertainty

1. 2,000 seeded bootstrap draws, stratified by class, 95% percentile intervals.
2. Paired differences between detectors use the same draws.
3. Sensitivity: moving-block bootstrap with block length 20 runs, because neighbouring runs may be correlated. Both are reported; if they disagree, the wider interval is quoted.
4. Every results table states the number of fails it rests on.

## 8. Sensor ranking

1. MSPC: contributions for runs flagged by T-squared or SPE.
2. XGBoost: SHAP TreeExplainer on raw margin, background from training runs only.
3. Reported per flagged run: top 5 per method, group members, and overlap between methods.
4. Quantitative check on synthetic data only (target O5). On SECOM the ranking is descriptive; no root-cause or accuracy claim.

## 9. Drift check

1. Reference: retained sensor columns of D1-D4. Current: any batch of at least 100 runs; fewer gives `insufficient_data`.
2. Evidently data-drift preset, dataset warning at 50% drifted columns; method and threshold recorded from the pinned version.
3. Added table: per-column missing-rate change between reference and current.
4. Controls: identical copy shows no drift; five sensors shifted by 2 reference standard deviations are flagged; a 50-run batch returns `insufficient_data`.
5. Reference versus holdout sensors is run after freeze. It uses no labels. Drift does not prove performance changed.

## 10. Final holdout procedure

1. Freeze cleaning, SPC parameters, PCA, Isolation Forest, XGBoost, thresholds, config and lock into a versioned bundle.
2. Score all 392 holdout runs; write and hash predictions before `labels_holdout.parquet` is opened.
3. Evaluate once. Record every metric, interval and target verdict, including misses.
4. The command refuses to overwrite an existing evaluation record. A bug-fix rerun keeps the original, explains the bug and states that the holdout has been seen.
5. No retuning afterwards under the same holdout claim.

## 11. Claims boundary

All figures produced under this protocol are offline benchmark results on a public dataset. They are not measured yield improvement, scrap reduction or cost savings.
