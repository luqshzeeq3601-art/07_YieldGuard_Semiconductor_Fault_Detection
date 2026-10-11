# 03. Data specification

Status: draft, planning v0.1, 11 October 2026. No data has been downloaded yet.

## 1. Dataset and provenance

1. Source: UCI Machine Learning Repository, SECOM, DOI 10.24432/C54305. Citation: McCann, M. & Johnston, A. (2008). Licence: CC BY 4.0 as stated on the UCI page on 11 October 2026.
2. Use the UCI archive only. Kaggle mirrors differ in format and stated licence.
3. Download `secom.zip`, extract only `secom.data`, `secom_labels.data`, `secom.names` to `data/raw/secom/` with path-traversal checks.
4. Record URL, download time, archive and file SHA-256, citation and licence text in `data/source_manifest.json`. First acquisition pins the hashes; later mismatches stop the pipeline.
5. Raw data is not committed by default (open question Q2 in [the plan](../tasks/plan.md)). Attribution appears in README, model card and `docs/10_SOURCES.md`.

Not yet verified, checked in T02 rather than assumed: timestamp string format, date range, duplicate or unsorted timestamps, per-column missingness, number of constant columns, fail rate over time.

## 2. Raw schema

| File | Content | Rules |
|---|---|---|
| `secom.data` | 1,567 lines, 590 space-separated values, `NaN` for missing | Exactly 590 fields per line; values finite or NaN |
| `secom_labels.data` | One line per run: label and quoted date-time | Label in {-1, 1}; timestamp parses with one declared format |

1. Internal names: `f000`-`f589` by position, `run_id` = original zero-based line number, `label_fail` = 1 when label is 1, `timestamp`.
2. The two files align by line number; row counts must match exactly.
3. T02 records the observed timestamp format, range, duplicates and whether the file is already sorted. The format string is then fixed in config; unparseable rows fail the load.
4. Counts expected from UCI: 1,567 runs, 104 fails. A mismatch stops the pipeline.

## 3. Ordering and chronological split

1. Order runs by `(timestamp, run_id)`. Ties keep file order.
2. **Holdout:** the last 25% of ordered runs (392). **Development:** the first 75% (1,175).
3. Development is cut into five contiguous equal blocks D1-D5 (235 runs each).
4. A boundary never separates runs with identical timestamps: move it later until the timestamp changes, and record the shift.
5. Forward folds: fit D1-D2 and score D3; fit D1-D3 and score D4; fit D1-D4 and score D5.
6. Final fit uses D1-D4. D5 is the calibration block for alarm thresholds. The released model is therefore exactly the fold-3 model; no refit happens after thresholds are set.
7. The split is computed from timestamps only and frozen in `data/processed/split_manifest.json` (indices, boundaries, hash) before any label statistics are produced.
8. No shuffling anywhere. No stratification by label.

## 4. Label isolation

1. `prepare` writes `labels_dev.parquet` and a separate `labels_holdout.parquet`.
2. Only the holdout evaluation command reads `labels_holdout.parquet`, and only with an explicit flag and a frozen bundle.
3. Data-quality reports show label statistics for development blocks only. The holdout fail count is derivable by subtraction from the published total of 104; it is not reported or used before the final evaluation.
4. Holdout sensor values may be used for structural checks (schema, parse) and for the post-freeze drift check. They never enter fitting, selection or threshold setting.

## 5. Data-quality transformers

All are fitted on the training runs of the current fold only. Applied in this order; each stores its decision list.

| Step | Rule | Default |
|---|---|---|
| 1. Missingness pruning | Drop columns whose training missing fraction exceeds the limit | 0.50 |
| 2. Constant pruning | Drop columns with one distinct non-missing training value, or one value covering at least 99% of non-missing training values | 0.99 |
| 3. Collinearity reduction | Pairwise-complete Pearson correlation on training runs; for each pair with absolute r at or above the limit, keep the column with less missingness, then the lower index | 0.95 |
| 4. Imputation | Training median per retained column | SPC, PCA and Isolation Forest only |
| 5. Scaling | Mean and standard deviation of **passing** training runs | PCA and Isolation Forest only |

1. XGBoost receives the retained columns with NaN left in place.
2. Step 3 stores a group map: each dropped column points to the kept representative. Sensor rankings report the representative together with its group members.
3. Thresholds 0.50, 0.99 and 0.95 are fixed before modelling. They are not tuned.
4. `reports/data_quality.json` records counts dropped at each step per fold.

## 6. Reference population for SPC

1. Phase I data: passing runs in the training blocks, in time order.
2. One declared trimming pass for the PCA model: remove Phase I runs above the 99.9% theoretical T-squared or SPE limit, refit once, record the number removed.
3. Univariate centre line is the Phase I mean; sigma is the average moving range divided by 1.128.
4. Phase II data: all later runs, scored with frozen parameters.

## 7. Predictor exclusions

Never predictors: `label_fail`, `run_id`, `timestamp`, block or fold identifiers, row position, any statistic computed from later runs. Past labels are not used as features because label delay after line test is unknown.

## 8. Synthetic SECOM-like data

1. `yieldguard.synthetic` writes the same two-file format so the real loader is exercised.
2. Contents: latent-factor correlated sensors, constant columns, duplicated (collinear) columns, columns above the missingness limit, random missing values, slow drift on a few sensors, and fails caused by shifts on a **known list of sensors**.
3. Uses: all required tests, CI, the dashboard demo and objective O5. It is never mixed with SECOM and never used for a performance claim.

## 9. Derived files

| Path | Purpose |
|---|---|
| `data/source_manifest.json` | Acquisition, licence, citation, checksums |
| `data/processed/runs.parquet` | Ordered runs with `run_id`, `timestamp`, 590 sensors, block label |
| `data/processed/split_manifest.json` | Block and holdout membership, boundaries, hash |
| `data/processed/labels_dev.parquet`, `labels_holdout.parquet` | Separated labels |
| `reports/data_quality.json` | Schema, counts, missingness, timestamp findings, pruning counts |
