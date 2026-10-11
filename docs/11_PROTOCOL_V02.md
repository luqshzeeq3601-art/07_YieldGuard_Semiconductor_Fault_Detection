# 11. Protocol v0.2: Prequential Drift-Adaptive Evaluation & Confirmatory Design

Status: Frozen before any `v0.2` experiment execution, 12 October 2026.

## 1. Motivation and Scope Boundary

1. **Preservation of `v0.1.0`:** Release `v0.1.0` (`models/v0.1.0/`, `reports/holdout_metrics.json`) remains the immutable, single-shot benchmark record on UCI SECOM. Under `v0.1.0`, no detector beat the `0.0612` failure prevalence on the 392-run chronological holdout, `E02`'s 12 alarms at budget 5 yielded `0 TP / 12 FP`, and `86.5%` of retained sensors exhibited statistically significant distribution drift between training (`D1`–`D4`) and holdout.
2. **Exploratory vs. Confirmatory Status:**
   - Because `data/processed/labels_holdout.parquet` was opened in `v0.1.0`, any subsequent experiment on UCI SECOM is strictly **exploratory** (`evaluation_type = "exploratory_post_holdout"`).
   - If the Phase 2 SECOM oracle upper bound (5-fold stratified shuffled CV across all 1,567 runs) fails to beat chance (`oracle_beats_chance == False`), SECOM sensors lack intrinsic predictive signal for the end-of-line label and `v0.2` is not run on SECOM.
   - **Confirmatory evaluation** of `v0.2` is executed once on an unseen second dataset: the **Semiconductor Wafer Etch dataset** (Bobick & Olszewski 2001, CMU / UCR `Wafer`: 7,164 wafer runs, 152 time-step sensor readings per wafer, 762 abnormal/faulty runs, `10.64%` fault prevalence).

---

## 2. Prequential (Rolling-Origin) Evaluation Design

Instead of a single static 25% holdout with only 24 fails, `v0.2` uses **prequential (test-then-train / rolling-origin) evaluation** across contiguous chronological blocks:

1. Order all $N$ runs strictly by `(timestamp, run_id)` (or acquisition run sequence).
2. Partition the timeline into $K$ equal contiguous blocks $B_1, B_2, \dots, B_K$ ($K = 10$ blocks of $\approx 716$ runs on Wafer Etch; $K = 8$ blocks of $\approx 196$ runs if run on SECOM).
3. **Warmup blocks:** $B_1 \cup B_2$ serve as the initial historical warmup window.
4. **Rolling evaluation loop ($k = 3, 4, \dots, K$):**
   - **Fit:** Fit cleaning, feature selection, and detectors using only past runs in blocks $\le k-1$.
   - **Calibrate threshold:** Set alarm budget thresholds $\tau_k(B)$ for budgets $B \in \{2, 5, 10\}$ per 100 runs using the **unlabelled score distribution of the most recent block $B_{k-1}$** under the current step-$k$ detector.
   - **Score:** Score all runs in block $B_k$ strictly out-of-sample before revealing $B_k$ labels.
   - **Advance:** Reveal $B_k$ labels and roll forward to $k+1$.

---

## 3. Drift-Adaptive Detectors & Regularised Logistic Baseline (`E01_v2`–`E09_v2`)

| ID | Detector | Baseline Window | Adaptation Mechanism |
|---|---|---|---|
| `E01_v2` | Prevalence baseline | Rolling past blocks $\le k-1$ | Chance reference |
| `E02_v2` | Rolling Robust Individuals SPC | Most recent $W=2$ blocks (passing runs) | Median $\mu_{\text{med}}$ and normalized MAD $\sigma_{\text{MAD}}$ refit every block; top-3 RMS tie-break |
| `E03_v2` | Rolling Robust EWMA ($\lambda=0.20$) | Most recent $W=2$ blocks (passing runs) | Anchored to rolling $\mu_{\text{med}}, \sigma_{\text{MAD}}$; state resets at block boundaries to prevent drift accumulation |
| `E04_v2` | Rolling Hotelling $T^2$ (PCA) | Most recent $W=2$ blocks (passing runs) | PCA subspace ($\le 30$ components, $80\%$ variance) refit every block on robust-scaled passing runs |
| `E05_v2` | Rolling SPE / $Q$-Residual | Most recent $W=2$ blocks (passing runs) | Residual distance orthogonal to rolling PCA subspace |
| `E06_v2` | Rolling Combined MSPC Index | Most recent $W=2$ blocks (passing runs) | $T^2 / \tau_{T^2, 99\%} + \text{SPE} / \tau_{\text{SPE}, 99\%}$ on rolling PCA |
| `E07_v2` | Rolling Isolation Forest | Most recent $W=2$ blocks (passing runs) | 200 trees (`seed=42`) refit every block on recent passing runs |
| `E08_v2` | Rolling Class-Weighted XGBoost | Expanding window $B_1 \dots B_{k-1}$ (recency-weighted) | Dynamic `scale_pos_weight` with exponential block recency weights ($\gamma = 0.85^{k-1-j}$) |
| `E09_v2` | Regularised Logistic Baseline | Expanding window $B_1 \dots B_{k-1}$ (recency-weighted) | L2-regularised `LogisticRegression(C=0.1, class_weight="balanced")` on robust-scaled features |

---

## 4. Robust Scaling (Median / MAD) and Deterministic Tie-Breaking

1. **Robust Feature Scaling:**
   - For each retained sensor $j$ on passing runs in the baseline window:
     $$\text{med}_j = \text{median}(x_{\text{pass}, j}), \qquad \text{MAD}_j = 1.4826 \times \text{median}(|x_{\text{pass}, j} - \text{med}_j|)$$
   - If $\text{MAD}_j < 10^{-9}$ (due to sensor quantization), fall back to $\text{IQR}_j / 1.349$, then sample standard deviation $s_j$, then $1.0$.
   - Standardized robust reading: $z_{t,j} = \text{clip}\big((x_{t,j} - \text{med}_j) / \text{MAD}_j, -25, +25\big)$.
2. **Deterministic Tie-Breaking for `E02_v2` and Budget Calibration:**
   - To eliminate single-sensor quantization ties in `E02_v2`, compute sorted absolute standardized values $|z|_{t,(1)} \ge |z|_{t,(2)} \ge \dots \ge |z|_{t,(m)}$ across the $m$ monitored sensors:
     $$S_{\text{E02\_v2}}(t) = |z|_{t,(1)} + 10^{-4} \sqrt{\frac{1}{3}\sum_{r=1}^{\min(3, m)} |z|_{t,(r)}^2} + 10^{-7} \overline{|z|}_t$$
   - Unlabelled threshold calibration on $B_{k-1}$ uses linear quantile `1 - B/100`; if boundary ties remain on $B_{k-1}$, break ties deterministically by secondary score rank so the calibration window alarm rate matches $B\%$ within $1 / |B_{k-1}|$.

---

## 5. Primary Detector Selection Rule (`v0.2`)

1. Warmup-adjacent evaluation blocks $B_3, B_4$ serve as inner selection blocks (`dev_blocks`), and blocks $B_5 \dots B_K$ serve as the out-of-sample prequential evaluation horizon (`eval_blocks`), or all post-warmup blocks $B_3 \dots B_K$ are reported with block-mean PR-AUC.
2. Specifically, to prevent selection bias on the confirmatory run:
   - Primary detector is selected on the first two post-warmup blocks ($B_3, B_4$) using the same parsimony rule as `v0.1`: highest mean block PR-AUC, except best SPC (`E02_v2`–`E06_v2`) is chosen if the best ML detector (`E07_v2`–`E09_v2`) does not beat best SPC on paired bootstrap 95% CI.
   - All 9 detectors (`E01_v2`–`E09_v2`) are scored across all evaluation blocks $B_3 \dots B_K$.

---

## 6. Predeclared `v0.2` Targets (`O1_v2`–`O6_v2`)

Set and frozen prior to executing any `v0.2` run:

| Target ID | Metric | Predeclared Target Criterion |
|---|---|---|
| **O1_v2** | Prequential Fault Ranking (PR-AUC) | Mean block PR-AUC $\ge 2.0 \times \text{prevalence}$, and 95% block-bootstrap lower bound $> \text{prevalence}$ |
| **O2_v2** | Operational Recall @ 5 FAR | Pooled prequential recall $\ge 0.25$ at 5 false alarms per 100 passing runs (chance $= 0.05$) |
| **O3_v2** | Adaptive Alarm Budget Stability | Realised prequential alarm rate at budget 5 within $[3.5, 6.5]$ alarms per 100 runs across post-warmup blocks |
| **O4_v2** | Adaptive SPC vs ML vs Logistic Comparison | Report paired bootstrap PR-AUC differences (`best_ML - best_SPC` and `E08_v2 - E09_v2`) with 95% CIs |
| **O5_v2** | Tie Elimination & Calibration Fidelity | Realised calibration-window alarm rate at budget 5 within $[4.5, 5.5]$ per 100 runs (resolving `v0.1` D5 tie under-firing) |
| **O6_v2** | Reproducibility | Identical data hash, config, and seed `42` reproduce all prequential metrics within $10^{-6}$ |
