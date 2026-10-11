# 10. Sources and evidence register

## 1. Primary sources

| ID | Source | What it supports |
|---|---|---|
| S1 | McCann, M. & Johnston, A. (2008). *SECOM* [Dataset]. UCI Machine Learning Repository. DOI: `10.24432/C54305`. URL: `https://archive.ics.uci.edu/dataset/179/secom` | Primary dataset (1,567 runs, 590 features, 104 fails), CC BY 4.0 licence and attribution |
| S2 | UCI static archive: `https://archive.ics.uci.edu/static/public/179/secom.zip` | Direct download archive containing `secom.data`, `secom_labels.data`, `secom.names` |
| S3 | Montgomery, D. C. (2019). *Introduction to Statistical Quality Control* (8th ed.). Wiley | Phase I/Phase II SPC, average moving range $\bar{MR}/1.128$, EWMA time-varying control limits ($\lambda=0.2, L=3$, in-control ARL $\approx 370.4$ for normal $3\sigma$ and $\approx 465$–$500$ for EWMA $\lambda=0.2, L=3$), Western Electric / Nelson run rules |
| S4 | Jackson, J. E. & Mudholkar, G. S. (1979); Nomikos, P. & MacGregor, J. F. (1995). Multivariate SPC ($T^2$ and $SPE/Q$) | PCA Hotelling $T^2$ $F$-distribution limit, Box/Jackson moment-matched $\chi^2$ limit for $SPE$, and sensor contribution decomposition |
| S5 | scikit-learn documentation (`PCA`, `IsolationForest`, `average_precision_score`, `roc_auc_score`) | Multivariate PCA, Isolation Forest (`score_samples`), and evaluation metrics |
| S6 | XGBoost documentation (`XGBClassifier`, `scale_pos_weight`) | Class-weighted gradient boosted trees with native missing-value handling |
| S7 | Lundberg, S. M. & Lee, S.-I. (2017); SHAP `TreeExplainer` documentation | Raw-margin tree SHAP attributions and additivity check |
| S8 | Evidently documentation (`evidently.legacy.report.Report`, `DataDriftPreset`) | Offline batch sensor drift analysis |
| S9 | Sibling project `01_TurbineGuard_Predictive_Maintenance` | Package structure, trusted bundle verification, Zip Slip guard, FastAPI and CI conventions |

## 2. Attribution and usage notes

- **Dataset citation:** McCann, M. & Johnston, A. (2008). SECOM [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C54305. Licensed under Creative Commons Attribution 4.0 International (CC BY 4.0).
- **Raw data policy:** Raw SECOM files are downloaded on demand into `data/raw/secom/` (ignored by Git) and verified via SHA-256 checksums recorded in `data/source_manifest.json`.
- **Claims boundary:** All metrics in this repository are offline benchmark results on the public UCI SECOM dataset. They do not represent measured fab yield improvement, scrap reduction or financial savings.
