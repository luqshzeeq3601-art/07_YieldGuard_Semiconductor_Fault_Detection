# 00. Execution guide

## 1. What we are building

**YieldGuard** is a local semiconductor process fault detection and SPC versus machine-learning demonstrator on UCI SECOM (1,567 runs, 590 anonymised sensors, 104 fails). It compares classical SPC (EWMA charts, run rules, Hotelling T-squared, SPE/Q) with machine-learning detectors (Isolation Forest, class-weighted XGBoost) under a fixed alarm budget, and ranks contributing sensors per flagged run.

The owner is the user. The implementer acts as ML engineer and project manager: keep the scope small, validate the science, record evidence and finish tasks in dependency order.

## 2. Reading order

1. `README.md` and `AGENTS.md`: scope and working instructions.
2. `01_PROBLEM_AND_OBJECTIVES.md` and `02_PRD.md`: intended outcome.
3. `tasks/plan.md` and `tasks/todo.md`: order and acceptance criteria.
4. `03_DATA_SPEC.md` and `05_EXPERIMENT_PLAN.md` before touching data or models.
5. `04_TECHNICAL_DESIGN.md` and `07_OPERATIONS_AND_COMMANDS.md` before implementation.
6. `06_VALIDATION_AND_RELEASE.md` before declaring a milestone complete.
7. `08_DECISIONS_LOG.md` and `09_PROGRESS_LOG.md` whenever resuming.

All files in this list are in `docs/` unless a path is explicitly given. [Sources](10_SOURCES.md) provide supporting references.

## 3. Defaults already chosen

| Decision | Default |
|---|---|
| Problem | Flag suspect runs under a fixed alarm budget and rank contributing sensors |
| Dataset | UCI SECOM only (DOI 10.24432/C54305, CC BY 4.0); download on demand with pinned hashes |
| Device | CPU, local Windows development, Linux Docker serving |
| Chronological split | First 75% development (1,175 runs across D1-D5 of 235 runs each); last 25% holdout (392 runs) |
| Cleaning | Train-only pruning (>50% missing, >=99% constant, \|r\| >= 0.95 collinearity with group map) |
| SPC detectors | Univariate individuals (E02), EWMA lambda=0.2 L=3 (E03), PCA Hotelling T-squared (E04), SPE/Q (E05), combined MSPC index (E06) |
| ML detectors | Isolation Forest on passing training runs (E07), class-weighted XGBoost <= 20 configs (E08) |
| Primary budget | 5 alarms per 100 runs (with 2 and 10 also reported), set on calibration block D5 |
| Primary selection rule | Highest mean fold PR-AUC on D3-D5, except SPC is primary if best ML's paired bootstrap 95% CI vs best SPC contains zero |
| Delivery | CLI, FastAPI (`/health`, `/ready`, `/model-info`, `/score`, `/spc`), Streamlit dashboard, Docker, CI, MLflow, Evidently |
| Task tracker | `tasks/todo.md` only |

## 4. Document authority

| Question | Owning file |
|---|---|
| Should a feature exist? | `02_PRD.md` |
| Is a value or split valid? | `03_DATA_SPEC.md` |
| Which model or threshold wins? | `05_EXPERIMENT_PLAN.md` |
| What is the API or artifact contract? | `04_TECHNICAL_DESIGN.md` |
| Does work pass? | `06_VALIDATION_AND_RELEASE.md` |
| What is next? | `tasks/todo.md` and `09_PROGRESS_LOG.md` |

Changes to these defaults require a decision entry in `08_DECISIONS_LOG.md` and updates to affected documents.
