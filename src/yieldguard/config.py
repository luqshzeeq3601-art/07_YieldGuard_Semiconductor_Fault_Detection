"""Typed YAML configuration schema and loader for YieldGuard."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class ProjectConfig(BaseModel):
    name: str = "YieldGuard"
    version: str = "0.1.0"
    seed: int = 42


class DataConfig(BaseModel):
    dataset_id: str = "SECOM"
    raw_dir: str = "data/raw/secom"
    processed_dir: str = "data/processed"
    source_manifest: str = "data/source_manifest.json"
    quality_report: str = "reports/data_quality.json"
    split_manifest: str = "data/processed/split_manifest.json"
    runs_parquet: str = "data/processed/runs.parquet"
    labels_dev_parquet: str = "data/processed/labels_dev.parquet"
    labels_holdout_parquet: str = "data/processed/labels_holdout.parquet"
    timestamp_format: str = "%d/%m/%Y %H:%M:%S"
    verify_uci_counts: bool = True


class SplitsConfig(BaseModel):
    dev_fraction: float = 0.75
    n_dev_blocks: int = 5


class QualityConfig(BaseModel):
    max_missing_fraction: float = 0.50
    max_constant_dominant_fraction: float = 0.99
    collinearity_threshold: float = 0.95


class SpcConfig(BaseModel):
    n_monitored_sensors: int = 10
    max_sensor_missing_fraction: float = 0.05
    ewma_lambda: float = 0.20
    ewma_l: float = 3.0
    mr_d2: float = 1.128


class MspcConfig(BaseModel):
    variance_explained_target: float = 0.80
    max_components: int = 50
    phase1_trim_alpha: float = 0.999
    limit_alpha: float = 0.99


class AnomalyConfig(BaseModel):
    n_estimators: int = 200
    max_samples: str = "auto"
    seed: int = 42


class XgboostSearchConfig(BaseModel):
    max_configs: int = 20
    seed: int = 42


class PolicyConfig(BaseModel):
    primary_alarm_budget_per_100: float = 5.0
    alarm_budgets_per_100: list[float] = Field(default_factory=lambda: [2.0, 5.0, 10.0])
    false_alarm_rates_per_100: list[float] = Field(default_factory=lambda: [2.0, 5.0, 10.0])
    n_bootstraps: int = 2000
    block_bootstrap_length: int = 20


class TargetsConfig(BaseModel):
    o1_pr_auc_min_prevalence_multiple: float = 2.0
    o1_require_ci_lower_above_prevalence: bool = True
    o2_min_recall_at_5_far: float = 0.20
    o3_alarm_rate_min_per_100: float = 2.0
    o3_alarm_rate_max_per_100: float = 8.0
    o4_paired_bootstrap_reported: bool = True
    o5_synthetic_top3_hit_rate_min: float = 0.90
    o5_shap_additivity_tol: float = 1e-4
    o6_reproducibility_tol: float = 1e-6
    o7_cli_api_parity_tol: float = 1e-9
    o7_warm_score_p95_max_ms: float = 200.0


class ModelsConfig(BaseModel):
    staging_dir: str = "models/staging"
    tracking_uri: str = "sqlite:///mlruns.db"
    reports_dir: str = "reports"


class Config(BaseModel):
    project: ProjectConfig = Field(default_factory=ProjectConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    splits: SplitsConfig = Field(default_factory=SplitsConfig)
    quality: QualityConfig = Field(default_factory=QualityConfig)
    spc: SpcConfig = Field(default_factory=SpcConfig)
    mspc: MspcConfig = Field(default_factory=MspcConfig)
    anomaly: AnomalyConfig = Field(default_factory=AnomalyConfig)
    xgboost: XgboostSearchConfig = Field(default_factory=XgboostSearchConfig)
    policy: PolicyConfig = Field(default_factory=PolicyConfig)
    targets: TargetsConfig = Field(default_factory=TargetsConfig)
    models: ModelsConfig = Field(default_factory=ModelsConfig)


def load_config(config_path: str | Path = "configs/default.yaml") -> Config:
    """Load and validate configuration from YAML file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with open(path, encoding="utf-8") as f:
        raw_dict = yaml.safe_load(f) or {}

    return Config(**raw_dict)
