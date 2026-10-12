"""Model bundle freeze, integrity verification, and bundle loading for YieldGuard."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import xgboost as xgb

from yieldguard.anomaly import FittedAnomalyModel
from yieldguard.config import load_config
from yieldguard.data import SENSOR_COLUMNS
from yieldguard.mspc import FittedMspcModel
from yieldguard.quality import QualityTransformer
from yieldguard.spc import FittedSpcModel

logger = logging.getLogger(__name__)


def compute_file_sha256(path: str | Path) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def _write_bundle_json(path: Path, value: dict[str, Any]) -> None:
    """Write deterministic UTF-8 / LF JSON bytes."""
    path.write_bytes((json.dumps(value, indent=2, allow_nan=False) + "\n").encode("utf-8"))


@dataclass
class LoadedBundle:
    """Container for verified model bundle components."""

    version: str
    bundle_dir: Path
    quality: QualityTransformer
    spc_model: FittedSpcModel
    mspc_model: FittedMspcModel
    iforest_model: FittedAnomalyModel
    xgb_model: xgb.Booster
    policy: dict[str, Any]
    feature_schema: dict[str, Any]
    metadata: dict[str, Any]


def freeze_model_bundle(
    config_path: str | Path = "configs/default.yaml",
    version: str = "v0.1.0",
    allow_overwrite: bool = False,
) -> Path:
    """Freeze validated staging models and policy into an immutable release bundle.

    Directory layout per docs/04_TECHNICAL_DESIGN.md section 7:
    models/<version>/
    1. quality.joblib: pruning lists, medians, scaling, group map
    2. spc.json: monitored sensors, Phase I limits, rules
    3. mspc.joblib: PCA loadings, eigenvalues, theoretical limits
    4. iforest.joblib: fitted Isolation Forest
    5. xgb_model.json: XGBoost booster
    6. policy.json: primary detector, thresholds for budgets 2, 5, 10
    7. feature_schema.json: 590 raw sensor names and retained columns
    8. metadata.json: file hashes, versions, seed, validation summary
    9. monitoring_reference.parquet & reference_manifest.json
    """
    cfg = load_config(config_path)
    staging_dir = Path(cfg.models.staging_dir)
    bundle_dir = Path("models") / version

    if bundle_dir.exists() and any(bundle_dir.iterdir()) and not allow_overwrite:
        raise FileExistsError(
            f"Frozen bundle directory {bundle_dir} already exists and is non-empty. "
            f"Refusing to overwrite immutable frozen release {version}."
        )
    bundle_dir.mkdir(parents=True, exist_ok=True)

    # Required staging files
    req_staging = [
        "quality.joblib",
        "spc.json",
        "mspc.joblib",
        "iforest.joblib",
        "xgb_model.json",
        "policy.json",
        "monitoring_reference.parquet",
    ]
    for fname in req_staging:
        src = staging_dir / fname
        if not src.exists():
            raise FileNotFoundError(f"Missing staging file {src}. Run training and evaluation first.")

    # 1. Copy core models and policy
    (bundle_dir / "quality.joblib").write_bytes((staging_dir / "quality.joblib").read_bytes())
    (bundle_dir / "spc.json").write_bytes((staging_dir / "spc.json").read_bytes())
    (bundle_dir / "mspc.joblib").write_bytes((staging_dir / "mspc.joblib").read_bytes())
    (bundle_dir / "iforest.joblib").write_bytes((staging_dir / "iforest.joblib").read_bytes())
    (bundle_dir / "xgb_model.json").write_bytes((staging_dir / "xgb_model.json").read_bytes())
    (bundle_dir / "policy.json").write_bytes((staging_dir / "policy.json").read_bytes())
    (bundle_dir / "monitoring_reference.parquet").write_bytes(
        (staging_dir / "monitoring_reference.parquet").read_bytes()
    )

    # Load copied models to build schemas
    qt = QualityTransformer.load(bundle_dir / "quality.joblib")
    spc = FittedSpcModel.load(bundle_dir / "spc.json")
    policy = json.loads((bundle_dir / "policy.json").read_text(encoding="utf-8"))

    # 2. Write feature_schema.json
    schema_dict = {
        "schema_version": "1.0",
        "raw_sensor_count": len(SENSOR_COLUMNS),
        "raw_sensor_columns": list(SENSOR_COLUMNS),
        "retained_columns": qt.retained_columns,
        "n_retained_columns": len(qt.retained_columns),
        "dropped_high_missing": qt.dropped_high_missing,
        "dropped_constant": qt.dropped_constant,
        "dropped_collinear": qt.dropped_collinear,
        "collinear_groups": qt.representative_to_members,
        "monitored_sensors": spc.monitored_sensors,
    }
    _write_bundle_json(bundle_dir / "feature_schema.json", schema_dict)

    # 3. Write reference_manifest.json
    ref_df = pd.read_parquet(bundle_dir / "monitoring_reference.parquet")
    ref_manifest = {
        "n_runs": len(ref_df),
        "columns": list(ref_df.columns),
        "parquet_sha256": compute_file_sha256(bundle_dir / "monitoring_reference.parquet"),
    }
    _write_bundle_json(bundle_dir / "reference_manifest.json", ref_manifest)

    # 4. Compute file hashes
    bundle_files = [
        "quality.joblib",
        "spc.json",
        "mspc.joblib",
        "iforest.joblib",
        "xgb_model.json",
        "policy.json",
        "feature_schema.json",
        "monitoring_reference.parquet",
        "reference_manifest.json",
    ]
    file_hashes = {fname: compute_file_sha256(bundle_dir / fname) for fname in bundle_files}

    # Load validation report summary if available
    val_report_path = Path("reports/validation_metrics.json")
    val_summary = {}
    if val_report_path.exists():
        val_summary = json.loads(val_report_path.read_text(encoding="utf-8")).get("selection", {})

    metadata_dict = {
        "bundle_version": version,
        "dataset_id": cfg.data.dataset_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_seed": cfg.project.seed,
        "primary_detector": policy.get("primary_detector", "E02"),
        "alarm_budgets_per_100": cfg.policy.alarm_budgets_per_100,
        "thresholds": policy.get("thresholds", {}),
        "selection_summary": val_summary,
        "file_hashes": file_hashes,
        "split_manifest_sha256": compute_file_sha256(cfg.data.split_manifest)
        if Path(cfg.data.split_manifest).exists()
        else None,
        "source_manifest_sha256": compute_file_sha256(cfg.data.source_manifest)
        if Path(cfg.data.source_manifest).exists()
        else None,
    }
    _write_bundle_json(bundle_dir / "metadata.json", metadata_dict)
    logger.info("Successfully frozen model bundle %s to %s", version, bundle_dir)
    return bundle_dir


def load_model_bundle(bundle_dir: str | Path) -> LoadedBundle:
    """Load and verify a trusted project bundle from bundle directory.

    Validates:
    - Expected files exist.
    - SHA-256 checksums match metadata.json.
    - Deserializes only project-generated bundle artifacts.
    """
    bundle_path = Path(bundle_dir)
    if not bundle_path.exists():
        raise FileNotFoundError(f"Bundle directory does not exist: {bundle_path}")

    meta_file = bundle_path / "metadata.json"
    if not meta_file.exists():
        raise ValueError(f"Invalid bundle: missing metadata.json in {bundle_path}")

    metadata = json.loads(meta_file.read_text(encoding="utf-8"))
    file_hashes = metadata.get("file_hashes", {})

    # Checksum verification for tamper evidence
    for fname, expected_hash in file_hashes.items():
        fpath = bundle_path / fname
        if not fpath.exists():
            raise FileNotFoundError(f"Missing bundle file {fname} in {bundle_path}")
        actual_hash = compute_file_sha256(fpath)
        if actual_hash != expected_hash and fname.endswith(".json"):
            raw_bytes = fpath.read_bytes()
            lf_hash = hashlib.sha256(raw_bytes.replace(b"\r\n", b"\n")).hexdigest()
            crlf_hash = hashlib.sha256(raw_bytes.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")).hexdigest()
            if expected_hash in (lf_hash, crlf_hash):
                actual_hash = expected_hash

        if actual_hash != expected_hash:
            raise ValueError(
                f"Integrity check failed: checksum mismatch for {fname} in {bundle_path}. "
                f"Expected {expected_hash}, got {actual_hash}."
            )

    # Load components
    quality = QualityTransformer.load(bundle_path / "quality.joblib")
    spc_model = FittedSpcModel.load(bundle_path / "spc.json")
    mspc_model = FittedMspcModel.load(bundle_path / "mspc.joblib")
    iforest_model = FittedAnomalyModel.load(bundle_path / "iforest.joblib")

    xgb_booster = xgb.Booster()
    xgb_booster.load_model(str(bundle_path / "xgb_model.json"))

    policy = json.loads((bundle_path / "policy.json").read_text(encoding="utf-8"))
    feature_schema = json.loads((bundle_path / "feature_schema.json").read_text(encoding="utf-8"))

    return LoadedBundle(
        version=metadata.get("bundle_version", "unknown"),
        bundle_dir=bundle_path,
        quality=quality,
        spc_model=spc_model,
        mspc_model=mspc_model,
        iforest_model=iforest_model,
        xgb_model=xgb_booster,
        policy=policy,
        feature_schema=feature_schema,
        metadata=metadata,
    )
