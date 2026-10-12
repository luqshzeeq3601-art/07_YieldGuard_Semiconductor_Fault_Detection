"""Tests for model bundle freezing, checksum verification, and bundle loader."""

import json
from pathlib import Path

import pytest

from yieldguard.artifacts import freeze_model_bundle, load_model_bundle


def test_bundle_freeze_and_load_roundtrip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    project_root = Path(__file__).resolve().parent.parent
    stg_src = (
        project_root / "models" / "staging"
        if (project_root / "models" / "staging" / "quality.joblib").exists()
        else project_root / "models" / "v0.1.0"
    )
    rep_src = project_root / "reports" / "validation_metrics.json"

    tmp_stg = tmp_path / "models" / "staging"
    tmp_stg.mkdir(parents=True, exist_ok=True)
    for f in stg_src.iterdir():
        if f.is_file():
            (tmp_stg / f.name).write_bytes(f.read_bytes())

    tmp_rep = tmp_path / "reports"
    tmp_rep.mkdir(parents=True, exist_ok=True)
    if rep_src.exists():
        (tmp_rep / "validation_metrics.json").write_bytes(rep_src.read_bytes())

    split_p = (project_root / "data" / "processed" / "split_manifest.json").as_posix()
    source_p = (project_root / "data" / "source_manifest.json").as_posix()
    stg_p = tmp_stg.as_posix()
    rep_p = tmp_rep.as_posix()

    cfg_path = tmp_path / "config.yaml"
    cfg_content = f"""
project:
  name: YieldGuard
  seed: 42
data:
  dataset_id: SECOM
  split_manifest: {split_p}
  source_manifest: {source_p}
models:
  staging_dir: {stg_p}
  reports_dir: {rep_p}
policy:
  alarm_budgets_per_100: [2.0, 5.0, 10.0]
"""
    cfg_path.write_text(cfg_content, encoding="utf-8")

    out_dir = freeze_model_bundle(config_path=cfg_path, version="v0.1.0-test")
    assert out_dir.exists()
    assert (out_dir / "metadata.json").exists()

    bundle = load_model_bundle(out_dir)
    assert bundle.version == "v0.1.0-test"
    assert bundle.quality is not None
    assert bundle.spc_model is not None
    assert bundle.mspc_model is not None
    assert bundle.iforest_model is not None
    assert bundle.xgb_model is not None
    assert bundle.policy["primary_detector"] == "E02"
    assert len(bundle.feature_schema["raw_sensor_columns"]) == 590


def test_tampered_bundle_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project_root = Path(__file__).resolve().parent.parent
    stg_src = (
        project_root / "models" / "staging"
        if (project_root / "models" / "staging" / "quality.joblib").exists()
        else project_root / "models" / "v0.1.0"
    )
    tmp_stg = tmp_path / "models" / "staging"
    tmp_stg.mkdir(parents=True, exist_ok=True)
    for f in stg_src.iterdir():
        if f.is_file():
            (tmp_stg / f.name).write_bytes(f.read_bytes())

    split_p = (project_root / "data" / "processed" / "split_manifest.json").as_posix()
    source_p = (project_root / "data" / "source_manifest.json").as_posix()
    stg_p = tmp_stg.as_posix()

    cfg_path = tmp_path / "config.yaml"
    cfg_content = f"""
project:
  name: YieldGuard
  seed: 42
data:
  dataset_id: SECOM
  split_manifest: {split_p}
  source_manifest: {source_p}
models:
  staging_dir: {stg_p}
policy:
  alarm_budgets_per_100: [2.0, 5.0, 10.0]
"""
    cfg_path.write_text(cfg_content, encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    out_dir = freeze_model_bundle(config_path=cfg_path, version="v_tamper")
    bundle = load_model_bundle(out_dir)
    assert bundle is not None

    spc_file = out_dir / "spc.json"
    data = json.loads(spc_file.read_text(encoding="utf-8"))
    data["ewma_l"] = 99.9
    spc_file.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValueError, match="Integrity check failed: checksum mismatch"):
        load_model_bundle(out_dir)


def test_refuse_overwrite_frozen_bundle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project_root = Path(__file__).resolve().parent.parent
    stg_src = (
        project_root / "models" / "staging"
        if (project_root / "models" / "staging" / "quality.joblib").exists()
        else project_root / "models" / "v0.1.0"
    )
    tmp_stg = tmp_path / "models" / "staging"
    tmp_stg.mkdir(parents=True, exist_ok=True)
    for f in stg_src.iterdir():
        if f.is_file():
            (tmp_stg / f.name).write_bytes(f.read_bytes())

    split_p = (project_root / "data" / "processed" / "split_manifest.json").as_posix()
    source_p = (project_root / "data" / "source_manifest.json").as_posix()
    stg_p = tmp_stg.as_posix()

    cfg_path = tmp_path / "config.yaml"
    cfg_content = f"""
project:
  name: YieldGuard
  seed: 42
data:
  dataset_id: SECOM
  split_manifest: {split_p}
  source_manifest: {source_p}
models:
  staging_dir: {stg_p}
policy:
  alarm_budgets_per_100: [2.0, 5.0, 10.0]
"""
    cfg_path.write_text(cfg_content, encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    freeze_model_bundle(config_path=cfg_path, version="v_no_overwrite")
    with pytest.raises(FileExistsError, match="Refusing to overwrite immutable frozen release"):
        freeze_model_bundle(config_path=cfg_path, version="v_no_overwrite", allow_overwrite=False)
