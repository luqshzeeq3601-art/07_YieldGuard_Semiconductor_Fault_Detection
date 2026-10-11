"""Tests for SECOM raw data validation, Zip Slip protection, and CSV batch validation."""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import pytest

from scripts.download_data import safe_extract_zip
from yieldguard.data import (
    N_SENSORS,
    SENSOR_COLUMNS,
    load_and_validate_runs_csv,
    load_raw_secom,
)
from yieldguard.synthetic import generate_synthetic_secom


def test_reject_wrong_field_count(tmp_path: Path) -> None:
    """Reject secom.data lines with 589 or 591 fields instead of 590."""
    generate_synthetic_secom(output_dir=tmp_path, n_runs=10, seed=42)
    data_file = tmp_path / "secom.data"
    lines = data_file.read_text(encoding="utf-8").splitlines()
    # Chop last token on row 3 -> 589 fields
    lines[3] = " ".join(lines[3].split()[:589])
    data_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Wrong field count at row 3"):
        load_raw_secom(tmp_path, verify_uci_counts=False)


def test_reject_bad_label(tmp_path: Path) -> None:
    """Reject labels outside {-1, 1}."""
    generate_synthetic_secom(output_dir=tmp_path, n_runs=10, seed=42)
    labels_file = tmp_path / "secom_labels.data"
    lines = labels_file.read_text(encoding="utf-8").splitlines()
    lines[2] = '0 "19/07/2008 13:00:00"'
    labels_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid label value at row 2"):
        load_raw_secom(tmp_path, verify_uci_counts=False)


def test_reject_unparseable_timestamp(tmp_path: Path) -> None:
    """Reject timestamps that do not match the declared format."""
    generate_synthetic_secom(output_dir=tmp_path, n_runs=10, seed=42)
    labels_file = tmp_path / "secom_labels.data"
    lines = labels_file.read_text(encoding="utf-8").splitlines()
    lines[4] = '-1 "2008/99/99 25:00:00"'
    labels_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Unparseable timestamp at row 4"):
        load_raw_secom(tmp_path, verify_uci_counts=False)


def test_reject_row_count_mismatch(tmp_path: Path) -> None:
    """Reject mismatched line counts between secom.data and secom_labels.data."""
    generate_synthetic_secom(output_dir=tmp_path, n_runs=10, seed=42)
    labels_file = tmp_path / "secom_labels.data"
    lines = labels_file.read_text(encoding="utf-8").splitlines()
    labels_file.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Row-count mismatch"):
        load_raw_secom(tmp_path, verify_uci_counts=False)


def test_reject_infinite_and_string_sensor_values(tmp_path: Path) -> None:
    """Reject non-numeric strings and infinities in secom.data."""
    generate_synthetic_secom(output_dir=tmp_path, n_runs=10, seed=42)
    data_file = tmp_path / "secom.data"
    lines = data_file.read_text(encoding="utf-8").splitlines()
    tokens = lines[1].split()
    tokens[5] = "inf"
    lines[1] = " ".join(tokens)
    data_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Infinite sensor value"):
        load_raw_secom(tmp_path, verify_uci_counts=False)

    tokens[5] = "bad_token"
    lines[1] = " ".join(tokens)
    data_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Non-numeric sensor token"):
        load_raw_secom(tmp_path, verify_uci_counts=False)


def test_reject_zip_path_traversal(tmp_path: Path) -> None:
    """Reject zip archives containing ../ path traversal entries (Zip Slip)."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("../evil.txt", b"malicious")
    zip_bytes = buf.getvalue()

    with pytest.raises(ValueError, match="path traversal"):
        safe_extract_zip(zip_bytes, tmp_path / "extract", ["evil.txt"])


def test_csv_batch_loader_validation(tmp_path: Path) -> None:
    """Verify CSV loader accepts valid 590-sensor CSV and rejects strings/infs/booleans."""
    generate_synthetic_secom(output_dir=tmp_path, n_runs=8, seed=42)
    loaded = load_raw_secom(tmp_path, verify_uci_counts=False)
    csv_path = tmp_path / "batch.csv"
    cols = ["run_id", "timestamp_str", *SENSOR_COLUMNS]
    df_out = loaded.runs_df[cols].rename(columns={"timestamp_str": "timestamp"})
    df_out.to_csv(csv_path, index=False)

    valid_df = load_and_validate_runs_csv(csv_path)
    assert len(valid_df) == 8
    assert list(valid_df[SENSOR_COLUMNS].columns) == SENSOR_COLUMNS

    # Inject boolean string into f002
    bad_df = df_out.copy().astype(object)
    bad_df.loc[2, "f002"] = "True"
    bad_csv = tmp_path / "bad_bool.csv"
    bad_df.to_csv(bad_csv, index=False)
    with pytest.raises(ValueError, match="Invalid boolean value"):
        load_and_validate_runs_csv(bad_csv)


@pytest.mark.integration
def test_real_secom_manifest_and_loader() -> None:
    """Verify real UCI SECOM download, manifest metadata, and 1,567 x 590 shape with 104 fails."""
    raw_dir = Path("data/raw/secom")
    manifest_path = Path("data/source_manifest.json")
    if not (raw_dir / "secom.data").exists() or not manifest_path.exists():
        pytest.skip("Real SECOM data not downloaded in this environment.")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert "archive.ics.uci.edu" in manifest["source_url"]
    assert manifest["doi"] == "10.24432/C54305"
    assert "CC BY 4.0" in manifest["licence"]
    assert all(check["matches"] for check in manifest["hash_checks"].values())

    loaded = load_raw_secom(raw_dir, verify_uci_counts=True)
    assert len(loaded.runs_df) == 1567
    assert len(SENSOR_COLUMNS) == N_SENSORS == 590
    assert int(loaded.runs_df["label_fail"].sum()) == 104
