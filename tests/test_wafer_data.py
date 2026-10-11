"""Unit tests for Semiconductor Wafer Etch dataset parsing and Zip-Slip security."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pytest

from yieldguard.wafer_data import (
    WAFER_N_FEATURES,
    _parse_ucr_txt_or_ts,
    _safe_extract_zip,
)


def test_safe_extract_zip_blocks_path_traversal(tmp_path: Path) -> None:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("../evil.txt", "malicious")
    with pytest.raises(ValueError, match="Unsafe zip member"):
        _safe_extract_zip(buf.getvalue(), tmp_path / "extract")


def test_parse_ucr_txt_valid_and_invalid_dimensions(tmp_path: Path) -> None:
    rng = np.random.default_rng(42)
    lines: list[str] = []
    for i in range(10):
        lbl = -1 if i < 2 else 1
        vals = " ".join(f"{v:.4f}" for v in rng.normal(0.0, 1.0, size=WAFER_N_FEATURES))
        lines.append(f"{lbl} {vals}")
    valid_file = tmp_path / "Wafer_TRAIN.txt"
    valid_file.write_text("\n".join(lines), encoding="utf-8")

    y, x = _parse_ucr_txt_or_ts(valid_file)
    assert len(y) == 10
    assert x.shape == (10, WAFER_N_FEATURES)
    assert int(np.sum(y == -1)) == 2

    # Wrong feature count must raise ValueError
    bad_file = tmp_path / "Wafer_BAD.txt"
    bad_file.write_text("1 0.1 0.2 0.3\n", encoding="utf-8")
    with pytest.raises(ValueError, match="expected 152 features"):
        _parse_ucr_txt_or_ts(bad_file)
