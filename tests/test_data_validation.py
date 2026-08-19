"""
tests/test_data_validation.py
================================
Tests for per-record and per-file validation logic.
"""

from __future__ import annotations

import numpy as np
import pytest
import rasterio
from pathlib import Path
from rasterio.transform import from_bounds
from datetime import datetime

from cloudremoval.data.metadata import DataRecord, ObsEntry
from cloudremoval.data.validation import validate_record, filter_records
from cloudremoval.data.loaders import S2_EXPECTED_BANDS, S1_EXPECTED_BANDS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_tiff(path: Path, n_bands: int, fill=1000.0, nodata=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    transform = from_bounds(0, 0, 1, 1, 32, 32)
    with rasterio.open(
        path, "w", driver="GTiff",
        height=32, width=32, count=n_bands, dtype="float64",
        crs="EPSG:32755", transform=transform, nodata=nodata
    ) as dst:
        data = np.full((n_bands, 32, 32), fill, dtype="float64")
        dst.write(data)


def _make_record(
    tmp_path: Path,
    roi_id: str = "roi1",
    s2_bands: int = S2_EXPECTED_BANDS,
    s1_bands: int = S1_EXPECTED_BANDS,
    has_s1: bool = True,
    create_files: bool = True,
) -> DataRecord:
    s2_path  = tmp_path / "s2.tif"
    s1_path  = tmp_path / "s1.tif"
    tgt_path = tmp_path / "target.tif"
    if create_files:
        _write_tiff(s2_path, s2_bands)
        _write_tiff(tgt_path, S2_EXPECTED_BANDS)
        if has_s1:
            _write_tiff(s1_path, s1_bands)
    dt = datetime(2022, 3, 13)
    return DataRecord(
        key="roi1_key",
        roi_id=roi_id,
        roi_coords=[0.0, 0.0],
        s2_inputs=[
            ObsEntry(date=dt, path=s2_path),
            ObsEntry(date=dt, path=s2_path),
            ObsEntry(date=dt, path=s2_path),
        ],
        s1_inputs=[ObsEntry(date=dt, path=s1_path)] if has_s1 else [],
        target=ObsEntry(date=dt, path=tgt_path),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestValidateRecord:
    def test_valid_record_passes(self, tmp_path) -> None:
        rec = _make_record(tmp_path)
        ok, reason = validate_record(rec, require_s1=True)
        assert ok is True
        assert reason is None

    def test_missing_s1_rejected_when_required(self, tmp_path) -> None:
        rec = _make_record(tmp_path, has_s1=False)
        ok, reason = validate_record(rec, require_s1=True)
        assert ok is False
        assert "no S1" in reason

    def test_missing_s1_accepted_when_not_required(self, tmp_path) -> None:
        rec = _make_record(tmp_path, has_s1=False, create_files=True)
        ok, reason = validate_record(rec, require_s1=False)
        assert ok is True

    def test_missing_target_file_rejected(self, tmp_path) -> None:
        rec = _make_record(tmp_path, create_files=True)
        # Remove target file
        rec.target.path.unlink()
        ok, reason = validate_record(rec)
        assert ok is False
        assert "not found" in reason.lower()

    def test_wrong_band_count_s2_rejected(self, tmp_path) -> None:
        rec = _make_record(tmp_path, s2_bands=6)  # wrong band count
        ok, reason = validate_record(rec)
        assert ok is False

    def test_wrong_band_count_s1_rejected(self, tmp_path) -> None:
        rec = _make_record(tmp_path, s1_bands=1)  # wrong band count
        ok, reason = validate_record(rec)
        assert ok is False


class TestFilterRecords:
    def test_filters_correctly(self, tmp_path) -> None:
        valid_rec = _make_record(tmp_path / "v1", has_s1=True)
        no_s1_rec = _make_record(tmp_path / "v2", has_s1=False)
        valid_list, rejected = filter_records([valid_rec, no_s1_rec], require_s1=True)
        assert len(valid_list) == 1
        assert len(rejected) == 1
        assert rejected[0]["reason"] is not None

    def test_rejection_log_has_reason(self, tmp_path) -> None:
        rec = _make_record(tmp_path, has_s1=False)
        _, rejected = filter_records([rec], require_s1=True)
        assert "reason" in rejected[0]
        assert rejected[0]["reason"]
