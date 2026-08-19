"""
tests/test_data_metadata.py
============================
Tests for metadata parsing and path remapping.

Uses real dataset files but never modifies them.
Tests are fast: they only read the metadata JSON and verify structure.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from cloudremoval.data.metadata import (
    DataRecord,
    ObsEntry,
    SCRATCH_PREFIX,
    parse_metadata,
    remap_path,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def minimal_metadata_json(tmp_path: Path) -> tuple[Path, Path]:
    """Write a minimal metadata JSON and return (meta_path, allclear_dir)."""
    allclear_dir = tmp_path / "allclear_dataset"
    # Create dummy files so path-existence checks can pass if needed
    roi_dir = allclear_dir / "roi99999" / "2022_3" / "s2_toa"
    roi_dir.mkdir(parents=True)
    (roi_dir / "roi99999_s2_toa_2022_3_13_median.tif").touch()

    s1_dir = allclear_dir / "roi99999" / "2022_3" / "s1"
    s1_dir.mkdir(parents=True)
    (s1_dir / "roi99999_s1_2022_3_24_median.tif").touch()

    meta = {
        "roi99999_2022-03-13_2022-03-28": {
            "roi": ["roi99999", [-24.4, 150.558]],
            "target": [
                ["2022-03-18 00:13:39", f"{SCRATCH_PREFIX}roi99999/2022_3/s2_toa/roi99999_s2_toa_2022_3_18_median.tif"]
            ],
            "s2_toa": [
                ["2022-03-13 00:13:45", f"{SCRATCH_PREFIX}roi99999/2022_3/s2_toa/roi99999_s2_toa_2022_3_13_median.tif"],
                ["2022-03-23 00:13:44", f"{SCRATCH_PREFIX}roi99999/2022_3/s2_toa/roi99999_s2_toa_2022_3_23_median.tif"],
                ["2022-03-28 00:13:38", f"{SCRATCH_PREFIX}roi99999/2022_3/s2_toa/roi99999_s2_toa_2022_3_28_median.tif"],
            ],
            "s1": [
                ["2022-03-24 19:21:24", f"{SCRATCH_PREFIX}roi99999/2022_3/s1/roi99999_s1_2022_3_24_median.tif"]
            ],
        },
        "roi88888_2022-11-29_2022-12-14": {
            "roi": ["roi88888", [10.0, 50.0]],
            "target": [
                ["2022-12-09 13:04:32", f"{SCRATCH_PREFIX}roi88888/2022_12/s2_toa/roi88888_s2_toa_2022_12_9_median.tif"]
            ],
            "s2_toa": [
                ["2022-11-29 13:04:29", f"{SCRATCH_PREFIX}roi88888/2022_11/s2_toa/roi88888_s2_toa_2022_11_29_median.tif"],
                ["2022-12-04 13:04:25", f"{SCRATCH_PREFIX}roi88888/2022_12/s2_toa/roi88888_s2_toa_2022_12_4_median.tif"],
                ["2022-12-14 13:04:28", f"{SCRATCH_PREFIX}roi88888/2022_12/s2_toa/roi88888_s2_toa_2022_12_14_median.tif"],
            ],
            "s1": [],   # no S1
        },
    }

    meta_path = tmp_path / "meta.json"
    meta_path.write_text(json.dumps(meta))
    return meta_path, allclear_dir


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestRemapPath:
    def test_removes_scratch_prefix(self, tmp_path: Path) -> None:
        allclear_dir = tmp_path / "allclear_dataset"
        path_str = f"{SCRATCH_PREFIX}roi83528/2022_3/s2_toa/file.tif"
        result = remap_path(path_str, allclear_dir)
        assert str(result).endswith("roi83528/2022_3/s2_toa/file.tif".replace("/", "\\") if __import__("os").name == "nt" else "roi83528/2022_3/s2_toa/file.tif")

    def test_result_under_allclear_dir(self, tmp_path: Path) -> None:
        allclear_dir = tmp_path / "allclear_dataset"
        path_str = f"{SCRATCH_PREFIX}roi83528/file.tif"
        result = remap_path(path_str, allclear_dir)
        assert str(allclear_dir) in str(result)


class TestParseMetadata:
    def test_returns_all_records(self, minimal_metadata_json: tuple) -> None:
        meta_path, allclear_dir = minimal_metadata_json
        records = parse_metadata(meta_path, allclear_dir)
        assert len(records) == 2

    def test_roi_id_parsed(self, minimal_metadata_json: tuple) -> None:
        meta_path, allclear_dir = minimal_metadata_json
        records = parse_metadata(meta_path, allclear_dir)
        roi_ids = {r.roi_id for r in records}
        assert "roi99999" in roi_ids
        assert "roi88888" in roi_ids

    def test_s2_inputs_count(self, minimal_metadata_json: tuple) -> None:
        meta_path, allclear_dir = minimal_metadata_json
        records = parse_metadata(meta_path, allclear_dir)
        for rec in records:
            assert len(rec.s2_inputs) == 3

    def test_target_is_single(self, minimal_metadata_json: tuple) -> None:
        meta_path, allclear_dir = minimal_metadata_json
        records = parse_metadata(meta_path, allclear_dir)
        for rec in records:
            assert isinstance(rec.target, ObsEntry)

    def test_has_s1_flag(self, minimal_metadata_json: tuple) -> None:
        meta_path, allclear_dir = minimal_metadata_json
        records = parse_metadata(meta_path, allclear_dir)
        by_roi = {r.roi_id: r for r in records}
        assert by_roi["roi99999"].has_s1 is True
        assert by_roi["roi88888"].has_s1 is False

    def test_paths_are_path_objects(self, minimal_metadata_json: tuple) -> None:
        meta_path, allclear_dir = minimal_metadata_json
        records = parse_metadata(meta_path, allclear_dir)
        for rec in records:
            assert isinstance(rec.target.path, Path)
            for obs in rec.s2_inputs:
                assert isinstance(obs.path, Path)

    def test_dates_parsed(self, minimal_metadata_json: tuple) -> None:
        meta_path, allclear_dir = minimal_metadata_json
        records = parse_metadata(meta_path, allclear_dir)
        for rec in records:
            assert rec.target.date is not None
            for obs in rec.s2_inputs:
                assert obs.date is not None
