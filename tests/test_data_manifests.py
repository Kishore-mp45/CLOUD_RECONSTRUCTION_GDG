"""
tests/test_data_manifests.py
=============================
Tests for manifest read/write roundtrips.
"""

from __future__ import annotations

import pytest
from pathlib import Path
from datetime import datetime

from cloudremoval.data.triplets import TripletRecord
from cloudremoval.data.patches import PatchRecord
from cloudremoval.data.manifests import (
    write_pairs_manifest,
    write_patch_manifest,
    read_patch_manifest,
)


def _fake_triplet(i: int) -> TripletRecord:
    return TripletRecord(
        triplet_id=f"trip_{i}",
        scene_key=f"key_{i}",
        roi_id=f"roi{i}",
        roi_coords=[0.0, 0.0],
        s2_path=Path(f"/fake/s2_{i}.tif"),
        s2_date=datetime(2022, 1, 1),
        s2_obs_index=0,
        s1_path=Path(f"/fake/s1_{i}.tif"),
        s1_date=datetime(2022, 1, 2),
        target_path=Path(f"/fake/target_{i}.tif"),
        target_date=datetime(2022, 1, 3),
        s2_to_target_days=2.0,
        s1_to_target_days=1.0,
    )


def _fake_patch(i: int) -> PatchRecord:
    return PatchRecord(
        patch_id=f"patch_{i}",
        triplet_id=f"trip_{i}",
        scene_key=f"key_{i}",
        roi_id=f"roi{i}",
        s2_path=Path(f"/fake/s2_{i}.tif"),
        s1_path=Path(f"/fake/s1_{i}.tif"),
        target_path=Path(f"/fake/target_{i}.tif"),
        row_off=0, col_off=0,
        patch_h=256, patch_w=256,
        norm_version="v1",
    )


class TestPairsManifest:
    def test_write_creates_file(self, tmp_path) -> None:
        triplets = [_fake_triplet(i) for i in range(5)]
        path = tmp_path / "pairs.json"
        write_pairs_manifest(triplets, path)
        assert path.exists()

    def test_count_in_file(self, tmp_path) -> None:
        import json
        triplets = [_fake_triplet(i) for i in range(7)]
        path = tmp_path / "pairs.json"
        write_pairs_manifest(triplets, path)
        with open(path) as fh:
            data = json.load(fh)
        assert data["count"] == 7


class TestPatchManifest:
    def test_roundtrip(self, tmp_path) -> None:
        patches = [_fake_patch(i) for i in range(10)]
        path = tmp_path / "patches.json"
        write_patch_manifest(patches, path)
        loaded = read_patch_manifest(path)
        assert len(loaded) == 10

    def test_patch_ids_preserved(self, tmp_path) -> None:
        patches = [_fake_patch(i) for i in range(5)]
        path = tmp_path / "patches.json"
        write_patch_manifest(patches, path)
        loaded = read_patch_manifest(path)
        assert [p.patch_id for p in loaded] == [p.patch_id for p in patches]

    def test_paths_are_path_objects(self, tmp_path) -> None:
        patches = [_fake_patch(0)]
        path = tmp_path / "p.json"
        write_patch_manifest(patches, path)
        loaded = read_patch_manifest(path)
        assert isinstance(loaded[0].s2_path, Path)
        assert isinstance(loaded[0].s1_path, Path)
        assert isinstance(loaded[0].target_path, Path)

    def test_window_offsets_preserved(self, tmp_path) -> None:
        p = _fake_patch(0)
        p.row_off = 128
        p.col_off = 64
        path = tmp_path / "p.json"
        write_patch_manifest([p], path)
        loaded = read_patch_manifest(path)
        assert loaded[0].row_off == 128
        assert loaded[0].col_off == 64
