"""
tests/test_data_normalization.py
==================================
Tests for normalization statistics computation, save/load, and transforms.
"""

from __future__ import annotations

import json
import numpy as np
import pytest
from pathlib import Path
from datetime import datetime

from cloudremoval.data.normalization import (
    _Z_CLAMP,
    compute_normalization_stats,
    denormalize_s2,
    load_normalization,
    normalize_s1,
    normalize_s2,
    save_normalization,
)
from cloudremoval.data.triplets import TripletRecord


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def fake_triplets(tmp_path: Path) -> list[TripletRecord]:
    """Return a list of TripletRecords pointing to real synthetic GeoTIFFs."""
    import rasterio
    from rasterio.transform import from_bounds

    def _make_tiff(path, n_bands, fill):
        path.parent.mkdir(parents=True, exist_ok=True)
        transform = from_bounds(0, 0, 1, 1, 32, 32)
        with rasterio.open(
            path, "w", driver="GTiff",
            height=32, width=32, count=n_bands, dtype="float64",
            crs="EPSG:32755", transform=transform
        ) as dst:
            dst.write(np.full((n_bands, 32, 32), fill, dtype="float64"))

    records = []
    for i in range(5):
        s2_path  = tmp_path / f"s2_{i}.tif"
        s1_path  = tmp_path / f"s1_{i}.tif"
        tgt_path = tmp_path / f"tgt_{i}.tif"
        _make_tiff(s2_path, 13, 1000.0 + i * 100)
        _make_tiff(s1_path,  2, -15.0  - i * 2)
        _make_tiff(tgt_path, 13, 1000.0 + i * 100)
        records.append(TripletRecord(
            triplet_id=f"fake__{i}__s2obs0",
            scene_key=f"fake_{i}",
            roi_id=f"roi{i}",
            roi_coords=[0.0, 0.0],
            s2_path=s2_path, s2_date=datetime(2022, 1, 1), s2_obs_index=0,
            s1_path=s1_path, s1_date=datetime(2022, 1, 2),
            target_path=tgt_path, target_date=datetime(2022, 1, 3),
            s2_to_target_days=2.0, s1_to_target_days=1.0,
        ))
    return records


@pytest.fixture()
def norm_stats(fake_triplets) -> dict:
    return compute_normalization_stats(fake_triplets, n_sample=5, seed=42)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestComputeNormStats:
    def test_returns_s2_and_s1_keys(self, norm_stats) -> None:
        assert "s2" in norm_stats
        assert "s1" in norm_stats

    def test_s2_has_13_bands(self, norm_stats) -> None:
        assert len(norm_stats["s2"]["mean"]) == 13
        assert len(norm_stats["s2"]["std"]) == 13

    def test_s1_has_2_bands(self, norm_stats) -> None:
        assert len(norm_stats["s1"]["mean"]) == 2
        assert len(norm_stats["s1"]["std"]) == 2

    def test_std_positive(self, norm_stats) -> None:
        for std in norm_stats["s2"]["std"]:
            assert std > 0
        for std in norm_stats["s1"]["std"]:
            assert std > 0


class TestSaveLoadNormalization:
    def test_roundtrip(self, tmp_path: Path, norm_stats) -> None:
        path = tmp_path / "norm" / "normalization.json"
        save_normalization(norm_stats, path, version="v1")
        loaded = load_normalization(path)
        assert loaded["version"] == "v1"
        assert "s2" in loaded
        assert "s1" in loaded

    def test_json_readable(self, tmp_path: Path, norm_stats) -> None:
        path = tmp_path / "norm.json"
        save_normalization(norm_stats, path)
        with open(path) as fh:
            raw = json.load(fh)
        assert "s2" in raw

    def test_creates_parent_dirs(self, tmp_path: Path, norm_stats) -> None:
        path = tmp_path / "deep" / "nested" / "norm.json"
        save_normalization(norm_stats, path)
        assert path.exists()


class TestNormalizeS2:
    def test_output_shape_preserved(self, norm_stats) -> None:
        arr = np.ones((13, 64, 64), dtype=np.float64) * 1500.0
        out = normalize_s2(arr, norm_stats)
        assert out.shape == arr.shape

    def test_values_clamped(self, norm_stats) -> None:
        # Extreme values should be clamped to ±Z_CLAMP
        arr = np.full((13, 4, 4), 1e8, dtype=np.float64)
        out = normalize_s2(arr, norm_stats)
        assert out.max() <= _Z_CLAMP

        arr2 = np.full((13, 4, 4), -1e8, dtype=np.float64)
        out2 = normalize_s2(arr2, norm_stats)
        assert out2.min() >= -_Z_CLAMP

    def test_nan_preserved(self, norm_stats) -> None:
        arr = np.ones((13, 4, 4), dtype=np.float64) * 1000.0
        arr[0, 0, 0] = np.nan
        out = normalize_s2(arr, norm_stats)
        assert np.isnan(out[0, 0, 0])


class TestNormalizeS1:
    def test_output_shape_preserved(self, norm_stats) -> None:
        arr = np.ones((2, 64, 64), dtype=np.float64) * -15.0
        out = normalize_s1(arr, norm_stats)
        assert out.shape == arr.shape

    def test_values_clamped(self, norm_stats) -> None:
        arr = np.full((2, 4, 4), 1e8, dtype=np.float64)
        out = normalize_s1(arr, norm_stats)
        assert out.max() <= _Z_CLAMP


class TestDenormalizeS2:
    def test_approximate_roundtrip(self, norm_stats) -> None:
        original = np.full((13, 4, 4), 1500.0, dtype=np.float64)
        normalized = normalize_s2(original, norm_stats)
        reconstructed = denormalize_s2(normalized, norm_stats)
        # Should be close but not exact (due to clamping)
        np.testing.assert_allclose(reconstructed, original, atol=50.0)

    def test_output_non_negative(self, norm_stats) -> None:
        arr = np.zeros((13, 4, 4), dtype=np.float64)
        out = denormalize_s2(arr, norm_stats)
        # Denormalized mean values should be positive (reflectance can't be negative)
        assert out.min() >= 0.0
