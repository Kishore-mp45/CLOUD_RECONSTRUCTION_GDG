"""
tests/test_data_loaders.py
============================
Tests for GeoTIFF loaders using synthetic in-memory GeoTIFFs.

Real dataset files are NOT read.  All tests create tiny valid GeoTIFFs
in a tmp_path fixture.
"""

from __future__ import annotations

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds

from cloudremoval.data.loaders import (
    S1_EXPECTED_BANDS,
    S2_EXPECTED_BANDS,
    GeoTIFFMeta,
    load_geotiff_meta,
    load_s1,
    load_s2,
    load_target,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _write_geotiff(path, n_bands: int, height: int = 32, width: int = 32,
                   dtype="float64", nodata=None, fill=1000.0) -> None:
    """Write a tiny synthetic GeoTIFF for testing."""
    transform = from_bounds(0, 0, 1, 1, width, height)
    with rasterio.open(
        path, "w",
        driver="GTiff",
        height=height, width=width,
        count=n_bands,
        dtype=dtype,
        crs="EPSG:32755",
        transform=transform,
        nodata=nodata,
    ) as dst:
        data = np.full((n_bands, height, width), fill, dtype=dtype)
        if nodata is not None:
            # Mark a corner pixel as nodata
            data[:, 0, 0] = nodata
        dst.write(data)


@pytest.fixture()
def s2_file(tmp_path):
    p = tmp_path / "s2.tif"
    _write_geotiff(p, S2_EXPECTED_BANDS)
    return p


@pytest.fixture()
def s1_file(tmp_path):
    p = tmp_path / "s1.tif"
    _write_geotiff(p, S1_EXPECTED_BANDS, fill=-15.0)
    return p


@pytest.fixture()
def target_file(tmp_path):
    p = tmp_path / "target.tif"
    _write_geotiff(p, S2_EXPECTED_BANDS)
    return p


@pytest.fixture()
def s2_nodata_file(tmp_path):
    p = tmp_path / "s2_nodata.tif"
    _write_geotiff(p, S2_EXPECTED_BANDS, nodata=-9999.0)
    return p


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestLoadGeotiffMeta:
    def test_returns_geotiff_meta(self, s2_file) -> None:
        meta = load_geotiff_meta(s2_file)
        assert isinstance(meta, GeoTIFFMeta)

    def test_band_count(self, s2_file) -> None:
        meta = load_geotiff_meta(s2_file)
        assert meta.count == S2_EXPECTED_BANDS

    def test_resolution(self, s2_file) -> None:
        meta = load_geotiff_meta(s2_file)
        assert meta.res_x > 0
        assert meta.res_y > 0

    def test_crs_set(self, s2_file) -> None:
        meta = load_geotiff_meta(s2_file)
        assert meta.crs is not None


class TestLoadS2:
    def test_shape(self, s2_file) -> None:
        arr = load_s2(s2_file)
        assert arr.ndim == 3
        assert arr.shape[0] == S2_EXPECTED_BANDS

    def test_dtype_float64(self, s2_file) -> None:
        arr = load_s2(s2_file)
        assert arr.dtype == np.float64

    def test_nodata_becomes_nan(self, s2_nodata_file) -> None:
        arr = load_s2(s2_nodata_file)
        # The corner pixel was set to nodata (-9999), should be NaN
        assert np.isnan(arr[:, 0, 0]).all()

    def test_valid_pixels_finite(self, s2_file) -> None:
        arr = load_s2(s2_file)
        assert np.isfinite(arr).all()


class TestLoadS1:
    def test_shape(self, s1_file) -> None:
        arr = load_s1(s1_file)
        assert arr.ndim == 3
        assert arr.shape[0] == S1_EXPECTED_BANDS

    def test_dtype_float64(self, s1_file) -> None:
        arr = load_s1(s1_file)
        assert arr.dtype == np.float64


class TestLoadTarget:
    def test_shape(self, target_file) -> None:
        arr = load_target(target_file)
        assert arr.ndim == 3
        assert arr.shape[0] == S2_EXPECTED_BANDS

    def test_dtype_float64(self, target_file) -> None:
        arr = load_target(target_file)
        assert arr.dtype == np.float64
