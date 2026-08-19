"""
tests/inference/test_inference.py
==================================
Pytest test suite for Phase 6 geospatial inference pipeline.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
import pytest
import numpy as np
import rasterio
from rasterio.transform import from_origin
import torch

from cloudremoval.models import DSen2CRConfig, build_model
from cloudremoval.geospatial.alignment import validate_geotiff, load_and_align_s1_to_s2, GeospatialValidationError
from cloudremoval.inference.tiled_inference import TiledInferenceEngine, create_2d_blending_window
from cloudremoval.inference.writer import write_reconstructed_geotiff, verify_reconstructed_geotiff
from cloudremoval.inference.preview import create_inference_preview


# Helper to create temporary synthetic GeoTIFFs
def _create_synthetic_geotiff(
    path: Path,
    bands: int = 13,
    height: int = 64,
    width: int = 64,
    crs: str = "EPSG:32643",
) -> None:
    transform = from_origin(500000.0, 3000000.0, 10.0, 10.0)
    data = np.random.uniform(0.01, 0.5, size=(bands, height, width)).astype(np.float32)
    profile = {
        "driver": "GTiff",
        "dtype": "float32",
        "width": width,
        "height": height,
        "count": bands,
        "crs": crs,
        "transform": transform,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data)


class TestGeospatialValidation:

    def test_validate_valid_s2(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "s2_test.tif"
            _create_synthetic_geotiff(p, bands=13)
            meta = validate_geotiff(p, expected_bands=13)
            assert meta["count"] == 13
            assert meta["width"] == 64
            assert meta["height"] == 64
            assert meta["crs"] is not None

    def test_validate_missing_file_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            validate_geotiff(Path("non_existent_file.tif"), expected_bands=13)

    def test_validate_wrong_band_count_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "s2_wrong.tif"
            _create_synthetic_geotiff(p, bands=3)
            with pytest.raises(GeospatialValidationError):
                validate_geotiff(p, expected_bands=13)


class TestAlignmentAndTiling:

    def test_load_and_align_s1_to_s2(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            s2_p = Path(tmpdir) / "s2.tif"
            s1_p = Path(tmpdir) / "s1.tif"
            _create_synthetic_geotiff(s2_p, bands=13, height=64, width=64)
            _create_synthetic_geotiff(s1_p, bands=2, height=64, width=64)

            s2_arr, s1_arr, s2_meta = load_and_align_s1_to_s2(s2_p, s1_p)
            assert s2_arr.shape == (13, 64, 64)
            assert s1_arr.shape == (2, 64, 64)
            assert s2_meta["width"] == 64

    def test_create_2d_blending_window(self) -> None:
        w = create_2d_blending_window(tile_size=64)
        assert w.shape == (64, 64)
        assert 0.0 < w.min() <= w.max() <= 1.0

    def test_tiled_inference_synthetic(self) -> None:
        cfg = DSen2CRConfig(base_features=32, num_res_blocks=2, device="cpu", amp_enabled=False)
        m = build_model(cfg).to("cpu")

        # Fake normalization path
        norm_path = Path("data/normalization/normalization.json")

        engine = TiledInferenceEngine(
            model=m,
            norm_path=norm_path,
            tile_size=64,
            overlap=16,
            batch_size=2,
            device="cpu",
            amp_enabled=False,
        )

        s2_data = np.random.uniform(0.01, 0.5, size=(13, 80, 80)).astype(np.float32)
        s1_data = np.random.uniform(0.01, 0.5, size=(2, 80, 80)).astype(np.float32)

        out, stats = engine.run_scene_inference(s2_data, s1_data)
        assert out.shape == (13, 80, 80)
        assert np.isfinite(out).all()
        assert stats["total_tiles"] > 0


class TestGeoTIFFWriterAndPreview:

    def test_write_and_verify_geotiff(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            s2_p = Path(tmpdir) / "s2_orig.tif"
            out_p = Path(tmpdir) / "reconstructed.tif"
            _create_synthetic_geotiff(s2_p, bands=13, height=64, width=64)
            meta = validate_geotiff(s2_p, expected_bands=13)

            recon_data = np.random.uniform(0.0, 0.5, size=(13, 64, 64)).astype(np.float32)
            write_reconstructed_geotiff(out_p, recon_data, s2_meta=meta)

            report = verify_reconstructed_geotiff(out_p, expected_s2_meta=meta)
            assert report["verified"] is True
            assert report["bands"] == 13
            assert report["width"] == 64
            assert report["height"] == 64

    def test_create_inference_preview(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out_png = Path(tmpdir) / "preview.png"
            c = np.random.uniform(0.0, 0.5, size=(13, 64, 64)).astype(np.float32)
            r = np.random.uniform(0.0, 0.5, size=(13, 64, 64)).astype(np.float32)

            create_inference_preview(c, r, out_png, job_id="test_job")
            assert out_png.exists()
            assert out_png.stat().st_size > 1000
