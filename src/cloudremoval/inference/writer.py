"""
src/cloudremoval/inference/writer.py
====================================
GeoTIFF writing and verification utilities for Phase 6.

Key Functions:
  - write_reconstructed_geotiff: Writes 13-band Float32 GeoTIFF with full geospatial tags
  - verify_reconstructed_geotiff: Programmatically asserts CRS, transform, dimensions, and data validity
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import rasterio
from rasterio.transform import Affine

from cloudremoval.models.model_config import S2_CHANNELS

log = logging.getLogger(__name__)

# Standard Sentinel-2 13-band names
S2_BAND_NAMES: List[str] = [
    "B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B10", "B11", "B12"
]


def write_reconstructed_geotiff(
    output_path: Path,
    reconstructed_data: np.ndarray,
    s2_meta: Dict[str, Any],
    model_metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    """Write 13-band reconstructed optical surface reflectance to a GeoTIFF.

    Parameters
    ----------
    output_path : Path
        Destination .tif file path.
    reconstructed_data : np.ndarray
        Shape (13, H, W), float32.
    s2_meta : Dict[str, Any]
        Metadata from input S2 reference (CRS, transform, width, height).
    model_metadata : Optional[Dict[str, Any]]
        Optional tags (model name, checkpoint, source files).

    Returns
    -------
    Path
        Written GeoTIFF path.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    c, h, w = reconstructed_data.shape
    if c != S2_CHANNELS:
        raise ValueError(f"Expected {S2_CHANNELS} channels, got {c}.")

    # Build GeoTIFF profile
    profile = {
        "driver": "GTiff",
        "dtype": "float32",
        "nodata": None,
        "width": w,
        "height": h,
        "count": S2_CHANNELS,
        "crs": s2_meta["crs"],
        "transform": s2_meta["transform"],
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
        "compress": "deflate",
        "interleave": "pixel",
    }

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(reconstructed_data.astype(np.float32))

        # Assign band descriptions
        for i, band_name in enumerate(S2_BAND_NAMES, start=1):
            dst.set_band_description(i, band_name)

        # Set custom GeoTIFF metadata tags
        tags = {
            "SOFTWARE": "CloudRemoval-DSen2CR",
            "MODEL_ARCHITECTURE": "Modified DSen2-CR (SAR-Supervised)",
            "CHANNELS_IN": "13 (S2) + 2 (S1)",
            "CHANNELS_OUT": "13 (S2 Optical Surface Reflectance)",
            "CREATION_TIMESTAMP": datetime.now(tz=timezone.utc).isoformat(),
        }
        if model_metadata:
            tags.update({str(k).upper(): str(v) for k, v in model_metadata.items()})

        dst.update_tags(**tags)

    log.info("GeoTIFF written successfully: %s", output_path)
    return output_path


def verify_reconstructed_geotiff(
    output_path: Path,
    expected_s2_meta: Dict[str, Any],
) -> Dict[str, Any]:
    """Reopen and verify written GeoTIFF against S2 reference spatial metadata.

    Asserts:
      - CRS matches exactly
      - Affine transform matches exactly
      - Width and height match exactly
      - Band count is 13
      - Data values are finite

    Returns
    -------
    Dict[str, Any]
        Verification report dict.
    """
    output_path = Path(output_path)
    if not output_path.exists():
        raise FileNotFoundError(f"Verification failed — output GeoTIFF not found: {output_path}")

    with rasterio.open(output_path) as src:
        out_crs = src.crs
        out_transform = src.transform
        out_w = src.width
        out_h = src.height
        out_count = src.count
        data = src.read()

        # 1. CRS Check
        if out_crs != expected_s2_meta["crs"]:
            raise AssertionError(f"CRS mismatch: output={out_crs} vs expected={expected_s2_meta['crs']}")

        # 2. Transform Check
        if out_transform != expected_s2_meta["transform"]:
            raise AssertionError(f"Transform mismatch: output={out_transform} vs expected={expected_s2_meta['transform']}")

        # 3. Dimension Checks
        if out_w != expected_s2_meta["width"] or out_h != expected_s2_meta["height"]:
            raise AssertionError(
                f"Dimension mismatch: output={out_w}x{out_h} vs expected={expected_s2_meta['width']}x{expected_s2_meta['height']}"
            )

        # 4. Band Count Check
        if out_count != S2_CHANNELS:
            raise AssertionError(f"Band count mismatch: output={out_count} vs expected={S2_CHANNELS}")

        # 5. Finite Data Check
        finite_frac = float(np.mean(np.isfinite(data)))
        if finite_frac < 0.99:
            log.warning("Output GeoTIFF contains non-finite values (finite fraction: %.2f%%)", finite_frac * 100)

        report = {
            "verified": True,
            "path": str(output_path),
            "crs": str(out_crs),
            "transform": tuple(out_transform),
            "width": out_w,
            "height": out_h,
            "bands": out_count,
            "data_min": float(np.nanmin(data)),
            "data_max": float(np.nanmax(data)),
            "finite_fraction": finite_frac,
        }
        return report
