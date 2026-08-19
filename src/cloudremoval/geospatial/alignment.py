"""
src/cloudremoval/geospatial/alignment.py
=========================================
Geospatial metadata validation and in-memory S1-to-S2 alignment for Phase 6.

Key Capabilities:
  - Strict GeoTIFF metadata validation (CRS, Transform, Dimensions, Band count)
  - In-memory S1 resampling & reprojection to S2 reference grid using rasterio
  - Zero modification of original source files
  - Robust nodata / NaN handling
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.warp import reproject

from cloudremoval.models.model_config import S2_CHANNELS, S1_CHANNELS

log = logging.getLogger(__name__)


class GeospatialValidationError(ValueError):
    """Raised when GeoTIFF metadata fails validation."""


def validate_geotiff(
    path: Path,
    expected_bands: int,
    modality_name: str = "GeoTIFF",
) -> Dict[str, Any]:
    """Validate that a GeoTIFF exists, is readable, and contains valid spatial metadata.

    Parameters
    ----------
    path : Path
        Path to the GeoTIFF file.
    expected_bands : int
        Expected number of raster bands (e.g. 13 for S2, 2 for S1).
    modality_name : str
        Human-readable name for logging (e.g. 'Sentinel-2 Optical').

    Returns
    -------
    Dict[str, Any]
        Metadata dictionary extracted from the GeoTIFF.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{modality_name} file not found: {path}")

    try:
        with rasterio.open(path) as src:
            crs = src.crs
            transform = src.transform
            width = src.width
            height = src.height
            count = src.count
            dtype = src.dtypes[0]
            bounds = src.bounds
            res = src.res

            if not crs:
                raise GeospatialValidationError(f"{modality_name} ({path.name}) has no valid CRS.")

            if count != expected_bands:
                raise GeospatialValidationError(
                    f"{modality_name} ({path.name}) has {count} bands, expected {expected_bands}."
                )

            if width <= 0 or height <= 0:
                raise GeospatialValidationError(
                    f"{modality_name} ({path.name}) has invalid dimensions: {width}x{height}."
                )

            meta = {
                "path": str(path),
                "crs": crs,
                "transform": transform,
                "width": width,
                "height": height,
                "count": count,
                "dtype": dtype,
                "bounds": bounds,
                "resolution": res,
                "nodata": src.nodata,
            }
            return meta

    except Exception as exc:
        if isinstance(exc, (GeospatialValidationError, FileNotFoundError)):
            raise
        raise GeospatialValidationError(f"Failed to read {modality_name} ({path}): {exc}") from exc


def load_and_align_s1_to_s2(
    s2_path: Path,
    s1_path: Path,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Load S2 optical and S1 SAR GeoTIFFs, aligning S1 to the S2 reference grid in-memory.

    Parameters
    ----------
    s2_path : Path
        Path to Sentinel-2 optical GeoTIFF (13 bands).
    s1_path : Path
        Path to Sentinel-1 SAR GeoTIFF (2 bands).

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, Dict[str, Any]]
        - s2_data: np.ndarray of shape (13, H, W), float32
        - s1_aligned: np.ndarray of shape (2, H, W), float32 (resampled to S2 grid)
        - s2_meta: Dict containing S2 CRS, transform, dimensions, and profile
    """
    s2_path = Path(s2_path)
    s1_path = Path(s1_path)

    # 1. Validate inputs
    s2_meta = validate_geotiff(s2_path, expected_bands=S2_CHANNELS, modality_name="Sentinel-2")
    s1_meta = validate_geotiff(s1_path, expected_bands=S1_CHANNELS, modality_name="Sentinel-1")

    # 2. Read Sentinel-2 data
    with rasterio.open(s2_path) as s2_src:
        s2_data = s2_src.read().astype(np.float32)  # (13, H, W)
        s2_profile = s2_src.profile.copy()

    # Clean non-finite values in S2
    s2_data = np.nan_to_num(s2_data, nan=0.0, posinf=1.0, neginf=0.0)

    # 3. Read Sentinel-1 data & Align to S2 Grid
    with rasterio.open(s1_path) as s1_src:
        s1_raw = s1_src.read().astype(np.float32)  # (2, H_s1, W_s1)

        # Check if alignment is already identical
        identical_grid = (
            s1_src.crs == s2_src.crs
            and s1_src.transform == s2_src.transform
            and s1_src.width == s2_src.width
            and s1_src.height == s2_src.height
        )

        if identical_grid:
            s1_aligned = s1_raw
        else:
            log.info("S1 grid differs from S2 reference. Reprojecting S1 in-memory to match S2...")
            s1_aligned = np.zeros((S1_CHANNELS, s2_meta["height"], s2_meta["width"]), dtype=np.float32)

            for b in range(S1_CHANNELS):
                reproject(
                    source=s1_raw[b],
                    destination=s1_aligned[b],
                    src_transform=s1_src.transform,
                    src_crs=s1_src.crs,
                    dst_transform=s2_src.transform,
                    dst_crs=s2_src.crs,
                    resampling=Resampling.bilinear,
                    src_nodata=s1_src.nodata,
                    dst_nodata=0.0,
                )

    # Clean non-finite values in S1
    s1_aligned = np.nan_to_num(s1_aligned, nan=0.0, posinf=0.0, neginf=0.0)

    # Attach profile to meta
    s2_meta["profile"] = s2_profile

    return s2_data, s1_aligned, s2_meta
