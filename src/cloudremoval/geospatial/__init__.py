"""
cloudremoval.geospatial
========================
Geospatial metadata validation and in-memory S1/S2 alignment.

Public API:
    validate_geotiff            -> Validate GeoTIFF metadata & headers
    load_and_align_s1_to_s2     -> In-memory alignment and resampling of S1 to S2
    GeospatialValidationError   -> Validation error class
"""

from cloudremoval.geospatial.alignment import (
    validate_geotiff,
    load_and_align_s1_to_s2,
    GeospatialValidationError,
)

__all__ = [
    "validate_geotiff",
    "load_and_align_s1_to_s2",
    "GeospatialValidationError",
]
