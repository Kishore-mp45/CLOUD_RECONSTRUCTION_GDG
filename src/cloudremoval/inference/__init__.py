"""
cloudremoval.inference
======================
Production local geospatial inference pipeline for Sentinel-2 cloud removal.

Public API:
    GeospatialInferencePipeline -> End-to-end inference pipeline
    TiledInferenceEngine        -> Sliding-window tiled inference engine
    write_reconstructed_geotiff -> Multi-band GeoTIFF writer
    verify_reconstructed_geotiff-> Post-write spatial verification
    create_inference_preview    -> True-color side-by-side PNG preview
"""

from cloudremoval.inference.tiled_inference import TiledInferenceEngine, create_2d_blending_window
from cloudremoval.inference.writer import write_reconstructed_geotiff, verify_reconstructed_geotiff, S2_BAND_NAMES
from cloudremoval.inference.preview import create_inference_preview
from cloudremoval.inference.pipeline import GeospatialInferencePipeline

__all__ = [
    "GeospatialInferencePipeline",
    "TiledInferenceEngine",
    "create_2d_blending_window",
    "write_reconstructed_geotiff",
    "verify_reconstructed_geotiff",
    "create_inference_preview",
    "S2_BAND_NAMES",
]
