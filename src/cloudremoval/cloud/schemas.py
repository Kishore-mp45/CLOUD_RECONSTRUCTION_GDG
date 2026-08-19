"""
src/cloudremoval/cloud/schemas.py
==================================
Pydantic schemas and dataclasses for Phase 7 Cloud-Density Analysis & Filtering.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Literal
from pydantic import BaseModel, Field


class CloudFilterConfig(BaseModel):
    """Configuration thresholds for pixel-level and scene-level cloud filtering."""

    pixel_probability_threshold: float = Field(
        default=60.0,
        ge=0.0,
        le=100.0,
        description="Pixel-level cloud-probability threshold (0-100%%). Pixels >= threshold are cloudy.",
    )
    scene_density_threshold: float = Field(
        default=60.0,
        ge=0.0,
        le=100.0,
        description="Scene-level cloud-density threshold (0-100%%). Scenes >= threshold are eligible for cloud removal.",
    )
    min_valid_pixel_ratio: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description="Minimum ratio of valid pixels in ROI (0-1). Below this, ROI is considered invalid.",
    )
    cloud_source_name: str = Field(
        default="Sentinel-2 Cloud Probability (s2cloudless / Earth Engine)",
        description="Name of the satellite cloud-probability provider.",
    )


class SceneMetadata(BaseModel):
    """Metadata describing an analyzed satellite scene."""

    scene_id: str
    acquisition_time: Optional[str] = None
    roi_id: Optional[str] = None
    cloud_source: str = "Sentinel-2 Cloud Probability (s2cloudless / Earth Engine)"
    extra: Dict[str, Any] = Field(default_factory=dict)


class CloudDensityResult(BaseModel):
    """Structured result of single scene cloud-density analysis."""

    scene_id: str
    acquisition_time: Optional[str] = None
    roi_id: Optional[str] = None
    cloud_probability_threshold: float
    scene_threshold_percent: float
    total_pixels: int
    valid_pixels: int
    cloudy_pixels: int
    cloud_density_percent: float
    mean_cloud_probability: float
    passes_threshold: bool
    status: Literal["eligible", "filtered", "error"]
    error_message: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.now(tz=timezone.utc).isoformat())
    cloud_source: str = "Sentinel-2 Cloud Probability (s2cloudless / Earth Engine)"


class BatchFilterResult(BaseModel):
    """Summary of batch scene analysis and filtering."""

    total_scenes: int
    eligible_count: int
    filtered_count: int
    error_count: int
    config: CloudFilterConfig
    eligible_scenes: List[CloudDensityResult] = Field(default_factory=list)
    filtered_scenes: List[CloudDensityResult] = Field(default_factory=list)
    error_scenes: List[CloudDensityResult] = Field(default_factory=list)
    timestamp: str = Field(default_factory=lambda: datetime.now(tz=timezone.utc).isoformat())
