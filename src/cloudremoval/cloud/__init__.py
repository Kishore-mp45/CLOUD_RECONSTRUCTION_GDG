"""
cloudremoval.cloud
==================
Modular Cloud-Density Analysis and Scene Filtering Layer (Phase 7).

Provides robust, ROI-aware calculation of cloud density from satellite
cloud-probability sources and applies configurable pixel-level and
scene-level filtering thresholds.

Public API:
    calculate_cloud_density    -> Compute ROI cloud-density percentage
    validate_cloud_probability -> Validate & clean 2D cloud probability array
    analyze_scene              -> Single-scene cloud analysis & eligibility gate
    filter_scenes_batch        -> Batch scene analysis with deterministic sorting
    CloudFilterConfig          -> Configuration threshold schema
    CloudDensityResult         -> Single-scene result schema
    BatchFilterResult          -> Batch summary schema
    SceneMetadata              -> Scene metadata schema
    CloudDataValidationError   -> Data validation exception
    InvalidROIDataError        -> ROI geometry/nodata exception
"""

from cloudremoval.cloud.schemas import (
    CloudFilterConfig,
    CloudDensityResult,
    BatchFilterResult,
    SceneMetadata,
)
from cloudremoval.cloud.density import (
    calculate_cloud_density,
    validate_cloud_probability,
    CloudDataValidationError,
    InvalidROIDataError,
)
from cloudremoval.cloud.filtering import (
    analyze_scene,
    filter_scenes_batch,
)

__all__ = [
    "calculate_cloud_density",
    "validate_cloud_probability",
    "analyze_scene",
    "filter_scenes_batch",
    "CloudFilterConfig",
    "CloudDensityResult",
    "BatchFilterResult",
    "SceneMetadata",
    "CloudDataValidationError",
    "InvalidROIDataError",
]
