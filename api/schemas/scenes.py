"""
api/schemas/scenes.py
=====================
Pydantic schemas for satellite scene listing and details.
"""

from __future__ import annotations

from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field


class SceneSummary(BaseModel):
    scene_id: str
    roi_id: str
    acquisition_time: Optional[str] = None
    cloud_density_percent: float
    cloud_probability_threshold: float = 60.0
    is_eligible: bool
    has_s2: bool = True
    has_s1: bool = True
    has_target: bool = False
    source_provider: str = "ALLClear"


class SceneDetail(BaseModel):
    scene_id: str
    roi_id: str
    acquisition_time: Optional[str] = None
    cloud_density_percent: float
    cloud_probability_threshold: float = 60.0
    is_eligible: bool
    crs: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    resolution: Optional[float] = None
    s2_available: bool = True
    s1_available: bool = True
    target_available: bool = False
    source_provider: str = "ALLClear"
    extra: Dict[str, Any] = Field(default_factory=dict)


class SceneListResponse(BaseModel):
    total_count: int
    eligible_count: int
    filtered_count: int
    scenes: List[SceneSummary]
