"""
src/cloudremoval/database/schemas.py
====================================
Pydantic schemas for data validation and API serialization in Phase 9.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional, Any, Dict, List
from pydantic import BaseModel, ConfigDict, Field


# --- Scene Schemas ---
class SceneBase(BaseModel):
    scene_id: str
    roi_id: str
    acquisition_time: str
    source_provider: str = "ALLClear"
    s2_path: str
    s1_path: str
    target_path: Optional[str] = None
    cloud_density_percent: float = Field(..., ge=0.0, le=100.0)
    cloud_probability_threshold: float = 60.0
    is_eligible: bool = False
    crs: str
    width: int
    height: int
    resolution: float
    bounds_json: Optional[str] = None
    extra_metadata: Optional[str] = None


class SceneCreate(SceneBase):
    external_scene_id: Optional[str] = None


class SceneRead(SceneBase):
    external_scene_id: Optional[str] = None
    created_at: datetime
    updated_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)


# --- Model Version Schemas ---
class ModelVersionBase(BaseModel):
    model_name: str
    architecture: str
    version: str
    checkpoint_path: str
    checkpoint_hash: Optional[str] = None
    best_epoch: Optional[int] = None
    s2_channels: int = 13
    s1_channels: int = 2
    target_channels: int = 13
    normalization_version: str = "v1"
    training_config_json: Optional[str] = None
    is_active: bool = True


class ModelVersionCreate(ModelVersionBase):
    pass


class ModelVersionRead(ModelVersionBase):
    id: int
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


# --- Inference Job Schemas ---
class InferenceJobCreate(BaseModel):
    job_id: str
    scene_id: str
    model_version_id: Optional[int] = None
    tile_size: int = 256
    overlap: int = 64
    batch_size: int = 4
    model_name: str = "Modified DSen2-CR (SAR-Supervised)"
    checkpoint_name: str = "best_model.pth"


class InferenceJobUpdate(BaseModel):
    status: Optional[str] = None
    error_message: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    inference_duration_s: Optional[float] = None
    total_duration_s: Optional[float] = None


class InferenceJobRead(BaseModel):
    job_id: str
    scene_id: str
    model_version_id: Optional[int] = None
    status: str
    tile_size: int
    overlap: int
    batch_size: int
    model_name: str
    checkpoint_name: str
    error_message: Optional[str] = None
    requested_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    inference_duration_s: Optional[float] = None
    total_duration_s: Optional[float] = None
    created_at: datetime
    updated_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)


# --- Result Schemas ---
class ResultCreate(BaseModel):
    result_id: str
    job_id: str
    scene_id: str
    geotiff_path: str
    preview_png_path: str
    metadata_path: str
    crs: str
    width: int
    height: int
    resolution: float
    band_count: int = 13
    bounds_json: Optional[str] = None
    inference_time_s: float
    total_time_s: float
    peak_vram_gb: Optional[float] = None


class ResultRead(ResultCreate):
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


# --- Metric Schemas ---
class MetricCreate(BaseModel):
    result_id: str
    psnr: Optional[float] = None
    ssim: Optional[float] = None
    mae: Optional[float] = None
    rmse: Optional[float] = None
    sam: Optional[float] = None
    is_available: bool = False
    evaluation_source: str = "ground_truth_target"


class MetricRead(MetricCreate):
    id: int
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


# --- Processing History Schemas ---
class ProcessingHistoryCreate(BaseModel):
    entity_type: str
    entity_id: str
    action: str
    status: str = "success"
    message: Optional[str] = None
    duration_s: Optional[float] = None
    details_json: Optional[str] = None


class ProcessingHistoryRead(ProcessingHistoryCreate):
    id: int
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)
