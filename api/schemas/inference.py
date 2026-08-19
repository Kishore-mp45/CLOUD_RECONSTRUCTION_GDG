"""
api/schemas/inference.py
========================
Pydantic schemas for inference requests and job tracking.
"""

from __future__ import annotations

from typing import Optional, Dict, Any
from pydantic import BaseModel, Field


class InferenceRequest(BaseModel):
    scene_id: str = Field(..., description="ID of the eligible satellite scene to reconstruct")
    tile_size: int = Field(default=256, ge=64, le=1024, description="Sliding window tile size")
    overlap: int = Field(default=64, ge=0, le=256, description="Overlap in pixels between adjacent tiles")
    batch_size: int = Field(default=4, ge=1, le=16, description="Tile batch size for GPU inference")


class InferenceJobResponse(BaseModel):
    job_id: str
    scene_id: str
    status: str
    model_name: str
    tile_size: int
    overlap: int
    created_at: str
    completion_time: Optional[str] = None
    inference_time_s: Optional[float] = None
    total_time_s: Optional[float] = None
    peak_vram_gb: Optional[float] = None
    error_message: Optional[str] = None
    result_id: Optional[str] = None
