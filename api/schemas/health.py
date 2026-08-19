"""
api/schemas/health.py
=====================
Health check response schemas.
"""

from __future__ import annotations

from typing import Optional, Dict, Any
from pydantic import BaseModel, Field


class StorageStatus(BaseModel):
    checkpoints_dir: bool = True
    outputs_dir: bool = True
    database_connected: bool = True


class HealthResponse(BaseModel):
    status: str = Field(default="ok", description="Overall health status")
    version: str = Field(default="1.0.0", description="Application version")
    python_version: str
    torch_version: str
    cuda_available: bool
    gpu_name: Optional[str] = None
    model_checkpoint_available: bool
    active_model: str = "Modified DSen2-CR (SAR-Supervised)"
    storage: StorageStatus = Field(default_factory=StorageStatus)
