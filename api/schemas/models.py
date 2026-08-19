"""
api/schemas/models.py
=====================
Pydantic schemas for model information.
"""

from __future__ import annotations

from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field


class ModelInfoResponse(BaseModel):
    model_name: str
    architecture: str
    version: str
    checkpoint_name: str
    checkpoint_status: str
    best_epoch: Optional[int] = None
    input_modalities: List[str]
    s2_channels: int
    s1_channels: int
    target_channels: int
    parameter_count: str
    patch_size: int
    normalization_version: str
