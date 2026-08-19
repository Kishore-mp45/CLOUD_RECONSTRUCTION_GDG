"""
api/schemas/metrics.py
======================
Pydantic schemas for quantitative metrics.
"""

from __future__ import annotations

from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field


class MetricStats(BaseModel):
    mean: float
    median: float
    std: float
    min: float
    max: float


class MetricsResponse(BaseModel):
    available: bool = Field(..., description="Whether ground-truth evaluation metrics are available")
    reason: Optional[str] = None
    result_id: Optional[str] = None
    psnr: Optional[float] = None
    ssim: Optional[float] = None
    mae: Optional[float] = None
    rmse: Optional[float] = None
    sam: Optional[float] = None
    latency_ms: Optional[float] = None
    aggregate_test_metrics: Optional[Dict[str, Any]] = None
