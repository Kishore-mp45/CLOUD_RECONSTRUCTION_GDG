"""
api/schemas/results.py
======================
Pydantic schemas for reconstructed geospatial results.
"""

from __future__ import annotations

from typing import Optional, Dict, Any
from pydantic import BaseModel, Field


class ResultResponse(BaseModel):
    result_id: str
    job_id: str
    scene_id: str
    status: str
    crs: str
    width: int
    height: int
    resolution: float
    band_count: int
    geotiff_download_url: str
    preview_png_download_url: str
    inference_time_s: Optional[float] = None
    created_at: str
