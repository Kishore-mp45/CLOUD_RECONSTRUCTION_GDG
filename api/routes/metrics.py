"""
api/routes/metrics.py
=====================
Quantitative metrics retrieval endpoint.
"""

from __future__ import annotations

from typing import Optional
from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from api.dependencies import get_db
from api.schemas.metrics import MetricsResponse
from api.services.metrics_service import get_metrics_for_result

router = APIRouter(prefix="/metrics", tags=["Metrics"])


@router.get("", response_model=MetricsResponse, summary="Get evaluation or inference metrics")
def get_metrics(
    result_id: Optional[str] = Query(None, description="Optional result ID to query specific metrics"),
    db: Session = Depends(get_db),
) -> MetricsResponse:
    """Retrieve quantitative metrics (PSNR, SSIM, MAE, RMSE, SAM) or aggregate test benchmarks."""
    return get_metrics_for_result(db=db, result_id=result_id)
