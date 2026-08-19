"""
api/routes/results.py
=====================
Inference result retrieval endpoint.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from api.dependencies import get_db
from api.db.models import InferenceJob, Result
from api.schemas.results import ResultResponse

router = APIRouter(prefix="/results", tags=["Results"])


@router.get("/{id}", response_model=ResultResponse, summary="Get inference result by Job or Result ID")
def get_result(
    id: str,
    db: Session = Depends(get_db),
) -> ResultResponse:
    """Retrieve spatial metadata and download URLs for a completed inference job."""
    # Look up by result_id or job_id
    res = db.query(Result).filter((Result.result_id == id) | (Result.job_id == id)).first()
    if not res:
        # Check if job is still in progress or failed
        job = db.query(InferenceJob).filter(InferenceJob.job_id == id).first()
        if job:
            raise HTTPException(
                status_code=400,
                detail=f"Inference job '{id}' is currently in state '{job.status}'. Output not yet ready.",
            )
        raise HTTPException(status_code=404, detail=f"Inference result '{id}' not found.")

    geotiff_url = f"/download?result_id={res.result_id}&file_type=geotiff"
    preview_url = f"/download?result_id={res.result_id}&file_type=png"

    infer_time = res.job.inference_time_s if res.job else None

    return ResultResponse(
        result_id=res.result_id,
        job_id=res.job_id,
        scene_id=res.scene_id,
        status="completed",
        crs=res.crs,
        width=res.width,
        height=res.height,
        resolution=res.resolution,
        band_count=res.band_count,
        geotiff_download_url=geotiff_url,
        preview_png_download_url=preview_url,
        inference_time_s=infer_time,
        created_at=res.created_at.isoformat(),
    )
