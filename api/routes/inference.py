"""
api/routes/inference.py
=======================
Inference triggering endpoint.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from api.dependencies import get_db
from api.schemas.inference import InferenceRequest, InferenceJobResponse
from api.services.inference_service import (
    execute_inference_job,
    IneligibleSceneError,
    SceneNotFoundError,
    CheckpointNotFoundError,
)

router = APIRouter(prefix="/inference", tags=["Inference"])


@router.post("", response_model=InferenceJobResponse, status_code=status.HTTP_201_CREATED, summary="Trigger cloud removal inference")
def run_inference(
    request: InferenceRequest,
    db: Session = Depends(get_db),
) -> InferenceJobResponse:
    """Create and run a local geospatial inference job on an eligible satellite scene."""
    try:
        job = execute_inference_job(db=db, request=request)
        return job
    except SceneNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    except IneligibleSceneError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except (FileNotFoundError, CheckpointNotFoundError) as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Inference execution failed: {exc}")
