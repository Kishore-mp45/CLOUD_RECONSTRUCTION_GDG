"""
api/routes/downloads.py
=======================
Secure file download endpoint with path-traversal protection and audit logging.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from api.dependencies import get_db
from api.db.models import Result
from cloudremoval.config import get_settings
from cloudremoval.database.repositories import ProcessingHistoryRepository

router = APIRouter(tags=["Downloads"])


@router.get("/download", summary="Download reconstructed GeoTIFF or preview PNG")
def download_file(
    result_id: str = Query(..., description="ID of the completed inference result"),
    file_type: Literal["geotiff", "png"] = Query("geotiff", description="Type of file to download: 'geotiff' or 'png'"),
    db: Session = Depends(get_db),
) -> FileResponse:
    """Download georeferenced output GeoTIFF or PNG preview with strict path security."""
    settings = get_settings()
    allowed_base_dir = Path(settings.OUTPUT_DIR).resolve()

    res = db.query(Result).filter(Result.result_id == result_id).first()
    if not res:
        raise HTTPException(status_code=404, detail=f"Inference result '{result_id}' not found.")

    if file_type == "geotiff":
        target_path = Path(res.geotiff_path).resolve()
        media_type = "image/tiff"
        filename = f"{res.result_id}_reconstructed.tif"
    elif file_type == "png":
        target_path = Path(res.preview_png_path).resolve()
        media_type = "image/png"
        filename = f"{res.result_id}_preview.png"
    else:
        raise HTTPException(status_code=400, detail="Invalid file_type parameter.")

    # Strict Path Traversal Check
    try:
        target_path.relative_to(allowed_base_dir)
    except ValueError:
        raise HTTPException(
            status_code=403,
            detail="Access denied: requested file is outside the authorized outputs directory.",
        )

    if not target_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Requested {file_type} file does not exist on server.",
        )

    # Log audit event
    ProcessingHistoryRepository.log_event(
        db=db,
        entity_type="download",
        entity_id=result_id,
        action="DOWNLOAD_REQUESTED",
        status="success",
        message=f"Downloaded {file_type} for result {result_id}",
    )

    return FileResponse(
        path=target_path,
        media_type=media_type,
        filename=filename,
    )
