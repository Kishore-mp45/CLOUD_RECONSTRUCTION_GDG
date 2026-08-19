"""
api/routes/scenes.py
====================
Scene listing, detail, and imagery preview endpoints.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Literal
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from api.dependencies import get_db
from api.schemas.scenes import SceneListResponse, SceneDetail
from api.services.scene_service import get_scenes, get_scene_by_id, get_or_generate_scene_preview

router = APIRouter(prefix="/scenes", tags=["Scenes"])


@router.get("", response_model=SceneListResponse, summary="List available satellite scenes")
def list_scenes(
    eligible: Optional[bool] = Query(None, description="Filter only eligible cloudy scenes (true) or non-eligible (false)"),
    limit: int = Query(50, ge=1, le=200, description="Max records to return"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
    db: Session = Depends(get_db),
) -> SceneListResponse:
    """List satellite scenes available for selection with cloud-density metadata."""
    return get_scenes(db=db, eligible_only=eligible, limit=limit, offset=offset)


@router.get("/{scene_id}", response_model=SceneDetail, summary="Get scene metadata details")
def get_scene(
    scene_id: str,
    db: Session = Depends(get_db),
) -> SceneDetail:
    """Retrieve detailed spatial and cloud metadata for a specific scene."""
    detail = get_scene_by_id(db=db, scene_id=scene_id)
    if not detail:
        raise HTTPException(status_code=404, detail=f"Scene '{scene_id}' not found.")
    return detail


@router.get("/{scene_id}/preview/{modality}", summary="Get scene PNG preview image")
def get_scene_preview_image(
    scene_id: str,
    modality: Literal["s2", "s1", "target"] = "s2",
    db: Session = Depends(get_db),
) -> FileResponse:
    """Return a true-color Sentinel-2 or Sentinel-1 SAR preview image for frontend rendering."""
    preview_path = get_or_generate_scene_preview(db=db, scene_id=scene_id, modality=modality)
    if not preview_path or not preview_path.exists():
        raise HTTPException(status_code=404, detail=f"Preview for scene '{scene_id}' ({modality}) not found.")

    return FileResponse(
        path=preview_path,
        media_type="image/png",
        filename=f"{scene_id}_{modality}.png",
    )
