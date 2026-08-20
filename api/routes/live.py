"""Endpoints for fetching and registering live Earth Engine imagery."""
from __future__ import annotations

from datetime import date
from pydantic import BaseModel, Field
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from api.dependencies import get_db
from api.services.earth_engine_service import LiveFetchError, fetch_live_scene
from api.services.scene_service import get_scene_by_id

router = APIRouter(prefix="/live", tags=["Live imagery"])


class LiveFetchRequest(BaseModel):
    location: str = Field(min_length=3, description="Latitude,longitude or a place name")
    acquisition_date: date


@router.post("/fetch", status_code=status.HTTP_201_CREATED)
def fetch(request: LiveFetchRequest, db: Session = Depends(get_db)):
    try:
        scene = fetch_live_scene(db, request.location, request.acquisition_date)
        return {"scene": get_scene_by_id(db, scene.scene_id), "source": "live"}
    except LiveFetchError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))
