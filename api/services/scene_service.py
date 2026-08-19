"""
api/services/scene_service.py
=============================
Service for querying and retrieving satellite scene records.
"""

from __future__ import annotations

import json
from typing import Optional, List
from sqlalchemy.orm import Session

from api.db.models import Scene
from api.schemas.scenes import SceneSummary, SceneDetail, SceneListResponse


def get_scenes(
    db: Session,
    eligible_only: Optional[bool] = None,
    limit: int = 50,
    offset: int = 0,
) -> SceneListResponse:
    """Retrieve paginated scenes with optional eligibility filtering."""
    query = db.query(Scene)

    if eligible_only is True:
        query = query.filter(Scene.is_eligible.is_(True))
    elif eligible_only is False:
        query = query.filter(Scene.is_eligible.is_(False))

    total_count = query.count()
    eligible_count = db.query(Scene).filter(Scene.is_eligible.is_(True)).count()
    filtered_count = total_count - eligible_count

    scenes_orm = query.order_by(Scene.cloud_density_percent.desc()).offset(offset).limit(limit).all()

    summaries = [
        SceneSummary(
            scene_id=s.scene_id,
            roi_id=s.roi_id,
            acquisition_time=s.acquisition_time,
            cloud_density_percent=s.cloud_density_percent,
            cloud_probability_threshold=s.cloud_probability_threshold,
            is_eligible=s.is_eligible,
            has_s2=bool(s.s2_path),
            has_s1=bool(s.s1_path),
            has_target=bool(s.target_path),
        )
        for s in scenes_orm
    ]

    return SceneListResponse(
        total_count=total_count,
        eligible_count=eligible_count,
        filtered_count=filtered_count,
        scenes=summaries,
    )


def get_scene_by_id(db: Session, scene_id: str) -> Optional[SceneDetail]:
    """Retrieve detailed metadata for a single scene."""
    scene = db.query(Scene).filter(Scene.scene_id == scene_id).first()
    if not scene:
        return None

    extra = {}
    if scene.extra_metadata:
        try:
            extra = json.loads(scene.extra_metadata)
        except Exception:
            pass

    return SceneDetail(
        scene_id=scene.scene_id,
        roi_id=scene.roi_id,
        acquisition_time=scene.acquisition_time,
        cloud_density_percent=scene.cloud_density_percent,
        cloud_probability_threshold=scene.cloud_probability_threshold,
        is_eligible=scene.is_eligible,
        crs=scene.crs,
        width=scene.width,
        height=scene.height,
        resolution=scene.resolution,
        s2_available=bool(scene.s2_path),
        s1_available=bool(scene.s1_path),
        target_available=bool(scene.target_path),
        extra=extra,
    )
