"""
api.services
============
Business logic services for FastAPI backend.
"""

from api.services.db_service import seed_scenes_from_manifests
from api.services.scene_service import get_scenes, get_scene_by_id
from api.services.inference_service import execute_inference_job, IneligibleSceneError, SceneNotFoundError, CheckpointNotFoundError
from api.services.metrics_service import get_metrics_for_result
from api.services.model_service import get_active_model_info

__all__ = [
    "seed_scenes_from_manifests",
    "get_scenes",
    "get_scene_by_id",
    "execute_inference_job",
    "IneligibleSceneError",
    "SceneNotFoundError",
    "CheckpointNotFoundError",
    "get_metrics_for_result",
    "get_active_model_info",
]
