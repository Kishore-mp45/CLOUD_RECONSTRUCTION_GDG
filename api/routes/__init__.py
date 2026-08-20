"""
api.routes
==========
API route controllers for Phase 8, 10, and 11.
"""

from api.routes.health import router as health_router
from api.routes.scenes import router as scenes_router
from api.routes.inference import router as inference_router
from api.routes.results import router as results_router
from api.routes.metrics import router as metrics_router
from api.routes.models import router as models_router
from api.routes.downloads import router as downloads_router
from api.routes.history import router as history_router
from api.routes.live import router as live_router

__all__ = [
    "health_router",
    "scenes_router",
    "inference_router",
    "results_router",
    "metrics_router",
    "models_router",
    "downloads_router",
    "history_router",
    "live_router",
]
