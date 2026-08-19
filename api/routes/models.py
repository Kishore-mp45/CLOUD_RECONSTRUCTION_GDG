"""
api/routes/models.py
====================
Model architecture and checkpoint inspection endpoint.
"""

from __future__ import annotations

from fastapi import APIRouter
from api.schemas.models import ModelInfoResponse
from api.services.model_service import get_active_model_info

router = APIRouter(prefix="/models", tags=["Models"])


@router.get("", response_model=ModelInfoResponse, summary="Get active DSen2-CR model specifications")
def get_models() -> ModelInfoResponse:
    """Return active model architecture, parameters, input channels, and checkpoint metadata."""
    return get_active_model_info()
