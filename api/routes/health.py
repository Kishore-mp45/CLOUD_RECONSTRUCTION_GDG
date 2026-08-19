"""
api/routes/health.py
====================
Health check endpoint.
"""

from __future__ import annotations

import sys
from pathlib import Path
from fastapi import APIRouter
import torch

from api.schemas.health import HealthResponse, StorageStatus
from cloudremoval.config import get_settings

router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthResponse, summary="Backend health status")
def health_check() -> HealthResponse:
    """Return backend status, CUDA availability, GPU name, and model readiness."""
    settings = get_settings()

    cuda_avail = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda_avail else None

    ckpt_path = Path(settings.CHECKPOINT_DIR) / settings.BEST_CHECKPOINT_NAME
    ckpt_ready = ckpt_path.exists()

    db_path = Path(settings.DB_PATH)

    storage = StorageStatus(
        checkpoints_dir=Path(settings.CHECKPOINT_DIR).exists(),
        outputs_dir=Path(settings.OUTPUT_DIR).exists(),
        database_connected=db_path.exists() or db_path.parent.exists(),
    )

    return HealthResponse(
        status="ok",
        version="1.0.0",
        python_version=sys.version.split()[0],
        torch_version=torch.__version__,
        cuda_available=cuda_avail,
        gpu_name=gpu_name,
        model_checkpoint_available=ckpt_ready,
        active_model="Modified DSen2-CR (SAR-Supervised)",
        storage=storage,
    )
