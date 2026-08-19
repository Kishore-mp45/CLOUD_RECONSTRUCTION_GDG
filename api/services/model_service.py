"""
api/services/model_service.py
=============================
Model architecture and checkpoint inspection service.
"""

from __future__ import annotations

from pathlib import Path
from api.schemas.models import ModelInfoResponse
from cloudremoval.config import get_settings
from cloudremoval.models.model_config import (
    S2_CHANNELS,
    S1_CHANNELS,
    TARGET_CHANNELS,
    PATCH_SIZE,
)


def get_active_model_info() -> ModelInfoResponse:
    """Retrieve specifications for the active DSen2-CR checkpoint."""
    settings = get_settings()
    ckpt_path = Path(settings.CHECKPOINT_DIR) / settings.BEST_CHECKPOINT_NAME
    ckpt_exists = ckpt_path.exists()

    return ModelInfoResponse(
        model_name="Modified DSen2-CR (SAR-Supervised)",
        architecture="Deep Residual Network with Early SAR Concatenation",
        version="1.0.0",
        checkpoint_name=settings.BEST_CHECKPOINT_NAME,
        checkpoint_status="ready" if ckpt_exists else "missing",
        best_epoch=44,
        input_modalities=[
            "Sentinel-2 Optical (13 multispectral bands)",
            "Sentinel-1 SAR Radar (VV, VH backscatter)",
        ],
        s2_channels=S2_CHANNELS,
        s1_channels=S1_CHANNELS,
        target_channels=TARGET_CHANNELS,
        parameter_count="18.95M Parameters",
        patch_size=PATCH_SIZE,
        normalization_version="v1",
    )
