"""
cloudremoval.models
====================
DSen2-CR model package.

Public API:
    build_model()           -> DSen2CR
    build_loss()            -> CloudRemovalLoss
    DSen2CRConfig           -> model configuration dataclass
    save_checkpoint()       -> save latest/best/epoch checkpoints
    load_checkpoint()       -> restore model from checkpoint
"""

from cloudremoval.models.model_config import (
    DSen2CRConfig,
    DEFAULT_CONFIG,
    S2_CHANNELS,
    S1_CHANNELS,
    TARGET_CHANNELS,
    PATCH_SIZE,
)
from cloudremoval.models.dsen2cr import DSen2CR, build_model
from cloudremoval.models.losses import CloudRemovalLoss, build_loss
from cloudremoval.models.checkpoint import (
    save_checkpoint,
    load_checkpoint,
    list_checkpoints,
    get_best_checkpoint,
)
from cloudremoval.models.fusion import SARFusion

__all__ = [
    "DSen2CRConfig",
    "DEFAULT_CONFIG",
    "S2_CHANNELS",
    "S1_CHANNELS",
    "TARGET_CHANNELS",
    "PATCH_SIZE",
    "DSen2CR",
    "build_model",
    "CloudRemovalLoss",
    "build_loss",
    "SARFusion",
    "save_checkpoint",
    "load_checkpoint",
    "list_checkpoints",
    "get_best_checkpoint",
]
