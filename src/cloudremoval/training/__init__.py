"""
cloudremoval.training
======================
Training loop, metrics, early stopping, logging, and related utilities.

Public API:
    Trainer             -> Phase 4 full training loop
    EarlyStopping       -> patience-based early stopping
    TrainingLogger      -> CSV + JSON + file log management
    run_preflight       -> pre-flight checks before training starts
    compute_all_metrics -> PSNR / SSIM / MAE / RMSE
    MetricTracker       -> batch-level metric accumulator
"""

from cloudremoval.training.trainer import Trainer, NaNLossError
from cloudremoval.training.early_stopping import EarlyStopping
from cloudremoval.training.train_logger import TrainingLogger
from cloudremoval.training.preflight import run_preflight, PreflightError
from cloudremoval.training.metrics import (
    compute_all_metrics,
    compute_psnr,
    compute_ssim_simple,
    compute_mae,
    compute_rmse,
    MetricTracker,
)

__all__ = [
    "Trainer",
    "NaNLossError",
    "EarlyStopping",
    "TrainingLogger",
    "run_preflight",
    "PreflightError",
    "compute_all_metrics",
    "compute_psnr",
    "compute_ssim_simple",
    "compute_mae",
    "compute_rmse",
    "MetricTracker",
]
