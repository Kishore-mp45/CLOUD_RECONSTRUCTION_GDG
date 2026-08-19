"""
src/cloudremoval/training/metrics.py
=======================================
Evaluation metrics for cloud removal quality assessment.

All metrics operate on normalised tensors (z-scored, clamped to [-5, 5]).

Metrics:
  PSNR  - Peak Signal-to-Noise Ratio
  SSIM  - Structural Similarity Index (simplified, per-batch)
  MAE   - Mean Absolute Error
  RMSE  - Root Mean Squared Error

All functions:
  - Accept (B, C, H, W) torch tensors
  - Return scalar Python floats
  - Are safe under torch.no_grad()
  - Work on both CPU and CUDA tensors
"""

from __future__ import annotations

import math
import torch


def compute_psnr(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 10.0,
) -> float:
    """Compute PSNR between prediction and target.

    Parameters
    ----------
    pred, target : torch.Tensor
        Shape (B, C, H, W), normalised.
    data_range : float
        Dynamic range of the data.  For z-score normalised data clamped to [-5, 5],
        the range is 10.0.

    Returns
    -------
    float
        PSNR in decibels.
    """
    with torch.no_grad():
        mse = torch.mean((pred.float() - target.float()) ** 2)
        if mse == 0.0:
            return float("inf")
        return float(20.0 * math.log10(data_range) - 10.0 * math.log10(mse.item()))


def compute_mae(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Compute Mean Absolute Error.

    Returns
    -------
    float
        MAE value.
    """
    with torch.no_grad():
        return float(torch.mean(torch.abs(pred.float() - target.float())).item())


def compute_rmse(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Compute Root Mean Squared Error.

    Returns
    -------
    float
        RMSE value.
    """
    with torch.no_grad():
        mse = torch.mean((pred.float() - target.float()) ** 2)
        return float(math.sqrt(mse.item()))


def compute_ssim_simple(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 10.0,
) -> float:
    """Compute a simplified SSIM (global statistics, no sliding window).

    This is faster than the full windowed SSIM and suitable for epoch-level
    logging.  For publication-quality SSIM, use the windowed SSIMLoss version.

    Returns
    -------
    float
        SSIM value in [0, 1].
    """
    with torch.no_grad():
        p = pred.float()
        t = target.float()

        mu_p = p.mean()
        mu_t = t.mean()

        sigma_p = ((p - mu_p) ** 2).mean()
        sigma_t = ((t - mu_t) ** 2).mean()
        sigma_pt = ((p - mu_p) * (t - mu_t)).mean()

        C1 = (0.01 * data_range) ** 2
        C2 = (0.03 * data_range) ** 2

        num   = (2 * mu_p * mu_t + C1) * (2 * sigma_pt + C2)
        denom = (mu_p ** 2 + mu_t ** 2 + C1) * (sigma_p + sigma_t + C2)

        return float((num / (denom + 1e-8)).item())


def compute_all_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 10.0,
) -> dict[str, float]:
    """Compute all metrics in one call.

    Parameters
    ----------
    pred, target : torch.Tensor
        Shape (B, C, H, W), normalised.
    data_range : float
        Dynamic range (default 10.0 for z-score [-5, 5]).

    Returns
    -------
    dict
        Keys: "psnr", "ssim", "mae", "rmse"
    """
    return {
        "psnr": compute_psnr(pred, target, data_range),
        "ssim": compute_ssim_simple(pred, target, data_range),
        "mae":  compute_mae(pred, target),
        "rmse": compute_rmse(pred, target),
    }


class MetricTracker:
    """Accumulate metrics across batches for epoch-level averaging."""

    def __init__(self) -> None:
        self._totals: dict[str, float] = {}
        self._counts: dict[str, int] = {}

    def update(self, metrics: dict[str, float], n: int = 1) -> None:
        """Add metrics from one batch."""
        for k, v in metrics.items():
            self._totals[k] = self._totals.get(k, 0.0) + v * n
            self._counts[k] = self._counts.get(k, 0) + n

    def compute(self) -> dict[str, float]:
        """Return weighted average over all accumulated batches."""
        return {
            k: self._totals[k] / self._counts[k]
            for k in self._totals
            if self._counts[k] > 0
        }

    def reset(self) -> None:
        """Reset all accumulators."""
        self._totals.clear()
        self._counts.clear()
