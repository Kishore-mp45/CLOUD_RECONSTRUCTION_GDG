"""
src/cloudremoval/evaluation/metrics.py
=======================================
Quantitative evaluation metrics for Phase 5 cloud removal evaluation.

Supports:
  - PSNR (Peak Signal-to-Noise Ratio)
  - SSIM (Multispectral Structural Similarity Index)
  - MAE (Mean Absolute Error)
  - RMSE (Root Mean Squared Error)
  - SAM (Spectral Angle Mapper in degrees)
  - Statistical aggregation (Mean, Median, Std, Min, Max, P25, P75)
  - Nodata / invalid pixel masking

All calculations operate on float32/float64 tensors or numpy arrays and are
safe under torch.no_grad().
"""

from __future__ import annotations

import math
from typing import Optional, Union, Dict, Any, List
import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# SSIM Window Helper
# ---------------------------------------------------------------------------

def _gaussian_window(window_size: int, sigma: float, channels: int) -> torch.Tensor:
    """Create a 2D Gaussian window kernel for SSIM calculation."""
    gauss = torch.tensor(
        [
            math.exp(-((x - window_size // 2) ** 2) / (2 * sigma ** 2))
            for x in range(window_size)
        ],
        dtype=torch.float32,
    )
    gauss = gauss / gauss.sum()
    _1d = gauss.unsqueeze(1)
    _2d = _1d.mm(_1d.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2d.expand(channels, 1, window_size, window_size).contiguous()
    return window


# ---------------------------------------------------------------------------
# Per-Image / Batch Metric Functions
# ---------------------------------------------------------------------------

def compute_psnr(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 10.0,
    mask: Optional[torch.Tensor] = None,
) -> float:
    """Compute PSNR between predicted and target tensors.

    Parameters
    ----------
    pred, target : torch.Tensor
        Tensors of shape (C, H, W) or (B, C, H, W).
    data_range : float
        Dynamic range of the data (default 10.0 for z-score clamped to [-5, 5]).
    mask : Optional[torch.Tensor]
        Optional binary mask (1 for valid pixels, 0 for nodata).

    Returns
    -------
    float
        PSNR value in decibels (dB). Returns float('inf') if perfect match.
    """
    with torch.no_grad():
        p = pred.float()
        t = target.float()

        if mask is not None:
            m = mask.bool()
            if m.sum() == 0:
                return 0.0
            p = p[m]
            t = t[m]

        mse = torch.mean((p - t) ** 2)
        if mse.item() == 0.0:
            return float("inf")
        if not torch.isfinite(mse):
            return 0.0

        psnr = 20.0 * math.log10(data_range) - 10.0 * math.log10(mse.item())
        return float(psnr)


def compute_ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 10.0,
    window_size: int = 11,
    sigma: float = 1.5,
) -> float:
    """Compute Multispectral SSIM across all channels.

    Parameters
    ----------
    pred, target : torch.Tensor
        Tensors of shape (C, H, W) or (B, C, H, W).
    data_range : float
        Dynamic range of the data.
    window_size : int
        Gaussian window kernel size.
    sigma : float
        Gaussian standard deviation.

    Returns
    -------
    float
        Mean SSIM in [0.0, 1.0].
    """
    with torch.no_grad():
        p = pred.float()
        t = target.float()

        if p.ndim == 3:
            p = p.unsqueeze(0)
            t = t.unsqueeze(0)

        b, c, h, w = p.shape
        if h < window_size or w < window_size:
            window_size = min(h, w)
            if window_size % 2 == 0:
                window_size -= 1

        device = p.device
        window = _gaussian_window(window_size, sigma, c).to(device)

        c1 = (0.01 * data_range) ** 2
        c2 = (0.03 * data_range) ** 2

        mu1 = F.conv2d(p, window, padding=window_size // 2, groups=c)
        mu2 = F.conv2d(t, window, padding=window_size // 2, groups=c)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(p * p, window, padding=window_size // 2, groups=c) - mu1_sq
        sigma2_sq = F.conv2d(t * t, window, padding=window_size // 2, groups=c) - mu2_sq
        sigma12 = F.conv2d(p * t, window, padding=window_size // 2, groups=c) - mu1_mu2

        num = (2.0 * mu1_mu2 + c1) * (2.0 * sigma12 + c2)
        denom = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
        ssim_map = num / (denom + 1e-8)

        ssim_val = ssim_map.mean().item()
        return float(np.clip(ssim_val, -1.0, 1.0))


def compute_mae(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> float:
    """Compute Mean Absolute Error (L1 distance)."""
    with torch.no_grad():
        p = pred.float()
        t = target.float()
        if mask is not None:
            m = mask.bool()
            if m.sum() == 0:
                return 0.0
            p = p[m]
            t = t[m]
        return float(torch.mean(torch.abs(p - t)).item())


def compute_rmse(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> float:
    """Compute Root Mean Squared Error (L2 distance)."""
    with torch.no_grad():
        p = pred.float()
        t = target.float()
        if mask is not None:
            m = mask.bool()
            if m.sum() == 0:
                return 0.0
            p = p[m]
            t = t[m]
        mse = torch.mean((p - t) ** 2).item()
        return float(math.sqrt(max(0.0, mse)))


def compute_sam(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-8,
) -> float:
    """Compute Spectral Angle Mapper (SAM) in degrees.

    Measures spectral angle distortion across multispectral bands.
    """
    with torch.no_grad():
        p = pred.double()
        t = target.double()
        if p.ndim == 4:
            p = p.squeeze(0)
            t = t.squeeze(0)

        # Reshape to (C, H*W) -> (H*W, C)
        c, h, w = p.shape
        p_flat = p.view(c, -1).t()
        t_flat = t.view(c, -1).t()

        dot_product = (p_flat * t_flat).sum(dim=1)
        norm_p = torch.norm(p_flat, dim=1)
        norm_t = torch.norm(t_flat, dim=1)

        cos_theta = dot_product / (norm_p * norm_t + eps)
        cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
        angle_rad = torch.acos(cos_theta)
        angle_deg = torch.rad2deg(angle_rad)

        valid_angles = angle_deg[torch.isfinite(angle_deg)]
        if len(valid_angles) == 0:
            return 0.0
        return float(valid_angles.mean().item())


def evaluate_single_sample(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 10.0,
    mask: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """Compute all evaluation metrics for a single prediction vs target.

    Parameters
    ----------
    pred, target : torch.Tensor
        Tensors of shape (C, H, W).
    data_range : float
        Dynamic range for PSNR/SSIM.

    Returns
    -------
    Dict[str, float]
        Dictionary with keys: psnr, ssim, mae, rmse, sam.
    """
    return {
        "psnr": compute_psnr(pred, target, data_range=data_range, mask=mask),
        "ssim": compute_ssim(pred, target, data_range=data_range),
        "mae":  compute_mae(pred, target, mask=mask),
        "rmse": compute_rmse(pred, target, mask=mask),
        "sam":  compute_sam(pred, target),
    }


# ---------------------------------------------------------------------------
# Statistical Aggregator
# ---------------------------------------------------------------------------

def compute_distribution_statistics(values: Union[List[float], np.ndarray]) -> Dict[str, float]:
    """Compute descriptive statistics for a list of metric values."""
    arr = np.array(values, dtype=np.float64)
    # Filter non-finite values for robust statistical aggregation
    finite_arr = arr[np.isfinite(arr)]
    if len(finite_arr) == 0:
        return {
            "mean": 0.0,
            "std": 0.0,
            "median": 0.0,
            "min": 0.0,
            "max": 0.0,
            "p25": 0.0,
            "p75": 0.0,
            "count": 0,
        }

    return {
        "mean":   float(np.mean(finite_arr)),
        "std":    float(np.std(finite_arr)),
        "median": float(np.median(finite_arr)),
        "min":    float(np.min(finite_arr)),
        "max":    float(np.max(finite_arr)),
        "p25":    float(np.percentile(finite_arr, 25)),
        "p75":    float(np.percentile(finite_arr, 75)),
        "count":  int(len(finite_arr)),
    }
