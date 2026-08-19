"""
src/cloudremoval/models/losses.py
====================================
Loss functions for DSen2-CR cloud removal training.

Loss formulation:
  loss = L1(pred, target) + ssim_weight * (1 - SSIM(pred, target))

Default (ssim_weight=0.0): pure L1.
  - L1 is the standard loss used in the original DSen2-CR paper.
  - It is robust to outliers in the satellite imagery domain.

SSIM term is optional and configurable.
  - Adds structural sensitivity at the cost of higher compute.
  - Recommended: start with pure L1 (ssim_weight=0.0), tune later.

The loss module is modular and supports:
  - Training (backward-capable)
  - Validation (no_grad context, same computation)
  - Per-sample loss logging
  - AMP (float16) safety via explicit float32 cast before loss computation
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from cloudremoval.models.model_config import DSen2CRConfig, DEFAULT_CONFIG


class SSIMLoss(nn.Module):
    """Differentiable SSIM loss (1 - SSIM), operating on normalised tensors.

    Uses a 11x11 Gaussian kernel as per the original SSIM paper.
    Computed per-channel and averaged.
    """

    def __init__(self, window_size: int = 11, sigma: float = 1.5) -> None:
        super().__init__()
        self.window_size = window_size
        kernel = self._gaussian_kernel(window_size, sigma)
        # Register as buffer so it moves with .to(device)
        self.register_buffer("kernel", kernel)

    @staticmethod
    def _gaussian_kernel(size: int, sigma: float) -> torch.Tensor:
        """1D Gaussian, outer-producted to 2D, then expanded for conv2d."""
        coords = torch.arange(size, dtype=torch.float32) - size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g /= g.sum()
        kernel_2d = g.unsqueeze(0) * g.unsqueeze(1)  # (size, size)
        # (1, 1, size, size) — will be broadcast over channels via groups
        return kernel_2d.unsqueeze(0).unsqueeze(0)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute 1 - SSIM(pred, target).

        Parameters
        ----------
        pred, target : torch.Tensor
            Shape (B, C, H, W), both normalised.

        Returns
        -------
        torch.Tensor
            Scalar loss value.
        """
        B, C, H, W = pred.shape
        # Expand kernel for all channels using grouped convolution
        kernel = self.kernel.expand(C, 1, self.window_size, self.window_size)
        pad = self.window_size // 2

        mu_x = F.conv2d(pred,   kernel, padding=pad, groups=C)
        mu_y = F.conv2d(target, kernel, padding=pad, groups=C)

        mu_x_sq = mu_x * mu_x
        mu_y_sq = mu_y * mu_y
        mu_xy   = mu_x * mu_y

        sigma_x  = F.conv2d(pred   * pred,   kernel, padding=pad, groups=C) - mu_x_sq
        sigma_y  = F.conv2d(target * target, kernel, padding=pad, groups=C) - mu_y_sq
        sigma_xy = F.conv2d(pred   * target, kernel, padding=pad, groups=C) - mu_xy

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        num   = (2 * mu_xy + C1) * (2 * sigma_xy + C2)
        denom = (mu_x_sq + mu_y_sq + C1) * (sigma_x + sigma_y + C2)

        ssim_map = num / (denom + 1e-8)
        return 1.0 - ssim_map.mean()


class CloudRemovalLoss(nn.Module):
    """Combined loss for DSen2-CR cloud removal.

    Formula: loss = L1(pred, target) + ssim_weight * SSIMLoss(pred, target)

    Default: pure L1 (ssim_weight=0.0) — matches original DSen2-CR paper.

    Parameters
    ----------
    config : DSen2CRConfig
        Uses config.loss_type and config.ssim_weight.
    """

    def __init__(self, config: DSen2CRConfig | None = None) -> None:
        super().__init__()
        if config is None:
            config = DEFAULT_CONFIG
        self.ssim_weight = config.ssim_weight
        self.loss_type = config.loss_type
        self.l1 = nn.L1Loss()
        if self.ssim_weight > 0.0:
            self.ssim = SSIMLoss()
        else:
            self.ssim = None

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """Compute loss between prediction and target.

        Parameters
        ----------
        pred : torch.Tensor
            Shape (B, 13, H, W) — model output (normalised)
        target : torch.Tensor
            Shape (B, 13, H, W) — ground truth cloud-free S2 (normalised)

        Returns
        -------
        total_loss : torch.Tensor
            Scalar loss tensor (backward-compatible).
        components : dict
            Dict with "l1", "ssim" (if used), "total" — for logging.
        """
        # Cast to float32 for numerical stability under AMP
        pred_f   = pred.float()
        target_f = target.float()

        l1_loss = self.l1(pred_f, target_f)
        components: dict[str, float] = {"l1": l1_loss.item()}

        total = l1_loss

        if self.ssim is not None and self.ssim_weight > 0.0:
            ssim_loss = self.ssim(pred_f, target_f)
            total = total + self.ssim_weight * ssim_loss
            components["ssim"] = ssim_loss.item()

        components["total"] = total.item()
        return total, components


def build_loss(config: DSen2CRConfig | None = None) -> CloudRemovalLoss:
    """Factory function for loss module."""
    return CloudRemovalLoss(config=config)
