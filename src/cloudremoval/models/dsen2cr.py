"""
src/cloudremoval/models/dsen2cr.py
=====================================
Modified DSen2-CR architecture for Sentinel-2 cloud removal using SAR.

ORIGINAL DSen2-CR (Meraner et al., 2020):
  - Deep convolutional network for S2 cloud removal
  - Dense Residual Blocks (DRBs) for feature extraction
  - Residual learning: output = input_s2 + network(input_s2)
  - Single modality: S2 optical only

THIS MODIFICATION (minimum change for SAR fusion):
  - [MODIFIED] First conv layer: Conv2d(15, F, 3, 1) instead of Conv2d(13, F, 3, 1)
    to accept the early-fused (S2 + S1) 15-channel input.
  - [ORIGINAL] All DRBs, feature processing, and output head are unchanged.
  - [ORIGINAL] Residual connection adds the 13-channel S2 input to the output.
  - [ORIGINAL] Output: (B, 13, H, W) cloud-free S2 reconstruction.

Architecture diagram:
    S2 (13, H, W) --|
                     |--> SARFusion --> (15, H, W)
    S1 ( 2, H, W) --|         |
                               v
                      head_conv (15 -> F)
                               |
                          [DRB x num_res_blocks]
                               |
                      tail_conv (F -> 13)
                               |
                         + S2 input (residual)
                               |
                      output (13, H, W)

References:
  Meraner et al. (2020). Cloud removal in Sentinel-2 imagery using a deep
  residual neural network and SAR-optical data fusion.
  ISPRS Journal of Photogrammetry and Remote Sensing, 166, 333-346.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn

from cloudremoval.models.model_config import DSen2CRConfig, DEFAULT_CONFIG
from cloudremoval.models.fusion import SARFusion

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class _ResidualBlock(nn.Module):
    """Single Dense Residual Block (DRB).

    [ORIGINAL DSen2-CR component]

    Structure:
        Conv -> ReLU -> Conv -> scale -> add input
    """

    def __init__(self, features: int, kernel_size: int = 3, res_scale: float = 0.1) -> None:
        super().__init__()
        pad = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(features, features, kernel_size, padding=pad, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, features, kernel_size, padding=pad, bias=True),
        )
        self.res_scale = res_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x) * self.res_scale


class _FeatureExtractor(nn.Module):
    """Stack of Dense Residual Blocks.

    [ORIGINAL DSen2-CR component]
    """

    def __init__(
        self,
        features: int,
        num_blocks: int,
        kernel_size: int = 3,
        res_scale: float = 0.1,
    ) -> None:
        super().__init__()
        self.blocks = nn.Sequential(
            *[_ResidualBlock(features, kernel_size, res_scale) for _ in range(num_blocks)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class DSen2CR(nn.Module):
    """Modified DSen2-CR for SAR-supervised Sentinel-2 cloud removal.

    Parameters
    ----------
    config : DSen2CRConfig
        Full model configuration.

    Usage
    -----
    >>> model = DSen2CR()
    >>> s2 = torch.randn(1, 13, 256, 256)
    >>> s1 = torch.randn(1,  2, 256, 256)
    >>> output = model(s2, s1)   # (1, 13, 256, 256)
    """

    def __init__(self, config: Optional[DSen2CRConfig] = None) -> None:
        super().__init__()
        if config is None:
            config = DEFAULT_CONFIG
        config.validate()
        self.config = config

        F = config.base_features
        K = config.kernel_size
        pad = K // 2

        # [MODIFIED] Head accepts 15-channel fused input (13 S2 + 2 S1)
        # Original DSen2-CR: Conv2d(s2_channels, F, K, padding=pad)
        self.fusion = SARFusion(config)
        self.head = nn.Conv2d(config.fused_channels, F, K, padding=pad, bias=True)

        # [ORIGINAL] Deep feature extractor — Dense Residual Blocks
        self.body = _FeatureExtractor(
            features=F,
            num_blocks=config.num_res_blocks,
            kernel_size=K,
            res_scale=config.res_scale,
        )

        # [ORIGINAL] Tail projects features back to S2 channel count
        self.tail = nn.Conv2d(F, config.target_channels, K, padding=pad, bias=True)

        self._init_weights()

        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        log.info(
            "DSen2CR initialised | params=%.2fM | s2=%d s1=%d fused=%d target=%d "
            "base_features=%d num_res_blocks=%d",
            n_params / 1e6,
            config.s2_channels, config.s1_channels, config.fused_channels,
            config.target_channels, config.base_features, config.num_res_blocks,
        )

    def _init_weights(self) -> None:
        """Kaiming normal initialisation for all conv layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, s2: torch.Tensor, s1: torch.Tensor) -> torch.Tensor:
        """Forward pass: reconstruct cloud-free S2 from cloudy S2 + SAR S1.

        Parameters
        ----------
        s2 : torch.Tensor
            Shape (B, 13, H, W) — normalised cloudy Sentinel-2 input
        s1 : torch.Tensor
            Shape (B, 2, H, W) — normalised Sentinel-1 SAR (VV, VH)

        Returns
        -------
        torch.Tensor
            Shape (B, 13, H, W) — reconstructed cloud-free Sentinel-2
        """
        # --- SAR + Optical fusion ---
        # [MODIFIED] S2 and S1 are concatenated into a 15-channel tensor
        fused = self.fusion(s2, s1)       # (B, 15, H, W)

        # --- Feature extraction ---
        # [ORIGINAL] Head projects to feature space
        feat = self.head(fused)           # (B, F, H, W)
        # [ORIGINAL] Deep residual blocks
        feat = self.body(feat)            # (B, F, H, W)
        # [ORIGINAL] Tail projects to S2 channel count
        residual = self.tail(feat)        # (B, 13, H, W)

        # --- Residual learning ---
        # [ORIGINAL] Add cloudy S2 input as global residual
        # The network learns to predict the correction, not the full image
        output = s2 + residual            # (B, 13, H, W)

        return output

    def parameter_count(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def parameter_count_str(self) -> str:
        """Return formatted parameter count string."""
        n = self.parameter_count()
        if n >= 1e6:
            return f"{n / 1e6:.2f}M"
        elif n >= 1e3:
            return f"{n / 1e3:.1f}K"
        return str(n)

    def to_device(self, device: str) -> "DSen2CR":
        """Move model to device and return self for chaining."""
        return self.to(device)

    def summary(self) -> str:
        """Return a human-readable model summary string."""
        cfg = self.config
        lines = [
            "=" * 60,
            "DSen2-CR Model Summary",
            "=" * 60,
            f"  S2 input channels    : {cfg.s2_channels}",
            f"  S1 input channels    : {cfg.s1_channels}",
            f"  Fused input channels : {cfg.fused_channels}",
            f"  Target channels      : {cfg.target_channels}",
            f"  Patch size           : {cfg.patch_size}x{cfg.patch_size}",
            f"  Base features        : {cfg.base_features}",
            f"  Residual blocks      : {cfg.num_res_blocks}",
            f"  Residual scale       : {cfg.res_scale}",
            f"  Kernel size          : {cfg.kernel_size}",
            f"  Fusion type          : {cfg.fusion_type}",
            f"  Parameters           : {self.parameter_count_str()}",
            "=" * 60,
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

def build_model(config: Optional[DSen2CRConfig] = None) -> DSen2CR:
    """Build and return a DSen2CR model.

    Parameters
    ----------
    config : DSen2CRConfig, optional
        If None, uses DEFAULT_CONFIG (all defaults from Phase 2 data shapes).

    Returns
    -------
    DSen2CR
        Initialised model (on CPU by default).
    """
    return DSen2CR(config=config)
