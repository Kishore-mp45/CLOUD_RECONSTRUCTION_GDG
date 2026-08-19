"""
src/cloudremoval/models/fusion.py
===================================
SAR + Optical fusion module for DSen2-CR.

ORIGINAL DSen2-CR: accepts only S2 optical channels.
THIS MODIFICATION: early channel concatenation of S2 + S1.

Fusion strategy: "early_concat"
  - S2 input : (B, 13, H, W)
  - S1 input : (B,  2, H, W)
  - Fused    : (B, 15, H, W)  <- concatenated along channel dim

This is the MINIMUM modification to the original DSen2-CR architecture.
Everything downstream (residual blocks, head) is unchanged.

Validation:
  - Checks S2 channel count matches config
  - Checks S1 channel count matches config
  - Checks spatial dimensions match between S2 and S1
  - Raises descriptive errors if anything is wrong (silent drop prevention)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from cloudremoval.models.model_config import DSen2CRConfig, S2_CHANNELS, S1_CHANNELS


class SARFusion(nn.Module):
    """Early channel concatenation of Sentinel-2 and Sentinel-1 inputs.

    [MODIFIED from original DSen2-CR]
    The original DSen2-CR did not support SAR input.
    This module concatenates S2 + S1 channels before the first conv layer.

    Parameters
    ----------
    config : DSen2CRConfig
        Model configuration.
    """

    def __init__(self, config: DSen2CRConfig) -> None:
        super().__init__()
        self.s2_channels = config.s2_channels
        self.s1_channels = config.s1_channels
        self.fused_channels = config.fused_channels  # 15

    def forward(self, s2: torch.Tensor, s1: torch.Tensor) -> torch.Tensor:
        """Concatenate S2 and S1 tensors along the channel dimension.

        Parameters
        ----------
        s2 : torch.Tensor
            Shape (B, 13, H, W) — normalised cloudy Sentinel-2
        s1 : torch.Tensor
            Shape (B, 2, H, W) — normalised SAR VV + VH

        Returns
        -------
        torch.Tensor
            Shape (B, 15, H, W) — fused input for the reconstruction network
        """
        self._validate(s2, s1)
        return torch.cat([s2, s1], dim=1)

    def _validate(self, s2: torch.Tensor, s1: torch.Tensor) -> None:
        """Validate tensor shapes before fusion.  Raises ValueError on mismatch."""
        if s2.ndim != 4:
            raise ValueError(f"s2 must be 4D (B,C,H,W), got shape {s2.shape}")
        if s1.ndim != 4:
            raise ValueError(f"s1 must be 4D (B,C,H,W), got shape {s1.shape}")

        if s2.shape[1] != self.s2_channels:
            raise ValueError(
                f"Expected {self.s2_channels} S2 channels, got {s2.shape[1]}. "
                f"Full s2 shape: {tuple(s2.shape)}"
            )
        if s1.shape[1] != self.s1_channels:
            raise ValueError(
                f"Expected {self.s1_channels} S1 channels, got {s1.shape[1]}. "
                f"Full s1 shape: {tuple(s1.shape)}"
            )

        # Spatial dimension check — S1 and S2 must have identical H and W
        if s2.shape[0] != s1.shape[0]:
            raise ValueError(
                f"Batch size mismatch: s2 batch={s2.shape[0]}, s1 batch={s1.shape[0]}"
            )
        if s2.shape[2:] != s1.shape[2:]:
            raise ValueError(
                f"Spatial dimension mismatch: "
                f"s2 HxW={s2.shape[2:]}, s1 HxW={s1.shape[2:]}. "
                f"Ensure preprocessing aligns S1 and S2 to the same patch grid."
            )

    def extra_repr(self) -> str:
        return (
            f"s2_channels={self.s2_channels}, "
            f"s1_channels={self.s1_channels}, "
            f"fused_channels={self.fused_channels}"
        )
