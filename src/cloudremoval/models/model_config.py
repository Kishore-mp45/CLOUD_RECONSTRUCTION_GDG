"""
src/cloudremoval/models/model_config.py
========================================
DSen2-CR model configuration.

All model constants live here.  Nothing is scattered across files.

Values are derived from the Phase 2 dataset audit:
  - S2 bands: 13 (B1-B12 incl. B8A)
  - S1 bands:  2 (VV, VH)
  - Patch size: 256x256
  - Target: 13 S2 bands (cloud-free reconstruction)
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional
import json


# ---------------------------------------------------------------------------
# Confirmed channel counts from Phase 2 / dataset audit
# ---------------------------------------------------------------------------
S2_CHANNELS: int = 13        # cloudy S2 input channels
S1_CHANNELS: int = 2         # SAR VV + VH input channels
TARGET_CHANNELS: int = 13    # cloud-free S2 output channels
FUSED_CHANNELS: int = S2_CHANNELS + S1_CHANNELS  # 15 -- early fusion input
PATCH_SIZE: int = 256


@dataclass
class DSen2CRConfig:
    """Full configuration for the DSen2-CR model.

    All values have documented defaults that can be overridden at init time.
    """

    # ------------------------------------------------------------------
    # Input / Output channels (authoritative from Phase 2)
    # ------------------------------------------------------------------
    s2_channels: int = S2_CHANNELS          # 13
    s1_channels: int = S1_CHANNELS          # 2
    target_channels: int = TARGET_CHANNELS  # 13
    patch_size: int = PATCH_SIZE            # 256

    # ------------------------------------------------------------------
    # Architecture
    # ------------------------------------------------------------------
    base_features: int = 256    # feature channels in the reconstruction network
    num_res_blocks: int = 16    # number of dense residual blocks
    res_scale: float = 0.1      # residual scaling factor (stabilises training)
    kernel_size: int = 3        # convolution kernel size throughout

    # ------------------------------------------------------------------
    # Fusion configuration
    # ------------------------------------------------------------------
    fusion_type: str = "early_concat"
    # "early_concat": S2 + S1 concatenated at channel dim before first conv
    # This is the only supported mode; it is the minimum DSen2-CR modification.

    # ------------------------------------------------------------------
    # Loss configuration
    # ------------------------------------------------------------------
    loss_type: str = "l1"           # "l1" | "l1_ssim"
    ssim_weight: float = 0.0        # weight on SSIM term (0.0 = pure L1)

    # ------------------------------------------------------------------
    # Training / AMP
    # ------------------------------------------------------------------
    amp_enabled: bool = True        # automatic mixed precision
    device: str = "cuda"            # "cuda" | "cpu"
    batch_size: int = 4             # initial batch size (memory test confirms)
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    max_epochs: int = 30
    grad_accum_steps: int = 1       # gradient accumulation (1 = disabled)

    # ------------------------------------------------------------------
    # Checkpoint settings
    # ------------------------------------------------------------------
    checkpoint_dir: str = "checkpoints"
    save_every_n_epochs: int = 5    # epoch checkpoint cadence
    norm_version: str = "v1"        # must match normalization.json version

    # ------------------------------------------------------------------
    # Validation / DataLoader
    # ------------------------------------------------------------------
    val_freq: int = 1               # validate every N epochs
    num_workers: int = 4

    # ------------------------------------------------------------------
    # Derived property
    # ------------------------------------------------------------------
    @property
    def fused_channels(self) -> int:
        """Total input channels after SAR + optical early fusion."""
        return self.s2_channels + self.s1_channels

    def to_dict(self) -> dict:
        """Serialise to a plain dict (for checkpoint embedding)."""
        d = asdict(self)
        d["fused_channels"] = self.fused_channels  # add derived field
        return d

    def to_json(self) -> str:
        """Serialise to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, d: dict) -> "DSen2CRConfig":
        """Restore from a plain dict (e.g. from a saved checkpoint)."""
        d = {k: v for k, v in d.items() if k != "fused_channels"}
        return cls(**d)

    def validate(self) -> None:
        """Raise AssertionError if any channel counts are inconsistent."""
        assert self.s2_channels == S2_CHANNELS, (
            f"s2_channels must be {S2_CHANNELS}, got {self.s2_channels}"
        )
        assert self.s1_channels == S1_CHANNELS, (
            f"s1_channels must be {S1_CHANNELS}, got {self.s1_channels}"
        )
        assert self.target_channels == TARGET_CHANNELS, (
            f"target_channels must be {TARGET_CHANNELS}, got {self.target_channels}"
        )
        assert self.fusion_type == "early_concat", (
            f"Only 'early_concat' fusion supported, got '{self.fusion_type}'"
        )
        assert self.base_features > 0, "base_features must be > 0"
        assert self.num_res_blocks > 0, "num_res_blocks must be > 0"
        assert 0.0 <= self.ssim_weight <= 1.0, "ssim_weight must be in [0, 1]"


# Module-level default config instance
DEFAULT_CONFIG = DSen2CRConfig()
