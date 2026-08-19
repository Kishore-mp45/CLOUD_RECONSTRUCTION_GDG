"""
src/cloudremoval/models/checkpoint.py
=======================================
Checkpoint save / load utilities for DSen2-CR training.

Checkpoint structure (saved as .pth via torch.save):
  {
      "epoch":                int,           # completed epoch number
      "model_state_dict":     OrderedDict,   # model weights
      "optimizer_state_dict": dict,          # optimizer state
      "scheduler_state_dict": dict | None,   # LR scheduler state (if used)
      "early_stopping":       dict | None,   # EarlyStopping.state_dict()
      "val_loss":             float,         # val loss at this epoch
      "train_loss":           float,         # train loss at this epoch
      "config":               dict,          # DSen2CRConfig.to_dict()
      "norm_version":         str,           # normalization.json version tag
      "amp_enabled":          bool,
      "timestamp":            str,           # ISO 8601
  }

Checkpoint files (Phase 4):
  checkpoints/
      latest.pth          <- saved every epoch (overwritten for recovery)
      best_model.pth      <- saved when val_loss improves (never overwritten by worse)
      epoch_NNN.pth       <- saved every epoch (epoch_001.pth, epoch_002.pth, …)

Resume: works from any of the three file types.
"""

from __future__ import annotations

import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import torch

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_checkpoint(
    checkpoint_dir: Path,
    model: "torch.nn.Module",
    optimizer: "torch.optim.Optimizer",
    epoch: int,
    val_loss: float,
    train_loss: float,
    config_dict: dict,
    norm_version: str = "v1",
    amp_enabled: bool = True,
    scheduler: Optional["torch.optim.lr_scheduler._LRScheduler"] = None,
    early_stopping_state: Optional[dict] = None,
    is_best: bool = False,
) -> dict[str, Path]:
    """Save model checkpoint files.

    Parameters
    ----------
    checkpoint_dir : Path
        Directory to write checkpoint files into.
    model : nn.Module
        The DSen2CR model.
    optimizer : Optimizer
        Current optimizer.
    epoch : int
        Completed epoch index (1-based).
    val_loss : float
        Validation loss at this epoch.
    train_loss : float
        Training loss at this epoch.
    config_dict : dict
        DSen2CRConfig.to_dict() — embedded in the checkpoint for reproducibility.
    norm_version : str
        Normalization version tag (must match normalization.json).
    amp_enabled : bool
        Whether AMP was active during training.
    scheduler : optional
        Learning rate scheduler.
    early_stopping_state : dict | None
        EarlyStopping.state_dict() to embed for resume support.
    is_best : bool
        If True, also copy this checkpoint to best_model.pth.

    Returns
    -------
    dict
        Mapping of checkpoint type -> saved path.
    """
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "epoch":                epoch,
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "early_stopping":       early_stopping_state,
        "val_loss":             val_loss,
        "train_loss":           train_loss,
        "config":               config_dict,
        "norm_version":         norm_version,
        "amp_enabled":          amp_enabled,
        "timestamp":            datetime.now(tz=timezone.utc).isoformat(),
    }

    saved: dict[str, Path] = {}

    # --- latest.pth (always overwritten — recovery checkpoint) ---
    latest_path = checkpoint_dir / "latest.pth"
    torch.save(payload, latest_path)
    saved["latest"] = latest_path
    log.info("Checkpoint saved: %s (epoch=%d val_loss=%.6f)", latest_path, epoch, val_loss)

    # --- best_model.pth (only when is_best=True) ---
    if is_best:
        best_path = checkpoint_dir / "best_model.pth"
        shutil.copy2(latest_path, best_path)
        saved["best"] = best_path
        log.info("Best checkpoint updated: %s", best_path)

    # --- epoch_NNN.pth (saved every epoch for full history) ---
    epoch_path = checkpoint_dir / f"epoch_{epoch:03d}.pth"
    shutil.copy2(latest_path, epoch_path)
    saved["epoch"] = epoch_path
    log.debug("Epoch checkpoint saved: %s", epoch_path)

    return saved


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_checkpoint(
    checkpoint_path: Path,
    model: "torch.nn.Module",
    optimizer: Optional["torch.optim.Optimizer"] = None,
    scheduler: Optional["torch.optim.lr_scheduler._LRScheduler"] = None,
    device: str = "cuda",
) -> dict:
    """Load a checkpoint and restore model (and optionally optimizer) state.

    Parameters
    ----------
    checkpoint_path : Path
        Path to a .pth checkpoint file (latest, best, or epoch_NNN).
    model : nn.Module
        Model to restore weights into.
    optimizer : optional
        If provided, optimizer state is restored.
    scheduler : optional
        If provided, LR scheduler state is restored.
    device : str
        Device to map tensors to.

    Returns
    -------
    dict
        The raw checkpoint dict.  Contains "epoch", "val_loss", etc.
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    log.info("Loading checkpoint: %s", checkpoint_path)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model.load_state_dict(ckpt["model_state_dict"])
    log.info("Model state restored from epoch %d", ckpt["epoch"])

    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        log.info("Optimizer state restored")

    if (
        scheduler is not None
        and ckpt.get("scheduler_state_dict") is not None
    ):
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        log.info("Scheduler state restored")

    return ckpt


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def list_checkpoints(checkpoint_dir: Path) -> dict[str, Path]:
    """Return a dict of available checkpoint files in the given directory."""
    if not checkpoint_dir.exists():
        return {}
    result: dict[str, Path] = {}
    for name in ("latest.pth", "best_model.pth"):
        p = checkpoint_dir / name
        if p.exists():
            result[name.replace(".pth", "")] = p
    # epoch checkpoints
    for p in sorted(checkpoint_dir.glob("epoch_*.pth")):
        result[p.stem] = p
    return result


def get_best_checkpoint(checkpoint_dir: Path) -> Optional[Path]:
    """Return best_model.pth path if it exists, else latest.pth, else None."""
    best = checkpoint_dir / "best_model.pth"
    if best.exists():
        return best
    latest = checkpoint_dir / "latest.pth"
    if latest.exists():
        return latest
    return None
