"""
src/cloudremoval/evaluation/visualizer.py
=========================================
Visualization utilities for Phase 5 cloud removal evaluation.

Generates 4-panel visual comparison figures:
  1. Input Cloudy Sentinel-2 (True-color RGB: B4, B3, B2)
  2. DSen2-CR Reconstructed Sentinel-2 (True-color RGB)
  3. Ground-Truth Target Sentinel-2 (True-color RGB)
  4. Absolute Error / Difference Heatmap

Features:
  - Strict band mapping from Phase 0/2:
      B4 (Red)   -> index 3
      B3 (Green) -> index 2
      B2 (Blue)  -> index 1
  - Percentile-based contrast stretch [2%, 98%] for realistic satellite imagery rendering
  - Denormalization using dataset normalization statistics
  - Saves standalone multi-panel comparison figures with metric annotations
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")  # Non-interactive headless backend
import matplotlib.pyplot as plt


# Sentinel-2 true-color band indices (from Phase 0/2 audit: B1..B12+B8A)
# B1=0, B2=1 (Blue), B3=2 (Green), B4=3 (Red), B5=4, ...
RGB_INDICES = (3, 2, 1)  # Red (B4), Green (B3), Blue (B2)


def to_rgb_numpy(
    tensor: torch.Tensor,
    rgb_indices: Tuple[int, int, int] = RGB_INDICES,
    p_low: float = 2.0,
    p_high: float = 98.0,
) -> np.ndarray:
    """Convert a 13-channel Sentinel-2 tensor to a normalized (H, W, 3) RGB numpy array.

    Parameters
    ----------
    tensor : torch.Tensor
        Shape (13, H, W) or (H, W, 13).
    rgb_indices : Tuple[int, int, int]
        Channel indices for (Red, Green, Blue). Default is (3, 2, 1).
    p_low, p_high : float
        Percentile stretch cutoffs.

    Returns
    -------
    np.ndarray
        Shape (H, W, 3), dtype float32, in range [0.0, 1.0].
    """
    if isinstance(tensor, torch.Tensor):
        arr = tensor.detach().cpu().float().numpy()
    else:
        arr = np.array(tensor, dtype=np.float32)

    # Ensure shape (C, H, W)
    if arr.ndim == 3 and arr.shape[0] != 13 and arr.shape[2] == 13:
        arr = np.transpose(arr, (2, 0, 1))

    # Extract RGB channels
    r = arr[rgb_indices[0]]
    g = arr[rgb_indices[1]]
    b = arr[rgb_indices[2]]

    rgb = np.stack([r, g, b], axis=-1)  # (H, W, 3)

    # Robust percentile stretching per channel for clear visualization
    rgb_out = np.zeros_like(rgb, dtype=np.float32)
    for c in range(3):
        ch = rgb[..., c]
        valid = ch[np.isfinite(ch)]
        if len(valid) == 0:
            continue
        vmin = np.percentile(valid, p_low)
        vmax = np.percentile(valid, p_high)
        if vmax > vmin:
            rgb_out[..., c] = np.clip((ch - vmin) / (vmax - vmin), 0.0, 1.0)
        else:
            rgb_out[..., c] = np.clip(ch, 0.0, 1.0)

    return np.nan_to_num(rgb_out, nan=0.0)


def create_4panel_comparison(
    cloudy_s2: torch.Tensor,
    reconstructed_s2: torch.Tensor,
    target_s2: torch.Tensor,
    sample_id: str,
    roi_id: str,
    metrics: Dict[str, float],
    output_path: Path,
    title_suffix: str = "",
) -> Path:
    """Create and save a 4-panel visual comparison figure.

    Panels:
      1. Input Cloudy S2
      2. Reconstructed S2
      3. Ground-Truth Target S2
      4. Absolute Error Map

    Parameters
    ----------
    cloudy_s2 : torch.Tensor
        Input cloudy optical image (13, H, W).
    reconstructed_s2 : torch.Tensor
        DSen2-CR model output (13, H, W).
    target_s2 : torch.Tensor
        Clear-sky target image (13, H, W).
    sample_id : str
        Patch or scene identifier.
    roi_id : str
        Region of interest identifier.
    metrics : Dict[str, float]
        Metrics dict containing 'psnr', 'ssim', 'mae', 'rmse'.
    output_path : Path
        Destination PNG path.
    title_suffix : str
        Optional tag (e.g. 'BEST', 'MEDIAN', 'WORST').

    Returns
    -------
    Path
        Saved PNG path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rgb_cloudy = to_rgb_numpy(cloudy_s2)
    rgb_recon  = to_rgb_numpy(reconstructed_s2)
    rgb_target = to_rgb_numpy(target_s2)

    # Compute RGB absolute error map
    rec_np = reconstructed_s2.detach().cpu().float().numpy()
    tgt_np = target_s2.detach().cpu().float().numpy()
    error_map = np.mean(np.abs(rec_np - tgt_np), axis=0)  # (H, W)

    fig, axes = plt.subplots(1, 4, figsize=(18, 5), dpi=150)
    fig.patch.set_facecolor("#1e1e1e")

    # Panel 1: Cloudy Input
    axes[0].imshow(rgb_cloudy)
    axes[0].set_title("1. Input Cloudy S2 (RGB)", color="white", fontsize=11, fontweight="bold", pad=8)
    axes[0].axis("off")

    # Panel 2: Reconstructed
    axes[1].imshow(rgb_recon)
    axes[1].set_title("2. DSen2-CR Output (RGB)", color="#00ffcc", fontsize=11, fontweight="bold", pad=8)
    axes[1].axis("off")

    # Panel 3: Ground Truth
    axes[2].imshow(rgb_target)
    axes[2].set_title("3. Target Ground Truth (RGB)", color="#66ff66", fontsize=11, fontweight="bold", pad=8)
    axes[2].axis("off")

    # Panel 4: Error Heatmap
    im4 = axes[3].imshow(error_map, cmap="inferno")
    axes[3].set_title("4. Mean Absolute Error", color="#ff9999", fontsize=11, fontweight="bold", pad=8)
    axes[3].axis("off")
    cbar = fig.colorbar(im4, ax=axes[3], fraction=0.046, pad=0.04)
    cbar.ax.yaxis.set_tick_params(color="white", labelcolor="white")

    # Main Super Title with Metrics
    tag_str = f" [{title_suffix}]" if title_suffix else ""
    psnr = metrics.get("psnr", 0.0)
    ssim = metrics.get("ssim", 0.0)
    mae  = metrics.get("mae", 0.0)
    rmse = metrics.get("rmse", 0.0)

    super_title = (
        f"DSen2-CR Evaluation{tag_str} | Sample: {sample_id} ({roi_id})\n"
        f"PSNR: {psnr:.2f} dB  |  SSIM: {ssim:.4f}  |  MAE: {mae:.4f}  |  RMSE: {rmse:.4f}"
    )
    plt.suptitle(super_title, color="white", fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout()

    plt.savefig(output_path, bbox_inches="tight", facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)

    return output_path
