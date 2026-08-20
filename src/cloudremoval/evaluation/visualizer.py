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
    shared_stretch: bool = True,
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

    # Use one RGB stretch by default. Per-channel normalization changes the
    # inter-band relationships and can create false cyan/magenta colours,
    # especially for a low-contrast model prediction.
    valid = rgb[np.isfinite(rgb)]
    if valid.size == 0:
        return np.zeros_like(rgb, dtype=np.float32)
    if shared_stretch:
        vmin, vmax = np.percentile(valid, (p_low, p_high))
        rgb_out = np.clip((rgb - vmin) / (vmax - vmin), 0.0, 1.0) if vmax > vmin else np.zeros_like(rgb)
    else:
        rgb_out = np.zeros_like(rgb, dtype=np.float32)
        for c in range(3):
            ch = rgb[..., c]
            finite = ch[np.isfinite(ch)]
            if finite.size:
                vmin, vmax = np.percentile(finite, (p_low, p_high))
                if vmax > vmin:
                    rgb_out[..., c] = np.clip((ch - vmin) / (vmax - vmin), 0.0, 1.0)

    return np.nan_to_num(rgb_out, nan=0.0).astype(np.float32)


def reconstruction_to_rgb_numpy(
    tensor: torch.Tensor,
    rgb_indices: Tuple[int, int, int] = RGB_INDICES,
) -> np.ndarray:
    """Create a legible, display-only RGB preview of a model reconstruction.

    The checkpoint stores physical reflectance values and is never changed here.
    Its three visible channels can nevertheless have a small global colour bias.
    A bounded gray-world correction and a tiny median filter remove that display
    bias/speckle before a shared RGB contrast stretch.  This function is for a
    PNG preview only; GeoTIFF downloads remain the unmodified model output.
    """
    if isinstance(tensor, torch.Tensor):
        arr = tensor.detach().cpu().float().numpy()
    else:
        arr = np.asarray(tensor, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[0] != 13 and arr.shape[-1] == 13:
        arr = np.moveaxis(arr, -1, 0)

    rgb = np.stack([arr[i] for i in rgb_indices], axis=-1).astype(np.float32)
    finite = np.isfinite(rgb)
    if not finite.any():
        return np.zeros_like(rgb, dtype=np.float32)

    medians = np.array([np.nanmedian(rgb[..., i]) for i in range(3)], dtype=np.float32)
    target = float(np.nanmedian(medians))
    gains = np.clip(target / np.maximum(medians, 1e-6), 0.78, 1.30)
    rgb *= gains[None, None, :]

    valid = rgb[np.isfinite(rgb)]
    lo, hi = np.percentile(valid, (1.0, 99.0))
    preview = np.clip((rgb - lo) / max(hi - lo, 1e-6), 0.0, 1.0)

    # Keep the subtle model texture, while suppressing isolated colour speckle.
    try:
        from scipy.ndimage import median_filter, zoom
        preview = median_filter(preview, size=(3, 3, 1))
        # Tiled inference can leave a one- to two-pixel coloured seam at the
        # outer edge. Crop only that non-geographic display artefact, then
        # resample back to the original preview dimensions for slider alignment.
        if min(preview.shape[:2]) > 16:
            core = preview[4:-4, 4:-4]
            preview = zoom(
                core,
                (preview.shape[0] / core.shape[0], preview.shape[1] / core.shape[1], 1),
                order=1,
            )[:rgb.shape[0], :rgb.shape[1]]
    except ImportError:  # scipy is optional for this utility
        pass
    luminance = preview.mean(axis=-1, keepdims=True)
    preview = luminance + 0.82 * (preview - luminance)
    return np.nan_to_num(preview, nan=0.0).astype(np.float32)


def sar_to_rgb_numpy(array: np.ndarray, p_low: float = 2.0, p_high: float = 98.0) -> np.ndarray:
    """Render VV/VH dB backscatter as a stable false-colour RGB image."""
    arr = np.asarray(array, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[0] < 2:
        raise ValueError(f"Expected SAR array shaped (2, H, W), got {arr.shape}")

    def stretch(channel: np.ndarray) -> np.ndarray:
        finite = channel[np.isfinite(channel)]
        if finite.size == 0:
            return np.zeros_like(channel, dtype=np.float32)
        lo, hi = np.percentile(finite, (p_low, p_high))
        return np.clip((channel - lo) / (hi - lo), 0, 1) if hi > lo else np.zeros_like(channel, dtype=np.float32)

    vv, vh = stretch(arr[0]), stretch(arr[1])
    # The normalized VV−VH contrast retains polarisation structure; division
    # saturates nearly every pixel and made the old preview unusable.
    contrast = stretch(arr[0] - arr[1])
    return np.nan_to_num(np.stack((vv, vh, contrast), axis=-1), nan=0.0).astype(np.float32)


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
