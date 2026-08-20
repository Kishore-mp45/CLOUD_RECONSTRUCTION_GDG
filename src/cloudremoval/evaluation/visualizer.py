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
  - render_s2_rgb(): authoritative fixed-scale renderer
    Uses a fixed reference S2 physical range so both original (cloudy)
    and reconstructed (clear-sky) images are visually comparable on the
    SAME absolute brightness scale. Without a fixed scale, independent
    per-image stretch makes a dark clear-sky reconstruction look as
    bright as a cloud-covered input - misleading.
  - to_rgb_numpy(): per-image percentile stretch (used internally for
    4-panel evaluation where each panel should fill its own contrast range)
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

# Fixed S2 TOA physical reference range for absolute-scale rendering.
# Derived from Sentinel-2 Level-1C DN characteristics:
#   - Min: 0 (dark pixels, shadows)
#   - Ref_high: ~3000 DN = ~0.30 surface reflectance (typical clear land)
#   - Soft saturation at 5000 DN (bright clouds/snow, clipped gracefully)
# Using a fixed scale means original-cloudy and reconstructed-clear-sky
# images are shown at the SAME absolute brightness, making them comparable.
_S2_FIXED_LOW  = 0.0       # DN - absolute lower reference
_S2_FIXED_HIGH = 3000.0    # DN - soft saturation reference (clips gracefully to 1.0)


def _to_numpy_chw(tensor_or_array) -> np.ndarray:
    """Coerce input to (C, H, W) float32 numpy array."""
    if isinstance(tensor_or_array, torch.Tensor):
        arr = tensor_or_array.detach().cpu().float().numpy()
    else:
        arr = np.asarray(tensor_or_array, dtype=np.float32)
    # Handle (H, W, C) arrays
    if arr.ndim == 3 and arr.shape[0] not in (13, 2) and arr.shape[2] in (13, 2):
        arr = np.moveaxis(arr, -1, 0)
    return arr


def render_s2_rgb(
    data,
    rgb_indices: Tuple[int, int, int] = RGB_INDICES,
    fixed_low: float = _S2_FIXED_LOW,
    fixed_high: float = _S2_FIXED_HIGH,
) -> np.ndarray:
    """**Authoritative** fixed-scale Sentinel-2 true-color renderer.

    This is the SINGLE function that must be used for BOTH the original
    (cloudy) and the reconstructed (cloud-free) image previews.  Both
    images share the same physical [fixed_low, fixed_high] reference
    range so that brightness is visually comparable between them.

    Parameters
    ----------
    data : array-like
        Shape (13, H, W) or (H, W, 13), denormalized S2 reflectance DN.
    rgb_indices : Tuple[int, int, int]
        (R, G, B) channel indices. Default (3, 2, 1) = (B4, B3, B2).
    fixed_low : float
        Lower clip / black point in S2 DN units. Default 0.
    fixed_high : float
        Upper clip / white point in S2 DN units. Default 3000.
        Values above this are simply clipped to white - graceful saturation.

    Returns
    -------
    np.ndarray
        Shape (H, W, 3), float32, range [0.0, 1.0].
    """
    arr = _to_numpy_chw(data)
    r = arr[rgb_indices[0]]
    g = arr[rgb_indices[1]]
    b = arr[rgb_indices[2]]
    rgb = np.stack([r, g, b], axis=-1)  # (H, W, 3)

    # Replace NaN with 0 before scaling
    rgb = np.nan_to_num(rgb, nan=0.0, posinf=fixed_high, neginf=0.0)

    # Fixed-range linear scale: both images use same absolute reference
    span = max(fixed_high - fixed_low, 1.0)
    rgb_out = np.clip((rgb - fixed_low) / span, 0.0, 1.0)

    return rgb_out.astype(np.float32)


def apply_chromaticity_match(
    recon_rgb: np.ndarray,
    orig_arr: np.ndarray,
    cloud_threshold: float = 2500.0
) -> np.ndarray:
    """
    Applies contextual chromaticity matching to fix AI color hallucinations.
    
    The SAR-to-Optical model often hallucinates green (vegetation) in completely
    cloud-obscured areas because of training data bias (SAR volume scattering = trees).
    This post-processing step measures the chromaticity (color balance) of the
    real clear-sky pixels in the image, and forces the hallucinated pixels to
    match that color distribution, while preserving the structural luminance predicted
    by the AI.
    
    Parameters
    ----------
    recon_rgb : np.ndarray
        Shape (H, W, 3), float32, range [0, 1]. The rendered reconstructed RGB.
    orig_arr : np.ndarray
        Shape (13, H, W). The original cloudy input GeoTIFF array (DN).
    cloud_threshold : float
        DN threshold in the blue band (B2, idx 1) to define thick clouds.
        
    Returns
    -------
    np.ndarray
        Shape (H, W, 3), float32, color-corrected RGB.
    """
    import numpy as np
    
    # Ensure orig_arr is (C, H, W)
    orig = _to_numpy_chw(orig_arr)
    if orig.shape[0] < 2:
        return recon_rgb
        
    # Blue band (B2) is at index 1
    cloud_mask = orig[1] > cloud_threshold
    clear_mask = ~cloud_mask
    
    if not np.any(cloud_mask) or not np.any(clear_mask):
        return recon_rgb
        
    # 1. Calculate Brightness (sum of RGB)
    eps = 1e-6
    recon_sum = np.sum(recon_rgb, axis=-1, keepdims=True) + eps
    
    # 2. Extract Chromaticity (Color fractions)
    r_chroma = recon_rgb[:, :, 0] / recon_sum[:, :, 0]
    g_chroma = recon_rgb[:, :, 1] / recon_sum[:, :, 0]
    b_chroma = recon_rgb[:, :, 2] / recon_sum[:, :, 0]
    
    # 3. Match Chromaticity distribution (shift cloudy colors to match clear colors)
    for chroma in [r_chroma, g_chroma, b_chroma]:
        c_mean_clear = np.mean(chroma[clear_mask])
        c_std_clear = np.std(chroma[clear_mask])
        c_mean_cloudy = np.mean(chroma[cloud_mask])
        c_std_cloudy = np.std(chroma[cloud_mask])
        
        # Apply shift only to cloudy pixels
        matched = (chroma[cloud_mask] - c_mean_cloudy) * (c_std_clear / max(c_std_cloudy, 1e-5)) + c_mean_clear
        chroma[cloud_mask] = np.clip(matched, 0.0, 1.0)
        
    # 4. Re-normalize chromaticity so R+G+B = 1
    c_sum = r_chroma + g_chroma + b_chroma + eps
    r_chroma /= c_sum
    g_chroma /= c_sum
    b_chroma /= c_sum
    
    # 5. Multiply back by the original structural brightness predicted by the AI
    matched_rgb = np.zeros_like(recon_rgb)
    matched_rgb[:, :, 0] = r_chroma * recon_sum[:, :, 0]
    matched_rgb[:, :, 1] = g_chroma * recon_sum[:, :, 0]
    matched_rgb[:, :, 2] = b_chroma * recon_sum[:, :, 0]
    
    return np.clip(matched_rgb, 0.0, 1.0)



def to_rgb_numpy(
    tensor,
    rgb_indices: Tuple[int, int, int] = RGB_INDICES,
    p_low: float = 2.0,
    p_high: float = 98.0,
    shared_stretch: bool = True,
) -> np.ndarray:
    """Per-image percentile-stretch RGB renderer.

    NOTE: For side-by-side original vs. reconstructed comparisons,
    use ``render_s2_rgb()`` instead — it applies a fixed physical-scale
    reference so both images are visually comparable.

    This function is retained for internal 4-panel evaluation figures
    where each panel benefits from its own full-range stretch.

    Parameters
    ----------
    tensor : array-like
        Shape (13, H, W) or (H, W, 13), denormalized S2 reflectance DN.
    rgb_indices : Tuple[int, int, int]
        Channel indices for (Red, Green, Blue). Default is (3, 2, 1).
    p_low, p_high : float
        Percentile stretch cutoffs.

    Returns
    -------
    np.ndarray
        Shape (H, W, 3), dtype float32, in range [0.0, 1.0].
    """
    arr = _to_numpy_chw(tensor)

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
    tensor,
    rgb_indices: Tuple[int, int, int] = RGB_INDICES,
    fixed_low: float = _S2_FIXED_LOW,
    fixed_high: float = _S2_FIXED_HIGH,
) -> np.ndarray:
    """Fixed-scale RGB renderer for reconstructed S2 output.

    Uses the SAME fixed physical reference range as render_s2_rgb() so
    reconstructed imagery is visually comparable to the original input.
    This is the function used by api/routes/results.py for browser previews.

    The GeoTIFF output is NEVER modified. This function only produces
    a display PNG.
    """
    return render_s2_rgb(
        data=tensor,
        rgb_indices=rgb_indices,
        fixed_low=fixed_low,
        fixed_high=fixed_high,
    )


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
