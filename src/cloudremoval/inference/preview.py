"""
src/cloudremoval/inference/preview.py
======================================
True-color PNG preview generation for local geospatial inference (Phase 6).

Produces a side-by-side comparison figure:
  - Left: Input Cloudy Sentinel-2 (True-color RGB)
  - Right: DSen2-CR Reconstructed Sentinel-2 (True-color RGB)

Strict Band Ordering:
  - B4 (Red)   -> index 3
  - B3 (Green) -> index 2
  - B2 (Blue)  -> index 1
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cloudremoval.evaluation.visualizer import to_rgb_numpy, RGB_INDICES


def create_inference_preview(
    cloudy_s2: np.ndarray,
    reconstructed_s2: np.ndarray,
    output_png_path: Path,
    job_id: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    """Create a side-by-side true-color PNG preview of cloudy input vs reconstructed output.

    Parameters
    ----------
    cloudy_s2 : np.ndarray
        Shape (13, H, W), raw surface reflectance.
    reconstructed_s2 : np.ndarray
        Shape (13, H, W), reconstructed surface reflectance.
    output_png_path : Path
        Destination PNG path.
    job_id : str
        Inference Job Identifier.
    metadata : Optional[Dict[str, Any]]
        Optional metadata to annotate on the figure.

    Returns
    -------
    Path
        Saved PNG path.
    """
    output_png_path = Path(output_png_path)
    output_png_path.parent.mkdir(parents=True, exist_ok=True)

    rgb_cloudy = to_rgb_numpy(cloudy_s2, rgb_indices=RGB_INDICES)
    rgb_recon  = to_rgb_numpy(reconstructed_s2, rgb_indices=RGB_INDICES)

    fig, axes = plt.subplots(1, 2, figsize=(14, 7), dpi=150)
    fig.patch.set_facecolor("#181818")

    # Left: Input Cloudy
    axes[0].imshow(rgb_cloudy)
    axes[0].set_title("Input Cloudy Sentinel-2 (RGB: B4-B3-B2)", color="white", fontsize=12, fontweight="bold", pad=10)
    axes[0].axis("off")

    # Right: Reconstructed Output
    axes[1].imshow(rgb_recon)
    axes[1].set_title("DSen2-CR Reconstructed Sentinel-2 (RGB: B4-B3-B2)", color="#00ffcc", fontsize=12, fontweight="bold", pad=10)
    axes[1].axis("off")

    # Subtitle with metadata
    meta_str = f"Job: {job_id}"
    if metadata:
        res = metadata.get("resolution", "")
        crs = metadata.get("crs", "")
        time_s = metadata.get("inference_time_s", "")
        if res:
            meta_str += f" | Res: {res}"
        if crs:
            meta_str += f" | CRS: {crs}"
        if time_s:
            meta_str += f" | Model Time: {time_s:.2f}s"

    plt.suptitle(f"DSen2-CR Local Geospatial Inference\n{meta_str}", color="white", fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout()

    plt.savefig(output_png_path, bbox_inches="tight", facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)

    return output_png_path
