"""
src/cloudremoval/cloud/density.py
=================================
Core Cloud-Density Calculation and Probability Validation for Phase 7.

Definitions:
  - Pixel Cloud Probability: Per-pixel confidence score [0, 100]% provided by satellite
    metadata (e.g., s2cloudless, QA60, or Google Earth Engine S2_CLOUD_PROBABILITY).
  - Pixel Cloud Threshold: Configurable threshold (default: 60%). A pixel is cloudy if prob >= threshold.
  - Scene Cloud Density (%): Proportion of valid pixels within an ROI classified as cloudy:
        (valid cloudy pixels / total valid pixels in ROI) * 100
"""

from __future__ import annotations

import logging
from typing import Optional, Dict, Any, Tuple
import numpy as np

log = logging.getLogger(__name__)


class CloudDataValidationError(ValueError):
    """Raised when cloud probability data fails validation."""


class InvalidROIDataError(ValueError):
    """Raised when ROI has no valid or non-nodata pixels."""


def validate_cloud_probability(
    cloud_prob: np.ndarray,
) -> np.ndarray:
    """Validate and format a 2D cloud-probability array.

    Parameters
    ----------
    cloud_prob : np.ndarray
        Array containing cloud-probability values.

    Returns
    -------
    np.ndarray
        2D float32 array scaled to the [0.0, 100.0] percentage range.
    """
    if cloud_prob is None:
        raise CloudDataValidationError("Cloud probability array is None.")

    if not isinstance(cloud_prob, np.ndarray):
        cloud_prob = np.asarray(cloud_prob, dtype=np.float32)

    # Squeeze channel dimension if shape is (1, H, W)
    if cloud_prob.ndim == 3 and cloud_prob.shape[0] == 1:
        cloud_prob = cloud_prob.squeeze(0)

    if cloud_prob.ndim != 2:
        raise CloudDataValidationError(
            f"Expected 2D cloud probability array (H, W), got shape: {cloud_prob.shape}."
        )

    if cloud_prob.size == 0:
        raise CloudDataValidationError("Cloud probability array is empty.")

    # Check for all NaNs/Infs
    finite_count = np.sum(np.isfinite(cloud_prob))
    if finite_count == 0:
        raise CloudDataValidationError("Cloud probability array contains zero finite numbers.")

    cloud_prob = cloud_prob.astype(np.float32)

    # If data is in [0.0, 1.0] fractional range, rescale to [0.0, 100.0]
    finite_vals = cloud_prob[np.isfinite(cloud_prob)]
    max_val = np.max(finite_vals)
    min_val = np.min(finite_vals)

    if min_val < -1e-4:
        raise CloudDataValidationError(
            f"Invalid negative values in cloud probability array (min: {min_val:.2f})."
        )

    if max_val <= 1.0 and max_val > 0.0:
        log.debug("Auto-rescaling fractional cloud probability [0.0, 1.0] to [0.0, 100.0]%%.")
        cloud_prob = cloud_prob * 100.0
    elif max_val > 100.0 + 1e-4:
        raise CloudDataValidationError(
            f"Cloud probability values exceed 100%% (max: {max_val:.2f})."
        )

    # Clip to exact [0.0, 100.0] bounds
    cloud_prob = np.clip(cloud_prob, 0.0, 100.0)

    return cloud_prob


def calculate_cloud_density(
    cloud_prob: np.ndarray,
    roi_mask: Optional[np.ndarray] = None,
    valid_mask: Optional[np.ndarray] = None,
    pixel_threshold: float = 60.0,
    min_valid_pixel_ratio: float = 0.05,
) -> Dict[str, Any]:
    """Calculate the precise cloud-density percentage over an ROI.

    Parameters
    ----------
    cloud_prob : np.ndarray
        2D array of cloud-probability values [0, 100].
    roi_mask : Optional[np.ndarray]
        2D boolean mask of the ROI (True = inside ROI). If None, entire scene is used.
    valid_mask : Optional[np.ndarray]
        2D boolean mask of valid pixels (True = valid, False = nodata/masked).
    pixel_threshold : float
        Pixel-level probability cutoff for cloud classification (default: 60.0).
    min_valid_pixel_ratio : float
        Minimum fraction of valid pixels required within the ROI.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
          - total_pixels: int
          - valid_pixels: int
          - cloudy_pixels: int
          - cloud_density_percent: float
          - mean_cloud_probability: float
          - pixel_threshold: float
    """
    prob_clean = validate_cloud_probability(cloud_prob)
    h, w = prob_clean.shape
    total_pixels = h * w

    # Construct ROI mask
    if roi_mask is None:
        roi_mask = np.ones((h, w), dtype=bool)
    else:
        if roi_mask.shape != (h, w):
            raise CloudDataValidationError(
                f"roi_mask shape {roi_mask.shape} does not match cloud_prob shape {(h, w)}."
            )
        roi_mask = roi_mask.astype(bool)

    roi_pixels = int(np.sum(roi_mask))
    if roi_pixels == 0:
        raise InvalidROIDataError("ROI mask contains zero pixels.")

    # Construct valid data mask
    if valid_mask is None:
        valid_mask = np.isfinite(prob_clean)
    else:
        if valid_mask.shape != (h, w):
            raise CloudDataValidationError(
                f"valid_mask shape {valid_mask.shape} does not match cloud_prob shape {(h, w)}."
            )
        valid_mask = valid_mask.astype(bool) & np.isfinite(prob_clean)

    # Combined active valid pixels within ROI
    active_mask = roi_mask & valid_mask
    valid_pixels = int(np.sum(active_mask))

    if valid_pixels == 0:
        raise InvalidROIDataError("ROI contains zero valid (non-nodata) pixels.")

    valid_ratio = valid_pixels / max(1, roi_pixels)
    if valid_ratio < min_valid_pixel_ratio:
        raise InvalidROIDataError(
            f"Valid pixel ratio ({valid_ratio*100:.1f}%%) is below minimum threshold ({min_valid_pixel_ratio*100:.1f}%%)."
        )

    # Pixel-level classification
    cloud_mask = active_mask & (prob_clean >= pixel_threshold)
    cloudy_pixels = int(np.sum(cloud_mask))

    # Cloud density percentage formula: (cloudy valid pixels / all valid pixels in ROI) * 100
    cloud_density_percent = (cloudy_pixels / valid_pixels) * 100.0
    mean_cloud_prob = float(np.mean(prob_clean[active_mask]))

    return {
        "total_pixels": total_pixels,
        "roi_pixels": roi_pixels,
        "valid_pixels": valid_pixels,
        "cloudy_pixels": cloudy_pixels,
        "cloud_density_percent": round(cloud_density_percent, 2),
        "mean_cloud_probability": round(mean_cloud_prob, 2),
        "pixel_threshold": pixel_threshold,
    }
