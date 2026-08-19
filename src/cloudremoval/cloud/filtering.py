"""
src/cloudremoval/cloud/filtering.py
===================================
Scene-level cloud filtering and batch processing for Phase 7.

Logic:
  1. Pixel-level check: pixel is cloudy if cloud_probability >= pixel_probability_threshold
  2. ROI-level calculation: cloud_density_percent = (valid cloudy pixels / total valid pixels in ROI) * 100
  3. Scene-level gate:
       - If cloud_density_percent >= scene_density_threshold -> ELIGIBLE (selected for cloud removal)
       - If cloud_density_percent < scene_density_threshold  -> FILTERED OUT
"""

from __future__ import annotations

import logging
from typing import List, Dict, Any, Optional
import numpy as np

from cloudremoval.cloud.schemas import (
    CloudFilterConfig,
    CloudDensityResult,
    BatchFilterResult,
    SceneMetadata,
)
from cloudremoval.cloud.density import calculate_cloud_density

log = logging.getLogger(__name__)


def analyze_scene(
    scene_id: str,
    cloud_prob: np.ndarray,
    roi_mask: Optional[np.ndarray] = None,
    valid_mask: Optional[np.ndarray] = None,
    config: Optional[CloudFilterConfig] = None,
    metadata: Optional[SceneMetadata] = None,
    verbose: bool = True,
) -> CloudDensityResult:
    """Analyze a single satellite scene and determine if it meets cloud-filtering criteria.

    Parameters
    ----------
    scene_id : str
        Unique scene or acquisition identifier.
    cloud_prob : np.ndarray
        2D array of pixel-level cloud probabilities [0, 100].
    roi_mask : Optional[np.ndarray]
        2D boolean mask for the target ROI.
    valid_mask : Optional[np.ndarray]
        2D boolean mask for valid/non-nodata pixels.
    config : Optional[CloudFilterConfig]
        Threshold configuration (pixel threshold & scene threshold).
    metadata : Optional[SceneMetadata]
        Additional scene metadata (e.g. acquisition time, ROI ID).
    verbose : bool
        Whether to print foreground Phase 7 logs.

    Returns
    -------
    CloudDensityResult
        Structured result including metrics, passes_threshold boolean, and status.
    """
    if config is None:
        config = CloudFilterConfig()

    acq_time = metadata.acquisition_time if metadata else None
    roi_id = metadata.roi_id if metadata else None
    source_name = config.cloud_source_name

    if verbose:
        print(f"[PHASE 7] Loading scene: {scene_id}")
        print(f"[PHASE 7] Loading cloud probability (Source: {source_name})...")

    try:
        if verbose:
            print(f"[PHASE 7] Applying pixel threshold: {config.pixel_probability_threshold:.1f}%...")
            print("[PHASE 7] Calculating ROI cloud density...")

        density_res = calculate_cloud_density(
            cloud_prob=cloud_prob,
            roi_mask=roi_mask,
            valid_mask=valid_mask,
            pixel_threshold=config.pixel_probability_threshold,
            min_valid_pixel_ratio=config.min_valid_pixel_ratio,
        )

        cloud_density = density_res["cloud_density_percent"]
        passes = cloud_density >= config.scene_density_threshold
        status = "eligible" if passes else "filtered"

        if verbose:
            print(f"[PHASE 7] Cloud density: {cloud_density:.1f}%")
            print(f"[PHASE 7] Scene threshold: {config.scene_density_threshold:.1f}%")
            print(f"[PHASE 7] Status: {status.upper()}")
            print("[PHASE 7] COMPLETE")

        return CloudDensityResult(
            scene_id=scene_id,
            acquisition_time=acq_time,
            roi_id=roi_id,
            cloud_probability_threshold=config.pixel_probability_threshold,
            scene_threshold_percent=config.scene_density_threshold,
            total_pixels=density_res["total_pixels"],
            valid_pixels=density_res["valid_pixels"],
            cloudy_pixels=density_res["cloudy_pixels"],
            cloud_density_percent=cloud_density,
            mean_cloud_probability=density_res["mean_cloud_probability"],
            passes_threshold=passes,
            status=status,
            cloud_source=source_name,
        )

    except Exception as exc:
        log.info("Scene %s failed cloud analysis: %s", scene_id, exc)
        if verbose:
            print(f"[PHASE 7] Error during analysis: {exc}")
            print("[PHASE 7] Status: ERROR")

        return CloudDensityResult(
            scene_id=scene_id,
            acquisition_time=acq_time,
            roi_id=roi_id,
            cloud_probability_threshold=config.pixel_probability_threshold,
            scene_threshold_percent=config.scene_density_threshold,
            total_pixels=cloud_prob.size if isinstance(cloud_prob, np.ndarray) else 0,
            valid_pixels=0,
            cloudy_pixels=0,
            cloud_density_percent=0.0,
            mean_cloud_probability=0.0,
            passes_threshold=False,
            status="error",
            error_message=str(exc),
            cloud_source=source_name,
        )


def filter_scenes_batch(
    scenes: List[Dict[str, Any]],
    config: Optional[CloudFilterConfig] = None,
    sort_by: str = "cloud_density_desc",
    verbose: bool = True,
) -> BatchFilterResult:
    """Analyze a batch of satellite scenes and classify them into eligible and filtered sets.

    Parameters
    ----------
    scenes : List[Dict[str, Any]]
        List of scene dicts, each containing:
          - 'scene_id': str
          - 'cloud_prob': np.ndarray
          - 'roi_mask': Optional[np.ndarray]
          - 'valid_mask': Optional[np.ndarray]
          - 'metadata': Optional[SceneMetadata] or dict
    config : Optional[CloudFilterConfig]
        Filter configuration thresholds.
    sort_by : str
        Sorting order for eligible scenes: 'cloud_density_desc', 'cloud_density_asc',
        'acquisition_date_desc', 'acquisition_date_asc'.
    verbose : bool
        Whether to print console logs.

    Returns
    -------
    BatchFilterResult
        Structured batch report.
    """
    if config is None:
        config = CloudFilterConfig()

    total_scenes = len(scenes)
    results: List[CloudDensityResult] = []

    if verbose:
        print(f"\n{'='*60}")
        print(f"  PHASE 7 — BATCH CLOUD-DENSITY ANALYSIS ({total_scenes} SCENES)")
        print(f"  Pixel Threshold: {config.pixel_probability_threshold:.1f}% | Scene Threshold: {config.scene_density_threshold:.1f}%")
        print(f"{'='*60}")

    for idx, sc in enumerate(scenes, start=1):
        scene_id = sc.get("scene_id", f"scene_{idx}")
        cloud_prob = sc.get("cloud_prob")
        roi_mask = sc.get("roi_mask")
        valid_mask = sc.get("valid_mask")
        meta_raw = sc.get("metadata")

        if isinstance(meta_raw, dict):
            metadata = SceneMetadata(**meta_raw)
        elif isinstance(meta_raw, SceneMetadata):
            metadata = meta_raw
        else:
            metadata = SceneMetadata(scene_id=scene_id)

        if verbose:
            print(f"\n[PHASE 7] Scene {idx}/{total_scenes} (ID: {scene_id})")

        res = analyze_scene(
            scene_id=scene_id,
            cloud_prob=cloud_prob,
            roi_mask=roi_mask,
            valid_mask=valid_mask,
            config=config,
            metadata=metadata,
            verbose=verbose,
        )
        results.append(res)

    eligible_scenes = [r for r in results if r.status == "eligible"]
    filtered_scenes = [r for r in results if r.status == "filtered"]
    error_scenes    = [r for r in results if r.status == "error"]

    # Deterministic sorting
    if sort_by == "cloud_density_desc":
        eligible_scenes.sort(key=lambda x: x.cloud_density_percent, reverse=True)
    elif sort_by == "cloud_density_asc":
        eligible_scenes.sort(key=lambda x: x.cloud_density_percent, reverse=False)
    elif sort_by == "acquisition_date_desc":
        eligible_scenes.sort(key=lambda x: x.acquisition_time or "", reverse=True)
    elif sort_by == "acquisition_date_asc":
        eligible_scenes.sort(key=lambda x: x.acquisition_time or "", reverse=False)

    batch_result = BatchFilterResult(
        total_scenes=total_scenes,
        eligible_count=len(eligible_scenes),
        filtered_count=len(filtered_scenes),
        error_count=len(error_scenes),
        config=config,
        eligible_scenes=eligible_scenes,
        filtered_scenes=filtered_scenes,
        error_scenes=error_scenes,
    )

    if verbose:
        print(f"\n{'='*60}")
        print("  PHASE 7 BATCH FILTERING SUMMARY")
        print(f"{'='*60}")
        print(f"  Total Analyzed : {total_scenes}")
        print(f"  Eligible (PASS): {len(eligible_scenes)}")
        print(f"  Filtered (OUT) : {len(filtered_scenes)}")
        print(f"  Errors         : {len(error_scenes)}")
        print(f"{'='*60}\n")

    return batch_result
