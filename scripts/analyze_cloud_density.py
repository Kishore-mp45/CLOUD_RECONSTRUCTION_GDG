"""
scripts/analyze_cloud_density.py
=================================
Cloud-Density Analysis & Filtering CLI — Phase 7

Usage
-----
# 1. Run automated multi-condition demo:
python scripts/analyze_cloud_density.py --demo

# 2. Analyze specific cloud-probability array:
python scripts/analyze_cloud_density.py \
    --scene-id S2A_MSIL2A_20220315 \
    --cloud-prob-file path/to/cloud_prob.npy \
    --pixel-thresh 60.0 \
    --scene-thresh 60.0
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any

# Ensure src/ is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

import numpy as np

from cloudremoval.cloud import (
    CloudFilterConfig,
    analyze_scene,
    filter_scenes_batch,
    SceneMetadata,
)

log = logging.getLogger(__name__)


def run_demo(pixel_thresh: float, scene_thresh: float, output_path: Path | None) -> None:
    """Run a comprehensive demonstration covering clear, moderate, dense, and edge-case scenes."""
    config = CloudFilterConfig(
        pixel_probability_threshold=pixel_thresh,
        scene_density_threshold=scene_thresh,
    )

    # 1. Clear Sky Scene (probabilities 0% - 25%)
    np.random.seed(42)
    clear_prob = np.random.uniform(0.0, 25.0, size=(256, 256)).astype(np.float32)

    # 2. Moderate Cloud Scene (mixed 30% - 85% with circular cloud cluster)
    mod_prob = np.random.uniform(10.0, 45.0, size=(256, 256)).astype(np.float32)
    # Add a cloud clump in center
    yy, xx = np.ogrid[:256, :256]
    dist_from_center = np.sqrt((xx - 128) ** 2 + (yy - 128) ** 2)
    mod_prob[dist_from_center <= 80] = np.random.uniform(70.0, 95.0, size=(dist_from_center <= 80).sum())

    # 3. Dense Monsoon Overcast Scene (probabilities 75% - 100%)
    dense_prob = np.random.uniform(70.0, 100.0, size=(256, 256)).astype(np.float32)

    # 4. Partial / Edge Nodata Scene (50% nodata/NaNs, rest cloudy)
    partial_prob = np.random.uniform(65.0, 95.0, size=(256, 256)).astype(np.float32)
    partial_valid = np.ones((256, 256), dtype=bool)
    partial_valid[:, :128] = False  # Left half nodata

    # 5. Invalid / Missing Data Scene (All NaNs)
    invalid_prob = np.full((256, 256), np.nan, dtype=np.float32)

    scenes: List[Dict[str, Any]] = [
        {
            "scene_id": "DEMO_SCENE_01_CLEAR_SKY",
            "cloud_prob": clear_prob,
            "metadata": {
                "scene_id": "DEMO_SCENE_01_CLEAR_SKY",
                "acquisition_time": "2022-03-10T10:30:00Z",
                "roi_id": "roi_india_south",
            },
        },
        {
            "scene_id": "DEMO_SCENE_02_MODERATE_CUMULUS",
            "cloud_prob": mod_prob,
            "metadata": {
                "scene_id": "DEMO_SCENE_02_MODERATE_CUMULUS",
                "acquisition_time": "2022-06-15T05:45:00Z",
                "roi_id": "roi_india_central",
            },
        },
        {
            "scene_id": "DEMO_SCENE_03_DENSE_MONSOON",
            "cloud_prob": dense_prob,
            "metadata": {
                "scene_id": "DEMO_SCENE_03_DENSE_MONSOON",
                "acquisition_time": "2022-08-20T06:15:00Z",
                "roi_id": "roi_india_west",
            },
        },
        {
            "scene_id": "DEMO_SCENE_04_PARTIAL_ROI_MASK",
            "cloud_prob": partial_prob,
            "valid_mask": partial_valid,
            "metadata": {
                "scene_id": "DEMO_SCENE_04_PARTIAL_ROI_MASK",
                "acquisition_time": "2022-09-02T11:00:00Z",
                "roi_id": "roi_india_north",
            },
        },
        {
            "scene_id": "DEMO_SCENE_05_ALL_NODATA_CORRUPT",
            "cloud_prob": invalid_prob,
            "metadata": {
                "scene_id": "DEMO_SCENE_05_ALL_NODATA_CORRUPT",
                "acquisition_time": "2022-10-01T04:20:00Z",
                "roi_id": "roi_india_east",
            },
        },
    ]

    batch_res = filter_scenes_batch(scenes, config=config, sort_by="cloud_density_desc", verbose=True)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(batch_res.model_dump_json(indent=2))
        print(f"[PHASE 7] Saved batch results JSON to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze Sentinel-2 scene cloud density and filter by threshold (Phase 7)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--demo", action="store_true", help="Run automated demonstration with synthetic scene suite")
    parser.add_argument("--scene-id", type=str, default="custom_scene_01", help="Scene identifier")
    parser.add_argument("--cloud-prob-file", type=str, default=None, help="Path to .npy array or GeoTIFF")
    parser.add_argument("--pixel-thresh", type=float, default=60.0, help="Pixel cloud-probability threshold (0-100%%)")
    parser.add_argument("--scene-thresh", type=float, default=60.0, help="Scene cloud-density threshold (0-100%%)")
    parser.add_argument("--output", type=str, default="outputs/cloud/cloud_density_results.json", help="Path to save result JSON")

    args = parser.parse_args()

    out_path = Path(args.output) if args.output else None

    if args.demo or args.cloud_prob_file is None:
        run_demo(args.pixel_thresh, args.scene_thresh, out_path)
    else:
        prob_file = Path(args.cloud_prob_file)
        if not prob_file.exists():
            print(f"[FATAL] Cloud probability file not found: {prob_file}", file=sys.stderr)
            sys.exit(1)

        if prob_file.suffix == ".npy":
            arr = np.load(prob_file)
        else:
            import rasterio
            with rasterio.open(prob_file) as src:
                arr = src.read(1)

        config = CloudFilterConfig(
            pixel_probability_threshold=args.pixel_thresh,
            scene_density_threshold=args.scene_thresh,
        )

        res = analyze_scene(
            scene_id=args.scene_id,
            cloud_prob=arr,
            config=config,
            verbose=True,
        )

        if out_path:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(res.model_dump_json(indent=2))
            print(f"[PHASE 7] Saved result to: {out_path}")


if __name__ == "__main__":
    main()
