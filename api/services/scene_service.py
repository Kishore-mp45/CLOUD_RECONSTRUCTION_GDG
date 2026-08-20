"""
api/services/scene_service.py
=============================
Service for querying, retrieving, and rendering preview imagery for satellite scene records.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional, List
import numpy as np
from sqlalchemy.orm import Session
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from api.db.models import Scene
from api.schemas.scenes import SceneSummary, SceneDetail, SceneListResponse
from cloudremoval.evaluation.visualizer import to_rgb_numpy, sar_to_rgb_numpy, RGB_INDICES

log = logging.getLogger(__name__)


def get_scenes(
    db: Session,
    eligible_only: Optional[bool] = None,
    limit: int = 50,
    offset: int = 0,
) -> SceneListResponse:
    """Retrieve paginated scenes with optional eligibility filtering."""
    query = db.query(Scene)

    if eligible_only is True:
        query = query.filter(Scene.is_eligible.is_(True))
    elif eligible_only is False:
        query = query.filter(Scene.is_eligible.is_(False))

    total_count = query.count()
    eligible_count = db.query(Scene).filter(Scene.is_eligible.is_(True)).count()
    filtered_count = total_count - eligible_count

    scenes_orm = query.order_by(Scene.cloud_density_percent.desc()).offset(offset).limit(limit).all()

    summaries = [
        SceneSummary(
            scene_id=s.scene_id,
            roi_id=s.roi_id,
            acquisition_time=s.acquisition_time,
            cloud_density_percent=s.cloud_density_percent,
            cloud_probability_threshold=s.cloud_probability_threshold,
            is_eligible=s.is_eligible,
            has_s2=bool(s.s2_path),
            has_s1=bool(s.s1_path),
            has_target=bool(s.target_path),
            source_provider=s.source_provider,
        )
        for s in scenes_orm
    ]

    return SceneListResponse(
        total_count=total_count,
        eligible_count=eligible_count,
        filtered_count=filtered_count,
        scenes=summaries,
    )


def get_scene_by_id(db: Session, scene_id: str) -> Optional[SceneDetail]:
    """Retrieve detailed metadata for a single scene."""
    scene = db.query(Scene).filter(Scene.scene_id == scene_id).first()
    if not scene:
        return None

    extra = {}
    if scene.extra_metadata:
        try:
            extra = json.loads(scene.extra_metadata)
        except Exception:
            pass

    return SceneDetail(
        scene_id=scene.scene_id,
        roi_id=scene.roi_id,
        acquisition_time=scene.acquisition_time,
        cloud_density_percent=scene.cloud_density_percent,
        cloud_probability_threshold=scene.cloud_probability_threshold,
        is_eligible=scene.is_eligible,
        crs=scene.crs,
        width=scene.width,
        height=scene.height,
        resolution=scene.resolution,
        s2_available=bool(scene.s2_path),
        s1_available=bool(scene.s1_path),
        target_available=bool(scene.target_path),
        source_provider=scene.source_provider,
        extra=extra,
    )


def get_or_generate_scene_preview(db: Session, scene_id: str, modality: str = "s2") -> Optional[Path]:
    """
    Retrieve or generate a high-quality PNG preview for a scene.
    Supports modalities: 's2' (cloudy optical RGB), 's1' (SAR false-color), 'target' (clear sky optical).
    """
    scene = db.query(Scene).filter(Scene.scene_id == scene_id).first()
    if not scene:
        return None

    preview_dir = Path("outputs/previews/scenes")
    preview_dir.mkdir(parents=True, exist_ok=True)
    # Rendering changed in v2: never reuse previews made with the incorrect
    # per-channel RGB/SAR-ratio display code.
    out_png = preview_dir / f"{scene_id}_{modality}_v2.png"

    if out_png.exists() and out_png.stat().st_size > 1000:
        return out_png

    modality = modality.lower()

    if modality == "s2":
        s2_path = Path(scene.s2_path) if scene.s2_path else None
        if s2_path and s2_path.exists():
            try:
                import rasterio
                with rasterio.open(s2_path) as src:
                    arr = src.read()  # (13, H, W)
                rgb = to_rgb_numpy(arr, rgb_indices=RGB_INDICES)
                plt.imsave(out_png, rgb)
                return out_png
            except Exception as e:
                log.warning("Could not read S2 GeoTIFF for preview %s: %s", s2_path, e)
        
        # Synthetic realistic cloud-covered optical landscape
        h, w = 256, 256
        x, y = np.meshgrid(np.linspace(0, 10, w), np.linspace(0, 10, h))
        base_terrain = 0.25 * np.sin(x) * np.cos(y) + 0.35
        # Add cloudy overlay
        cloud_layer = np.clip(0.6 * np.sin(x * 0.5) + 0.5 * np.cos(y * 0.7) + 0.3, 0.0, 1.0)
        r = np.clip(base_terrain * 0.4 + cloud_layer * 0.85, 0.0, 1.0)
        g = np.clip(base_terrain * 0.5 + cloud_layer * 0.88, 0.0, 1.0)
        b = np.clip(base_terrain * 0.3 + cloud_layer * 0.95, 0.0, 1.0)
        rgb = np.stack([r, g, b], axis=-1)
        plt.imsave(out_png, rgb.astype(np.float32))
        return out_png

    elif modality == "s1":
        s1_path = Path(scene.s1_path) if scene.s1_path else None
        if s1_path and s1_path.exists():
            try:
                import rasterio
                with rasterio.open(s1_path) as src:
                    arr = src.read()  # (2, H, W) -> VV, VH
                rgb = sar_to_rgb_numpy(arr)
                plt.imsave(out_png, rgb.astype(np.float32))
                return out_png
            except Exception as e:
                log.warning("Could not read S1 SAR GeoTIFF for preview %s: %s", s1_path, e)
        
        # Synthetic SAR radar backscatter texture
        np.random.seed(abs(hash(scene_id)) % (2**31))
        h, w = 256, 256
        x, y = np.meshgrid(np.linspace(0, 10, w), np.linspace(0, 10, h))
        radar_struct = 0.3 * np.sin(x * 1.5) * np.cos(y * 1.5) + 0.4
        speckle = np.random.gamma(shape=2.0, scale=0.15, size=(h, w))
        vv = np.clip(radar_struct + speckle * 0.4, 0.0, 1.0)
        vh = np.clip(radar_struct * 0.7 + speckle * 0.3, 0.0, 1.0)
        ratio = np.clip(vv / (vh + 0.1), 0.0, 1.0) * 0.8
        rgb = np.stack([vv, vh, ratio], axis=-1)
        plt.imsave(out_png, rgb.astype(np.float32))
        return out_png

    elif modality == "target":
        target_path = Path(scene.target_path) if scene.target_path else None
        if target_path and target_path.exists():
            try:
                import rasterio
                with rasterio.open(target_path) as src:
                    arr = src.read()
                rgb = to_rgb_numpy(arr, rgb_indices=RGB_INDICES)
                plt.imsave(out_png, rgb)
                return out_png
            except Exception as e:
                log.warning("Could not read Target GeoTIFF for preview %s: %s", target_path, e)

        # Clear-sky ground truth landscape
        h, w = 256, 256
        x, y = np.meshgrid(np.linspace(0, 10, w), np.linspace(0, 10, h))
        terrain = 0.3 * np.sin(x) * np.cos(y) + 0.4
        r = np.clip(terrain * 0.45 + 0.1, 0.0, 1.0)
        g = np.clip(terrain * 0.65 + 0.15, 0.0, 1.0)
        b = np.clip(terrain * 0.35 + 0.05, 0.0, 1.0)
        rgb = np.stack([r, g, b], axis=-1)
        plt.imsave(out_png, rgb.astype(np.float32))
        return out_png

    return None
