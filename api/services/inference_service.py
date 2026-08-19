"""
api/services/inference_service.py
=================================
Inference execution service reusing the Phase 6 GeospatialInferencePipeline.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any

from sqlalchemy.orm import Session

from api.db.models import Scene, InferenceJob, Result, MetricRecord
from api.schemas.inference import InferenceRequest, InferenceJobResponse
from cloudremoval.config import get_settings
from cloudremoval.inference.pipeline import GeospatialInferencePipeline

log = logging.getLogger(__name__)


class IneligibleSceneError(ValueError):
    """Raised when an ineligible scene (cloud density < 60%) is requested for inference."""


class SceneNotFoundError(ValueError):
    """Raised when a scene is not found in database."""


class CheckpointNotFoundError(FileNotFoundError):
    """Raised when model checkpoint is missing."""


def execute_inference_job(
    db: Session,
    request: InferenceRequest,
) -> InferenceJobResponse:
    """Execute end-to-end geospatial inference on an eligible satellite scene."""
    settings = get_settings()

    # 1. Validate Scene Existence
    scene = db.query(Scene).filter(Scene.scene_id == request.scene_id).first()
    if not scene:
        raise SceneNotFoundError(f"Scene '{request.scene_id}' not found in database.")

    # 2. Validate Scene Eligibility
    if not scene.is_eligible:
        raise IneligibleSceneError(
            f"Scene '{request.scene_id}' is ineligible for reconstruction: "
            f"cloud density is {scene.cloud_density_percent:.1f}%% (below the required threshold {scene.cloud_probability_threshold:.1f}%%)."
        )

    # 3. Validate File Existence
    s2_path = Path(scene.s2_path)
    s1_path = Path(scene.s1_path)

    if not s2_path.exists():
        raise FileNotFoundError(f"Input Sentinel-2 GeoTIFF file not found: {s2_path}")
    if not s1_path.exists():
        raise FileNotFoundError(f"Input Sentinel-1 GeoTIFF file not found: {s1_path}")

    checkpoint_path = Path(settings.CHECKPOINT_DIR) / settings.BEST_CHECKPOINT_NAME
    if not checkpoint_path.exists():
        raise CheckpointNotFoundError(f"Model checkpoint not found: {checkpoint_path}")

    norm_path = Path("data/normalization/normalization.json")

    # 4. Generate Job ID & Initial DB Record
    job_id = f"inf_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    job = InferenceJob(
        job_id=job_id,
        scene_id=scene.scene_id,
        status="running",
        model_name="Modified DSen2-CR",
        checkpoint_name=settings.BEST_CHECKPOINT_NAME,
        tile_size=request.tile_size,
        overlap=request.overlap,
        start_time=datetime.now(tz=timezone.utc),
    )
    db.add(job)
    db.commit()

    print(f"\n[API] Inference request received: {job_id}")
    print(f"[API] Scene validated: {scene.scene_id}")
    print(f"[API] Cloud threshold: PASS ({scene.cloud_density_percent:.1f}%% >= {scene.cloud_probability_threshold:.1f}%%)")
    print(f"[API] Starting inference on {settings.DEVICE.upper()}...")

    try:
        # 5. Initialize and Run Phase 6 Pipeline
        pipeline = GeospatialInferencePipeline(
            checkpoint_path=checkpoint_path,
            norm_path=norm_path,
            device=settings.DEVICE,
            amp_enabled=settings.AMP,
            tile_size=request.tile_size,
            overlap=request.overlap,
            batch_size=request.batch_size,
        )

        out_dir = Path(settings.OUTPUT_DIR) / "inference"
        result_meta = pipeline.run_inference(
            s2_path=s2_path,
            s1_path=s1_path,
            output_dir=out_dir,
            job_id=job_id,
        )

        # 6. Record Results
        geo = result_meta["geospatial"]
        perf = result_meta["performance"]

        result_orm = Result(
            result_id=job_id,
            job_id=job_id,
            scene_id=scene.scene_id,
            geotiff_path=result_meta["output_geotiff"],
            preview_png_path=result_meta["output_preview_png"],
            metadata_path=str(out_dir / f"{job_id}_metadata.json"),
            crs=geo["crs"],
            width=geo["width"],
            height=geo["height"],
            resolution=geo["resolution"][0],
            band_count=geo["band_count"],
        )
        db.add(result_orm)

        # 7. Update Job Status to Completed
        job.status = "completed"
        job.completion_time = datetime.now(tz=timezone.utc)
        job.inference_time_s = perf["model_inference_time_s"]
        job.total_time_s = perf["total_pipeline_time_s"]
        job.peak_vram_gb = perf["peak_vram_gb"]
        db.commit()

        print("[API] Inference completed successfully")
        print(f"[API] Result stored: {result_orm.result_id}\n")

        return InferenceJobResponse(
            job_id=job.job_id,
            scene_id=job.scene_id,
            status=job.status,
            model_name=job.model_name,
            tile_size=job.tile_size,
            overlap=job.overlap,
            created_at=job.created_at.isoformat(),
            completion_time=job.completion_time.isoformat() if job.completion_time else None,
            inference_time_s=job.inference_time_s,
            total_time_s=job.total_time_s,
            peak_vram_gb=job.peak_vram_gb,
            result_id=result_orm.result_id,
        )

    except Exception as exc:
        log.exception("Inference execution failed for job %s: %s", job_id, exc)
        job.status = "failed"
        job.completion_time = datetime.now(tz=timezone.utc)
        job.error_message = str(exc)
        db.commit()
        print(f"[API] Inference failed: {exc}\n")
        raise
