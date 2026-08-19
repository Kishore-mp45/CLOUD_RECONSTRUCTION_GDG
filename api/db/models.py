"""
api/db/models.py
================
SQLAlchemy database models for Phase 8.

Tables:
  - scenes: Stored satellite scenes with cloud-density metadata & eligibility status
  - inference_jobs: Execution status and tracking of inference runs
  - results: Output GeoTIFF and preview PNG references and spatial tags
  - metric_records: Evaluation metrics (PSNR, SSIM, latency) when available
"""

from __future__ import annotations

from datetime import datetime, timezone
from sqlalchemy import Column, String, Integer, Float, Boolean, DateTime, Text, ForeignKey
from sqlalchemy.orm import relationship

from api.db.database import Base


def utcnow() -> datetime:
    return datetime.now(tz=timezone.utc)


class Scene(Base):
    __tablename__ = "scenes"

    scene_id = Column(String(128), primary_key=True, index=True)
    roi_id = Column(String(64), index=True, nullable=False)
    acquisition_time = Column(String(64), nullable=True)
    s2_path = Column(String(512), nullable=False)
    s1_path = Column(String(512), nullable=False)
    target_path = Column(String(512), nullable=True)

    # Cloud Density metrics from Phase 7
    cloud_density_percent = Column(Float, nullable=False, default=0.0)
    cloud_probability_threshold = Column(Float, nullable=False, default=60.0)
    is_eligible = Column(Boolean, nullable=False, default=False, index=True)

    # Spatial metadata
    crs = Column(String(64), nullable=True)
    width = Column(Integer, nullable=True)
    height = Column(Integer, nullable=True)
    resolution = Column(Float, nullable=True)
    extra_metadata = Column(Text, nullable=True)

    created_at = Column(DateTime, default=utcnow, nullable=False)

    # Relationships
    jobs = relationship("InferenceJob", back_populates="scene", cascade="all, delete-orphan")


class InferenceJob(Base):
    __tablename__ = "inference_jobs"

    job_id = Column(String(128), primary_key=True, index=True)
    scene_id = Column(String(128), ForeignKey("scenes.scene_id"), nullable=False, index=True)
    status = Column(String(32), default="queued", index=True, nullable=False)  # queued, running, completed, failed

    model_name = Column(String(128), default="Modified DSen2-CR", nullable=False)
    checkpoint_name = Column(String(128), default="best_model.pth", nullable=False)
    tile_size = Column(Integer, default=256, nullable=False)
    overlap = Column(Integer, default=64, nullable=False)

    start_time = Column(DateTime, default=utcnow, nullable=False)
    completion_time = Column(DateTime, nullable=True)

    inference_time_s = Column(Float, nullable=True)
    total_time_s = Column(Float, nullable=True)
    peak_vram_gb = Column(Float, nullable=True)
    error_message = Column(Text, nullable=True)

    created_at = Column(DateTime, default=utcnow, nullable=False)

    # Relationships
    scene = relationship("Scene", back_populates="jobs")
    result = relationship("Result", back_populates="job", uselist=False, cascade="all, delete-orphan")


class Result(Base):
    __tablename__ = "results"

    result_id = Column(String(128), primary_key=True, index=True)
    job_id = Column(String(128), ForeignKey("inference_jobs.job_id"), unique=True, nullable=False, index=True)
    scene_id = Column(String(128), index=True, nullable=False)

    geotiff_path = Column(String(512), nullable=False)
    preview_png_path = Column(String(512), nullable=False)
    metadata_path = Column(String(512), nullable=False)

    crs = Column(String(64), nullable=False)
    width = Column(Integer, nullable=False)
    height = Column(Integer, nullable=False)
    resolution = Column(Float, nullable=False)
    band_count = Column(Integer, default=13, nullable=False)

    created_at = Column(DateTime, default=utcnow, nullable=False)

    # Relationships
    job = relationship("InferenceJob", back_populates="result")


class MetricRecord(Base):
    __tablename__ = "metric_records"

    id = Column(Integer, primary_key=True, autoincrement=True)
    result_id = Column(String(128), index=True, nullable=True)
    scene_id = Column(String(128), index=True, nullable=True)

    psnr = Column(Float, nullable=True)
    ssim = Column(Float, nullable=True)
    mae = Column(Float, nullable=True)
    rmse = Column(Float, nullable=True)
    sam = Column(Float, nullable=True)
    latency_ms = Column(Float, nullable=True)

    is_ground_truth_available = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime, default=utcnow, nullable=False)
