"""
src/cloudremoval/database/models.py
===================================
SQLAlchemy ORM models for the persistent application database (Phase 9).

Entities:
  1. Scene: Multi-sensor satellite scenes and cloud metrics
  2. InferenceJob: Cloud removal processing jobs and statuses
  3. Result: Output GeoTIFF/PNG file references and spatial bounds
  4. Metric: Radiometric evaluation metrics (PSNR, SSIM, MAE, RMSE, SAM)
  5. ModelVersion: Checkpoint versioning and architecture configuration
  6. ProcessingHistory: Audit log of all operational events
"""

from __future__ import annotations

from datetime import datetime, timezone
from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    Boolean,
    DateTime,
    ForeignKey,
    Text,
    Index,
)
from sqlalchemy.orm import relationship

from cloudremoval.database.database import Base


def utc_now() -> datetime:
    """Return current timestamp in UTC timezone."""
    return datetime.now(timezone.utc)


class Scene(Base):
    """Satellite scene record (Sentinel-2 optical + Sentinel-1 SAR)."""
    __tablename__ = "scenes"

    scene_id = Column(String(128), primary_key=True, index=True)
    external_scene_id = Column(String(256), index=True, nullable=True)
    roi_id = Column(String(64), index=True, nullable=False)
    acquisition_time = Column(String(64), index=True, nullable=False)
    source_provider = Column(String(64), default="ALLClear", index=True, nullable=False)
    s2_path = Column(String(512), nullable=False)
    s1_path = Column(String(512), nullable=False)
    target_path = Column(String(512), nullable=True)
    cloud_density_percent = Column(Float, nullable=False)
    cloud_probability_threshold = Column(Float, default=60.0, nullable=False)
    is_eligible = Column(Boolean, default=False, index=True, nullable=False)
    crs = Column(String(64), nullable=False)
    width = Column(Integer, nullable=False)
    height = Column(Integer, nullable=False)
    resolution = Column(Float, nullable=False)
    bounds_json = Column(Text, nullable=True)
    extra_metadata = Column(Text, nullable=True)
    created_at = Column(DateTime, default=utc_now, nullable=False)
    updated_at = Column(DateTime, default=utc_now, onupdate=utc_now, nullable=False)

    # Relationships
    inference_jobs = relationship("InferenceJob", back_populates="scene", cascade="all, delete-orphan")
    jobs = relationship("InferenceJob", back_populates="scene", viewonly=True)

    def __repr__(self) -> str:
        return f"<Scene(scene_id='{self.scene_id}', cloud={self.cloud_density_percent}%, eligible={self.is_eligible})>"


class ModelVersion(Base):
    """Trained model version and checkpoint reference."""
    __tablename__ = "model_versions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    model_name = Column(String(128), nullable=False)
    architecture = Column(String(128), nullable=False)
    version = Column(String(64), index=True, nullable=False)
    checkpoint_path = Column(String(512), nullable=False)
    checkpoint_hash = Column(String(128), nullable=True)
    best_epoch = Column(Integer, nullable=True)
    s2_channels = Column(Integer, default=13, nullable=False)
    s1_channels = Column(Integer, default=2, nullable=False)
    target_channels = Column(Integer, default=13, nullable=False)
    normalization_version = Column(String(32), default="v1", nullable=False)
    training_config_json = Column(Text, nullable=True)
    is_active = Column(Boolean, default=True, index=True, nullable=False)
    created_at = Column(DateTime, default=utc_now, nullable=False)

    # Relationships
    inference_jobs = relationship("InferenceJob", back_populates="model_version")

    def __repr__(self) -> str:
        return f"<ModelVersion(model='{self.model_name}', version='{self.version}', active={self.is_active})>"


class InferenceJob(Base):
    """Inference job execution record."""
    __tablename__ = "inference_jobs"

    job_id = Column(String(128), primary_key=True, index=True)
    scene_id = Column(String(128), ForeignKey("scenes.scene_id", ondelete="CASCADE"), index=True, nullable=False)
    model_version_id = Column(Integer, ForeignKey("model_versions.id"), nullable=True, index=True)
    status = Column(String(32), default="queued", index=True, nullable=False)  # queued, running, completed, failed
    tile_size = Column(Integer, default=256, nullable=False)
    overlap = Column(Integer, default=64, nullable=False)
    batch_size = Column(Integer, default=4, nullable=False)
    model_name = Column(String(128), default="Modified DSen2-CR", nullable=False)
    checkpoint_name = Column(String(128), default="best_model.pth", nullable=False)
    error_message = Column(Text, nullable=True)
    requested_at = Column(DateTime, default=utc_now, nullable=False)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    start_time = Column(DateTime, default=utc_now, nullable=False)
    completion_time = Column(DateTime, nullable=True)
    inference_duration_s = Column(Float, nullable=True)
    total_duration_s = Column(Float, nullable=True)
    inference_time_s = Column(Float, nullable=True)
    total_time_s = Column(Float, nullable=True)
    peak_vram_gb = Column(Float, nullable=True)
    created_at = Column(DateTime, default=utc_now, nullable=False)
    updated_at = Column(DateTime, default=utc_now, onupdate=utc_now, nullable=False)

    # Relationships
    scene = relationship("Scene", back_populates="inference_jobs")
    model_version = relationship("ModelVersion", back_populates="inference_jobs")
    result = relationship("Result", back_populates="inference_job", uselist=False, cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<InferenceJob(job_id='{self.job_id}', scene_id='{self.scene_id}', status='{self.status}')>"


class Result(Base):
    """Geospatial reconstruction results and file references."""
    __tablename__ = "results"

    result_id = Column(String(128), primary_key=True, index=True)
    job_id = Column(String(128), ForeignKey("inference_jobs.job_id", ondelete="CASCADE"), unique=True, index=True, nullable=False)
    scene_id = Column(String(128), ForeignKey("scenes.scene_id", ondelete="CASCADE"), index=True, nullable=False)
    geotiff_path = Column(String(512), nullable=False)
    preview_png_path = Column(String(512), nullable=False)
    metadata_path = Column(String(512), nullable=False)
    crs = Column(String(64), nullable=False)
    width = Column(Integer, nullable=False)
    height = Column(Integer, nullable=False)
    resolution = Column(Float, nullable=False)
    band_count = Column(Integer, default=13, nullable=False)
    bounds_json = Column(Text, nullable=True)
    inference_time_s = Column(Float, nullable=False)
    total_time_s = Column(Float, nullable=False)
    peak_vram_gb = Column(Float, nullable=True)
    created_at = Column(DateTime, default=utc_now, nullable=False)

    # Relationships
    inference_job = relationship("InferenceJob", back_populates="result")
    job = relationship("InferenceJob", back_populates="result", viewonly=True)
    metric = relationship("Metric", back_populates="result", uselist=False, cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<Result(result_id='{self.result_id}', job_id='{self.job_id}')>"


class Metric(Base):
    """Quantitative radiometric metrics for reconstructed imagery."""
    __tablename__ = "metric_records"

    id = Column(Integer, primary_key=True, autoincrement=True)
    result_id = Column(String(128), ForeignKey("results.result_id", ondelete="CASCADE"), index=True, nullable=False)
    psnr = Column(Float, nullable=True)
    ssim = Column(Float, nullable=True)
    mae = Column(Float, nullable=True)
    rmse = Column(Float, nullable=True)
    sam = Column(Float, nullable=True)
    is_available = Column(Boolean, default=False, index=True, nullable=False)
    evaluation_source = Column(String(128), default="ground_truth_target", nullable=False)
    created_at = Column(DateTime, default=utc_now, nullable=False)

    # Relationships
    result = relationship("Result", back_populates="metric")

    def __repr__(self) -> str:
        return f"<Metric(result_id='{self.result_id}', psnr={self.psnr}, available={self.is_available})>"


# Alias for backwards compatibility
MetricRecord = Metric


class ProcessingHistory(Base):
    """Audit log of application events and pipeline operations."""
    __tablename__ = "processing_history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    entity_type = Column(String(64), index=True, nullable=False)
    entity_id = Column(String(128), index=True, nullable=False)
    action = Column(String(64), index=True, nullable=False)
    status = Column(String(32), default="success", index=True, nullable=False)
    message = Column(Text, nullable=True)
    duration_s = Column(Float, nullable=True)
    details_json = Column(Text, nullable=True)
    created_at = Column(DateTime, default=utc_now, index=True, nullable=False)

    def __repr__(self) -> str:
        return f"<ProcessingHistory(action='{self.action}', entity='{self.entity_type}:{self.entity_id}', status='{self.status}')>"


# Additional composite indexes for optimized query performance
Index("ix_scenes_roi_eligible", Scene.roi_id, Scene.is_eligible)
Index("ix_jobs_scene_status", InferenceJob.scene_id, InferenceJob.status)
Index("ix_history_entity_action", ProcessingHistory.entity_type, ProcessingHistory.action)
