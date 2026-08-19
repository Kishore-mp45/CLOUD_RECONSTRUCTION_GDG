"""
src/cloudremoval/database/repositories.py
=========================================
Data access repositories for database entities in Phase 9.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Optional, Tuple, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import desc, func

from cloudremoval.database.models import (
    Scene,
    InferenceJob,
    Result,
    Metric,
    ModelVersion,
    ProcessingHistory,
    utc_now,
)
from cloudremoval.database.schemas import (
    SceneCreate,
    InferenceJobCreate,
    InferenceJobUpdate,
    ResultCreate,
    MetricCreate,
    ModelVersionCreate,
    ProcessingHistoryCreate,
)


class SceneRepository:
    """Data access repository for satellite scenes."""

    @staticmethod
    def create(db: Session, scene_in: SceneCreate | Scene) -> Scene:
        if isinstance(scene_in, Scene):
            scene = scene_in
        else:
            scene = Scene(**scene_in.model_dump())
        db.add(scene)
        db.commit()
        db.refresh(scene)
        return scene

    @staticmethod
    def get_by_scene_id(db: Session, scene_id: str) -> Optional[Scene]:
        return db.query(Scene).filter(Scene.scene_id == scene_id).first()

    @staticmethod
    def list(
        db: Session,
        skip: int = 0,
        limit: int = 100,
        eligible_only: Optional[bool] = None,
        roi_id: Optional[str] = None,
    ) -> List[Scene]:
        query = db.query(Scene)
        if eligible_only is not None:
            query = query.filter(Scene.is_eligible == eligible_only)
        if roi_id:
            query = query.filter(Scene.roi_id == roi_id)
        return query.order_by(desc(Scene.created_at)).offset(skip).limit(limit).all()

    @staticmethod
    def count(db: Session, eligible_only: Optional[bool] = None) -> int:
        query = db.query(func.count(Scene.scene_id))
        if eligible_only is not None:
            query = query.filter(Scene.is_eligible == eligible_only)
        return query.scalar() or 0

    @staticmethod
    def update(db: Session, scene_id: str, **kwargs) -> Optional[Scene]:
        scene = SceneRepository.get_by_scene_id(db, scene_id)
        if not scene:
            return None
        for key, value in kwargs.items():
            if hasattr(scene, key):
                setattr(scene, key, value)
        scene.updated_at = utc_now()
        db.commit()
        db.refresh(scene)
        return scene


class ModelVersionRepository:
    """Data access repository for model versions."""

    @staticmethod
    def register(db: Session, model_in: ModelVersionCreate | ModelVersion) -> ModelVersion:
        if isinstance(model_in, ModelVersion):
            model = model_in
        else:
            model = ModelVersion(**model_in.model_dump())
        db.add(model)
        db.commit()
        db.refresh(model)
        return model

    @staticmethod
    def get_active(db: Session) -> Optional[ModelVersion]:
        return db.query(ModelVersion).filter(ModelVersion.is_active == True).order_by(desc(ModelVersion.created_at)).first()

    @staticmethod
    def get_by_version(db: Session, version: str) -> Optional[ModelVersion]:
        return db.query(ModelVersion).filter(ModelVersion.version == version).first()

    @staticmethod
    def list(db: Session) -> List[ModelVersion]:
        return db.query(ModelVersion).order_by(desc(ModelVersion.created_at)).all()

    @staticmethod
    def ensure_default_active_model(db: Session) -> ModelVersion:
        active = ModelVersionRepository.get_active(db)
        if not active:
            active = ModelVersion(
                model_name="Modified DSen2-CR (SAR-Supervised)",
                architecture="DSen2-CR with SAR-Optical Fusion",
                version="v1.0.0",
                checkpoint_path="checkpoints/best_model.pth",
                best_epoch=44,
                s2_channels=13,
                s1_channels=2,
                target_channels=13,
                normalization_version="v1",
                is_active=True,
            )
            db.add(active)
            db.commit()
            db.refresh(active)
        return active


class InferenceJobRepository:
    """Data access repository for inference processing jobs."""

    @staticmethod
    def create(db: Session, job_in: InferenceJobCreate | InferenceJob) -> InferenceJob:
        if isinstance(job_in, InferenceJob):
            job = job_in
        else:
            job = InferenceJob(**job_in.model_dump())
        db.add(job)
        db.commit()
        db.refresh(job)
        return job

    @staticmethod
    def get_by_job_id(db: Session, job_id: str) -> Optional[InferenceJob]:
        return db.query(InferenceJob).filter(InferenceJob.job_id == job_id).first()

    @staticmethod
    def update_status(
        db: Session,
        job_id: str,
        status: str,
        error_message: Optional[str] = None,
        started_at: Optional[datetime] = None,
        completed_at: Optional[datetime] = None,
        inference_duration_s: Optional[float] = None,
        total_duration_s: Optional[float] = None,
    ) -> Optional[InferenceJob]:
        job = InferenceJobRepository.get_by_job_id(db, job_id)
        if not job:
            return None
        job.status = status
        if error_message is not None:
            job.error_message = error_message
        if started_at is not None:
            job.started_at = started_at
            job.start_time = started_at
        if completed_at is not None:
            job.completed_at = completed_at
            job.completion_time = completed_at
        if inference_duration_s is not None:
            job.inference_duration_s = inference_duration_s
            job.inference_time_s = inference_duration_s
        if total_duration_s is not None:
            job.total_duration_s = total_duration_s
            job.total_time_s = total_duration_s
        job.updated_at = utc_now()
        db.commit()
        db.refresh(job)
        return job

    @staticmethod
    def list(db: Session, skip: int = 0, limit: int = 100, status: Optional[str] = None) -> List[InferenceJob]:
        query = db.query(InferenceJob)
        if status:
            query = query.filter(InferenceJob.status == status)
        return query.order_by(desc(InferenceJob.created_at)).offset(skip).limit(limit).all()

    @staticmethod
    def count(db: Session, status: Optional[str] = None) -> int:
        query = db.query(func.count(InferenceJob.job_id))
        if status:
            query = query.filter(InferenceJob.status == status)
        return query.scalar() or 0


class ResultRepository:
    """Data access repository for reconstruction results."""

    @staticmethod
    def create(db: Session, result_in: ResultCreate | Result) -> Result:
        if isinstance(result_in, Result):
            result = result_in
        else:
            result = Result(**result_in.model_dump())
        db.add(result)
        db.commit()
        db.refresh(result)
        return result

    @staticmethod
    def get_by_result_id(db: Session, result_id: str) -> Optional[Result]:
        return db.query(Result).filter(Result.result_id == result_id).first()

    @staticmethod
    def get_by_job_id(db: Session, job_id: str) -> Optional[Result]:
        return db.query(Result).filter(Result.job_id == job_id).first()

    @staticmethod
    def count(db: Session) -> int:
        return db.query(func.count(Result.result_id)).scalar() or 0


class MetricRepository:
    """Data access repository for evaluation metrics."""

    @staticmethod
    def create(db: Session, metric_in: MetricCreate | Metric) -> Metric:
        if isinstance(metric_in, Metric):
            metric = metric_in
        else:
            metric = Metric(**metric_in.model_dump())
        db.add(metric)
        db.commit()
        db.refresh(metric)
        return metric

    @staticmethod
    def get_by_result_id(db: Session, result_id: str) -> Optional[Metric]:
        return db.query(Metric).filter(Metric.result_id == result_id).first()

    @staticmethod
    def count(db: Session) -> int:
        return db.query(func.count(Metric.id)).scalar() or 0


class ProcessingHistoryRepository:
    """Data access repository for application audit events."""

    @staticmethod
    def log_event(
        db: Session,
        entity_type: str,
        entity_id: str,
        action: str,
        status: str = "success",
        message: Optional[str] = None,
        duration_s: Optional[float] = None,
        details_json: Optional[str] = None,
    ) -> ProcessingHistory:
        event = ProcessingHistory(
            entity_type=entity_type,
            entity_id=entity_id,
            action=action,
            status=status,
            message=message,
            duration_s=duration_s,
            details_json=details_json,
        )
        db.add(event)
        db.commit()
        db.refresh(event)
        return event

    @staticmethod
    def list(
        db: Session,
        entity_type: Optional[str] = None,
        entity_id: Optional[str] = None,
        action: Optional[str] = None,
        limit: int = 100,
    ) -> List[ProcessingHistory]:
        query = db.query(ProcessingHistory)
        if entity_type:
            query = query.filter(ProcessingHistory.entity_type == entity_type)
        if entity_id:
            query = query.filter(ProcessingHistory.entity_id == entity_id)
        if action:
            query = query.filter(ProcessingHistory.action == action)
        return query.order_by(desc(ProcessingHistory.id)).limit(limit).all()

    @staticmethod
    def count(db: Session) -> int:
        return db.query(func.count(ProcessingHistory.id)).scalar() or 0
