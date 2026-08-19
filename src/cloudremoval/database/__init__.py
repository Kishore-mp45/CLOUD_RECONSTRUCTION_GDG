"""
cloudremoval.database
====================
Persistent SQLite + SQLAlchemy database layer for ALLClear Cloud Removal (Phase 9).

Public API:
    Base, engine, SessionLocal, get_db, init_db
    Scene, InferenceJob, Result, Metric, ModelVersion, ProcessingHistory
    SceneRepository, InferenceJobRepository, ResultRepository, MetricRepository
    ModelVersionRepository, ProcessingHistoryRepository
"""

from cloudremoval.database.database import (
    Base,
    engine,
    SessionLocal,
    get_db,
    init_db,
    create_db_engine,
    get_default_database_url,
)
from cloudremoval.database.models import (
    Scene,
    InferenceJob,
    Result,
    Metric,
    ModelVersion,
    ProcessingHistory,
)
from cloudremoval.database.repositories import (
    SceneRepository,
    InferenceJobRepository,
    ResultRepository,
    MetricRepository,
    ModelVersionRepository,
    ProcessingHistoryRepository,
)
from cloudremoval.database.schemas import (
    SceneCreate,
    SceneRead,
    InferenceJobCreate,
    InferenceJobRead,
    InferenceJobUpdate,
    ResultCreate,
    ResultRead,
    MetricCreate,
    MetricRead,
    ModelVersionCreate,
    ModelVersionRead,
    ProcessingHistoryCreate,
    ProcessingHistoryRead,
)

__all__ = [
    "Base",
    "engine",
    "SessionLocal",
    "get_db",
    "init_db",
    "create_db_engine",
    "get_default_database_url",
    "Scene",
    "InferenceJob",
    "Result",
    "Metric",
    "ModelVersion",
    "ProcessingHistory",
    "SceneRepository",
    "InferenceJobRepository",
    "ResultRepository",
    "MetricRepository",
    "ModelVersionRepository",
    "ProcessingHistoryRepository",
    "SceneCreate",
    "SceneRead",
    "InferenceJobCreate",
    "InferenceJobRead",
    "InferenceJobUpdate",
    "ResultCreate",
    "ResultRead",
    "MetricCreate",
    "MetricRead",
    "ModelVersionCreate",
    "ModelVersionRead",
    "ProcessingHistoryCreate",
    "ProcessingHistoryRead",
]
