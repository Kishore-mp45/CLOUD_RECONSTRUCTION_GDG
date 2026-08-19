"""
tests/test_database.py
======================
Pytest unit and integration test suite for the Phase 9 Database Layer.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from cloudremoval.database.models import (
    Base,
    Scene,
    InferenceJob,
    Result,
    Metric,
    ModelVersion,
    ProcessingHistory,
    utc_now,
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
    InferenceJobCreate,
    ResultCreate,
    MetricCreate,
    ModelVersionCreate,
    ProcessingHistoryCreate,
)


@pytest.fixture(scope="function")
def test_db():
    """Create an isolated temporary SQLite database for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp_file:
        db_path = tmp_file.name

    test_engine = create_engine(
        f"sqlite:///{Path(db_path).as_posix()}",
        connect_args={"check_same_thread": False},
    )
    # Enable foreign keys
    with test_engine.connect() as conn:
        conn.execute(text("PRAGMA foreign_keys=ON"))

    Base.metadata.create_all(bind=test_engine)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)
    session = TestingSessionLocal()

    yield session, test_engine, db_path

    session.close()
    test_engine.dispose()
    try:
        Path(db_path).unlink(missing_ok=True)
    except Exception:
        pass


class TestDatabaseInitialization:

    def test_tables_created(self, test_db) -> None:
        session, engine, _ = test_db
        from sqlalchemy import inspect
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        expected = ["scenes", "model_versions", "inference_jobs", "results", "metric_records", "processing_history"]
        for t in expected:
            assert t in tables


class TestSceneRepositoryCRUD:

    def test_create_and_get_scene(self, test_db) -> None:
        session, _, _ = test_db
        scene_in = SceneCreate(
            scene_id="scene_test_001",
            roi_id="roi_001",
            acquisition_time="2022-06-15",
            source_provider="ALLClear",
            s2_path="data/s2.tif",
            s1_path="data/s1.tif",
            cloud_density_percent=78.5,
            cloud_probability_threshold=60.0,
            is_eligible=True,
            crs="EPSG:32643",
            width=256,
            height=256,
            resolution=10.0,
        )
        created = SceneRepository.create(session, scene_in)
        assert created.scene_id == "scene_test_001"

        fetched = SceneRepository.get_by_scene_id(session, "scene_test_001")
        assert fetched is not None
        assert fetched.roi_id == "roi_001"
        assert fetched.is_eligible is True
        assert fetched.cloud_density_percent == 78.5

    def test_list_and_count_scenes(self, test_db) -> None:
        session, _, _ = test_db
        # Seed 2 scenes
        SceneRepository.create(
            session,
            SceneCreate(
                scene_id="sc_el",
                roi_id="roi_a",
                acquisition_time="2022-01-01",
                s2_path="s2.tif",
                s1_path="s1.tif",
                cloud_density_percent=85.0,
                is_eligible=True,
                crs="EPSG:32643",
                width=256,
                height=256,
                resolution=10.0,
            ),
        )
        SceneRepository.create(
            session,
            SceneCreate(
                scene_id="sc_inel",
                roi_id="roi_b",
                acquisition_time="2022-01-02",
                s2_path="s2.tif",
                s1_path="s1.tif",
                cloud_density_percent=20.0,
                is_eligible=False,
                crs="EPSG:32643",
                width=256,
                height=256,
                resolution=10.0,
            ),
        )

        assert SceneRepository.count(session) == 2
        assert SceneRepository.count(session, eligible_only=True) == 1
        assert SceneRepository.count(session, eligible_only=False) == 1

        all_scenes = SceneRepository.list(session)
        assert len(all_scenes) == 2
        eligible_scenes = SceneRepository.list(session, eligible_only=True)
        assert len(eligible_scenes) == 1
        assert eligible_scenes[0].scene_id == "sc_el"


class TestModelVersionRepository:

    def test_register_and_get_active_model(self, test_db) -> None:
        session, _, _ = test_db
        model_in = ModelVersionCreate(
            model_name="Modified DSen2-CR (SAR-Supervised)",
            architecture="DSen2-CR",
            version="v1.0.0",
            checkpoint_path="checkpoints/best_model.pth",
            best_epoch=44,
            s2_channels=13,
            s1_channels=2,
            target_channels=13,
            is_active=True,
        )
        registered = ModelVersionRepository.register(session, model_in)
        assert registered.id is not None
        assert registered.version == "v1.0.0"

        active = ModelVersionRepository.get_active(session)
        assert active is not None
        assert active.checkpoint_path == "checkpoints/best_model.pth"


class TestInferenceJobAndResultFlow:

    def test_job_and_result_lifecycle(self, test_db) -> None:
        session, _, _ = test_db
        # 1. Create Scene
        scene = SceneRepository.create(
            session,
            SceneCreate(
                scene_id="sc_job_test",
                roi_id="roi_002",
                acquisition_time="2022-05-01",
                s2_path="s2.tif",
                s1_path="s1.tif",
                cloud_density_percent=90.0,
                is_eligible=True,
                crs="EPSG:32643",
                width=256,
                height=256,
                resolution=10.0,
            ),
        )

        # 2. Create InferenceJob
        job_in = InferenceJobCreate(
            job_id="job_001",
            scene_id="sc_job_test",
            tile_size=256,
            overlap=64,
            batch_size=4,
        )
        job = InferenceJobRepository.create(session, job_in)
        assert job.status == "queued"

        # 3. Update status to running then completed
        job_updated = InferenceJobRepository.update_status(
            session,
            job_id="job_001",
            status="completed",
            inference_duration_s=1.25,
            total_duration_s=3.50,
        )
        assert job_updated.status == "completed"
        assert job_updated.inference_duration_s == 1.25

        # 4. Create Result
        result_in = ResultCreate(
            result_id="res_001",
            job_id="job_001",
            scene_id="sc_job_test",
            geotiff_path="outputs/inference/job_001_reconstructed.tif",
            preview_png_path="outputs/inference/job_001_preview.png",
            metadata_path="outputs/inference/job_001_metadata.json",
            crs="EPSG:32643",
            width=256,
            height=256,
            resolution=10.0,
            band_count=13,
            inference_time_s=1.25,
            total_time_s=3.50,
            peak_vram_gb=1.85,
        )
        result = ResultRepository.create(session, result_in)
        assert result.result_id == "res_001"

        # Verify relational access
        fetched_job = InferenceJobRepository.get_by_job_id(session, "job_001")
        assert fetched_job.result is not None
        assert fetched_job.result.result_id == "res_001"
        assert fetched_job.scene.scene_id == "sc_job_test"


class TestMetricRepositoryAndNullSupport:

    def test_metric_creation_with_null_ground_truth(self, test_db) -> None:
        session, _, _ = test_db
        # Seed scene, job, result
        SceneRepository.create(
            session,
            SceneCreate(
                scene_id="sc_live",
                roi_id="roi_live",
                acquisition_time="2022-08-01",
                s2_path="s2.tif",
                s1_path="s1.tif",
                cloud_density_percent=88.0,
                is_eligible=True,
                crs="EPSG:32643",
                width=256,
                height=256,
                resolution=10.0,
            ),
        )
        InferenceJobRepository.create(
            session,
            InferenceJobCreate(job_id="job_live", scene_id="sc_live"),
        )
        ResultRepository.create(
            session,
            ResultCreate(
                result_id="res_live",
                job_id="job_live",
                scene_id="sc_live",
                geotiff_path="outputs/inf.tif",
                preview_png_path="outputs/prev.png",
                metadata_path="outputs/meta.json",
                crs="EPSG:32643",
                width=256,
                height=256,
                resolution=10.0,
                inference_time_s=1.0,
                total_time_s=2.0,
            ),
        )

        # Metric without ground truth -> PSNR/SSIM/MAE/RMSE should be NULL
        metric_null = MetricRepository.create(
            session,
            MetricCreate(
                result_id="res_live",
                psnr=None,
                ssim=None,
                mae=None,
                rmse=None,
                sam=None,
                is_available=False,
                evaluation_source="live_scene_no_ground_truth",
            ),
        )
        assert metric_null.id is not None
        assert metric_null.psnr is None
        assert metric_null.is_available is False

        # Metric with ground truth
        metric_gt = MetricRepository.create(
            session,
            MetricCreate(
                result_id="res_live",
                psnr=38.82,
                ssim=0.9396,
                mae=0.0125,
                rmse=0.0210,
                sam=2.15,
                is_available=True,
                evaluation_source="ground_truth_target",
            ),
        )
        assert metric_gt.psnr == 38.82
        assert metric_gt.is_available is True


class TestProcessingHistoryAuditTrail:

    def test_log_and_list_audit_events(self, test_db) -> None:
        session, _, _ = test_db
        ProcessingHistoryRepository.log_event(
            session,
            entity_type="scene",
            entity_id="scene_001",
            action="SCENE_ANALYZED",
            status="success",
            message="Cloud density 85.2% computed",
        )
        ProcessingHistoryRepository.log_event(
            session,
            entity_type="inference_job",
            entity_id="job_001",
            action="INFERENCE_STARTED",
            status="success",
        )
        ProcessingHistoryRepository.log_event(
            session,
            entity_type="inference_job",
            entity_id="job_001",
            action="INFERENCE_COMPLETED",
            status="success",
            duration_s=2.45,
        )

        assert ProcessingHistoryRepository.count(session) == 3
        events = ProcessingHistoryRepository.list(session, entity_id="job_001")
        assert len(events) == 2
        assert events[0].action == "INFERENCE_COMPLETED"  # latest first
        assert events[1].action == "INFERENCE_STARTED"


class TestPersistenceAcrossSessions:

    def test_data_persists_after_session_close(self, test_db) -> None:
        session, engine, _ = test_db
        SceneRepository.create(
            session,
            SceneCreate(
                scene_id="sc_persist_test",
                roi_id="roi_persist",
                acquisition_time="2022-07-01",
                s2_path="s2.tif",
                s1_path="s1.tif",
                cloud_density_percent=75.0,
                is_eligible=True,
                crs="EPSG:32643",
                width=256,
                height=256,
                resolution=10.0,
            ),
        )
        session.close()

        # Reopen a new fresh session on the same database
        NewSession = sessionmaker(bind=engine)
        new_session = NewSession()
        try:
            persisted = SceneRepository.get_by_scene_id(new_session, "sc_persist_test")
            assert persisted is not None
            assert persisted.scene_id == "sc_persist_test"
            assert persisted.cloud_density_percent == 75.0
        finally:
            new_session.close()
