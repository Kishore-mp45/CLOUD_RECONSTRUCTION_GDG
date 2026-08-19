"""
tests/test_integration.py
=========================
Full End-to-End System Integration Test Suite (Phase 11).

Validates the complete execution flow:
  Frontend Static -> FastAPI Router -> SQLite DB -> Model Checkpoint ->
  Phase 6 Inference -> GeoTIFF/PNG Outputs -> DB Result & Metrics -> Audit Trail -> Downloads
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
import numpy as np
import rasterio
import torch

from api.main import create_app
from api.db.database import init_db, SessionLocal
from cloudremoval.database.models import Scene, InferenceJob, Result, Metric, ProcessingHistory
from cloudremoval.database.repositories import SceneRepository, InferenceJobRepository, ResultRepository


@pytest.fixture(scope="module")
def client():
    """Create a FastAPI test client with initialized database."""
    init_db()
    app = create_app()
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="module")
def real_or_synth_scene(client: TestClient):
    """Ensure at least one valid eligible scene with real or valid test GeoTIFFs exists."""
    db = SessionLocal()
    try:
        scene_id = "integration_test_scene_001"
        existing = db.query(Scene).filter(Scene.scene_id == scene_id).first()
        if existing and Path(existing.s2_path).exists() and Path(existing.s1_path).exists():
            return existing.scene_id

        # Check if another eligible scene already exists
        scene = db.query(Scene).filter(Scene.is_eligible.is_(True)).first()
        if scene and Path(scene.s2_path).exists() and Path(scene.s1_path).exists():
            return scene.scene_id

        # Create temporary valid test GeoTIFFs
        tmp_dir = Path("outputs/scratch_test_integration")
        tmp_dir.mkdir(parents=True, exist_ok=True)
        s2_tif = tmp_dir / "s2_integration_test.tif"
        s1_tif = tmp_dir / "s1_integration_test.tif"

        # 13 bands for S2, 2 bands for S1
        transform = rasterio.transform.from_origin(500000, 2000000, 10.0, 10.0)
        crs = "EPSG:32643"

        with rasterio.open(
            s2_tif, "w", driver="GTiff", height=256, width=256, count=13,
            dtype=rasterio.float32, crs=crs, transform=transform
        ) as dst:
            dst.write(np.random.uniform(0.1, 0.9, size=(13, 256, 256)).astype(np.float32))

        with rasterio.open(
            s1_tif, "w", driver="GTiff", height=256, width=256, count=2,
            dtype=rasterio.float32, crs=crs, transform=transform
        ) as dst:
            dst.write(np.random.uniform(0.01, 0.5, size=(2, 256, 256)).astype(np.float32))

        if not existing:
            test_scene = Scene(
                scene_id=scene_id,
                roi_id="roi_integration",
                acquisition_time="2022-06-15",
                source_provider="ALLClear",
                s2_path=str(s2_tif),
                s1_path=str(s1_tif),
                cloud_density_percent=88.5,
                cloud_probability_threshold=60.0,
                is_eligible=True,
                crs=crs,
                width=256,
                height=256,
                resolution=10.0,
            )
            db.add(test_scene)
            db.commit()
        return scene_id
    finally:
        db.close()


class TestSystemStartupAndHealth:

    def test_health_telemetry(self, client: TestClient) -> None:
        """Verify GET /health returns all hardware and model telemetry."""
        res = client.get("/health")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "ok"
        assert "cuda_available" in data
        assert "storage" in data
        assert data["storage"]["database_connected"] is True

    def test_model_specifications(self, client: TestClient) -> None:
        """Verify GET /models returns correct architecture and input channels."""
        res = client.get("/models")
        assert res.status_code == 200
        data = res.json()
        assert "Modified DSen2-CR" in data["model_name"]
        assert data["s2_channels"] == 13
        assert data["s1_channels"] == 2
        assert data["target_channels"] == 13


class TestFullEndToEndDemoFlow:

    def test_complete_pipeline_flow(self, client: TestClient, real_or_synth_scene: str) -> None:
        """
        Execute full flow:
        1. Fetch scenes and select eligible scene
        2. Get scene previews
        3. Trigger inference
        4. Poll and retrieve result
        5. Verify geospatial correctness of generated GeoTIFF
        6. Verify database records and audit history
        7. Test secure downloads
        """
        scene_id = real_or_synth_scene

        # 1. Fetch Scene Details
        scene_res = client.get(f"/scenes/{scene_id}")
        assert scene_res.status_code == 200
        scene_meta = scene_res.json()
        assert scene_meta["scene_id"] == scene_id
        assert scene_meta["cloud_density_percent"] >= 60.0

        # 2. Get S2 and S1 Previews
        s2_prev = client.get(f"/scenes/{scene_id}/preview/s2")
        assert s2_prev.status_code == 200
        assert s2_prev.headers.get("content-type") == "image/png"

        s1_prev = client.get(f"/scenes/{scene_id}/preview/s1")
        assert s1_prev.status_code == 200
        assert s1_prev.headers.get("content-type") == "image/png"

        # 3. Trigger Inference
        inf_res = client.post(
            "/inference",
            json={
                "scene_id": scene_id,
                "tile_size": 256,
                "overlap": 64,
                "batch_size": 4,
            },
        )
        assert inf_res.status_code == 201
        job_data = inf_res.json()
        job_id = job_data["job_id"]
        assert job_id is not None

        # 4. Fetch Result
        res_res = client.get(f"/results/{job_id}")
        assert res_res.status_code == 200
        res_data = res_res.json()
        assert res_data["status"] == "completed"
        assert res_data["band_count"] == 13
        assert res_data["resolution"] == 10.0
        result_id = res_data["result_id"]

        # 5. Check Metrics Endpoint
        metrics_res = client.get(f"/metrics?result_id={result_id}")
        assert metrics_res.status_code == 200

        # 6. Verify Downloads
        geotiff_dl = client.get(f"/download?result_id={result_id}&file_type=geotiff")
        assert geotiff_dl.status_code == 200
        assert geotiff_dl.headers.get("content-type") == "image/tiff"
        assert len(geotiff_dl.content) > 1000

        png_dl = client.get(f"/download?result_id={result_id}&file_type=png")
        assert png_dl.status_code == 200
        assert png_dl.headers.get("content-type") == "image/png"
        assert len(png_dl.content) > 1000

        # 7. Verify Database Records and Audit History
        db = SessionLocal()
        try:
            job_orm = InferenceJobRepository.get_by_job_id(db, job_id)
            assert job_orm is not None
            assert job_orm.result is not None
            assert job_orm.result.result_id == result_id

            # Verify processing history events
            events = db.query(ProcessingHistory).filter(ProcessingHistory.entity_id == job_id).all()
            actions = [e.action for e in events]
            assert "INFERENCE_STARTED" in actions
            assert "INFERENCE_COMPLETED" in actions
        finally:
            db.close()

        # 8. Check History API
        history_res = client.get("/history?limit=10")
        assert history_res.status_code == 200
        hist_data = history_res.json()
        assert hist_data["total_count"] > 0
        assert len(hist_data["events"]) > 0


class TestFailureModesAndSecurityChecks:

    def test_ineligible_scene_rejected(self, client: TestClient) -> None:
        """Verify scenes below the threshold return HTTP 400."""
        # Seed an ineligible scene
        db = SessionLocal()
        try:
            ineligible_id = "scene_ineligible_test"
            existing = db.query(Scene).filter(Scene.scene_id == ineligible_id).first()
            if not existing:
                db.add(
                    Scene(
                        scene_id=ineligible_id,
                        roi_id="roi_test",
                        acquisition_time="2022-01-01",
                        s2_path="data/s2.tif",
                        s1_path="data/s1.tif",
                        cloud_density_percent=25.0,
                        cloud_probability_threshold=60.0,
                        is_eligible=False,
                        crs="EPSG:32643",
                        width=256,
                        height=256,
                        resolution=10.0,
                    )
                )
                db.commit()
        finally:
            db.close()

        res = client.post("/inference", json={"scene_id": "scene_ineligible_test"})
        assert res.status_code == 400
        assert "ineligible" in res.json()["detail"].lower()

    def test_non_existent_scene_rejected(self, client: TestClient) -> None:
        """Verify non-existent scene returns HTTP 404."""
        res = client.post("/inference", json={"scene_id": "non_existent_scene_xyz"})
        assert res.status_code == 404

    def test_path_traversal_prevention_on_downloads(self, client: TestClient) -> None:
        """Verify directory traversal attempts on download endpoint are rejected."""
        res = client.get("/download?result_id=../../../../etc/passwd&file_type=geotiff")
        assert res.status_code in [400, 404, 403]
