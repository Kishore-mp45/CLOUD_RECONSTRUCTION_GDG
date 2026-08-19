"""
tests/test_api.py
=================
Pytest test suite for Phase 8 FastAPI Backend endpoints.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

import pytest
from fastapi.testclient import TestClient

from api.main import app
from api.db.database import Base, engine, SessionLocal
from api.db.models import Scene, InferenceJob, Result, MetricRecord


@pytest.fixture(scope="module")
def client():
    """Create FastAPI TestClient and ensure database tables exist."""
    Base.metadata.create_all(bind=engine)
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="module", autouse=True)
def setup_test_scenes():
    """Ensure at least one eligible and one ineligible scene exists in the database for testing."""
    from api.db.database import init_db
    init_db()
    db = SessionLocal()
    try:
        # Eligible scene
        s_eligible = db.query(Scene).filter(Scene.scene_id == "test_scene_eligible").first()
        if not s_eligible:
            s_eligible = Scene(
                scene_id="test_scene_eligible",
                roi_id="roi_test_01",
                acquisition_time="2022-06-01",
                s2_path="data/test_s2.tif",
                s1_path="data/test_s1.tif",
                cloud_density_percent=85.0,
                cloud_probability_threshold=60.0,
                is_eligible=True,
                crs="EPSG:32643",
                width=256,
                height=256,
                resolution=10.0,
            )
            db.add(s_eligible)

        # Ineligible scene
        s_ineligible = db.query(Scene).filter(Scene.scene_id == "test_scene_ineligible").first()
        if not s_ineligible:
            s_ineligible = Scene(
                scene_id="test_scene_ineligible",
                roi_id="roi_test_02",
                acquisition_time="2022-06-02",
                s2_path="data/test_s2.tif",
                s1_path="data/test_s1.tif",
                cloud_density_percent=20.0,
                cloud_probability_threshold=60.0,
                is_eligible=False,
                crs="EPSG:32643",
                width=256,
                height=256,
                resolution=10.0,
            )
            db.add(s_ineligible)

        # Dummy result for download testing
        dummy_out_dir = Path("outputs/inference")
        dummy_out_dir.mkdir(parents=True, exist_ok=True)
        dummy_tif = dummy_out_dir / "test_dummy_reconstructed.tif"
        dummy_png = dummy_out_dir / "test_dummy_preview.png"

        if not dummy_tif.exists():
            dummy_tif.write_bytes(b"dummy_geotiff_bytes")
        if not dummy_png.exists():
            dummy_png.write_bytes(b"dummy_png_bytes")

        r_test = db.query(Result).filter(Result.result_id == "res_test_123").first()
        if not r_test:
            # Need matching job
            job = InferenceJob(
                job_id="job_test_123",
                scene_id="test_scene_eligible",
                status="completed",
                model_name="Modified DSen2-CR",
                checkpoint_name="best_model.pth",
            )
            db.add(job)
            db.commit()

            r_test = Result(
                result_id="res_test_123",
                job_id="job_test_123",
                scene_id="test_scene_eligible",
                geotiff_path=str(dummy_tif),
                preview_png_path=str(dummy_png),
                metadata_path=str(dummy_out_dir / "test_dummy_metadata.json"),
                crs="EPSG:32643",
                width=256,
                height=256,
                resolution=10.0,
            )
            db.add(r_test)

        db.commit()
    finally:
        db.close()


class TestHealthEndpoint:

    def test_get_health(self, client: TestClient) -> None:
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "version" in data
        assert "python_version" in data
        assert "torch_version" in data
        assert "cuda_available" in data
        assert "active_model" in data
        assert data["storage"]["database_connected"] is True


class TestScenesEndpoints:

    def test_get_scenes_list(self, client: TestClient) -> None:
        response = client.get("/scenes")
        assert response.status_code == 200
        data = response.json()
        assert "total_count" in data
        assert "scenes" in data
        assert data["total_count"] > 0

    def test_get_scenes_filter_eligible(self, client: TestClient) -> None:
        response = client.get("/scenes?eligible=true")
        assert response.status_code == 200
        data = response.json()
        for sc in data["scenes"]:
            assert sc["is_eligible"] is True

    def test_get_scene_by_id_valid(self, client: TestClient) -> None:
        response = client.get("/scenes/test_scene_eligible")
        assert response.status_code == 200
        data = response.json()
        assert data["scene_id"] == "test_scene_eligible"
        assert data["is_eligible"] is True
        assert data["cloud_density_percent"] == 85.0

    def test_get_scene_by_id_not_found(self, client: TestClient) -> None:
        response = client.get("/scenes/non_existent_scene_xyz")
        assert response.status_code == 404


class TestModelsEndpoint:

    def test_get_models(self, client: TestClient) -> None:
        response = client.get("/models")
        assert response.status_code == 200
        data = response.json()
        assert data["model_name"] == "Modified DSen2-CR (SAR-Supervised)"
        assert data["s2_channels"] == 13
        assert data["s1_channels"] == 2
        assert data["target_channels"] == 13
        assert data["checkpoint_name"] == "best_model.pth"


class TestMetricsEndpoint:

    def test_get_aggregate_metrics(self, client: TestClient) -> None:
        response = client.get("/metrics")
        assert response.status_code == 200
        data = response.json()
        assert "available" in data

    def test_get_metrics_for_live_result_without_gt(self, client: TestClient) -> None:
        response = client.get("/metrics?result_id=res_test_123")
        assert response.status_code == 200
        data = response.json()
        assert data["available"] is False
        assert "Ground-truth" in data["reason"]


class TestInferenceValidationAndRejection:

    def test_inference_rejects_ineligible_scene(self, client: TestClient) -> None:
        payload = {
            "scene_id": "test_scene_ineligible",
            "tile_size": 256,
            "overlap": 64,
            "batch_size": 4,
        }
        response = client.post("/inference", json=payload)
        assert response.status_code == 400
        assert "ineligible" in response.json()["detail"].lower()

    def test_inference_rejects_missing_scene(self, client: TestClient) -> None:
        payload = {
            "scene_id": "missing_scene_999",
            "tile_size": 256,
            "overlap": 64,
            "batch_size": 4,
        }
        response = client.post("/inference", json=payload)
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()


class TestResultsEndpoint:

    def test_get_result_by_id_valid(self, client: TestClient) -> None:
        response = client.get("/results/res_test_123")
        assert response.status_code == 200
        data = response.json()
        assert data["result_id"] == "res_test_123"
        assert "/download?result_id=res_test_123&file_type=geotiff" in data["geotiff_download_url"]

    def test_get_result_by_id_not_found(self, client: TestClient) -> None:
        response = client.get("/results/non_existent_result_000")
        assert response.status_code == 404


class TestDownloadEndpointAndSecurity:

    def test_download_geotiff_valid(self, client: TestClient) -> None:
        response = client.get("/download?result_id=res_test_123&file_type=geotiff")
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/tiff"
        assert response.content == b"dummy_geotiff_bytes"

    def test_download_png_valid(self, client: TestClient) -> None:
        response = client.get("/download?result_id=res_test_123&file_type=png")
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/png"
        assert response.content == b"dummy_png_bytes"

    def test_download_non_existent_result(self, client: TestClient) -> None:
        response = client.get("/download?result_id=fake_result_999&file_type=geotiff")
        assert response.status_code == 404


class TestInferenceExecutionIntegration:

    def test_full_inference_flow(self, client: TestClient) -> None:
        """Integration test executing full Phase 6 inference through the FastAPI endpoint."""
        s2_real = Path("allclear_dataset/roi502413/2022_10/s2_toa/roi502413_s2_toa_2022_10_19_median.tif")
        s1_real = Path("allclear_dataset/roi502413/2022_10/s1/roi502413_s1_2022_10_29_median.tif")

        if not s2_real.exists() or not s1_real.exists():
            pytest.skip("Real dataset sample not found for integration test.")

        # Seed real scene into db
        db = SessionLocal()
        try:
            real_scene = db.query(Scene).filter(Scene.scene_id == "real_scene_integration_test").first()
            if not real_scene:
                real_scene = Scene(
                    scene_id="real_scene_integration_test",
                    roi_id="roi502413",
                    acquisition_time="2022-10-19",
                    s2_path=str(s2_real),
                    s1_path=str(s1_real),
                    cloud_density_percent=92.0,
                    cloud_probability_threshold=60.0,
                    is_eligible=True,
                    crs="EPSG:32645",
                    width=309,
                    height=309,
                    resolution=10.0,
                )
                db.add(real_scene)
                db.commit()
        finally:
            db.close()

        # Trigger POST /inference
        req = {
            "scene_id": "real_scene_integration_test",
            "tile_size": 256,
            "overlap": 64,
            "batch_size": 4,
        }
        res_post = client.post("/inference", json=req)
        assert res_post.status_code == 201
        job_data = res_post.json()
        assert job_data["status"] == "completed"
        assert job_data["result_id"] is not None
        job_id = job_data["job_id"]

        # Fetch GET /results/{id}
        res_get = client.get(f"/results/{job_id}")
        assert res_get.status_code == 200
        result_data = res_get.json()
        assert result_data["crs"] == "EPSG:32645"
        assert result_data["band_count"] == 13

        # Test GeoTIFF download
        res_down_tif = client.get(f"/download?result_id={job_id}&file_type=geotiff")
        assert res_down_tif.status_code == 200
        assert res_down_tif.headers["content-type"] == "image/tiff"
        assert len(res_down_tif.content) > 1000

        # Test PNG preview download
        res_down_png = client.get(f"/download?result_id={job_id}&file_type=png")
        assert res_down_png.status_code == 200
        assert res_down_png.headers["content-type"] == "image/png"
        assert len(res_down_png.content) > 1000

