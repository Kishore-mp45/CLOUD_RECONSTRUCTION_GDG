"""
tests/test_frontend.py
======================
Unit and integration tests for Phase 10 Frontend static serving and preview endpoints.
"""

from __future__ import annotations

from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from api.main import create_app
from api.db.database import init_db, SessionLocal
from cloudremoval.database.models import Scene, Result, InferenceJob


@pytest.fixture(scope="module")
def client():
    """Create a FastAPI test client with initialized database."""
    init_db()
    app = create_app()
    with TestClient(app) as test_client:
        yield test_client


class TestFrontendStaticServing:

    def test_get_index_html(self, client: TestClient) -> None:
        """Verify index.html is served cleanly at root /."""
        response = client.get("/")
        assert response.status_code == 200
        assert "ALLClear" in response.text
        assert "Cloud Scene Selector" in response.text
        assert "Split Slider" in response.text

    def test_get_css_assets(self, client: TestClient) -> None:
        """Verify CSS assets load with correct content type."""
        for css in ["styles.css", "dashboard.css", "components.css"]:
            res = client.get(f"/css/{css}")
            assert res.status_code == 200
            assert "text/css" in res.headers.get("content-type", "")

    def test_get_js_modules(self, client: TestClient) -> None:
        """Verify all JS modules load cleanly."""
        for js in ["app.js", "api.js", "scenes.js", "viewer.js", "inference.js", "metrics.js", "ui.js"]:
            res = client.get(f"/js/{js}")
            assert res.status_code == 200


class TestImageryPreviewEndpoints:

    def test_get_scene_preview_s2(self, client: TestClient) -> None:
        """Verify /scenes/{id}/preview/s2 returns a valid PNG preview image."""
        # Get a scene from the API
        scenes_res = client.get("/scenes?limit=1")
        assert scenes_res.status_code == 200
        scenes_data = scenes_res.json().get("scenes", [])
        if not scenes_data:
            pytest.skip("No scenes available to test preview.")

        scene_id = scenes_data[0]["scene_id"]
        res = client.get(f"/scenes/{scene_id}/preview/s2")
        assert res.status_code == 200
        assert res.headers.get("content-type") == "image/png"
        assert len(res.content) > 100

    def test_get_scene_preview_s1(self, client: TestClient) -> None:
        """Verify /scenes/{id}/preview/s1 returns a valid SAR PNG preview image."""
        scenes_res = client.get("/scenes?limit=1")
        assert scenes_res.status_code == 200
        scenes_data = scenes_res.json().get("scenes", [])
        if not scenes_data:
            pytest.skip("No scenes available to test SAR preview.")

        scene_id = scenes_data[0]["scene_id"]
        res = client.get(f"/scenes/{scene_id}/preview/s1")
        assert res.status_code == 200
        assert res.headers.get("content-type") == "image/png"
        assert len(res.content) > 100
