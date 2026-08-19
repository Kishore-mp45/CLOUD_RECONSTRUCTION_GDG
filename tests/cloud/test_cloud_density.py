"""
tests/cloud/test_cloud_density.py
==================================
Pytest test suite for Phase 7 cloud-density analysis & filtering layer.
"""

from __future__ import annotations

import pytest
import numpy as np

from cloudremoval.cloud import (
    calculate_cloud_density,
    validate_cloud_probability,
    analyze_scene,
    filter_scenes_batch,
    CloudFilterConfig,
    CloudDataValidationError,
    InvalidROIDataError,
    SceneMetadata,
)


class TestCloudProbabilityValidation:

    def test_valid_percentage_array(self) -> None:
        arr = np.array([[10.0, 50.0], [80.0, 95.0]], dtype=np.float32)
        val = validate_cloud_probability(arr)
        assert val.shape == (2, 2)
        assert np.isclose(val.max(), 95.0)

    def test_rescaling_fractional_array(self) -> None:
        arr = np.array([[0.1, 0.5], [0.8, 0.95]], dtype=np.float32)
        val = validate_cloud_probability(arr)
        assert np.isclose(val.max(), 95.0)
        assert np.isclose(val.min(), 10.0)

    def test_negative_values_raise(self) -> None:
        arr = np.array([[-5.0, 50.0], [80.0, 95.0]], dtype=np.float32)
        with pytest.raises(CloudDataValidationError):
            validate_cloud_probability(arr)

    def test_excessive_values_raise(self) -> None:
        arr = np.array([[10.0, 150.0], [80.0, 95.0]], dtype=np.float32)
        with pytest.raises(CloudDataValidationError):
            validate_cloud_probability(arr)

    def test_all_nans_raise(self) -> None:
        arr = np.full((10, 10), np.nan, dtype=np.float32)
        with pytest.raises(CloudDataValidationError):
            validate_cloud_probability(arr)

    def test_none_or_empty_raise(self) -> None:
        with pytest.raises(CloudDataValidationError):
            validate_cloud_probability(None)
        with pytest.raises(CloudDataValidationError):
            validate_cloud_probability(np.array([]))


class TestCloudDensityCalculation:

    def test_exact_50_percent_cloud_density(self) -> None:
        # 100 pixels: 50 with prob 80% (>=60%), 50 with prob 20% (<60%)
        arr = np.zeros((10, 10), dtype=np.float32)
        arr[:5, :] = 80.0
        arr[5:, :] = 20.0

        res = calculate_cloud_density(arr, pixel_threshold=60.0)
        assert res["total_pixels"] == 100
        assert res["valid_pixels"] == 100
        assert res["cloudy_pixels"] == 50
        assert res["cloud_density_percent"] == 50.0

    def test_100_percent_cloudy(self) -> None:
        arr = np.full((10, 10), 90.0, dtype=np.float32)
        res = calculate_cloud_density(arr, pixel_threshold=60.0)
        assert res["cloud_density_percent"] == 100.0

    def test_0_percent_cloudy(self) -> None:
        arr = np.full((10, 10), 10.0, dtype=np.float32)
        res = calculate_cloud_density(arr, pixel_threshold=60.0)
        assert res["cloud_density_percent"] == 0.0

    def test_roi_and_valid_masking(self) -> None:
        # 10x10 array
        arr = np.zeros((10, 10), dtype=np.float32)
        arr[:5, :] = 80.0  # top half cloudy (50 px)
        arr[5:, :] = 10.0  # bottom half clear (50 px)

        # ROI selects top 5 rows (50 px total)
        roi_mask = np.zeros((10, 10), dtype=bool)
        roi_mask[:5, :] = True

        # Valid mask excludes first column (10 px total, 5 in ROI)
        valid_mask = np.ones((10, 10), dtype=bool)
        valid_mask[:, 0] = False

        res = calculate_cloud_density(arr, roi_mask=roi_mask, valid_mask=valid_mask, pixel_threshold=60.0)
        assert res["roi_pixels"] == 50
        assert res["valid_pixels"] == 45
        assert res["cloudy_pixels"] == 45
        assert res["cloud_density_percent"] == 100.0

    def test_empty_roi_raises(self) -> None:
        arr = np.full((10, 10), 50.0, dtype=np.float32)
        empty_roi = np.zeros((10, 10), dtype=bool)
        with pytest.raises(InvalidROIDataError):
            calculate_cloud_density(arr, roi_mask=empty_roi)


class TestSceneAnalysisAndBatchFiltering:

    def test_analyze_scene_eligible(self) -> None:
        arr = np.full((20, 20), 85.0, dtype=np.float32)
        cfg = CloudFilterConfig(pixel_probability_threshold=60.0, scene_density_threshold=70.0)
        meta = SceneMetadata(scene_id="scene_001", acquisition_time="2022-05-01T00:00:00Z")

        res = analyze_scene("scene_001", arr, config=cfg, metadata=meta, verbose=False)
        assert res.passes_threshold is True
        assert res.status == "eligible"
        assert res.cloud_density_percent == 100.0

    def test_analyze_scene_filtered(self) -> None:
        arr = np.full((20, 20), 40.0, dtype=np.float32)
        cfg = CloudFilterConfig(pixel_probability_threshold=60.0, scene_density_threshold=50.0)

        res = analyze_scene("scene_002", arr, config=cfg, verbose=False)
        assert res.passes_threshold is False
        assert res.status == "filtered"
        assert res.cloud_density_percent == 0.0

    def test_analyze_scene_error_handling(self) -> None:
        arr = np.full((10, 10), np.nan, dtype=np.float32)
        res = analyze_scene("scene_corrupt", arr, verbose=False)
        assert res.status == "error"
        assert res.passes_threshold is False
        assert res.error_message is not None

    def test_filter_scenes_batch_deterministic_sorting(self) -> None:
        # Scene 1: 30% cloud
        arr1 = np.zeros((10, 10), dtype=np.float32)
        arr1[:3, :] = 90.0

        # Scene 2: 90% cloud
        arr2 = np.zeros((10, 10), dtype=np.float32)
        arr2[:9, :] = 90.0

        # Scene 3: 70% cloud
        arr3 = np.zeros((10, 10), dtype=np.float32)
        arr3[:7, :] = 90.0

        scenes = [
            {"scene_id": "S1", "cloud_prob": arr1, "metadata": {"scene_id": "S1", "acquisition_time": "2022-01-01"}},
            {"scene_id": "S2", "cloud_prob": arr2, "metadata": {"scene_id": "S2", "acquisition_time": "2022-02-01"}},
            {"scene_id": "S3", "cloud_prob": arr3, "metadata": {"scene_id": "S3", "acquisition_time": "2022-03-01"}},
        ]

        cfg = CloudFilterConfig(pixel_probability_threshold=60.0, scene_density_threshold=50.0)

        # Filter with sorting by cloud_density_desc
        batch = filter_scenes_batch(scenes, config=cfg, sort_by="cloud_density_desc", verbose=False)
        assert batch.total_scenes == 3
        assert batch.eligible_count == 2
        assert batch.filtered_count == 1
        assert batch.error_count == 0

        # Verify ordering: S2 (90%) then S3 (70%)
        assert batch.eligible_scenes[0].scene_id == "S2"
        assert batch.eligible_scenes[1].scene_id == "S3"
        assert batch.filtered_scenes[0].scene_id == "S1"
