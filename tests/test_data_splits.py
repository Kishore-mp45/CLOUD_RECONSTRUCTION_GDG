"""
tests/test_data_splits.py
===========================
Tests for ROI-level scene splitting and leakage verification.
"""

from __future__ import annotations

import pytest
from datetime import datetime
from pathlib import Path

from cloudremoval.data.splits import scene_level_split, verify_no_leakage
from cloudremoval.data.triplets import TripletRecord


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_triplets(roi_ids: list[str], obs_per_roi: int = 3) -> list[TripletRecord]:
    """Create fake TripletRecords for given ROI IDs."""
    triplets = []
    for roi in roi_ids:
        for i in range(obs_per_roi):
            triplets.append(TripletRecord(
                triplet_id=f"{roi}__s2obs{i}",
                scene_key=f"{roi}_key",
                roi_id=roi,
                roi_coords=[0.0, 0.0],
                s2_path=Path(f"/fake/{roi}/s2_{i}.tif"),
                s2_date=datetime(2022, 1, 1),
                s2_obs_index=i,
                s1_path=Path(f"/fake/{roi}/s1.tif"),
                s1_date=datetime(2022, 1, 2),
                target_path=Path(f"/fake/{roi}/target.tif"),
                target_date=datetime(2022, 1, 3),
                s2_to_target_days=2.0,
                s1_to_target_days=1.0,
            ))
    return triplets


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSceneLevelSplit:
    def test_all_triplets_present_in_splits(self) -> None:
        rois = [f"roi{i:05d}" for i in range(100)]
        triplets = _make_triplets(rois, obs_per_roi=3)
        train, val, test = scene_level_split(triplets, seed=42)
        assert len(train) + len(val) + len(test) == len(triplets)

    def test_approximate_ratios(self) -> None:
        rois = [f"roi{i:05d}" for i in range(200)]
        triplets = _make_triplets(rois, obs_per_roi=1)
        train, val, test = scene_level_split(triplets, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42)
        n = len(triplets)
        assert abs(len(train) / n - 0.8) < 0.05
        assert abs(len(val) / n - 0.1) < 0.05
        assert abs(len(test) / n - 0.1) < 0.05

    def test_deterministic_with_same_seed(self) -> None:
        rois = [f"roi{i:05d}" for i in range(50)]
        triplets = _make_triplets(rois)
        train1, val1, test1 = scene_level_split(triplets, seed=42)
        train2, val2, test2 = scene_level_split(triplets, seed=42)
        assert [t.triplet_id for t in train1] == [t.triplet_id for t in train2]
        assert [t.triplet_id for t in val1]   == [t.triplet_id for t in val2]
        assert [t.triplet_id for t in test1]  == [t.triplet_id for t in test2]

    def test_different_seeds_give_different_splits(self) -> None:
        rois = [f"roi{i:05d}" for i in range(50)]
        triplets = _make_triplets(rois)
        train1, _, _ = scene_level_split(triplets, seed=42)
        train2, _, _ = scene_level_split(triplets, seed=99)
        # Very unlikely to be identical with 50 ROIs
        assert set(t.roi_id for t in train1) != set(t.roi_id for t in train2)

    def test_invalid_ratio_raises(self) -> None:
        rois = [f"roi{i}" for i in range(10)]
        triplets = _make_triplets(rois)
        with pytest.raises(ValueError):
            scene_level_split(triplets, train_ratio=0.8, val_ratio=0.1, test_ratio=0.2)

    def test_no_roi_appears_in_multiple_splits(self) -> None:
        rois = [f"roi{i:05d}" for i in range(100)]
        triplets = _make_triplets(rois)
        train, val, test = scene_level_split(triplets, seed=42)
        train_rois = {t.roi_id for t in train}
        val_rois   = {t.roi_id for t in val}
        test_rois  = {t.roi_id for t in test}
        assert train_rois & val_rois == set()
        assert train_rois & test_rois == set()
        assert val_rois & test_rois == set()


class TestVerifyNoLeakage:
    def test_passes_with_clean_splits(self) -> None:
        rois = [f"roi{i:05d}" for i in range(30)]
        triplets = _make_triplets(rois)
        train, val, test = scene_level_split(triplets, seed=42)
        assert verify_no_leakage(train, val, test) is True

    def test_raises_on_train_val_overlap(self) -> None:
        rois_a = [f"roi{i:05d}" for i in range(10)]
        rois_b = [f"roi{i:05d}" for i in range(10)]  # same ROIs!
        train = _make_triplets(rois_a, obs_per_roi=1)
        val   = _make_triplets(rois_b, obs_per_roi=1)
        test  = _make_triplets([f"roi{i+20:05d}" for i in range(5)], obs_per_roi=1)
        with pytest.raises(AssertionError, match="LEAKAGE"):
            verify_no_leakage(train, val, test)
