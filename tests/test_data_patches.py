"""
tests/test_data_patches.py
============================
Tests for patch extraction, quality filtering, and ROI sampling.
"""

from __future__ import annotations

import numpy as np
import pytest
from pathlib import Path
from datetime import datetime

from cloudremoval.data.patches import (
    PatchRecord,
    extract_patches_from_arrays,
    sample_patches_by_roi,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_arrays(h: int = 320, w: int = 320, fill_s2=1000.0, fill_s1=-15.0):
    s2  = np.full((13, h, w), fill_s2, dtype=np.float64)
    s1  = np.full((2,  h, w), fill_s1, dtype=np.float64)
    tgt = np.full((13, h, w), fill_s2, dtype=np.float64)
    return s2, s1, tgt


def _make_patch_record(roi_id: str, patch_id: str) -> PatchRecord:
    return PatchRecord(
        patch_id=patch_id,
        triplet_id="fake_triplet",
        scene_key="fake_key",
        roi_id=roi_id,
        s2_path=Path("/fake/s2.tif"),
        s1_path=Path("/fake/s1.tif"),
        target_path=Path("/fake/target.tif"),
        row_off=0, col_off=0,
        patch_h=256, patch_w=256,
        norm_version="v1",
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestExtractPatches:
    def test_returns_patches_for_valid_image(self) -> None:
        s2, s1, tgt = _make_arrays(320, 320)
        patches, n_rej = extract_patches_from_arrays(
            "fake_id", "fake_key", "roi1",
            Path("/s2.tif"), Path("/s1.tif"), Path("/tgt.tif"),
            s2, s1, tgt,
            patch_size=256, stride=128, max_nodata_frac=0.05,
        )
        assert len(patches) > 0
        assert n_rej == 0  # clean arrays

    def test_image_too_small_gives_no_patches(self) -> None:
        """If image is smaller than patch_size in both dims, no windows fit."""
        s2, s1, tgt = _make_arrays(200, 200)
        patches, _ = extract_patches_from_arrays(
            "fake_id", "fake_key", "roi1",
            Path("/s2.tif"), Path("/s1.tif"), Path("/tgt.tif"),
            s2, s1, tgt,
            patch_size=256, stride=128, max_nodata_frac=0.05,
        )
        assert len(patches) == 0

    def test_patch_ids_are_unique(self) -> None:
        s2, s1, tgt = _make_arrays(320, 320)
        patches, _ = extract_patches_from_arrays(
            "fake_id", "fake_key", "roi1",
            Path("/s2.tif"), Path("/s1.tif"), Path("/tgt.tif"),
            s2, s1, tgt,
            patch_size=256, stride=64,
        )
        ids = [p.patch_id for p in patches]
        assert len(ids) == len(set(ids))

    def test_patch_window_within_bounds(self) -> None:
        H, W = 320, 320
        s2, s1, tgt = _make_arrays(H, W)
        patches, _ = extract_patches_from_arrays(
            "fake_id", "fake_key", "roi1",
            Path("/s2.tif"), Path("/s1.tif"), Path("/tgt.tif"),
            s2, s1, tgt,
            patch_size=256, stride=128,
        )
        for p in patches:
            assert p.row_off + p.patch_h <= H
            assert p.col_off + p.patch_w <= W

    def test_nodata_patch_rejected(self) -> None:
        s2, s1, tgt = _make_arrays(320, 320)
        # Flood the entire S2 with NaN
        s2[:] = np.nan
        patches, n_rej = extract_patches_from_arrays(
            "fake_id", "fake_key", "roi1",
            Path("/s2.tif"), Path("/s1.tif"), Path("/tgt.tif"),
            s2, s1, tgt,
            patch_size=256, stride=128, max_nodata_frac=0.05,
        )
        assert len(patches) == 0
        assert n_rej > 0

    def test_patch_record_fields(self) -> None:
        s2, s1, tgt = _make_arrays(320, 320)
        patches, _ = extract_patches_from_arrays(
            "my_triplet", "my_key", "my_roi",
            Path("/s2.tif"), Path("/s1.tif"), Path("/tgt.tif"),
            s2, s1, tgt,
            patch_size=256, stride=128,
        )
        assert len(patches) > 0
        p = patches[0]
        assert p.roi_id == "my_roi"
        assert p.triplet_id == "my_triplet"
        assert p.patch_h == 256
        assert p.patch_w == 256
        assert p.norm_version == "v1"


class TestSamplePatchesByRoi:
    def test_no_limit_returns_all(self) -> None:
        patches = [_make_patch_record("roi1", f"p{i}") for i in range(20)]
        result = sample_patches_by_roi(patches, max_patches_per_roi=None, seed=42)
        assert len(result) == 20

    def test_zero_limit_returns_all(self) -> None:
        patches = [_make_patch_record("roi1", f"p{i}") for i in range(20)]
        result = sample_patches_by_roi(patches, max_patches_per_roi=0, seed=42)
        assert len(result) == 20

    def test_cap_applied_per_roi(self) -> None:
        patches = (
            [_make_patch_record("roi1", f"r1_p{i}") for i in range(30)]
            + [_make_patch_record("roi2", f"r2_p{i}") for i in range(30)]
        )
        result = sample_patches_by_roi(patches, max_patches_per_roi=10, seed=42)
        from collections import Counter
        counts = Counter(p.roi_id for p in result)
        assert counts["roi1"] <= 10
        assert counts["roi2"] <= 10

    def test_deterministic(self) -> None:
        patches = [_make_patch_record("roi1", f"p{i}") for i in range(50)]
        result1 = sample_patches_by_roi(patches, max_patches_per_roi=10, seed=42)
        result2 = sample_patches_by_roi(patches, max_patches_per_roi=10, seed=42)
        assert [p.patch_id for p in result1] == [p.patch_id for p in result2]
