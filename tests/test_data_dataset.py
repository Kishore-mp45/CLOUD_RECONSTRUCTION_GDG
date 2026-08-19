"""
tests/test_data_dataset.py
============================
Tests for AllClearDataset (PyTorch Dataset) using synthetic GeoTIFFs.

No real dataset files are read.
"""

from __future__ import annotations

import numpy as np
import pytest
import rasterio
import torch
from pathlib import Path
from rasterio.transform import from_bounds
from datetime import datetime

from cloudremoval.data.patches import PatchRecord
from cloudremoval.data.dataset import AllClearDataset
from cloudremoval.data.loaders import S2_EXPECTED_BANDS, S1_EXPECTED_BANDS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_tiff(path: Path, n_bands: int, h: int, w: int, fill=1000.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    transform = from_bounds(0, 0, 1, 1, w, h)
    with rasterio.open(
        path, "w", driver="GTiff",
        height=h, width=w, count=n_bands, dtype="float64",
        crs="EPSG:32755", transform=transform,
    ) as dst:
        dst.write(np.full((n_bands, h, w), fill, dtype="float64"))


def _fake_norm_stats() -> dict:
    return {
        "s2": {
            "mean": [1000.0] * S2_EXPECTED_BANDS,
            "std":  [500.0]  * S2_EXPECTED_BANDS,
        },
        "s1": {
            "mean": [-15.0, -20.0],
            "std":  [3.0,    3.0],
        },
        "z_clamp": 5.0,
    }


@pytest.fixture()
def patch_dataset(tmp_path: Path):
    """Create 3 synthetic triplet GeoTIFFs and return an AllClearDataset."""
    H, W = 320, 320

    patches = []
    for i in range(3):
        s2_p  = tmp_path / f"s2_{i}.tif"
        s1_p  = tmp_path / f"s1_{i}.tif"
        tgt_p = tmp_path / f"tgt_{i}.tif"
        _write_tiff(s2_p,  S2_EXPECTED_BANDS, H, W, fill=1500.0)
        _write_tiff(s1_p,  S1_EXPECTED_BANDS, H, W, fill=-14.0)
        _write_tiff(tgt_p, S2_EXPECTED_BANDS, H, W, fill=1200.0)

        patches.append(PatchRecord(
            patch_id=f"patch_{i}",
            triplet_id=f"trip_{i}",
            scene_key=f"key_{i}",
            roi_id=f"roi{i}",
            s2_path=s2_p,
            s1_path=s1_p,
            target_path=tgt_p,
            row_off=0, col_off=0,
            patch_h=256, patch_w=256,
            norm_version="v1",
        ))

    return AllClearDataset(patches, _fake_norm_stats(), augment=False)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestAllClearDataset:
    def test_len(self, patch_dataset) -> None:
        assert len(patch_dataset) == 3

    def test_getitem_returns_dict(self, patch_dataset) -> None:
        item = patch_dataset[0]
        assert isinstance(item, dict)
        assert set(item.keys()) >= {"s2", "s1", "target", "meta"}

    def test_s2_tensor_shape(self, patch_dataset) -> None:
        item = patch_dataset[0]
        assert item["s2"].shape == (S2_EXPECTED_BANDS, 256, 256)

    def test_s1_tensor_shape(self, patch_dataset) -> None:
        item = patch_dataset[0]
        assert item["s1"].shape == (S1_EXPECTED_BANDS, 256, 256)

    def test_target_tensor_shape(self, patch_dataset) -> None:
        item = patch_dataset[0]
        assert item["target"].shape == (S2_EXPECTED_BANDS, 256, 256)

    def test_tensors_are_float32(self, patch_dataset) -> None:
        item = patch_dataset[0]
        assert item["s2"].dtype == torch.float32
        assert item["s1"].dtype == torch.float32
        assert item["target"].dtype == torch.float32

    def test_no_nan_in_output(self, patch_dataset) -> None:
        item = patch_dataset[0]
        assert not torch.isnan(item["s2"]).any()
        assert not torch.isnan(item["s1"]).any()
        assert not torch.isnan(item["target"]).any()

    def test_values_within_clamp_range(self, patch_dataset) -> None:
        item = patch_dataset[0]
        assert item["s2"].abs().max().item() <= 5.0 + 1e-5  # Z_CLAMP
        assert item["s1"].abs().max().item() <= 5.0 + 1e-5

    def test_meta_contains_patch_id(self, patch_dataset) -> None:
        item = patch_dataset[0]
        assert "patch_id" in item["meta"]
        assert item["meta"]["patch_id"] == "patch_0"

    def test_all_items_accessible(self, patch_dataset) -> None:
        for i in range(len(patch_dataset)):
            item = patch_dataset[i]
            assert "s2" in item
