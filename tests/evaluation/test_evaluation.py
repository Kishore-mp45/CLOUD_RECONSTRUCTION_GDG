"""
tests/evaluation/test_evaluation.py
====================================
Pytest tests for Phase 5 evaluation module.

Tests:
  - Metric calculations (PSNR, SSIM, MAE, RMSE, SAM)
  - Statistical aggregation
  - Visualizer RGB conversion and 4-panel generation
  - Latency benchmarker
"""

from __future__ import annotations

import tempfile
from pathlib import Path
import pytest
import numpy as np
import torch

from cloudremoval.models import DSen2CRConfig, build_model
from cloudremoval.evaluation.metrics import (
    compute_psnr,
    compute_ssim,
    compute_mae,
    compute_rmse,
    compute_sam,
    evaluate_single_sample,
    compute_distribution_statistics,
)
from cloudremoval.evaluation.visualizer import to_rgb_numpy, create_4panel_comparison
from cloudremoval.evaluation.benchmarker import benchmark_inference


# ---------------------------------------------------------------------------
# Metric Tests
# ---------------------------------------------------------------------------

class TestEvaluationMetrics:

    def test_psnr_identical(self) -> None:
        x = torch.randn(13, 64, 64)
        psnr = compute_psnr(x, x, data_range=10.0)
        assert psnr == float("inf")

    def test_psnr_noisy(self) -> None:
        x = torch.zeros(13, 64, 64)
        noisy = x + 0.1
        psnr = compute_psnr(noisy, x, data_range=10.0)
        assert 20.0 <= psnr <= 50.0

    def test_ssim_identical(self) -> None:
        x = torch.randn(13, 64, 64)
        ssim = compute_ssim(x, x, data_range=10.0)
        assert abs(ssim - 1.0) < 1e-4

    def test_ssim_different(self) -> None:
        x = torch.randn(13, 64, 64)
        y = torch.randn(13, 64, 64)
        ssim = compute_ssim(x, y, data_range=10.0)
        assert -1.0 <= ssim < 0.95

    def test_mae_rmse_values(self) -> None:
        x = torch.zeros(13, 32, 32)
        y = torch.ones(13, 32, 32) * 2.0
        mae = compute_mae(x, y)
        rmse = compute_rmse(x, y)
        assert abs(mae - 2.0) < 1e-5
        assert abs(rmse - 2.0) < 1e-5

    def test_sam_zero_angle(self) -> None:
        x = torch.abs(torch.randn(13, 32, 32)) + 0.1
        sam = compute_sam(x, x * 2.0)
        assert abs(sam - 0.0) < 1e-2

    def test_evaluate_single_sample(self) -> None:
        pred = torch.randn(13, 64, 64)
        target = torch.randn(13, 64, 64)
        res = evaluate_single_sample(pred, target)
        assert "psnr" in res
        assert "ssim" in res
        assert "mae" in res
        assert "rmse" in res
        assert "sam" in res
        assert all(isinstance(v, float) for v in res.values())

    def test_distribution_statistics(self) -> None:
        vals = [10.0, 20.0, 30.0, 40.0, 50.0]
        stats = compute_distribution_statistics(vals)
        assert stats["mean"] == 30.0
        assert stats["median"] == 30.0
        assert stats["min"] == 10.0
        assert stats["max"] == 50.0
        assert stats["count"] == 5


# ---------------------------------------------------------------------------
# Visualizer Tests
# ---------------------------------------------------------------------------

class TestVisualizer:

    def test_to_rgb_numpy_shape(self) -> None:
        t = torch.randn(13, 64, 64)
        rgb = to_rgb_numpy(t)
        assert rgb.shape == (64, 64, 3)
        assert 0.0 <= rgb.min() <= rgb.max() <= 1.0

    def test_create_4panel_comparison_file(self) -> None:
        c = torch.randn(13, 64, 64)
        r = torch.randn(13, 64, 64)
        t = torch.randn(13, 64, 64)
        metrics = {"psnr": 32.5, "ssim": 0.85, "mae": 0.12, "rmse": 0.18}

        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = Path(tmpdir) / "test_visual.png"
            create_4panel_comparison(
                cloudy_s2=c,
                reconstructed_s2=r,
                target_s2=t,
                sample_id="patch_001",
                roi_id="roi_001",
                metrics=metrics,
                output_path=out_file,
                title_suffix="TEST",
            )
            assert out_file.exists()
            assert out_file.stat().st_size > 1000  # Non-empty image file


# ---------------------------------------------------------------------------
# Benchmarker Tests
# ---------------------------------------------------------------------------

class TestBenchmarker:

    def test_benchmark_inference_cpu(self) -> None:
        cfg = DSen2CRConfig(base_features=32, num_res_blocks=2, device="cpu", amp_enabled=False)
        m = build_model(cfg).to("cpu")
        res = benchmark_inference(
            model=m,
            device="cpu",
            amp_enabled=False,
            warmup_iters=2,
            measured_iters=5,
            batch_size=1,
            patch_size=64,
        )
        assert res["batch_size"] == 1
        assert res["mean_latency_ms"] > 0.0
        assert "throughput_patches_per_sec" in res
