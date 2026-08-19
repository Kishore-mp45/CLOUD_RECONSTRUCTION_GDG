"""
src/cloudremoval/evaluation/benchmarker.py
===========================================
Inference latency and hardware benchmarking for Phase 5.

Measures strictly model forward pass latency on the GPU:
  - Excludes disk I/O, GeoTIFF parsing, and metric calculation
  - Uses torch.cuda.synchronize() before and after timing
  - Runs warmup iterations to initialize CUDA execution contexts
  - Evaluates both single-sample (batch=1) and batch (batch=4) latency
  - Tracks peak VRAM usage during inference
"""

from __future__ import annotations

import time
import json
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import torch
import torch.nn as nn

from cloudremoval.models.model_config import S2_CHANNELS, S1_CHANNELS, PATCH_SIZE


def benchmark_inference(
    model: nn.Module,
    device: str = "cuda",
    amp_enabled: bool = True,
    warmup_iters: int = 10,
    measured_iters: int = 100,
    batch_size: int = 1,
    patch_size: int = PATCH_SIZE,
) -> Dict[str, Any]:
    """Benchmark raw model inference latency.

    Parameters
    ----------
    model : nn.Module
        Loaded DSen2CR model.
    device : str
        Target device ('cuda' or 'cpu').
    amp_enabled : bool
        Whether AMP is enabled.
    warmup_iters : int
        Number of untimed warmup passes.
    measured_iters : int
        Number of timed repetitions.
    batch_size : int
        Batch size to test.
    patch_size : int
        Spatial dimensions (default 256).

    Returns
    -------
    Dict[str, Any]
        Dictionary with timing metrics and memory stats.
    """
    model.eval()
    dev = torch.device(device)
    is_cuda = dev.type == "cuda" and torch.cuda.is_available()

    # Create synthetic test tensors on device
    s2 = torch.randn(batch_size, S2_CHANNELS, patch_size, patch_size, device=dev, dtype=torch.float32)
    s1 = torch.randn(batch_size, S1_CHANNELS, patch_size, patch_size, device=dev, dtype=torch.float32)

    # Reset CUDA memory stats
    if is_cuda:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(0)

    # 1. Warmup passes
    with torch.no_grad():
        for _ in range(warmup_iters):
            if is_cuda and amp_enabled:
                with torch.amp.autocast(device_type="cuda"):
                    _ = model(s2, s1)
            else:
                _ = model(s2, s1)
        if is_cuda:
            torch.cuda.synchronize()

    # 2. Timed iterations
    latencies_ms = []
    with torch.no_grad():
        for _ in range(measured_iters):
            if is_cuda:
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            if is_cuda and amp_enabled:
                with torch.amp.autocast(device_type="cuda"):
                    _ = model(s2, s1)
            else:
                _ = model(s2, s1)

            if is_cuda:
                torch.cuda.synchronize()
            t1 = time.perf_counter()

            latencies_ms.append((t1 - t0) * 1000.0)

    latencies_arr = np.array(latencies_ms, dtype=np.float64)

    # 3. GPU Memory
    gpu_name = torch.cuda.get_device_name(0) if is_cuda else "CPU"
    total_vram_gb = (torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)) if is_cuda else 0.0
    peak_vram_gb = (torch.cuda.max_memory_allocated(0) / (1024 ** 3)) if is_cuda else 0.0
    reserved_vram_gb = (torch.cuda.memory_reserved(0) / (1024 ** 3)) if is_cuda else 0.0

    mean_lat = float(np.mean(latencies_arr))
    median_lat = float(np.median(latencies_arr))
    std_lat = float(np.std(latencies_arr))
    min_lat = float(np.min(latencies_arr))
    max_lat = float(np.max(latencies_arr))
    p95_lat = float(np.percentile(latencies_arr, 95))
    throughput = float((batch_size / (mean_lat / 1000.0))) if mean_lat > 0 else 0.0

    return {
        "device": device,
        "gpu_name": gpu_name,
        "total_vram_gb": round(total_vram_gb, 2),
        "peak_vram_allocated_gb": round(peak_vram_gb, 4),
        "peak_vram_reserved_gb": round(reserved_vram_gb, 4),
        "amp_enabled": amp_enabled,
        "batch_size": batch_size,
        "patch_size": f"{patch_size}x{patch_size}",
        "warmup_iterations": warmup_iters,
        "measured_iterations": measured_iters,
        "mean_latency_ms": round(mean_lat, 3),
        "median_latency_ms": round(median_lat, 3),
        "std_latency_ms": round(std_lat, 3),
        "min_latency_ms": round(min_lat, 3),
        "max_latency_ms": round(max_lat, 3),
        "p95_latency_ms": round(p95_lat, 3),
        "throughput_patches_per_sec": round(throughput, 2),
    }


def run_full_benchmark(
    model: nn.Module,
    device: str = "cuda",
    amp_enabled: bool = True,
    output_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run both single-sample (batch=1) and batch (batch=4) benchmarks."""
    single_res = benchmark_inference(
        model=model,
        device=device,
        amp_enabled=amp_enabled,
        batch_size=1,
        warmup_iters=10,
        measured_iters=100,
    )
    batch_res = benchmark_inference(
        model=model,
        device=device,
        amp_enabled=amp_enabled,
        batch_size=4,
        warmup_iters=10,
        measured_iters=100,
    )

    combined = {
        "single_sample_batch_1": single_res,
        "batch_size_4": batch_res,
    }

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(combined, f, indent=2)

    return combined
