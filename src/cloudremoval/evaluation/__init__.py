"""
cloudremoval.evaluation
========================
Evaluation metrics, visual comparison, latency benchmarking, and reporting.

Public API:
    Phase5Evaluator            -> Full evaluation pipeline orchestrator
    evaluate_single_sample     -> Per-sample metrics dictionary
    compute_psnr               -> Peak Signal-to-Noise Ratio
    compute_ssim               -> Multispectral Structural Similarity
    compute_mae                -> Mean Absolute Error
    compute_rmse               -> Root Mean Squared Error
    compute_sam                -> Spectral Angle Mapper
    create_4panel_comparison   -> Visual comparison generator
    benchmark_inference        -> Latency & throughput benchmarker
    run_full_benchmark         -> Multi-batch latency suite
"""

from cloudremoval.evaluation.metrics import (
    evaluate_single_sample,
    compute_distribution_statistics,
    compute_psnr,
    compute_ssim,
    compute_mae,
    compute_rmse,
    compute_sam,
)
from cloudremoval.evaluation.visualizer import (
    create_4panel_comparison,
    to_rgb_numpy,
    RGB_INDICES,
)
from cloudremoval.evaluation.benchmarker import (
    benchmark_inference,
    run_full_benchmark,
)
from cloudremoval.evaluation.evaluator import Phase5Evaluator

__all__ = [
    "Phase5Evaluator",
    "evaluate_single_sample",
    "compute_distribution_statistics",
    "compute_psnr",
    "compute_ssim",
    "compute_mae",
    "compute_rmse",
    "compute_sam",
    "create_4panel_comparison",
    "to_rgb_numpy",
    "RGB_INDICES",
    "benchmark_inference",
    "run_full_benchmark",
]
