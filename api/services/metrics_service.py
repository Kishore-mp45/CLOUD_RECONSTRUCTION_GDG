"""
api/services/metrics_service.py
===============================
Service for retrieving evaluation metrics and ground-truth comparisons.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any

from sqlalchemy.orm import Session

from api.db.models import MetricRecord, Result
from api.schemas.metrics import MetricsResponse

log = logging.getLogger(__name__)


def get_metrics_for_result(
    db: Session,
    result_id: Optional[str] = None,
) -> MetricsResponse:
    """Retrieve quantitative metrics for a result, or return aggregate Phase 5 benchmarks."""
    eval_metrics_path = Path("outputs/evaluation/metrics.json")
    aggregate_metrics: Optional[Dict[str, Any]] = None

    if eval_metrics_path.exists():
        try:
            with open(eval_metrics_path, "r", encoding="utf-8") as f:
                aggregate_metrics = json.load(f)
        except Exception as exc:
            log.warning("Could not read aggregate metrics from %s: %s", eval_metrics_path, exc)

    if result_id is not None:
        rec = db.query(MetricRecord).filter(MetricRecord.result_id == result_id).first()
        if rec and rec.is_ground_truth_available:
            return MetricsResponse(
                available=True,
                result_id=result_id,
                psnr=rec.psnr,
                ssim=rec.ssim,
                mae=rec.mae,
                rmse=rec.rmse,
                sam=rec.sam,
                latency_ms=rec.latency_ms,
                aggregate_test_metrics=aggregate_metrics,
            )
        else:
            # Check if result exists
            res = db.query(Result).filter(Result.result_id == result_id).first()
            if res:
                return MetricsResponse(
                    available=False,
                    reason="Ground-truth clear-sky Sentinel-2 target is not available for this live inference result.",
                    result_id=result_id,
                    aggregate_test_metrics=aggregate_metrics,
                )
            else:
                return MetricsResponse(
                    available=False,
                    reason=f"Result ID '{result_id}' not found.",
                    result_id=result_id,
                    aggregate_test_metrics=aggregate_metrics,
                )

    # General / aggregate metrics from Phase 5
    if aggregate_metrics:
        psnr_mean = aggregate_metrics.get("psnr", {}).get("mean")
        ssim_mean = aggregate_metrics.get("ssim", {}).get("mean")
        mae_mean = aggregate_metrics.get("mae", {}).get("mean")
        rmse_mean = aggregate_metrics.get("rmse", {}).get("mean")
        sam_mean = aggregate_metrics.get("sam", {}).get("mean")

        return MetricsResponse(
            available=True,
            reason="Aggregate test dataset evaluation benchmarks (Phase 5).",
            psnr=psnr_mean,
            ssim=ssim_mean,
            mae=mae_mean,
            rmse=rmse_mean,
            sam=sam_mean,
            aggregate_test_metrics=aggregate_metrics,
        )

    return MetricsResponse(
        available=False,
        reason="Evaluation metrics not found. Run Phase 5 evaluation to generate benchmark metrics.",
    )
