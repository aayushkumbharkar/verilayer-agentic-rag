"""
VeriLayer — Phase 7: Metrics API route.
GET /metrics — returns computed evaluation metrics from audit_logs.
"""
from __future__ import annotations

import structlog
from fastapi import APIRouter, HTTPException

from src.evaluation.evaluator import compute_metrics
from src.models.schemas import MetricsResponse

logger = structlog.get_logger("verilayer.api.metrics")
router = APIRouter(tags=["Evaluation"])


@router.get(
    "/metrics",
    response_model=MetricsResponse,
    summary="Pipeline evaluation metrics",
)
async def get_metrics() -> MetricsResponse:
    """
    Compute and return pipeline health metrics from the audit_logs table.

    Metrics include:
    - hallucination_rate: fraction of claims marked unsupported
    - avg_confidence: mean confidence score across all runs
    - retry_rate: fraction of queries needing >= 1 retry
    - avg_latency_ms: mean total pipeline latency
    - verified_rate / partial_rate / unsafe_rate: status distribution
    """
    try:
        return await compute_metrics()
    except Exception as exc:
        logger.error("metrics_computation_failed", error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc)) from exc
