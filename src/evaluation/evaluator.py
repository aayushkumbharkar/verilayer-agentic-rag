"""
VeriLayer — Phase 7: Evaluation layer.
Reads from PostgreSQL audit_logs and computes pipeline health metrics.

Metrics computed:
  - hallucination_rate  = unsupported claims / total claims
  - avg_confidence      = mean claim confidence across all runs
  - retry_rate          = queries with >= 1 retry / total queries
  - avg_latency_ms      = mean total pipeline latency
  - verified_rate       = queries with status=verified / total
  - partial_rate        = queries with status=partial / total
  - unsafe_rate         = queries with status=unsafe / total
"""
from __future__ import annotations

import json
import structlog
from datetime import datetime

import asyncpg

from src.core.config import settings
from src.models.schemas import MetricsResponse

logger = structlog.get_logger("verilayer.evaluation.evaluator")


async def _get_conn() -> asyncpg.Connection:
    return await asyncpg.connect(
        host=settings.postgres_host,
        port=settings.postgres_port,
        database=settings.postgres_db,
        user=settings.postgres_user,
        password=settings.postgres_password,
    )


async def compute_metrics() -> MetricsResponse:
    """
    Fetch all audit_logs rows and compute evaluation metrics.
    Returns MetricsResponse with zero-safe defaults if no data.
    """
    conn = await _get_conn()
    try:
        rows = await conn.fetch(
            """
            SELECT status, confidence, claims_json, retries, latency_ms
            FROM audit_logs
            ORDER BY created_at DESC
            LIMIT 10000
            """
        )
    finally:
        await conn.close()

    total = len(rows)
    if total == 0:
        logger.info("no_audit_data_yet")
        return MetricsResponse(
            total_queries=0,
            hallucination_rate=0.0,
            avg_confidence=0.0,
            retry_rate=0.0,
            avg_latency_ms=0.0,
            verified_rate=0.0,
            partial_rate=0.0,
            unsafe_rate=0.0,
            computed_at=datetime.utcnow(),
        )

    # Aggregate counters
    total_claims = 0
    unsupported_claims = 0
    confidence_sum = 0.0
    retry_count = 0
    latency_sum = 0.0
    status_counts: dict[str, int] = {"verified": 0, "partial": 0, "unsafe": 0}

    for row in rows:
        status = row["status"] or "unsafe"
        status_counts[status] = status_counts.get(status, 0) + 1

        confidence_sum += float(row["confidence"] or 0.0)

        retries = int(row["retries"] or 0)
        if retries >= 1:
            retry_count += 1

        latency_sum += float(row["latency_ms"] or 0)

        # Parse claims JSON to count hallucinations
        raw_claims = row["claims_json"]
        if raw_claims:
            try:
                claims = json.loads(raw_claims) if isinstance(raw_claims, str) else raw_claims
                if isinstance(claims, list):
                    for c in claims:
                        total_claims += 1
                        if isinstance(c, dict) and c.get("verdict") == "unsupported":
                            unsupported_claims += 1
            except (json.JSONDecodeError, TypeError):
                pass

    hallucination_rate = unsupported_claims / total_claims if total_claims > 0 else 0.0
    avg_confidence = confidence_sum / total
    retry_rate = retry_count / total
    avg_latency_ms = latency_sum / total
    verified_rate = status_counts.get("verified", 0) / total
    partial_rate = status_counts.get("partial", 0) / total
    unsafe_rate = status_counts.get("unsafe", 0) / total

    logger.info(
        "metrics_computed",
        total_queries=total,
        hallucination_rate=round(hallucination_rate, 4),
        avg_confidence=round(avg_confidence, 4),
        retry_rate=round(retry_rate, 4),
    )

    return MetricsResponse(
        total_queries=total,
        hallucination_rate=round(hallucination_rate, 4),
        avg_confidence=round(avg_confidence, 4),
        retry_rate=round(retry_rate, 4),
        avg_latency_ms=round(avg_latency_ms, 2),
        verified_rate=round(verified_rate, 4),
        partial_rate=round(partial_rate, 4),
        unsafe_rate=round(unsafe_rate, 4),
        computed_at=datetime.utcnow(),
    )
