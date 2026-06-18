"""
VeriLayer — Phase 7: Structured audit logger.
Writes each /verify pipeline run to the PostgreSQL audit_logs table.
Wraps postgres_writer.save_audit_log with Langfuse span observability.
"""
from __future__ import annotations

import structlog

from src.ingestion.postgres_writer import save_audit_log
from src.observability.langfuse_client import observe

logger = structlog.get_logger("verilayer.observability.audit")


@observe(name="audit-logger")
async def log_verify_result(
    query: str,
    final_answer: str,
    status: str,
    confidence: float,
    claims: list[dict],
    retries: int,
    latency_ms: int,
) -> None:
    """
    Persist a full pipeline audit record to PostgreSQL.

    Args:
        query: The original user query.
        final_answer: The final generated answer.
        status: Pipeline status: verified | partial | unsafe.
        confidence: Average claim confidence score.
        claims: List of claim dicts (text, verdict, confidence, sources).
        retries: Number of retry cycles performed.
        latency_ms: Total pipeline wall-clock latency in ms.
    """
    try:
        await save_audit_log(
            query=query,
            final_answer=final_answer,
            status=status,
            confidence=confidence,
            claims=claims,
            retries=retries,
            latency_ms=latency_ms,
        )
        logger.info(
            "audit_logged",
            status=status,
            confidence=round(confidence, 3),
            claims=len(claims),
            retries=retries,
            latency_ms=latency_ms,
        )
    except Exception as exc:
        # Audit logging must never crash the pipeline
        logger.error("audit_log_failed", error=str(exc))
