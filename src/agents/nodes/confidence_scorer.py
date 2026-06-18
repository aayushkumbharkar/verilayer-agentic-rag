"""
VeriLayer — Agent Node: ConfidenceScorer.
Computes average confidence across all verified claims and assigns pipeline status.
"""
from __future__ import annotations
import time
import structlog

from src.agents.state import GraphState
from src.core.config import settings
from src.observability.langfuse_client import observe, update_span_metadata

logger = structlog.get_logger("verilayer.agents.confidence_scorer")


@observe(name="confidence-scorer")
async def confidence_scorer_node(state: GraphState) -> GraphState:
    """Compute avg_confidence and tentative status from verified claims."""
    t0 = time.monotonic()

    claims = state.claims
    if not claims:
        latency = int((time.monotonic() - t0) * 1000)
        return state.model_copy(update={
            "avg_confidence": 0.0,
            "status": "unsafe",
            "trace": state.trace + [{
                "step": "scorer",
                "details": "No claims to score — status=unsafe",
                "latency_ms": latency,
            }],
        })

    confidences = [c.get("confidence", 0.0) for c in claims]
    avg_conf = sum(confidences) / len(confidences)

    verdicts = [c.get("verdict") for c in claims]
    all_supported = all(v == "supported" for v in verdicts)
    any_supported = any(v == "supported" for v in verdicts)

    # Apply status rules from the plan
    if all_supported and avg_conf >= settings.confidence_verified_threshold:
        status = "verified"
    elif avg_conf >= settings.confidence_partial_threshold or any_supported:
        status = "partial"
    else:
        status = "unsafe"

    latency = int((time.monotonic() - t0) * 1000)
    update_span_metadata(output={"avg_confidence": avg_conf, "status": status})
    logger.info(
        "confidence_scored",
        avg_confidence=round(avg_conf, 3),
        status=status,
        claims=len(claims),
        latency_ms=latency,
    )

    return state.model_copy(update={
        "avg_confidence": avg_conf,
        "status": status,
        "trace": state.trace + [{
            "step": "scorer",
            "details": (
                f"avg_confidence={avg_conf:.3f}, status={status}, "
                f"claims={len(claims)}"
            ),
            "latency_ms": latency,
        }],
    })
