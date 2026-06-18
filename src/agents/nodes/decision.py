"""
VeriLayer — Agent Node: DecisionNode.
Implements the retry/confidence loop from the implementation plan:

    if avg_confidence >= 0.8  → finalize (verified)
    elif avg_confidence >= 0.5 → rewrite low-confidence claims → retry verification
    else:
        if retries < 2 → full re-retrieval with expanded query → retry
        else           → finalize (unsafe)

This node is used as a CONDITIONAL EDGE router in LangGraph — it returns
a routing key string, not a new GraphState.
"""
from __future__ import annotations
import structlog

from src.agents.state import GraphState
from src.core.config import settings

logger = structlog.get_logger("verilayer.agents.decision")

# Routing keys — must match edge names in the graph
ROUTE_FINALIZE = "finalize"
ROUTE_REWRITE  = "rewrite"
ROUTE_RETRIEVE = "retrieve"


def decision_node(state: GraphState) -> str:
    """
    Pure routing function — returns a string key to select the next edge.
    Called by LangGraph as a conditional edge after the confidence scorer.
    """
    avg_conf   = state.avg_confidence
    retry_cnt  = state.retry_count
    max_retries = state.max_retries

    logger.info(
        "decision_node",
        avg_confidence=round(avg_conf, 3),
        retry_count=retry_cnt,
        max_retries=max_retries,
        status=state.status,
    )

    if avg_conf >= settings.confidence_verified_threshold:
        # All good — finalize as verified
        logger.info("decision → finalize (verified)")
        return ROUTE_FINALIZE

    if avg_conf >= settings.confidence_partial_threshold:
        if retry_cnt < max_retries:
            # Partial confidence — rewrite bad claims and retry verification
            logger.info("decision → rewrite (partial confidence, retry)")
            return ROUTE_REWRITE
        else:
            # Exhausted retries — partial but done
            logger.info("decision → finalize (partial, retries exhausted)")
            return ROUTE_FINALIZE

    # Low confidence path
    if retry_cnt < max_retries:
        # Re-retrieve with expanded query
        logger.info("decision → retrieve (low confidence, expanding query)")
        return ROUTE_RETRIEVE
    else:
        # Unsafe — give up
        logger.info("decision → finalize (unsafe, retries exhausted)")
        return ROUTE_FINALIZE
