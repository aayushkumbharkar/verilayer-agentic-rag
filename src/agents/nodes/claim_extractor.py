"""
VeriLayer — Agent Node: ClaimExtractor.
Extracts atomic verifiable claims from the generated answer.
"""
from __future__ import annotations
import time
import structlog
from src.agents.state import GraphState
from src.observability.langfuse_client import observe, update_span_metadata
from src.rag.generator import call_llm_json
from src.rag.prompt_templates import CLAIM_EXTRACTOR_SYSTEM_PROMPT, CLAIM_EXTRACTOR_USER_PROMPT

logger = structlog.get_logger("verilayer.agents.claim_extractor")


@observe(name="claim-extractor")
async def claim_extractor_node(state: GraphState) -> GraphState:
    """Extract atomic claims from the answer."""
    t0 = time.monotonic()
    update_span_metadata(input={"answer_preview": state.answer[:200]})

    claims_raw: list[str] = []
    try:
        result = await call_llm_json(
            CLAIM_EXTRACTOR_SYSTEM_PROMPT,
            CLAIM_EXTRACTOR_USER_PROMPT.format(answer=state.answer),
        )
        claims_raw = [str(c) for c in result] if isinstance(result, list) else []
    except Exception as exc:
        logger.warning("claim_extraction_failed", error=str(exc))
        # Fallback: treat the whole answer as one claim
        if state.answer and state.answer != "Insufficient source documents to answer this query.":
            claims_raw = [state.answer[:500]]

    claims = [{"text": c, "verdict": None, "confidence": 0.0, "sources": []} for c in claims_raw]
    latency = int((time.monotonic() - t0) * 1000)
    update_span_metadata(output={"claim_count": len(claims)})
    logger.info("claims_extracted", count=len(claims), latency_ms=latency)

    return state.model_copy(update={
        "claims": claims,
        "trace": state.trace + [{"step": "extractor", "details": f"Extracted {len(claims)} atomic claims", "latency_ms": latency}],
    })
