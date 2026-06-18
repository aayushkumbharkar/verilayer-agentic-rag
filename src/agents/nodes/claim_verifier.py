"""
VeriLayer — Agent Node: ClaimVerifier.
Verifies each claim against retrieved source documents.
"""
from __future__ import annotations
import asyncio, time
import structlog
from src.agents.state import GraphState
from src.observability.langfuse_client import observe, update_span_metadata
from src.rag.generator import call_llm_json
from src.rag.prompt_templates import CLAIM_VERIFIER_SYSTEM_PROMPT, CLAIM_VERIFIER_USER_PROMPT

logger = structlog.get_logger("verilayer.agents.claim_verifier")

VALID_VERDICTS = {"supported", "unsupported", "partial"}


@observe(name="claim-verifier")
async def claim_verifier_node(state: GraphState) -> GraphState:
    """Verify each claim against graded source documents."""
    t0 = time.monotonic()
    docs = state.graded_docs or state.retrieved_docs
    sources_text = "\n\n---\n\n".join(
        f"[{d.get('source','?')}]: {d['text'][:400]}" for d in docs[:5]
    )

    async def verify_claim(claim: dict) -> dict:
        try:
            result = await call_llm_json(
                CLAIM_VERIFIER_SYSTEM_PROMPT,
                CLAIM_VERIFIER_USER_PROMPT.format(claim=claim["text"], sources=sources_text),
            )
            verdict = str(result.get("verdict", "unsupported")).lower()
            if verdict not in VALID_VERDICTS:
                verdict = "unsupported"
            confidence = float(result.get("confidence", 0.0))
            confidence = max(0.0, min(1.0, confidence))

            supporting = [
                {"document_id": d.get("document_id",""), "chunk_id": d.get("chunk_id",""), "text": d["text"][:200]}
                for d in docs[:3]
            ] if verdict in ("supported", "partial") else []

            return {**claim, "verdict": verdict, "confidence": confidence, "sources": supporting}
        except Exception as exc:
            logger.warning("claim_verification_failed", claim=claim["text"][:80], error=str(exc))
            return {**claim, "verdict": "unsupported", "confidence": 0.0, "sources": []}

    verified = await asyncio.gather(*[verify_claim(c) for c in state.claims])
    verified_list = list(verified)
    latency = int((time.monotonic() - t0) * 1000)

    supported = sum(1 for c in verified_list if c["verdict"] == "supported")
    update_span_metadata(output={"verified": len(verified_list), "supported": supported})
    logger.info("verification_complete", total=len(verified_list), supported=supported)

    return state.model_copy(update={
        "claims": verified_list,
        "trace": state.trace + [{"step": "verifier", "details": f"Verified {len(verified_list)} claims: {supported} supported", "latency_ms": latency}],
    })
