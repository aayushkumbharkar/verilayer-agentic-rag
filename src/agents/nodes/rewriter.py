"""
VeriLayer — Agent Node: Rewriter.
Rewrites unsupported/partial claims to accurately reflect source documents.
Low-confidence claims are rewritten before the next verification retry.
"""
from __future__ import annotations
import time
import structlog

from src.agents.state import GraphState
from src.observability.langfuse_client import observe, update_span_metadata
from src.rag.generator import call_llm
from src.rag.prompt_templates import REWRITER_SYSTEM_PROMPT, REWRITER_USER_PROMPT

logger = structlog.get_logger("verilayer.agents.rewriter")


@observe(name="claim-rewriter")
async def rewriter_node(state: GraphState) -> GraphState:
    """
    Rewrite claims that are unsupported or partial.
    Called before a retry cycle to improve claim quality.
    """
    t0 = time.monotonic()
    docs = state.graded_docs or state.retrieved_docs
    sources_text = "\n\n---\n\n".join(
        f"[{d.get('source', '?')}]: {d['text'][:400]}" for d in docs[:5]
    )

    rewritten_claims = []
    rewrite_count = 0

    for claim in state.claims:
        verdict = claim.get("verdict", "unsupported")
        confidence = claim.get("confidence", 0.0)

        # Only rewrite low-confidence or unsupported/partial claims
        if verdict in ("unsupported", "partial") or confidence < 0.6:
            try:
                rewritten_text = await call_llm(
                    REWRITER_SYSTEM_PROMPT,
                    REWRITER_USER_PROMPT.format(
                        claim=claim["text"],
                        verdict=verdict,
                        sources=sources_text,
                    ),
                    temperature=0.0,
                    max_tokens=256,
                )
                rewritten_claims.append({
                    **claim,
                    "text": rewritten_text.strip(),
                    # Reset verdict/confidence — will be re-verified
                    "verdict": None,
                    "confidence": 0.0,
                    "sources": [],
                })
                rewrite_count += 1
            except Exception as exc:
                logger.warning(
                    "rewrite_failed",
                    claim=claim["text"][:80],
                    error=str(exc),
                )
                rewritten_claims.append(claim)
        else:
            rewritten_claims.append(claim)

    latency = int((time.monotonic() - t0) * 1000)
    update_span_metadata(output={"rewritten": rewrite_count, "total": len(rewritten_claims)})
    logger.info("claims_rewritten", rewritten=rewrite_count, total=len(rewritten_claims))

    return state.model_copy(update={
        "claims": rewritten_claims,
        "trace": state.trace + [{
            "step": "rewriter",
            "details": f"Rewrote {rewrite_count}/{len(rewritten_claims)} claims",
            "latency_ms": latency,
        }],
    })
