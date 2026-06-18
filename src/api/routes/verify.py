from __future__ import annotations

import structlog
from fastapi import APIRouter, HTTPException

from src.models.schemas import (
    Claim,
    Source,
    TraceStep,
    VerifyRequest,
    VerifyResponse,
    VerifyResponseMetadata,
)
from src.agents.graph.verilayer_graph import run_pipeline
from src.observability.redis_cache import get_cached_response, set_cached_response
from src.observability.audit_logger import log_verify_result

logger = structlog.get_logger("verilayer.api.verify")
router = APIRouter(tags=["Verification"])


# ── Phase 6: Full Agentic Verification ────────────────────────────────────────

@router.post(
    "/verify",
    response_model=VerifyResponse,
    summary="Full agentic verification pipeline",
)
async def verify_endpoint(request: VerifyRequest) -> VerifyResponse:
    """
    Run the full VeriLayer agentic pipeline:
    planner → retriever → grader → generator → extractor → verifier →
    scorer → decision (retry loop) → finalize.

    Returns grounded answer with per-claim verdicts, confidence scores,
    source citations, and full pipeline trace. Results are Redis-cached.
    """
    # ── Cache-aside: check Redis first ───────────────────────────────────────
    cached = await get_cached_response(request.query, request.top_k)
    if cached:
        logger.info("serving_from_cache", query=request.query[:80])
        return VerifyResponse(**cached)

    try:
        state = await run_pipeline(
            query=request.query,
            top_k=request.top_k,
            session_id=request.session_id,
        )

        # Coerce claims dict list → Claim models
        claims: list[Claim] = []
        for c in state.claims:
            sources = [
                Source(
                    document_id=s.get("document_id", ""),
                    chunk_id=s.get("chunk_id", ""),
                    text=s.get("text", ""),
                    score=s.get("score", 0.0),
                )
                for s in c.get("sources", [])
            ]
            verdict = c.get("verdict") or "unsupported"
            claims.append(
                Claim(
                    text=c.get("text", ""),
                    verdict=verdict,  # type: ignore[arg-type]
                    confidence=float(c.get("confidence", 0.0)),
                    sources=sources,
                )
            )

        # Coerce trace list → TraceStep models
        trace: list[TraceStep] = [
            TraceStep(
                step=t.get("step", "unknown"),
                details=t.get("details", ""),
                latency_ms=int(t.get("latency_ms", 0)),
            )
            for t in state.trace
        ]

        response = VerifyResponse(
            query=state.query,
            final_answer=state.answer,
            confidence=round(state.avg_confidence, 4),
            status=state.status,  # type: ignore[arg-type]
            claims=claims,
            trace=trace,
            metadata=VerifyResponseMetadata(
                retrieval_docs=len(state.retrieved_docs),
                retries=state.retry_count,
                latency_total_ms=state.total_latency_ms,
            ),
        )

        # ── Store in Redis cache ──────────────────────────────────────────────
        await set_cached_response(request.query, request.top_k, response.model_dump())

        # ── Audit log to PostgreSQL ───────────────────────────────────────────
        await log_verify_result(
            query=state.query,
            final_answer=state.answer,
            status=state.status,
            confidence=state.avg_confidence,
            claims=state.claims,
            retries=state.retry_count,
            latency_ms=state.total_latency_ms,
        )

        return response

    except Exception as exc:
        logger.error("verify_pipeline_failed", query=request.query[:80], error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ── Phase 5: Basic RAG (no claim verification) ────────────────────────────────

@router.post(
    "/rag/query",
    summary="Basic RAG answer (no verification)",
)
async def rag_query_endpoint(request: VerifyRequest) -> dict:
    """
    Lightweight RAG: retrieve + generate, skip claim extraction/verification.
    Useful for latency-sensitive use cases.
    """
    try:
        from src.retrieval.hybrid import hybrid_search
        from src.models.schemas import SearchRequest
        from src.rag.generator import call_llm
        from src.rag.prompt_templates import RAG_SYSTEM_PROMPT, RAG_USER_PROMPT

        search_resp = await hybrid_search(
            SearchRequest(query=request.query, top_k=request.top_k)
        )
        docs = search_resp.results

        if not docs:
            return {
                "query": request.query,
                "answer": "Insufficient source documents to answer this query.",
                "retrieval_docs": 0,
            }

        context = "\n\n---\n\n".join(
            f"[Source: {d.source} | Section: {d.section or '?'}]\n{d.text}"
            for d in docs
        )
        answer = await call_llm(
            RAG_SYSTEM_PROMPT,
            RAG_USER_PROMPT.format(context=context, query=request.query),
            temperature=0.1,
            max_tokens=2048,
        )
        return {
            "query": request.query,
            "answer": answer,
            "retrieval_docs": len(docs),
        }

    except Exception as exc:
        logger.error("rag_query_failed", query=request.query[:80], error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc)) from exc
