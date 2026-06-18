"""
VeriLayer — Phases 3 + 4 + 5: Search & Verify API routes.
  POST /search/bm25   — BM25 keyword search
  POST /search/hybrid — BM25 + semantic hybrid search (RRF)
  POST /rag/query     — Basic RAG answer generation
  POST /verify        — Full agentic verification pipeline
  GET  /metrics       — Evaluation metrics from audit logs
"""
from __future__ import annotations

import structlog
from fastapi import APIRouter, HTTPException, status

from src.models.schemas import (
    SearchRequest,
    SearchResponse,
    VerifyRequest,
    VerifyResponse,
    MetricsResponse,
)
from src.retrieval.bm25 import bm25_search
from src.retrieval.hybrid import hybrid_search
from src.retrieval.ranker import rank_results

logger = structlog.get_logger("verilayer.api.search")
router = APIRouter(tags=["Search & Retrieval"])


# ── Phase 3: BM25 Search ──────────────────────────────────────────────────────

@router.post(
    "/search/bm25",
    response_model=SearchResponse,
    summary="BM25 keyword search",
)
async def search_bm25(request: SearchRequest) -> SearchResponse:
    """Keyword search using BM25 over OpenSearch."""
    try:
        response = await bm25_search(request)
        response.results = rank_results(response.results, top_k=request.top_k)
        response.total = len(response.results)
        return response
    except Exception as exc:
        logger.error("bm25_search_failed", error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ── Phase 4: Hybrid Search ────────────────────────────────────────────────────

@router.post(
    "/search/hybrid",
    response_model=SearchResponse,
    summary="Hybrid BM25 + semantic search",
)
async def search_hybrid(request: SearchRequest) -> SearchResponse:
    """Hybrid search combining BM25 and Jina semantic embeddings via RRF."""
    try:
        response = await hybrid_search(request)
        response.results = rank_results(response.results, top_k=request.top_k)
        response.total = len(response.results)
        return response
    except Exception as exc:
        logger.error("hybrid_search_failed", error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc)) from exc
