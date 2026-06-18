"""
VeriLayer — Phase 4: Hybrid Search (BM25 + Semantic fusion).
Uses Reciprocal Rank Fusion (RRF) to combine BM25 and KNN results.
"""
from __future__ import annotations

import structlog

from src.models.schemas import SearchRequest, SearchResponse, SearchResult
from src.retrieval.bm25 import bm25_search
from src.retrieval.vector_search import vector_search

logger = structlog.get_logger("verilayer.retrieval.hybrid")

RRF_K = 60  # RRF constant — standard value per literature


def _reciprocal_rank_fusion(
    bm25_results: list[SearchResult],
    semantic_results: list[SearchResult],
    bm25_weight: float = 0.5,
    semantic_weight: float = 0.5,
) -> list[SearchResult]:
    """
    Merge BM25 and semantic results using Reciprocal Rank Fusion.
    Documents appearing in both lists get boosted scores.
    """
    scores: dict[str, float] = {}
    chunks: dict[str, SearchResult] = {}

    for rank, result in enumerate(bm25_results, start=1):
        rrf_score = bm25_weight * (1.0 / (RRF_K + rank))
        scores[result.chunk_id] = scores.get(result.chunk_id, 0.0) + rrf_score
        chunks[result.chunk_id] = result

    for rank, result in enumerate(semantic_results, start=1):
        rrf_score = semantic_weight * (1.0 / (RRF_K + rank))
        scores[result.chunk_id] = scores.get(result.chunk_id, 0.0) + rrf_score
        if result.chunk_id not in chunks:
            chunks[result.chunk_id] = result

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    fused: list[SearchResult] = []
    for chunk_id, fused_score in ranked:
        result = chunks[chunk_id]
        fused.append(result.model_copy(update={"score": round(fused_score, 6)}))

    return fused


async def hybrid_search(request: SearchRequest) -> SearchResponse:
    """
    Perform hybrid search combining BM25 and semantic results via RRF.
    Fetches 2× top_k from each retriever then fuses and trims to top_k.
    """
    expanded = SearchRequest(
        query=request.query,
        top_k=request.top_k * 2,
        filters=request.filters,
    )

    bm25_resp, semantic_resp = await _run_both(expanded)

    fused = _reciprocal_rank_fusion(bm25_resp.results, semantic_resp.results)
    fused = fused[: request.top_k]

    logger.info(
        "hybrid_search_complete",
        query=request.query[:80],
        bm25_count=len(bm25_resp.results),
        semantic_count=len(semantic_resp.results),
        fused_count=len(fused),
    )
    return SearchResponse(
        query=request.query,
        results=fused,
        total=len(fused),
        retrieval_type="hybrid",
    )


async def _run_both(request: SearchRequest):
    """Run BM25 and vector search concurrently."""
    import asyncio
    return await asyncio.gather(bm25_search(request), vector_search(request))
