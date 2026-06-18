"""
VeriLayer — Phase 3: BM25 search via OpenSearch.
"""
from __future__ import annotations

import structlog
from opensearchpy import AsyncOpenSearch

from src.core.config import settings
from src.models.schemas import SearchRequest, SearchResponse, SearchResult

logger = structlog.get_logger("verilayer.retrieval.bm25")


def _get_client() -> AsyncOpenSearch:
    return AsyncOpenSearch(
        hosts=[{"host": settings.opensearch_host, "port": settings.opensearch_port}],
        use_ssl=False,
        verify_certs=False,
        http_compress=True,
    )


def _build_bm25_query(query: str, top_k: int, filters: dict | None = None) -> dict:
    """Build a BM25 multi_match query with optional metadata filters."""
    must = [
        {
            "multi_match": {
                "query": query,
                "fields": ["text^3", "source^1", "section^2", "clause^2"],
                "type": "best_fields",
                "fuzziness": "AUTO",
            }
        }
    ]
    body: dict = {"query": {"bool": {"must": must}}, "size": top_k}

    if filters:
        filter_clauses = []
        for field, value in filters.items():
            filter_clauses.append({"term": {field: value}})
        body["query"]["bool"]["filter"] = filter_clauses

    return body


async def bm25_search(request: SearchRequest) -> SearchResponse:
    """
    Perform BM25 keyword search against OpenSearch.
    Returns ranked results by relevance score.
    """
    client = _get_client()
    try:
        body = _build_bm25_query(request.query, request.top_k, request.filters)
        response = await client.search(index=settings.opensearch_index, body=body)

        hits = response.get("hits", {}).get("hits", [])
        results = [
            SearchResult(
                chunk_id=hit["_source"]["chunk_id"],
                document_id=hit["_source"]["document_id"],
                text=hit["_source"]["text"],
                score=hit["_score"],
                source=hit["_source"].get("source", ""),
                section=hit["_source"].get("section"),
                clause=hit["_source"].get("clause"),
            )
            for hit in hits
        ]

        logger.info(
            "bm25_search_complete",
            query=request.query[:80],
            results=len(results),
            top_score=results[0].score if results else 0.0,
        )
        return SearchResponse(
            query=request.query,
            results=results,
            total=len(results),
            retrieval_type="bm25",
        )
    finally:
        await client.close()
