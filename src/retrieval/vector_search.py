"""
VeriLayer — Phase 4: KNN vector search via OpenSearch.
"""
from __future__ import annotations

import structlog
from opensearchpy import AsyncOpenSearch

from src.core.config import settings
from src.models.schemas import SearchRequest, SearchResponse, SearchResult
from src.retrieval.embeddings import embed_query

logger = structlog.get_logger("verilayer.retrieval.vector")


def _get_client() -> AsyncOpenSearch:
    return AsyncOpenSearch(
        hosts=[{"host": settings.opensearch_host, "port": settings.opensearch_port}],
        use_ssl=False,
        verify_certs=False,
        http_compress=True,
    )


async def vector_search(request: SearchRequest) -> SearchResponse:
    """
    Perform KNN semantic search using Jina embeddings.
    Returns the top_k most similar chunks by cosine similarity.
    """
    query_embedding = await embed_query(request.query)
    if not query_embedding:
        return SearchResponse(query=request.query, results=[], total=0, retrieval_type="semantic")

    client = _get_client()
    try:
        body = {
            "size": request.top_k,
            "query": {
                "knn": {
                    "embedding": {
                        "vector": query_embedding,
                        "k": request.top_k,
                    }
                }
            },
        }
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
        logger.info("vector_search_complete", query=request.query[:80], results=len(results))
        return SearchResponse(query=request.query, results=results, total=len(results), retrieval_type="semantic")
    finally:
        await client.close()
