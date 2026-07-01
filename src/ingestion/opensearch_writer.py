"""
VeriLayer — Phase 2: OpenSearch index writer.
Creates the verilayer-docs index and writes chunks to it.
"""
from __future__ import annotations

import structlog
from opensearchpy import AsyncOpenSearch, NotFoundError
from opensearchpy.helpers import async_bulk

from src.core.config import settings
from src.models.schemas import Chunk

logger = structlog.get_logger("verilayer.ingestion.opensearch")

INDEX_MAPPING = {
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0,
        "analysis": {
            "analyzer": {
                "verilayer_analyzer": {
                    "type": "standard",
                    "stopwords": "_english_",
                }
            }
        },
    },
    "mappings": {
        "properties": {
            "chunk_id":     {"type": "keyword"},
            "document_id":  {"type": "keyword"},
            "text":         {"type": "text", "analyzer": "verilayer_analyzer"},
            "source":       {"type": "keyword"},
            "section":      {"type": "keyword"},
            "clause":       {"type": "keyword"},
            "token_count":  {"type": "integer"},
            "embedding": {
                "type": "knn_vector",
                "dimension": settings.jina_embed_dimensions,
                "method": {
                    "name": "hnsw",
                    "space_type": "cosinesimil",
                    "engine": "lucene",
                },
            },
        }
    },
}


def _get_client() -> AsyncOpenSearch:
    return AsyncOpenSearch(
        hosts=[{"host": settings.opensearch_host, "port": settings.opensearch_port}],
        use_ssl=False,
        verify_certs=False,
        http_compress=True,
    )


async def ensure_index_exists() -> None:
    """Create the verilayer-docs index if it doesn't already exist."""
    client = _get_client()
    try:
        exists = await client.indices.exists(index=settings.opensearch_index)
        if not exists:
            await client.indices.create(index=settings.opensearch_index, body=INDEX_MAPPING)
            logger.info("opensearch_index_created", index=settings.opensearch_index)
        else:
            logger.info("opensearch_index_exists", index=settings.opensearch_index)
    finally:
        await client.close()


async def index_chunks(chunks: list[Chunk]) -> int:
    """
    Bulk-index a list of Chunk objects into OpenSearch.
    Returns count of successfully indexed chunks.
    """
    if not chunks:
        return 0

    client = _get_client()
    try:
        actions = []
        for chunk in chunks:
            source = {
                "chunk_id":    chunk.chunk_id,
                "document_id": chunk.document_id,
                "text":        chunk.text,
                "source":      chunk.metadata.source,
                "section":     chunk.metadata.section,
                "clause":      chunk.metadata.clause,
                "token_count": chunk.token_count,
            }
            # Only include embedding when a real vector is available
            if chunk.embedding:
                source["embedding"] = chunk.embedding
            actions.append({
                "_index": settings.opensearch_index,
                "_id":    chunk.chunk_id,
                "_source": source,
            })

        success, errors = await async_bulk(client, actions, raise_on_error=False)
        if errors:
            logger.warning("opensearch_bulk_errors", count=len(errors), errors=errors[:3])
        logger.info(
            "chunks_indexed",
            index=settings.opensearch_index,
            success=success,
            total=len(chunks),
        )
        return success
    finally:
        await client.close()


async def delete_document_chunks(document_id: str) -> int:
    """Delete all chunks belonging to a document."""
    client = _get_client()
    try:
        resp = await client.delete_by_query(
            index=settings.opensearch_index,
            body={"query": {"term": {"document_id": document_id}}},
        )
        deleted = resp.get("deleted", 0)
        logger.info("document_chunks_deleted", document_id=document_id, count=deleted)
        return deleted
    finally:
        await client.close()
