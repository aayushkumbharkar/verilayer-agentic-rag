"""
VeriLayer — Phase 4: Jina Embeddings client.
Calls the Jina Embeddings v3 REST API to generate dense vectors.
"""
from __future__ import annotations

import structlog
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from src.core.config import settings

logger = structlog.get_logger("verilayer.retrieval.embeddings")

JINA_EMBED_URL = "https://api.jina.ai/v1/embeddings"


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
async def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Generate embeddings for a list of texts using Jina v3.
    Returns a list of float vectors (one per text).
    """
    if not texts:
        return []

    headers = {
        "Authorization": f"Bearer {settings.jina_api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": settings.jina_embed_model,
        "input": texts,
        "dimensions": settings.jina_embed_dimensions,
        "normalized": True,
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(JINA_EMBED_URL, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()

    embeddings = [item["embedding"] for item in data["data"]]
    logger.info(
        "embeddings_generated",
        count=len(embeddings),
        model=settings.jina_embed_model,
        dimensions=settings.jina_embed_dimensions,
    )
    return embeddings


async def embed_query(query: str) -> list[float]:
    """Embed a single query string."""
    results = await embed_texts([query])
    return results[0] if results else []
