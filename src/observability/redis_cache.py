"""
VeriLayer — Phase 7: Redis cache for /verify responses.
Caches query → VerifyResponse JSON by SHA-256 of the query string.
TTL defaults to settings.redis_ttl (3600s).
"""
from __future__ import annotations

import hashlib
import json
import structlog
import redis.asyncio as aioredis

from src.core.config import settings

logger = structlog.get_logger("verilayer.observability.cache")

_client: aioredis.Redis | None = None


def _get_client() -> aioredis.Redis:
    global _client
    if _client is None:
        _client = aioredis.Redis(
            host=settings.redis_host,
            port=settings.redis_port,
            decode_responses=True,
        )
    return _client


def _cache_key(query: str, top_k: int) -> str:
    """Deterministic cache key from query + top_k."""
    raw = f"{query.strip().lower()}|top_k={top_k}"
    return "verilayer:verify:" + hashlib.sha256(raw.encode()).hexdigest()


async def get_cached_response(query: str, top_k: int) -> dict | None:
    """
    Retrieve a cached verify response.
    Returns the parsed dict if found, None on miss or error.
    """
    key = _cache_key(query, top_k)
    try:
        client = _get_client()
        raw = await client.get(key)
        if raw:
            logger.info("cache_hit", key=key[:24])
            return json.loads(raw)
    except Exception as exc:
        logger.warning("cache_get_failed", error=str(exc))
    return None


async def set_cached_response(query: str, top_k: int, response_dict: dict) -> None:
    """
    Store a verify response in Redis with TTL.
    Silently skips on error (cache is best-effort).
    """
    key = _cache_key(query, top_k)
    try:
        client = _get_client()
        await client.setex(key, settings.redis_ttl, json.dumps(response_dict))
        logger.info("cache_set", key=key[:24], ttl=settings.redis_ttl)
    except Exception as exc:
        logger.warning("cache_set_failed", error=str(exc))


async def invalidate_cache(query: str, top_k: int) -> None:
    """Manually invalidate a cached response."""
    key = _cache_key(query, top_k)
    try:
        client = _get_client()
        await client.delete(key)
        logger.info("cache_invalidated", key=key[:24])
    except Exception as exc:
        logger.warning("cache_invalidate_failed", error=str(exc))
