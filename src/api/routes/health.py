"""
VeriLayer — Health check routes.

GET /health       → liveness probe (always 200 if API is alive)
GET /health/services → readiness probe (checks PostgreSQL, OpenSearch, Redis)
"""
import time
from typing import Any

import asyncpg
import redis.asyncio as aioredis
import structlog
from fastapi import APIRouter
from opensearchpy import AsyncOpenSearch

from src.core.config import settings
from src.models.schemas import HealthResponse, ServiceStatus

logger = structlog.get_logger("verilayer.health")
router = APIRouter(prefix="/health", tags=["Health"])


@router.get("", summary="Liveness probe")
async def health() -> dict[str, str]:
    """
    Simple liveness check — returns 200 as long as the API process is running.
    """
    return {"status": "ok", "service": settings.app_name, "version": settings.app_version}


@router.get(
    "/services",
    response_model=HealthResponse,
    summary="Readiness probe — checks all backing services",
)
async def health_services() -> HealthResponse:
    """
    Checks connectivity to PostgreSQL, OpenSearch, and Redis.
    Returns 200 with per-service status. Status field is:
      - healthy:   all services reachable
      - degraded:  some services unreachable
    """
    services: dict[str, ServiceStatus] = {}

    # ── PostgreSQL ──────────────────────────────────────────────────────────
    t0 = time.monotonic()
    try:
        conn = await asyncpg.connect(
            host=settings.postgres_host,
            port=settings.postgres_port,
            database=settings.postgres_db,
            user=settings.postgres_user,
            password=settings.postgres_password,
            timeout=5,
        )
        await conn.execute("SELECT 1")
        await conn.close()
        latency = int((time.monotonic() - t0) * 1000)
        services["postgresql"] = ServiceStatus(
            status="healthy",
            version=f"latency={latency}ms",
        )
        logger.info("health_check_postgres", status="healthy", latency_ms=latency)
    except Exception as exc:
        logger.warning("health_check_postgres", status="unhealthy", error=str(exc))
        services["postgresql"] = ServiceStatus(status="unhealthy", error=str(exc))

    # ── OpenSearch ──────────────────────────────────────────────────────────
    t0 = time.monotonic()
    os_client: AsyncOpenSearch | None = None
    try:
        os_client = AsyncOpenSearch(
            hosts=[{"host": settings.opensearch_host, "port": settings.opensearch_port}],
            use_ssl=False,
            verify_certs=False,
            http_compress=True,
        )
        info: dict[str, Any] = await os_client.info()
        latency = int((time.monotonic() - t0) * 1000)
        version = info.get("version", {}).get("number", "unknown")
        services["opensearch"] = ServiceStatus(
            status="healthy",
            version=f"{version} latency={latency}ms",
        )
        logger.info("health_check_opensearch", status="healthy", version=version, latency_ms=latency)
    except Exception as exc:
        logger.warning("health_check_opensearch", status="unhealthy", error=str(exc))
        services["opensearch"] = ServiceStatus(status="unhealthy", error=str(exc))
    finally:
        if os_client:
            await os_client.close()

    # ── Redis ───────────────────────────────────────────────────────────────
    t0 = time.monotonic()
    redis_client: aioredis.Redis | None = None
    try:
        redis_client = aioredis.Redis(
            host=settings.redis_host,
            port=settings.redis_port,
            socket_connect_timeout=5,
        )
        await redis_client.ping()
        latency = int((time.monotonic() - t0) * 1000)
        services["redis"] = ServiceStatus(
            status="healthy",
            version=f"latency={latency}ms",
        )
        logger.info("health_check_redis", status="healthy", latency_ms=latency)
    except Exception as exc:
        logger.warning("health_check_redis", status="unhealthy", error=str(exc))
        services["redis"] = ServiceStatus(status="unhealthy", error=str(exc))
    finally:
        if redis_client:
            await redis_client.aclose()

    overall = (
        "healthy" if all(s.status == "healthy" for s in services.values()) else "degraded"
    )
    logger.info("health_check_complete", overall=overall)
    return HealthResponse(status=overall, services=services)
