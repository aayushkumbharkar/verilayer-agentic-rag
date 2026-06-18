"""
VeriLayer — FastAPI application entrypoint.

Configures:
- Structured logging (structlog)
- Langfuse tracing client (initialized and flushed via lifespan)
- CORS middleware
- All route groups
- OpenAPI metadata
"""
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.core.config import settings
from src.core.logging import setup_logging
from src.observability.langfuse_client import flush_langfuse, get_langfuse_client

# Import routers — all phases
from src.api.routes import health
from src.api.routes import ingest
from src.api.routes import search
from src.api.routes import verify
from src.api.routes import metrics

_logger = structlog.get_logger("verilayer.startup")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan:
    - startup: configure logging, initialize Langfuse
    - shutdown: flush Langfuse events
    """
    # ── Startup ──────────────────────────────────────────────────────────────
    setup_logging(log_level="DEBUG" if settings.debug else "INFO")
    _logger.info(
        "verilayer_starting",
        app=settings.app_name,
        version=settings.app_version,
        debug=settings.debug,
    )

    # Initialize Langfuse client (cached singleton)
    lf_client = get_langfuse_client()
    if lf_client:
        _logger.info("langfuse_enabled", host=settings.langfuse_host)
    else:
        _logger.warning("langfuse_disabled", reason="API keys not configured")

    yield

    # ── Shutdown ─────────────────────────────────────────────────────────────
    _logger.info("verilayer_shutting_down")
    flush_langfuse()  # Ensure all traces are flushed before exit


# ── FastAPI App ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="VeriLayer API",
    description=(
        "**VeriLayer** — Trust Layer for LLM Systems.\n\n"
        "An agentic RAG system that plans queries, retrieves documents via hybrid search, "
        "generates answers, extracts atomic claims, verifies each claim against sources, "
        "assigns confidence scores, rewrites hallucinated claims, and returns "
        "grounded answers with full citations and audit trails.\n\n"
        "Powered by: **Groq** (LLM) · **Jina** (embeddings) · **OpenSearch** (hybrid search) "
        "· **LangGraph** (agentic pipeline) · **Langfuse** (observability)"
    ),
    version=settings.app_version,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# ── Middleware ────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routes ───────────────────────────────────────────────────────────────────
app.include_router(health.router)
app.include_router(ingest.router)         # Phase 2: POST /ingest, POST /ingest/pdf
app.include_router(search.router)         # Phase 3+4: POST /search/bm25, POST /search/hybrid
app.include_router(verify.router)         # Phase 5+6: POST /rag/query, POST /verify
app.include_router(metrics.router)        # Phase 7: GET /metrics


@app.get("/", tags=["Root"])
async def root() -> dict[str, str]:
    """Root endpoint — returns service info."""
    return {
        "service": settings.app_name,
        "version": settings.app_version,
        "docs": "/docs",
        "health": "/health",
        "services": "/health/services",
    }
