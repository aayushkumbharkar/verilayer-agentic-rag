"""
VeriLayer — Canonical Pydantic schemas.
Defined in Phase 1. Used by ALL layers: API, agents, observability, UI.

This is the single source of truth for:
- /verify API response contract
- LangGraph GraphState types
- Ingestion pipeline models
- Search models
"""
from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Literal, Optional

from pydantic import BaseModel, Field


# ═══════════════════════════════════════════════════════════════════════════════
# /verify Response Contract
# ═══════════════════════════════════════════════════════════════════════════════

class Source(BaseModel):
    """A document chunk used as evidence for a claim."""
    document_id: str
    chunk_id: str
    text: str
    score: float = Field(default=0.0, ge=0.0, description="Relevance score")


class Claim(BaseModel):
    """An atomic factual claim extracted from the generated answer."""
    text: str = Field(description="The atomic claim text")
    verdict: Literal["supported", "unsupported", "partial"] = Field(
        description="Verification verdict against retrieved sources"
    )
    confidence: Annotated[float, Field(ge=0.0, le=1.0)] = Field(
        description="Confidence score for the verdict"
    )
    sources: list[Source] = Field(default_factory=list, description="Supporting source chunks")


class TraceStep(BaseModel):
    """A single agent step in the pipeline trace."""
    step: str = Field(
        description="Agent node name: planner | retriever | grader | generator | "
                    "extractor | verifier | scorer | rewriter | decision"
    )
    details: str = Field(description="Human-readable description of what this step did")
    latency_ms: int = Field(ge=0, description="Wall-clock latency for this step in ms")


class VerifyResponseMetadata(BaseModel):
    """Pipeline execution metadata."""
    retrieval_docs: int = Field(ge=0, description="Number of documents retrieved")
    retries: int = Field(ge=0, description="Number of verification retries performed")
    latency_total_ms: int = Field(ge=0, description="Total pipeline latency in ms")


class VerifyResponse(BaseModel):
    """
    Canonical /verify endpoint response.
    This is the single source of truth for what the system produces.
    Status rules:
      - verified: all claims supported, avg confidence >= 0.8
      - partial: mixed verdicts OR avg confidence 0.5-0.8
      - unsafe: retrieval failed OR confidence < 0.5 after max retries
    """
    query: str
    final_answer: str
    confidence: Annotated[float, Field(ge=0.0, le=1.0)]
    status: Literal["verified", "partial", "unsafe"]
    claims: list[Claim] = Field(default_factory=list)
    trace: list[TraceStep] = Field(default_factory=list)
    metadata: VerifyResponseMetadata


class VerifyRequest(BaseModel):
    """Request body for POST /verify."""
    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)
    session_id: Optional[str] = Field(default=None, description="Optional session ID for tracing")


# ═══════════════════════════════════════════════════════════════════════════════
# Ingestion Models
# ═══════════════════════════════════════════════════════════════════════════════

class DocumentMetadata(BaseModel):
    """Metadata extracted from an ingested document."""
    document_id: str
    source: str = Field(description="Filename or URL of the source document")
    section: Optional[str] = Field(default=None, description="Document section (e.g. Article 3)")
    clause: Optional[str] = Field(default=None, description="Specific clause or paragraph ID")
    page_number: Optional[int] = Field(default=None)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class Chunk(BaseModel):
    """A text chunk ready for indexing."""
    chunk_id: str
    document_id: str
    text: str = Field(min_length=1)
    metadata: DocumentMetadata
    embedding: Optional[list[float]] = Field(default=None, description="Dense embedding vector")
    token_count: int = Field(default=0, ge=0)


class IngestRequest(BaseModel):
    """Request body for POST /ingest (text content)."""
    source_name: str = Field(min_length=1, description="Filename or document identifier")
    content: str = Field(min_length=1, description="Raw text content to ingest")
    section: Optional[str] = Field(default=None)
    clause: Optional[str] = Field(default=None)


class IngestResponse(BaseModel):
    """Response from POST /ingest."""
    document_id: str
    chunks_created: int
    source: str
    message: str


# ═══════════════════════════════════════════════════════════════════════════════
# Search Models
# ═══════════════════════════════════════════════════════════════════════════════

class SearchRequest(BaseModel):
    """Request body for search endpoints."""
    query: str = Field(min_length=1, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=20)
    filters: Optional[dict[str, Any]] = Field(
        default=None, description="Optional metadata filters"
    )


class SearchResult(BaseModel):
    """A single search result (BM25 or hybrid)."""
    chunk_id: str
    document_id: str
    text: str
    score: float = Field(ge=0.0)
    source: str
    section: Optional[str] = None
    clause: Optional[str] = None


class SearchResponse(BaseModel):
    """Response from search endpoints."""
    query: str
    results: list[SearchResult]
    total: int
    retrieval_type: Literal["bm25", "semantic", "hybrid"] = "bm25"


# ═══════════════════════════════════════════════════════════════════════════════
# Evaluation / Metrics Models
# ═══════════════════════════════════════════════════════════════════════════════

class MetricsResponse(BaseModel):
    """Response from GET /metrics — computed from audit logs."""
    total_queries: int
    hallucination_rate: float = Field(ge=0.0, le=1.0, description="Fraction of unsupported claims")
    avg_confidence: float = Field(ge=0.0, le=1.0)
    retry_rate: float = Field(ge=0.0, le=1.0, description="Fraction of queries needing >= 1 retry")
    avg_latency_ms: float = Field(ge=0.0)
    verified_rate: float = Field(ge=0.0, le=1.0)
    partial_rate: float = Field(ge=0.0, le=1.0)
    unsafe_rate: float = Field(ge=0.0, le=1.0)
    computed_at: datetime = Field(default_factory=datetime.utcnow)


# ═══════════════════════════════════════════════════════════════════════════════
# Service Health
# ═══════════════════════════════════════════════════════════════════════════════

class ServiceStatus(BaseModel):
    """Status of a single infrastructure service."""
    status: Literal["healthy", "unhealthy"]
    error: Optional[str] = None
    version: Optional[str] = None


class HealthResponse(BaseModel):
    """Response from GET /health/services."""
    status: Literal["healthy", "degraded", "unhealthy"]
    services: dict[str, ServiceStatus]
