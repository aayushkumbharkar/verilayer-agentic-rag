"""
Phase 1 — Unit tests for Pydantic schemas and config loading.
"""
import pytest
from pydantic import ValidationError

from src.models.schemas import (
    Claim,
    HealthResponse,
    ServiceStatus,
    Source,
    TraceStep,
    VerifyRequest,
    VerifyResponse,
    VerifyResponseMetadata,
)


class TestVerifyResponse:
    def test_valid_verified_response(self) -> None:
        source = Source(document_id="doc1", chunk_id="c1", text="Relevant passage.", score=0.9)
        claim = Claim(text="X is true.", verdict="supported", confidence=0.92, sources=[source])
        step = TraceStep(step="planner", details="Decomposed into 2 sub-queries", latency_ms=45)
        meta = VerifyResponseMetadata(retrieval_docs=5, retries=0, latency_total_ms=800)

        resp = VerifyResponse(
            query="Is X true?",
            final_answer="Yes, X is true.",
            confidence=0.92,
            status="verified",
            claims=[claim],
            trace=[step],
            metadata=meta,
        )
        assert resp.status == "verified"
        assert resp.confidence == 0.92
        assert len(resp.claims) == 1
        assert resp.claims[0].verdict == "supported"

    def test_confidence_out_of_range(self) -> None:
        with pytest.raises(ValidationError):
            Claim(text="bad", verdict="supported", confidence=1.5, sources=[])

    def test_invalid_verdict(self) -> None:
        with pytest.raises(ValidationError):
            Claim(text="bad", verdict="maybe", confidence=0.5, sources=[])  # type: ignore

    def test_invalid_status(self) -> None:
        with pytest.raises(ValidationError):
            VerifyResponse(
                query="q",
                final_answer="a",
                confidence=0.5,
                status="unknown",  # type: ignore
                claims=[],
                trace=[],
                metadata=VerifyResponseMetadata(
                    retrieval_docs=0, retries=0, latency_total_ms=0
                ),
            )


class TestVerifyRequest:
    def test_defaults(self) -> None:
        req = VerifyRequest(query="What is the penalty for breach of contract?")
        assert req.top_k == 5
        assert req.session_id is None

    def test_empty_query_rejected(self) -> None:
        with pytest.raises(ValidationError):
            VerifyRequest(query="")


class TestHealthResponse:
    def test_all_healthy(self) -> None:
        resp = HealthResponse(
            status="healthy",
            services={
                "postgresql": ServiceStatus(status="healthy"),
                "opensearch": ServiceStatus(status="healthy"),
                "redis": ServiceStatus(status="healthy"),
            },
        )
        assert resp.status == "healthy"

    def test_degraded(self) -> None:
        resp = HealthResponse(
            status="degraded",
            services={
                "postgresql": ServiceStatus(status="healthy"),
                "opensearch": ServiceStatus(status="unhealthy", error="Connection refused"),
                "redis": ServiceStatus(status="healthy"),
            },
        )
        assert resp.status == "degraded"
        assert resp.services["opensearch"].error == "Connection refused"


class TestConfig:
    def test_settings_load(self) -> None:
        from src.core.config import settings
        assert settings.app_name == "VeriLayer"
        assert settings.app_version == "0.1.0"

    def test_postgres_dsn_format(self) -> None:
        from src.core.config import settings
        assert settings.postgres_dsn.startswith("postgresql://")
        assert settings.postgres_async_dsn.startswith("postgresql+asyncpg://")
