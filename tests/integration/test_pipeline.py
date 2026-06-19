"""
Integration tests for the VeriLayer API.
Tests full request/response cycle using FastAPI's TestClient (no external services needed
for schema/routing checks; external-service tests are marked and skipped by default).

All external service mocking (Redis, Langfuse, audit logger) is handled by
tests/integration/conftest.py via a session-scoped autouse fixture.
"""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def client():
    """Create a synchronous test client for the FastAPI app."""
    from src.api.main import app
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c


# ── Root + Health ──────────────────────────────────────────────────────────────

class TestRootAndHealth:
    def test_root_returns_service_info(self, client: TestClient) -> None:
        resp = client.get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert data["service"] == "VeriLayer"
        assert "docs" in data
        assert "health" in data

    def test_health_endpoint_ok(self, client: TestClient) -> None:
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"

    def test_docs_accessible(self, client: TestClient) -> None:
        resp = client.get("/docs")
        assert resp.status_code == 200


# ── Ingest API ─────────────────────────────────────────────────────────────────

class TestIngestAPI:
    @patch("src.api.routes.ingest.ingest_text")
    def test_ingest_text_endpoint(self, mock_ingest, client: TestClient) -> None:
        from src.models.schemas import IngestResponse
        mock_ingest.return_value = IngestResponse(
            document_id="doc-abc-123",
            chunks_created=10,
            source="test.txt",
            message="Successfully ingested 10 chunks from 'test.txt'.",
        )
        resp = client.post("/ingest", json={
            "source_name": "test.txt",
            "content": "This is a test document about contract law and force majeure clauses.",
        })
        assert resp.status_code == 201
        data = resp.json()
        assert data["document_id"] == "doc-abc-123"
        assert data["chunks_created"] == 10
        assert data["source"] == "test.txt"

    def test_ingest_text_empty_content_rejected(self, client: TestClient) -> None:
        resp = client.post("/ingest", json={
            "source_name": "empty.txt",
            "content": "",
        })
        assert resp.status_code == 422  # Pydantic validation error

    def test_ingest_pdf_non_pdf_rejected(self, client: TestClient) -> None:
        resp = client.post(
            "/ingest/pdf",
            files={"file": ("not_a_pdf.txt", b"some content", "text/plain")},
        )
        assert resp.status_code == 400

    @patch("src.api.routes.ingest.ingest_pdf")
    def test_ingest_pdf_endpoint(self, mock_pdf, client: TestClient) -> None:
        from src.models.schemas import IngestResponse
        mock_pdf.return_value = IngestResponse(
            document_id="pdf-doc-456",
            chunks_created=25,
            source="contract.pdf",
            message="Successfully ingested 25 chunks from 'contract.pdf'.",
        )
        # Minimal valid PDF bytes header
        minimal_pdf = b"%PDF-1.4 fake pdf content"
        resp = client.post(
            "/ingest/pdf",
            files={"file": ("contract.pdf", minimal_pdf, "application/pdf")},
        )
        # mock intercepts so we should get 201
        assert resp.status_code == 201
        data = resp.json()
        assert data["chunks_created"] == 25


# ── Search API ─────────────────────────────────────────────────────────────────

class TestSearchAPI:
    @patch("src.api.routes.search.bm25_search")
    @patch("src.api.routes.search.rank_results")
    def test_bm25_search_endpoint(self, mock_rank, mock_bm25, client: TestClient) -> None:
        from src.models.schemas import SearchResponse, SearchResult
        fake_result = SearchResult(
            chunk_id="c1", document_id="doc1", text="Negligence requires duty of care.",
            score=0.87, source="tort_law.pdf"
        )
        mock_bm25.return_value = SearchResponse(
            query="what is negligence", results=[fake_result], total=1, retrieval_type="bm25"
        )
        mock_rank.return_value = [fake_result]

        resp = client.post("/search/bm25", json={"query": "what is negligence", "top_k": 5})
        assert resp.status_code == 200
        data = resp.json()
        assert data["retrieval_type"] == "bm25"
        assert data["total"] == 1
        assert data["results"][0]["chunk_id"] == "c1"

    def test_bm25_empty_query_rejected(self, client: TestClient) -> None:
        resp = client.post("/search/bm25", json={"query": "", "top_k": 5})
        assert resp.status_code == 422

    @patch("src.api.routes.search.hybrid_search")
    @patch("src.api.routes.search.rank_results")
    def test_hybrid_search_endpoint(self, mock_rank, mock_hybrid, client: TestClient) -> None:
        from src.models.schemas import SearchResponse, SearchResult
        fake_result = SearchResult(
            chunk_id="c2", document_id="doc2", text="Force majeure exempts performance.",
            score=0.014, source="contracts.pdf"
        )
        mock_hybrid.return_value = SearchResponse(
            query="force majeure", results=[fake_result], total=1, retrieval_type="hybrid"
        )
        mock_rank.return_value = [fake_result]

        resp = client.post("/search/hybrid", json={"query": "force majeure", "top_k": 3})
        assert resp.status_code == 200
        data = resp.json()
        assert data["retrieval_type"] == "hybrid"


# ── Verify API ─────────────────────────────────────────────────────────────────

class TestVerifyAPI:
    def test_verify_empty_query_rejected(self, client: TestClient) -> None:
        resp = client.post("/verify", json={"query": ""})
        assert resp.status_code == 422

    def test_verify_query_too_long_rejected(self, client: TestClient) -> None:
        resp = client.post("/verify", json={"query": "q" * 2001})
        assert resp.status_code == 422

    def test_verify_response_schema_contract(self) -> None:
        """Verify the VerifyResponse schema matches the /verify API contract."""
        from src.models.schemas import (
            VerifyResponse, VerifyResponseMetadata, Claim, Source, TraceStep
        )
        # Build a full response — validates schema shape without HTTP call
        response = VerifyResponse(
            query="What is the penalty for late payment?",
            final_answer="A 5% penalty applies for payments delayed beyond 30 days.",
            confidence=0.88,
            status="verified",
            claims=[
                Claim(
                    text="5% penalty applies after 30 days.",
                    verdict="supported",
                    confidence=0.88,
                    sources=[Source(document_id="doc1", chunk_id="c1", text="penalty clause text")]
                )
            ],
            trace=[TraceStep(step="planner", details="Decomposed query", latency_ms=50)],
            metadata=VerifyResponseMetadata(retrieval_docs=5, retries=0, latency_total_ms=1200),
        )
        data = response.model_dump()
        assert data["status"] == "verified"
        assert data["confidence"] == 0.88
        assert len(data["claims"]) == 1
        assert data["claims"][0]["verdict"] == "supported"
        assert "retrieval_docs" in data["metadata"]
        assert data["metadata"]["retrieval_docs"] == 5

    @pytest.mark.live
    def test_verify_endpoint_full_stack(self, client: TestClient) -> None:
        """Full E2E test — requires live Redis, PostgreSQL, OpenSearch, and Groq API key.
        Run with: pytest -m live tests/integration/test_pipeline.py
        """
        resp = client.post("/verify", json={"query": "What is force majeure?", "top_k": 3})
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] in ("verified", "partial", "unsafe")
        assert 0.0 <= data["confidence"] <= 1.0




# ── Metrics API ────────────────────────────────────────────────────────────────

class TestMetricsAPI:
    @patch("src.api.routes.metrics.compute_metrics")
    def test_metrics_endpoint(self, mock_metrics, client: TestClient) -> None:
        from src.models.schemas import MetricsResponse
        from datetime import datetime
        mock_metrics.return_value = MetricsResponse(
            total_queries=42,
            hallucination_rate=0.12,
            avg_confidence=0.83,
            retry_rate=0.15,
            avg_latency_ms=1450.0,
            verified_rate=0.71,
            partial_rate=0.19,
            unsafe_rate=0.10,
            computed_at=datetime.utcnow(),
        )

        resp = client.get("/metrics")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_queries"] == 42
        assert data["hallucination_rate"] == 0.12
        assert data["avg_confidence"] == 0.83
        assert "computed_at" in data
