"""
conftest.py — Integration test fixtures and global patches.
Prevents all external network I/O during test runs.
"""
from __future__ import annotations
import pytest
from unittest.mock import AsyncMock, patch


@pytest.fixture(scope="session", autouse=True)
def mock_all_external_services():
    """
    Session-scoped fixture that mocks ALL external service calls
    before any test module is loaded. Prevents Redis, PostgreSQL,
    OpenSearch, Langfuse, and Groq from being called during tests.
    """
    patches = [
        # Langfuse (observability) — called in lifespan
        patch("src.observability.langfuse_client.get_langfuse_client", return_value=None),
        patch("src.observability.langfuse_client.flush_langfuse", return_value=None),
        patch("src.observability.langfuse_client.observe", lambda name=None, **kw: lambda f: f),
        patch("src.observability.langfuse_client.update_span_metadata", return_value=None),
        # Redis cache — called in verify route
        patch("src.api.routes.verify.get_cached_response", new_callable=AsyncMock, return_value=None),
        patch("src.api.routes.verify.set_cached_response", new_callable=AsyncMock),
        # Audit logger — called in verify route
        patch("src.api.routes.verify.log_verify_result", new_callable=AsyncMock),
    ]
    started = [p.start() for p in patches]
    yield
    for p in patches:
        p.stop()
