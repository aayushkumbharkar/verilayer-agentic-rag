"""
VeriLayer — Phase 8: HTTP client for the /verify API.
Used by the Gradio UI to communicate with the FastAPI backend.
"""
from __future__ import annotations

import httpx

BASE_URL = "http://localhost:8000"
TIMEOUT = 120.0  # LLM calls can take time


async def call_verify(query: str, top_k: int = 5) -> dict:
    """POST /verify and return the response JSON."""
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        resp = await client.post(
            f"{BASE_URL}/verify",
            json={"query": query, "top_k": top_k},
        )
        resp.raise_for_status()
        return resp.json()


async def call_rag_query(query: str, top_k: int = 5) -> dict:
    """POST /rag/query and return the response JSON."""
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        resp = await client.post(
            f"{BASE_URL}/rag/query",
            json={"query": query, "top_k": top_k},
        )
        resp.raise_for_status()
        return resp.json()


async def call_health() -> dict:
    """GET /health — returns service status."""
    async with httpx.AsyncClient(timeout=5.0) as client:
        resp = await client.get(f"{BASE_URL}/health")
        resp.raise_for_status()
        return resp.json()


async def call_metrics() -> dict:
    """GET /metrics — returns evaluation metrics."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(f"{BASE_URL}/metrics")
        resp.raise_for_status()
        return resp.json()
