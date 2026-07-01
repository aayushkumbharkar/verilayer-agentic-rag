"""
VeriLayer — Phase 8: HTTP client for the /verify API.
Used by the Gradio UI to communicate with the FastAPI backend.
"""
from __future__ import annotations

import os
import httpx

BASE_URL = os.environ.get("VERILAYER_API_URL", "http://localhost:8000").rstrip("/")
TIMEOUT = 120.0  # LLM calls can take time
HEADERS = {
    "ngrok-skip-browser-warning": "true",
    "User-Agent": "VeriLayer-UI/1.0"
}


async def call_verify(query: str, top_k: int = 5) -> dict:
    """POST /verify and return the response JSON."""
    async with httpx.AsyncClient(timeout=TIMEOUT, headers=HEADERS) as client:
        resp = await client.post(
            f"{BASE_URL}/verify",
            json={"query": query, "top_k": top_k},
        )
        resp.raise_for_status()
        return resp.json()


async def call_rag_query(query: str, top_k: int = 5) -> dict:
    """POST /rag/query and return the response JSON."""
    async with httpx.AsyncClient(timeout=TIMEOUT, headers=HEADERS) as client:
        resp = await client.post(
            f"{BASE_URL}/rag/query",
            json={"query": query, "top_k": top_k},
        )
        resp.raise_for_status()
        return resp.json()


async def call_health() -> dict:
    """GET /health — returns service status."""
    async with httpx.AsyncClient(timeout=5.0, headers=HEADERS) as client:
        resp = await client.get(f"{BASE_URL}/health")
        resp.raise_for_status()
        return resp.json()


async def call_metrics() -> dict:
    """GET /metrics — returns evaluation metrics."""
    async with httpx.AsyncClient(timeout=10.0, headers=HEADERS) as client:
        resp = await client.get(f"{BASE_URL}/metrics")
        resp.raise_for_status()
        return resp.json()


async def call_ingest_text(
    source_name: str,
    content: str,
    section: str | None = None,
    clause: str | None = None,
) -> dict:
    """POST /ingest — ingest plain text content."""
    payload = {"source_name": source_name, "content": content}
    if section:
        payload["section"] = section
    if clause:
        payload["clause"] = clause
    async with httpx.AsyncClient(timeout=60.0, headers=HEADERS) as client:
        resp = await client.post(f"{BASE_URL}/ingest", json=payload)
        resp.raise_for_status()
        return resp.json()


async def call_ingest_pdf(
    pdf_path: str,
    section: str | None = None,
    clause: str | None = None,
) -> dict:
    """POST /ingest/pdf — upload a PDF file for ingestion."""
    import os
    filename = os.path.basename(pdf_path)
    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()
    files = {"file": (filename, pdf_bytes, "application/pdf")}
    data = {}
    if section:
        data["section"] = section
    if clause:
        data["clause"] = clause
    async with httpx.AsyncClient(timeout=120.0, headers=HEADERS) as client:
        resp = await client.post(f"{BASE_URL}/ingest/pdf", files=files, data=data)
        resp.raise_for_status()
        return resp.json()
