"""
VeriLayer — Phase 2: Ingest API route.
POST /ingest — accepts text body or PDF file upload.
"""
from __future__ import annotations

import structlog
from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status

from src.ingestion.pipeline import ingest_pdf, ingest_text
from src.models.schemas import IngestRequest, IngestResponse

logger = structlog.get_logger("verilayer.api.ingest")
router = APIRouter(prefix="/ingest", tags=["Ingestion"])


@router.post(
    "",
    response_model=IngestResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Ingest a text document",
)
async def ingest_text_endpoint(request: IngestRequest) -> IngestResponse:
    """Ingest plain text content directly."""
    try:
        return await ingest_text(
            source_name=request.source_name,
            content=request.content,
            section=request.section,
            clause=request.clause,
        )
    except Exception as exc:
        logger.error("ingest_text_failed", error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post(
    "/pdf",
    response_model=IngestResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Ingest a PDF file",
)
async def ingest_pdf_endpoint(
    file: UploadFile = File(..., description="PDF file to ingest"),
    section: str | None = Form(default=None),
    clause: str | None = Form(default=None),
) -> IngestResponse:
    """Upload and ingest a PDF document."""
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    pdf_bytes = await file.read()
    if len(pdf_bytes) > 50 * 1024 * 1024:  # 50MB limit
        raise HTTPException(status_code=413, detail="PDF too large (max 50MB).")

    try:
        return await ingest_pdf(source_name=file.filename, pdf_bytes=pdf_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("ingest_pdf_failed", error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc)) from exc
