"""
VeriLayer — Phase 2: Ingestion pipeline orchestrator.
Handles PDF + plain text ingestion end-to-end.
"""
from __future__ import annotations

import io
import structlog
from pypdf import PdfReader

from src.ingestion.chunker import chunk_text
from src.ingestion.metadata_extractor import extract_metadata
from src.ingestion.opensearch_writer import ensure_index_exists, index_chunks
from src.ingestion.postgres_writer import ensure_tables_exist, save_document_metadata
from src.models.schemas import IngestResponse

logger = structlog.get_logger("verilayer.ingestion.pipeline")


async def ingest_text(
    source_name: str,
    content: str,
    section: str | None = None,
    clause: str | None = None,
) -> IngestResponse:
    """
    Ingest raw text content:
    1. Extract metadata
    2. Chunk text
    3. Index chunks in OpenSearch
    4. Save document metadata to PostgreSQL
    """
    logger.info("ingestion_started", source=source_name, content_len=len(content))

    # Ensure storage is ready
    await ensure_index_exists()
    await ensure_tables_exist()

    # Extract metadata
    metadata = extract_metadata(source_name, content, section, clause)

    # Chunk text
    chunks = chunk_text(content, metadata.document_id, metadata)

    if not chunks:
        return IngestResponse(
            document_id=metadata.document_id,
            chunks_created=0,
            source=source_name,
            message="No content to index after chunking.",
        )

    # Index into OpenSearch (no embeddings yet — added in Phase 4)
    indexed = await index_chunks(chunks)

    # Save metadata to PostgreSQL
    await save_document_metadata(metadata, chunk_count=indexed)

    logger.info(
        "ingestion_complete",
        document_id=metadata.document_id,
        source=source_name,
        chunks=indexed,
    )
    return IngestResponse(
        document_id=metadata.document_id,
        chunks_created=indexed,
        source=source_name,
        message=f"Successfully ingested {indexed} chunks from '{source_name}'.",
    )


async def ingest_pdf(source_name: str, pdf_bytes: bytes) -> IngestResponse:
    """
    Extract text from PDF bytes and ingest.
    """
    logger.info("pdf_ingestion_started", source=source_name, size_bytes=len(pdf_bytes))
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        pages_text = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages_text.append(text)
        content = "\n\n".join(pages_text)
    except Exception as exc:
        logger.error("pdf_extraction_failed", source=source_name, error=str(exc))
        raise ValueError(f"Failed to extract text from PDF: {exc}") from exc

    return await ingest_text(source_name, content)
