"""
VeriLayer — Phase 2: Ingestion pipeline orchestrator.
Handles PDF + plain text ingestion end-to-end.
"""
from __future__ import annotations

import io
import math
import structlog
from pypdf import PdfReader

from src.ingestion.chunker import chunk_text
from src.ingestion.metadata_extractor import extract_metadata
from src.ingestion.opensearch_writer import ensure_index_exists, index_chunks
from src.ingestion.postgres_writer import ensure_tables_exist, save_document_metadata
from src.models.schemas import IngestResponse
from src.core.config import settings

logger = structlog.get_logger("verilayer.ingestion.pipeline")

EMBED_BATCH_SIZE = 64  # Jina API max batch size


async def _embed_chunks(chunks) -> None:
    """
    Generate and attach Jina embeddings to each chunk in-place.
    Silently skips if JINA_API_KEY is not configured.
    """
    if not settings.jina_api_key:
        logger.warning("jina_key_missing_skipping_embeddings")
        return
    try:
        from src.retrieval.embeddings import embed_texts
        texts = [c.text for c in chunks]
        # Process in batches to stay within API limits
        for i in range(0, len(texts), EMBED_BATCH_SIZE):
            batch_texts = texts[i : i + EMBED_BATCH_SIZE]
            batch_chunks = chunks[i : i + EMBED_BATCH_SIZE]
            vectors = await embed_texts(batch_texts)
            for chunk, vec in zip(batch_chunks, vectors):
                chunk.embedding = vec
        logger.info("embeddings_attached", count=len(chunks))
    except Exception as exc:
        logger.warning("embedding_failed_continuing_without", error=str(exc))


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

    # Generate embeddings (attaches vectors in-place; no-op if key missing)
    await _embed_chunks(chunks)

    # Index into OpenSearch
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
