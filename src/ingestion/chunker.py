"""
VeriLayer — Phase 2: Document Chunker.
Splits raw text into overlapping chunks with token counting.
"""
from __future__ import annotations
import uuid
from typing import Iterator

import structlog

from src.core.config import settings
from src.models.schemas import Chunk, DocumentMetadata

logger = structlog.get_logger("verilayer.ingestion.chunker")


def _count_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token."""
    return max(1, len(text) // 4)


def chunk_text(
    text: str,
    document_id: str,
    metadata: DocumentMetadata,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Chunk]:
    """
    Split text into overlapping chunks using a sliding window.

    Args:
        text: Raw text to chunk.
        document_id: Parent document ID.
        metadata: Document metadata applied to every chunk.
        chunk_size: Characters per chunk (defaults to settings.chunk_size).
        chunk_overlap: Overlap between chunks (defaults to settings.chunk_overlap).

    Returns:
        List of Chunk objects ready for indexing.
    """
    size = chunk_size or settings.chunk_size
    overlap = chunk_overlap or settings.chunk_overlap
    chunks: list[Chunk] = []

    text = text.strip()
    if not text:
        return chunks

    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        chunk_text_content = text[start:end].strip()

        if chunk_text_content:
            chunk = Chunk(
                chunk_id=str(uuid.uuid4()),
                document_id=document_id,
                text=chunk_text_content,
                metadata=metadata,
                token_count=_count_tokens(chunk_text_content),
            )
            chunks.append(chunk)

        if end == len(text):
            break
        start = end - overlap

    logger.info(
        "text_chunked",
        document_id=document_id,
        total_chars=len(text),
        chunks_created=len(chunks),
        chunk_size=size,
        overlap=overlap,
    )
    return chunks
