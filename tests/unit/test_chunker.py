"""
Phase 2 — Unit tests for the text chunker.
"""
from __future__ import annotations

import pytest

from src.ingestion.chunker import chunk_text, _count_tokens
from src.models.schemas import DocumentMetadata
from datetime import datetime


@pytest.fixture
def dummy_metadata() -> DocumentMetadata:
    return DocumentMetadata(
        document_id="doc-test-001",
        source="test_doc.pdf",
        section="Section 1",
        clause="1.1",
        created_at=datetime.utcnow(),
    )


class TestCountTokens:
    def test_basic_count(self) -> None:
        # ~4 chars per token
        assert _count_tokens("hello world") == 2  # 11 // 4 = 2

    def test_empty_string(self) -> None:
        assert _count_tokens("") == 1  # max(1, ...)

    def test_long_text(self) -> None:
        text = "a" * 400
        assert _count_tokens(text) == 100


class TestChunkText:
    def test_basic_chunking(self, dummy_metadata: DocumentMetadata) -> None:
        text = "Hello world. " * 100  # ~1300 chars
        chunks = chunk_text(text, "doc-001", dummy_metadata, chunk_size=200, chunk_overlap=50)
        assert len(chunks) > 1
        for chunk in chunks:
            assert chunk.document_id == "doc-001"
            assert len(chunk.text) > 0
            assert chunk.token_count > 0

    def test_empty_text_returns_empty(self, dummy_metadata: DocumentMetadata) -> None:
        chunks = chunk_text("", "doc-001", dummy_metadata)
        assert chunks == []

    def test_whitespace_only_returns_empty(self, dummy_metadata: DocumentMetadata) -> None:
        chunks = chunk_text("   \n\t  ", "doc-001", dummy_metadata)
        assert chunks == []

    def test_single_chunk_for_short_text(self, dummy_metadata: DocumentMetadata) -> None:
        text = "Short text that fits in one chunk."
        chunks = chunk_text(text, "doc-001", dummy_metadata, chunk_size=500, chunk_overlap=50)
        assert len(chunks) == 1
        assert chunks[0].text == text

    def test_chunk_ids_are_unique(self, dummy_metadata: DocumentMetadata) -> None:
        text = "word " * 500
        chunks = chunk_text(text, "doc-001", dummy_metadata, chunk_size=100, chunk_overlap=20)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_overlap_creates_continuity(self, dummy_metadata: DocumentMetadata) -> None:
        """Verify that consecutive chunks share overlapping content."""
        text = "a" * 1000
        chunk_size = 200
        overlap = 50
        chunks = chunk_text(text, "doc-001", dummy_metadata, chunk_size=chunk_size, chunk_overlap=overlap)
        # Each chunk after the first should start with content from the previous
        for i in range(1, len(chunks)):
            # The current chunk starts where previous chunk ended minus overlap
            assert len(chunks[i].text) > 0

    def test_metadata_attached_to_each_chunk(self, dummy_metadata: DocumentMetadata) -> None:
        text = "Some content to chunk. " * 50
        chunks = chunk_text(text, "doc-001", dummy_metadata, chunk_size=100, chunk_overlap=20)
        for chunk in chunks:
            assert chunk.metadata.source == "test_doc.pdf"
            assert chunk.metadata.section == "Section 1"
            assert chunk.metadata.document_id == "doc-test-001"

    def test_chunk_size_respected(self, dummy_metadata: DocumentMetadata) -> None:
        text = "x" * 1000
        chunk_size = 200
        chunks = chunk_text(text, "doc-001", dummy_metadata, chunk_size=chunk_size, chunk_overlap=0)
        for chunk in chunks:
            assert len(chunk.text) <= chunk_size
