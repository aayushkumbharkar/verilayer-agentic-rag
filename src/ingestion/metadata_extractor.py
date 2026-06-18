"""
VeriLayer — Phase 2: Metadata Extractor.
Extracts source, section, and clause from document text/filename.
"""
from __future__ import annotations
import re
import uuid
from pathlib import Path

import structlog

from src.models.schemas import DocumentMetadata

logger = structlog.get_logger("verilayer.ingestion.metadata")


_SECTION_PATTERNS = [
    re.compile(r"(?i)(section|article|chapter)\s+(\d+[\.\d]*)", re.MULTILINE),
    re.compile(r"(?i)^(\d+\.\d+)\s+\w", re.MULTILINE),
]
_CLAUSE_PATTERNS = [
    re.compile(r"(?i)(clause|para|paragraph)\s+(\d+[\.\d]*)", re.MULTILINE),
    re.compile(r"(?i)^\(([a-z])\)\s+", re.MULTILINE),
]


def extract_metadata(
    source_name: str,
    content: str,
    section_hint: str | None = None,
    clause_hint: str | None = None,
) -> DocumentMetadata:
    """
    Build DocumentMetadata from a source filename and raw content.

    Tries to auto-detect section and clause from content if not provided.
    """
    document_id = str(uuid.uuid4())

    # Auto-detect section
    section = section_hint
    if not section:
        for pattern in _SECTION_PATTERNS:
            m = pattern.search(content)
            if m:
                section = m.group(0).strip()[:100]
                break

    # Auto-detect clause
    clause = clause_hint
    if not clause:
        for pattern in _CLAUSE_PATTERNS:
            m = pattern.search(content)
            if m:
                clause = m.group(0).strip()[:100]
                break

    meta = DocumentMetadata(
        document_id=document_id,
        source=Path(source_name).name,
        section=section,
        clause=clause,
    )
    logger.info(
        "metadata_extracted",
        document_id=document_id,
        source=meta.source,
        section=section,
        clause=clause,
    )
    return meta
