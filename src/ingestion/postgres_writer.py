"""
VeriLayer — Phase 2: PostgreSQL metadata writer.
Stores document-level metadata in a relational table for audit + querying.
"""
from __future__ import annotations

import structlog
import asyncpg

from src.core.config import settings
from src.models.schemas import DocumentMetadata

logger = structlog.get_logger("verilayer.ingestion.postgres")

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS documents (
    document_id   TEXT PRIMARY KEY,
    source        TEXT NOT NULL,
    section       TEXT,
    clause        TEXT,
    chunk_count   INTEGER DEFAULT 0,
    created_at    TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS audit_logs (
    id              BIGSERIAL PRIMARY KEY,
    query           TEXT NOT NULL,
    final_answer    TEXT,
    status          TEXT,
    confidence      FLOAT,
    claims_json     JSONB,
    verdicts_json   JSONB,
    retries         INTEGER DEFAULT 0,
    latency_ms      INTEGER DEFAULT 0,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);
"""


async def _get_conn() -> asyncpg.Connection:
    return await asyncpg.connect(
        host=settings.postgres_host,
        port=settings.postgres_port,
        database=settings.postgres_db,
        user=settings.postgres_user,
        password=settings.postgres_password,
    )


async def ensure_tables_exist() -> None:
    """Create documents and audit_logs tables if they don't exist."""
    conn = await _get_conn()
    try:
        await conn.execute(CREATE_TABLE_SQL)
        logger.info("postgres_tables_ready")
    finally:
        await conn.close()


async def save_document_metadata(metadata: DocumentMetadata, chunk_count: int) -> None:
    """Insert or update a document metadata row."""
    conn = await _get_conn()
    try:
        await conn.execute(
            """
            INSERT INTO documents (document_id, source, section, clause, chunk_count, created_at)
            VALUES ($1, $2, $3, $4, $5, $6)
            ON CONFLICT (document_id) DO UPDATE
                SET chunk_count = EXCLUDED.chunk_count,
                    source = EXCLUDED.source
            """,
            metadata.document_id,
            metadata.source,
            metadata.section,
            metadata.clause,
            chunk_count,
            metadata.created_at,
        )
        logger.info(
            "document_metadata_saved",
            document_id=metadata.document_id,
            source=metadata.source,
            chunk_count=chunk_count,
        )
    finally:
        await conn.close()


async def save_audit_log(
    query: str,
    final_answer: str,
    status: str,
    confidence: float,
    claims: list[dict],
    retries: int,
    latency_ms: int,
) -> None:
    """Write a full query audit log entry."""
    import json

    conn = await _get_conn()
    try:
        verdicts = [c.get("verdict") for c in claims]
        await conn.execute(
            """
            INSERT INTO audit_logs
                (query, final_answer, status, confidence, claims_json, verdicts_json, retries, latency_ms)
            VALUES ($1, $2, $3, $4, $5::jsonb, $6::jsonb, $7, $8)
            """,
            query,
            final_answer,
            status,
            confidence,
            json.dumps(claims),
            json.dumps(verdicts),
            retries,
            latency_ms,
        )
        logger.info(
            "audit_log_saved",
            status=status,
            confidence=confidence,
            retries=retries,
            latency_ms=latency_ms,
        )
    finally:
        await conn.close()
