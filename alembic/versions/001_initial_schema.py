"""Initial schema: documents and audit_logs tables.

Revision ID: 001_initial
Revises: 
Create Date: 2026-06-22 00:00:00.000000
"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic
revision = '001_initial'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create the documents and audit_logs tables."""
    op.execute("""
    CREATE TABLE IF NOT EXISTS documents (
        document_id   TEXT PRIMARY KEY,
        source        TEXT NOT NULL,
        section       TEXT,
        clause        TEXT,
        chunk_count   INTEGER DEFAULT 0,
        created_at    TIMESTAMPTZ DEFAULT NOW()
    )
    """)

    op.execute("""
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
    )
    """)

    # Index for faster metric queries on audit_logs
    op.execute("""
    CREATE INDEX IF NOT EXISTS idx_audit_logs_status ON audit_logs (status)
    """)
    op.execute("""
    CREATE INDEX IF NOT EXISTS idx_audit_logs_created_at ON audit_logs (created_at DESC)
    """)


def downgrade() -> None:
    """Drop all tables in reverse order."""
    op.execute("DROP TABLE IF EXISTS audit_logs")
    op.execute("DROP TABLE IF EXISTS documents")
