"""initial schema: documents, chunks, HNSW + BM25 indexes

Revision ID: 0001
Revises:
Create Date: 2026-10-23
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

document_status = postgresql.ENUM(
    "pending", "processing", "ready", "failed", name="document_status", create_type=False
)


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_search")
    document_status.create(op.get_bind(), checkfirst=True)

    op.create_table(
        "documents",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("source_uri", sa.Text()),
        sa.Column("mime_type", sa.Text(), nullable=False),
        sa.Column("sha256", sa.Text(), nullable=False, unique=True),
        sa.Column("page_count", sa.Integer()),
        sa.Column("status", document_status, nullable=False, server_default="pending"),
        sa.Column("error", sa.Text()),
        sa.Column("meta", postgresql.JSONB(), nullable=False, server_default="{}"),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
    )

    op.create_table(
        "chunks",
        sa.Column("id", sa.BigInteger(), sa.Identity(), primary_key=True),
        sa.Column(
            "document_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("documents.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("page_start", sa.Integer(), nullable=False),
        sa.Column("page_end", sa.Integer(), nullable=False),
        sa.Column("char_start", sa.Integer(), nullable=False),
        sa.Column("char_end", sa.Integer(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("token_count", sa.Integer(), nullable=False),
        sa.Column("embedding", Vector(384)),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.UniqueConstraint("document_id", "ordinal"),
    )
    op.create_index("ix_chunks_document_id", "chunks", ["document_id"])

    # Dense: cosine HNSW. m/ef_construction are pgvector defaults, written out so they're
    # easy to tune; ef_search is set per query.
    op.execute(
        "CREATE INDEX ix_chunks_embedding_hnsw ON chunks "
        "USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64)"
    )
    # Lexical: pg_search BM25 over chunk text, keyed by chunks.id.
    op.execute("CREATE INDEX ix_chunks_bm25 ON chunks USING bm25 (id, content)")


def downgrade() -> None:
    op.drop_table("chunks")
    op.drop_table("documents")
    document_status.drop(op.get_bind(), checkfirst=True)
