"""
VeriLayer — Centralized configuration via pydantic-settings.
All values sourced from environment variables / .env file.
No hardcoded values anywhere in the codebase.
"""
from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ── App ──────────────────────────────────────────────────────────────────
    app_name: str = Field(default="VeriLayer")
    app_version: str = Field(default="0.1.0")
    debug: bool = Field(default=False)

    # ── PostgreSQL ────────────────────────────────────────────────────────────
    postgres_host: str = Field(default="localhost")
    postgres_port: int = Field(default=5432)
    postgres_db: str = Field(default="verilayer")
    postgres_user: str = Field(default="verilayer")
    postgres_password: str = Field(default="verilayer_secret")

    @property
    def postgres_dsn(self) -> str:
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def postgres_async_dsn(self) -> str:
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    # ── OpenSearch ────────────────────────────────────────────────────────────
    opensearch_host: str = Field(default="localhost")
    opensearch_port: int = Field(default=9200)
    opensearch_index: str = Field(default="verilayer-docs")

    # ── Redis ─────────────────────────────────────────────────────────────────
    redis_host: str = Field(default="localhost")
    redis_port: int = Field(default=6379)
    redis_ttl: int = Field(default=3600)

    # ── Groq LLM ──────────────────────────────────────────────────────────────
    groq_api_key: str = Field(default="")
    groq_model: str = Field(default="openai/gpt-oss-20b")
    groq_timeout: int = Field(default=30)
    groq_max_retries: int = Field(default=3)

    # ── Jina Embeddings ───────────────────────────────────────────────────────
    jina_api_key: str = Field(default="")
    jina_embed_model: str = Field(default="jina-embeddings-v3")
    jina_embed_dimensions: int = Field(default=1024)

    # ── Langfuse ──────────────────────────────────────────────────────────────
    langfuse_public_key: str = Field(default="")
    langfuse_secret_key: str = Field(default="")
    langfuse_host: str = Field(default="https://cloud.langfuse.com")

    # ── Agentic Pipeline ──────────────────────────────────────────────────────
    confidence_verified_threshold: float = Field(default=0.8)
    confidence_partial_threshold: float = Field(default=0.5)
    max_retries: int = Field(default=2)
    max_retrieval_docs: int = Field(default=5)
    chunk_size: int = Field(default=512)
    chunk_overlap: int = Field(default=64)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached singleton Settings instance."""
    return Settings()


# Module-level singleton for convenience imports
settings: Settings = get_settings()
