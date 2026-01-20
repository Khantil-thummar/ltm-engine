"""
Configuration management for LTM Engine.
Uses pydantic-settings for type-safe configuration from environment variables.
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Provider Selection
    use_ollama: bool = Field(default=False, description="Use Ollama instead of OpenAI")

    # OpenAI Configuration (required if use_ollama=False)
    openai_api_key: str = Field(default="", description="OpenAI API key")

    # Ollama Configuration (used if use_ollama=True)
    ollama_llm_model: str = Field(default="llama3.2", description="Ollama LLM model")
    ollama_embedding_model: str = Field(
        default="nomic-embed-text", description="Ollama embedding model"
    )

    # LLM Settings
    llm_model: str = Field(default="gpt-4o-mini", description="LLM model for reasoning")
    llm_temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    llm_max_tokens: int = Field(default=1024, ge=1)

    # Embedding Settings
    embedding_model: str = Field(
        default="text-embedding-3-small", description="Embedding model"
    )
    embedding_dimensions: int = Field(default=1536, description="Embedding vector size")

    # PostgreSQL Configuration
    postgres_host: str = Field(default="localhost")
    postgres_port: int = Field(default=5432)
    postgres_user: str = Field(default="ltm_user")
    postgres_password: str = Field(default="ltm_password")
    postgres_db: str = Field(default="ltm_db")

    # Qdrant Configuration
    qdrant_host: str = Field(default="localhost")
    qdrant_port: int = Field(default=6333)
    qdrant_collection_name: str = Field(default="ltm_memories")

    # Server Configuration
    server_host: str = Field(default="0.0.0.0")
    server_port: int = Field(default=8000)
    debug: bool = Field(default=False)

    # Memory Configuration
    default_top_k: int = Field(default=10, ge=1, le=100)
    decay_half_life_days: float = Field(default=30.0, gt=0)
    consolidation_min_memories: int = Field(
        default=5, ge=1, description="Minimum episodic memories required for consolidation"
    )
    consolidation_max_memories: int = Field(
        default=50, ge=1, description="Maximum episodic memories per consolidation batch"
    )
    min_confidence_threshold: float = Field(
        default=0.1, ge=0.0, le=1.0, description="Minimum confidence to keep memory"
    )

    # Multi-Agent Configuration
    default_agent_id: str = Field(default="default")
    enable_agent_isolation: bool = Field(default=True)

    # Scoring Weights (for hybrid retrieval)
    weight_semantic_similarity: float = Field(default=0.4, ge=0.0, le=1.0)
    weight_recency: float = Field(default=0.25, ge=0.0, le=1.0)
    weight_frequency: float = Field(default=0.15, ge=0.0, le=1.0)
    weight_confidence: float = Field(default=0.2, ge=0.0, le=1.0)

    @computed_field
    @property
    def postgres_dsn(self) -> str:
        """Construct PostgreSQL connection string."""
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @computed_field
    @property
    def postgres_dsn_sync(self) -> str:
        """Construct synchronous PostgreSQL connection string for Alembic."""
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
