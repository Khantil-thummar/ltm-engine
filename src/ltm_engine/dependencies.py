"""
Dependency injection for LTM Engine.

Provides FastAPI dependencies for database sessions, repositories, and services.
"""

from typing import AsyncGenerator

import structlog
from fastapi import Depends

from ltm_engine.config import Settings, get_settings
from ltm_engine.providers import (
    OpenAIEmbeddingProvider,
    OpenAILLMProvider,
    OllamaEmbeddingProvider,
    OllamaLLMProvider,
)
from ltm_engine.providers.base import EmbeddingProvider, LLMProvider
from ltm_engine.repositories import PostgresRepository, QdrantRepository
from ltm_engine.services import (
    MemoryService,
    RetrievalService,
    ConsolidationService,
    LifecycleService,
    ConflictService,
    ConfidenceService,
    ReplayService,
)

logger = structlog.get_logger(__name__)

# Global instances (initialized at startup)
_postgres_repo: PostgresRepository | None = None
_qdrant_repo: QdrantRepository | None = None
_embedding_provider: EmbeddingProvider | None = None
_llm_provider: LLMProvider | None = None


async def init_dependencies(settings: Settings) -> None:
    """Initialize global dependencies at application startup."""
    global _postgres_repo, _qdrant_repo, _embedding_provider, _llm_provider

    logger.info("Initializing dependencies...")

    # Initialize PostgreSQL
    _postgres_repo = PostgresRepository(settings)
    await _postgres_repo.init_db()

    # Initialize Qdrant
    _qdrant_repo = QdrantRepository(settings)
    await _qdrant_repo.init_collection()

    # Initialize providers based on configuration
    if settings.use_ollama:
        logger.info("Using Ollama providers")
        _embedding_provider = OllamaEmbeddingProvider(settings)
        _llm_provider = OllamaLLMProvider(settings)
    else:
        logger.info("Using OpenAI providers")
        _embedding_provider = OpenAIEmbeddingProvider(settings)
        _llm_provider = OpenAILLMProvider(settings)

    logger.info("Dependencies initialized successfully")


async def close_dependencies() -> None:
    """Close global dependencies at application shutdown."""
    global _postgres_repo, _qdrant_repo

    logger.info("Closing dependencies...")

    if _postgres_repo:
        await _postgres_repo.close()
    if _qdrant_repo:
        await _qdrant_repo.close()

    logger.info("Dependencies closed")


def get_postgres_repo() -> PostgresRepository:
    """Get PostgreSQL repository."""
    if _postgres_repo is None:
        raise RuntimeError("PostgreSQL repository not initialized")
    return _postgres_repo


def get_qdrant_repo() -> QdrantRepository:
    """Get Qdrant repository."""
    if _qdrant_repo is None:
        raise RuntimeError("Qdrant repository not initialized")
    return _qdrant_repo


def get_embedding_provider() -> EmbeddingProvider:
    """Get embedding provider."""
    if _embedding_provider is None:
        raise RuntimeError("Embedding provider not initialized")
    return _embedding_provider


def get_llm_provider() -> LLMProvider:
    """Get LLM provider."""
    if _llm_provider is None:
        raise RuntimeError("LLM provider not initialized")
    return _llm_provider


async def get_db_session() -> AsyncGenerator:
    """Get database session for request."""
    postgres = get_postgres_repo()
    async with postgres.session() as session:
        try:
            yield session
        finally:
            await session.close()


async def get_memory_service(
    session=Depends(get_db_session),
    settings: Settings = Depends(get_settings),
) -> MemoryService:
    """Get memory service."""
    return MemoryService(
        session=session,
        qdrant=get_qdrant_repo(),
        embedding_provider=get_embedding_provider(),
        llm_provider=get_llm_provider(),
        settings=settings,
    )


async def get_retrieval_service(
    session=Depends(get_db_session),
    settings: Settings = Depends(get_settings),
) -> RetrievalService:
    """Get retrieval service."""
    return RetrievalService(
        session=session,
        qdrant=get_qdrant_repo(),
        embedding_provider=get_embedding_provider(),
        settings=settings,
    )


async def get_consolidation_service(
    session=Depends(get_db_session),
    settings: Settings = Depends(get_settings),
) -> ConsolidationService:
    """Get consolidation service."""
    return ConsolidationService(
        session=session,
        qdrant=get_qdrant_repo(),
        embedding_provider=get_embedding_provider(),
        llm_provider=get_llm_provider(),
        settings=settings,
    )


async def get_lifecycle_service(
    session=Depends(get_db_session),
    settings: Settings = Depends(get_settings),
) -> LifecycleService:
    """Get lifecycle service."""
    return LifecycleService(
        session=session,
        qdrant=get_qdrant_repo(),
        settings=settings,
    )


async def get_conflict_service(
    session=Depends(get_db_session),
    settings: Settings = Depends(get_settings),
) -> ConflictService:
    """Get conflict service."""
    return ConflictService(
        session=session,
        qdrant=get_qdrant_repo(),
        embedding_provider=get_embedding_provider(),
        llm_provider=get_llm_provider(),
        settings=settings,
    )


async def get_confidence_service(
    session=Depends(get_db_session),
    settings: Settings = Depends(get_settings),
) -> ConfidenceService:
    """Get confidence service."""
    return ConfidenceService(
        session=session,
        qdrant=get_qdrant_repo(),
        settings=settings,
    )


async def get_replay_service(
    session=Depends(get_db_session),
    settings: Settings = Depends(get_settings),
) -> ReplayService:
    """Get replay service."""
    return ReplayService(
        session=session,
        settings=settings,
    )
