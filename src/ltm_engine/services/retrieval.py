"""
Retrieval Service - Hybrid memory retrieval with scoring.

Implements hybrid ranking combining:
- Semantic similarity
- Recency decay
- Access frequency
- Confidence score
"""

import uuid
from datetime import datetime, timezone
from typing import Any

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from ltm_engine.config import Settings
from ltm_engine.models import EpisodicMemory, SemanticMemory, ProceduralMemory
from ltm_engine.models.base import MemoryType, MemoryStatus
from ltm_engine.providers import EmbeddingProvider
from ltm_engine.repositories import (
    EpisodicRepository,
    SemanticRepository,
    ProceduralRepository,
    QdrantRepository,
)
from ltm_engine.schemas.requests import RetrieveRequest, RetrieveFilters, TimeWindow
from ltm_engine.schemas.responses import RetrievedMemory, RetrieveResponse
from ltm_engine.utils.scoring import (
    calculate_recency_score,
    calculate_frequency_score,
    calculate_combined_score,
)

logger = structlog.get_logger(__name__)


class RetrievalService:
    """
    Hybrid retrieval service with configurable scoring.
    
    Combines vector similarity search with metadata-based scoring
    to provide relevance-ranked memory retrieval.
    """

    def __init__(
        self,
        session: AsyncSession,
        qdrant: QdrantRepository,
        embedding_provider: EmbeddingProvider,
        settings: Settings,
    ) -> None:
        self._session = session
        self._qdrant = qdrant
        self._embedding = embedding_provider
        self._settings = settings

        # Initialize repositories
        self._episodic_repo = EpisodicRepository(session)
        self._semantic_repo = SemanticRepository(session)
        self._procedural_repo = ProceduralRepository(session)

    async def retrieve(
        self,
        request: RetrieveRequest,
        agent_id: str | None = None,
    ) -> RetrieveResponse:
        """
        Retrieve memories matching the query with hybrid ranking.
        
        The ranking combines:
        1. Semantic similarity from vector search
        2. Recency decay based on creation time
        3. Access frequency (log-scaled)
        4. Confidence score from the memory
        """
        import time
        start_time = time.time()

        agent_id = agent_id or request.agent_id or self._settings.default_agent_id

        # Get scoring weights (use request overrides or defaults)
        weight_semantic = request.weight_semantic or self._settings.weight_semantic_similarity
        weight_recency = request.weight_recency or self._settings.weight_recency
        weight_frequency = request.weight_frequency or self._settings.weight_frequency
        weight_confidence = request.weight_confidence or self._settings.weight_confidence

        # Generate query embedding
        query_embedding = await self._embedding.embed_text(request.query)

        # Build filters for Qdrant
        memory_types = None
        if request.filters.memory_types:
            memory_types = request.filters.memory_types

        status_filter = [MemoryStatus.ACTIVE.value]
        if request.filters.include_superseded:
            status_filter.append(MemoryStatus.SUPERSEDED.value)
        if request.filters.status:
            status_filter = request.filters.status

        # Time window handling
        time_start = None
        time_end = None
        if request.time_window:
            time_start = request.time_window.start
            time_end = request.time_window.end

        # Perform vector search
        # Retrieve more than needed to allow for re-ranking
        search_limit = min(request.top_k * 3, 100)
        
        vector_results = await self._qdrant.search(
            query_vector=query_embedding,
            top_k=search_limit,
            agent_id=agent_id,
            memory_types=memory_types,
            status=status_filter,
            min_confidence=request.filters.min_confidence,
            time_start=time_start,
            time_end=time_end,
        )

        if not vector_results:
            return RetrieveResponse(
                query=request.query,
                total_found=0,
                memories=[],
                agent_id=agent_id,
                as_of=request.as_of,
                execution_time_ms=(time.time() - start_time) * 1000,
            )

        # Enrich results with full memory data and calculate scores
        enriched_results = await self._enrich_and_score(
            vector_results=vector_results,
            agent_id=agent_id,
            weight_semantic=weight_semantic,
            weight_recency=weight_recency,
            weight_frequency=weight_frequency,
            weight_confidence=weight_confidence,
            as_of=request.as_of,
        )

        # Apply additional filters
        filtered_results = self._apply_filters(
            results=enriched_results,
            filters=request.filters,
        )

        # Sort by combined score and take top_k
        sorted_results = sorted(
            filtered_results,
            key=lambda x: x.score,
            reverse=True,
        )[:request.top_k]

        # Update access counts for retrieved memories
        await self._update_access_counts(sorted_results)
        await self._session.commit()

        execution_time_ms = (time.time() - start_time) * 1000

        logger.info(
            "Retrieved memories",
            query=request.query[:50],
            agent_id=agent_id,
            total_found=len(sorted_results),
            execution_time_ms=execution_time_ms,
        )

        return RetrieveResponse(
            query=request.query,
            total_found=len(sorted_results),
            memories=sorted_results,
            agent_id=agent_id,
            as_of=request.as_of,
            execution_time_ms=execution_time_ms,
        )

    async def _enrich_and_score(
        self,
        vector_results: list[dict[str, Any]],
        agent_id: str,
        weight_semantic: float,
        weight_recency: float,
        weight_frequency: float,
        weight_confidence: float,
        as_of: datetime | None = None,
    ) -> list[RetrievedMemory]:
        """Enrich vector results with full data and calculate combined scores."""
        reference_time = as_of or datetime.now(timezone.utc)
        enriched = []

        # Group by memory type for batch retrieval
        episodic_ids = []
        semantic_ids = []
        procedural_ids = []

        for result in vector_results:
            payload = result["payload"]
            memory_type = payload.get("memory_type")
            memory_id = payload.get("memory_id")

            if memory_type == MemoryType.EPISODIC.value:
                episodic_ids.append(uuid.UUID(memory_id))
            elif memory_type == MemoryType.SEMANTIC.value:
                semantic_ids.append(uuid.UUID(memory_id))
            elif memory_type == MemoryType.PROCEDURAL.value:
                procedural_ids.append(uuid.UUID(memory_id))

        # Batch retrieve memories
        episodic_memories = {
            m.id: m for m in await self._episodic_repo.get_by_ids(episodic_ids, agent_id)
        }
        semantic_memories = {
            m.id: m for m in await self._semantic_repo.get_by_ids(semantic_ids, agent_id)
        }
        procedural_memories = {
            m.id: m for m in await self._procedural_repo.get_by_ids(procedural_ids, agent_id)
        }

        # Calculate scores and build results
        for result in vector_results:
            payload = result["payload"]
            memory_type = payload.get("memory_type")
            memory_id_str = payload.get("memory_id")
            memory_id = uuid.UUID(memory_id_str)
            semantic_score = result["score"]

            # Get the full memory object
            memory = None
            if memory_type == MemoryType.EPISODIC.value:
                memory = episodic_memories.get(memory_id)
            elif memory_type == MemoryType.SEMANTIC.value:
                memory = semantic_memories.get(memory_id)
            elif memory_type == MemoryType.PROCEDURAL.value:
                memory = procedural_memories.get(memory_id)

            if not memory:
                continue

            # Calculate component scores
            recency_score = calculate_recency_score(
                created_at=memory.created_at,
                half_life_days=self._settings.decay_half_life_days,
                reference_time=reference_time,
            )

            access_count = getattr(memory, "access_count", 0)
            frequency_score = calculate_frequency_score(access_count)

            confidence_score = getattr(memory, "confidence", 1.0)

            # Calculate combined score
            combined_score = calculate_combined_score(
                semantic_score=semantic_score,
                recency_score=recency_score,
                frequency_score=frequency_score,
                confidence_score=confidence_score,
                weight_semantic=weight_semantic,
                weight_recency=weight_recency,
                weight_frequency=weight_frequency,
                weight_confidence=weight_confidence,
            )

            # Build retrieved memory response
            retrieved = RetrievedMemory(
                memory_id=memory_id,
                memory_type=memory_type,
                content=memory.content if hasattr(memory, "content") else str(getattr(memory, "value", "")),
                score=combined_score,
                semantic_score=semantic_score,
                recency_score=recency_score,
                frequency_score=frequency_score,
                confidence_score=confidence_score,
                metadata=getattr(memory, "metadata_", {}),
                created_at=memory.created_at,
            )

            # Add type-specific fields
            if memory_type == MemoryType.EPISODIC.value:
                retrieved.event_timestamp = memory.event_timestamp
                retrieved.source = memory.source
                retrieved.session_id = memory.session_id
            elif memory_type == MemoryType.SEMANTIC.value:
                retrieved.subject = memory.subject
                retrieved.category = memory.category
                retrieved.version = memory.version
                retrieved.valid_from = memory.valid_from
                retrieved.valid_until = memory.valid_until
            elif memory_type == MemoryType.PROCEDURAL.value:
                retrieved.key = memory.key
                retrieved.category = memory.category

            enriched.append(retrieved)

        return enriched

    def _apply_filters(
        self,
        results: list[RetrievedMemory],
        filters: RetrieveFilters,
    ) -> list[RetrievedMemory]:
        """Apply additional filters that couldn't be done in vector search."""
        filtered = results

        # Filter by categories
        if filters.categories:
            filtered = [r for r in filtered if r.category in filters.categories]

        # Filter by subjects
        if filters.subjects:
            filtered = [
                r for r in filtered
                if r.subject and any(s.lower() in r.subject.lower() for s in filters.subjects)
            ]

        # Filter by sources
        if filters.sources:
            filtered = [r for r in filtered if r.source in filters.sources]

        # Filter by session_id (for episodic memories)
        if filters.session_id:
            filtered = [
                r for r in filtered
                if r.session_id == filters.session_id
            ]

        return filtered

    async def _update_access_counts(
        self,
        results: list[RetrievedMemory],
    ) -> None:
        """Update access counts for retrieved memories."""
        for result in results:
            if result.memory_type == MemoryType.EPISODIC.value:
                await self._episodic_repo.increment_access(result.memory_id)
            elif result.memory_type == MemoryType.SEMANTIC.value:
                await self._semantic_repo.increment_access(result.memory_id)
            elif result.memory_type == MemoryType.PROCEDURAL.value:
                await self._procedural_repo.reinforce(result.memory_id)

    async def search_by_subject(
        self,
        subject: str,
        agent_id: str,
        include_superseded: bool = False,
    ) -> list[SemanticMemory]:
        """Search semantic memories by subject."""
        return list(await self._semantic_repo.get_by_subject(
            agent_id=agent_id,
            subject=subject,
            include_superseded=include_superseded,
        ))
