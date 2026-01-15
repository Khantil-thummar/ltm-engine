"""
Consolidation Service - Summarize episodic memories into semantic memories.

Periodically consolidates episodic memories (events, conversations)
into semantic memories (facts, knowledge) using LLM summarization.
"""

import uuid
from datetime import datetime, timezone
from typing import Any

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from ltm_engine.config import Settings
from ltm_engine.models import EpisodicMemory, SemanticMemory, SemanticMemoryVersion, MemoryEvent
from ltm_engine.models.base import MemoryType, MemoryStatus, utc_now
from ltm_engine.models.events import EventType
from ltm_engine.providers import EmbeddingProvider, LLMProvider
from ltm_engine.repositories import (
    EpisodicRepository,
    SemanticRepository,
    EventRepository,
    QdrantRepository,
)
from ltm_engine.schemas.requests import ConsolidateRequest
from ltm_engine.schemas.responses import ConsolidateResponse, ConsolidationResult

logger = structlog.get_logger(__name__)


class ConsolidationService:
    """
    Service for consolidating episodic memories into semantic memories.
    
    Uses LLM to summarize groups of related episodic memories into
    distilled knowledge that is easier to retrieve and reason about.
    """

    def __init__(
        self,
        session: AsyncSession,
        qdrant: QdrantRepository,
        embedding_provider: EmbeddingProvider,
        llm_provider: LLMProvider,
        settings: Settings,
    ) -> None:
        self._session = session
        self._qdrant = qdrant
        self._embedding = embedding_provider
        self._llm = llm_provider
        self._settings = settings

        # Initialize repositories
        self._episodic_repo = EpisodicRepository(session)
        self._semantic_repo = SemanticRepository(session)
        self._event_repo = EventRepository(session)

    async def consolidate(
        self,
        request: ConsolidateRequest,
    ) -> ConsolidateResponse:
        """
        Consolidate episodic memories into semantic memories.
        
        Groups related episodic memories and uses LLM to extract
        key facts and knowledge from them.
        """
        agent_id = request.agent_id or self._settings.default_agent_id
        
        # Use config values as defaults if not specified in request
        min_memories = request.min_memories or self._settings.consolidation_min_memories
        max_memories = request.max_memories or self._settings.consolidation_max_memories

        # Get unconsolidated episodic memories
        episodic_memories = await self._episodic_repo.list_unconsolidated(
            agent_id=agent_id,
            limit=max_memories,
        )

        # Filter by time window if specified
        if request.time_window:
            if request.time_window.start:
                episodic_memories = [
                    m for m in episodic_memories
                    if m.event_timestamp >= request.time_window.start
                ]
            if request.time_window.end:
                episodic_memories = [
                    m for m in episodic_memories
                    if m.event_timestamp <= request.time_window.end
                ]

        # Filter by session if specified
        if request.session_id:
            episodic_memories = [
                m for m in episodic_memories
                if m.session_id == request.session_id
            ]

        # Check if we have enough memories to consolidate
        if len(episodic_memories) < min_memories and not request.force:
            return ConsolidateResponse(
                success=True,
                consolidated_count=0,
                results=[],
                agent_id=agent_id,
                message=f"Not enough memories to consolidate. Have {len(episodic_memories)}, need {min_memories}.",
            )

        if not episodic_memories:
            return ConsolidateResponse(
                success=True,
                consolidated_count=0,
                results=[],
                agent_id=agent_id,
                message="No unconsolidated memories found.",
            )

        # Group memories by session or theme
        grouped = await self._group_memories(episodic_memories)

        results: list[ConsolidationResult] = []

        for group_key, memories in grouped.items():
            if len(memories) < 2:
                continue

            # Summarize the group
            summary = await self._summarize_group(memories, group_key)

            # Extract subject
            subject = await self._llm.extract_subject(summary)

            # Create semantic memory
            semantic_memory = SemanticMemory(
                agent_id=agent_id,
                content=summary,
                subject=subject,
                category="consolidated",
                valid_from=memories[0].event_timestamp,
                source_episodic_ids=[str(m.id) for m in memories],
                metadata_={
                    "consolidation_group": group_key,
                    "source_count": len(memories),
                    "time_range": {
                        "start": memories[0].event_timestamp.isoformat(),
                        "end": memories[-1].event_timestamp.isoformat(),
                    },
                },
                confidence=1.0,
                importance_score=1.0,
                version=1,
            )

            semantic_memory = await self._semantic_repo.create(semantic_memory)

            # Create version
            version = SemanticMemoryVersion(
                memory_id=semantic_memory.id,
                version=1,
                content=summary,
                confidence=1.0,
                valid_from=semantic_memory.valid_from,
                metadata_snapshot=semantic_memory.metadata_,
                change_reason="Consolidated from episodic memories",
            )
            await self._semantic_repo.create_version(version)

            # Generate and store embedding
            embedding = await self._embedding.embed_text(summary)
            vector_id = str(semantic_memory.id)
            semantic_memory.vector_id = vector_id

            await self._qdrant.upsert(
                vector_id=vector_id,
                vector=embedding,
                payload={
                    "memory_id": str(semantic_memory.id),
                    "memory_type": MemoryType.SEMANTIC.value,
                    "agent_id": agent_id,
                    "content": summary,
                    "subject": subject,
                    "category": "consolidated",
                    "status": MemoryStatus.ACTIVE.value,
                    "confidence": 1.0,
                    "created_at_ts": semantic_memory.created_at.timestamp(),
                    "valid_from_ts": semantic_memory.valid_from.timestamp(),
                    "version": 1,
                },
            )

            # Mark episodic memories as consolidated
            memory_ids = [m.id for m in memories]
            await self._episodic_repo.mark_consolidated(memory_ids, semantic_memory.id)

            # Record consolidation event
            await self._record_consolidation_event(
                semantic_memory=semantic_memory,
                source_memories=memories,
                agent_id=agent_id,
            )

            results.append(
                ConsolidationResult(
                    semantic_memory_id=semantic_memory.id,
                    subject=subject,
                    summary=summary,
                    source_episodic_count=len(memories),
                    source_episodic_ids=[str(m.id) for m in memories],
                )
            )

        await self._session.commit()

        logger.info(
            "Consolidated memories",
            agent_id=agent_id,
            consolidated_count=len(results),
            total_episodic=sum(r.source_episodic_count for r in results),
        )

        return ConsolidateResponse(
            success=True,
            consolidated_count=len(results),
            results=results,
            agent_id=agent_id,
            message=f"Successfully consolidated {len(results)} groups.",
        )

    async def _group_memories(
        self,
        memories: list[EpisodicMemory],
    ) -> dict[str, list[EpisodicMemory]]:
        """
        Group episodic memories by session or theme.
        
        Uses session_id if available, otherwise groups by time proximity.
        """
        groups: dict[str, list[EpisodicMemory]] = {}

        # First, group by session_id if available
        session_groups: dict[str, list[EpisodicMemory]] = {}
        no_session: list[EpisodicMemory] = []

        for memory in memories:
            if memory.session_id:
                if memory.session_id not in session_groups:
                    session_groups[memory.session_id] = []
                session_groups[memory.session_id].append(memory)
            else:
                no_session.append(memory)

        # Add session groups
        for session_id, session_memories in session_groups.items():
            groups[f"session:{session_id}"] = sorted(
                session_memories, key=lambda m: m.event_timestamp
            )

        # Group remaining memories by time window (1 hour blocks)
        if no_session:
            time_groups: dict[str, list[EpisodicMemory]] = {}
            for memory in no_session:
                # Round to nearest hour
                hour_key = memory.event_timestamp.replace(
                    minute=0, second=0, microsecond=0
                ).isoformat()
                if hour_key not in time_groups:
                    time_groups[hour_key] = []
                time_groups[hour_key].append(memory)

            for time_key, time_memories in time_groups.items():
                groups[f"time:{time_key}"] = sorted(
                    time_memories, key=lambda m: m.event_timestamp
                )

        return groups

    async def _summarize_group(
        self,
        memories: list[EpisodicMemory],
        group_key: str,
    ) -> str:
        """Summarize a group of episodic memories using LLM."""
        contents = [m.content for m in memories]
        context = f"Group: {group_key}, Time range: {memories[0].event_timestamp} to {memories[-1].event_timestamp}"

        summary = await self._llm.summarize(contents, context)
        return summary

    async def _record_consolidation_event(
        self,
        semantic_memory: SemanticMemory,
        source_memories: list[EpisodicMemory],
        agent_id: str,
    ) -> None:
        """Record a consolidation event for replay."""
        event = MemoryEvent(
            event_type=EventType.MEMORY_CONSOLIDATED.value,
            memory_type=MemoryType.SEMANTIC.value,
            memory_id=semantic_memory.id,
            agent_id=agent_id,
            payload_before=None,
            payload_after={
                "semantic_memory_id": str(semantic_memory.id),
                "content": semantic_memory.content,
                "subject": semantic_memory.subject,
                "source_episodic_ids": [str(m.id) for m in source_memories],
            },
            event_metadata={
                "source_count": len(source_memories),
            },
            actor="consolidation_service",
        )
        await self._event_repo.create(event)
