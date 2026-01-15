"""
Main Memory Service - Orchestrates all memory operations.

This is the primary service interface that coordinates between:
- Storage repositories (PostgreSQL, Qdrant)
- Embedding/LLM providers
- Specialized services (retrieval, consolidation, lifecycle, conflict)
"""

import uuid
from datetime import datetime, timezone
from typing import Any

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from ltm_engine.config import Settings
from ltm_engine.models import EpisodicMemory, SemanticMemory, SemanticMemoryVersion, ProceduralMemory, MemoryEvent
from ltm_engine.models.base import MemoryType, MemoryStatus, utc_now
from ltm_engine.models.events import EventType
from ltm_engine.providers import EmbeddingProvider, LLMProvider
from ltm_engine.repositories import (
    EpisodicRepository,
    SemanticRepository,
    ProceduralRepository,
    EventRepository,
    QdrantRepository,
)
from ltm_engine.schemas.memory import (
    EpisodicMemoryCreate,
    SemanticMemoryCreate,
    SemanticMemoryUpdate,
    ProceduralMemoryCreate,
    ProceduralMemoryUpdate,
)

logger = structlog.get_logger(__name__)


class MemoryService:
    """
    Main memory service orchestrating all memory operations.
    
    Provides high-level operations for:
    - Storing memories (with automatic embedding and conflict detection)
    - Updating memories (with versioning)
    - Basic CRUD operations
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
        self._procedural_repo = ProceduralRepository(session)
        self._event_repo = EventRepository(session)

    # =========================================================================
    # Episodic Memory Operations
    # =========================================================================

    async def store_episodic(
        self,
        data: EpisodicMemoryCreate,
        agent_id: str | None = None,
    ) -> EpisodicMemory:
        """
        Store a new episodic memory.
        
        Episodic memories are immutable (append-only) and represent
        specific events or interactions.
        """
        agent_id = agent_id or data.agent_id or self._settings.default_agent_id
        event_timestamp = data.event_timestamp or utc_now()

        # Create memory record
        memory = EpisodicMemory(
            agent_id=agent_id,
            content=data.content,
            event_timestamp=event_timestamp,
            source=data.source,
            session_id=data.session_id,
            actor=data.actor,
            action_type=data.action_type,
            entities=data.entities,
            metadata_=data.metadata,
            confidence=1.0,  # Episodic memories start with full confidence
            importance_score=1.0,
        )

        # Generate embedding
        embedding = await self._embedding.embed_text(data.content)
        
        # Store in database
        memory = await self._episodic_repo.create(memory)
        
        # Store vector in Qdrant
        vector_id = str(memory.id)
        memory.vector_id = vector_id
        
        await self._qdrant.upsert(
            vector_id=vector_id,
            vector=embedding,
            payload={
                "memory_id": str(memory.id),
                "memory_type": MemoryType.EPISODIC.value,
                "agent_id": agent_id,
                "content": data.content,
                "status": MemoryStatus.ACTIVE.value,
                "confidence": memory.confidence,
                "created_at_ts": memory.created_at.timestamp(),
                "event_timestamp_ts": event_timestamp.timestamp(),
                "source": data.source,
                "session_id": data.session_id,
            },
        )

        # Record event for replay
        await self._record_event(
            event_type=EventType.EPISODIC_CREATED,
            memory_type=MemoryType.EPISODIC.value,
            memory_id=memory.id,
            agent_id=agent_id,
            payload_after=self._memory_to_dict(memory),
        )

        await self._session.commit()
        
        logger.info(
            "Stored episodic memory",
            memory_id=str(memory.id),
            agent_id=agent_id,
        )
        
        return memory

    async def get_episodic(
        self,
        memory_id: uuid.UUID,
        agent_id: str | None = None,
    ) -> EpisodicMemory | None:
        """Get an episodic memory by ID."""
        return await self._episodic_repo.get_by_id(memory_id, agent_id)

    # =========================================================================
    # Semantic Memory Operations
    # =========================================================================

    async def store_semantic(
        self,
        data: SemanticMemoryCreate,
        agent_id: str | None = None,
        check_conflicts: bool = True,
    ) -> tuple[SemanticMemory, dict[str, Any] | None]:
        """
        Store a new semantic memory.
        
        Semantic memories represent facts and can be updated/versioned.
        Returns (memory, conflict_info) where conflict_info is set if
        a conflict was detected.
        """
        agent_id = agent_id or data.agent_id or self._settings.default_agent_id
        valid_from = data.valid_from or utc_now()

        conflict_info = None

        # Generate embedding
        embedding = await self._embedding.embed_text(data.content)

        # Check for conflicts with existing semantic memories
        if check_conflicts:
            conflict_info = await self._check_semantic_conflict(
                content=data.content,
                embedding=embedding,
                agent_id=agent_id,
                subject=data.subject,
            )

        # Skip duplicates: if very similar (>=98%) and not a contradiction, return existing
        if conflict_info and not conflict_info.get("is_contradiction", False):
            similarity = conflict_info.get("similarity_score", 0)
            if similarity >= 0.98:
                # It's a duplicate - return existing memory instead
                existing_id = conflict_info.get("existing_memory_id")
                if existing_id:
                    existing_memory = await self._semantic_repo.get_by_id(
                        uuid.UUID(existing_id) if isinstance(existing_id, str) else existing_id
                    )
                    if existing_memory:
                        # Increment access count as reinforcement
                        existing_memory.access_count += 1
                        existing_memory.last_accessed_at = utc_now()
                        await self._session.commit()
                        # Return existing memory, no conflict
                        return existing_memory, None

        # Create memory record
        memory = SemanticMemory(
            agent_id=agent_id,
            content=data.content,
            subject=data.subject,
            category=data.category,
            valid_from=valid_from,
            source_episodic_ids=data.source_episodic_ids,
            related_memory_ids=data.related_memory_ids,
            metadata_=data.metadata,
            confidence=data.confidence,
            importance_score=1.0,
            version=1,
        )

        # Handle conflict resolution
        if conflict_info and conflict_info.get("is_contradiction"):
            memory.conflict_metadata = {
                "detected_at": utc_now().isoformat(),
                "existing_memory_id": conflict_info.get("existing_memory_id"),
                "resolution": conflict_info.get("resolution", "supersede"),
                "explanation": conflict_info.get("llm_analysis"),
            }

        # Store in database
        memory = await self._semantic_repo.create(memory)

        # Create initial version
        version = SemanticMemoryVersion(
            memory_id=memory.id,
            version=1,
            content=data.content,
            confidence=data.confidence,
            valid_from=valid_from,
            metadata_snapshot=data.metadata,
            change_reason="Initial creation",
        )
        await self._semantic_repo.create_version(version)

        # Store vector in Qdrant
        vector_id = str(memory.id)
        memory.vector_id = vector_id

        await self._qdrant.upsert(
            vector_id=vector_id,
            vector=embedding,
            payload={
                "memory_id": str(memory.id),
                "memory_type": MemoryType.SEMANTIC.value,
                "agent_id": agent_id,
                "content": data.content,
                "subject": data.subject,
                "category": data.category,
                "status": MemoryStatus.ACTIVE.value,
                "confidence": memory.confidence,
                "created_at_ts": memory.created_at.timestamp(),
                "valid_from_ts": valid_from.timestamp(),
                "version": 1,
            },
        )

        # Handle superseding if conflict detected
        if conflict_info and conflict_info.get("resolution") == "supersede":
            existing_id = conflict_info.get("existing_memory_id")
            if existing_id:
                await self._semantic_repo.supersede(
                    uuid.UUID(existing_id),
                    memory.id,
                )

        # Record event
        await self._record_event(
            event_type=EventType.SEMANTIC_CREATED,
            memory_type=MemoryType.SEMANTIC.value,
            memory_id=memory.id,
            agent_id=agent_id,
            payload_after=self._memory_to_dict(memory),
            event_metadata={"conflict_info": conflict_info} if conflict_info else None,
        )

        await self._session.commit()

        logger.info(
            "Stored semantic memory",
            memory_id=str(memory.id),
            agent_id=agent_id,
            subject=data.subject,
            has_conflict=bool(conflict_info),
        )

        return memory, conflict_info

    async def update_semantic(
        self,
        memory_id: uuid.UUID,
        data: SemanticMemoryUpdate,
        agent_id: str | None = None,
    ) -> SemanticMemory | None:
        """
        Update a semantic memory, creating a new version.
        
        The old version is preserved for temporal queries.
        """
        memory = await self._semantic_repo.get_by_id(memory_id, agent_id)
        if not memory:
            return None

        old_content = memory.content
        valid_from = data.valid_from or utc_now()

        # Close the previous version's validity
        if memory.versions:
            latest_version = memory.versions[0]
            latest_version.valid_until = valid_from

        # Create new version
        new_version = SemanticMemoryVersion(
            memory_id=memory.id,
            version=memory.version + 1,
            content=data.content,
            confidence=data.confidence or memory.confidence,
            valid_from=valid_from,
            metadata_snapshot=data.metadata or memory.metadata_,
            change_reason=data.change_reason,
        )
        await self._semantic_repo.create_version(new_version)

        # Update memory
        memory.content = data.content
        memory.version += 1
        memory.valid_from = valid_from
        if data.confidence is not None:
            memory.confidence = data.confidence
        if data.metadata is not None:
            memory.metadata_ = data.metadata

        await self._session.flush()

        # Update embedding in Qdrant
        embedding = await self._embedding.embed_text(data.content)
        await self._qdrant.upsert(
            vector_id=str(memory.id),
            vector=embedding,
            payload={
                "memory_id": str(memory.id),
                "memory_type": MemoryType.SEMANTIC.value,
                "agent_id": memory.agent_id,
                "content": data.content,
                "subject": memory.subject,
                "category": memory.category,
                "status": memory.status,
                "confidence": memory.confidence,
                "created_at_ts": memory.created_at.timestamp(),
                "valid_from_ts": valid_from.timestamp(),
                "version": memory.version,
            },
        )

        # Record event
        await self._record_event(
            event_type=EventType.SEMANTIC_UPDATED,
            memory_type=MemoryType.SEMANTIC.value,
            memory_id=memory.id,
            agent_id=memory.agent_id,
            payload_before={"content": old_content, "version": memory.version - 1},
            payload_after=self._memory_to_dict(memory),
            event_metadata={"change_reason": data.change_reason},
        )

        await self._session.commit()

        logger.info(
            "Updated semantic memory",
            memory_id=str(memory.id),
            new_version=memory.version,
        )

        return memory

    async def get_semantic(
        self,
        memory_id: uuid.UUID,
        agent_id: str | None = None,
    ) -> SemanticMemory | None:
        """Get a semantic memory by ID."""
        return await self._semantic_repo.get_by_id(memory_id, agent_id)

    async def get_semantic_at_time(
        self,
        subject: str,
        as_of: datetime,
        agent_id: str,
    ) -> tuple[SemanticMemory | None, SemanticMemoryVersion | None]:
        """
        Get what the system believed about a subject at a specific time.
        
        Returns (memory, version_at_time).
        """
        memory = await self._semantic_repo.get_valid_at(agent_id, subject, as_of)
        if not memory:
            return None, None

        version = await self._semantic_repo.get_version_at(memory.id, as_of)
        return memory, version

    async def get_semantic_evolution(
        self,
        memory_id: uuid.UUID,
    ) -> list[SemanticMemoryVersion]:
        """Get the version history of a semantic memory."""
        return list(await self._semantic_repo.get_versions(memory_id))

    # =========================================================================
    # Procedural Memory Operations
    # =========================================================================

    async def store_procedural(
        self,
        data: ProceduralMemoryCreate,
        agent_id: str | None = None,
    ) -> tuple[ProceduralMemory, bool]:
        """
        Store or update a procedural memory.
        
        Procedural memories are key-value based and can be upserted.
        Returns (memory, is_new).
        """
        agent_id = agent_id or data.agent_id or self._settings.default_agent_id

        # Upsert the memory
        memory, is_new = await self._procedural_repo.upsert(
            agent_id=agent_id,
            key=data.key,
            value=data.value,
            category=data.category,
            value_text=data.value_text,
            confidence=data.confidence,
            source=data.source,
            metadata=data.metadata,
        )

        # Generate embedding for semantic search
        content = f"{data.key}: {data.value_text or str(data.value)}"
        embedding = await self._embedding.embed_text(content)

        # Store/update vector in Qdrant
        vector_id = str(memory.id)
        memory.vector_id = vector_id

        await self._qdrant.upsert(
            vector_id=vector_id,
            vector=embedding,
            payload={
                "memory_id": str(memory.id),
                "memory_type": MemoryType.PROCEDURAL.value,
                "agent_id": agent_id,
                "content": content,
                "key": data.key,
                "category": data.category,
                "status": MemoryStatus.ACTIVE.value,
                "confidence": memory.confidence,
                "created_at_ts": memory.created_at.timestamp(),
            },
        )

        # Record event
        event_type = EventType.PROCEDURAL_CREATED if is_new else EventType.PROCEDURAL_UPDATED
        await self._record_event(
            event_type=event_type,
            memory_type=MemoryType.PROCEDURAL.value,
            memory_id=memory.id,
            agent_id=agent_id,
            payload_after=self._memory_to_dict(memory),
        )

        await self._session.commit()

        logger.info(
            "Stored procedural memory",
            memory_id=str(memory.id),
            key=data.key,
            is_new=is_new,
        )

        return memory, is_new

    async def get_procedural(
        self,
        memory_id: uuid.UUID,
        agent_id: str | None = None,
    ) -> ProceduralMemory | None:
        """Get a procedural memory by ID."""
        return await self._procedural_repo.get_by_id(memory_id, agent_id)

    async def get_procedural_by_key(
        self,
        key: str,
        agent_id: str,
        category: str = "preference",
    ) -> ProceduralMemory | None:
        """Get a procedural memory by key."""
        return await self._procedural_repo.get_by_key(agent_id, key, category)

    async def reinforce_procedural(
        self,
        memory_id: uuid.UUID,
    ) -> None:
        """Reinforce a procedural memory (increment usage count)."""
        await self._procedural_repo.reinforce(memory_id)
        await self._session.commit()

    # =========================================================================
    # Helper Methods
    # =========================================================================

    async def _check_semantic_conflict(
        self,
        content: str,
        embedding: list[float],
        agent_id: str,
        subject: str,
    ) -> dict[str, Any] | None:
        """Check for conflicts with existing semantic memories."""
        # Find similar existing memories
        similar = await self._qdrant.find_similar(
            query_vector=embedding,
            agent_id=agent_id,
            memory_type=MemoryType.SEMANTIC.value,
            threshold=0.8,  # High similarity threshold
        )

        if not similar:
            return None

        # Get the most similar memory
        most_similar = similar[0]
        existing_content = most_similar["payload"].get("content", "")

        # Use LLM to analyze if this is a contradiction
        analysis = await self._llm.analyze_conflict(
            existing_content=existing_content,
            new_content=content,
            context=f"Subject: {subject}",
        )

        return {
            "existing_memory_id": most_similar["payload"].get("memory_id"),
            "existing_content": existing_content,
            "new_content": content,
            "similarity_score": most_similar["score"],
            "is_contradiction": analysis.get("is_contradiction", False),
            "confidence": analysis.get("confidence", 0.0),
            "llm_analysis": analysis.get("explanation", ""),
            "resolution": analysis.get("resolution_suggestion", "keep_both"),
        }

    async def _record_event(
        self,
        event_type: EventType,
        memory_type: str,
        memory_id: uuid.UUID,
        agent_id: str,
        payload_after: dict[str, Any],
        payload_before: dict[str, Any] | None = None,
        event_metadata: dict[str, Any] | None = None,
        correlation_id: str | None = None,
    ) -> MemoryEvent:
        """Record an event for deterministic replay."""
        event = MemoryEvent(
            event_type=event_type.value,
            memory_type=memory_type,
            memory_id=memory_id,
            agent_id=agent_id,
            payload_before=payload_before,
            payload_after=payload_after,
            event_metadata=event_metadata or {},
            correlation_id=correlation_id,
            actor="system",
        )
        return await self._event_repo.create(event)

    def _memory_to_dict(self, memory: Any) -> dict[str, Any]:
        """Convert a memory model to a dictionary for event storage."""
        result = {}
        for column in memory.__table__.columns:
            value = getattr(memory, column.name)
            if value is None:
                result[column.name] = None
            elif isinstance(value, datetime):
                result[column.name] = value.isoformat()
            elif isinstance(value, uuid.UUID):
                result[column.name] = str(value)
            elif isinstance(value, (str, int, float, bool)):
                result[column.name] = value
            elif isinstance(value, (list, dict)):
                result[column.name] = value
            else:
                # Skip non-serializable objects
                result[column.name] = str(value)
        return result
