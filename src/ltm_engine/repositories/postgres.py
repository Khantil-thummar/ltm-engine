"""PostgreSQL repository implementations."""

import uuid
from datetime import datetime, timezone
from typing import Any, Generic, Sequence, TypeVar

import structlog
from sqlalchemy import and_, func, or_, select, update, delete
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from ltm_engine.config import Settings
from ltm_engine.models import (
    EpisodicMemory,
    SemanticMemory,
    SemanticMemoryVersion,
    ProceduralMemory,
    MemoryEvent,
)
from ltm_engine.models.base import Base, MemoryStatus

logger = structlog.get_logger(__name__)

T = TypeVar("T", bound=Base)


class PostgresRepository:
    """
    Base PostgreSQL repository with connection management.
    
    Handles database engine and session lifecycle.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._engine = create_async_engine(
            settings.postgres_dsn,
            echo=settings.debug,
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True,
        )
        self._session_factory = async_sessionmaker(
            self._engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )

    async def init_db(self) -> None:
        """Initialize database tables."""
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("Database tables initialized")

    async def close(self) -> None:
        """Close database connections."""
        await self._engine.dispose()

    def session(self) -> AsyncSession:
        """Get a new database session."""
        return self._session_factory()

    async def health_check(self) -> bool:
        """Check database connectivity."""
        try:
            async with self.session() as session:
                await session.execute(select(1))
            return True
        except Exception as e:
            logger.error("Database health check failed", error=str(e))
            return False


class EpisodicRepository:
    """Repository for episodic memories."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(self, memory: EpisodicMemory) -> EpisodicMemory:
        """Create a new episodic memory."""
        self._session.add(memory)
        await self._session.flush()
        await self._session.refresh(memory)
        return memory

    async def get_by_id(
        self, memory_id: uuid.UUID, agent_id: str | None = None
    ) -> EpisodicMemory | None:
        """Get episodic memory by ID."""
        query = select(EpisodicMemory).where(EpisodicMemory.id == memory_id)
        if agent_id:
            query = query.where(EpisodicMemory.agent_id == agent_id)
        result = await self._session.execute(query)
        return result.scalar_one_or_none()

    async def get_by_ids(
        self, memory_ids: list[uuid.UUID], agent_id: str | None = None
    ) -> Sequence[EpisodicMemory]:
        """Get multiple episodic memories by IDs."""
        query = select(EpisodicMemory).where(EpisodicMemory.id.in_(memory_ids))
        if agent_id:
            query = query.where(EpisodicMemory.agent_id == agent_id)
        result = await self._session.execute(query)
        return result.scalars().all()

    async def list_by_agent(
        self,
        agent_id: str,
        status: str | None = None,
        session_id: str | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> Sequence[EpisodicMemory]:
        """List episodic memories for an agent with filters."""
        query = select(EpisodicMemory).where(EpisodicMemory.agent_id == agent_id)

        if status:
            query = query.where(EpisodicMemory.status == status)
        if session_id:
            query = query.where(EpisodicMemory.session_id == session_id)
        if start_time:
            query = query.where(EpisodicMemory.event_timestamp >= start_time)
        if end_time:
            query = query.where(EpisodicMemory.event_timestamp <= end_time)

        query = query.order_by(EpisodicMemory.event_timestamp.desc())
        query = query.limit(limit).offset(offset)

        result = await self._session.execute(query)
        return result.scalars().all()

    async def list_unconsolidated(
        self,
        agent_id: str,
        limit: int = 100,
    ) -> Sequence[EpisodicMemory]:
        """List episodic memories not yet consolidated."""
        query = (
            select(EpisodicMemory)
            .where(EpisodicMemory.agent_id == agent_id)
            .where(EpisodicMemory.consolidated_into_id.is_(None))
            .where(EpisodicMemory.status == MemoryStatus.ACTIVE.value)
            .order_by(EpisodicMemory.event_timestamp.asc())
            .limit(limit)
        )
        result = await self._session.execute(query)
        return result.scalars().all()

    async def mark_consolidated(
        self, memory_ids: list[uuid.UUID], semantic_memory_id: uuid.UUID
    ) -> int:
        """Mark episodic memories as consolidated."""
        stmt = (
            update(EpisodicMemory)
            .where(EpisodicMemory.id.in_(memory_ids))
            .values(consolidated_into_id=str(semantic_memory_id))
        )
        result = await self._session.execute(stmt)
        return result.rowcount

    async def increment_access(self, memory_id: uuid.UUID) -> None:
        """Increment access count for a memory."""
        stmt = (
            update(EpisodicMemory)
            .where(EpisodicMemory.id == memory_id)
            .values(
                access_count=EpisodicMemory.access_count + 1,
                last_accessed_at=datetime.now(timezone.utc),
            )
        )
        await self._session.execute(stmt)

    async def update_importance(
        self, memory_id: uuid.UUID, importance_score: float
    ) -> None:
        """Update importance score for a memory."""
        stmt = (
            update(EpisodicMemory)
            .where(EpisodicMemory.id == memory_id)
            .values(importance_score=importance_score)
        )
        await self._session.execute(stmt)

    async def update_status(self, memory_id: uuid.UUID, status: str) -> None:
        """Update memory status."""
        stmt = (
            update(EpisodicMemory)
            .where(EpisodicMemory.id == memory_id)
            .values(status=status)
        )
        await self._session.execute(stmt)

    async def delete_hard(self, memory_id: uuid.UUID) -> bool:
        """Permanently delete a memory."""
        stmt = delete(EpisodicMemory).where(EpisodicMemory.id == memory_id)
        result = await self._session.execute(stmt)
        return result.rowcount > 0

    async def count_by_agent(self, agent_id: str) -> int:
        """Count episodic memories for an agent."""
        query = (
            select(func.count())
            .select_from(EpisodicMemory)
            .where(EpisodicMemory.agent_id == agent_id)
        )
        result = await self._session.execute(query)
        return result.scalar() or 0

    async def get_memories_as_of(
        self,
        agent_id: str,
        as_of: datetime,
    ) -> Sequence[EpisodicMemory]:
        """Get episodic memories that existed as of a given time."""
        query = (
            select(EpisodicMemory)
            .where(EpisodicMemory.agent_id == agent_id)
            .where(EpisodicMemory.created_at <= as_of)
            .order_by(EpisodicMemory.event_timestamp.desc())
        )
        result = await self._session.execute(query)
        return result.scalars().all()


class SemanticRepository:
    """Repository for semantic memories."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(self, memory: SemanticMemory) -> SemanticMemory:
        """Create a new semantic memory."""
        self._session.add(memory)
        await self._session.flush()
        await self._session.refresh(memory)
        return memory

    async def create_version(
        self, version: SemanticMemoryVersion
    ) -> SemanticMemoryVersion:
        """Create a new version of a semantic memory."""
        self._session.add(version)
        await self._session.flush()
        return version

    async def get_by_id(
        self, memory_id: uuid.UUID, agent_id: str | None = None
    ) -> SemanticMemory | None:
        """Get semantic memory by ID."""
        query = select(SemanticMemory).where(SemanticMemory.id == memory_id)
        if agent_id:
            query = query.where(SemanticMemory.agent_id == agent_id)
        result = await self._session.execute(query)
        return result.scalar_one_or_none()

    async def get_by_ids(
        self, memory_ids: list[uuid.UUID], agent_id: str | None = None
    ) -> Sequence[SemanticMemory]:
        """Get multiple semantic memories by IDs."""
        query = select(SemanticMemory).where(SemanticMemory.id.in_(memory_ids))
        if agent_id:
            query = query.where(SemanticMemory.agent_id == agent_id)
        result = await self._session.execute(query)
        return result.scalars().all()

    async def get_by_subject(
        self,
        agent_id: str,
        subject: str,
        include_superseded: bool = False,
    ) -> Sequence[SemanticMemory]:
        """Get semantic memories by subject."""
        query = (
            select(SemanticMemory)
            .where(SemanticMemory.agent_id == agent_id)
            .where(SemanticMemory.subject.ilike(f"%{subject}%"))
        )
        if not include_superseded:
            query = query.where(SemanticMemory.status == MemoryStatus.ACTIVE.value)
        result = await self._session.execute(query)
        return result.scalars().all()

    async def list_by_agent(
        self,
        agent_id: str,
        status: str | None = None,
        category: str | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> Sequence[SemanticMemory]:
        """List semantic memories for an agent with filters."""
        query = select(SemanticMemory).where(SemanticMemory.agent_id == agent_id)

        if status:
            query = query.where(SemanticMemory.status == status)
        if category:
            query = query.where(SemanticMemory.category == category)
        if start_time:
            query = query.where(SemanticMemory.valid_from >= start_time)
        if end_time:
            query = query.where(
                or_(
                    SemanticMemory.valid_until.is_(None),
                    SemanticMemory.valid_until <= end_time,
                )
            )

        query = query.order_by(SemanticMemory.updated_at.desc())
        query = query.limit(limit).offset(offset)

        result = await self._session.execute(query)
        return result.scalars().all()

    async def get_valid_at(
        self,
        agent_id: str,
        subject: str,
        as_of: datetime,
    ) -> SemanticMemory | None:
        """Get the semantic memory that had a version valid at a specific time."""
        # First find the memory by subject (the main record's valid_from changes on update)
        # Then we check versions to see if any was valid at the given time
        # Order by created_at DESC for stable, predictable ordering (immutable field)
        query = (
            select(SemanticMemory)
            .where(SemanticMemory.agent_id == agent_id)
            .where(SemanticMemory.subject.ilike(f"%{subject}%"))
            .where(SemanticMemory.created_at <= as_of)  # Memory must have existed
            .order_by(SemanticMemory.created_at.desc())
        )
        result = await self._session.execute(query)
        memories = result.scalars().all()
        
        # For each memory (ordered by most recent first), check if it had a valid version at as_of
        for memory in memories:
            version = await self.get_version_at(memory.id, as_of)
            if version:
                return memory
        
        return None

    async def get_version_at(
        self,
        memory_id: uuid.UUID,
        as_of: datetime,
    ) -> SemanticMemoryVersion | None:
        """Get the version of a memory valid at a specific time."""
        query = (
            select(SemanticMemoryVersion)
            .where(SemanticMemoryVersion.memory_id == memory_id)
            .where(SemanticMemoryVersion.valid_from <= as_of)
            .where(
                or_(
                    SemanticMemoryVersion.valid_until.is_(None),
                    SemanticMemoryVersion.valid_until > as_of,
                )
            )
            .order_by(SemanticMemoryVersion.version.desc())
            .limit(1)
        )
        result = await self._session.execute(query)
        return result.scalar_one_or_none()

    async def get_versions(
        self, memory_id: uuid.UUID
    ) -> Sequence[SemanticMemoryVersion]:
        """Get all versions of a semantic memory."""
        query = (
            select(SemanticMemoryVersion)
            .where(SemanticMemoryVersion.memory_id == memory_id)
            .order_by(SemanticMemoryVersion.version.desc())
        )
        result = await self._session.execute(query)
        return result.scalars().all()

    async def update(
        self,
        memory_id: uuid.UUID,
        content: str,
        confidence: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> SemanticMemory | None:
        """Update a semantic memory."""
        memory = await self.get_by_id(memory_id)
        if not memory:
            return None

        memory.content = content
        memory.version += 1
        if confidence is not None:
            memory.confidence = confidence
        if metadata is not None:
            memory.metadata_ = metadata

        await self._session.flush()
        await self._session.refresh(memory)
        return memory

    async def supersede(
        self,
        old_memory_id: uuid.UUID,
        new_memory_id: uuid.UUID,
    ) -> None:
        """Mark a memory as superseded by another."""
        now = datetime.now(timezone.utc)

        # Update old memory
        stmt = (
            update(SemanticMemory)
            .where(SemanticMemory.id == old_memory_id)
            .values(
                status=MemoryStatus.SUPERSEDED.value,
                superseded_by_id=new_memory_id,
                valid_until=now,
            )
        )
        await self._session.execute(stmt)

        # Update new memory
        stmt = (
            update(SemanticMemory)
            .where(SemanticMemory.id == new_memory_id)
            .values(supersedes_id=old_memory_id)
        )
        await self._session.execute(stmt)

    async def increment_access(self, memory_id: uuid.UUID) -> None:
        """Increment access count for a memory."""
        stmt = (
            update(SemanticMemory)
            .where(SemanticMemory.id == memory_id)
            .values(
                access_count=SemanticMemory.access_count + 1,
                last_accessed_at=datetime.now(timezone.utc),
            )
        )
        await self._session.execute(stmt)

    async def update_importance(
        self, memory_id: uuid.UUID, importance_score: float
    ) -> None:
        """Update importance score for a memory."""
        stmt = (
            update(SemanticMemory)
            .where(SemanticMemory.id == memory_id)
            .values(importance_score=importance_score)
        )
        await self._session.execute(stmt)

    async def update_status(self, memory_id: uuid.UUID, status: str) -> None:
        """Update memory status."""
        stmt = (
            update(SemanticMemory)
            .where(SemanticMemory.id == memory_id)
            .values(status=status)
        )
        await self._session.execute(stmt)

    async def delete_hard(self, memory_id: uuid.UUID) -> bool:
        """Permanently delete a memory."""
        stmt = delete(SemanticMemory).where(SemanticMemory.id == memory_id)
        result = await self._session.execute(stmt)
        return result.rowcount > 0

    async def count_by_agent(self, agent_id: str) -> int:
        """Count semantic memories for an agent."""
        query = (
            select(func.count())
            .select_from(SemanticMemory)
            .where(SemanticMemory.agent_id == agent_id)
        )
        result = await self._session.execute(query)
        return result.scalar() or 0

    async def get_memories_as_of(
        self,
        agent_id: str,
        as_of: datetime,
    ) -> Sequence[SemanticMemory]:
        """Get semantic memories that were valid as of a given time."""
        query = (
            select(SemanticMemory)
            .where(SemanticMemory.agent_id == agent_id)
            .where(SemanticMemory.valid_from <= as_of)
            .where(
                or_(
                    SemanticMemory.valid_until.is_(None),
                    SemanticMemory.valid_until > as_of,
                )
            )
            .order_by(SemanticMemory.valid_from.desc())
        )
        result = await self._session.execute(query)
        return result.scalars().all()


class ProceduralRepository:
    """Repository for procedural memories."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(self, memory: ProceduralMemory) -> ProceduralMemory:
        """Create a new procedural memory."""
        self._session.add(memory)
        await self._session.flush()
        await self._session.refresh(memory)
        return memory

    async def get_by_id(
        self, memory_id: uuid.UUID, agent_id: str | None = None
    ) -> ProceduralMemory | None:
        """Get procedural memory by ID."""
        query = select(ProceduralMemory).where(ProceduralMemory.id == memory_id)
        if agent_id:
            query = query.where(ProceduralMemory.agent_id == agent_id)
        result = await self._session.execute(query)
        return result.scalar_one_or_none()

    async def get_by_key(
        self, agent_id: str, key: str, category: str = "preference"
    ) -> ProceduralMemory | None:
        """Get procedural memory by key."""
        query = (
            select(ProceduralMemory)
            .where(ProceduralMemory.agent_id == agent_id)
            .where(ProceduralMemory.key == key)
            .where(ProceduralMemory.category == category)
        )
        result = await self._session.execute(query)
        return result.scalar_one_or_none()

    async def get_by_ids(
        self, memory_ids: list[uuid.UUID], agent_id: str | None = None
    ) -> Sequence[ProceduralMemory]:
        """Get multiple procedural memories by IDs."""
        query = select(ProceduralMemory).where(ProceduralMemory.id.in_(memory_ids))
        if agent_id:
            query = query.where(ProceduralMemory.agent_id == agent_id)
        result = await self._session.execute(query)
        return result.scalars().all()

    async def list_by_agent(
        self,
        agent_id: str,
        category: str | None = None,
        status: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> Sequence[ProceduralMemory]:
        """List procedural memories for an agent with filters."""
        query = select(ProceduralMemory).where(ProceduralMemory.agent_id == agent_id)

        if category:
            query = query.where(ProceduralMemory.category == category)
        if status:
            query = query.where(ProceduralMemory.status == status)

        query = query.order_by(ProceduralMemory.last_used_at.desc())
        query = query.limit(limit).offset(offset)

        result = await self._session.execute(query)
        return result.scalars().all()

    async def upsert(
        self,
        agent_id: str,
        key: str,
        value: dict[str, Any],
        category: str = "preference",
        value_text: str | None = None,
        confidence: float = 1.0,
        source: str = "explicit",
        metadata: dict[str, Any] | None = None,
    ) -> tuple[ProceduralMemory, bool]:
        """Create or update a procedural memory. Returns (memory, created)."""
        existing = await self.get_by_key(agent_id, key, category)

        if existing:
            existing.value = value
            existing.value_text = value_text
            existing.confidence = confidence
            existing.source = source
            existing.last_used_at = datetime.now(timezone.utc)
            if metadata:
                existing.metadata_ = metadata
            await self._session.flush()
            await self._session.refresh(existing)
            return existing, False

        memory = ProceduralMemory(
            agent_id=agent_id,
            key=key,
            value=value,
            value_text=value_text,
            category=category,
            confidence=confidence,
            source=source,
            metadata_=metadata or {},
        )
        self._session.add(memory)
        await self._session.flush()
        await self._session.refresh(memory)
        return memory, True

    async def reinforce(self, memory_id: uuid.UUID) -> None:
        """Reinforce a procedural memory (increment count)."""
        stmt = (
            update(ProceduralMemory)
            .where(ProceduralMemory.id == memory_id)
            .values(
                reinforcement_count=ProceduralMemory.reinforcement_count + 1,
                last_used_at=datetime.now(timezone.utc),
            )
        )
        await self._session.execute(stmt)

    async def update_confidence(
        self, memory_id: uuid.UUID, confidence: float
    ) -> None:
        """Update confidence for a procedural memory."""
        stmt = (
            update(ProceduralMemory)
            .where(ProceduralMemory.id == memory_id)
            .values(confidence=confidence)
        )
        await self._session.execute(stmt)

    async def update_status(self, memory_id: uuid.UUID, status: str) -> None:
        """Update memory status."""
        stmt = (
            update(ProceduralMemory)
            .where(ProceduralMemory.id == memory_id)
            .values(status=status)
        )
        await self._session.execute(stmt)

    async def delete_hard(self, memory_id: uuid.UUID) -> bool:
        """Permanently delete a memory."""
        stmt = delete(ProceduralMemory).where(ProceduralMemory.id == memory_id)
        result = await self._session.execute(stmt)
        return result.rowcount > 0

    async def count_by_agent(self, agent_id: str) -> int:
        """Count procedural memories for an agent."""
        query = (
            select(func.count())
            .select_from(ProceduralMemory)
            .where(ProceduralMemory.agent_id == agent_id)
        )
        result = await self._session.execute(query)
        return result.scalar() or 0

    async def get_memories_as_of(
        self,
        agent_id: str,
        as_of: datetime,
    ) -> Sequence[ProceduralMemory]:
        """Get procedural memories that existed as of a given time."""
        query = (
            select(ProceduralMemory)
            .where(ProceduralMemory.agent_id == agent_id)
            .where(ProceduralMemory.created_at <= as_of)
            .order_by(ProceduralMemory.last_used_at.desc())
        )
        result = await self._session.execute(query)
        return result.scalars().all()


class EventRepository:
    """Repository for memory events (event sourcing)."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session
        self._sequence_counter = 0

    async def create(self, event: MemoryEvent) -> MemoryEvent:
        """Create a new event."""
        # Get next sequence number
        query = select(func.coalesce(func.max(MemoryEvent.sequence_number), 0) + 1)
        result = await self._session.execute(query)
        event.sequence_number = result.scalar() or 1

        self._session.add(event)
        await self._session.flush()
        return event

    async def get_events_for_memory(
        self,
        memory_id: uuid.UUID,
        start_sequence: int | None = None,
        end_sequence: int | None = None,
    ) -> Sequence[MemoryEvent]:
        """Get events for a specific memory."""
        query = (
            select(MemoryEvent)
            .where(MemoryEvent.memory_id == memory_id)
            .order_by(MemoryEvent.sequence_number.asc())
        )
        if start_sequence is not None:
            query = query.where(MemoryEvent.sequence_number >= start_sequence)
        if end_sequence is not None:
            query = query.where(MemoryEvent.sequence_number <= end_sequence)
        result = await self._session.execute(query)
        return result.scalars().all()

    async def get_events_by_agent(
        self,
        agent_id: str,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        event_types: list[str] | None = None,
        limit: int = 1000,
    ) -> Sequence[MemoryEvent]:
        """Get events for an agent with filters."""
        query = (
            select(MemoryEvent)
            .where(MemoryEvent.agent_id == agent_id)
            .order_by(MemoryEvent.sequence_number.asc())
        )
        if start_time:
            query = query.where(MemoryEvent.event_timestamp >= start_time)
        if end_time:
            query = query.where(MemoryEvent.event_timestamp <= end_time)
        if event_types:
            query = query.where(MemoryEvent.event_type.in_(event_types))
        query = query.limit(limit)
        result = await self._session.execute(query)
        return result.scalars().all()

    async def get_events_up_to(
        self,
        agent_id: str,
        as_of: datetime,
        memory_types: list[str] | None = None,
    ) -> Sequence[MemoryEvent]:
        """Get all events up to a point in time for replay."""
        query = (
            select(MemoryEvent)
            .where(MemoryEvent.agent_id == agent_id)
            .where(MemoryEvent.event_timestamp <= as_of)
            .order_by(MemoryEvent.sequence_number.asc())
        )
        if memory_types:
            query = query.where(MemoryEvent.memory_type.in_(memory_types))
        result = await self._session.execute(query)
        return result.scalars().all()

    async def count_events(
        self,
        agent_id: str | None = None,
        memory_id: uuid.UUID | None = None,
    ) -> int:
        """Count events with optional filters."""
        query = select(func.count()).select_from(MemoryEvent)
        if agent_id:
            query = query.where(MemoryEvent.agent_id == agent_id)
        if memory_id:
            query = query.where(MemoryEvent.memory_id == memory_id)
        result = await self._session.execute(query)
        return result.scalar() or 0
