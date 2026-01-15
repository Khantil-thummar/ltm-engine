"""
Replay Service - Deterministic replay of memory states.

Enables reconstruction of memory state at any point in time
using event sourcing patterns.
"""

import uuid
from datetime import datetime, timezone
from typing import Any

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from ltm_engine.config import Settings
from ltm_engine.models import MemoryEvent
from ltm_engine.models.base import MemoryType, MemoryStatus
from ltm_engine.models.events import EventType
from ltm_engine.repositories import (
    EpisodicRepository,
    SemanticRepository,
    ProceduralRepository,
    EventRepository,
)
from ltm_engine.schemas.requests import ReplayRequest
from ltm_engine.schemas.responses import (
    ReplayResponse,
    ReplayMemoryState,
    ReplayEventEntry,
)
from ltm_engine.schemas.memory import (
    EpisodicMemoryResponse,
    SemanticMemoryResponse,
    ProceduralMemoryResponse,
)

logger = structlog.get_logger(__name__)


class ReplayService:
    """
    Service for deterministic memory state replay.
    
    Uses event sourcing to reconstruct the memory state
    at any point in time, enabling:
    - Point-in-time queries
    - Debugging and auditing
    - State reconstruction after failures
    """

    def __init__(
        self,
        session: AsyncSession,
        settings: Settings,
    ) -> None:
        self._session = session
        self._settings = settings

        self._episodic_repo = EpisodicRepository(session)
        self._semantic_repo = SemanticRepository(session)
        self._procedural_repo = ProceduralRepository(session)
        self._event_repo = EventRepository(session)

    async def replay(
        self,
        request: ReplayRequest,
    ) -> ReplayResponse:
        """
        Replay memory state as of a specific time.
        
        Reconstructs what the memory state looked like at the given
        point in time by either:
        1. Querying current state with time filters (fast)
        2. Replaying events from the beginning (accurate but slow)
        """
        agent_id = request.agent_id
        as_of = request.as_of
        memory_types = request.memory_types or ["episodic", "semantic", "procedural"]

        # Get memories that existed at that time
        state = await self._get_state_at_time(agent_id, as_of, memory_types)

        # Get events if requested
        events: list[ReplayEventEntry] = []
        event_count = 0

        if request.include_events:
            raw_events = await self._event_repo.get_events_up_to(
                agent_id=agent_id,
                as_of=as_of,
                memory_types=memory_types,
            )
            event_count = len(raw_events)
            events = [self._event_to_entry(e) for e in raw_events[-100:]]  # Last 100

        logger.info(
            "Replayed memory state",
            agent_id=agent_id,
            as_of=as_of.isoformat(),
            episodic_count=state.episodic_count,
            semantic_count=state.semantic_count,
            procedural_count=state.procedural_count,
            event_count=event_count,
        )

        return ReplayResponse(
            success=True,
            agent_id=agent_id,
            as_of=as_of,
            state=state,
            events=events,
            event_count=event_count,
            message=f"Replayed state as of {as_of.isoformat()}",
        )

    async def _get_state_at_time(
        self,
        agent_id: str,
        as_of: datetime,
        memory_types: list[str],
    ) -> ReplayMemoryState:
        """Get memory state at a specific time."""
        episodic_memories: list[EpisodicMemoryResponse] = []
        semantic_memories: list[SemanticMemoryResponse] = []
        procedural_memories: list[ProceduralMemoryResponse] = []

        if "episodic" in memory_types:
            memories = await self._episodic_repo.get_memories_as_of(agent_id, as_of)
            episodic_memories = [
                EpisodicMemoryResponse.model_validate(m) for m in memories
            ]

        if "semantic" in memory_types:
            memories = await self._semantic_repo.get_memories_as_of(agent_id, as_of)
            semantic_memories = [
                SemanticMemoryResponse.model_validate(m) for m in memories
            ]

        if "procedural" in memory_types:
            memories = await self._procedural_repo.get_memories_as_of(agent_id, as_of)
            procedural_memories = [
                ProceduralMemoryResponse.model_validate(m) for m in memories
            ]

        return ReplayMemoryState(
            episodic_count=len(episodic_memories),
            semantic_count=len(semantic_memories),
            procedural_count=len(procedural_memories),
            episodic_memories=episodic_memories,
            semantic_memories=semantic_memories,
            procedural_memories=procedural_memories,
        )

    async def replay_from_events(
        self,
        agent_id: str,
        as_of: datetime,
        memory_types: list[str] | None = None,
    ) -> dict[str, dict[str, Any]]:
        """
        Reconstruct state by replaying all events.
        
        This is more accurate than querying current state but slower.
        Useful for debugging or when current state might be inconsistent.
        """
        memory_types = memory_types or ["episodic", "semantic", "procedural"]

        # Get all events up to the timestamp
        events = await self._event_repo.get_events_up_to(
            agent_id=agent_id,
            as_of=as_of,
            memory_types=memory_types,
        )

        # Replay events to reconstruct state
        state: dict[str, dict[str, Any]] = {}

        for event in events:
            memory_id = str(event.memory_id)

            if event.event_type in [
                EventType.EPISODIC_CREATED.value,
                EventType.SEMANTIC_CREATED.value,
                EventType.PROCEDURAL_CREATED.value,
            ]:
                state[memory_id] = {
                    "memory_type": event.memory_type,
                    "data": event.payload_after,
                    "status": MemoryStatus.ACTIVE.value,
                }

            elif event.event_type in [
                EventType.SEMANTIC_UPDATED.value,
                EventType.PROCEDURAL_UPDATED.value,
            ]:
                if memory_id in state:
                    state[memory_id]["data"].update(event.payload_after)

            elif event.event_type == EventType.CONFIDENCE_UPDATED.value:
                if memory_id in state:
                    state[memory_id]["data"]["confidence"] = event.payload_after.get(
                        "confidence"
                    )

            elif event.event_type == EventType.MEMORY_DELETED.value:
                if memory_id in state:
                    state[memory_id]["status"] = MemoryStatus.DELETED.value

            elif event.event_type == EventType.MEMORY_SUPERSEDED.value:
                if memory_id in state:
                    state[memory_id]["status"] = MemoryStatus.SUPERSEDED.value

            elif event.event_type == EventType.MEMORY_DECAYED.value:
                if memory_id in state:
                    state[memory_id]["status"] = MemoryStatus.DECAYED.value

        # Filter to only active memories
        active_state = {
            k: v for k, v in state.items()
            if v["status"] == MemoryStatus.ACTIVE.value
        }

        return active_state

    async def get_memory_history(
        self,
        memory_id: uuid.UUID,
    ) -> list[dict[str, Any]]:
        """
        Get the complete history of a memory.
        
        Returns all events related to a specific memory in chronological order.
        """
        events = await self._event_repo.get_events_for_memory(memory_id)

        return [
            {
                "sequence": e.sequence_number,
                "timestamp": e.event_timestamp.isoformat(),
                "event_type": e.event_type,
                "actor": e.actor,
                "payload_before": e.payload_before,
                "payload_after": e.payload_after,
                "metadata": e.event_metadata,
            }
            for e in events
        ]

    async def get_agent_timeline(
        self,
        agent_id: str,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        event_types: list[str] | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """
        Get a timeline of events for an agent.
        
        Useful for understanding what happened during a specific period.
        """
        events = await self._event_repo.get_events_by_agent(
            agent_id=agent_id,
            start_time=start_time,
            end_time=end_time,
            event_types=event_types,
            limit=limit,
        )

        return [
            {
                "sequence": e.sequence_number,
                "timestamp": e.event_timestamp.isoformat(),
                "event_type": e.event_type,
                "memory_type": e.memory_type,
                "memory_id": str(e.memory_id),
                "actor": e.actor,
                "summary": self._get_event_summary(e),
            }
            for e in events
        ]

    def _event_to_entry(self, event: MemoryEvent) -> ReplayEventEntry:
        """Convert a MemoryEvent to a ReplayEventEntry."""
        return ReplayEventEntry(
            sequence_number=event.sequence_number,
            event_type=event.event_type,
            memory_type=event.memory_type,
            memory_id=event.memory_id,
            event_timestamp=event.event_timestamp,
            actor=event.actor,
            summary=self._get_event_summary(event),
        )

    def _get_event_summary(self, event: MemoryEvent) -> str:
        """Generate a human-readable summary of an event."""
        summaries = {
            EventType.EPISODIC_CREATED.value: "Episodic memory created",
            EventType.SEMANTIC_CREATED.value: "Semantic memory created",
            EventType.PROCEDURAL_CREATED.value: "Procedural memory created",
            EventType.SEMANTIC_UPDATED.value: "Semantic memory updated",
            EventType.PROCEDURAL_UPDATED.value: "Procedural memory updated",
            EventType.CONFIDENCE_UPDATED.value: "Confidence score updated",
            EventType.MEMORY_ACCESSED.value: "Memory accessed",
            EventType.MEMORY_DECAYED.value: "Memory importance decayed",
            EventType.MEMORY_CONSOLIDATED.value: "Episodic memories consolidated",
            EventType.MEMORY_SUPERSEDED.value: "Memory superseded by newer version",
            EventType.MEMORY_DELETED.value: "Memory deleted",
            EventType.MEMORY_COMPRESSED.value: "Memory compressed",
            EventType.CONFLICT_DETECTED.value: "Conflict detected",
            EventType.CONFLICT_RESOLVED.value: "Conflict resolved",
        }
        return summaries.get(event.event_type, event.event_type)

    async def count_events(
        self,
        agent_id: str | None = None,
        memory_id: uuid.UUID | None = None,
    ) -> int:
        """Count events with optional filters."""
        return await self._event_repo.count_events(agent_id, memory_id)
