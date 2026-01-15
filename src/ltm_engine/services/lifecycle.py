"""
Lifecycle Service - Memory decay and forget operations.

Manages the lifecycle of memories including:
- Importance decay over time
- Soft/hard deletion
- Compression of old memories
"""

import uuid
from datetime import datetime, timezone
from typing import Any

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from ltm_engine.config import Settings
from ltm_engine.models import EpisodicMemory, SemanticMemory, ProceduralMemory, MemoryEvent
from ltm_engine.models.base import MemoryType, MemoryStatus
from ltm_engine.models.events import EventType
from ltm_engine.repositories import (
    EpisodicRepository,
    SemanticRepository,
    ProceduralRepository,
    EventRepository,
    QdrantRepository,
)
from ltm_engine.schemas.requests import DecayRequest, ForgetRequest, ForgetPolicy
from ltm_engine.schemas.responses import DecayResponse, ForgetResponse
from ltm_engine.utils.scoring import calculate_importance_decay

logger = structlog.get_logger(__name__)


class LifecycleService:
    """
    Service for managing memory lifecycle.
    
    Handles:
    - Decay: Reduce importance scores over time
    - Forget: Remove or compress old/unimportant memories
    """

    def __init__(
        self,
        session: AsyncSession,
        qdrant: QdrantRepository,
        settings: Settings,
    ) -> None:
        self._session = session
        self._qdrant = qdrant
        self._settings = settings

        # Initialize repositories
        self._episodic_repo = EpisodicRepository(session)
        self._semantic_repo = SemanticRepository(session)
        self._procedural_repo = ProceduralRepository(session)
        self._event_repo = EventRepository(session)

    async def decay(
        self,
        request: DecayRequest,
    ) -> DecayResponse:
        """
        Apply decay to memory importance scores.
        
        Uses exponential decay based on time since last access.
        Memories below min_importance threshold may be marked for cleanup.
        """
        agent_id = request.agent_id or self._settings.default_agent_id
        half_life = request.half_life_days or self._settings.decay_half_life_days
        memory_types = request.memory_types or ["episodic", "semantic", "procedural"]

        reference_time = datetime.now(timezone.utc)
        affected_count = 0
        affected_types: list[str] = []

        # Process episodic memories
        if "episodic" in memory_types:
            count = await self._decay_episodic_memories(
                agent_id=agent_id,
                half_life=half_life,
                min_importance=request.min_importance,
                reference_time=reference_time,
            )
            affected_count += count
            if count > 0:
                affected_types.append("episodic")

        # Process semantic memories
        if "semantic" in memory_types:
            count = await self._decay_semantic_memories(
                agent_id=agent_id,
                half_life=half_life,
                min_importance=request.min_importance,
                reference_time=reference_time,
            )
            affected_count += count
            if count > 0:
                affected_types.append("semantic")

        # Process procedural memories
        if "procedural" in memory_types:
            count = await self._decay_procedural_memories(
                agent_id=agent_id,
                half_life=half_life,
                min_importance=request.min_importance,
                reference_time=reference_time,
            )
            affected_count += count
            if count > 0:
                affected_types.append("procedural")

        await self._session.commit()

        logger.info(
            "Applied decay to memories",
            agent_id=agent_id,
            affected_count=affected_count,
            half_life_days=half_life,
        )

        return DecayResponse(
            success=True,
            affected_count=affected_count,
            agent_id=agent_id,
            memory_types=affected_types,
            half_life_days=half_life,
            message=f"Decayed {affected_count} memories across {len(affected_types)} types.",
        )

    async def _decay_episodic_memories(
        self,
        agent_id: str,
        half_life: float,
        min_importance: float,
        reference_time: datetime,
    ) -> int:
        """Apply decay to episodic memories."""
        memories = await self._episodic_repo.list_by_agent(
            agent_id=agent_id,
            status=MemoryStatus.ACTIVE.value,
            limit=1000,
        )

        count = 0
        for memory in memories:
            new_importance = calculate_importance_decay(
                current_importance=memory.importance_score,
                last_accessed=memory.last_accessed_at or memory.created_at,
                half_life_days=half_life,
                reference_time=reference_time,
            )

            if new_importance != memory.importance_score:
                await self._episodic_repo.update_importance(memory.id, new_importance)
                count += 1

                # Mark as decayed if below threshold
                if new_importance < min_importance:
                    await self._episodic_repo.update_status(
                        memory.id, MemoryStatus.DECAYED.value
                    )
                    await self._qdrant.update_payload(
                        str(memory.id),
                        {"status": MemoryStatus.DECAYED.value},
                    )

        return count

    async def _decay_semantic_memories(
        self,
        agent_id: str,
        half_life: float,
        min_importance: float,
        reference_time: datetime,
    ) -> int:
        """Apply decay to semantic memories."""
        memories = await self._semantic_repo.list_by_agent(
            agent_id=agent_id,
            status=MemoryStatus.ACTIVE.value,
            limit=1000,
        )

        count = 0
        for memory in memories:
            new_importance = calculate_importance_decay(
                current_importance=memory.importance_score,
                last_accessed=memory.last_accessed_at or memory.created_at,
                half_life_days=half_life,
                reference_time=reference_time,
            )

            if new_importance != memory.importance_score:
                await self._semantic_repo.update_importance(memory.id, new_importance)
                count += 1

                if new_importance < min_importance:
                    await self._semantic_repo.update_status(
                        memory.id, MemoryStatus.DECAYED.value
                    )
                    await self._qdrant.update_payload(
                        str(memory.id),
                        {"status": MemoryStatus.DECAYED.value},
                    )

        return count

    async def _decay_procedural_memories(
        self,
        agent_id: str,
        half_life: float,
        min_importance: float,
        reference_time: datetime,
    ) -> int:
        """Apply decay to procedural memories (based on last_used_at)."""
        memories = await self._procedural_repo.list_by_agent(
            agent_id=agent_id,
            status=MemoryStatus.ACTIVE.value,
            limit=1000,
        )

        count = 0
        for memory in memories:
            # For procedural, decay confidence based on usage
            new_confidence = calculate_importance_decay(
                current_importance=memory.confidence,
                last_accessed=memory.last_used_at,
                half_life_days=half_life,
                reference_time=reference_time,
            )

            if new_confidence != memory.confidence:
                await self._procedural_repo.update_confidence(memory.id, new_confidence)
                count += 1

                if new_confidence < min_importance:
                    await self._procedural_repo.update_status(
                        memory.id, MemoryStatus.DECAYED.value
                    )
                    await self._qdrant.update_payload(
                        str(memory.id),
                        {"status": MemoryStatus.DECAYED.value},
                    )

        return count

    async def forget(
        self,
        request: ForgetRequest,
    ) -> ForgetResponse:
        """
        Forget memories based on policy.
        
        Policies:
        - HARD_DELETE: Permanently remove from database and vector store
        - SOFT_DELETE: Mark as deleted but keep in database
        - COMPRESS: Summarize and reduce storage (for future implementation)
        """
        if not request.confirm and request.policy == ForgetPolicy.HARD_DELETE:
            return ForgetResponse(
                success=False,
                policy=request.policy.value,
                affected_count=0,
                agent_id=request.agent_id,
                memory_ids=[],
                message="Hard delete requires confirmation. Set confirm=true.",
            )

        agent_id = request.agent_id or self._settings.default_agent_id
        memory_types = request.memory_types or ["episodic", "semantic", "procedural"]

        affected_ids: list[str] = []

        # If specific memory IDs provided
        if request.memory_ids:
            for memory_id_str in request.memory_ids:
                memory_id = uuid.UUID(memory_id_str)
                success = await self._forget_memory(
                    memory_id=memory_id,
                    agent_id=agent_id,
                    policy=request.policy,
                    memory_types=memory_types,
                )
                if success:
                    affected_ids.append(memory_id_str)
        else:
            # Find memories matching criteria
            memories_to_forget = await self._find_memories_to_forget(
                agent_id=agent_id,
                memory_types=memory_types,
                max_importance=request.max_importance,
                older_than=request.older_than,
                filters=request.filters,
            )

            for memory_id, memory_type in memories_to_forget:
                success = await self._forget_single(
                    memory_id=memory_id,
                    memory_type=memory_type,
                    policy=request.policy,
                )
                if success:
                    affected_ids.append(str(memory_id))

        await self._session.commit()

        logger.info(
            "Forgot memories",
            agent_id=agent_id,
            policy=request.policy.value,
            affected_count=len(affected_ids),
        )

        return ForgetResponse(
            success=True,
            policy=request.policy.value,
            affected_count=len(affected_ids),
            agent_id=agent_id,
            memory_ids=affected_ids,
            message=f"Applied {request.policy.value} to {len(affected_ids)} memories.",
        )

    async def _forget_memory(
        self,
        memory_id: uuid.UUID,
        agent_id: str,
        policy: ForgetPolicy,
        memory_types: list[str],
    ) -> bool:
        """Forget a specific memory by ID."""
        # Try to find the memory in each type
        for memory_type in memory_types:
            if memory_type == "episodic":
                memory = await self._episodic_repo.get_by_id(memory_id, agent_id)
                if memory:
                    return await self._forget_single(memory_id, "episodic", policy)
            elif memory_type == "semantic":
                memory = await self._semantic_repo.get_by_id(memory_id, agent_id)
                if memory:
                    return await self._forget_single(memory_id, "semantic", policy)
            elif memory_type == "procedural":
                memory = await self._procedural_repo.get_by_id(memory_id, agent_id)
                if memory:
                    return await self._forget_single(memory_id, "procedural", policy)

        return False

    async def _forget_single(
        self,
        memory_id: uuid.UUID,
        memory_type: str,
        policy: ForgetPolicy,
    ) -> bool:
        """Apply forget policy to a single memory."""
        if policy == ForgetPolicy.HARD_DELETE:
            # Delete from database
            if memory_type == "episodic":
                await self._episodic_repo.delete_hard(memory_id)
            elif memory_type == "semantic":
                await self._semantic_repo.delete_hard(memory_id)
            elif memory_type == "procedural":
                await self._procedural_repo.delete_hard(memory_id)

            # Delete from vector store
            await self._qdrant.delete(str(memory_id))
            return True

        elif policy == ForgetPolicy.SOFT_DELETE:
            # Mark as deleted
            if memory_type == "episodic":
                await self._episodic_repo.update_status(
                    memory_id, MemoryStatus.DELETED.value
                )
            elif memory_type == "semantic":
                await self._semantic_repo.update_status(
                    memory_id, MemoryStatus.DELETED.value
                )
            elif memory_type == "procedural":
                await self._procedural_repo.update_status(
                    memory_id, MemoryStatus.DELETED.value
                )

            # Update vector store status
            await self._qdrant.update_payload(
                str(memory_id),
                {"status": MemoryStatus.DELETED.value},
            )
            return True

        elif policy == ForgetPolicy.COMPRESS:
            # Mark as compressed (actual compression would be more complex)
            if memory_type == "episodic":
                await self._episodic_repo.update_status(
                    memory_id, MemoryStatus.COMPRESSED.value
                )
            elif memory_type == "semantic":
                await self._semantic_repo.update_status(
                    memory_id, MemoryStatus.COMPRESSED.value
                )
            elif memory_type == "procedural":
                await self._procedural_repo.update_status(
                    memory_id, MemoryStatus.COMPRESSED.value
                )

            await self._qdrant.update_payload(
                str(memory_id),
                {"status": MemoryStatus.COMPRESSED.value},
            )
            return True

        return False

    async def _find_memories_to_forget(
        self,
        agent_id: str,
        memory_types: list[str],
        max_importance: float | None,
        older_than: datetime | None,
        filters: Any | None,
    ) -> list[tuple[uuid.UUID, str]]:
        """Find memories matching forget criteria."""
        results: list[tuple[uuid.UUID, str]] = []

        if "episodic" in memory_types:
            memories = await self._episodic_repo.list_by_agent(
                agent_id=agent_id,
                limit=1000,
            )
            for m in memories:
                if self._matches_forget_criteria(m, max_importance, older_than):
                    results.append((m.id, "episodic"))

        if "semantic" in memory_types:
            memories = await self._semantic_repo.list_by_agent(
                agent_id=agent_id,
                limit=1000,
            )
            for m in memories:
                if self._matches_forget_criteria(m, max_importance, older_than):
                    results.append((m.id, "semantic"))

        if "procedural" in memory_types:
            memories = await self._procedural_repo.list_by_agent(
                agent_id=agent_id,
                limit=1000,
            )
            for m in memories:
                if self._matches_forget_criteria(m, max_importance, older_than, use_confidence=True):
                    results.append((m.id, "procedural"))

        return results

    def _matches_forget_criteria(
        self,
        memory: Any,
        max_importance: float | None,
        older_than: datetime | None,
        use_confidence: bool = False,
    ) -> bool:
        """Check if a memory matches forget criteria."""
        if max_importance is not None:
            importance = memory.confidence if use_confidence else getattr(memory, "importance_score", 1.0)
            if importance > max_importance:
                return False

        if older_than is not None:
            if memory.created_at > older_than:
                return False

        return True
