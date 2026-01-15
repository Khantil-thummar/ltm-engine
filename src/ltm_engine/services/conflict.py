"""
Conflict Service - Detect and resolve memory conflicts.

Handles conflict detection and resolution when new memories
contradict existing memories using embedding similarity and LLM analysis.
"""

import uuid
from datetime import datetime, timezone
from typing import Any

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from ltm_engine.config import Settings
from ltm_engine.models import SemanticMemory, MemoryEvent
from ltm_engine.models.base import MemoryType, MemoryStatus
from ltm_engine.models.events import EventType
from ltm_engine.providers import EmbeddingProvider, LLMProvider
from ltm_engine.repositories import (
    SemanticRepository,
    EventRepository,
    QdrantRepository,
)
from ltm_engine.schemas.responses import ConflictInfo

logger = structlog.get_logger(__name__)


class ConflictService:
    """
    Service for detecting and resolving memory conflicts.
    
    Uses a two-stage approach:
    1. Embedding similarity to find potentially conflicting memories
    2. LLM analysis to determine if there's an actual contradiction
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

        self._semantic_repo = SemanticRepository(session)
        self._event_repo = EventRepository(session)

    async def detect_conflict(
        self,
        content: str,
        agent_id: str,
        subject: str | None = None,
        similarity_threshold: float = 0.8,
    ) -> ConflictInfo | None:
        """
        Detect if new content conflicts with existing memories.
        
        Args:
            content: New content to check
            agent_id: Agent's memory space
            subject: Optional subject to narrow search
            similarity_threshold: Minimum similarity to consider
            
        Returns:
            ConflictInfo if conflict detected, None otherwise
        """
        # Generate embedding for new content
        embedding = await self._embedding.embed_text(content)

        # Find similar semantic memories
        similar = await self._qdrant.find_similar(
            query_vector=embedding,
            agent_id=agent_id,
            memory_type=MemoryType.SEMANTIC.value,
            threshold=similarity_threshold,
        )

        if not similar:
            return None

        # Get the most similar memory
        most_similar = similar[0]
        existing_content = most_similar["payload"].get("content", "")
        existing_id = most_similar["payload"].get("memory_id")

        # Use LLM to analyze conflict
        analysis = await self._llm.analyze_conflict(
            existing_content=existing_content,
            new_content=content,
            context=f"Subject: {subject}" if subject else None,
        )

        if not analysis.get("is_contradiction", False):
            return None

        return ConflictInfo(
            existing_memory_id=uuid.UUID(existing_id),
            existing_content=existing_content,
            new_content=content,
            similarity_score=most_similar["score"],
            llm_analysis=analysis.get("explanation", ""),
            is_contradiction=True,
            confidence=analysis.get("confidence", 0.5),
            resolution=analysis.get("resolution_suggestion", "supersede"),
        )

    async def resolve_conflict(
        self,
        existing_memory_id: uuid.UUID,
        new_memory: SemanticMemory,
        resolution: str,
        agent_id: str,
    ) -> dict[str, Any]:
        """
        Resolve a detected conflict.
        
        Resolution strategies:
        - supersede: Mark old as superseded, new becomes active
        - keep_both: Both remain active with conflict metadata
        - reject_new: Don't store the new memory
        - merge: Combine both into a new memory (requires LLM)
        """
        existing_memory = await self._semantic_repo.get_by_id(existing_memory_id, agent_id)
        if not existing_memory:
            return {"success": False, "error": "Existing memory not found"}

        result: dict[str, Any] = {
            "success": True,
            "resolution": resolution,
            "existing_memory_id": str(existing_memory_id),
            "new_memory_id": str(new_memory.id),
        }

        if resolution == "supersede":
            # Mark existing as superseded
            await self._semantic_repo.supersede(existing_memory_id, new_memory.id)
            
            # Update vector store
            await self._qdrant.update_payload(
                str(existing_memory_id),
                {"status": MemoryStatus.SUPERSEDED.value},
            )

            result["message"] = "Existing memory superseded by new memory"

        elif resolution == "keep_both":
            # Update conflict metadata on both
            existing_memory.conflict_metadata = {
                **existing_memory.conflict_metadata,
                "conflicting_memory_id": str(new_memory.id),
                "conflict_detected_at": datetime.now(timezone.utc).isoformat(),
            }
            new_memory.conflict_metadata = {
                **new_memory.conflict_metadata,
                "conflicting_memory_id": str(existing_memory_id),
                "conflict_detected_at": datetime.now(timezone.utc).isoformat(),
            }
            
            result["message"] = "Both memories kept with conflict metadata"

        elif resolution == "reject_new":
            # Mark new memory as deleted
            await self._semantic_repo.update_status(
                new_memory.id, MemoryStatus.DELETED.value
            )
            await self._qdrant.delete(str(new_memory.id))
            
            result["message"] = "New memory rejected due to conflict"

        elif resolution == "merge":
            # Use LLM to merge both memories
            merged_content = await self._merge_memories(
                existing_memory.content,
                new_memory.content,
            )
            
            # Update the new memory with merged content
            new_memory.content = merged_content
            new_memory.conflict_metadata = {
                "merged_from": [str(existing_memory_id)],
                "merge_timestamp": datetime.now(timezone.utc).isoformat(),
            }
            
            # Mark existing as superseded
            await self._semantic_repo.supersede(existing_memory_id, new_memory.id)
            
            # Update embedding for merged content
            embedding = await self._embedding.embed_text(merged_content)
            await self._qdrant.upsert(
                vector_id=str(new_memory.id),
                vector=embedding,
                payload={
                    "memory_id": str(new_memory.id),
                    "memory_type": MemoryType.SEMANTIC.value,
                    "agent_id": agent_id,
                    "content": merged_content,
                    "subject": new_memory.subject,
                    "category": new_memory.category,
                    "status": MemoryStatus.ACTIVE.value,
                    "confidence": new_memory.confidence,
                    "created_at_ts": new_memory.created_at.timestamp(),
                },
            )
            
            result["message"] = "Memories merged into new memory"
            result["merged_content"] = merged_content

        # Record conflict resolution event
        await self._record_conflict_event(
            existing_memory_id=existing_memory_id,
            new_memory_id=new_memory.id,
            resolution=resolution,
            agent_id=agent_id,
        )

        await self._session.commit()

        logger.info(
            "Resolved memory conflict",
            existing_id=str(existing_memory_id),
            new_id=str(new_memory.id),
            resolution=resolution,
        )

        return result

    async def _merge_memories(
        self,
        content1: str,
        content2: str,
    ) -> str:
        """Merge two memory contents using LLM."""
        prompt = f"""Merge these two pieces of information into a single, coherent statement that captures the most accurate and complete understanding:

Memory 1:
{content1}

Memory 2:
{content2}

Provide a merged statement that:
1. Resolves any contradictions by favoring more recent/specific information
2. Combines complementary information
3. Is clear and factual"""

        return await self._llm.generate(prompt)

    async def _record_conflict_event(
        self,
        existing_memory_id: uuid.UUID,
        new_memory_id: uuid.UUID,
        resolution: str,
        agent_id: str,
    ) -> None:
        """Record a conflict resolution event."""
        event = MemoryEvent(
            event_type=EventType.CONFLICT_RESOLVED.value,
            memory_type=MemoryType.SEMANTIC.value,
            memory_id=new_memory_id,
            agent_id=agent_id,
            payload_before={"existing_memory_id": str(existing_memory_id)},
            payload_after={
                "new_memory_id": str(new_memory_id),
                "resolution": resolution,
            },
            event_metadata={"resolution_type": resolution},
            actor="conflict_service",
        )
        await self._event_repo.create(event)

    async def find_all_conflicts(
        self,
        agent_id: str,
        subject: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Find all potential conflicts in an agent's semantic memories.
        
        Useful for periodic conflict audits.
        """
        memories = await self._semantic_repo.list_by_agent(
            agent_id=agent_id,
            status=MemoryStatus.ACTIVE.value,
            limit=500,
        )

        conflicts: list[dict[str, Any]] = []

        # Check each memory against others
        for i, memory in enumerate(memories):
            if memory.conflict_metadata:
                # Already has conflict info
                continue

            # Generate embedding
            embedding = await self._embedding.embed_text(memory.content)

            # Find similar (excluding self)
            similar = await self._qdrant.find_similar(
                query_vector=embedding,
                agent_id=agent_id,
                memory_type=MemoryType.SEMANTIC.value,
                threshold=0.85,
                exclude_ids=[str(memory.id)],
            )

            for match in similar:
                # Use LLM to check for contradiction
                analysis = await self._llm.analyze_conflict(
                    existing_content=memory.content,
                    new_content=match["payload"].get("content", ""),
                )

                if analysis.get("is_contradiction"):
                    conflicts.append({
                        "memory_1_id": str(memory.id),
                        "memory_1_content": memory.content,
                        "memory_2_id": match["payload"].get("memory_id"),
                        "memory_2_content": match["payload"].get("content"),
                        "similarity": match["score"],
                        "analysis": analysis,
                    })

        return conflicts
