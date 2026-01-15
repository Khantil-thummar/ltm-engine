"""
Confidence Calibration Service.

Implements Bayesian-style confidence updates based on:
- Feedback (correct/incorrect)
- Reinforcement from usage
- Contradiction detection
"""

import uuid
from datetime import datetime, timezone
from typing import Any

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from ltm_engine.config import Settings
from ltm_engine.models import SemanticMemory, ProceduralMemory, MemoryEvent
from ltm_engine.models.base import MemoryType
from ltm_engine.models.events import EventType
from ltm_engine.repositories import (
    SemanticRepository,
    ProceduralRepository,
    EventRepository,
    QdrantRepository,
)
from ltm_engine.utils.scoring import calibrate_confidence

logger = structlog.get_logger(__name__)


class ConfidenceService:
    """
    Service for calibrating memory confidence scores.
    
    Provides Bayesian-style updates to confidence based on:
    - Explicit feedback (was the memory correct?)
    - Implicit signals (usage frequency, contradictions)
    - Source reliability
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

        self._semantic_repo = SemanticRepository(session)
        self._procedural_repo = ProceduralRepository(session)
        self._event_repo = EventRepository(session)

    async def update_confidence_from_feedback(
        self,
        memory_id: uuid.UUID,
        memory_type: str,
        is_correct: bool,
        agent_id: str,
        learning_rate: float = 0.1,
    ) -> dict[str, Any]:
        """
        Update confidence based on explicit feedback.
        
        Uses a Bayesian-style update:
        - If correct: confidence increases (diminishing returns near 1.0)
        - If incorrect: confidence decreases
        """
        if memory_type == MemoryType.SEMANTIC.value:
            memory = await self._semantic_repo.get_by_id(memory_id, agent_id)
        elif memory_type == MemoryType.PROCEDURAL.value:
            memory = await self._procedural_repo.get_by_id(memory_id, agent_id)
        else:
            return {"success": False, "error": f"Unsupported memory type: {memory_type}"}

        if not memory:
            return {"success": False, "error": "Memory not found"}

        old_confidence = memory.confidence
        new_confidence = calibrate_confidence(
            current_confidence=old_confidence,
            is_correct=is_correct,
            learning_rate=learning_rate,
        )

        # Update in database
        if memory_type == MemoryType.SEMANTIC.value:
            memory.confidence = new_confidence
        else:
            await self._procedural_repo.update_confidence(memory_id, new_confidence)

        # Update in vector store
        await self._qdrant.update_payload(
            str(memory_id),
            {"confidence": new_confidence},
        )

        # Record event
        await self._record_confidence_event(
            memory_id=memory_id,
            memory_type=memory_type,
            old_confidence=old_confidence,
            new_confidence=new_confidence,
            reason="feedback",
            agent_id=agent_id,
            metadata={"is_correct": is_correct, "learning_rate": learning_rate},
        )

        await self._session.commit()

        logger.info(
            "Updated confidence from feedback",
            memory_id=str(memory_id),
            is_correct=is_correct,
            old_confidence=old_confidence,
            new_confidence=new_confidence,
        )

        return {
            "success": True,
            "memory_id": str(memory_id),
            "old_confidence": old_confidence,
            "new_confidence": new_confidence,
            "change": new_confidence - old_confidence,
        }

    async def boost_confidence_from_reinforcement(
        self,
        memory_id: uuid.UUID,
        memory_type: str,
        agent_id: str,
        boost_factor: float = 0.05,
    ) -> dict[str, Any]:
        """
        Boost confidence when a memory is used/retrieved.
        
        Memories that are frequently used get a small confidence boost,
        reflecting implicit validation through use.
        """
        if memory_type == MemoryType.SEMANTIC.value:
            memory = await self._semantic_repo.get_by_id(memory_id, agent_id)
        elif memory_type == MemoryType.PROCEDURAL.value:
            memory = await self._procedural_repo.get_by_id(memory_id, agent_id)
        else:
            return {"success": False, "error": f"Unsupported memory type: {memory_type}"}

        if not memory:
            return {"success": False, "error": "Memory not found"}

        old_confidence = memory.confidence
        
        # Small boost with diminishing returns
        boost = boost_factor * (1.0 - old_confidence)
        new_confidence = min(1.0, old_confidence + boost)

        # Update in database
        if memory_type == MemoryType.SEMANTIC.value:
            memory.confidence = new_confidence
        else:
            await self._procedural_repo.update_confidence(memory_id, new_confidence)

        # Update in vector store
        await self._qdrant.update_payload(
            str(memory_id),
            {"confidence": new_confidence},
        )

        await self._session.commit()

        return {
            "success": True,
            "memory_id": str(memory_id),
            "old_confidence": old_confidence,
            "new_confidence": new_confidence,
            "boost": new_confidence - old_confidence,
        }

    async def penalize_confidence_from_contradiction(
        self,
        memory_id: uuid.UUID,
        memory_type: str,
        agent_id: str,
        penalty_factor: float = 0.15,
    ) -> dict[str, Any]:
        """
        Reduce confidence when a memory is found to contradict others.
        
        Contradictions indicate uncertainty, so both conflicting
        memories should have reduced confidence.
        """
        if memory_type == MemoryType.SEMANTIC.value:
            memory = await self._semantic_repo.get_by_id(memory_id, agent_id)
        else:
            return {"success": False, "error": "Only semantic memories support contradiction penalty"}

        if not memory:
            return {"success": False, "error": "Memory not found"}

        old_confidence = memory.confidence
        new_confidence = max(0.0, old_confidence * (1.0 - penalty_factor))

        memory.confidence = new_confidence

        await self._qdrant.update_payload(
            str(memory_id),
            {"confidence": new_confidence},
        )

        await self._record_confidence_event(
            memory_id=memory_id,
            memory_type=memory_type,
            old_confidence=old_confidence,
            new_confidence=new_confidence,
            reason="contradiction",
            agent_id=agent_id,
            metadata={"penalty_factor": penalty_factor},
        )

        await self._session.commit()

        logger.info(
            "Reduced confidence due to contradiction",
            memory_id=str(memory_id),
            old_confidence=old_confidence,
            new_confidence=new_confidence,
        )

        return {
            "success": True,
            "memory_id": str(memory_id),
            "old_confidence": old_confidence,
            "new_confidence": new_confidence,
            "penalty": old_confidence - new_confidence,
        }

    async def set_confidence(
        self,
        memory_id: uuid.UUID,
        memory_type: str,
        confidence: float,
        agent_id: str,
        reason: str = "manual",
    ) -> dict[str, Any]:
        """
        Directly set confidence to a specific value.
        
        Useful for manual corrections or importing from external sources.
        """
        if not 0.0 <= confidence <= 1.0:
            return {"success": False, "error": "Confidence must be between 0.0 and 1.0"}

        if memory_type == MemoryType.SEMANTIC.value:
            memory = await self._semantic_repo.get_by_id(memory_id, agent_id)
            if memory:
                old_confidence = memory.confidence
                memory.confidence = confidence
        elif memory_type == MemoryType.PROCEDURAL.value:
            memory = await self._procedural_repo.get_by_id(memory_id, agent_id)
            if memory:
                old_confidence = memory.confidence
                await self._procedural_repo.update_confidence(memory_id, confidence)
        else:
            return {"success": False, "error": f"Unsupported memory type: {memory_type}"}

        if not memory:
            return {"success": False, "error": "Memory not found"}

        await self._qdrant.update_payload(
            str(memory_id),
            {"confidence": confidence},
        )

        await self._record_confidence_event(
            memory_id=memory_id,
            memory_type=memory_type,
            old_confidence=old_confidence,
            new_confidence=confidence,
            reason=reason,
            agent_id=agent_id,
        )

        await self._session.commit()

        return {
            "success": True,
            "memory_id": str(memory_id),
            "old_confidence": old_confidence,
            "new_confidence": confidence,
        }

    async def get_confidence_history(
        self,
        memory_id: uuid.UUID,
        agent_id: str,
    ) -> list[dict[str, Any]]:
        """Get the history of confidence changes for a memory."""
        events = await self._event_repo.get_events_for_memory(memory_id)
        
        confidence_events = [
            {
                "timestamp": e.event_timestamp.isoformat(),
                "old_confidence": e.payload_before.get("confidence") if e.payload_before else None,
                "new_confidence": e.payload_after.get("confidence") if e.payload_after else None,
                "reason": e.event_metadata.get("reason", "unknown"),
            }
            for e in events
            if e.event_type == EventType.CONFIDENCE_UPDATED.value
        ]

        return confidence_events

    async def _record_confidence_event(
        self,
        memory_id: uuid.UUID,
        memory_type: str,
        old_confidence: float,
        new_confidence: float,
        reason: str,
        agent_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record a confidence update event."""
        event = MemoryEvent(
            event_type=EventType.CONFIDENCE_UPDATED.value,
            memory_type=memory_type,
            memory_id=memory_id,
            agent_id=agent_id,
            payload_before={"confidence": old_confidence},
            payload_after={"confidence": new_confidence},
            event_metadata={"reason": reason, **(metadata or {})},
            actor="confidence_service",
        )
        await self._event_repo.create(event)
