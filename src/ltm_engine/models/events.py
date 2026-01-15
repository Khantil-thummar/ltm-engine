"""
Memory Event Model for Deterministic Replay.

Stores all memory operations as events to enable:
- Deterministic replay of memory states
- Audit trail of all changes
- Point-in-time reconstruction
"""

import enum
import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import BigInteger, DateTime, Index, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from ltm_engine.models.base import Base, utc_now


class EventType(str, enum.Enum):
    """Types of memory events."""

    # Create events
    EPISODIC_CREATED = "episodic_created"
    SEMANTIC_CREATED = "semantic_created"
    PROCEDURAL_CREATED = "procedural_created"

    # Update events
    SEMANTIC_UPDATED = "semantic_updated"
    PROCEDURAL_UPDATED = "procedural_updated"
    CONFIDENCE_UPDATED = "confidence_updated"

    # Access events
    MEMORY_ACCESSED = "memory_accessed"

    # Lifecycle events
    MEMORY_DECAYED = "memory_decayed"
    MEMORY_CONSOLIDATED = "memory_consolidated"
    MEMORY_SUPERSEDED = "memory_superseded"
    MEMORY_DELETED = "memory_deleted"
    MEMORY_COMPRESSED = "memory_compressed"

    # Conflict events
    CONFLICT_DETECTED = "conflict_detected"
    CONFLICT_RESOLVED = "conflict_resolved"


class MemoryEvent(Base):
    """
    Event sourcing model for memory operations.
    
    Enables deterministic replay by storing all state changes
    as immutable events with full payload snapshots.
    """

    __tablename__ = "memory_events"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )

    # Sequence number for ordering (auto-increment)
    sequence_number: Mapped[int] = mapped_column(
        BigInteger,
        autoincrement=True,
        nullable=False,
        unique=True,
        index=True,
    )

    # Event timestamp
    event_timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        nullable=False,
        index=True,
    )

    # Event type
    event_type: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        index=True,
    )

    # Agent that triggered this event
    agent_id: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        index=True,
    )

    # Memory type affected
    memory_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
    )

    # Target memory ID
    memory_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        nullable=False,
        index=True,
    )

    # Full payload before the change (for rollback)
    payload_before: Mapped[dict[str, Any] | None] = mapped_column(
        JSONB,
        nullable=True,
    )

    # Full payload after the change
    payload_after: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
    )

    # Additional event metadata
    event_metadata: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        default=dict,
        nullable=False,
    )

    # Correlation ID for grouping related events
    correlation_id: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
        index=True,
    )

    # Actor/source of the event
    actor: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        default="system",
    )

    __table_args__ = (
        Index("idx_event_agent_timestamp", "agent_id", "event_timestamp"),
        Index("idx_event_memory_sequence", "memory_id", "sequence_number"),
        Index("idx_event_type_timestamp", "event_type", "event_timestamp"),
    )

    def __repr__(self) -> str:
        return (
            f"<MemoryEvent(id={self.id}, type={self.event_type}, "
            f"memory_id={self.memory_id}, seq={self.sequence_number})>"
        )
