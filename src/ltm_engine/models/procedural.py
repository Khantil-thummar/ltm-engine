"""
Procedural / Preference Memory Model.

Procedural memories are:
- User or system preferences, patterns
- Key-value with confidence and last-updated time
- Lightweight and fast to access
"""

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, Index, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from ltm_engine.models.base import Base, TimestampMixin, utc_now


class ProceduralMemory(Base, TimestampMixin):
    """
    Procedural memory stores preferences, patterns, and behavioral knowledge.
    
    These are lightweight key-value memories with confidence scores,
    optimized for fast retrieval and updates.
    """

    __tablename__ = "procedural_memories"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )

    agent_id: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        index=True,
        default="default",
    )

    # Key for fast lookup
    key: Mapped[str] = mapped_column(
        String(512),
        nullable=False,
        index=True,
    )

    # Value as JSON for flexibility
    value: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
    )

    # Optional string representation for simple values
    value_text: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    # Category/namespace for organization
    category: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        default="preference",
        index=True,
    )

    # Confidence in this preference/pattern (0.0 to 1.0)
    confidence: Mapped[float] = mapped_column(
        default=1.0,
        nullable=False,
    )

    # Number of times this preference was reinforced
    reinforcement_count: Mapped[int] = mapped_column(
        default=1,
        nullable=False,
    )

    # Last time this preference was used/accessed
    last_used_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        nullable=False,
    )

    # Source of this preference
    source: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        default="inferred",
        comment="Source: explicit, inferred, default",
    )

    # Additional metadata
    metadata_: Mapped[dict[str, Any]] = mapped_column(
        "metadata",
        JSONB,
        default=dict,
        nullable=False,
    )

    # Status for lifecycle management
    status: Mapped[str] = mapped_column(
        String(50),
        default="active",
        nullable=False,
        index=True,
    )

    # Vector ID for semantic search (optional for procedural)
    vector_id: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
    )

    __table_args__ = (
        UniqueConstraint("agent_id", "key", "category", name="uq_procedural_agent_key_category"),
        Index("idx_procedural_agent_category", "agent_id", "category"),
        Index("idx_procedural_agent_key", "agent_id", "key"),
    )

    def __repr__(self) -> str:
        return (
            f"<ProceduralMemory(id={self.id}, key={self.key}, "
            f"category={self.category}, agent_id={self.agent_id})>"
        )
