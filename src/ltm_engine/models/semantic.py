"""
Semantic Memory Model.

Semantic memories are:
- Facts, summaries, distilled knowledge
- Updatable and versioned
- Support contradiction resolution
"""

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, ForeignKey, Index, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ltm_engine.models.base import Base, BaseMemory, TimestampMixin, utc_now


class SemanticMemory(Base, BaseMemory):
    """
    Semantic memory stores facts, summaries, and distilled knowledge.
    
    These memories are updatable and versioned, with support for
    temporal validity intervals and contradiction resolution.
    """

    __tablename__ = "semantic_memories"

    # Subject/topic of this semantic memory
    subject: Mapped[str] = mapped_column(
        String(512),
        nullable=False,
        index=True,
    )

    # Category for organization
    category: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
        index=True,
    )

    # Current version number
    version: Mapped[int] = mapped_column(
        Integer,
        default=1,
        nullable=False,
    )

    # Validity interval for temporal reasoning
    valid_from: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        nullable=False,
    )
    valid_until: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="NULL means currently valid",
    )

    # Source episodic memories that contributed to this semantic memory
    source_episodic_ids: Mapped[list[str]] = mapped_column(
        JSONB,
        default=list,
        nullable=False,
    )

    # Reference to the memory this supersedes (for conflict resolution)
    supersedes_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        nullable=True,
    )

    # Reference to the memory that superseded this one
    superseded_by_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        nullable=True,
    )

    # Conflict detection metadata
    conflict_metadata: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        default=dict,
        nullable=False,
        comment="Stores conflict detection results and resolution info",
    )

    # Related semantic memories
    related_memory_ids: Mapped[list[str]] = mapped_column(
        JSONB,
        default=list,
        nullable=False,
    )

    # Versions relationship
    versions: Mapped[list["SemanticMemoryVersion"]] = relationship(
        "SemanticMemoryVersion",
        back_populates="memory",
        order_by="SemanticMemoryVersion.version.desc()",
        lazy="selectin",
    )

    __table_args__ = (
        Index("idx_semantic_agent_subject", "agent_id", "subject"),
        Index("idx_semantic_agent_category", "agent_id", "category"),
        Index("idx_semantic_validity", "valid_from", "valid_until"),
        Index("idx_semantic_status_agent", "status", "agent_id"),
    )

    def __repr__(self) -> str:
        return (
            f"<SemanticMemory(id={self.id}, subject={self.subject}, "
            f"version={self.version}, agent_id={self.agent_id})>"
        )


class SemanticMemoryVersion(Base, TimestampMixin):
    """
    Version history for semantic memories.
    
    Enables temporal reasoning by tracking how facts evolve over time.
    """

    __tablename__ = "semantic_memory_versions"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )

    memory_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("semantic_memories.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    version: Mapped[int] = mapped_column(Integer, nullable=False)

    content: Mapped[str] = mapped_column(Text, nullable=False)

    # Snapshot of metadata at this version
    metadata_snapshot: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        default=dict,
        nullable=False,
    )

    confidence: Mapped[float] = mapped_column(default=1.0, nullable=False)

    # Validity interval for this version
    valid_from: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )
    valid_until: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    # Change reason/source
    change_reason: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    # Vector ID for this specific version
    vector_id: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
    )

    # Relationship back to parent
    memory: Mapped["SemanticMemory"] = relationship(
        "SemanticMemory",
        back_populates="versions",
    )

    __table_args__ = (
        Index("idx_version_memory_version", "memory_id", "version"),
        Index("idx_version_validity", "valid_from", "valid_until"),
    )

    def __repr__(self) -> str:
        return (
            f"<SemanticMemoryVersion(memory_id={self.memory_id}, "
            f"version={self.version})>"
        )
