"""Base model and common utilities for SQLAlchemy models."""

import enum
import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import DateTime, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class MemoryType(str, enum.Enum):
    """Types of memory supported by the LTM Engine."""

    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"


class MemoryStatus(str, enum.Enum):
    """Status of a memory item."""

    ACTIVE = "active"
    SUPERSEDED = "superseded"
    DECAYED = "decayed"
    DELETED = "deleted"
    COMPRESSED = "compressed"


def utc_now() -> datetime:
    """Get current UTC datetime."""
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy models."""

    type_annotation_map = {
        dict[str, Any]: JSONB,
    }


class TimestampMixin:
    """Mixin for created_at and updated_at timestamps."""

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        server_default=func.now(),
        onupdate=utc_now,
        nullable=False,
    )


class BaseMemory(TimestampMixin):
    """Base mixin for all memory types."""

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
    content: Mapped[str] = mapped_column(Text, nullable=False)
    metadata_: Mapped[dict[str, Any]] = mapped_column(
        "metadata",
        JSONB,
        default=dict,
        nullable=False,
    )
    vector_id: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
        comment="Reference to vector in Qdrant",
    )
    access_count: Mapped[int] = mapped_column(default=0, nullable=False)
    last_accessed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    confidence: Mapped[float] = mapped_column(default=1.0, nullable=False)
    importance_score: Mapped[float] = mapped_column(
        default=1.0,
        nullable=False,
        comment="Decayable importance score",
    )
    status: Mapped[str] = mapped_column(
        String(50),
        default=MemoryStatus.ACTIVE.value,
        nullable=False,
        index=True,
    )
