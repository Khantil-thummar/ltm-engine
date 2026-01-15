"""
Episodic Memory Model.

Episodic memories are:
- Conversation snippets, events, interactions
- Time-stamped
- Immutable (append-only)
"""

from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, Index, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from ltm_engine.models.base import Base, BaseMemory


class EpisodicMemory(Base, BaseMemory):
    """
    Episodic memory stores time-stamped events and interactions.
    
    These memories are immutable (append-only) and represent
    specific moments in time like conversations, events, or interactions.
    """

    __tablename__ = "episodic_memories"

    # Event timestamp (when the event occurred, not when stored)
    event_timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        index=True,
    )

    # Source of the episodic memory
    source: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        default="conversation",
        comment="Source type: conversation, event, interaction, etc.",
    )

    # Session identifier for grouping related episodes
    session_id: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
        index=True,
    )

    # Actor in the episode (user, system, etc.)
    actor: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
    )

    # Action or event type
    action_type: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
    )

    # Related entities mentioned in this episode
    entities: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        default=dict,
        nullable=False,
    )

    # Reference to consolidated semantic memory (if summarized)
    consolidated_into_id: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
        comment="UUID of semantic memory this was consolidated into",
    )

    __table_args__ = (
        Index("idx_episodic_agent_timestamp", "agent_id", "event_timestamp"),
        Index("idx_episodic_agent_session", "agent_id", "session_id"),
        Index("idx_episodic_status_timestamp", "status", "event_timestamp"),
    )

    def __repr__(self) -> str:
        return (
            f"<EpisodicMemory(id={self.id}, agent_id={self.agent_id}, "
            f"source={self.source}, event_timestamp={self.event_timestamp})>"
        )
