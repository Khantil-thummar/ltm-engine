"""Pydantic schemas for API requests."""

import enum
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from ltm_engine.schemas.memory import (
    EpisodicMemoryCreate,
    SemanticMemoryCreate,
    ProceduralMemoryCreate,
)


class ForgetPolicy(str, enum.Enum):
    """Policy for forgetting memories."""

    HARD_DELETE = "hard_delete"  # Permanently remove
    SOFT_DELETE = "soft_delete"  # Mark as deleted, keep in DB
    COMPRESS = "compress"  # Summarize and reduce storage


class TimeWindow(BaseModel):
    """Time window for filtering memories."""

    start: datetime | None = Field(default=None, description="Start of time window")
    end: datetime | None = Field(default=None, description="End of time window")


class RetrieveFilters(BaseModel):
    """Filters for memory retrieval."""

    memory_types: list[Literal["episodic", "semantic", "procedural"]] | None = Field(
        default=None, description="Filter by memory types"
    )
    categories: list[str] | None = Field(default=None, description="Filter by categories")
    subjects: list[str] | None = Field(default=None, description="Filter by subjects")
    sources: list[str] | None = Field(default=None, description="Filter by sources")
    min_confidence: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Minimum confidence threshold"
    )
    status: list[str] | None = Field(
        default=None, description="Filter by status (active, superseded, etc.)"
    )
    session_id: str | None = Field(default=None, description="Filter by session ID")
    include_superseded: bool = Field(
        default=False, description="Include superseded memories"
    )


# =============================================================================
# Store Request
# =============================================================================


class StoreRequest(BaseModel):
    """
    Request to store a memory item.
    
    The memory_type determines which create schema to use for the item.
    """

    memory_type: Literal["episodic", "semantic", "procedural"] = Field(
        ..., description="Type of memory to store"
    )
    item: EpisodicMemoryCreate | SemanticMemoryCreate | ProceduralMemoryCreate = Field(
        ..., description="Memory item to store"
    )


# =============================================================================
# Retrieve Request
# =============================================================================


class RetrieveRequest(BaseModel):
    """Request to retrieve memories."""

    query: str = Field(..., min_length=1, description="Search query")
    top_k: int = Field(default=10, ge=1, le=100, description="Number of results")
    filters: RetrieveFilters = Field(
        default_factory=RetrieveFilters, description="Optional filters"
    )
    time_window: TimeWindow | None = Field(
        default=None, description="Optional time window filter"
    )
    agent_id: str | None = Field(
        default=None, description="Agent ID (uses default if not set)"
    )
    # Scoring weights (optional overrides)
    weight_semantic: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Override semantic similarity weight"
    )
    weight_recency: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Override recency weight"
    )
    weight_frequency: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Override frequency weight"
    )
    weight_confidence: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Override confidence weight"
    )
    # Temporal query options
    as_of: datetime | None = Field(
        default=None, description="Query memory state as of this time"
    )


# =============================================================================
# Consolidate Request
# =============================================================================


class ConsolidateRequest(BaseModel):
    """Request to consolidate episodic memories into semantic memories."""

    agent_id: str | None = Field(
        default=None, description="Agent ID (uses default if not set)"
    )
    session_id: str | None = Field(
        default=None, description="Only consolidate memories from this session"
    )
    time_window: TimeWindow | None = Field(
        default=None, description="Only consolidate memories in this time window"
    )
    min_memories: int = Field(
        default=5, ge=1, description="Minimum episodic memories to consolidate"
    )
    max_memories: int = Field(
        default=50, ge=1, description="Maximum episodic memories per consolidation"
    )
    force: bool = Field(
        default=False, description="Force consolidation even below threshold"
    )


# =============================================================================
# Decay Request
# =============================================================================


class DecayRequest(BaseModel):
    """Request to decay memory importance scores."""

    agent_id: str | None = Field(
        default=None, description="Agent ID (uses default if not set, 'all' for all agents)"
    )
    memory_types: list[Literal["episodic", "semantic", "procedural"]] | None = Field(
        default=None, description="Memory types to decay (default: all)"
    )
    half_life_days: float | None = Field(
        default=None, gt=0, description="Override half-life for decay"
    )
    min_importance: float = Field(
        default=0.01, ge=0.0, le=1.0, description="Minimum importance score to keep"
    )


# =============================================================================
# Forget Request
# =============================================================================


class ForgetRequest(BaseModel):
    """Request to forget memories based on policy."""

    policy: ForgetPolicy = Field(..., description="Forget policy to apply")
    agent_id: str | None = Field(
        default=None, description="Agent ID (uses default if not set)"
    )
    memory_ids: list[str] | None = Field(
        default=None, description="Specific memory IDs to forget"
    )
    memory_types: list[Literal["episodic", "semantic", "procedural"]] | None = Field(
        default=None, description="Memory types to forget"
    )
    filters: RetrieveFilters | None = Field(
        default=None, description="Filter criteria for memories to forget"
    )
    max_importance: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Only forget memories below this importance"
    )
    older_than: datetime | None = Field(
        default=None, description="Only forget memories older than this"
    )
    max_count: int | None = Field(
        default=None, ge=1, description="Maximum number of memories to forget"
    )
    confirm: bool = Field(
        default=False, description="Confirmation flag for destructive operations"
    )


# =============================================================================
# Temporal Query Requests
# =============================================================================


class TemporalQueryRequest(BaseModel):
    """Request for temporal reasoning queries."""

    subject: str = Field(..., description="Subject to query about")
    as_of: datetime = Field(..., description="Point in time to query")
    agent_id: str | None = Field(default=None, description="Agent ID")


class EvolutionQueryRequest(BaseModel):
    """Request to see how memory evolved over time."""

    subject: str = Field(..., description="Subject to track evolution of")
    agent_id: str | None = Field(default=None, description="Agent ID")
    start_time: datetime | None = Field(default=None, description="Start of time range")
    end_time: datetime | None = Field(default=None, description="End of time range")


# =============================================================================
# Replay Request
# =============================================================================


class ReplayRequest(BaseModel):
    """Request to replay memory state."""

    agent_id: str = Field(..., description="Agent ID to replay")
    as_of: datetime = Field(..., description="Replay state as of this time")
    memory_types: list[Literal["episodic", "semantic", "procedural"]] | None = Field(
        default=None, description="Memory types to include"
    )
    include_events: bool = Field(
        default=False, description="Include event log in response"
    )
