"""Pydantic schemas for memory types."""

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field, ConfigDict


# =============================================================================
# Episodic Memory Schemas
# =============================================================================


class EpisodicMemoryCreate(BaseModel):
    """Schema for creating an episodic memory."""

    content: str = Field(..., min_length=1, description="Memory content")
    event_timestamp: datetime | None = Field(
        default=None, description="When the event occurred (defaults to now)"
    )
    source: str = Field(default="conversation", description="Source type")
    session_id: str | None = Field(default=None, description="Session identifier")
    actor: str | None = Field(default=None, description="Actor in the episode")
    action_type: str | None = Field(default=None, description="Type of action/event")
    entities: dict[str, Any] = Field(default_factory=dict, description="Related entities")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    agent_id: str | None = Field(default=None, description="Agent ID (uses default if not set)")


class EpisodicMemoryResponse(BaseModel):
    """Schema for episodic memory response."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    content: str
    event_timestamp: datetime
    source: str
    session_id: str | None
    actor: str | None
    action_type: str | None
    entities: dict[str, Any]
    metadata: dict[str, Any] = Field(alias="metadata_")
    agent_id: str
    confidence: float
    importance_score: float
    access_count: int
    status: str
    created_at: datetime
    updated_at: datetime
    consolidated_into_id: str | None = None


# =============================================================================
# Semantic Memory Schemas
# =============================================================================


class SemanticMemoryCreate(BaseModel):
    """Schema for creating a semantic memory."""

    content: str = Field(..., min_length=1, description="Fact or knowledge content")
    subject: str = Field(..., min_length=1, description="Subject/topic of this memory")
    category: str | None = Field(default=None, description="Category for organization")
    valid_from: datetime | None = Field(
        default=None, description="When this fact became valid (defaults to now)"
    )
    source_episodic_ids: list[str] = Field(
        default_factory=list, description="Source episodic memory IDs"
    )
    related_memory_ids: list[str] = Field(
        default_factory=list, description="Related semantic memory IDs"
    )
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence score")
    agent_id: str | None = Field(default=None, description="Agent ID (uses default if not set)")


class SemanticMemoryUpdate(BaseModel):
    """Schema for updating a semantic memory."""

    content: str = Field(..., min_length=1, description="Updated content")
    change_reason: str | None = Field(default=None, description="Reason for the change")
    confidence: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Updated confidence"
    )
    valid_from: datetime | None = Field(
        default=None, description="When this update becomes valid"
    )
    metadata: dict[str, Any] | None = Field(default=None, description="Updated metadata")


class MemoryVersionResponse(BaseModel):
    """Schema for a memory version."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    version: int
    content: str
    confidence: float
    valid_from: datetime
    valid_until: datetime | None
    change_reason: str | None
    created_at: datetime


class SemanticMemoryResponse(BaseModel):
    """Schema for semantic memory response."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    content: str
    subject: str
    category: str | None
    version: int
    valid_from: datetime
    valid_until: datetime | None
    source_episodic_ids: list[str]
    related_memory_ids: list[str]
    supersedes_id: UUID | None
    superseded_by_id: UUID | None
    conflict_metadata: dict[str, Any]
    metadata: dict[str, Any] = Field(alias="metadata_")
    agent_id: str
    confidence: float
    importance_score: float
    access_count: int
    status: str
    created_at: datetime
    updated_at: datetime
    versions: list[MemoryVersionResponse] = Field(default_factory=list)


# =============================================================================
# Procedural Memory Schemas
# =============================================================================


class ProceduralMemoryCreate(BaseModel):
    """Schema for creating a procedural memory."""

    key: str = Field(..., min_length=1, description="Preference/pattern key")
    value: dict[str, Any] = Field(..., description="Preference value as JSON")
    value_text: str | None = Field(default=None, description="Simple string value")
    category: str = Field(default="preference", description="Category/namespace")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence score")
    source: str = Field(default="explicit", description="Source: explicit, inferred, default")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    agent_id: str | None = Field(default=None, description="Agent ID (uses default if not set)")


class ProceduralMemoryUpdate(BaseModel):
    """Schema for updating a procedural memory."""

    value: dict[str, Any] | None = Field(default=None, description="Updated value")
    value_text: str | None = Field(default=None, description="Updated text value")
    confidence: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Updated confidence"
    )
    reinforce: bool = Field(
        default=False, description="If true, increment reinforcement count"
    )
    metadata: dict[str, Any] | None = Field(default=None, description="Updated metadata")


class ProceduralMemoryResponse(BaseModel):
    """Schema for procedural memory response."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    key: str
    value: dict[str, Any]
    value_text: str | None
    category: str
    confidence: float
    reinforcement_count: int
    last_used_at: datetime
    source: str
    metadata: dict[str, Any] = Field(alias="metadata_")
    agent_id: str
    status: str
    created_at: datetime
    updated_at: datetime
