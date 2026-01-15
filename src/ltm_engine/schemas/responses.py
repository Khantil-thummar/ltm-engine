"""Pydantic schemas for API responses."""

from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, Field

from ltm_engine.schemas.memory import (
    EpisodicMemoryResponse,
    SemanticMemoryResponse,
    ProceduralMemoryResponse,
    MemoryVersionResponse,
)


# =============================================================================
# Store Response
# =============================================================================


class StoreResponse(BaseModel):
    """Response after storing a memory."""

    success: bool = True
    memory_id: UUID
    memory_type: str
    message: str = "Memory stored successfully"
    conflict_detected: bool = False
    conflict_info: "ConflictInfo | None" = None


# =============================================================================
# Retrieve Response
# =============================================================================


class RetrievedMemory(BaseModel):
    """A retrieved memory with scoring information."""

    memory_id: UUID
    memory_type: Literal["episodic", "semantic", "procedural"]
    content: str
    score: float = Field(..., description="Final combined score")
    semantic_score: float = Field(..., description="Semantic similarity score")
    recency_score: float = Field(..., description="Recency decay score")
    frequency_score: float = Field(..., description="Access frequency score")
    confidence_score: float = Field(..., description="Confidence score")
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    # Type-specific fields
    subject: str | None = None  # For semantic
    category: str | None = None  # For semantic/procedural
    key: str | None = None  # For procedural
    event_timestamp: datetime | None = None  # For episodic
    source: str | None = None  # For episodic
    version: int | None = None  # For semantic
    valid_from: datetime | None = None  # For semantic
    valid_until: datetime | None = None  # For semantic


class RetrieveResponse(BaseModel):
    """Response containing retrieved memories."""

    success: bool = True
    query: str
    total_found: int
    memories: list[RetrievedMemory]
    agent_id: str
    as_of: datetime | None = None
    execution_time_ms: float


# =============================================================================
# Consolidate Response
# =============================================================================


class ConsolidationResult(BaseModel):
    """Result of consolidating memories."""

    semantic_memory_id: UUID
    subject: str
    summary: str
    source_episodic_count: int
    source_episodic_ids: list[str]


class ConsolidateResponse(BaseModel):
    """Response after consolidating memories."""

    success: bool = True
    consolidated_count: int
    results: list[ConsolidationResult]
    agent_id: str
    message: str


# =============================================================================
# Decay Response
# =============================================================================


class DecayResponse(BaseModel):
    """Response after decaying memories."""

    success: bool = True
    affected_count: int
    agent_id: str | None
    memory_types: list[str]
    half_life_days: float
    message: str


# =============================================================================
# Forget Response
# =============================================================================


class ForgetResponse(BaseModel):
    """Response after forgetting memories."""

    success: bool = True
    policy: str
    affected_count: int
    agent_id: str | None
    memory_ids: list[str]
    message: str


# =============================================================================
# Conflict Detection
# =============================================================================


class ConflictInfo(BaseModel):
    """Information about a detected conflict."""

    existing_memory_id: UUID
    existing_content: str
    new_content: str
    similarity_score: float
    llm_analysis: str
    is_contradiction: bool
    confidence: float
    resolution: str = Field(
        ..., description="How the conflict was resolved: superseded, merged, rejected"
    )


# =============================================================================
# Temporal Query Responses
# =============================================================================


class TemporalQueryResponse(BaseModel):
    """Response for temporal queries."""

    success: bool = True
    subject: str
    as_of: datetime
    memory: SemanticMemoryResponse | None
    version_at_time: MemoryVersionResponse | None
    message: str


class EvolutionEntry(BaseModel):
    """A single entry in memory evolution."""

    version: int
    content: str
    confidence: float
    valid_from: datetime
    valid_until: datetime | None
    change_reason: str | None
    created_at: datetime


class EvolutionResponse(BaseModel):
    """Response showing how memory evolved over time."""

    success: bool = True
    subject: str
    agent_id: str
    memory_id: UUID | None
    total_versions: int
    evolution: list[EvolutionEntry]
    message: str


# =============================================================================
# Replay Response
# =============================================================================


class ReplayMemoryState(BaseModel):
    """Memory state at a point in time."""

    episodic_count: int
    semantic_count: int
    procedural_count: int
    episodic_memories: list[EpisodicMemoryResponse] = Field(default_factory=list)
    semantic_memories: list[SemanticMemoryResponse] = Field(default_factory=list)
    procedural_memories: list[ProceduralMemoryResponse] = Field(default_factory=list)


class ReplayEventEntry(BaseModel):
    """An event in the replay log."""

    sequence_number: int
    event_type: str
    memory_type: str
    memory_id: UUID
    event_timestamp: datetime
    actor: str
    summary: str


class ReplayResponse(BaseModel):
    """Response for deterministic replay."""

    success: bool = True
    agent_id: str
    as_of: datetime
    state: ReplayMemoryState
    events: list[ReplayEventEntry] = Field(default_factory=list)
    event_count: int
    message: str


# =============================================================================
# Health Response
# =============================================================================


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    postgres_connected: bool
    qdrant_connected: bool
    version: str
    timestamp: datetime


# =============================================================================
# Agent Responses
# =============================================================================


class AgentStats(BaseModel):
    """Statistics for an agent's memory."""

    agent_id: str
    episodic_count: int
    semantic_count: int
    procedural_count: int
    total_memories: int
    oldest_memory: datetime | None
    newest_memory: datetime | None
    average_confidence: float


class AgentListResponse(BaseModel):
    """Response listing all agents."""

    agents: list[AgentStats]
    total_agents: int
