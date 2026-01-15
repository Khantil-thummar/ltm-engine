"""
Main API Router for LTM Engine.

Provides RESTful endpoints for all memory operations.
"""

from datetime import datetime, timezone
from typing import Any
from uuid import UUID

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query

from ltm_engine import __version__
from ltm_engine.config import Settings, get_settings
from ltm_engine.dependencies import (
    get_memory_service,
    get_retrieval_service,
    get_consolidation_service,
    get_lifecycle_service,
    get_conflict_service,
    get_confidence_service,
    get_replay_service,
    get_postgres_repo,
    get_qdrant_repo,
)
from ltm_engine.models.base import MemoryType
from ltm_engine.schemas.memory import (
    EpisodicMemoryCreate,
    EpisodicMemoryResponse,
    SemanticMemoryCreate,
    SemanticMemoryResponse,
    SemanticMemoryUpdate,
    ProceduralMemoryCreate,
    ProceduralMemoryResponse,
    ProceduralMemoryUpdate,
    MemoryVersionResponse,
)
from ltm_engine.schemas.requests import (
    RetrieveRequest,
    ConsolidateRequest,
    DecayRequest,
    ForgetRequest,
    ReplayRequest,
)
from ltm_engine.schemas.responses import (
    StoreResponse,
    RetrieveResponse,
    ConsolidateResponse,
    DecayResponse,
    ForgetResponse,
    HealthResponse,
    TemporalQueryResponse,
    EvolutionResponse,
    EvolutionEntry,
    ReplayResponse,
    AgentStats,
    AgentListResponse,
)
from ltm_engine.services import (
    MemoryService,
    RetrievalService,
    ConsolidationService,
    LifecycleService,
    ConflictService,
    ConfidenceService,
    ReplayService,
)

logger = structlog.get_logger(__name__)


def create_router() -> APIRouter:
    """Create and configure the main API router."""
    router = APIRouter()

    # ==========================================================================
    # Health & Status
    # ==========================================================================

    @router.get("/health", response_model=HealthResponse, tags=["System"])
    async def health_check():
        """Check system health."""
        postgres = get_postgres_repo()
        qdrant = get_qdrant_repo()

        return HealthResponse(
            status="healthy",
            postgres_connected=await postgres.health_check(),
            qdrant_connected=await qdrant.health_check(),
            version=__version__,
            timestamp=datetime.now(timezone.utc),
        )

    # ==========================================================================
    # Store Operations
    # ==========================================================================

    @router.post(
        "/memory/episodic",
        response_model=StoreResponse,
        tags=["Memory - Store"],
    )
    async def store_episodic(
        data: EpisodicMemoryCreate,
        agent_id: str = Query(None, description="Agent ID"),
        service: MemoryService = Depends(get_memory_service),
    ):
        """
        Store an episodic memory.
        
        Episodic memories are immutable (append-only) records of events,
        conversations, or interactions.
        """
        memory = await service.store_episodic(data, agent_id)
        return StoreResponse(
            memory_id=memory.id,
            memory_type=MemoryType.EPISODIC.value,
        )

    @router.post(
        "/memory/semantic",
        response_model=StoreResponse,
        tags=["Memory - Store"],
    )
    async def store_semantic(
        data: SemanticMemoryCreate,
        agent_id: str = Query(None, description="Agent ID"),
        check_conflicts: bool = Query(True, description="Check for conflicts"),
        service: MemoryService = Depends(get_memory_service),
    ):
        """
        Store a semantic memory.
        
        Semantic memories are facts and knowledge that can be updated
        and versioned. Conflict detection is enabled by default.
        """
        memory, conflict_info = await service.store_semantic(
            data, agent_id, check_conflicts
        )

        # Only show conflict_info if it's an actual contradiction
        is_contradiction = conflict_info is not None and conflict_info.get("is_contradiction", False)
        
        return StoreResponse(
            memory_id=memory.id,
            memory_type=MemoryType.SEMANTIC.value,
            conflict_detected=is_contradiction,
            conflict_info=conflict_info if is_contradiction else None,
        )

    @router.post(
        "/memory/procedural",
        response_model=StoreResponse,
        tags=["Memory - Store"],
    )
    async def store_procedural(
        data: ProceduralMemoryCreate,
        agent_id: str = Query(None, description="Agent ID"),
        service: MemoryService = Depends(get_memory_service),
    ):
        """
        Store a procedural memory (preference/pattern).
        
        Procedural memories are key-value pairs representing preferences
        or behavioral patterns. If a memory with the same key exists,
        it will be updated.
        """
        memory, is_new = await service.store_procedural(data, agent_id)
        return StoreResponse(
            memory_id=memory.id,
            memory_type=MemoryType.PROCEDURAL.value,
            message="Created" if is_new else "Updated existing memory",
        )

    # ==========================================================================
    # Retrieve Operations
    # ==========================================================================

    @router.post(
        "/memory/retrieve",
        response_model=RetrieveResponse,
        tags=["Memory - Retrieve"],
    )
    async def retrieve_memories(
        request: RetrieveRequest,
        service: RetrievalService = Depends(get_retrieval_service),
    ):
        """
        Retrieve memories matching a query.
        
        Uses hybrid ranking combining:
        - Semantic similarity
        - Recency decay
        - Access frequency
        - Confidence score
        """
        return await service.retrieve(request)

    @router.get(
        "/memory/episodic/{memory_id}",
        response_model=EpisodicMemoryResponse,
        tags=["Memory - Retrieve"],
    )
    async def get_episodic_memory(
        memory_id: UUID,
        agent_id: str = Query(None, description="Agent ID"),
        service: MemoryService = Depends(get_memory_service),
    ):
        """Get an episodic memory by ID."""
        memory = await service.get_episodic(memory_id, agent_id)
        if not memory:
            raise HTTPException(status_code=404, detail="Memory not found")
        return EpisodicMemoryResponse.model_validate(memory)

    @router.get(
        "/memory/semantic/{memory_id}",
        response_model=SemanticMemoryResponse,
        tags=["Memory - Retrieve"],
    )
    async def get_semantic_memory(
        memory_id: UUID,
        agent_id: str = Query(None, description="Agent ID"),
        service: MemoryService = Depends(get_memory_service),
    ):
        """Get a semantic memory by ID."""
        memory = await service.get_semantic(memory_id, agent_id)
        if not memory:
            raise HTTPException(status_code=404, detail="Memory not found")
        return SemanticMemoryResponse.model_validate(memory)

    @router.get(
        "/memory/procedural/{memory_id}",
        response_model=ProceduralMemoryResponse,
        tags=["Memory - Retrieve"],
    )
    async def get_procedural_memory(
        memory_id: UUID,
        agent_id: str = Query(None, description="Agent ID"),
        service: MemoryService = Depends(get_memory_service),
    ):
        """Get a procedural memory by ID."""
        memory = await service.get_procedural(memory_id, agent_id)
        if not memory:
            raise HTTPException(status_code=404, detail="Memory not found")
        return ProceduralMemoryResponse.model_validate(memory)

    @router.get(
        "/memory/procedural/key/{key}",
        response_model=ProceduralMemoryResponse,
        tags=["Memory - Retrieve"],
    )
    async def get_procedural_by_key(
        key: str,
        agent_id: str = Query(..., description="Agent ID"),
        category: str = Query("preference", description="Category"),
        service: MemoryService = Depends(get_memory_service),
    ):
        """Get a procedural memory by key."""
        memory = await service.get_procedural_by_key(key, agent_id, category)
        if not memory:
            raise HTTPException(status_code=404, detail="Memory not found")
        return ProceduralMemoryResponse.model_validate(memory)

    # ==========================================================================
    # Update Operations
    # ==========================================================================

    @router.put(
        "/memory/semantic/{memory_id}",
        response_model=SemanticMemoryResponse,
        tags=["Memory - Update"],
    )
    async def update_semantic_memory(
        memory_id: UUID,
        data: SemanticMemoryUpdate,
        agent_id: str = Query(None, description="Agent ID"),
        service: MemoryService = Depends(get_memory_service),
    ):
        """
        Update a semantic memory.
        
        Creates a new version while preserving history for temporal queries.
        """
        memory = await service.update_semantic(memory_id, data, agent_id)
        if not memory:
            raise HTTPException(status_code=404, detail="Memory not found")
        return SemanticMemoryResponse.model_validate(memory)

    # ==========================================================================
    # Temporal Operations
    # ==========================================================================

    @router.get(
        "/memory/temporal/at",
        response_model=TemporalQueryResponse,
        tags=["Memory - Temporal"],
    )
    async def query_memory_at_time(
        subject: str = Query(..., description="Subject to query"),
        as_of: datetime = Query(..., description="Point in time"),
        agent_id: str = Query(..., description="Agent ID"),
        service: MemoryService = Depends(get_memory_service),
    ):
        """
        Query what the system believed about a subject at a specific time.
        
        Returns the semantic memory that was valid at the given timestamp.
        """
        memory, version = await service.get_semantic_at_time(subject, as_of, agent_id)

        return TemporalQueryResponse(
            subject=subject,
            as_of=as_of,
            memory=SemanticMemoryResponse.model_validate(memory) if memory else None,
            version_at_time=MemoryVersionResponse.model_validate(version) if version else None,
            message="Found" if memory else "No memory found for that time",
        )

    @router.get(
        "/memory/temporal/evolution/{memory_id}",
        response_model=EvolutionResponse,
        tags=["Memory - Temporal"],
    )
    async def get_memory_evolution(
        memory_id: UUID,
        agent_id: str = Query(..., description="Agent ID"),
        service: MemoryService = Depends(get_memory_service),
    ):
        """
        Get the evolution of a semantic memory over time.
        
        Returns all versions showing how the memory changed.
        """
        memory = await service.get_semantic(memory_id, agent_id)
        if not memory:
            raise HTTPException(status_code=404, detail="Memory not found")

        versions = await service.get_semantic_evolution(memory_id)

        return EvolutionResponse(
            subject=memory.subject,
            agent_id=agent_id,
            memory_id=memory_id,
            total_versions=len(versions),
            evolution=[
                EvolutionEntry(
                    version=v.version,
                    content=v.content,
                    confidence=v.confidence,
                    valid_from=v.valid_from,
                    valid_until=v.valid_until,
                    change_reason=v.change_reason,
                    created_at=v.created_at,
                )
                for v in versions
            ],
            message=f"Found {len(versions)} versions",
        )

    # ==========================================================================
    # Consolidation Operations
    # ==========================================================================

    @router.post(
        "/memory/consolidate",
        response_model=ConsolidateResponse,
        tags=["Memory - Lifecycle"],
    )
    async def consolidate_memories(
        request: ConsolidateRequest,
        service: ConsolidationService = Depends(get_consolidation_service),
    ):
        """
        Consolidate episodic memories into semantic memories.
        
        Groups related episodic memories and uses LLM to extract
        key facts and knowledge from them.
        """
        return await service.consolidate(request)

    # ==========================================================================
    # Decay Operations
    # ==========================================================================

    @router.post(
        "/memory/decay",
        response_model=DecayResponse,
        tags=["Memory - Lifecycle"],
    )
    async def decay_memories(
        request: DecayRequest,
        service: LifecycleService = Depends(get_lifecycle_service),
    ):
        """
        Apply decay to memory importance scores.
        
        Uses exponential decay based on time since last access.
        """
        return await service.decay(request)

    # ==========================================================================
    # Forget Operations
    # ==========================================================================

    @router.post(
        "/memory/forget",
        response_model=ForgetResponse,
        tags=["Memory - Lifecycle"],
    )
    async def forget_memories(
        request: ForgetRequest,
        service: LifecycleService = Depends(get_lifecycle_service),
    ):
        """
        Forget memories based on policy.
        
        Policies:
        - hard_delete: Permanently remove
        - soft_delete: Mark as deleted
        - compress: Summarize and reduce storage
        """
        return await service.forget(request)

    # ==========================================================================
    # Confidence Operations
    # ==========================================================================

    @router.post(
        "/memory/confidence/feedback",
        tags=["Memory - Confidence"],
    )
    async def update_confidence_feedback(
        memory_id: UUID = Query(..., description="Memory ID"),
        memory_type: str = Query(..., description="Memory type (semantic/procedural)"),
        is_correct: bool = Query(..., description="Was the memory correct?"),
        agent_id: str = Query(..., description="Agent ID"),
        learning_rate: float = Query(0.1, description="Learning rate"),
        service: ConfidenceService = Depends(get_confidence_service),
    ):
        """
        Update memory confidence based on feedback.
        
        Uses Bayesian-style updates to adjust confidence.
        """
        return await service.update_confidence_from_feedback(
            memory_id=memory_id,
            memory_type=memory_type,
            is_correct=is_correct,
            agent_id=agent_id,
            learning_rate=learning_rate,
        )

    @router.post(
        "/memory/confidence/set",
        tags=["Memory - Confidence"],
    )
    async def set_confidence(
        memory_id: UUID = Query(..., description="Memory ID"),
        memory_type: str = Query(..., description="Memory type"),
        confidence: float = Query(..., ge=0.0, le=1.0, description="Confidence value"),
        agent_id: str = Query(..., description="Agent ID"),
        reason: str = Query("manual", description="Reason for change"),
        service: ConfidenceService = Depends(get_confidence_service),
    ):
        """Directly set confidence to a specific value."""
        return await service.set_confidence(
            memory_id=memory_id,
            memory_type=memory_type,
            confidence=confidence,
            agent_id=agent_id,
            reason=reason,
        )

    @router.get(
        "/memory/confidence/history/{memory_id}",
        tags=["Memory - Confidence"],
    )
    async def get_confidence_history(
        memory_id: UUID,
        agent_id: str = Query(..., description="Agent ID"),
        service: ConfidenceService = Depends(get_confidence_service),
    ):
        """Get the history of confidence changes for a memory."""
        return await service.get_confidence_history(memory_id, agent_id)

    # ==========================================================================
    # Replay Operations
    # ==========================================================================

    @router.post(
        "/memory/replay",
        response_model=ReplayResponse,
        tags=["Memory - Replay"],
    )
    async def replay_memory_state(
        request: ReplayRequest,
        service: ReplayService = Depends(get_replay_service),
    ):
        """
        Replay memory state as of a specific time.
        
        Enables deterministic reconstruction of what the memory
        state looked like at any point in the past.
        """
        return await service.replay(request)

    @router.get(
        "/memory/history/{memory_id}",
        tags=["Memory - Replay"],
    )
    async def get_memory_history(
        memory_id: UUID,
        service: ReplayService = Depends(get_replay_service),
    ):
        """Get the complete event history of a memory."""
        return await service.get_memory_history(memory_id)

    @router.get(
        "/memory/timeline",
        tags=["Memory - Replay"],
    )
    async def get_agent_timeline(
        agent_id: str = Query(..., description="Agent ID"),
        start_time: datetime = Query(None, description="Start time"),
        end_time: datetime = Query(None, description="End time"),
        limit: int = Query(100, description="Max events to return"),
        service: ReplayService = Depends(get_replay_service),
    ):
        """Get a timeline of memory events for an agent."""
        return await service.get_agent_timeline(
            agent_id=agent_id,
            start_time=start_time,
            end_time=end_time,
            limit=limit,
        )

    # ==========================================================================
    # Conflict Operations
    # ==========================================================================

    @router.post(
        "/memory/conflict/detect",
        tags=["Memory - Conflict"],
    )
    async def detect_conflict(
        content: str = Query(..., description="Content to check"),
        agent_id: str = Query(..., description="Agent ID"),
        subject: str = Query(None, description="Subject/topic"),
        threshold: float = Query(0.8, description="Similarity threshold"),
        service: ConflictService = Depends(get_conflict_service),
    ):
        """
        Detect if content conflicts with existing memories.
        
        Uses embedding similarity and LLM analysis.
        """
        conflict = await service.detect_conflict(
            content=content,
            agent_id=agent_id,
            subject=subject,
            similarity_threshold=threshold,
        )
        if conflict:
            return conflict.model_dump()
        return {"conflict_detected": False}

    @router.get(
        "/memory/conflict/audit",
        tags=["Memory - Conflict"],
    )
    async def audit_conflicts(
        agent_id: str = Query(..., description="Agent ID"),
        service: ConflictService = Depends(get_conflict_service),
    ):
        """Find all potential conflicts in an agent's memories."""
        return await service.find_all_conflicts(agent_id)

    # ==========================================================================
    # Agent Management
    # ==========================================================================

    @router.get(
        "/agents",
        response_model=AgentListResponse,
        tags=["Agents"],
    )
    async def list_agents(
        service: MemoryService = Depends(get_memory_service),
    ):
        """List all agents with memory statistics."""
        agents_data = await service.get_all_agents()
        
        agents = [
            AgentStats(
                agent_id=a["agent_id"],
                episodic_count=a["episodic_count"],
                semantic_count=a["semantic_count"],
                procedural_count=a["procedural_count"],
                total_memories=a["total_memories"],
                oldest_memory=None,
                newest_memory=None,
                average_confidence=1.0,
            )
            for a in agents_data
        ]
        
        return AgentListResponse(
            agents=agents,
            total_agents=len(agents),
        )

    return router
