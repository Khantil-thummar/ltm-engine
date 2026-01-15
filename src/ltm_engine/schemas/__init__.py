"""Pydantic schemas for LTM Engine API."""

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
    StoreRequest,
    RetrieveRequest,
    ConsolidateRequest,
    DecayRequest,
    ForgetRequest,
    ForgetPolicy,
    TimeWindow,
    RetrieveFilters,
)
from ltm_engine.schemas.responses import (
    StoreResponse,
    RetrieveResponse,
    RetrievedMemory,
    ConsolidateResponse,
    DecayResponse,
    ForgetResponse,
    HealthResponse,
    TemporalQueryResponse,
    EvolutionResponse,
    ConflictInfo,
    ReplayResponse,
)

__all__ = [
    # Memory schemas
    "EpisodicMemoryCreate",
    "EpisodicMemoryResponse",
    "SemanticMemoryCreate",
    "SemanticMemoryResponse",
    "SemanticMemoryUpdate",
    "ProceduralMemoryCreate",
    "ProceduralMemoryResponse",
    "ProceduralMemoryUpdate",
    "MemoryVersionResponse",
    # Request schemas
    "StoreRequest",
    "RetrieveRequest",
    "ConsolidateRequest",
    "DecayRequest",
    "ForgetRequest",
    "ForgetPolicy",
    "TimeWindow",
    "RetrieveFilters",
    # Response schemas
    "StoreResponse",
    "RetrieveResponse",
    "RetrievedMemory",
    "ConsolidateResponse",
    "DecayResponse",
    "ForgetResponse",
    "HealthResponse",
    "TemporalQueryResponse",
    "EvolutionResponse",
    "ConflictInfo",
    "ReplayResponse",
]
