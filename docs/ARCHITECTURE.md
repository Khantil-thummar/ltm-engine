# LTM Engine Architecture

## Overview

The LTM (Long-Term Memory) Engine is designed to provide a robust, scalable memory subsystem for LLM-based applications. This document describes the system architecture, data models, retrieval flow, and key design decisions.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         FastAPI Layer                            │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐            │
│  │  Store   │ │ Retrieve │ │ Temporal │ │  Admin   │            │
│  │ Endpoints│ │ Endpoints│ │ Endpoints│ │ Endpoints│            │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘            │
└───────┼────────────┼────────────┼────────────┼──────────────────┘
        │            │            │            │
        ▼            ▼            ▼            ▼
┌─────────────────────────────────────────────────────────────────┐
│                       Service Layer                              │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐             │
│  │   Memory     │ │  Retrieval   │ │ Consolidation│             │
│  │   Service    │ │   Service    │ │   Service    │             │
│  └──────────────┘ └──────────────┘ └──────────────┘             │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐             │
│  │  Lifecycle   │ │   Conflict   │ │  Confidence  │             │
│  │   Service    │ │   Service    │ │   Service    │             │
│  └──────────────┘ └──────────────┘ └──────────────┘             │
│  ┌──────────────┐                                               │
│  │   Replay     │                                               │
│  │   Service    │                                               │
│  └──────────────┘                                               │
└─────────────────────────────────────────────────────────────────┘
        │                    │                    │
        ▼                    ▼                    ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│   PostgreSQL    │  │     Qdrant      │  │   LLM/Embedding │
│   Repository    │  │   Repository    │  │    Providers    │
└─────────────────┘  └─────────────────┘  └─────────────────┘
        │                    │                    │
        ▼                    ▼                    ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│   PostgreSQL    │  │     Qdrant      │  │   OpenAI API    │
│   (Structured)  │  │   (Vectors)     │  │  GPT-4o-mini    │
└─────────────────┘  └─────────────────┘  └─────────────────┘
```

## Memory Model

### Memory Types

#### 1. Episodic Memory
- **Purpose**: Store time-stamped events, conversations, interactions
- **Characteristics**:
  - Immutable (append-only)
  - Rich temporal metadata
  - Session-based grouping
- **Use Cases**: Conversation history, user actions, system events

```
┌─────────────────────────────────────┐
│         Episodic Memory             │
├─────────────────────────────────────┤
│ id: UUID                            │
│ agent_id: string                    │
│ content: text                       │
│ event_timestamp: datetime           │
│ source: string                      │
│ session_id: string (optional)       │
│ actor: string                       │
│ action_type: string                 │
│ entities: JSON                      │
│ confidence: float                   │
│ importance_score: float             │
│ status: enum                        │
│ consolidated_into_id: UUID          │
└─────────────────────────────────────┘
```

#### 2. Semantic Memory
- **Purpose**: Store facts, summaries, distilled knowledge
- **Characteristics**:
  - Versioned with full history
  - Supports validity intervals
  - Conflict resolution metadata
- **Use Cases**: User profiles, learned facts, extracted knowledge

```
┌─────────────────────────────────────┐
│         Semantic Memory             │
├─────────────────────────────────────┤
│ id: UUID                            │
│ agent_id: string                    │
│ content: text                       │
│ subject: string                     │
│ category: string                    │
│ version: int                        │
│ valid_from: datetime                │
│ valid_until: datetime (nullable)    │
│ source_episodic_ids: JSON[]         │
│ supersedes_id: UUID                 │
│ superseded_by_id: UUID              │
│ conflict_metadata: JSON             │
│ confidence: float                   │
│ importance_score: float             │
└─────────────────────────────────────┘
         │
         │ 1:N
         ▼
┌─────────────────────────────────────┐
│     Semantic Memory Version         │
├─────────────────────────────────────┤
│ id: UUID                            │
│ memory_id: UUID (FK)                │
│ version: int                        │
│ content: text                       │
│ confidence: float                   │
│ valid_from: datetime                │
│ valid_until: datetime               │
│ change_reason: text                 │
│ metadata_snapshot: JSON             │
└─────────────────────────────────────┘
```

#### 3. Procedural Memory
- **Purpose**: Store preferences, patterns, behavioral knowledge
- **Characteristics**:
  - Key-value structure
  - Reinforcement tracking
  - Fast lookup by key
- **Use Cases**: User preferences, system settings, learned patterns

```
┌─────────────────────────────────────┐
│        Procedural Memory            │
├─────────────────────────────────────┤
│ id: UUID                            │
│ agent_id: string                    │
│ key: string                         │
│ value: JSON                         │
│ value_text: string                  │
│ category: string                    │
│ confidence: float                   │
│ reinforcement_count: int            │
│ last_used_at: datetime              │
│ source: string                      │
└─────────────────────────────────────┘
```

## Data Schemas

### PostgreSQL Schema

```sql
-- Episodic Memories
CREATE TABLE episodic_memories (
    id UUID PRIMARY KEY,
    agent_id VARCHAR(255) NOT NULL,
    content TEXT NOT NULL,
    event_timestamp TIMESTAMPTZ NOT NULL,
    source VARCHAR(255) NOT NULL,
    session_id VARCHAR(255),
    actor VARCHAR(255),
    action_type VARCHAR(255),
    entities JSONB DEFAULT '{}',
    metadata JSONB DEFAULT '{}',
    vector_id VARCHAR(255),
    access_count INT DEFAULT 0,
    last_accessed_at TIMESTAMPTZ,
    confidence FLOAT DEFAULT 1.0,
    importance_score FLOAT DEFAULT 1.0,
    status VARCHAR(50) DEFAULT 'active',
    consolidated_into_id VARCHAR(255),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Semantic Memories
CREATE TABLE semantic_memories (
    id UUID PRIMARY KEY,
    agent_id VARCHAR(255) NOT NULL,
    content TEXT NOT NULL,
    subject VARCHAR(512) NOT NULL,
    category VARCHAR(255),
    version INT DEFAULT 1,
    valid_from TIMESTAMPTZ NOT NULL,
    valid_until TIMESTAMPTZ,
    source_episodic_ids JSONB DEFAULT '[]',
    supersedes_id UUID,
    superseded_by_id UUID,
    conflict_metadata JSONB DEFAULT '{}',
    related_memory_ids JSONB DEFAULT '[]',
    -- ... common fields
);

-- Memory Events (Event Sourcing)
CREATE TABLE memory_events (
    id UUID PRIMARY KEY,
    sequence_number BIGSERIAL UNIQUE,
    event_timestamp TIMESTAMPTZ DEFAULT NOW(),
    event_type VARCHAR(100) NOT NULL,
    agent_id VARCHAR(255) NOT NULL,
    memory_type VARCHAR(50) NOT NULL,
    memory_id UUID NOT NULL,
    payload_before JSONB,
    payload_after JSONB NOT NULL,
    event_metadata JSONB DEFAULT '{}',
    correlation_id VARCHAR(255),
    actor VARCHAR(255) DEFAULT 'system'
);
```

### Qdrant Vector Schema

```json
{
  "collection_name": "ltm_memories",
  "vectors": {
    "size": 1536,
    "distance": "Cosine"
  },
  "payload_schema": {
    "memory_id": "keyword",
    "memory_type": "keyword",
    "agent_id": "keyword",
    "content": "text",
    "status": "keyword",
    "confidence": "float",
    "created_at_ts": "float",
    "subject": "keyword",
    "category": "keyword"
  }
}
```

## Retrieval Flow

### Hybrid Ranking Algorithm

```
Final Score = w₁ × Semantic + w₂ × Recency + w₃ × Frequency + w₄ × Confidence

Where:
- Semantic = Cosine similarity from Qdrant (0-1)
- Recency = e^(-λt) where λ = ln(2)/half_life_days
- Frequency = log(1 + access_count) / log(1 + max_accesses)
- Confidence = Memory confidence score (0-1)

Default weights: w₁=0.4, w₂=0.25, w₃=0.15, w₄=0.2
```

### Retrieval Sequence

```
1. Query → Generate Embedding (OpenAI)
           ↓
2. Vector Search (Qdrant)
   - Filter by agent_id, memory_type, status
   - Get top-K × 3 candidates
           ↓
3. Enrich (PostgreSQL)
   - Fetch full memory records
   - Get access counts, timestamps
           ↓
4. Score & Rank
   - Calculate component scores
   - Apply weighted combination
   - Sort by combined score
           ↓
5. Return top-K results
```

## Consolidation Strategy

### Process

1. **Selection**: Find unconsolidated episodic memories
2. **Grouping**: Group by session_id or time proximity
3. **Summarization**: Use LLM to extract key facts
4. **Storage**: Create semantic memory with source references
5. **Linking**: Mark episodic memories as consolidated

### Trigger Conditions

- Threshold: N episodic memories accumulated
- Scheduled: Periodic consolidation job
- Manual: API trigger

```
Episodic Memories          →    LLM Summarization    →    Semantic Memory
┌────────────────┐              ┌──────────────┐         ┌──────────────┐
│ User asked     │              │              │         │ User is      │
│ about Python   │──────────────│   GPT-4o     │────────→│ learning     │
│                │              │   mini       │         │ Python for   │
├────────────────┤              │              │         │ data science │
│ Discussed      │──────────────│              │         │              │
│ pandas basics  │              └──────────────┘         └──────────────┘
├────────────────┤
│ Asked about    │
│ visualization  │
└────────────────┘
```

## Decay & Lifecycle

### Importance Decay

Uses exponential decay based on time since last access:

```
importance(t) = importance₀ × e^(-λt)

Where:
- λ = ln(2) / half_life_days
- t = days since last access
```

### Lifecycle States

```
┌────────┐    ┌────────────┐    ┌─────────┐    ┌─────────┐
│ ACTIVE │───→│ SUPERSEDED │───→│ DECAYED │───→│ DELETED │
└────────┘    └────────────┘    └─────────┘    └─────────┘
     │                               │
     └───────────────────────────────┘
                  decay()

ACTIVE → memory is current and retrievable
SUPERSEDED → replaced by newer version, kept for history
DECAYED → importance below threshold
DELETED → soft or hard deleted
COMPRESSED → summarized to reduce storage
```

## Conflict Detection & Resolution

### Detection Process

1. **Embedding Similarity**: Find memories with similarity > 0.3 (configurable threshold)
2. **LLM Analysis**: Use GPT-4o-mini to analyze if content contradicts
3. **Confidence Score**: Rate conflict certainty (0.0 to 1.0)

### Duplicate Detection

Before storing semantic memories, the system checks for near-duplicates:
- If similarity >= 98% AND not a contradiction → Skip storing, return existing memory
- This prevents redundant storage of identical information

### Resolution Strategies

| Strategy | Description | When to Use |
|----------|-------------|-------------|
| `supersede` | Mark old as superseded | New info more recent/reliable |
| `keep_both` | Keep both with metadata | Uncertainty about which is correct |
| `reject_new` | Don't store new | Existing memory more reliable |
| `merge` | Combine into new | Both contain partial truth |

## Trade-offs & Design Decisions

### PostgreSQL + Qdrant (vs. Single DB)

**Chosen**: Separate stores for structured data and vectors

**Pros**:
- Optimal for each use case
- Rich relational queries for temporal reasoning
- Best-in-class vector search performance

**Cons**:
- Increased operational complexity
- Need to maintain consistency

### Event Sourcing for Replay

**Chosen**: Full event log for deterministic replay

**Pros**:
- Complete audit trail
- Point-in-time reconstruction
- Debugging capability

**Cons**:
- Storage overhead
- Query complexity for some operations

### LLM for Conflict Detection

**Chosen**: Embedding + LLM hybrid approach

**Pros**:
- More accurate than pure embedding similarity
- Understands semantic contradictions
- Provides explanations

**Cons**:
- API cost for LLM calls
- Latency for conflict check

### Versioning vs. Immutable Append

**Chosen**: Versioned semantic memories, immutable episodic

**Pros**:
- Full history preserved
- Temporal queries supported
- Aligns with memory type semantics

**Cons**:
- More complex queries
- Additional storage for versions

## Scalability Considerations

### Database Scaling

- **PostgreSQL**: Read replicas, partitioning by agent_id
- **Qdrant**: Cluster mode, sharding

### Performance Optimizations

- Connection pooling (10 + 20 overflow)
- Batch embedding generation
- Index on frequently filtered columns
- Vector index optimization in Qdrant

### Future Improvements

1. Caching layer (Redis) for hot memories
2. Background workers for consolidation
3. Async batch processing
4. Memory compression for old data

## Multi-Agent Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    LTM Engine                            │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │   Agent A    │  │   Agent B    │  │   Agent C    │  │
│  │   Memories   │  │   Memories   │  │   Memories   │  │
│  │              │  │              │  │              │  │
│  │  agent_id:   │  │  agent_id:   │  │  agent_id:   │  │
│  │  "agent_a"   │  │  "agent_b"   │  │  "agent_c"   │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
│                                                         │
│  Isolation: All queries filtered by agent_id            │
│  Sharing: Possible via explicit memory copying          │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

## Security Considerations

1. **Agent Isolation**: Strict filtering by agent_id
2. **API Authentication**: Add auth middleware (not in scope)
3. **Data Encryption**: Use encrypted connections
4. **Input Validation**: Pydantic schemas validate all input

## Monitoring & Observability

- Structured logging with structlog
- Health check endpoint
- Event timestamps for debugging
- Sequence numbers for ordering

## API Endpoints Summary

### Memory Operations
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/memory/episodic` | POST | Store episodic memory |
| `/api/v1/memory/semantic` | POST | Store semantic memory (with conflict detection) |
| `/api/v1/memory/procedural` | POST | Store procedural memory |
| `/api/v1/memory/retrieve` | POST | Hybrid search with ranking |
| `/api/v1/memory/semantic/{id}` | PUT | Update semantic memory (creates version) |
| `/api/v1/memory/procedural/key/{key}` | GET | Get preference by key |

### Temporal Operations
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/memory/temporal/evolution/{id}` | GET | Get version history |
| `/api/v1/memory/replay` | POST | Reconstruct state at timestamp |
| `/api/v1/memory/timeline` | GET | Get agent event log |

### Lifecycle Operations
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/memory/consolidate` | POST | Episodic → Semantic via LLM |
| `/api/v1/memory/decay` | POST | Apply importance decay |
| `/api/v1/memory/forget` | POST | Soft/hard delete memories |

### System
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/health` | GET | Health check |
| `/api/v1/agents` | GET | List all agents with stats |

## Testing

See the **[API Testing Guide](API_TESTING_GUIDE.md)** for comprehensive curl commands to test all endpoints.

---

*This architecture document is part of the LTM Engine project deliverables.*
