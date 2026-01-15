# LTM Engine

**Long-Term Memory Engine for LLM-based Applications**

A production-ready memory subsystem that can be used by LLM-based applications (agents, copilots, assistants) to store, retrieve, update, and reason over information across sessions and time.

## Features

### Memory Types

- **Episodic Memory**: Time-stamped, immutable records of events, conversations, and interactions
- **Semantic Memory**: Versioned facts and knowledge with conflict resolution
- **Procedural Memory**: User preferences and behavioral patterns (key-value)

### Core Capabilities

- ✅ **Hybrid Retrieval**: Semantic similarity + recency decay + access frequency + confidence scoring
- ✅ **Temporal Reasoning**: Query historical states, track memory evolution over time
- ✅ **Conflict Detection**: Embedding similarity + LLM-based contradiction analysis
- ✅ **Memory Consolidation**: Summarize episodic memories into semantic knowledge
- ✅ **Lifecycle Management**: Decay, forget (soft/hard delete), compress
- ✅ **Confidence Calibration**: Bayesian-style confidence updates
- ✅ **Deterministic Replay**: Event sourcing for point-in-time reconstruction
- ✅ **Multi-Agent Support**: Agent isolation with shared memory capabilities

## Tech Stack

| Component | Technology |
|-----------|------------|
| API Framework | FastAPI (async) |
| Structured Storage | PostgreSQL 16 |
| Vector Storage | Qdrant |
| ORM | SQLAlchemy 2.0 (async) |
| LLM | OpenAI GPT-4o-mini |
| Embeddings | OpenAI text-embedding-3-small |

## Quick Start

### Prerequisites

- Docker & Docker Compose
- OpenAI API Key

### 1. Setup Environment

```bash
cd ltm-engine

# Copy environment file
cp .env.example .env

# Edit .env and add your OpenAI API key
nano .env  # or use any editor
# Set: OPENAI_API_KEY=your-key-here
```

### 2. Run with Docker (Recommended)

```bash
# Start everything (API + PostgreSQL + Qdrant)
docker-compose up --build

# Or run in background
docker-compose up --build -d
```

That's it! The API will be available at:
- API: http://localhost:8000
- Docs: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
- Health: http://localhost:8000/api/v1/health

### 3. Run the Demo

```bash
# In another terminal (requires Python + httpx)
pip install httpx
python scripts/demo.py
```

### Stop Services

```bash
docker-compose down

# To also remove data volumes
docker-compose down -v
```

---

### Alternative: Run Locally (Development)

If you want to run the API outside Docker:

```bash
# Start only databases
docker-compose up postgres qdrant -d

# Install Python dependencies
python -m venv venv
source venv/bin/activate
pip install -e .

# Run the server
python -m ltm_engine.main
```

## API Overview

### Memory Operations

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/memory/episodic` | POST | Store episodic memory |
| `/api/v1/memory/semantic` | POST | Store semantic memory |
| `/api/v1/memory/procedural` | POST | Store procedural memory |
| `/api/v1/memory/retrieve` | POST | Hybrid search |
| `/api/v1/memory/semantic/{id}` | PUT | Update semantic memory |
| `/api/v1/memory/consolidate` | POST | Consolidate episodic → semantic |
| `/api/v1/memory/decay` | POST | Apply importance decay |
| `/api/v1/memory/forget` | POST | Remove memories |

### Temporal Operations

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/memory/temporal/at` | GET | Query memory at specific time |
| `/api/v1/memory/temporal/evolution/{id}` | GET | Get memory version history |
| `/api/v1/memory/replay` | POST | Replay state at time |
| `/api/v1/memory/timeline` | GET | Get agent event timeline |

### Confidence & Conflict

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/memory/confidence/feedback` | POST | Update confidence from feedback |
| `/api/v1/memory/conflict/detect` | POST | Detect conflicts |
| `/api/v1/memory/conflict/audit` | GET | Find all conflicts |

## Usage Examples

### Store Episodic Memory

```python
import httpx

async with httpx.AsyncClient() as client:
    response = await client.post(
        "http://localhost:8000/api/v1/memory/episodic",
        params={"agent_id": "my_agent"},
        json={
            "content": "User asked about Python programming",
            "source": "conversation",
            "session_id": "session_001",
            "actor": "user",
        }
    )
```

### Store Semantic Memory

```python
response = await client.post(
    "http://localhost:8000/api/v1/memory/semantic",
    params={"agent_id": "my_agent"},
    json={
        "content": "The user is a Python developer",
        "subject": "user_profile",
        "category": "skills",
    }
)
```

### Retrieve Memories

```python
response = await client.post(
    "http://localhost:8000/api/v1/memory/retrieve",
    json={
        "query": "What does the user know about programming?",
        "top_k": 5,
        "agent_id": "my_agent",
        "filters": {
            "memory_types": ["semantic", "episodic"]
        }
    }
)
```

### Query Historical State

```python
from datetime import datetime, timezone

response = await client.get(
    "http://localhost:8000/api/v1/memory/temporal/at",
    params={
        "subject": "user_profile",
        "as_of": "2024-01-15T10:00:00Z",
        "agent_id": "my_agent",
    }
)
```

## Configuration

Key environment variables (see `.env.example` for full list):

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key | (required) |
| `LLM_MODEL` | LLM model name | gpt-4o-mini |
| `EMBEDDING_MODEL` | Embedding model | text-embedding-3-small |
| `POSTGRES_HOST` | PostgreSQL host | localhost |
| `QDRANT_HOST` | Qdrant host | localhost |
| `DECAY_HALF_LIFE_DAYS` | Memory decay half-life | 30 |

## Architecture

See [ARCHITECTURE.md](docs/ARCHITECTURE.md) for detailed system design documentation.

## Project Structure

```
ltm-engine/
├── src/ltm_engine/
│   ├── api/            # FastAPI routes
│   ├── models/         # SQLAlchemy models
│   ├── schemas/        # Pydantic schemas
│   ├── repositories/   # Storage layer
│   ├── services/       # Business logic
│   ├── providers/      # Embedding & LLM providers
│   └── utils/          # Scoring utilities
├── scripts/
│   └── demo.py         # Demo script
├── docs/
│   └── ARCHITECTURE.md # Architecture documentation
├── docker-compose.yml  # Infrastructure
└── pyproject.toml      # Project configuration
```

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run linting
ruff check src/

# Run type checking
mypy src/

# Run tests
pytest
```

## License

MIT License
