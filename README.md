# LTM Engine

**Long-Term Memory Engine for LLM-based Applications**

A production-ready memory subsystem that enables LLM-based applications (agents, copilots, assistants) to store, retrieve, update, and reason over information across sessions and time.

## Features

### Memory Types

- **Episodic Memory**: Time-stamped, immutable records of events, conversations, and interactions
- **Semantic Memory**: Versioned facts and knowledge with conflict resolution
- **Procedural Memory**: User preferences and behavioral patterns (key-value)

### Core Capabilities

- **Hybrid Retrieval**: Semantic similarity + recency decay + access frequency + confidence scoring
- **Temporal Reasoning**: Query historical states, track memory evolution over time
- **Conflict Detection**: Embedding similarity + LLM-based contradiction analysis
- **Memory Consolidation**: Summarize episodic memories into semantic knowledge
- **Lifecycle Management**: Decay, forget (soft/hard delete), compress
- **Confidence Calibration**: Bayesian-style confidence updates
- **Deterministic Replay**: Event sourcing for point-in-time reconstruction
- **Multi-Agent Support**: Agent isolation with shared memory capabilities
- **Duplicate Detection**: Automatic skip of identical memories

## Tech Stack

| Component | Technology |
|-----------|------------|
| API Framework | FastAPI (async) |
| Structured Storage | PostgreSQL 16 |
| Vector Storage | Qdrant |
| ORM | SQLAlchemy 2.0 (async) |
| LLM | OpenAI GPT-4o-mini |
| Embeddings | OpenAI text-embedding-3-small |

---

## Quick Start

### Prerequisites

- Python 3.11+
- Docker & Docker Compose
- OpenAI API Key

### 1. Clone and Setup Environment

```bash
cd ltm-engine

# Copy environment file
cp .env.example .env

# Edit .env and add your OpenAI API key
nano .env  # or use any editor
# Set: OPENAI_API_KEY=your-key-here
```

### 2. Start Database Services

```bash
# Start PostgreSQL and Qdrant in Docker
docker compose up -d

# Verify containers are running
docker compose ps
```

### 3. Install Dependencies

**Option A: Using pip (Recommended)**

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install the package
pip install -e .
```

**Option B: Using Conda**

```bash
# Create conda environment
conda create -n ltm python=3.11
conda activate ltm

# Install the package
pip install -e .
```

**Option C: Using Poetry**

```bash
# Install Poetry if not already installed
curl -sSL https://install.python-poetry.org | python3 -

# Install dependencies
poetry install

# Activate shell
poetry shell
```

### 4. Run the API Server

```bash
python -m ltm_engine.main
```

The API will be available at:
- **API**: http://localhost:8000
- **Swagger Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/api/v1/health

### 5. Test the API

See the **[API Testing Guide](docs/API_TESTING_GUIDE.md)** for step-by-step curl commands to test all endpoints.

Quick health check:
```bash
curl http://localhost:8000/api/v1/health
```

### 6. Run the Demo Script

```bash
python scripts/demo.py
```

### Stop Services

```bash
# Stop API server: Ctrl+C

# Stop Docker containers
docker compose down

# To also remove data volumes (fresh start)
docker compose down -v
```

---

## API Overview

### Memory Operations

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/memory/episodic` | POST | Store episodic memory |
| `/api/v1/memory/semantic` | POST | Store semantic memory |
| `/api/v1/memory/procedural` | POST | Store procedural memory |
| `/api/v1/memory/retrieve` | POST | Hybrid search |
| `/api/v1/memory/semantic/{id}` | PUT | Update semantic memory |
| `/api/v1/memory/procedural/key/{key}` | GET | Get preference by key |

### Temporal Operations

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/memory/temporal/evolution/{id}` | GET | Get memory version history |
| `/api/v1/memory/replay` | POST | Replay state at point in time |
| `/api/v1/memory/timeline` | GET | Get agent event timeline |

### Lifecycle Management

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/memory/consolidate` | POST | Consolidate episodic → semantic |
| `/api/v1/memory/decay` | POST | Apply importance decay |
| `/api/v1/memory/forget` | POST | Remove memories (soft/hard) |

### Agent Management

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/agents` | GET | List all agents with stats |
| `/api/v1/health` | GET | System health check |

---

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
            "action_type": "question"
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

### Retrieve Memories (Hybrid Search)

```python
response = await client.post(
    "http://localhost:8000/api/v1/memory/retrieve",
    json={
        "query": "What does the user know about programming?",
        "top_k": 5,
        "agent_id": "my_agent"
    }
)
```

---

## Configuration

Key environment variables (see `.env.example` for full list):

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key | (required) |
| `LLM_MODEL` | LLM model name | gpt-4o-mini |
| `EMBEDDING_MODEL` | Embedding model | text-embedding-3-small |
| `POSTGRES_HOST` | PostgreSQL host | localhost |
| `POSTGRES_PORT` | PostgreSQL port | 5432 |
| `QDRANT_HOST` | Qdrant host | localhost |
| `QDRANT_PORT` | Qdrant port | 6333 |
| `DECAY_HALF_LIFE_DAYS` | Memory decay half-life | 30 |

---

## Documentation

- **[API Testing Guide](docs/API_TESTING_GUIDE.md)** - Step-by-step curl commands to test all endpoints
- **[Architecture](docs/ARCHITECTURE.md)** - Detailed system design documentation

---

## Project Structure

```
ltm-engine/
├── src/ltm_engine/
│   ├── api/            # FastAPI routes
│   ├── models/         # SQLAlchemy models
│   ├── schemas/        # Pydantic schemas
│   ├── repositories/   # Storage layer (PostgreSQL, Qdrant)
│   ├── services/       # Business logic
│   ├── providers/      # Embedding & LLM providers
│   └── utils/          # Scoring utilities
├── scripts/
│   └── demo.py         # Comprehensive demo script
├── docs/
│   ├── ARCHITECTURE.md # Architecture documentation
│   └── API_TESTING_GUIDE.md # API testing guide
├── docker-compose.yml  # Database infrastructure
├── pyproject.toml      # Project configuration
└── .env.example        # Environment template
```

---

## Future Scope & Enhancements

1. **Redis Caching Layer**: Cache frequently accessed memories for sub-millisecond retrieval
2. **Cross-Encoder Re-ranking**: Use cross-encoder models for more accurate final ranking after initial retrieval
3. **Multi-Modal Memory**: Support for image and audio memory storage with vision/audio embeddings
4. **Memory Visualization Dashboard**: Web UI to explore, debug, and visualize memory graphs
5. **Horizontal Scaling**: Database sharding by agent_id + Qdrant cluster mode for enterprise workloads

---

## License

MIT License
