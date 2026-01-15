# API Testing Guide

This guide provides step-by-step curl commands to test all LTM Engine endpoints. Run these commands in order to test the complete functionality.

## Prerequisites

1. Docker containers running: `docker compose up -d`
2. API server running: `python -m ltm_engine.main`
3. API available at: http://localhost:8000

---

## Test 1: Health Check

Verify the system is running and connected to databases.

```bash
curl -X GET "http://localhost:8000/api/v1/health"
```

**Expected Response:**
```json
{
  "status": "healthy",
  "postgres_connected": true,
  "qdrant_connected": true,
  "version": "1.0.0",
  "timestamp": "2026-01-15T06:00:00.000000Z"
}
```

---

## Test 2: Store Episodic Memory (Conversation)

Store a user message as an episodic memory.

```bash
curl -X POST "http://localhost:8000/api/v1/memory/episodic?agent_id=test_agent" \
  -H "Content-Type: application/json" \
  -d '{
    "content": "User asked how to learn Python programming",
    "source": "conversation",
    "session_id": "session_001",
    "actor": "user",
    "action_type": "question"
  }'
```

**Expected Response:**
```json
{
  "success": true,
  "memory_id": "uuid-here",
  "memory_type": "episodic",
  "message": "Memory stored successfully"
}
```

---

## Test 3: Store Another Episodic Memory (Assistant Response)

```bash
curl -X POST "http://localhost:8000/api/v1/memory/episodic?agent_id=test_agent" \
  -H "Content-Type: application/json" \
  -d '{
    "content": "Assistant recommended starting with Python basics and practicing on coding challenges",
    "source": "conversation",
    "session_id": "session_001",
    "actor": "assistant",
    "action_type": "response"
  }'
```

---

## Test 4: Store Semantic Memory (Fact)

Store a fact about the user. **Save the `memory_id` from the response for later tests.**

```bash
curl -X POST "http://localhost:8000/api/v1/memory/semantic?agent_id=test_agent" \
  -H "Content-Type: application/json" \
  -d '{
    "content": "The user is a beginner programmer with no prior coding experience",
    "subject": "user_skill_level",
    "category": "user_profile"
  }'
```

---

## Test 5: Duplicate Detection

Store the same content again. Should return the **same `memory_id`** (no duplicate created).

```bash
curl -X POST "http://localhost:8000/api/v1/memory/semantic?agent_id=test_agent" \
  -H "Content-Type: application/json" \
  -d '{
    "content": "The user is a beginner programmer with no prior coding experience",
    "subject": "user_skill_level",
    "category": "user_profile"
  }'
```

**Expected:** Same `memory_id` as Test 4.

---

## Test 6: Update Semantic Memory (Versioning)

Update the semantic memory to create Version 2. Replace `<MEMORY_ID>` with the ID from Test 4.

```bash
curl -X PUT "http://localhost:8000/api/v1/memory/semantic/<MEMORY_ID>?agent_id=test_agent" \
  -H "Content-Type: application/json" \
  -d '{
    "content": "The user now has basic Python knowledge after completing a tutorial",
    "change_reason": "User completed Python basics tutorial"
  }'
```

**Expected:** Response shows `version: 2`.

---

## Test 7: Store Procedural Memory (Preference)

Store a user preference.

```bash
curl -X POST "http://localhost:8000/api/v1/memory/procedural?agent_id=test_agent" \
  -H "Content-Type: application/json" \
  -d '{
    "key": "ui_theme",
    "value": {"mode": "dark", "font_size": 14},
    "value_text": "dark mode with 14px font",
    "category": "preference"
  }'
```

---

## Test 8: Get Preference by Key

Retrieve the stored preference.

```bash
curl -X GET "http://localhost:8000/api/v1/memory/procedural/key/ui_theme?agent_id=test_agent"
```

**Expected:** Returns the preference with `key: "ui_theme"`.

---

## Test 9: Hybrid Retrieval (Search)

Search for relevant memories using semantic + recency + frequency scoring.

```bash
curl -X POST "http://localhost:8000/api/v1/memory/retrieve" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What programming skills does the user have?",
    "agent_id": "test_agent",
    "top_k": 5
  }'
```

**Expected:** Returns ranked memories with individual scores:
- `semantic_score`: Embedding similarity
- `recency_score`: Time-based decay
- `frequency_score`: Access count
- `confidence_score`: Memory confidence
- `score`: Combined final score

---

## Test 10: Memory Evolution (Version History)

View how a semantic memory evolved over time. Replace `<MEMORY_ID>` with the ID from Test 4.

```bash
curl -X GET "http://localhost:8000/api/v1/memory/temporal/evolution/<MEMORY_ID>?agent_id=test_agent"
```

**Expected:** Shows all versions with timestamps and change reasons.

---

## Test 11: Conflict Detection

Store a contradicting fact. The system should detect the conflict.

```bash
curl -X POST "http://localhost:8000/api/v1/memory/semantic?agent_id=test_agent" \
  -H "Content-Type: application/json" \
  -d '{
    "content": "The user is an expert Python developer with 10 years experience",
    "subject": "user_skill_level",
    "category": "user_profile"
  }'
```

**Expected Response:**
```json
{
  "success": true,
  "memory_id": "uuid-here",
  "memory_type": "semantic",
  "conflict_detected": true,
  "conflict_info": {
    "existing_memory_id": "...",
    "existing_content": "The user now has basic Python knowledge...",
    "new_content": "The user is an expert Python developer...",
    "is_contradiction": true,
    "llm_analysis": "These statements contradict each other...",
    "resolution": "reject_new"
  }
}
```

---

## Test 12: Multi-Agent Isolation (Store for Different Agent)

Store a memory for a different agent.

```bash
curl -X POST "http://localhost:8000/api/v1/memory/semantic?agent_id=agent_bob" \
  -H "Content-Type: application/json" \
  -d '{
    "content": "Bob user prefers formal communication style",
    "subject": "communication_preference",
    "category": "preference"
  }'
```

---

## Test 13: Multi-Agent Isolation (Verify Isolation)

Search as `test_agent` - should NOT see Bob's memory.

```bash
curl -X POST "http://localhost:8000/api/v1/memory/retrieve" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "communication preference",
    "agent_id": "test_agent",
    "top_k": 5
  }'
```

**Expected:** Bob's memory should NOT appear in results.

---

## Test 14: Multi-Agent Isolation (Bob's View)

Search as `agent_bob` - should see Bob's memory.

```bash
curl -X POST "http://localhost:8000/api/v1/memory/retrieve" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "communication preference",
    "agent_id": "agent_bob",
    "top_k": 5
  }'
```

**Expected:** Returns Bob's memory.

---

## Test 15: List All Agents

View all agents with their memory statistics.

```bash
curl -X GET "http://localhost:8000/api/v1/agents"
```

**Expected Response:**
```json
{
  "agents": [
    {
      "agent_id": "test_agent",
      "episodic_count": 2,
      "semantic_count": 3,
      "procedural_count": 1,
      "total_memories": 6
    },
    {
      "agent_id": "agent_bob",
      "episodic_count": 0,
      "semantic_count": 1,
      "procedural_count": 0,
      "total_memories": 1
    }
  ],
  "total_agents": 2
}
```

---

## Test 16: Agent Timeline (Event History)

View the event log for an agent.

```bash
curl -X GET "http://localhost:8000/api/v1/memory/timeline?agent_id=test_agent&limit=10"
```

**Expected:** Returns chronological list of memory events.

---

## Test 17: Deterministic Replay

Reconstruct memory state at a specific point in time.

```bash
curl -X POST "http://localhost:8000/api/v1/memory/replay" \
  -H "Content-Type: application/json" \
  -d '{
    "agent_id": "test_agent",
    "as_of": "2026-01-15T06:05:00Z"
  }'
```

**Expected:** Returns the exact memory state as it existed at that timestamp.

---

## Test 18: Memory Decay

Apply time-based importance decay to memories.

```bash
curl -X POST "http://localhost:8000/api/v1/memory/decay" \
  -H "Content-Type: application/json" \
  -d '{
    "agent_id": "test_agent",
    "memory_types": ["episodic", "semantic"],
    "half_life_days": 30
  }'
```

**Expected:** Returns count of affected memories.

---

## Test 19: Forget Memory (Soft Delete)

Soft delete memories (marks as inactive but keeps data).

```bash
curl -X POST "http://localhost:8000/api/v1/memory/forget" \
  -H "Content-Type: application/json" \
  -d '{
    "agent_id": "test_agent",
    "memory_types": ["semantic"],
    "policy": "soft_delete",
    "max_count": 1
  }'
```

**Expected:** Only 1 memory is soft-deleted.

---

## Test 20: Verify Soft Delete

Search again - soft-deleted memories should not appear.

```bash
curl -X POST "http://localhost:8000/api/v1/memory/retrieve" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Python programming",
    "agent_id": "test_agent",
    "top_k": 10
  }'
```

**Expected:** Fewer results than before (soft-deleted memories hidden).

---

## Test 21: Memory Consolidation

Consolidate episodic memories into semantic knowledge (requires 5+ episodic memories).

First, add more episodic memories:

```bash
curl -X POST "http://localhost:8000/api/v1/memory/episodic?agent_id=test_agent" \
  -H "Content-Type: application/json" \
  -d '{"content": "User discussed data analysis with pandas", "source": "conversation", "session_id": "session_001", "actor": "user", "action_type": "question"}'
```

```bash
curl -X POST "http://localhost:8000/api/v1/memory/episodic?agent_id=test_agent" \
  -H "Content-Type: application/json" \
  -d '{"content": "Assistant explained DataFrame operations", "source": "conversation", "session_id": "session_001", "actor": "assistant", "action_type": "response"}'
```

```bash
curl -X POST "http://localhost:8000/api/v1/memory/episodic?agent_id=test_agent" \
  -H "Content-Type: application/json" \
  -d '{"content": "User asked about data visualization", "source": "conversation", "session_id": "session_001", "actor": "user", "action_type": "question"}'
```

Then consolidate:

```bash
curl -X POST "http://localhost:8000/api/v1/memory/consolidate" \
  -H "Content-Type: application/json" \
  -d '{
    "agent_id": "test_agent",
    "group_by": "session"
  }'
```

**Expected:** Creates a new semantic memory summarizing the episodic memories.

---

## Reset for Fresh Testing

To start fresh:

```bash
# Stop API server (Ctrl+C)

# Remove all data
docker compose down -v

# Start fresh
docker compose up -d

# Start API
python -m ltm_engine.main
```

---

## Quick Reference

| Feature | Endpoint | Method |
|---------|----------|--------|
| Health Check | `/api/v1/health` | GET |
| Store Episodic | `/api/v1/memory/episodic` | POST |
| Store Semantic | `/api/v1/memory/semantic` | POST |
| Store Procedural | `/api/v1/memory/procedural` | POST |
| Update Semantic | `/api/v1/memory/semantic/{id}` | PUT |
| Get Preference | `/api/v1/memory/procedural/key/{key}` | GET |
| Search | `/api/v1/memory/retrieve` | POST |
| Version History | `/api/v1/memory/temporal/evolution/{id}` | GET |
| Timeline | `/api/v1/memory/timeline` | GET |
| Replay | `/api/v1/memory/replay` | POST |
| Consolidate | `/api/v1/memory/consolidate` | POST |
| Decay | `/api/v1/memory/decay` | POST |
| Forget | `/api/v1/memory/forget` | POST |
| List Agents | `/api/v1/agents` | GET |

---

## Swagger UI

For interactive testing, use the built-in Swagger documentation:

**http://localhost:8000/docs**

This provides a visual interface to test all endpoints with request/response examples.
