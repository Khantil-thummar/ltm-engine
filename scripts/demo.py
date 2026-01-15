#!/usr/bin/env python3
"""
LTM Engine Demo Script

Demonstrates key features:
1. Storing episodic memories
2. Creating and updating semantic memories with evolving facts
3. Querying historical vs current memory state
4. Conflict detection and resolution
5. Consolidation of episodic memories into semantic
6. Multi-agent memory isolation

Usage:
    python scripts/demo.py

Requirements:
    - LTM Engine server running at http://localhost:8000
    - PostgreSQL and Qdrant running (via docker-compose)
"""

import asyncio
import time
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx

BASE_URL = "http://localhost:8000/api/v1"
AGENT_1 = "agent_alice"
AGENT_2 = "agent_bob"


async def print_response(title: str, response: dict[str, Any]) -> None:
    """Pretty print a response."""
    print(f"\n{'='*60}")
    print(f"📌 {title}")
    print(f"{'='*60}")
    
    import json
    print(json.dumps(response, indent=2, default=str))


async def main():
    """Run the demo."""
    async with httpx.AsyncClient(timeout=60.0) as client:
        print("\n" + "="*60)
        print("🚀 LTM ENGINE DEMO")
        print("="*60)

        # Check health
        print("\n📍 Checking system health...")
        response = await client.get(f"{BASE_URL}/health")
        health = response.json()
        await print_response("Health Check", health)

        if health["status"] != "healthy":
            print("❌ System is not healthy. Please check the services.")
            return

        # =====================================================================
        # 1. EPISODIC MEMORIES
        # =====================================================================
        print("\n\n" + "="*60)
        print("📝 SECTION 1: EPISODIC MEMORIES (Conversations/Events)")
        print("="*60)

        # Store some conversation episodes
        episodes = [
            {
                "content": "User asked about Python programming basics",
                "source": "conversation",
                "session_id": "session_001",
                "actor": "user",
                "action_type": "question",
            },
            {
                "content": "Assistant explained variables, loops, and functions in Python",
                "source": "conversation",
                "session_id": "session_001",
                "actor": "assistant",
                "action_type": "response",
            },
            {
                "content": "User requested help with data analysis using pandas",
                "source": "conversation",
                "session_id": "session_001",
                "actor": "user",
                "action_type": "question",
            },
            {
                "content": "Assistant provided pandas DataFrame examples and plotting tips",
                "source": "conversation",
                "session_id": "session_001",
                "actor": "assistant",
                "action_type": "response",
            },
            {
                "content": "User mentioned they prefer dark mode interfaces",
                "source": "conversation",
                "session_id": "session_001",
                "actor": "user",
                "action_type": "preference",
            },
        ]

        episode_ids = []
        for episode in episodes:
            response = await client.post(
                f"{BASE_URL}/memory/episodic",
                params={"agent_id": AGENT_1},
                json=episode,
            )
            result = response.json()
            episode_ids.append(result["memory_id"])
            print(f"✅ Stored episodic memory: {episode['content'][:50]}...")

        await print_response("Last Episodic Memory Stored", result)

        # =====================================================================
        # 2. SEMANTIC MEMORIES WITH EVOLVING FACTS
        # =====================================================================
        print("\n\n" + "="*60)
        print("🧠 SECTION 2: SEMANTIC MEMORIES (Facts that evolve)")
        print("="*60)

        # Create initial fact
        initial_fact = {
            "content": "The user is learning Python and is at a beginner level",
            "subject": "user_skill_python",
            "category": "user_knowledge",
            "metadata": {"skill_level": "beginner"},
        }

        response = await client.post(
            f"{BASE_URL}/memory/semantic",
            params={"agent_id": AGENT_1},
            json=initial_fact,
        )
        semantic_result = response.json()
        semantic_id = semantic_result["memory_id"]
        await print_response("Initial Semantic Memory (V1)", semantic_result)

        # Store the timestamp for later temporal query
        time_after_v1 = datetime.now(timezone.utc)

        # Wait a moment to ensure different timestamps
        await asyncio.sleep(1)

        # Update the fact (user has progressed)
        update_1 = {
            "content": "The user has intermediate Python skills and is learning data science",
            "change_reason": "User completed Python basics, moved to pandas and data analysis",
            "confidence": 0.9,
        }

        response = await client.put(
            f"{BASE_URL}/memory/semantic/{semantic_id}",
            params={"agent_id": AGENT_1},
            json=update_1,
        )
        await print_response("Updated Semantic Memory (V2)", response.json())

        time_after_v2 = datetime.now(timezone.utc)
        await asyncio.sleep(1)

        # Another update
        update_2 = {
            "content": "The user is proficient in Python data science, including pandas, numpy, and matplotlib",
            "change_reason": "User demonstrated proficiency in data visualization project",
            "confidence": 0.95,
        }

        response = await client.put(
            f"{BASE_URL}/memory/semantic/{semantic_id}",
            params={"agent_id": AGENT_1},
            json=update_2,
        )
        await print_response("Updated Semantic Memory (V3 - Current)", response.json())

        # =====================================================================
        # 3. TEMPORAL QUERIES
        # =====================================================================
        print("\n\n" + "="*60)
        print("⏰ SECTION 3: TEMPORAL QUERIES (What did we believe when?)")
        print("="*60)

        # Query what we believed at different times
        print("\n📍 Querying historical memory states...")

        response = await client.get(
            f"{BASE_URL}/memory/temporal/at",
            params={
                "subject": "user_skill_python",
                "as_of": time_after_v1.isoformat(),
                "agent_id": AGENT_1,
            },
        )
        await print_response(f"Memory at {time_after_v1.strftime('%H:%M:%S')} (V1)", response.json())

        response = await client.get(
            f"{BASE_URL}/memory/temporal/at",
            params={
                "subject": "user_skill_python",
                "as_of": time_after_v2.isoformat(),
                "agent_id": AGENT_1,
            },
        )
        await print_response(f"Memory at {time_after_v2.strftime('%H:%M:%S')} (V2)", response.json())

        # Get evolution
        response = await client.get(
            f"{BASE_URL}/memory/temporal/evolution/{semantic_id}",
            params={"agent_id": AGENT_1},
        )
        await print_response("Memory Evolution Over Time", response.json())

        # =====================================================================
        # 4. CONFLICT DETECTION
        # =====================================================================
        print("\n\n" + "="*60)
        print("⚔️ SECTION 4: CONFLICT DETECTION")
        print("="*60)

        # Create a potentially conflicting fact
        conflicting_fact = {
            "content": "The user is a complete beginner with no programming experience",
            "subject": "user_skill_python",
            "category": "user_knowledge",
        }

        print("\n📍 Storing potentially conflicting fact...")
        response = await client.post(
            f"{BASE_URL}/memory/semantic",
            params={"agent_id": AGENT_1, "check_conflicts": "true"},
            json=conflicting_fact,
        )
        await print_response("Conflict Detection Result", response.json())

        # =====================================================================
        # 5. PROCEDURAL MEMORIES (Preferences)
        # =====================================================================
        print("\n\n" + "="*60)
        print("⚙️ SECTION 5: PROCEDURAL MEMORIES (Preferences)")
        print("="*60)

        preferences = [
            {
                "key": "ui_theme",
                "value": {"mode": "dark", "accent_color": "blue"},
                "value_text": "dark mode with blue accent",
                "category": "preference",
                "source": "explicit",
            },
            {
                "key": "communication_style",
                "value": {"verbosity": "concise", "formality": "casual"},
                "value_text": "concise and casual",
                "category": "preference",
                "source": "inferred",
                "confidence": 0.8,
            },
            {
                "key": "code_style",
                "value": {"language": "python", "comments": True, "type_hints": True},
                "value_text": "Python with comments and type hints",
                "category": "preference",
                "source": "explicit",
            },
        ]

        for pref in preferences:
            response = await client.post(
                f"{BASE_URL}/memory/procedural",
                params={"agent_id": AGENT_1},
                json=pref,
            )
            result = response.json()
            print(f"✅ Stored preference: {pref['key']}")

        await print_response("Last Procedural Memory", result)

        # Retrieve a preference by key
        response = await client.get(
            f"{BASE_URL}/memory/procedural/key/ui_theme",
            params={"agent_id": AGENT_1},
        )
        await print_response("Retrieved Preference by Key", response.json())

        # =====================================================================
        # 6. HYBRID RETRIEVAL
        # =====================================================================
        print("\n\n" + "="*60)
        print("🔍 SECTION 6: HYBRID RETRIEVAL (Semantic + Recency + Frequency)")
        print("="*60)

        # Retrieve memories
        retrieve_request = {
            "query": "What does the user know about programming and data science?",
            "top_k": 5,
            "agent_id": AGENT_1,
            "filters": {
                "memory_types": ["semantic", "episodic"],
            },
        }

        response = await client.post(
            f"{BASE_URL}/memory/retrieve",
            json=retrieve_request,
        )
        await print_response("Hybrid Retrieval Results", response.json())

        # =====================================================================
        # 7. CONSOLIDATION
        # =====================================================================
        print("\n\n" + "="*60)
        print("🔄 SECTION 7: MEMORY CONSOLIDATION (Episodic → Semantic)")
        print("="*60)

        consolidate_request = {
            "agent_id": AGENT_1,
            "min_memories": 3,
            "max_memories": 10,
            "force": True,  # Force even if below threshold
        }

        response = await client.post(
            f"{BASE_URL}/memory/consolidate",
            json=consolidate_request,
        )
        await print_response("Consolidation Result", response.json())

        # =====================================================================
        # 8. MULTI-AGENT ISOLATION
        # =====================================================================
        print("\n\n" + "="*60)
        print("👥 SECTION 8: MULTI-AGENT MEMORY ISOLATION")
        print("="*60)

        # Store memory for Agent 2
        agent2_memory = {
            "content": "This user prefers light mode and formal communication",
            "subject": "user_preferences",
            "category": "preference",
        }

        response = await client.post(
            f"{BASE_URL}/memory/semantic",
            params={"agent_id": AGENT_2},
            json=agent2_memory,
        )
        await print_response(f"Memory stored for {AGENT_2}", response.json())

        # Query each agent's memories
        for agent in [AGENT_1, AGENT_2]:
            retrieve_request = {
                "query": "user preferences",
                "top_k": 3,
                "agent_id": agent,
            }
            response = await client.post(
                f"{BASE_URL}/memory/retrieve",
                json=retrieve_request,
            )
            await print_response(f"Memories for {agent}", response.json())

        # =====================================================================
        # 9. REPLAY / EVENT SOURCING
        # =====================================================================
        print("\n\n" + "="*60)
        print("🔁 SECTION 9: DETERMINISTIC REPLAY")
        print("="*60)

        # Get timeline
        response = await client.get(
            f"{BASE_URL}/memory/timeline",
            params={"agent_id": AGENT_1, "limit": 10},
        )
        await print_response("Agent Timeline (Last 10 Events)", response.json())

        # Replay state at a specific time
        replay_request = {
            "agent_id": AGENT_1,
            "as_of": time_after_v1.isoformat(),
            "include_events": True,
        }
        response = await client.post(
            f"{BASE_URL}/memory/replay",
            json=replay_request,
        )
        await print_response(f"Replayed State at {time_after_v1.strftime('%H:%M:%S')}", response.json())

        # =====================================================================
        # 10. DECAY
        # =====================================================================
        print("\n\n" + "="*60)
        print("📉 SECTION 10: MEMORY DECAY")
        print("="*60)

        decay_request = {
            "agent_id": AGENT_1,
            "memory_types": ["episodic", "semantic"],
            "half_life_days": 30,
            "min_importance": 0.1,
        }

        response = await client.post(
            f"{BASE_URL}/memory/decay",
            json=decay_request,
        )
        await print_response("Decay Result", response.json())

        # =====================================================================
        # SUMMARY
        # =====================================================================
        print("\n\n" + "="*60)
        print("✅ DEMO COMPLETE")
        print("="*60)
        print("""
Demonstrated Features:
1. ✅ Episodic Memory (append-only events)
2. ✅ Semantic Memory (versioned facts)
3. ✅ Procedural Memory (preferences)
4. ✅ Temporal Queries (historical state)
5. ✅ Memory Evolution Tracking
6. ✅ Conflict Detection
7. ✅ Hybrid Retrieval (semantic + recency + frequency)
8. ✅ Memory Consolidation
9. ✅ Multi-Agent Isolation
10. ✅ Deterministic Replay
11. ✅ Memory Decay
        """)


if __name__ == "__main__":
    asyncio.run(main())
