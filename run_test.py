import asyncio
import time
from httpx import AsyncClient, ASGITransport
from app.main import app
from app.database import init_db

SESSION_ID = f"session_spec_{int(time.time())}"


async def main():
    print("=" * 60)
    print("RUNNING END-TO-END HYBRID MEMORY PIPELINE TEST")
    print(f"Session ID: {SESSION_ID}")
    print("=" * 60)

    # Initialize DB schema
    await init_db()

    # Use ASGITransport to test FastAPI in-process
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver", timeout=90.0) as client:
        # Trigger lifespan startup by hitting health
        print("\n--- 1. Testing Health Endpoint ---")
        health = await client.get("/health")
        print(f"Status: {health.status_code}")
        print(f"Payload: {health.json()}")
        assert health.status_code == 200

        # Turn 1: Cold start turn
        print("\n--- 2. Turn 1: Cold Start ---")
        prompt1 = "We are building a FinTech digital banking microservice. We use PostgreSQL for our database and Kubernetes for deployment."
        r1 = await client.post("/api/v1/chat", json={"session_id": SESSION_ID, "message": prompt1})
        d1 = r1.json()
        print(f"Turn Index: {d1['turn_index']}")
        print(f"Matched Branch: {d1['matched_branch']}")
        print(f"Resolved Query: '{d1['resolved_query']}'")
        print(f"Assistant Response:\n{d1['response'][:150]}...")

        # Wait for async background task to complete summarization, embedding & DB insert
        print("Waiting for Step E background task...")
        await asyncio.sleep(4)

        # Turn 2: Second turn (still cold start / baseline building)
        print("\n--- 3. Turn 2: Second Turn ---")
        prompt2 = "For API security, we require OAuth2 with asymmetric JWT RS256 token verification."
        r2 = await client.post("/api/v1/chat", json={"session_id": SESSION_ID, "message": prompt2})
        d2 = r2.json()
        print(f"Turn Index: {d2['turn_index']}")
        print(f"Matched Branch: {d2['matched_branch']}")
        print(f"Assistant Response:\n{d2['response'][:150]}...")

        print("Waiting for Step E background task...")
        await asyncio.sleep(4)

        # Turn 3: The Pronoun Fix & Semantic Memory Search
        print("\n--- 4. Turn 3: Query Resolution & Vector Similarity Search ---")
        prompt3 = "How do I deploy it?"
        r3 = await client.post("/api/v1/chat", json={"session_id": SESSION_ID, "message": prompt3})
        d3 = r3.json()
        print(f"Turn Index: {d3['turn_index']}")
        print(f"Resolved Query: '{d3['resolved_query']}'")
        print(f"Matched Branch: {d3['matched_branch']}")
        print(f"Similarity Score: {d3.get('similarity_score')}")
        print(f"Retrieved Historical Items: {len(d3.get('historical_context', []))}")
        for item in d3.get("historical_context", []):
            print(f"  -> Turn {item['turn_index']} (sim: {item['similarity']}): {item['prompt_summary']}")
        print(f"Assistant Response:\n{d3['response'][:250]}...")

        print("Waiting for Step E background task...")
        await asyncio.sleep(4)

        # Turn 4: State Overwrite Conflict (Explicit constraint update)
        print("\n--- 5. Turn 4: State Overwrite / Constraint Update ---")
        prompt4 = "Change of plans: Ignore Kubernetes completely. Our team has decided to deploy strictly to AWS ECS Fargate."
        r4 = await client.post("/api/v1/chat", json={"session_id": SESSION_ID, "message": prompt4})
        d4 = r4.json()
        print(f"Turn Index: {d4['turn_index']}")
        print(f"Matched Branch: {d4['matched_branch']}")
        print(f"Assistant Response:\n{d4['response'][:150]}...")

        print("Waiting for Step E background task...")
        await asyncio.sleep(4)

        # Turn 5: Recency Verification
        print("\n--- 6. Turn 5: Recency Precedence Verification ---")
        prompt5 = "Where are we deploying our service now?"
        r5 = await client.post("/api/v1/chat", json={"session_id": SESSION_ID, "message": prompt5})
        d5 = r5.json()
        print(f"Turn Index: {d5['turn_index']}")
        print(f"Resolved Query: '{d5['resolved_query']}'")
        print(f"Matched Branch: {d5['matched_branch']}")
        print(f"Assistant Response:\n{d5['response']}")

        print("Waiting for Step E background task...")
        await asyncio.sleep(4)

        # 7. Check Memory Store
        print("\n--- 7. Inspecting Stored Memory in PostgreSQL ---")
        mem = await client.get(f"/api/v1/memory/{SESSION_ID}")
        mem_data = mem.json()
        print(f"Total turns stored in chat_memory_store: {mem_data['total_turns']}")
        for t in mem_data.get("turns", []):
            print(f"\n[Turn {t['turn_index']}]")
            print(f"  Prompt Summary: {t['prompt_summary']}")
            print(f"  Answer Summary: {t['answer_summary']}")

    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
