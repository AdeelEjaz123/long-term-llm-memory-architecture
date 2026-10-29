import asyncio
import time
import httpx


BASE_URL = "http://127.0.0.1:8000"
SESSION_ID = f"test_session_{int(time.time())}"


async def run_pipeline_test():
    print(f"==================================================")
    print(f"Starting E2E Semantic RAG Conversational Memory Test")
    print(f"Session ID: {SESSION_ID}")
    print(f"==================================================\n")

    async with httpx.AsyncClient(base_url=BASE_URL, timeout=60.0) as client:
        # 1. Health Check
        print("1. Checking /health endpoint...")
        health_resp = await client.get("/health")
        print(f"Health status: {health_resp.status_code}")
        print(f"Response: {health_resp.json()}\n")
        assert health_resp.status_code == 200, "Health check failed!"

        # 2. Turn 1 - Cold Start
        print("--------------------------------------------------")
        print("Turn 1: Project definition (Cold Start expected)")
        turn1_msg = "We are designing a FinTech digital banking microservice architecture. It uses PostgreSQL for transactions and Kubernetes for container deployment."
        resp1 = await client.post("/api/v1/chat", json={"session_id": SESSION_ID, "message": turn1_msg})
        data1 = resp1.json()
        print(f"Turn: {data1.get('turn_index')}")
        print(f"Branch: {data1.get('matched_branch')}")
        print(f"Assistant: {data1.get('response')[:120]}...\n")

        # Allow background post-processing task to finish embedding and storing
        print("Waiting 3s for background summarization and embedding...")
        await asyncio.sleep(3)

        # 3. Turn 2 - Adding details
        print("--------------------------------------------------")
        print("Turn 2: Adding security constraints (Cold Start / Recency)")
        turn2_msg = "For authentication, we decided to use OAuth2 with RS256 signed JWT tokens."
        resp2 = await client.post("/api/v1/chat", json={"session_id": SESSION_ID, "message": turn2_msg})
        data2 = resp2.json()
        print(f"Turn: {data2.get('turn_index')}")
        print(f"Branch: {data2.get('matched_branch')}")
        print(f"Assistant: {data2.get('response')[:120]}...\n")

        print("Waiting 3s for background processing...")
        await asyncio.sleep(3)

        # 4. Turn 3 - The Pronoun Fix & Semantic Memory Retrieval
        print("--------------------------------------------------")
        print("Turn 3: Pronoun query ('How do I deploy it?')")
        turn3_msg = "How do I deploy it?"
        resp3 = await client.post("/api/v1/chat", json={"session_id": SESSION_ID, "message": turn3_msg})
        data3 = resp3.json()
        print(f"Turn: {data3.get('turn_index')}")
        print(f"Resolved Query: '{data3.get('resolved_query')}'")
        print(f"Matched Branch: {data3.get('matched_branch')}")
        print(f"Similarity Score: {data3.get('similarity_score')}")
        print(f"Historical Context Retrieved: {len(data3.get('historical_context', []))} item(s)")
        for idx, item in enumerate(data3.get('historical_context', []), 1):
            print(f"  [{idx}] Turn {item['turn_index']} (sim: {item['similarity']}): {item['prompt_summary']}")
        print(f"Assistant: {data3.get('response')[:200]}...\n")

        print("Waiting 3s for background processing...")
        await asyncio.sleep(3)

        # 5. Turn 4 - State Overwrite Conflict
        print("--------------------------------------------------")
        print("Turn 4: Overriding constraint (Ignore Kubernetes, switch to AWS ECS Fargate)")
        turn4_msg = "Important update: Ignore Kubernetes completely. We have decided to switch our deployment target exclusively to AWS ECS Fargate."
        resp4 = await client.post("/api/v1/chat", json={"session_id": SESSION_ID, "message": turn4_msg})
        data4 = resp4.json()
        print(f"Turn: {data4.get('turn_index')}")
        print(f"Branch: {data4.get('matched_branch')}")
        print(f"Assistant: {data4.get('response')[:120]}...\n")

        print("Waiting 3s for background processing...")
        await asyncio.sleep(3)

        # 6. Turn 5 - Verifying Recency Precedence
        print("--------------------------------------------------")
        print("Turn 5: Verifying state override precedence ('Where are we deploying?')")
        turn5_msg = "Where are we deploying our microservice now?"
        resp5 = await client.post("/api/v1/chat", json={"session_id": SESSION_ID, "message": turn5_msg})
        data5 = resp5.json()
        print(f"Turn: {data5.get('turn_index')}")
        print(f"Resolved Query: '{data5.get('resolved_query')}'")
        print(f"Branch: {data5.get('matched_branch')}")
        print(f"Assistant: {data5.get('response')}\n")

        print("Waiting 3s for background processing...")
        await asyncio.sleep(3)

        # 7. Check Memory Store
        print("--------------------------------------------------")
        print("6. Inspecting chat_memory_store via /api/v1/memory/{session_id}...")
        mem_resp = await client.get(f"/api/v1/memory/{SESSION_ID}")
        mem_data = mem_resp.json()
        print(f"Total turns stored in PostgreSQL: {mem_data.get('total_turns')}")
        for t in mem_data.get("turns", []):
            print(f"  Turn {t['turn_index']}:")
            print(f"    User Prompt Summary:   {t['prompt_summary']}")
            print(f"    Answer Summary:        {t['answer_summary']}")

    print("\n==================================================")
    print("E2E Test Completed Successfully!")
    print("==================================================")


if __name__ == "__main__":
    asyncio.run(run_pipeline_test())

