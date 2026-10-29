# Semantic RAG Conversational Memory Engine

A high-performance hybrid conversational memory engine built with **FastAPI**, **PostgreSQL with `pgvector`**, and **Ollama** (`llama3.2` & `nomic-embed-text`).

Instead of relying on naive growing chat histories or pure RAG, this system combines **Vector Similarity Retrieval (RAG)** on turn summaries with a **Recency/Chronology Fallback** to resolve pronoun fragmentation and prioritize real-time state overrides.

---

## 🏗️ Architectural Flow

```
Incoming User Message
         │
         ▼
[Step 1: Cold Start Check] ──(≤ 1 turns)──► Bypass vector search (Linear baseline)
         │ (> 1 turns)
         ▼
[Step 2: Query Resolution]
  Rewrite: raw prompt + last 2-3 turns ──► Standalone, context-independent query
         │
         ▼
[Step 3: Vector Similarity Search]
  Embed standalone query (nomic-embed-text) ──► Query pgvector (top k=2, similarity ≥ 0.80)
         │
         ├─── (Similarity ≥ 0.80) ────► Branch A: Explicit Semantic Memory Block
         └─── (Below threshold)   ────► Branch B: Rolling Chronological Recency Block
         │
         ▼
[Step 4: LLM Generation]
  Assemble: System Rules + Context Block + Current Prompt ──► Stream/Return to User
         │
         ▼
[Step 5: Async Post-Processing (Write Path)] (BackgroundTasks)
  ├── 1-2 sentence summarization of turn
  ├── Generate 768-dim vector embedding of prompt summary
  └── Persist to `chat_memory_store` with incremented `turn_index`
```

---

## 🚀 Key Features

* **Query Resolution (The Pronoun Fix):** Uses lightweight LLM rewriting to resolve vague queries like *"How do I deploy it?"* into *"How do I deploy the FinTech digital banking microservice?"* using the immediate 2-3 turns.
* **Turn Summary Embeddings:** Stores concise 1-2 sentence summaries instead of noisy raw transcripts for higher-quality semantic retrieval.
* **Strict Cosine Thresholding ($\ge 0.80$):** Employs pgvector cosine similarity (`1 - (prompt_embedding <=> query_vec)`) to ensure only strictly relevant past decisions are injected.
* **State Overwrite & Recency Precedence:** When a user changes constraints mid-conversation (e.g. *"Ignore Kubernetes, use AWS ECS Fargate"*), recent chronology explicitly overrides older vector memory.
* **Non-Blocking Post-Processing:** Summarization, embedding, and database writes run asynchronously in FastAPI `BackgroundTasks`, keeping user chat responses immediate.
* **Graceful Degradation & Cold Start:** Sessions with $\le 1$ turn bypass vector search automatically. DB/Embedding timeouts degrade gracefully to linear conversation.

---

## 📦 Tech Stack

- **Python:** 3.11+
- **Backend:** FastAPI, AsyncIO, Uvicorn, httpx
- **Database:** PostgreSQL 16 + `pgvector` (HNSW cosine index)
- **Embedding Model:** `nomic-embed-text:latest` (768 dimensions via Ollama)
- **Language Model:** `llama3.2:latest` (3B via Ollama)

---

## 🛠️ Quick Start

### 1. Prerequisites

Ensure you have Ollama installed with the required models:
```bash
ollama pull llama3.2:latest
ollama pull nomic-embed-text:latest
```

### 2. Start PostgreSQL with pgvector

Launch the included Docker Compose container:
```bash
docker compose up -d
```

### 3. Setup Virtual Environment & Install Dependencies

```bash
python -m venv .venv

# On Windows:
.\.venv\Scripts\activate

# On Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

### 4. Configure Environment

Copy `.env.example` to `.env` (pre-configured with defaults):
```env
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=semantic_memory
POSTGRES_USER=rag_user
POSTGRES_PASSWORD=rag_password

OLLAMA_BASE_URL=http://localhost:11434
LLM_MODEL=llama3.2:latest
FAST_RESOLVER_MODEL=llama3.2:latest
EMBEDDING_MODEL=nomic-embed-text:latest
EMBEDDING_DIM=768

SIMILARITY_THRESHOLD=0.80
TOP_K=2
RECENT_TURNS_LIMIT=3
```

### 5. Run the API Server

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Interactive API documentation will be available at:
- **Swagger UI:** `http://localhost:8000/docs`
- **ReDoc:** `http://localhost:8000/redoc`

---

## 🧪 Running the E2E Verification Test

Run the automated integration test to see the 5-step pipeline in action:
```bash
python run_test.py
```

This test simulates:
1. **Cold Start** project definition
2. Second conversation turn
3. **Pronoun query** (`"How do I deploy it?"`), resolving to the microservice and matching Turn 1 via pgvector ($\text{sim} \ge 0.80$)
4. **State override** (`"Ignore Kubernetes, deploy to AWS ECS Fargate"`)
5. **Recency precedence verification** (`"Where are we deploying now?"`)
6. Database inspection via `/api/v1/memory/{session_id}`

---

## 📡 API Reference

### 1. Chat Turn
`POST /api/v1/chat`

**Request:**
```json
{
  "session_id": "session_user_42",
  "message": "How do I deploy it?"
}
```

**Response:**
```json
{
  "session_id": "session_user_42",
  "turn_index": 3,
  "response": "To deploy the FinTech digital banking microservice...",
  "resolved_query": "deploy digital banking microservice using Kubernetes",
  "matched_branch": "Branch A (Semantic Match)",
  "similarity_score": 0.8201,
  "historical_context": [
    {
      "turn_index": 1,
      "prompt_summary": "Building a FinTech digital banking microservice with PostgreSQL and Kubernetes",
      "answer_summary": "Discussed architecture and containerized deployment with Kubernetes",
      "similarity": 0.8201
    }
  ]
}
```

### 2. Inspect Session Memory
`GET /api/v1/memory/{session_id}`

Returns all persisted turns, summaries, and timestamps for the session.

### 3. Health Check
`GET /health`

Verifies status of FastAPI, PostgreSQL `pgvector`, and Ollama 768-dim embeddings.

