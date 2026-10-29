import logging
from fastapi import APIRouter, BackgroundTasks, HTTPException, status
from app.database import get_all_session_turns, get_session_turn_count
from app.models.schemas import (
    ChatRequest,
    ChatResponse,
    MemoryTurnRecord,
    SessionMemoryResponse,
)
from app.services.memory_engine import memory_engine
from app.services.ollama_client import ollama_client

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/chat", response_model=ChatResponse, summary="Send message through Hybrid Memory Engine")
async def chat_endpoint(
    request: ChatRequest,
    background_tasks: BackgroundTasks,
):
    """
    Main conversational endpoint:
    - Step A: Query Resolution (Resolves pronouns with recent history)
    - Step B: Vector Similarity Search (Top-k past turns from pgvector)
    - Step C: Branching Context Injection (Branch A: Semantic Match vs Branch B: Recency Fallback)
    - Step D: Prompt Construction & LLM Generation
    - Step E: Asynchronous Post-Processing (Write Path scheduled in BackgroundTasks)
    """
    try:
        chat_response, post_process_data = await memory_engine.process_chat_turn(
            session_id=request.session_id,
            raw_user_prompt=request.message,
        )

        # Enqueue async post-processing into BackgroundTasks so response is sent immediately
        background_tasks.add_task(
            memory_engine.post_process_turn,
            session_id=post_process_data["session_id"],
            turn_index=post_process_data["turn_index"],
            raw_user_prompt=post_process_data["raw_user_prompt"],
            assistant_response=post_process_data["assistant_response"],
        )

        return chat_response
    except Exception as exc:
        logger.error(f"Error in chat endpoint for session {request.session_id}: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while processing your message: {str(exc)}",
        )


@router.get(
    "/memory/{session_id}",
    response_model=SessionMemoryResponse,
    summary="Get stored memory turns for a session",
)
async def get_session_memory(session_id: str):
    """Retrieve all summarized turns and metadata stored for a session."""
    try:
        turns = await get_all_session_turns(session_id)
        records = [
            MemoryTurnRecord(
                id=t["id"],
                session_id=t["session_id"],
                turn_index=t["turn_index"],
                prompt_summary=t["prompt_summary"],
                answer_summary=t["answer_summary"],
                user_message=t.get("user_message"),
                assistant_message=t.get("assistant_message"),
                created_at=t["created_at"],
            )
            for t in turns
        ]
        return SessionMemoryResponse(
            session_id=session_id,
            total_turns=len(records),
            turns=records,
        )
    except Exception as exc:
        logger.error(f"Failed to fetch session memory for {session_id}: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch session memory: {str(exc)}",
        )


@router.get("/health", summary="Health check endpoint")
async def health_check():
    """Verify health of FastAPI, PostgreSQL pgvector, and Ollama."""
    health_status = {
        "status": "healthy",
        "database": "unknown",
        "ollama": "unknown",
    }

    try:
        # Check DB
        await get_session_turn_count("health_check_dummy")
        health_status["database"] = "connected"
    except Exception as e:
        health_status["database"] = f"unhealthy: {str(e)}"
        health_status["status"] = "degraded"

    try:
        # Check Ollama
        emb = await ollama_client.get_embedding("health check")
        if len(emb) == 768:
            health_status["ollama"] = "connected (768-dim embeddings verified)"
        else:
            health_status["ollama"] = f"connected (unexpected dim: {len(emb)})"
    except Exception as e:
        health_status["ollama"] = f"unhealthy: {str(e)}"
        health_status["status"] = "degraded"

    return health_status

