import logging
from typing import Any, Dict, List, Optional
import asyncpg
from pgvector.asyncpg import register_vector
from app.config import settings

logger = logging.getLogger(__name__)

pool: Optional[asyncpg.Pool] = None


async def init_connection(conn: asyncpg.Connection):
    """Register pgvector codec on new connections."""
    await register_vector(conn)


async def get_db_pool() -> asyncpg.Pool:
    global pool
    if pool is None:
        # First ensure extension exists before pool connections try to register the vector type
        conn = await asyncpg.connect(
            host=settings.POSTGRES_HOST,
            port=settings.POSTGRES_PORT,
            user=settings.POSTGRES_USER,
            password=settings.POSTGRES_PASSWORD,
            database=settings.POSTGRES_DB,
        )
        try:
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        finally:
            await conn.close()

        pool = await asyncpg.create_pool(
            host=settings.POSTGRES_HOST,
            port=settings.POSTGRES_PORT,
            user=settings.POSTGRES_USER,
            password=settings.POSTGRES_PASSWORD,
            database=settings.POSTGRES_DB,
            min_size=2,
            max_size=10,
            init=init_connection,
        )
    return pool


async def close_db_pool():
    global pool
    if pool:
        await pool.close()
        pool = None
        logger.info("Database connection pool closed.")


async def init_db():
    """Create extensions and table schema matching specifications."""
    db_pool = await get_db_pool()
    async with db_pool.acquire() as conn:
        # Enable pgvector extension
        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")

        # Create chat_memory_store table
        await conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS chat_memory_store (
                id SERIAL PRIMARY KEY,
                session_id VARCHAR(255) NOT NULL,
                turn_index INT NOT NULL,
                prompt_summary TEXT NOT NULL,
                answer_summary TEXT NOT NULL,
                user_message TEXT,
                assistant_message TEXT,
                prompt_embedding VECTOR({settings.EMBEDDING_DIM}),
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            );
            """
        )

        # Create session + turn_index composite index
        await conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_chat_memory_session_turn 
            ON chat_memory_store (session_id, turn_index);
            """
        )

        # Create HNSW index for vector cosine similarity search
        await conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_chat_memory_hnsw 
            ON chat_memory_store USING hnsw (prompt_embedding vector_cosine_ops);
            """
        )
        logger.info("PostgreSQL chat_memory_store schema and HNSW indexes verified.")


async def get_session_turn_count(session_id: str) -> int:
    """Return the total number of past turns recorded for a given session."""
    db_pool = await get_db_pool()
    async with db_pool.acquire() as conn:
        count = await conn.fetchval(
            "SELECT COUNT(*) FROM chat_memory_store WHERE session_id = $1;",
            session_id,
        )
        return count or 0


async def get_recent_turns(session_id: str, limit: int = 3) -> List[Dict[str, Any]]:
    """Retrieve the most recent N turns in chronological order."""
    db_pool = await get_db_pool()
    async with db_pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, session_id, turn_index, prompt_summary, answer_summary,
                   user_message, assistant_message, created_at
            FROM chat_memory_store
            WHERE session_id = $1
            ORDER BY turn_index DESC
            LIMIT $2;
            """,
            session_id,
            limit,
        )
        # Reverse to chronological order (oldest to newest among the recent window)
        records = [dict(row) for row in reversed(rows)]
        return records


async def search_vector_memory(
    session_id: str,
    query_embedding: List[float],
    limit: int = 2,
    threshold: float = 0.80,
) -> List[Dict[str, Any]]:
    """
    Search past turns in session using cosine similarity:
    similarity = 1 - (prompt_embedding <=> query_embedding)
    Returns only matches where similarity >= threshold.
    """
    db_pool = await get_db_pool()
    async with db_pool.acquire() as conn:
        # Cosine distance operator is <=>
        # Cosine similarity = 1 - distance
        rows = await conn.fetch(
            """
            SELECT id, session_id, turn_index, prompt_summary, answer_summary,
                   1 - (prompt_embedding <=> $2) AS similarity
            FROM chat_memory_store
            WHERE session_id = $1 AND prompt_embedding IS NOT NULL
            ORDER BY prompt_embedding <=> $2 ASC
            LIMIT $3;
            """,
            session_id,
            query_embedding,
            limit,
        )

        matches = []
        for r in rows:
            sim = float(r["similarity"])
            if sim >= threshold:
                matches.append({
                    "id": r["id"],
                    "session_id": r["session_id"],
                    "turn_index": r["turn_index"],
                    "prompt_summary": r["prompt_summary"],
                    "answer_summary": r["answer_summary"],
                    "similarity": round(sim, 4),
                })
        return matches


async def insert_memory_turn(
    session_id: str,
    turn_index: int,
    prompt_summary: str,
    answer_summary: str,
    prompt_embedding: List[float],
    user_message: Optional[str] = None,
    assistant_message: Optional[str] = None,
) -> int:
    """Insert a new summarized turn with its embedding vector."""
    db_pool = await get_db_pool()
    async with db_pool.acquire() as conn:
        inserted_id = await conn.fetchval(
            """
            INSERT INTO chat_memory_store (
                session_id, turn_index, prompt_summary, answer_summary,
                prompt_embedding, user_message, assistant_message
            ) VALUES ($1, $2, $3, $4, $5, $6, $7)
            RETURNING id;
            """,
            session_id,
            turn_index,
            prompt_summary,
            answer_summary,
            prompt_embedding,
            user_message,
            assistant_message,
        )
        return inserted_id


async def get_all_session_turns(session_id: str) -> List[Dict[str, Any]]:
    """Retrieve full history of a session for debugging and inspection."""
    db_pool = await get_db_pool()
    async with db_pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, session_id, turn_index, prompt_summary, answer_summary,
                   user_message, assistant_message, created_at
            FROM chat_memory_store
            WHERE session_id = $1
            ORDER BY turn_index ASC;
            """,
            session_id,
        )
        return [dict(r) for r in rows]
