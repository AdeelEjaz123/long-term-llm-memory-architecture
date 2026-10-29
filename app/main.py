import os
from contextlib import asynccontextmanager
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from app.api.routes import router as api_router
from app.config import settings
from app.database import close_db_pool, init_db

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("semantic_rag_engine")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing Semantic RAG Conversational Memory Engine...")
    # Initialize database tables and pgvector extension
    try:
        await init_db()
        logger.info("PostgreSQL & pgvector initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}", exc_info=True)
        raise

    yield

    logger.info("Shutting down Semantic RAG Conversational Memory Engine...")
    await close_db_pool()


app = FastAPI(
    title="Semantic RAG Conversational Memory Engine",
    description="Production-grade hybrid conversational memory combining pgvector similarity with recency fallback.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api/v1")
app.include_router(api_router)  # mount /health at root level


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def serve_gui():
    """Serves the interactive Semantic RAG Memory GUI."""
    static_file = os.path.join(os.path.dirname(__file__), "static", "index.html")
    with open(static_file, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


