from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    session_id: str = Field(..., description="Unique identifier for the chat session")
    message: str = Field(..., description="User's incoming raw prompt")


class HistoricalContextItem(BaseModel):
    turn_index: int
    prompt_summary: str
    answer_summary: str
    similarity: float


class ChatResponse(BaseModel):
    session_id: str
    turn_index: int
    response: str
    resolved_query: str
    matched_branch: str  # "Branch A (Semantic Match)", "Branch B (Recency Fallback)", or "Cold Start"
    similarity_score: Optional[float] = None
    historical_context: List[HistoricalContextItem] = []


class MemoryTurnRecord(BaseModel):
    id: int
    session_id: str
    turn_index: int
    prompt_summary: str
    answer_summary: str
    user_message: Optional[str] = None
    assistant_message: Optional[str] = None
    created_at: datetime


class SessionMemoryResponse(BaseModel):
    session_id: str
    total_turns: int
    turns: List[MemoryTurnRecord]

