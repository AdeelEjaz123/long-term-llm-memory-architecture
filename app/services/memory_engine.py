import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple
from app.config import settings
from app.database import (
    get_recent_turns,
    get_session_turn_count,
    insert_memory_turn,
    search_vector_memory,
)
from app.models.schemas import ChatResponse, HistoricalContextItem
from app.services.ollama_client import ollama_client
from app.services.prompt_templates import (
    build_main_generation_messages,
    build_query_resolution_prompt,
    build_summarization_prompt,
    format_branch_a_context,
    format_branch_b_context,
)

logger = logging.getLogger(__name__)


def sanitize_llm_response(text: str) -> str:
    """Clean up any accidental leaked headers or repetitive boilerplate."""
    text = re.sub(r"\[Context Memory Block\].*?\[Response\]", "", text, flags=re.DOTALL)
    text = re.sub(r"^\[Context Memory Block\].*?\n\n", "", text, flags=re.DOTALL)
    text = re.sub(r"^\[Response\]\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^(Assistant|AI):\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^I(?:'ve| have) taken note of (?:your new question|the correction|your question)\.\s*(?:For the purpose of our conversation, [^\n]+)?\n*", "", text, flags=re.IGNORECASE)
    return text.strip()


class SemanticMemoryEngine:
    async def process_chat_turn(
        self,
        session_id: str,
        raw_user_prompt: str,
    ) -> Tuple[ChatResponse, Dict[str, Any]]:
        """
        Executes Steps A, B, C, D of the specification.
        Returns:
            - ChatResponse to return to caller
            - Post-processing payload needed for Step E background task
        """
        # Determine current turn index
        turn_count = await get_session_turn_count(session_id)
        current_turn_index = turn_count + 1
        recent_turns = await get_recent_turns(session_id, limit=settings.RECENT_TURNS_LIMIT)

        resolved_query = raw_user_prompt
        matched_branch = "Cold Start"
        similarity_score: Optional[float] = None
        historical_items: List[HistoricalContextItem] = []
        context_block = ""

        # Check Cold Start (0 or 1 prior turns)
        if turn_count <= 1:
            logger.info(f"[{session_id}] Cold start (turns={turn_count}). Bypassing vector retrieval.")
            matched_branch = "Cold Start"
            context_block = format_branch_b_context(recent_turns)
        else:
            # Step A: Query Resolution (The Pronoun Fix)
            try:
                resolution_messages = build_query_resolution_prompt(recent_turns, raw_user_prompt)
                resolved_query_raw = await ollama_client.chat_completion(
                    messages=resolution_messages,
                    model=settings.FAST_RESOLVER_MODEL,
                    temperature=0.0,
                    max_tokens=64,
                )
                # Clean up quotes, markdown, or newlines if present
                clean_resolved = resolved_query_raw.strip().strip('"').strip("'").split("\n")[0]
                if clean_resolved:
                    resolved_query = clean_resolved
                logger.info(f"[{session_id}] Resolved query: '{raw_user_prompt}' -> '{resolved_query}'")
            except Exception as e:
                logger.warning(f"[{session_id}] Query resolution failed, degrading to raw prompt: {e}")
                resolved_query = raw_user_prompt

            # Step B: Vector Similarity Search & Matching (with Graceful Degradation)
            matches = []
            try:
                query_embedding = await ollama_client.get_embedding(
                    text=resolved_query,
                    model=settings.EMBEDDING_MODEL,
                )
                matches = await search_vector_memory(
                    session_id=session_id,
                    query_embedding=query_embedding,
                    limit=settings.TOP_K,
                    threshold=settings.SIMILARITY_THRESHOLD,
                )
            except Exception as e:
                logger.error(f"[{session_id}] Vector search / embedding failed: {e}. Falling back to recency.")
                matches = []

            # Step C: Branching Context Injection
            if matches:
                matched_branch = "Branch A (Semantic Match)"
                similarity_score = matches[0]["similarity"]
                context_block = format_branch_a_context(matches)
                historical_items = [
                    HistoricalContextItem(
                        turn_index=m["turn_index"],
                        prompt_summary=m["prompt_summary"],
                        answer_summary=m["answer_summary"],
                        similarity=m["similarity"],
                    )
                    for m in matches
                ]
                logger.info(f"[{session_id}] Branch A selected with {len(matches)} match(es). Top sim: {similarity_score}")
            else:
                matched_branch = "Branch B (Recency Fallback)"
                context_block = format_branch_b_context(recent_turns)
                logger.info(f"[{session_id}] Branch B selected (no vector matches >= {settings.SIMILARITY_THRESHOLD}).")

        # Step D: Prompt Construction & LLM Generation
        generation_messages = build_main_generation_messages(
            context_block=context_block,
            raw_user_prompt=raw_user_prompt,
            recent_turns=recent_turns,
        )

        try:
            raw_assistant_response = await ollama_client.chat_completion(
                messages=generation_messages,
                model=settings.LLM_MODEL,
                temperature=0.3,
            )
            assistant_response = sanitize_llm_response(raw_assistant_response)
        except Exception as e:
            logger.error(f"[{session_id}] Primary LLM generation failed: {e}")
            assistant_response = "I apologize, but I encountered an error processing your request."

        chat_response = ChatResponse(
            session_id=session_id,
            turn_index=current_turn_index,
            response=assistant_response,
            resolved_query=resolved_query,
            matched_branch=matched_branch,
            similarity_score=similarity_score,
            historical_context=historical_items,
        )

        post_process_data = {
            "session_id": session_id,
            "turn_index": current_turn_index,
            "raw_user_prompt": raw_user_prompt,
            "assistant_response": assistant_response,
        }

        return chat_response, post_process_data

    async def post_process_turn(
        self,
        session_id: str,
        turn_index: int,
        raw_user_prompt: str,
        assistant_response: str,
    ):
        """
        Step E: Asynchronous Post-Processing (Write Path)
        1. Summarize latest user prompt and assistant response (1-2 sentences).
        2. Generate embedding vector for the prompt summary.
        3. Insert record into chat_memory_store.
        """
        logger.info(f"[{session_id}] Starting async post-processing for turn {turn_index}...")
        try:
            # 1. Summarize turn
            prompt_summary, answer_summary = await self._generate_summaries(
                raw_user_prompt, assistant_response
            )

            # 2. Generate embedding for prompt summary
            prompt_embedding = await ollama_client.get_embedding(
                text=prompt_summary,
                model=settings.EMBEDDING_MODEL,
            )

            # 3. Store in PostgreSQL chat_memory_store
            inserted_id = await insert_memory_turn(
                session_id=session_id,
                turn_index=turn_index,
                prompt_summary=prompt_summary,
                answer_summary=answer_summary,
                prompt_embedding=prompt_embedding,
                user_message=raw_user_prompt,
                assistant_message=assistant_response,
            )
            logger.info(
                f"[{session_id}] Turn {turn_index} successfully persisted to chat_memory_store (ID={inserted_id})."
            )
        except Exception as e:
            logger.error(f"[{session_id}] Error in async post-processing turn {turn_index}: {e}", exc_info=True)

    async def _generate_summaries(self, user_prompt: str, assistant_response: str) -> Tuple[str, str]:
        """Generate 1-2 sentence summaries for user prompt and assistant response."""
        try:
            summarization_messages = build_summarization_prompt(user_prompt, assistant_response)
            raw_json = await ollama_client.chat_completion(
                messages=summarization_messages,
                model=settings.FAST_RESOLVER_MODEL,
                temperature=0.1,
                max_tokens=150,
            )

            # Strip possible markdown code block wrapper
            cleaned = re.sub(r"^```json\s*", "", raw_json.strip())
            cleaned = re.sub(r"```$", "", cleaned.strip())

            data = json.loads(cleaned)
            p_sum = data.get("prompt_summary", user_prompt[:200])
            a_sum = data.get("answer_summary", assistant_response[:200])
            return p_sum, a_sum
        except Exception as e:
            logger.warning(f"Summarization JSON parsing failed: {e}. Using extractive fallback.")
            # Fallback to truncated text if summarization fails
            p_fallback = user_prompt.strip()[:150] + ("..." if len(user_prompt) > 150 else "")
            a_fallback = assistant_response.strip()[:200] + ("..." if len(assistant_response) > 200 else "")
            return p_fallback, a_fallback


memory_engine = SemanticMemoryEngine()

