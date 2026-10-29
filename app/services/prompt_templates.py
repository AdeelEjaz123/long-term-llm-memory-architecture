import json
import re
from typing import Any, Dict, List


def build_query_resolution_prompt(recent_turns: List[Dict[str, Any]], current_prompt: str) -> List[Dict[str, str]]:
    """
    Constructs messages to resolve pronouns and context dependencies
    in the raw user prompt into a standalone search query.
    """
    history_lines = []
    for turn in recent_turns:
        u_msg = turn.get("user_message") or turn.get("prompt_summary") or ""
        a_msg = turn.get("assistant_message") or turn.get("answer_summary") or ""
        history_lines.append(f"User: {u_msg}")
        history_lines.append(f"Assistant: {a_msg}")

    history_str = "\n".join(history_lines) if history_lines else "None"

    system_instruction = (
        "You are an expert query refactorer for a conversational vector memory engine.\n"
        "Your ONLY job is to rewrite the user's latest prompt into a concise standalone search query.\n\n"
        "CRITICAL RULES:\n"
        "1. DO NOT answer the question. NEVER return the answer or a single person's name.\n"
        "2. The output MUST be a standalone search query or question.\n"
        "3. Replace vague pronouns ('it', 'that', 'this', 'they') with the specific entities from history.\n"
        "4. Output ONLY the rewritten search query. No preamble, no quotes, no conversational filler.\n\n"
        "Examples:\n"
        "User: How do I deploy it?\n"
        "Standalone Query: How to deploy the FinTech digital banking microservice\n\n"
        "User: what is your name?\n"
        "Standalone Query: What is the assistant's name\n\n"
        "User: whats my name?\n"
        "Standalone Query: What is the user's name\n\n"
        "User: tell me something about america\n"
        "Standalone Query: facts and overview of America\n\n"
        "User: ok so for now on your name is lucky, remember that ok\n"
        "Standalone Query: User assigns assistant name as Lucky"
    )

    user_instruction = (
        f"Recent Conversation History:\n"
        f"{history_str}\n\n"
        f"User Prompt: {current_prompt}\n\n"
        f"Standalone Query:"
    )

    return [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": user_instruction},
    ]


def build_summarization_prompt(user_prompt: str, assistant_response: str) -> List[Dict[str, str]]:
    """
    Constructs messages to summarize a single conversational turn into concise
    prompt_summary and answer_summary strings. Returns strict JSON.
    """
    system_instruction = (
        "You are a conversation summarizer for long-term memory storage.\n"
        "Summarize the user prompt and assistant response into 1-2 concise factual sentences.\n"
        "Respond ONLY with valid JSON in this exact schema:\n"
        "{\n"
        '  "prompt_summary": "1 sentence describing what user asked or stated",\n'
        '  "answer_summary": "1 sentence describing key answer or outcome"\n'
        "}\n"
        "Do not include markdown blocks or any other text."
    )

    user_instruction = (
        f"User Prompt:\n{user_prompt}\n\n"
        f"Assistant Response:\n{assistant_response}\n\n"
        f"JSON Summary:"
    )

    return [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": user_instruction},
    ]


def format_branch_a_context(matches: List[Dict[str, Any]]) -> str:
    """Format Branch A (Semantic Match) context block."""
    lines = ["[Relevant Historical Long-Term Memory (via Semantic Match)]"]
    for match in matches:
        lines.append(f"- Past User: {match['prompt_summary']}")
        lines.append(f"- Past Answer: {match['answer_summary']}")
    return "\n".join(lines)


def format_branch_b_context(recent_turns: List[Dict[str, Any]]) -> str:
    """Format Branch B (Recency Fallback) context block."""
    lines = ["[Recent Conversation Chronology & Baseline State]"]
    if not recent_turns:
        lines.append("- (No prior conversation history)")
    else:
        for turn in recent_turns:
            u_text = turn.get("user_message") or turn.get("prompt_summary")
            a_text = turn.get("assistant_message") or turn.get("answer_summary")
            lines.append(f"- User: {u_text}")
            lines.append(f"- Assistant: {a_text}")
    return "\n".join(lines)


def build_main_generation_messages(
    context_block: str,
    raw_user_prompt: str,
    recent_turns: List[Dict[str, Any]] = None,
) -> List[Dict[str, str]]:
    """
    Assembles a clean, native multi-turn message payload for the LLM.
    - System message holds persona, memory context block, and recency rules.
    - Previous turns are formatted as native user/assistant role turns.
    - Final turn is the raw user prompt.
    This prevents prompt leakage, role confusion, and repetitive boilerplate.
    """
    system_content = (
        "You are an intelligent, helpful AI assistant.\n"
        "Keep track of conversational identity: you are the assistant, and the human user is speaking to you.\n"
        f"{context_block}\n\n"
        "CRITICAL RULES:\n"
        "1. Answer naturally, directly, and concisely.\n"
        "2. NEVER output internal system headers like '[Context Memory Block]', '[Response]', or repeat your prompt.\n"
        "3. NEVER say repetitive boilerplate like 'I have taken note of your new question'.\n"
        "4. If the user gave corrections or new constraints, the most recent instruction strictly takes precedence over older memory."
    )

    messages = [{"role": "system", "content": system_content}]

    # Inject recent conversation as native chat messages (last 3 turns max)
    if recent_turns:
        for turn in recent_turns[-3:]:
            u = turn.get("user_message") or turn.get("prompt_summary")
            a = turn.get("assistant_message") or turn.get("answer_summary")
            if u and a:
                # Clean up any previously stored leaked headers in historical text
                clean_a = re.sub(r"\[Context Memory Block\].*?\[Response\]", "", a, flags=re.DOTALL).strip()
                clean_a = re.sub(r"^\[Response\]\s*", "", clean_a, flags=re.IGNORECASE).strip()
                messages.append({"role": "user", "content": u})
                messages.append({"role": "assistant", "content": clean_a})

    # Append current user prompt
    messages.append({"role": "user", "content": raw_user_prompt})

    return messages
