import logging
from typing import Any, Dict, List, Optional
import httpx
from app.config import settings

logger = logging.getLogger(__name__)


class OllamaServiceError(Exception):
    """Custom exception raised when Ollama service fails."""
    pass


class OllamaClient:
    def __init__(self, base_url: str = settings.OLLAMA_BASE_URL):
        self.base_url = base_url.rstrip("/")
        # Generous timeout for local inference
        self.timeout = httpx.Timeout(180.0, connect=10.0)

    async def get_embedding(self, text: str, model: Optional[str] = None) -> List[float]:
        """Generate embedding vector using nomic-embed-text via Ollama."""
        model_name = model or settings.EMBEDDING_MODEL
        url = f"{self.base_url}/api/embeddings"
        payload = {
            "model": model_name,
            "prompt": text.strip(),
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(url, json=payload)
                if response.status_code != 200:
                    raise OllamaServiceError(
                        f"Ollama embedding error (status {response.status_code}): {response.text}"
                    )
                data = response.json()
                embedding = data.get("embedding")
                if not embedding or not isinstance(embedding, list):
                    raise OllamaServiceError(f"Invalid embedding returned: {data}")
                return embedding
        except httpx.RequestError as exc:
            logger.error(f"Failed to connect to Ollama embedding API: {exc}")
            raise OllamaServiceError(f"Connection error to Ollama: {exc}") from exc

    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: Optional[int] = 350,
    ) -> str:
        """Call Ollama /api/chat with provided message history."""
        model_name = model or settings.LLM_MODEL
        url = f"{self.base_url}/api/chat"
        options: Dict[str, Any] = {
            "temperature": temperature,
        }
        if max_tokens:
            options["num_predict"] = max_tokens

        payload: Dict[str, Any] = {
            "model": model_name,
            "messages": messages,
            "stream": False,
            "options": options,
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(url, json=payload)
                if response.status_code != 200:
                    raise OllamaServiceError(
                        f"Ollama chat error (status {response.status_code}): {response.text}"
                    )
                data = response.json()
                content = data.get("message", {}).get("content", "")
                return content.strip()
        except httpx.RequestError as exc:
            logger.error(f"Failed to connect to Ollama chat API: {exc}")
            raise OllamaServiceError(f"Connection error to Ollama: {exc}") from exc


ollama_client = OllamaClient()
