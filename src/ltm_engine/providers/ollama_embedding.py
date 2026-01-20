"""Ollama Embedding Provider implementation."""

import ollama
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

from ltm_engine.config import Settings
from ltm_engine.providers.base import EmbeddingProvider

logger = structlog.get_logger(__name__)


class OllamaEmbeddingProvider(EmbeddingProvider):
    """
    Ollama embedding provider using nomic-embed-text.
    
    Features:
    - Async API calls using official ollama library
    - Automatic retries with exponential backoff
    - Batch processing support
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._model = settings.ollama_embedding_model
        self._dimensions = settings.embedding_dimensions
        self._client = ollama.AsyncClient()

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def model_name(self) -> str:
        return self._model

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
    )
    async def embed_text(self, text: str) -> list[float]:
        """Generate embedding for a single text."""
        if not text.strip():
            raise ValueError("Cannot embed empty text")

        logger.debug("Generating embedding", model=self._model, text_length=len(text))

        response = await self._client.embed(
            model=self._model,
            input=text,
        )

        return response["embeddings"][0]

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
    )
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        if not texts:
            return []

        # Filter out empty texts
        valid_texts = [t for t in texts if t.strip()]
        if not valid_texts:
            raise ValueError("All texts are empty")

        logger.debug(
            "Generating batch embeddings",
            model=self._model,
            batch_size=len(valid_texts),
        )

        # Ollama supports batch embedding via the input parameter
        response = await self._client.embed(
            model=self._model,
            input=valid_texts,
        )

        return response["embeddings"]
