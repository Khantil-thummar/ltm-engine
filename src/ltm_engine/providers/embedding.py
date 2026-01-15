"""OpenAI Embedding Provider implementation."""

import structlog
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from ltm_engine.config import Settings
from ltm_engine.providers.base import EmbeddingProvider

logger = structlog.get_logger(__name__)


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """
    OpenAI embedding provider using text-embedding-3-small.
    
    Features:
    - Async API calls
    - Automatic retries with exponential backoff
    - Batch processing support
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client = AsyncOpenAI(api_key=settings.openai_api_key)
        self._model = settings.embedding_model
        self._dimensions = settings.embedding_dimensions

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

        response = await self._client.embeddings.create(
            model=self._model,
            input=text,
            dimensions=self._dimensions,
        )

        return response.data[0].embedding

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

        # OpenAI has a limit on batch size, split if needed
        max_batch_size = 2048
        all_embeddings: list[list[float]] = []

        for i in range(0, len(valid_texts), max_batch_size):
            batch = valid_texts[i : i + max_batch_size]
            response = await self._client.embeddings.create(
                model=self._model,
                input=batch,
                dimensions=self._dimensions,
            )
            all_embeddings.extend([d.embedding for d in response.data])

        return all_embeddings
