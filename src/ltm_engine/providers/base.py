"""Abstract base classes for pluggable providers."""

from abc import ABC, abstractmethod
from typing import Any


class EmbeddingProvider(ABC):
    """
    Abstract base class for embedding providers.
    
    Implement this interface to add support for different embedding models
    (OpenAI, Cohere, local models, etc.)
    """

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Return the dimensionality of the embeddings."""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model name."""
        ...

    @abstractmethod
    async def embed_text(self, text: str) -> list[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            List of floats representing the embedding vector
        """
        ...

    @abstractmethod
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        ...


class LLMProvider(ABC):
    """
    Abstract base class for LLM providers.
    
    Implement this interface to add support for different LLM models
    (OpenAI, Anthropic, local models, etc.)
    """

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model name."""
        ...

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """
        Generate text from a prompt.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            temperature: Sampling temperature (optional override)
            max_tokens: Maximum tokens to generate (optional override)
            
        Returns:
            Generated text
        """
        ...

    @abstractmethod
    async def generate_json(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float | None = None,
    ) -> dict[str, Any]:
        """
        Generate structured JSON response.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            temperature: Sampling temperature (optional override)
            
        Returns:
            Parsed JSON as dictionary
        """
        ...

    @abstractmethod
    async def analyze_conflict(
        self,
        existing_content: str,
        new_content: str,
        context: str | None = None,
    ) -> dict[str, Any]:
        """
        Analyze whether two pieces of content conflict.
        
        Args:
            existing_content: Existing memory content
            new_content: New memory content
            context: Optional context about the memories
            
        Returns:
            Analysis result with is_contradiction, confidence, explanation
        """
        ...

    @abstractmethod
    async def summarize(
        self,
        contents: list[str],
        context: str | None = None,
    ) -> str:
        """
        Summarize multiple pieces of content into one.
        
        Args:
            contents: List of content strings to summarize
            context: Optional context about the content
            
        Returns:
            Summarized content
        """
        ...

    @abstractmethod
    async def extract_subject(self, content: str) -> str:
        """
        Extract the main subject/topic from content.
        
        Args:
            content: Content to analyze
            
        Returns:
            Main subject/topic
        """
        ...
