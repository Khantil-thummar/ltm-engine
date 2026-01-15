"""OpenAI LLM Provider implementation."""

import json
from typing import Any

import structlog
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from ltm_engine.config import Settings
from ltm_engine.providers.base import LLMProvider

logger = structlog.get_logger(__name__)


class OpenAILLMProvider(LLMProvider):
    """
    OpenAI LLM provider using GPT-4o-mini.
    
    Features:
    - Async API calls
    - Automatic retries with exponential backoff
    - Structured JSON output support
    - Specialized prompts for memory operations
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client = AsyncOpenAI(api_key=settings.openai_api_key)
        self._model = settings.llm_model
        self._temperature = settings.llm_temperature
        self._max_tokens = settings.llm_max_tokens

    @property
    def model_name(self) -> str:
        return self._model

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
    )
    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Generate text from a prompt."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        logger.debug(
            "Generating LLM response",
            model=self._model,
            prompt_length=len(prompt),
        )

        response = await self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=temperature or self._temperature,
            max_tokens=max_tokens or self._max_tokens,
        )

        return response.choices[0].message.content or ""

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
    )
    async def generate_json(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float | None = None,
    ) -> dict[str, Any]:
        """Generate structured JSON response."""
        full_system = (system_prompt or "") + (
            "\n\nYou must respond with valid JSON only. No markdown, no explanation."
        )

        messages = [
            {"role": "system", "content": full_system},
            {"role": "user", "content": prompt},
        ]

        response = await self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=temperature or self._temperature,
            max_tokens=self._max_tokens,
            response_format={"type": "json_object"},
        )

        content = response.choices[0].message.content or "{}"
        return json.loads(content)

    async def analyze_conflict(
        self,
        existing_content: str,
        new_content: str,
        context: str | None = None,
    ) -> dict[str, Any]:
        """Analyze whether two pieces of content conflict."""
        system_prompt = """You are a memory conflict analyzer. Analyze whether two pieces of information contradict each other.

Consider:
1. Direct contradictions (A says X, B says not-X)
2. Implicit contradictions (A implies X, B implies not-X)
3. Updates vs contradictions (new info that replaces vs conflicts with old)
4. Temporal context (things that could have changed over time)

Respond with JSON containing:
- is_contradiction: boolean
- confidence: float (0.0 to 1.0)
- explanation: string explaining the analysis
- resolution_suggestion: one of "supersede", "merge", "keep_both", "reject_new"
"""

        prompt = f"""Analyze these two pieces of memory content for conflicts:

EXISTING MEMORY:
{existing_content}

NEW MEMORY:
{new_content}
"""
        if context:
            prompt += f"\nCONTEXT:\n{context}"

        return await self.generate_json(prompt, system_prompt)

    async def summarize(
        self,
        contents: list[str],
        context: str | None = None,
    ) -> str:
        """Summarize multiple pieces of content into one."""
        system_prompt = """You are a memory consolidation system. Your task is to summarize multiple episodic memories (events, conversations, interactions) into a single semantic memory (fact, knowledge).

Guidelines:
1. Extract the key facts and insights
2. Preserve important details and relationships
3. Remove redundancy
4. Maintain temporal accuracy when relevant
5. Be concise but complete
6. Use clear, factual language"""

        numbered_contents = "\n\n".join(
            f"[Memory {i+1}]:\n{c}" for i, c in enumerate(contents)
        )

        prompt = f"""Consolidate these episodic memories into a single semantic memory:

{numbered_contents}

Provide a clear, consolidated summary that captures the essential knowledge."""

        if context:
            prompt += f"\n\nContext: {context}"

        return await self.generate(prompt, system_prompt)

    async def extract_subject(self, content: str) -> str:
        """Extract the main subject/topic from content."""
        system_prompt = """You extract the main subject or topic from a piece of text.
Respond with just the subject/topic phrase, nothing else.
Keep it short (1-5 words) and specific."""

        prompt = f"Extract the main subject from:\n\n{content}"

        result = await self.generate(prompt, system_prompt, temperature=0.0)
        return result.strip().strip('"\'')
