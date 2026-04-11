"""Google Gemini embedding provider."""

from __future__ import annotations

import asyncio
import logging
import os

from google import genai

from .base import EmbeddingProvider

logger = logging.getLogger(__name__)

# Gemini batch limits
_MAX_BATCH_SIZE = 100
_GEMINI_DIMENSION = 3072


class GeminiEmbeddingProvider(EmbeddingProvider):
    """Gemini embedding provider using google-genai SDK.

    Uses gemini-embedding-001 (768 dimensions) by default.
    API key from GEMINI_API_KEY environment variable.
    """

    def __init__(self, model: str = "gemini-embedding-001", api_key: str | None = None):
        self._model = model
        self._api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self._api_key:
            raise ValueError(
                "Gemini API key required — set GEMINI_API_KEY env var or pass api_key"
            )
        self._client = genai.Client(api_key=self._api_key)

    @property
    def model_name(self) -> str:
        return f"gemini/{self._model}"

    @property
    def dimension(self) -> int:
        return _GEMINI_DIMENSION

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings via Gemini API.

        Handles batching (max 100 texts per request) and basic retry.
        """
        if not texts:
            return []

        all_embeddings: list[list[float]] = []

        # Process in batches
        for i in range(0, len(texts), _MAX_BATCH_SIZE):
            batch = texts[i : i + _MAX_BATCH_SIZE]
            embeddings = await self._embed_batch(batch)
            all_embeddings.extend(embeddings)

        return all_embeddings

    async def _embed_batch(
        self, texts: list[str], max_retries: int = 3
    ) -> list[list[float]]:
        """Embed a single batch with retry logic."""
        for attempt in range(max_retries):
            try:
                # google-genai embed_content is sync — run in executor
                result = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self._client.models.embed_content(
                        model=self._model,
                        contents=texts,
                    ),
                )
                return [e.values for e in result.embeddings]

            except Exception as e:
                if attempt < max_retries - 1:
                    wait = 2 ** attempt
                    logger.warning(
                        "Gemini embed attempt %d failed (%s), retrying in %ds",
                        attempt + 1, e, wait,
                    )
                    await asyncio.sleep(wait)
                else:
                    raise RuntimeError(
                        f"Gemini embedding failed after {max_retries} attempts: {e}"
                    ) from e
