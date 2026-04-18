"""Google Gemini embedding provider."""

from __future__ import annotations

import asyncio
import logging
import os

from google import genai

from .base import EmbeddingProvider

logger = logging.getLogger(__name__)

# Gemini API limits
_MAX_BATCH_SIZE = 100       # texts per embed_content call
_MAX_PARALLEL_BATCHES = 4   # concurrent requests during index builds
_GEMINI_DIMENSION = 3072


class GeminiEmbeddingProvider(EmbeddingProvider):
    """Gemini embedding provider using google-genai SDK.

    Uses gemini-embedding-001 (3072 dimensions) by default.
    API key from GEMINI_API_KEY environment variable.

    Batching strategy:
    - Splits input into chunks of _MAX_BATCH_SIZE (100) texts.
    - Sends up to _MAX_PARALLEL_BATCHES (4) concurrent requests during
      bulk index builds, using an asyncio.Semaphore to cap parallelism.
    - Single-text queries (embed_query) bypass batch splitting for lower
      latency.
    - Each batch retries up to 3 times with exponential backoff.
    """

    def __init__(
        self,
        model: str = "gemini-embedding-001",
        api_key: str | None = None,
        max_parallel_batches: int = _MAX_PARALLEL_BATCHES,
        output_dimensionality: int | None = None,
    ):
        self._model = model
        self._api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self._api_key:
            raise ValueError(
                "Gemini API key required — set GEMINI_API_KEY env var or pass api_key"
            )
        self._client = genai.Client(api_key=self._api_key)
        self._semaphore = asyncio.Semaphore(max_parallel_batches)
        self._output_dimensionality = output_dimensionality  # None = full 3072d

    @property
    def model_name(self) -> str:
        return f"gemini/{self._model}"

    @property
    def dimension(self) -> int:
        return self._output_dimensionality or _GEMINI_DIMENSION

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings via Gemini API with parallel batch requests."""
        if not texts:
            return []

        batches = [
            texts[i : i + _MAX_BATCH_SIZE]
            for i in range(0, len(texts), _MAX_BATCH_SIZE)
        ]

        if len(batches) == 1:
            return await self._embed_batch_with_semaphore(batches[0])

        logger.info(
            "Embedding %d texts across %d batches (%d parallel max)",
            len(texts), len(batches), self._semaphore._value,
        )

        tasks = [
            asyncio.ensure_future(self._embed_batch_with_semaphore(batch))
            for batch in batches
        ]
        batch_results = await asyncio.gather(*tasks)

        all_embeddings: list[list[float]] = []
        for batch_emb in batch_results:
            all_embeddings.extend(batch_emb)
        return all_embeddings

    async def _embed_batch_with_semaphore(
        self, texts: list[str]
    ) -> list[list[float]]:
        async with self._semaphore:
            return await self._embed_batch(texts)

    async def _embed_batch(
        self, texts: list[str], max_retries: int = 3
    ) -> list[list[float]]:
        """Embed a single batch with exponential-backoff retry."""
        for attempt in range(max_retries):
            try:
                kwargs: dict = dict(model=self._model, contents=texts)
                if self._output_dimensionality is not None:
                    kwargs["config"] = {"output_dimensionality": self._output_dimensionality}
                result = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self._client.models.embed_content(**kwargs),
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
