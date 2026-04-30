"""Azure OpenAI embedding provider."""

from __future__ import annotations

import asyncio
import logging
import os

from openai import AsyncAzureOpenAI

from .base import EmbeddingProvider

logger = logging.getLogger(__name__)

# Azure OpenAI model dimensions
_MODEL_DIMENSIONS: dict[str, int] = {
    "text-embedding-3-large": 3072,
    "text-embedding-3-small": 1536,
    "text-embedding-ada-002": 1536,
}

# Batch / concurrency limits
_MAX_BATCH_SIZE = 512       # Azure OpenAI accepts up to 2048 inputs; 512 is safe
_MAX_PARALLEL_BATCHES = 4   # concurrent requests during index builds


class AzureOpenAIEmbeddingProvider(EmbeddingProvider):
    """Azure OpenAI embedding provider.

    Supports text-embedding-3-large (3072d, recommended), text-embedding-3-small
    (1536d), and text-embedding-ada-002 (1536d).

    Required config / env vars:
        AZURE_OPENAI_ENDPOINT   — e.g. https://<resource>.openai.azure.com/
        AZURE_OPENAI_API_KEY    — API key for the Azure OpenAI resource

    Optional config:
        azure_deployment        — deployment name (defaults to model name)
        azure_api_version       — defaults to "2024-10-21"
        output_dimensionality   — MRL shortening (3-large only, must be <= 3072)

    Batching strategy mirrors the Gemini provider:
    - Splits large inputs into chunks of _MAX_BATCH_SIZE texts.
    - Sends up to _MAX_PARALLEL_BATCHES concurrent requests with a semaphore.
    - Single-query embed_query() uses input_type hint for better retrieval quality.
    - Each batch retries up to 3 times with exponential backoff.
    """

    def __init__(
        self,
        model: str = "text-embedding-3-large",
        azure_endpoint: str | None = None,
        api_key: str | None = None,
        api_version: str = "2024-10-21",
        azure_deployment: str | None = None,
        output_dimensionality: int | None = None,
        max_parallel_batches: int = _MAX_PARALLEL_BATCHES,
    ):
        self._model = model
        self._deployment = azure_deployment or model
        self._output_dimensionality = output_dimensionality

        endpoint = azure_endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT")
        if not endpoint:
            raise ValueError(
                "Azure OpenAI endpoint required — set AZURE_OPENAI_ENDPOINT env var "
                "or set embedding.azure_endpoint in config"
            )

        key = api_key or os.environ.get("AZURE_OPENAI_API_KEY")
        if not key:
            raise ValueError(
                "Azure OpenAI API key required — set AZURE_OPENAI_API_KEY env var "
                "or set embedding.azure_api_key in config"
            )

        self._client = AsyncAzureOpenAI(
            azure_endpoint=endpoint,
            api_key=key,
            api_version=api_version,
        )
        self._semaphore = asyncio.Semaphore(max_parallel_batches)

    @property
    def model_name(self) -> str:
        return f"azure-openai/{self._model}"

    @property
    def dimension(self) -> int:
        if self._output_dimensionality is not None:
            return self._output_dimensionality
        return _MODEL_DIMENSIONS.get(self._model, 1536)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate document embeddings with parallel batching."""
        if not texts:
            return []

        batches = [
            texts[i : i + _MAX_BATCH_SIZE]
            for i in range(0, len(texts), _MAX_BATCH_SIZE)
        ]

        if len(batches) == 1:
            return await self._embed_batch_with_semaphore(batches[0], input_type="document")

        logger.info(
            "Embedding %d texts across %d batches (%d parallel max)",
            len(texts), len(batches), self._semaphore._value,
        )

        tasks = [
            asyncio.ensure_future(
                self._embed_batch_with_semaphore(batch, input_type="document")
            )
            for batch in batches
        ]
        batch_results = await asyncio.gather(*tasks)

        all_embeddings: list[list[float]] = []
        for batch_emb in batch_results:
            all_embeddings.extend(batch_emb)
        return all_embeddings

    async def embed_query(self, query: str) -> list[float]:
        """Embed a single query string with query-optimised input type."""
        result = await self._embed_batch_with_semaphore([query], input_type="query")
        return result[0]

    async def _embed_batch_with_semaphore(
        self, texts: list[str], input_type: str = "document"
    ) -> list[list[float]]:
        async with self._semaphore:
            return await self._embed_batch(texts, input_type=input_type)

    async def _embed_batch(
        self,
        texts: list[str],
        input_type: str = "document",
        max_retries: int = 3,
    ) -> list[list[float]]:
        """Embed a single batch with exponential-backoff retry."""
        for attempt in range(max_retries):
            try:
                kwargs: dict = dict(
                    model=self._deployment,
                    input=texts,
                )
                # MRL dimensionality reduction (text-embedding-3-* only)
                if self._output_dimensionality is not None:
                    kwargs["dimensions"] = self._output_dimensionality

                response = await self._client.embeddings.create(**kwargs)
                # Sort by index to ensure order matches input
                sorted_data = sorted(response.data, key=lambda e: e.index)
                return [e.embedding for e in sorted_data]

            except Exception as e:
                if attempt < max_retries - 1:
                    wait = 2 ** attempt
                    logger.warning(
                        "Azure OpenAI embed attempt %d failed (%s), retrying in %ds",
                        attempt + 1, e, wait,
                    )
                    await asyncio.sleep(wait)
                else:
                    raise RuntimeError(
                        f"Azure OpenAI embedding failed after {max_retries} attempts: {e}"
                    ) from e
