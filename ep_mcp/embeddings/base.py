"""Abstract embedding provider interface."""

from __future__ import annotations

from abc import ABC, abstractmethod


class EmbeddingProvider(ABC):
    """Abstract interface for text embedding providers.

    Implementations handle rate limiting, batching, and error propagation.
    """

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Unique identifier for cache keying (e.g., 'gemini/gemini-embedding-001')."""
        ...

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Output embedding dimension."""
        ...

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a batch of texts.

        Implementations must handle:
        - Rate limiting / retry with backoff
        - Batch size limits (split large batches internally)
        - Error propagation with meaningful messages
        """
        ...

    async def embed_query(self, query: str) -> list[float]:
        """Embed a single query string.

        Override if the provider distinguishes query vs document embeddings.
        """
        result = await self.embed([query])
        return result[0]
