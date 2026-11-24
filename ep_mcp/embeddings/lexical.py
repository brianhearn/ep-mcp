"""Zero-key lexical (BM25-only) embedding provider.

Allows ep-mcp to index and serve packs completely offline with no cloud API key.
SQLite FTS5 handles full-text search while vector tables are skipped.
"""

from __future__ import annotations

from .base import EmbeddingProvider


class LexicalOnlyEmbeddingProvider(EmbeddingProvider):
    """No-op embedding provider (dimension=0) for zero-key BM25 retrieval."""

    @property
    def model_name(self) -> str:
        return "lexical/bm25-only"

    @property
    def dimension(self) -> int:
        return 0

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [[] for _ in texts]

    async def embed_query(self, query: str) -> list[float]:
        return []
