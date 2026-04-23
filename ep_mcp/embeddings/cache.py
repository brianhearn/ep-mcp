"""SQLite-backed query embedding cache.

Wraps any EmbeddingProvider and caches embed_query() results by
(model_name, dimension, query_text) key. Bulk embed() calls are
always passed through uncached (index builds must stay fresh).

Cache location: <first_pack_index_dir>/../query_embed_cache.db
or a custom path via QueryEmbeddingCache(provider, cache_path=...).

Typical warm latency: ~1ms vs ~3-5s for a live Gemini API call.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import sqlite3
import struct
import time
from pathlib import Path

from .base import EmbeddingProvider

logger = logging.getLogger(__name__)


class QueryEmbeddingCache(EmbeddingProvider):
    """Transparent cache wrapper around any EmbeddingProvider.

    Only caches single-query embed_query() calls. Bulk embed() is
    passed through — index builds always use fresh embeddings.
    """

    def __init__(self, provider: EmbeddingProvider, cache_path: str | Path):
        self._provider = provider
        self._cache_path = Path(cache_path)
        self._conn: sqlite3.Connection | None = None
        self._lock = asyncio.Lock()
        #: Set after each embed_query() call — True = served from cache.
        self.last_cache_hit: bool = False

    # ── EmbeddingProvider interface ───────────────────────────────────────────

    @property
    def model_name(self) -> str:
        return self._provider.model_name

    @property
    def dimension(self) -> int:
        return self._provider.dimension

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Pass-through — bulk embeds are not cached."""
        return await self._provider.embed(texts)

    async def embed_query(self, query: str) -> list[float]:
        """Cache-aware single query embedding."""
        self._ensure_open()
        key = self._make_key(query)

        hit = self._get(key)
        if hit is not None:
            logger.debug("Query embed cache HIT (dim=%d, query=%r)", self.dimension, query[:60])
            self.last_cache_hit = True
            return hit

        logger.debug("Query embed cache MISS — calling provider (query=%r)", query[:60])
        self.last_cache_hit = False
        t0 = time.monotonic()
        embedding = await self._provider.embed_query(query)
        elapsed = time.monotonic() - t0
        logger.info("Query embed: %.0fms (cache miss, stored)", elapsed * 1000)

        self._put(key, embedding)
        return embedding

    # ── Cache internals ───────────────────────────────────────────────────────

    def _ensure_open(self) -> None:
        if self._conn is not None:
            return
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._cache_path), check_same_thread=False)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS query_embeddings (
                cache_key  TEXT PRIMARY KEY,
                embedding  BLOB NOT NULL,
                created_at INTEGER NOT NULL
            )
        """)
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_created ON query_embeddings(created_at)")
        self._conn.commit()
        count = self._conn.execute("SELECT COUNT(*) FROM query_embeddings").fetchone()[0]
        logger.info("Query embed cache opened: %s (%d entries)", self._cache_path, count)

    def _make_key(self, query: str) -> str:
        """Stable cache key: model + dimension + sha256(query)."""
        payload = f"{self.model_name}|{self.dimension}|{query}"
        return hashlib.sha256(payload.encode()).hexdigest()

    def _get(self, key: str) -> list[float] | None:
        row = self._conn.execute(
            "SELECT embedding FROM query_embeddings WHERE cache_key = ?", (key,)
        ).fetchone()
        if row is None:
            return None
        raw: bytes = row[0]
        n = len(raw) // 4
        return list(struct.unpack(f"{n}f", raw))

    def _put(self, key: str, embedding: list[float]) -> None:
        raw = struct.pack(f"{len(embedding)}f", *embedding)
        self._conn.execute(
            "INSERT OR REPLACE INTO query_embeddings (cache_key, embedding, created_at) VALUES (?, ?, ?)",
            (key, raw, int(time.time())),
        )
        self._conn.commit()

    def cache_stats(self) -> dict:
        """Return cache statistics."""
        if self._conn is None:
            return {"entries": 0, "path": str(self._cache_path)}
        count = self._conn.execute("SELECT COUNT(*) FROM query_embeddings").fetchone()[0]
        return {"entries": count, "path": str(self._cache_path)}
