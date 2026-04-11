"""Hybrid retrieval engine: vector + BM25 → fusion → boosting → MMR."""

from __future__ import annotations

import logging
import struct
from dataclasses import dataclass

from ..config import RetrievalConfig
from ..embeddings.base import EmbeddingProvider
from ..index.sqlite_store import SQLiteStore
from ..pack.models import Pack
from .models import SearchRequest, SearchResult
from .scorer import (
    apply_metadata_boosts,
    fuse_scores,
    mmr_rerank,
    normalize_bm25_scores,
    normalize_vector_scores,
)

logger = logging.getLogger(__name__)


class RetrievalEngine:
    """Hybrid search engine for a single pack.

    Pipeline (ARCHITECTURE.md §5.1):
    1. Vector search (sqlite-vec) + BM25 search (FTS5) in parallel
    2. Score normalization
    3. Score fusion (weighted combination)
    4. Metadata boosting (type, tags, always-tier)
    5. MMR re-ranking (diversity)
    6. Return top K
    """

    def __init__(
        self,
        pack: Pack,
        store: SQLiteStore,
        embedding_provider: EmbeddingProvider,
        config: RetrievalConfig | None = None,
    ):
        self.pack = pack
        self.store = store
        self.provider = embedding_provider
        self.config = config or RetrievalConfig()

        # Pre-compute always-tier files
        self._always_files = set(self.pack.manifest.context.always)

    async def search(self, request: SearchRequest) -> list[SearchResult]:
        """Execute a hybrid search against the pack index.

        Args:
            request: Search query with optional type/tag filters

        Returns:
            Ranked list of SearchResult objects
        """
        max_results = request.max_results
        candidate_count = max_results * self.config.candidate_multiplier

        # Step 1: Dual search
        query_embedding = await self.provider.embed_query(request.query)
        vec_results = self.store.vector_search(query_embedding, limit=candidate_count)
        bm25_results = self.store.bm25_search(request.query, limit=candidate_count)

        logger.debug(
            "Search '%s': %d vector candidates, %d BM25 candidates",
            request.query, len(vec_results), len(bm25_results),
        )

        if not vec_results and not bm25_results:
            return []

        # Step 2: Normalize scores
        vec_results = normalize_vector_scores(vec_results)
        bm25_results = normalize_bm25_scores(bm25_results)

        # Step 3: Score fusion
        fused = fuse_scores(
            vec_results, bm25_results,
            vector_weight=self.config.vector_weight,
            text_weight=self.config.text_weight,
        )

        # Filter by min_score
        fused = {cid: s for cid, s in fused.items() if s >= self.config.min_score}

        if not fused:
            return []

        # Load chunk data for boosting and result building
        chunk_ids = list(fused.keys())
        chunks = self.store.get_chunks_by_ids(chunk_ids)

        # Step 4: Metadata boosting
        fused = apply_metadata_boosts(
            fused, chunks,
            type_filter=request.type,
            tag_filter=request.tags,
            always_files=self._always_files,
            type_match_boost=self.config.type_match_boost,
            tag_match_boost=self.config.tag_match_boost,
            always_tier_boost=self.config.always_tier_boost,
        )

        # Sort by score for MMR input
        scored_list = sorted(fused.items(), key=lambda x: x[1], reverse=True)

        # Step 5: MMR re-ranking
        if self.config.mmr_enabled and len(scored_list) > 1:
            # Load embeddings for MMR diversity calculation
            embeddings = self._load_embeddings_for_chunks(chunk_ids)
            scored_list = mmr_rerank(
                scored_list, embeddings,
                lambda_param=self.config.mmr_lambda,
                k=max_results,
            )
        else:
            scored_list = scored_list[:max_results]

        # Step 6: Build results
        results = []
        for chunk_id, score in scored_list:
            chunk = chunks.get(chunk_id)
            if not chunk:
                continue

            tags = _parse_tags(chunk.get("tags", "[]"))

            results.append(SearchResult(
                text=chunk["content"],
                source_file=chunk["file_path"],
                id=chunk.get("prov_id"),
                content_hash=chunk.get("content_hash"),
                verified_at=chunk.get("verified_at"),
                score=round(score, 4),
                type=chunk.get("type"),
                tags=tags,
                chunk_index=chunk.get("chunk_index", 0),
                title=chunk.get("title"),
            ))

        return results

    def _load_embeddings_for_chunks(self, chunk_ids: list[int]) -> dict[int, list[float]]:
        """Load embedding vectors for MMR calculation.

        Reads directly from the vec table.
        """
        embeddings: dict[int, list[float]] = {}
        for cid in chunk_ids:
            row = self.store.conn.execute(
                "SELECT embedding FROM chunks_vec WHERE chunk_id = ?", (cid,)
            ).fetchone()
            if row and row["embedding"]:
                blob = row["embedding"]
                n = len(blob) // 4
                embeddings[cid] = list(struct.unpack(f"{n}f", blob))
        return embeddings


import json


def _parse_tags(tags_json: str) -> list[str]:
    """Parse tags from JSON string."""
    try:
        tags = json.loads(tags_json)
        return tags if isinstance(tags, list) else []
    except (json.JSONDecodeError, TypeError):
        return []
