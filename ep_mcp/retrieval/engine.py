"""Hybrid retrieval engine: vector + BM25 → fusion → boosting → MMR → post-top-K graph expansion.

Updated pipeline (per new additive design):
1. Vector + BM25 search → normalize → fuse → min_score filter
2. Metadata boosting
3. MMR re-ranking → top-K (finalized here, no interference from expansion)
4. Post-top-K graph expansion (additive bonus neighbors only)
5. Return top-K + bonus neighbors (flagged with graph_expanded=True)
"""

import json
import logging
import struct
from dataclasses import dataclass

import numpy as np

from ..config import RetrievalConfig
from ..embeddings.base import EmbeddingProvider
from ..index.sqlite_store import SQLiteStore
from ..pack.models import Pack
from .graph_helpers import GraphLookup
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
    4. Filter by min_score
    5. Metadata boosting
    6. MMR re-ranking (final top-K)
    7. Post-top-K additive graph expansion (bonus results only)
    8. Return results
    """

    def __init__(
        self,
        pack: Pack,
        store: SQLiteStore,
        embedding_provider: EmbeddingProvider,
        config: RetrievalConfig | None = None,
        graph_lookup: GraphLookup | None = None,
    ):
        self.pack = pack
        self.store = store
        self.provider = embedding_provider
        self.config = config or RetrievalConfig()

        # Pre-compute always-tier files
        self._always_files = set(self.pack.manifest.context.always)

        # Graph lookup for file_path ↔ node_id mapping
        self._graph_lookup = graph_lookup or GraphLookup.from_pack(pack)

    async def search(self, request: SearchRequest) -> list[SearchResult]:
        """Execute a hybrid search against the pack index.

        Args:
            request: Search query with optional type/tag filters

        Returns:
            Ranked list of SearchResult objects (top-K + optional graph-expanded bonus results)
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

        # Step 5: MMR re-ranking → finalized top-K
        scored_list = sorted(fused.items(), key=lambda x: x[1], reverse=True)

        if self.config.mmr_enabled and len(scored_list) > 1:
            all_chunk_ids = list(fused.keys())
            embeddings = self._load_embeddings_for_chunks(all_chunk_ids)
            scored_list = mmr_rerank(
                scored_list, embeddings,
                lambda_param=self.config.mmr_lambda,
                k=max_results,
            )
        else:
            scored_list = scored_list[:max_results]

        # Step 6: Build core results from MMR top-K (before expansion)
        top_k_results = []
        top_k_chunk_ids_for_lookup = {}  # chunk_id -> result for later reference if needed
        for chunk_id, score in scored_list:
            chunk = chunks.get(chunk_id)
            if not chunk:
                continue

            tags = _parse_tags(chunk.get("tags", "[]"))

            result = SearchResult(
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
                graph_expanded=False,
            )
            top_k_results.append(result)
            top_k_chunk_ids_for_lookup[chunk_id] = result

        # Step 7: Post-top-K additive graph expansion
        bonus_results = self._apply_graph_expansion(
            results=top_k_results,
            query_embedding=query_embedding,
        )

        # Combine: top-K + bonus neighbors (bonus come after)
        return top_k_results + bonus_results

    def _apply_graph_expansion(
        self,
        results: list[SearchResult],
        query_embedding: list[float],
    ) -> list[SearchResult]:
        """Additive post-top-K graph expansion.

        Identifies high-confidence seeds from finalized top-K, loads 1-hop neighbors
        (excluding anything already in top-K), scores them independently against the
        original query_embedding using cosine similarity, and includes qualifying
        neighbors as bonus results flagged with graph_expanded=True.

        Does NOT displace or alter the top-K results. Gracefully no-ops when
        graph is unavailable, expansion is disabled, or no high-confidence seeds.

        Args:
            results: Finalized top-K SearchResult list (post-MMR).
            query_embedding: Original query embedding vector for independent neighbor scoring.

        Returns:
            List of bonus SearchResult objects (may be empty).
        """
        if not self.config.graph_expansion_enabled:
            return []
        if self.pack.graph is None or self._graph_lookup is None:
            return []

        graph = self.pack.graph
        lookup = self._graph_lookup
        confidence_threshold = self.config.graph_expansion_confidence_threshold
        min_score = self.config.graph_expansion_min_score
        structural_bonus = self.config.graph_expansion_structural_bonus

        # Identify high-confidence seeds
        seeds = [r for r in results if r.score >= confidence_threshold]
        if not seeds:
            return []

        top_k_files = {r.source_file for r in results}
        bonus: list[SearchResult] = []
        seen_files: set[str] = set(top_k_files)

        for seed in seeds:
            file_path = seed.source_file
            neighbor_fps = lookup.get_neighbor_file_paths(file_path, graph)

            for n_fp in neighbor_fps:
                if n_fp in seen_files:
                    continue

                # Load primary chunk (chunk_index=0) for the neighbor file
                neighbor_chunks_by_fp = self.store.get_chunks_by_file_paths([n_fp])
                if not neighbor_chunks_by_fp or n_fp not in neighbor_chunks_by_fp:
                    continue

                fp_chunks = neighbor_chunks_by_fp[n_fp]
                # Select primary chunk (chunk_index=0 preferred)
                primary_chunk = next(
                    (nc for nc in fp_chunks if nc.get("chunk_index", 0) == 0),
                    None,
                )
                if not primary_chunk:
                    continue

                chunk_id = primary_chunk["id"]

                # Load neighbor embedding (primary chunk)
                neighbor_embeddings = self._load_embeddings_for_chunks([chunk_id])
                if chunk_id not in neighbor_embeddings:
                    continue

                neighbor_emb = neighbor_embeddings[chunk_id]

                # Independent cosine similarity (embeddings are normalized, dot = cosine)
                cosine_sim = float(np.dot(query_embedding, neighbor_emb))

                if cosine_sim < min_score:
                    continue

                final_score = cosine_sim * structural_bonus

                tags = _parse_tags(primary_chunk.get("tags", "[]"))

                bonus_result = SearchResult(
                    text=primary_chunk["content"],
                    source_file=primary_chunk.get("file_path", n_fp),
                    id=primary_chunk.get("prov_id"),
                    content_hash=primary_chunk.get("content_hash"),
                    verified_at=primary_chunk.get("verified_at"),
                    score=round(final_score, 4),
                    type=primary_chunk.get("type"),
                    tags=tags,
                    chunk_index=primary_chunk.get("chunk_index", 0),
                    title=primary_chunk.get("title"),
                    graph_expanded=True,
                )

                bonus.append(bonus_result)
                seen_files.add(n_fp)

        if bonus:
            logger.debug(
                "Post-top-K graph expansion added %d bonus neighbor(s) from %d seed(s)",
                len(bonus), len(seeds),
            )

        return bonus

    def _load_embeddings_for_chunks(self, chunk_ids: list[int]) -> dict[int, list[float]]:
        """Load embedding vectors for MMR or graph expansion scoring.

        Reads directly from the vec table. Returns {chunk_id: embedding_list}.
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


def _parse_tags(tags_json: str) -> list[str]:
    """Parse tags from JSON string."""
    try:
        tags = json.loads(tags_json)
        return tags if isinstance(tags, list) else []
    except (json.JSONDecodeError, TypeError):
        return []
