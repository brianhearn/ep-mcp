"""Hybrid retrieval engine: vector + BM25 → fusion → boosting → graph expansion → MMR."""

from __future__ import annotations

import json
import logging
import struct
from dataclasses import dataclass

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

        # Step 4.5: Graph expansion
        fused, chunks = self._apply_graph_expansion(
            fused, chunks, max_results=max_results,
        )

        # Sort by score for MMR input
        scored_list = sorted(fused.items(), key=lambda x: x[1], reverse=True)

        # Step 5: MMR re-ranking
        if self.config.mmr_enabled and len(scored_list) > 1:
            # Load embeddings for MMR diversity calculation (use current fused keys
            # which may include graph-expanded chunks)
            all_chunk_ids = list(fused.keys())
            embeddings = self._load_embeddings_for_chunks(all_chunk_ids)
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

    def _apply_graph_expansion(
        self,
        fused: dict[int, float],
        chunks: dict[int, dict],
        max_results: int,
    ) -> tuple[dict[int, float], dict[int, dict]]:
        """Expand candidate set by adding graph-connected neighbor chunks.

        For each top-scoring result, finds neighbors via the knowledge graph
        and adds their chunks with a discounted score. Gracefully no-ops when
        graph data is unavailable or expansion is disabled.

        Args:
            fused: Current {chunk_id: score} mapping after metadata boosting.
            chunks: Current {chunk_id: chunk_data} mapping.
            max_results: Cap on total graph-expanded additions.

        Returns:
            Updated (fused, chunks) tuple with graph neighbors included.
        """
        # Guard: no-op conditions
        if not self.config.graph_expansion_enabled:
            return fused, chunks
        if self.pack.graph is None:
            return fused, chunks
        if self._graph_lookup is None:
            return fused, chunks

        graph = self.pack.graph
        lookup = self._graph_lookup
        discount = self.config.graph_expansion_discount
        depth = self.config.graph_expansion_depth
        graph_min_score = self.config.graph_expansion_min_score

        # Sort current candidates by score descending, take top max_results as seeds
        sorted_candidates = sorted(fused.items(), key=lambda x: x[1], reverse=True)
        seed_chunks = sorted_candidates[:max_results]

        # Collect neighbor file paths across all depth levels
        added_count = 0
        existing_chunk_ids = set(fused.keys())

        for chunk_id, parent_score in seed_chunks:
            if added_count >= max_results:
                break

            chunk = chunks.get(chunk_id)
            if not chunk:
                continue

            file_path = chunk.get("file_path")
            if not file_path:
                continue

            # Multi-hop: BFS through the graph with visited tracking
            visited_files: set[str] = {file_path}
            current_level_files = {file_path}
            current_discount = discount

            for hop in range(depth):
                if added_count >= max_results:
                    break

                next_level_files: set[str] = set()
                for fp in current_level_files:
                    neighbor_fps = lookup.get_neighbor_file_paths(fp, graph)
                    next_level_files.update(neighbor_fps)

                # Remove all previously visited files (prevents cycles)
                next_level_files -= visited_files

                if not next_level_files:
                    break

                # Look up chunks for neighbor file paths
                neighbor_chunks_by_fp = self.store.get_chunks_by_file_paths(
                    list(next_level_files)
                )

                for fp, fp_chunks in neighbor_chunks_by_fp.items():
                    if added_count >= max_results:
                        break
                    for nc in fp_chunks:
                        nc_id = nc["id"]
                        if nc_id in existing_chunk_ids:
                            continue

                        neighbor_score = parent_score * current_discount
                        if neighbor_score < graph_min_score:
                            continue

                        fused[nc_id] = neighbor_score
                        chunks[nc_id] = nc
                        existing_chunk_ids.add(nc_id)
                        added_count += 1

                        if added_count >= max_results:
                            break

                # Next hop: deeper discount, advance frontier
                visited_files.update(next_level_files)
                current_level_files = next_level_files
                current_discount *= discount

        if added_count > 0:
            logger.debug(
                "Graph expansion added %d neighbor chunks (depth=%d, discount=%.2f)",
                added_count, depth, discount,
            )

        return fused, chunks

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


def _parse_tags(tags_json: str) -> list[str]:
    """Parse tags from JSON string."""
    try:
        tags = json.loads(tags_json)
        return tags if isinstance(tags, list) else []
    except (json.JSONDecodeError, TypeError):
        return []
