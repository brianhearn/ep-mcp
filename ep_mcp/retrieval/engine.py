"""Hybrid retrieval engine: vector + BM25 → fusion → boosting → MMR → graph expansion → file dedup.

Pipeline:
1. Intent classification → adaptive weight selection
2. Vector + BM25 search → normalize → fuse (with intent-adjusted weights)
3. Adaptive threshold filter (ratio-based) or legacy flat min_score
4. Metadata boosting
5. Length penalty
6. MMR re-ranking → top-K
7. Graph expansion → bonus neighbors scored and merged into results by score
8. File-level dedup → max_chunks_per_file per source file
9. Return final top-K
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
from .intent import IntentClassifier, QueryIntent
from .models import SearchRequest, SearchResult
from .scorer import (
    apply_adaptive_threshold,
    apply_length_penalty,
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
    1. Intent classification → adaptive weight selection
    2. Vector search (sqlite-vec) + BM25 search (FTS5) in parallel
    3. Score normalization
    4. Score fusion (intent-adjusted weighted combination)
    5. Adaptive threshold filter (or legacy flat min_score)
    6. Metadata boosting
    7. Length penalty
    8. MMR re-ranking (top-K candidates)
    9. Graph expansion → bonus neighbors merged into results by score
    10. File-level dedup (max_chunks_per_file per source file)
    11. Return final top-K
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

        # Intent classifier for query-adaptive weight routing
        self._intent_classifier = IntentClassifier()

    async def search(
        self,
        request: SearchRequest,
        graph_expansion_confidence_threshold: float | None = None,
        graph_expansion_min_score: float | None = None,
    ) -> list[SearchResult]:
        """Execute a hybrid search against the pack index.

        Args:
            request: Search query with optional type/tag filters

        Returns:
            Ranked list of SearchResult objects (top-K + optional graph-expanded bonus results)
        """
        max_results = request.max_results
        candidate_count = max_results * self.config.candidate_multiplier

        # Step 1: Intent classification → adaptive weight selection
        vector_weight = self.config.vector_weight
        text_weight = self.config.text_weight
        if self.config.intent_routing_enabled:
            intent_result = self._intent_classifier.classify(request.query)
            if intent_result.vector_weight is not None:
                vector_weight = intent_result.vector_weight
                text_weight = intent_result.text_weight
            logger.debug(
                "intent_routing | query='%s' intent=%s vector_weight=%.2f text_weight=%.2f",
                request.query, intent_result.intent.value, vector_weight, text_weight,
            )

        # Step 2: Dual search
        query_embedding = await self.provider.embed_query(request.query)
        vec_results = self.store.vector_search(query_embedding, limit=candidate_count)
        bm25_results = self.store.bm25_search(request.query, limit=candidate_count)

        logger.debug(
            "Search '%s': %d vector candidates, %d BM25 candidates",
            request.query, len(vec_results), len(bm25_results),
        )

        if not vec_results and not bm25_results:
            return []

        # Step 3: Normalize scores
        vec_results = normalize_vector_scores(vec_results)
        bm25_results = normalize_bm25_scores(bm25_results)

        # Step 4: Score fusion (with intent-adjusted weights)
        fused = fuse_scores(
            vec_results, bm25_results,
            vector_weight=vector_weight,
            text_weight=text_weight,
        )

        # Step 3b: Score filtering — adaptive threshold or legacy flat cutoff
        if self.config.adaptive_threshold:
            fused = apply_adaptive_threshold(
                fused,
                activation_floor=self.config.activation_floor,
                score_ratio=self.config.score_ratio,
                absolute_floor=self.config.absolute_floor,
            )
        else:
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

        # Step 4b: Length penalty — discount very short chunks
        fused = apply_length_penalty(
            fused, chunks,
            short_threshold=self.config.length_penalty_threshold,
            short_penalty=self.config.length_penalty_factor,
        )

        # Step 5: MMR re-ranking → top candidates (fetch more than max_results to allow dedup)
        scored_list = sorted(fused.items(), key=lambda x: x[1], reverse=True)
        mmr_k = max_results * max(2, self.config.max_chunks_per_file)  # fetch extra for dedup headroom

        if self.config.mmr_enabled and len(scored_list) > 1:
            all_chunk_ids = list(fused.keys())
            embeddings = self._load_embeddings_for_chunks(all_chunk_ids)
            scored_list = mmr_rerank(
                scored_list, embeddings,
                lambda_param=self.config.mmr_lambda,
                k=mmr_k,
            )
        else:
            scored_list = scored_list[:mmr_k]

        # Step 5b: File-level dedup on MMR candidates → limit chunks per source file
        if self.config.max_chunks_per_file > 0:
            file_counts_pre: dict[str, int] = {}
            deduped_scored: list[tuple[int, float]] = []
            for chunk_id, score in scored_list:
                chunk = chunks.get(chunk_id)
                fp = chunk["file_path"] if chunk else ""
                count = file_counts_pre.get(fp, 0)
                if count < self.config.max_chunks_per_file:
                    deduped_scored.append((chunk_id, score))
                    file_counts_pre[fp] = count + 1
            logger.debug(
                "file_dedup(pre-build) | before=%d after=%d (max_per_file=%d)",
                len(scored_list), len(deduped_scored), self.config.max_chunks_per_file,
            )
            scored_list = deduped_scored[:max_results]
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

        # Step 7: Graph expansion → score and merge bonus neighbors into result pool
        bonus_results = self._apply_graph_expansion(
            results=top_k_results,
            query_embedding=query_embedding,
            confidence_threshold_override=graph_expansion_confidence_threshold,
            min_score_override=graph_expansion_min_score,
        )

        # Merge top-K + bonus, re-sort by score descending
        # (bonus neighbors may slot in above lower-scoring core results)
        merged = top_k_results + bonus_results
        merged.sort(key=lambda r: r.score, reverse=True)

        # Return final top-K (file dedup already applied pre-build; bonus slots in cleanly)
        return merged[:max_results]

    def _apply_graph_expansion(
        self,
        results: list[SearchResult],
        query_embedding: list[float],
        confidence_threshold_override: float | None = None,
        min_score_override: float | None = None,
    ) -> list[SearchResult]:
        """Additive post-top-K graph expansion.

        Two modes controlled by config.graph_expansion_deep:

        **Standard (deep=False, default):** 1-hop expansion from high-confidence
        seeds. Loads immediate neighbors, scores them independently against the
        query, and adds qualifying neighbors as bonus results.

        **Deep (deep=True):** Multi-hop BFS expansion. Starting from
        high-confidence seeds, traverses up to config.graph_expansion_depth hops.
        Each hop applies a score discount (config.graph_expansion_discount).
        At each hop, only neighbors that clear the min_score threshold (after
        discount) are included AND eligible to seed the next hop. This prevents
        drift: low-relevance nodes don't propagate further. Capped at
        config.graph_expansion_deep_max_bonus bonus results.

        Both modes:
        - Do NOT displace or alter the top-K results
        - Gracefully no-op when graph is unavailable, expansion is disabled,
          or no high-confidence seeds
        - All bonus results are flagged with graph_expanded=True

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
        confidence_threshold = (
            confidence_threshold_override
            if confidence_threshold_override is not None
            else self.config.graph_expansion_confidence_threshold
        )
        min_score = (
            min_score_override
            if min_score_override is not None
            else self.config.graph_expansion_min_score
        )
        structural_bonus = self.config.graph_expansion_structural_bonus

        # Identify high-confidence seeds
        seeds = [r for r in results if r.score >= confidence_threshold]
        logger.info(
            "graph_expansion | seeds=%d/%d (threshold=%.2f)",
            len(seeds), len(results), confidence_threshold,
        )
        if not seeds:
            return []

        top_k_files = {r.source_file for r in results}
        seen_files: set[str] = set(top_k_files)

        if self.config.graph_expansion_deep:
            return self._deep_graph_expansion(
                seeds=seeds,
                query_embedding=query_embedding,
                graph=graph,
                lookup=lookup,
                seen_files=seen_files,
                min_score=min_score,
                structural_bonus=structural_bonus,
            )
        else:
            return self._shallow_graph_expansion(
                seeds=seeds,
                query_embedding=query_embedding,
                graph=graph,
                lookup=lookup,
                seen_files=seen_files,
                min_score=min_score,
                structural_bonus=structural_bonus,
            )

    def _shallow_graph_expansion(
        self,
        seeds: list[SearchResult],
        query_embedding: list[float],
        graph,
        lookup: GraphLookup,
        seen_files: set[str],
        min_score: float,
        structural_bonus: float,
    ) -> list[SearchResult]:
        """Original 1-hop graph expansion."""
        bonus: list[SearchResult] = []
        total_candidates = 0

        for seed in seeds:
            file_path = seed.source_file
            neighbor_fps = lookup.get_neighbor_file_paths(file_path, graph)

            for n_fp in neighbor_fps:
                if n_fp in seen_files:
                    continue

                total_candidates += 1
                result = self._score_neighbor(
                    n_fp, query_embedding, min_score, structural_bonus,
                )
                if result:
                    bonus.append(result)
                    seen_files.add(n_fp)

        bonus_files = [r.source_file for r in bonus]
        logger.info(
            "graph_expansion(shallow) | candidates=%d bonus=%d bonus_files=%s",
            total_candidates, len(bonus), bonus_files,
        )
        return bonus

    def _deep_graph_expansion(
        self,
        seeds: list[SearchResult],
        query_embedding: list[float],
        graph,
        lookup: GraphLookup,
        seen_files: set[str],
        min_score: float,
        structural_bonus: float,
    ) -> list[SearchResult]:
        """Multi-hop BFS graph expansion with per-hop score decay.

        BFS frontier starts with seed file_paths. At each hop:
        1. Expand all frontier nodes to their 1-hop neighbors
        2. Score each unseen neighbor independently against the query
        3. Apply discount^hop to the structural bonus
        4. Only neighbors clearing min_score (after discount) join the
           bonus list AND become seeds for the next hop
        5. Stop at max depth or when bonus cap is reached
        """
        max_depth = self.config.graph_expansion_depth
        max_bonus = self.config.graph_expansion_deep_max_bonus
        discount = self.config.graph_expansion_discount

        bonus: list[SearchResult] = []
        total_candidates = 0

        # BFS frontier: start with seed file paths
        frontier: list[str] = [s.source_file for s in seeds]

        for hop in range(1, max_depth + 1):
            if len(bonus) >= max_bonus:
                break

            hop_discount = discount ** hop
            next_frontier: list[str] = []

            for fp in frontier:
                neighbor_fps = lookup.get_neighbor_file_paths(fp, graph)

                for n_fp in neighbor_fps:
                    if n_fp in seen_files:
                        continue

                    total_candidates += 1
                    seen_files.add(n_fp)  # mark seen immediately to prevent re-visit

                    result = self._score_neighbor(
                        n_fp, query_embedding, min_score,
                        structural_bonus * hop_discount,
                    )
                    if result:
                        bonus.append(result)
                        # This neighbor qualifies — it can seed the next hop
                        next_frontier.append(n_fp)

                        if len(bonus) >= max_bonus:
                            break

                if len(bonus) >= max_bonus:
                    break

            frontier = next_frontier
            if not frontier:
                break

            logger.debug(
                "graph_expansion(deep) | hop=%d discount=%.3f frontier=%d bonus_so_far=%d",
                hop, hop_discount, len(frontier), len(bonus),
            )

        bonus_files = [r.source_file for r in bonus]
        logger.info(
            "graph_expansion(deep) | depth=%d candidates=%d bonus=%d bonus_files=%s",
            max_depth, total_candidates, len(bonus), bonus_files,
        )
        return bonus

    def _score_neighbor(
        self,
        file_path: str,
        query_embedding: list[float],
        min_score: float,
        effective_bonus: float,
    ) -> SearchResult | None:
        """Load, embed-score, and build a SearchResult for a neighbor file.

        The min_score check is applied to the *final* score (cosine_sim × effective_bonus),
        not the raw cosine similarity alone. This ensures that per-hop decay in
        deep expansion is reflected in the filtering threshold.

        Returns None if the neighbor can't be loaded or doesn't clear min_score.
        """
        neighbor_chunks_by_fp = self.store.get_chunks_by_file_paths([file_path])
        if not neighbor_chunks_by_fp or file_path not in neighbor_chunks_by_fp:
            return None

        fp_chunks = neighbor_chunks_by_fp[file_path]
        primary_chunk = next(
            (nc for nc in fp_chunks if nc.get("chunk_index", 0) == 0),
            None,
        )
        if not primary_chunk:
            return None

        chunk_id = primary_chunk["id"]
        neighbor_embeddings = self._load_embeddings_for_chunks([chunk_id])
        if chunk_id not in neighbor_embeddings:
            return None

        neighbor_emb = neighbor_embeddings[chunk_id]
        cosine_sim = float(np.dot(query_embedding, neighbor_emb))
        final_score = cosine_sim * effective_bonus

        if final_score < min_score:
            logger.debug(
                "graph_expansion | neighbor '%s' rejected (final=%.4f < min_score=%.2f, "
                "cosine=%.4f, bonus=%.3f)",
                file_path, final_score, min_score, cosine_sim, effective_bonus,
            )
            return None

        tags = _parse_tags(primary_chunk.get("tags", "[]"))

        return SearchResult(
            text=primary_chunk["content"],
            source_file=primary_chunk.get("file_path", file_path),
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
