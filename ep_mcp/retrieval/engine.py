"""Hybrid retrieval engine: vector + BM25 → fusion → boosting → [graph widen] → MMR → graph expansion → reserved-slot merge.

Pipeline:
1. Intent classification → adaptive weight selection
2. Vector + BM25 search → normalize → fuse (with intent-adjusted weights)
3. Adaptive threshold filter (ratio-based) or legacy flat min_score
4. Metadata boosting
5. Length penalty
5b. [OPT] Pre-MMR graph widen: pull neighbors of high-confidence fused candidates
    into the pool BEFORE MMR so they compete on merit (config.graph_widen_enabled).
6. MMR re-ranking → top-K
7. File-level dedup → max_chunks_per_file per source file
8. Post-K graph expansion → bonus neighbors scored
9. Reserved-slot merge: guarantee `graph_expansion_reserved_slots` slots for the
   best graph-expanded bonuses when they'd otherwise lose the merge-sort. Falls
   back to legacy merge-sort if reserved_slots=0.
10. Cross-encoder reranker (if configured)
11. Return final top-K
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
from .reranker import Reranker
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
    score_bm25_fallback,
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
        reranker: "Reranker | None" = None,
    ):
        self.pack = pack
        self.store = store
        self.provider = embedding_provider
        self.config = config or RetrievalConfig()
        self._reranker = reranker

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
        # Embedding may fail (provider outage, quota, etc.). Gracefully degrade to
        # BM25-only mode using lexical scoring rather than crashing the whole request.
        query_embedding: list[float] | None = None
        embedding_failed = False
        try:
            query_embedding = await self.provider.embed_query(request.query)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "[PIPELINE] embedding failed (%s: %s) — degrading to BM25-only mode",
                type(exc).__name__, exc,
            )
            embedding_failed = True

        if embedding_failed:
            return await self._search_bm25_fallback(request, max_results, candidate_count)

        vec_results = self.store.vector_search(query_embedding, limit=candidate_count)
        bm25_results = self.store.bm25_search(
            request.query,
            limit=candidate_count,
            min_token_match_ratio=self.config.bm25_min_token_match_ratio,
        )

        logger.info(
            "[PIPELINE] query='%s' | vector=%d BM25=%d candidates (pool=%d)",
            request.query, len(vec_results), len(bm25_results), candidate_count,
        )

        if not vec_results and not bm25_results:
            return []

        # Step 3: Normalize scores
        vec_results = normalize_vector_scores(vec_results)
        bm25_results = normalize_bm25_scores(bm25_results, bm25_cap=self.config.bm25_cap)

        # Capture the best vector-only score before fusion.
        # Used as the anchor for adaptive threshold so a BM25 spike
        # doesn't inflate the cutoff and kill vector-matched results.
        # Best vector-only score, weighted to be on the same scale as fused scores.
        # This prevents a BM25 spike from inflating the adaptive threshold anchor.
        vec_best_score = vector_weight * max(
            (r["vec_score"] for r in vec_results), default=0.0,
        ) if vec_results else 0.0

        # Step 4: Score fusion (with intent-adjusted weights)
        fused = fuse_scores(
            vec_results, bm25_results,
            vector_weight=vector_weight,
            text_weight=text_weight,
            bm25_cap=self.config.bm25_cap,
        )

        # Debug: log top-20 fused scores before threshold
        _log_top_chunks("after_fusion", fused, chunks_store=self.store, top_n=20)

        # Step 3b: Score filtering — adaptive threshold or legacy flat cutoff
        # Anchor on the vector-only best score, not the fused best score,
        # so BM25 keyword spikes don't raise the threshold above genuine
        # semantic matches.
        pre_threshold_count = len(fused)
        if self.config.adaptive_threshold:
            fused = apply_adaptive_threshold(
                fused,
                activation_floor=self.config.activation_floor,
                score_ratio=self.config.score_ratio,
                absolute_floor=self.config.absolute_floor,
                anchor_score=vec_best_score,
            )
        else:
            fused = {cid: s for cid, s in fused.items() if s >= self.config.min_score}

        logger.info(
            "[PIPELINE] threshold_filter | before=%d after=%d dropped=%d",
            pre_threshold_count, len(fused), pre_threshold_count - len(fused),
        )

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

        # Debug: log top-20 after boosts + penalty
        _log_top_chunks("after_boosts", fused, chunks_store=self.store, top_n=20)

        # Step 5b: Pre-MMR graph widen — pull graph neighbors of top pre-cutoff
        # files into the candidate pool as regular candidates. Unlike post-K
        # graph expansion (which appends discounted bonuses), this widen lets
        # related files compete for top-K slots on their raw cosine score.
        # Targets the secondary-file miss pattern (e.g. interface files for
        # workflow queries).
        if (
            self.config.graph_widen_enabled
            and self.pack.graph is not None
            and self._graph_lookup is not None
        ):
            fused, chunks = self._graph_widen_candidates(
                fused, chunks, query_embedding,
                max_seeds=self.config.graph_widen_max_seeds,
                min_seed_score=self.config.graph_widen_min_seed_score,
                min_neighbor_score=self.config.graph_widen_min_neighbor_score,
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
            # Debug: log top-20 after MMR
            _log_top_chunks_scored("after_mmr", scored_list[:20], chunks_store=self.store)
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
            logger.info(
                "[PIPELINE] file_dedup | before=%d after=%d (max_per_file=%d)",
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

        # Step 7b: Merge top-K + bonus with reserved-slot guarantee.
        #
        # Legacy behavior (reserved_slots=0): concat + sort by score,
        # truncate to max_results. Bonus neighbors compete directly;
        # because structural_bonus typically discounts them below core
        # results, they rarely surface. Secondary-file misses persist.
        #
        # Reserved-slot behavior (reserved_slots>0): guarantee up to N
        # slots for the best graph-expanded bonuses, filled only with
        # files NOT already in the core result set. This turns graph
        # expansion into a real candidate-widener for secondary files.
        reserved = self.config.graph_expansion_reserved_slots
        if reserved > 0 and bonus_results:
            merged = self._reserved_slot_merge(
                top_k_results, bonus_results, max_results, reserved,
            )
        else:
            merged = top_k_results + bonus_results
            merged.sort(key=lambda r: r.score, reverse=True)

        # Step 8: Cross-encoder rerank (second-pass precision layer)
        if self._reranker is not None:
            merged = self._reranker.rerank(request.query, merged)

        # Return final top-K (file dedup already applied pre-build; bonus slots in cleanly)
        return merged[:max_results]

    async def _search_bm25_fallback(
        self,
        request: SearchRequest,
        max_results: int,
        candidate_count: int,
    ) -> list[SearchResult]:
        """Degraded BM25-only search when the embedding provider is unavailable.

        Uses :func:`score_bm25_fallback` for richer lexical scoring (term coverage,
        density, path boost, length boost) similar to OpenClaw's fallback mode.
        Does NOT perform MMR or graph expansion (both need embeddings).
        Adaptive threshold still applies with BM25-only anchor.
        """
        logger.info("[BM25_FALLBACK] running BM25-only search for query='%s'", request.query)
        bm25_results = self.store.bm25_search(
            request.query,
            limit=candidate_count,
            min_token_match_ratio=self.config.bm25_min_token_match_ratio,
        )
        if not bm25_results:
            logger.info("[BM25_FALLBACK] no BM25 results")
            return []

        bm25_results = normalize_bm25_scores(bm25_results)

        # Build chunk map
        chunk_ids = [r["chunk_id"] for r in bm25_results]
        chunks = self.store.get_chunks_by_ids(chunk_ids)

        # Score with lexical fallback scorer
        scores = score_bm25_fallback(bm25_results, chunks, request.query)

        # Adaptive threshold (no vector anchor available — use best fused score)
        if self.config.adaptive_threshold:
            scores = apply_adaptive_threshold(
                scores,
                activation_floor=self.config.activation_floor,
                score_ratio=self.config.score_ratio,
                absolute_floor=self.config.absolute_floor,
                anchor_score=None,  # no vector anchor in fallback
            )
        else:
            scores = {cid: s for cid, s in scores.items() if s >= self.config.min_score}

        if not scores:
            return []

        # Metadata boosts + length penalty
        scores = apply_metadata_boosts(
            scores, chunks,
            type_filter=request.type,
            tag_filter=request.tags,
            always_files=self._always_files,
            type_match_boost=self.config.type_match_boost,
            tag_match_boost=self.config.tag_match_boost,
            always_tier_boost=self.config.always_tier_boost,
        )
        scores = apply_length_penalty(
            scores, chunks,
            short_threshold=self.config.length_penalty_threshold,
            short_penalty=self.config.length_penalty_factor,
        )

        # Sort, dedup per file, truncate
        scored_list = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        if self.config.max_chunks_per_file > 0:
            file_counts: dict[str, int] = {}
            deduped: list[tuple[int, float]] = []
            for chunk_id, score in scored_list:
                chunk = chunks.get(chunk_id)
                fp = chunk["file_path"] if chunk else ""
                if file_counts.get(fp, 0) < self.config.max_chunks_per_file:
                    deduped.append((chunk_id, score))
                    file_counts[fp] = file_counts.get(fp, 0) + 1
            scored_list = deduped
        scored_list = scored_list[:max_results]

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
                graph_expanded=False,
            ))

        logger.info("[BM25_FALLBACK] returning %d results", len(results))
        return results

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

    def _graph_widen_candidates(
        self,
        fused: dict[int, float],
        chunks: dict[int, dict],
        query_embedding: list[float],
        max_seeds: int = 5,
        min_seed_score: float = 0.38,
        min_neighbor_score: float = 0.25,
    ) -> tuple[dict[int, float], dict[int, dict]]:
        """Widen the candidate pool with graph neighbors BEFORE MMR.

        Identifies the top `max_seeds` files (post-boosting, pre-MMR) that
        clear `min_seed_score`, pulls their 1-hop graph neighbors, scores each
        neighbor's primary chunk via cosine similarity against the query
        embedding, and injects qualifying neighbors as regular candidates.

        Unlike post-K graph expansion (which appends structurally-discounted
        bonus results), this widen produces candidates that flow through MMR
        and the normal top-K cutoff with their raw cosine score — letting
        them compete on merit.

        Targets the secondary-file miss pattern: the right file exists and
        is graph-linked to a retrieved file, but never entered the vector
        candidate pool because the vector search didn't rank it high enough.

        Returns the augmented (fused, chunks) pair. Existing candidates are
        untouched; only new neighbor candidates are added.
        """
        graph = self.pack.graph
        lookup = self._graph_lookup
        if graph is None or lookup is None:
            return fused, chunks

        # Identify seed files (top-N distinct files by score, clearing threshold)
        ranked = sorted(fused.items(), key=lambda x: x[1], reverse=True)
        seed_files: list[str] = []
        for cid, score in ranked:
            if score < min_seed_score:
                break
            chunk = chunks.get(cid)
            if not chunk:
                continue
            fp = chunk.get("file_path")
            if fp and fp not in seed_files:
                seed_files.append(fp)
            if len(seed_files) >= max_seeds:
                break

        if not seed_files:
            logger.info(
                "[PIPELINE] graph_widen | no seeds cleared min_seed_score=%.2f",
                min_seed_score,
            )
            return fused, chunks

        # Collect candidate neighbor file paths (exclude files already in pool)
        existing_files = {
            chunks[cid].get("file_path") for cid in fused if cid in chunks
        }
        existing_files.discard(None)
        neighbor_candidates: set[str] = set()
        for fp in seed_files:
            for n_fp in lookup.get_neighbor_file_paths(fp, graph):
                if n_fp not in existing_files:
                    neighbor_candidates.add(n_fp)

        if not neighbor_candidates:
            logger.info(
                "[PIPELINE] graph_widen | seeds=%d, no new neighbors",
                len(seed_files),
            )
            return fused, chunks

        # Load primary chunks + embeddings for all neighbor files in one batch
        neighbor_chunks_by_fp = self.store.get_chunks_by_file_paths(list(neighbor_candidates))

        # Collect all primary chunk ids to load embeddings in one shot
        primary_by_fp: dict[str, dict] = {}
        for n_fp, fp_chunks in neighbor_chunks_by_fp.items():
            primary = next((c for c in fp_chunks if c.get("chunk_index", 0) == 0), None)
            if primary:
                primary_by_fp[n_fp] = primary

        if not primary_by_fp:
            return fused, chunks

        primary_chunk_ids = [p["id"] for p in primary_by_fp.values()]
        neighbor_embs = self._load_embeddings_for_chunks(primary_chunk_ids)

        added = 0
        added_files: list[str] = []
        for n_fp, primary in primary_by_fp.items():
            cid = primary["id"]
            if cid not in neighbor_embs:
                continue
            cosine = float(np.dot(query_embedding, neighbor_embs[cid]))
            if cosine < min_neighbor_score:
                continue
            # Inject as a regular candidate (raw cosine, no structural discount)
            fused[cid] = cosine
            chunks[cid] = primary
            added += 1
            added_files.append(n_fp)

        logger.info(
            "[PIPELINE] graph_widen | seeds=%d neighbor_candidates=%d added=%d added_files=%s",
            len(seed_files), len(neighbor_candidates), added, added_files,
        )
        return fused, chunks

    def _reserved_slot_merge(
        self,
        core: list[SearchResult],
        bonus: list[SearchResult],
        max_results: int,
        reserved_slots: int,
    ) -> list[SearchResult]:
        """Merge core top-K with graph-expanded bonuses, reserving slots for bonuses.

        Guarantees up to `reserved_slots` final slots for the best graph-expanded
        bonus results (files not already in core), even when their structurally-
        discounted score would lose a raw merge-sort competition.

        Layout:
          final[0..core_slots-1]     → top core results by score
          final[core_slots..total-1] → top reserved-slot bonuses by score

        Where core_slots = max_results - min(reserved_slots, len(unique_bonuses)).
        If fewer bonuses qualify than reserved slots, core expands to fill the gap.

        Within each section, results are sorted by score. The merged list is
        returned with core results first, then bonus results — NOT re-sorted
        globally, because the whole point is to guarantee bonus representation.
        """
        # Deduplicate bonus files against core files
        core_files = {r.source_file for r in core}
        unique_bonuses = [b for b in bonus if b.source_file not in core_files]

        # How many bonus slots can we actually fill?
        actual_reserved = min(reserved_slots, len(unique_bonuses), max_results)
        core_slots = max_results - actual_reserved

        # Take top core_slots core results (already score-sorted upstream, but be safe)
        core_sorted = sorted(core, key=lambda r: r.score, reverse=True)[:core_slots]

        # Take top actual_reserved bonuses by score
        bonus_sorted = sorted(unique_bonuses, key=lambda r: r.score, reverse=True)[:actual_reserved]

        logger.info(
            "[PIPELINE] reserved_slot_merge | core=%d/%d bonus=%d/%d (reserved=%d)",
            len(core_sorted), len(core), len(bonus_sorted), len(unique_bonuses),
            reserved_slots,
        )

        return core_sorted + bonus_sorted

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


def _log_top_chunks(
    stage: str,
    fused: dict,
    chunks_store,
    top_n: int = 20,
) -> None:
    """Log top-N chunk scores at a pipeline stage, showing file path."""
    if not fused:
        return
    top = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:top_n]
    chunk_ids = [cid for cid, _ in top]
    chunks = chunks_store.get_chunks_by_ids(chunk_ids)
    logger.info("[STAGE:%s] top-%d chunks:", stage, top_n)
    for i, (cid, score) in enumerate(top):
        chunk = chunks.get(cid, {})
        fp = chunk.get("file_path", "?")
        logger.info("  %2d. score=%.4f  %s  (chunk_id=%d)", i + 1, score, fp, cid)


def _log_top_chunks_scored(
    stage: str,
    scored_list: list,
    chunks_store,
) -> None:
    """Log scored list (list of (chunk_id, score) tuples) at a pipeline stage."""
    if not scored_list:
        return
    chunk_ids = [cid for cid, _ in scored_list]
    chunks = chunks_store.get_chunks_by_ids(chunk_ids)
    logger.info("[STAGE:%s] top-%d chunks:", stage, len(scored_list))
    for i, (cid, score) in enumerate(scored_list):
        chunk = chunks.get(cid, {})
        fp = chunk.get("file_path", "?")
        logger.info("  %2d. score=%.4f  %s  (chunk_id=%d)", i + 1, score, fp, cid)


def _parse_tags(tags_json: str) -> list[str]:
    """Parse tags from JSON string."""
    try:
        tags = json.loads(tags_json)
        return tags if isinstance(tags, list) else []
    except (json.JSONDecodeError, TypeError):
        return []
