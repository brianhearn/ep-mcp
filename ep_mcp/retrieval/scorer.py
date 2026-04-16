"""Score fusion, metadata boosting, MMR re-ranking, and adaptive threshold filtering."""

from __future__ import annotations

import json
import logging
import math

logger = logging.getLogger(__name__)


def normalize_vector_scores(results: list[dict], max_distance: float = 2.0) -> list[dict]:
    """Normalize sqlite-vec distances to 0-1 similarity scores.

    sqlite-vec returns L2 distances (lower = more similar).
    Convert to similarity: score = 1 - (distance / max_distance), clamped to [0, 1].
    """
    for r in results:
        dist = r["distance"]
        r["vec_score"] = max(0.0, min(1.0, 1.0 - (dist / max_distance)))
    return results


def normalize_bm25_scores(results: list[dict]) -> list[dict]:
    """Normalize FTS5 BM25 scores to 0-1 range.

    FTS5 BM25 returns negative scores (more negative = more relevant).
    Normalize across the result set: best = 1.0, worst = 0.0.
    """
    if not results:
        return results

    scores = [abs(r["bm25_score"]) for r in results]
    max_score = max(scores)
    min_score = min(scores)
    score_range = max_score - min_score

    for r in results:
        abs_score = abs(r["bm25_score"])
        if score_range > 0:
            r["bm25_norm"] = (abs_score - min_score) / score_range
        else:
            r["bm25_norm"] = 1.0  # All same score
    return results


def fuse_scores(
    vec_results: list[dict],
    bm25_results: list[dict],
    vector_weight: float = 0.7,
    text_weight: float = 0.3,
) -> dict[int, float]:
    """Fuse vector and BM25 scores into a single hybrid score per chunk.

    Returns {chunk_id: hybrid_score}.
    """
    scores: dict[int, float] = {}

    # Accumulate vector scores
    for r in vec_results:
        cid = r["chunk_id"]
        scores[cid] = scores.get(cid, 0.0) + vector_weight * r.get("vec_score", 0.0)

    # Accumulate BM25 scores
    for r in bm25_results:
        cid = r["chunk_id"]
        scores[cid] = scores.get(cid, 0.0) + text_weight * r.get("bm25_norm", 0.0)

    return scores


def apply_metadata_boosts(
    scores: dict[int, float],
    chunks: dict[int, dict],
    type_filter: str | None = None,
    tag_filter: list[str] | None = None,
    always_files: set[str] | None = None,
    type_match_boost: float = 0.05,
    tag_match_boost: float = 0.03,
    always_tier_boost: float = 0.02,
) -> dict[int, float]:
    """Apply metadata-based score boosts.

    Boosts applied:
    - type_match_boost: when chunk type matches the filter
    - tag_match_boost: per matching tag
    - always_tier_boost: for files in context.always tier
    """
    for cid, score in scores.items():
        chunk = chunks.get(cid)
        if not chunk:
            continue

        # Type match boost
        if type_filter and chunk.get("type") == type_filter:
            score += type_match_boost

        # Tag match boost
        if tag_filter:
            chunk_tags = _parse_tags(chunk.get("tags", "[]"))
            matching = len(set(tag_filter) & set(chunk_tags))
            score += tag_match_boost * matching

        # Always-tier boost
        if always_files and chunk.get("file_path") in always_files:
            score += always_tier_boost

        scores[cid] = score

    return scores


def mmr_rerank(
    scored_chunks: list[tuple[int, float]],
    embeddings: dict[int, list[float]],
    lambda_param: float = 0.7,
    k: int = 10,
) -> list[tuple[int, float]]:
    """Maximal Marginal Relevance re-ranking.

    Balances relevance vs diversity:
    MMR(d) = λ × Sim(d, query) - (1-λ) × max(Sim(d, selected))

    Args:
        scored_chunks: [(chunk_id, relevance_score), ...] sorted by score desc
        embeddings: {chunk_id: embedding_vector} for cosine similarity
        lambda_param: Balance factor (1.0 = pure relevance, 0.0 = pure diversity)
        k: Number of results to return

    Returns:
        Re-ranked [(chunk_id, mmr_score), ...] of length min(k, len(scored_chunks))
    """
    if not scored_chunks:
        return []

    if not embeddings or lambda_param >= 1.0:
        # No embeddings available or pure relevance — just truncate
        return scored_chunks[:k]

    remaining = dict(scored_chunks)
    selected: list[tuple[int, float]] = []

    while remaining and len(selected) < k:
        best_id = None
        best_mmr = -float("inf")

        for cid, relevance in remaining.items():
            # Relevance component
            rel_component = lambda_param * relevance

            # Diversity component: max similarity to any already-selected chunk
            div_component = 0.0
            if selected and cid in embeddings:
                for sel_id, _ in selected:
                    if sel_id in embeddings:
                        sim = _cosine_similarity(embeddings[cid], embeddings[sel_id])
                        div_component = max(div_component, sim)

            mmr = rel_component - (1 - lambda_param) * div_component

            if mmr > best_mmr:
                best_mmr = mmr
                best_id = cid

        if best_id is not None:
            selected.append((best_id, best_mmr))
            del remaining[best_id]

    return selected


def apply_length_penalty(
    scores: dict[int, float],
    chunks: dict[int, dict],
    short_threshold: int = 80,
    short_penalty: float = 0.15,
) -> dict[int, float]:
    """Penalize very short chunks that are unlikely to contain a complete answer.

    Chunks below `short_threshold` characters are probably stubs, headings, or
    navigation artefacts. Apply a multiplicative penalty so they can still win
    on relevance but at a discounted rate.

    Args:
        scores: {chunk_id: score} dict (modified in-place and returned)
        chunks: {chunk_id: chunk_row} from SQLite — must include 'content'
        short_threshold: Character count below which a chunk is considered short
        short_penalty: Multiplicative penalty applied to short chunks (0–1)

    Returns:
        Updated {chunk_id: score} dict
    """
    penalized = 0
    for cid, score in scores.items():
        chunk = chunks.get(cid)
        if not chunk:
            continue
        content = chunk.get("content", "")
        if len(content) < short_threshold:
            scores[cid] = score * (1.0 - short_penalty)
            penalized += 1

    if penalized:
        logger.debug(
            "length_penalty | penalized=%d/%d chunks (threshold=%d chars, penalty=%.2f)",
            penalized, len(scores), short_threshold, short_penalty,
        )

    return scores


def apply_adaptive_threshold(
    scores: dict[int, float],
    activation_floor: float = 0.15,
    score_ratio: float = 0.55,
    absolute_floor: float = 0.10,
) -> dict[int, float]:
    """Filter scores using adaptive ratio-based thresholds.

    Two-step filter:
    1. Activation floor — if the best score is below this, return empty.
       The query is too far from anything in the pack.
    2. Score ratio — keep results within `score_ratio` of the best score,
       with `absolute_floor` as a hard minimum.

    This adapts to pack size, embedding model, and query specificity
    instead of relying on a fixed absolute cutoff.

    Args:
        scores: {chunk_id: fused_score} dict
        activation_floor: Minimum best-score to return any results
        score_ratio: Keep results within this fraction of best score
        absolute_floor: Never filter below this regardless of ratio

    Returns:
        Filtered {chunk_id: score} dict (may be empty)
    """
    if not scores:
        return {}

    best_score = max(scores.values())

    if best_score < activation_floor:
        logger.debug(
            "adaptive_threshold | best_score=%.4f < activation_floor=%.2f → empty",
            best_score, activation_floor,
        )
        return {}

    ratio_cutoff = best_score * score_ratio
    effective_min = max(ratio_cutoff, absolute_floor)

    filtered = {cid: s for cid, s in scores.items() if s >= effective_min}

    logger.debug(
        "adaptive_threshold | best=%.4f ratio_cutoff=%.4f effective_min=%.4f "
        "kept=%d/%d",
        best_score, ratio_cutoff, effective_min, len(filtered), len(scores),
    )

    return filtered


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if len(a) != len(b):
        return 0.0

    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))

    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _parse_tags(tags_json: str) -> list[str]:
    """Parse tags from JSON string stored in SQLite."""
    try:
        tags = json.loads(tags_json)
        return tags if isinstance(tags, list) else []
    except (json.JSONDecodeError, TypeError):
        return []
