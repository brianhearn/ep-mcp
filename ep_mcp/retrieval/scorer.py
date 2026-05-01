"""Score fusion, metadata boosting, MMR re-ranking, and adaptive threshold filtering."""

from __future__ import annotations

import json
import logging
import math

import numpy as np

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


def normalize_bm25_scores(results: list[dict], bm25_cap: float = 1.0) -> list[dict]:
    """Normalize FTS5 BM25 scores to 0-1 range using an absolute transform.

    FTS5 BM25 returns negative scores (more negative = more relevant).
    Uses the monotonic transform ``rank / (1 + rank)`` on the absolute value,
    which maps [0, ∞) → [0, 1) without any per-query normalization.

    This is more stable than min-max normalization because BM25 scores are
    comparable across queries — the same numeric value always means the same
    relative relevance, regardless of what other results are in the set.
    Min-max normalization inflates weak BM25 matches (when all scores are low,
    the worst result still gets 0.0 and the best gets 1.0), distorting fusion.

    BM25 saturation cap (``bm25_cap < 1.0``): clamps the normalized score so
    that keyword-dense files cannot dominate fusion by an unlimited margin.
    When a pack has files where the query term appears throughout the body
    (high term frequency → high absolute BM25), they can score 3x higher in
    BM25 than related files where the term appears only in the title/lead.
    The cap limits that gap without eliminating BM25 signal entirely.
    Default: 1.0 (no cap, backward-compatible).
    """
    for r in results:
        abs_score = abs(r["bm25_score"])
        raw_norm = abs_score / (1.0 + abs_score)
        r["bm25_norm"] = min(raw_norm, bm25_cap)
    return results


def fuse_scores(
    vec_results: list[dict],
    bm25_results: list[dict],
    vector_weight: float = 0.7,
    text_weight: float = 0.3,
    bm25_cap: float = 1.0,
) -> dict[int, float]:
    """Fuse vector and BM25 scores into a single hybrid score per chunk.

    ``bm25_cap`` limits the normalized BM25 score used in fusion.  When set
    below 1.0 (e.g. 0.7), files with extremely high term-frequency (keyword-
    dense) cannot dominate fusion over files that match the query semantically
    but have the term only in their title or lead.  Only affects the BM25
    contribution; vector scores are unaffected.

    Returns {chunk_id: hybrid_score}.
    """
    scores: dict[int, float] = {}

    # Accumulate vector scores
    for r in vec_results:
        cid = r["chunk_id"]
        scores[cid] = scores.get(cid, 0.0) + vector_weight * r.get("vec_score", 0.0)

    # Accumulate BM25 scores (cap applied during normalization, enforced here too)
    for r in bm25_results:
        cid = r["chunk_id"]
        bm25_contribution = min(r.get("bm25_norm", 0.0), bm25_cap)
        scores[cid] = scores.get(cid, 0.0) + text_weight * bm25_contribution

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

    Uses MMR to select and order results for diversity, but preserves the
    original relevance scores in the output.  This prevents the MMR diversity
    penalty from collapsing scores when input relevance scores have low
    variance (common for ExpertPack-style packs where many chunks are
    semantically similar).

    Selection criterion per step:
        MMR(d) = λ × Rel(d) - (1-λ) × max(CosSim(d, already_selected))

    The chunk with the highest MMR is picked next, but its *output* score
    is the original ``Rel(d)`` — not the MMR value.
    """
    if not scored_chunks:
        return []

    if not embeddings or lambda_param >= 1.0:
        # No embeddings available or pure relevance — just truncate
        return scored_chunks[:k]

    # Keep only candidates with vectors; append any vectorless candidates after MMR.
    candidate_ids: list[int] = []
    relevance_scores: list[float] = []
    vector_rows: list[list[float]] = []
    vectorless: list[tuple[int, float]] = []
    for cid, score in scored_chunks:
        emb = embeddings.get(cid)
        if emb is None:
            vectorless.append((cid, score))
            continue
        candidate_ids.append(cid)
        relevance_scores.append(score)
        vector_rows.append(emb)

    if not candidate_ids:
        return scored_chunks[:k]

    # Vectorized implementation. The old loop recomputed Python generator-based
    # dot products and norms for every candidate/selected pair, which made MMR a
    # multi-second bottleneck on 3072-d embeddings. Normalize once, then use BLAS
    # matrix-vector products to update max similarity to selected candidates.
    vectors = np.asarray(vector_rows, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    vectors = vectors / norms
    relevance = np.asarray(relevance_scores, dtype=np.float32)
    max_similarity = np.zeros(len(candidate_ids), dtype=np.float32)
    remaining = np.ones(len(candidate_ids), dtype=bool)

    selected: list[tuple[int, float]] = []
    target_k = min(k, len(candidate_ids))
    for _ in range(target_k):
        mmr_scores = (lambda_param * relevance) - ((1.0 - lambda_param) * max_similarity)
        mmr_scores[~remaining] = -np.inf
        best_idx = int(np.argmax(mmr_scores))
        if not remaining[best_idx]:
            break

        cid = candidate_ids[best_idx]
        selected.append((cid, relevance_scores[best_idx]))
        remaining[best_idx] = False

        if len(selected) >= target_k:
            break

        similarities = vectors @ vectors[best_idx]
        max_similarity = np.maximum(max_similarity, similarities)

    # Preserve previous behavior for rare vectorless candidates: they can still
    # appear after vector-ranked selections if there is room.
    if len(selected) < k and vectorless:
        selected.extend(vectorless[: k - len(selected)])

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
    anchor_score: float | None = None,
) -> dict[int, float]:
    """Filter scores using adaptive ratio-based thresholds.

    Two-step filter:
    1. Activation floor — if the best score is below this, return empty.
       The query is too far from anything in the pack.
    2. Score ratio — keep results within `score_ratio` of the *anchor* score,
       with `absolute_floor` as a hard minimum.

    The anchor score defaults to the best fused score when ``anchor_score``
    is None.  Pass the best *vector-only* score as ``anchor_score`` in
    hybrid pipelines so that a BM25 keyword spike does not inflate the
    cutoff and kill genuine vector-matched results.

    Args:
        scores: {chunk_id: fused_score} dict
        activation_floor: Minimum best-score to return any results
        score_ratio: Keep results within this fraction of anchor score
        absolute_floor: Never filter below this regardless of ratio
        anchor_score: Explicit anchor for ratio computation (default: best fused score)

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

    # Use explicit anchor (e.g. vector-only best) when provided,
    # otherwise fall back to best fused score.
    anchor = anchor_score if anchor_score is not None else best_score
    ratio_cutoff = anchor * score_ratio
    effective_min = max(ratio_cutoff, absolute_floor)

    filtered = {cid: s for cid, s in scores.items() if s >= effective_min}

    logger.info(
        "adaptive_threshold | best_fused=%.4f anchor=%.4f ratio_cutoff=%.4f effective_min=%.4f "
        "kept=%d/%d",
        best_score, anchor, ratio_cutoff, effective_min, len(filtered), len(scores),
    )

    return filtered


def score_bm25_fallback(
    bm25_results: list[dict],
    chunks: dict[int, dict],
    query: str,
    path_boost: float = 0.18,
    length_boost_max: float = 0.18,
    length_boost_target: int = 300,
) -> dict[int, float]:
    """Richer lexical scoring for BM25-only fallback mode (no embeddings available).

    Augments the normalized BM25 score with:
    - **Term coverage:** fraction of query content-tokens present in the chunk.
      Rewards chunks that answer more of what was asked.
    - **Token density:** coverage / chunk_token_count. Rewards focused chunks
      over long documents where the matching terms are buried.
    - **Path boost:** ``+path_boost`` per query token found in the file path.
      Helps exact-match files (e.g. query token ``"auto-build"`` matching
      ``auto-build.md``) surface to the top.
    - **Length boost:** small linear bonus for longer chunks, capped at
      ``length_boost_max``. Prevents very short stubs from winning purely
      on density.

    Inspired by OpenClaw's ``scoreFallbackKeywordResult()``.

    Args:
        bm25_results: Output of ``normalize_bm25_scores()``.
        chunks: {chunk_id: chunk_row} from SQLite.
        query: Original natural language query string.
        path_boost: Additive boost per query token found in the file path.
        length_boost_max: Maximum additive length bonus.
        length_boost_target: Chunk length (chars) at which the length bonus maxes out.

    Returns:
        {chunk_id: combined_score} dict.
    """
    # Tokenize query: lowercase, strip short/stop tokens (mirrors _sanitize_fts5_query logic)
    import re
    raw_tokens = re.sub(r'[^\w\s]', ' ', query).lower().split()
    query_tokens = [
        t for t in raw_tokens
        if len(t) >= 3 and t not in _STOPWORDS
    ] or raw_tokens[:5]  # fallback: keep first 5 tokens

    scores: dict[int, float] = {}
    for r in bm25_results:
        cid = r["chunk_id"]
        bm25_norm = r.get("bm25_norm", 0.0)
        chunk = chunks.get(cid)
        if not chunk:
            scores[cid] = bm25_norm
            continue

        content_lower = chunk.get("content", "").lower()
        file_path_lower = chunk.get("file_path", "").lower()
        content_tokens = content_lower.split()
        total_tokens = max(len(content_tokens), 1)

        # Term coverage: how many query tokens appear in the chunk content?
        matched_in_content = sum(1 for t in query_tokens if t in content_lower)
        coverage = matched_in_content / max(len(query_tokens), 1)

        # Token density: matched tokens relative to chunk size
        density = coverage / total_tokens * 100  # scale: 0 → ~1 for typical chunks
        density = min(density, 1.0)

        # Path boost: query tokens in file path
        matched_in_path = sum(1 for t in query_tokens if t in file_path_lower)
        p_boost = path_boost * matched_in_path

        # Length boost: small reward for longer (more complete) chunks
        content_len = len(chunk.get("content", ""))
        l_boost = length_boost_max * min(content_len / length_boost_target, 1.0)

        scores[cid] = bm25_norm + coverage * 0.4 + density * 0.2 + p_boost + l_boost

    logger.debug(
        "score_bm25_fallback | scored %d chunks from query '%s' (%d content tokens)",
        len(scores), query[:60], len(query_tokens),
    )
    return scores


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
