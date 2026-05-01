"""Cross-encoder reranker — second-pass precision layer.

Uses a lightweight cross-encoder model (default: ms-marco-MiniLM-L-6-v2)
to rescore the top-N candidates from the hybrid retrieval pipeline.

The reranker sees the full (query, document) pair together, giving it
significantly better precision than the bi-encoder vector similarity used
in the first pass. This is especially effective for domain-specific packs
where many files are topically similar (high inter-chunk cosine similarity)
and MMR displacement causes relevant files to rank 9-15 instead of top 8.

Design principles:
- Lazy model load on first request (avoids startup delay)
- Original hybrid scores are preserved; reranker scores are used for
  ordering ONLY (the reported score stays the hybrid fused score)
- Configurable candidate_pool_size: rerank top-N before final slice
- Graceful degradation: if the model fails to load, log a warning and
  return candidates unmodified
"""

from __future__ import annotations

import logging
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class Reranker:
    """Lazy-loading cross-encoder reranker.

    Args:
        model_name: HuggingFace model id for the cross-encoder.
        candidate_pool_size: How many candidates to rerank (top-N from hybrid).
            Reranking is applied to this many results before the final slice.
        enabled: If False, rerank() is a no-op (passthrough).
    """

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
        candidate_pool_size: int = 20,
        enabled: bool = True,
        max_chars: int = 512,
        batch_size: int = 32,
    ):
        self._model_name = model_name
        self._candidate_pool_size = candidate_pool_size
        self._enabled = enabled
        self._max_chars = max_chars
        self._batch_size = batch_size
        self._model = None  # lazy load
        self._load_failed = False

    def _load_model(self) -> bool:
        """Load the CrossEncoder model. Returns True if successful."""
        if self._model is not None:
            return True
        if self._load_failed:
            return False
        try:
            from sentence_transformers import CrossEncoder  # type: ignore
            logger.info("Reranker: loading model '%s'...", self._model_name)
            self._model = CrossEncoder(self._model_name)
            logger.info("Reranker: model loaded OK")
            return True
        except Exception as exc:
            logger.warning(
                "Reranker: failed to load model '%s' (%s) — reranking disabled",
                self._model_name, exc,
            )
            self._load_failed = True
            return False

    def rerank(self, query: str, results: list) -> list:
        """Rerank results using cross-encoder scoring.

        Takes the top `candidate_pool_size` results, scores each (query, text)
        pair with the cross-encoder, re-sorts by cross-encoder score, then
        returns the full list (reranked portion first, remainder appended).

        The original hybrid score is preserved on each SearchResult; only
        ordering changes. This keeps the reported score meaningful for callers
        while improving precision through reranking.

        Args:
            query: The original search query string.
            results: List of SearchResult objects from the retrieval pipeline.

        Returns:
            Reranked list (same length, different order for top candidates).
        """
        if not self._enabled or not results:
            return results

        if not self._load_model():
            return results  # graceful degradation

        pool_size = min(self._candidate_pool_size, len(results))
        pool = results[:pool_size]
        remainder = results[pool_size:]

        try:
            pairs = [(query, self._truncate_text(r.text)) for r in pool]
            ce_scores = self._model.predict(
                pairs,
                batch_size=self._batch_size,
                show_progress_bar=False,
            )

            # Zip candidates with their cross-encoder scores and re-sort
            scored = sorted(
                zip(ce_scores, pool),
                key=lambda x: x[0],
                reverse=True,
            )
            reranked = [r for _, r in scored]

            logger.info(
                "Reranker: reranked %d candidates (pool=%d, remainder=%d)",
                len(reranked), pool_size, len(remainder),
            )
            return reranked + remainder

        except Exception as exc:
            logger.warning("Reranker: scoring failed (%s) — returning original order", exc)
            return results

    def _truncate_text(self, text: str) -> str:
        """Bound document length before cross-encoder scoring.

        Cross-encoder cost scales with token count. The model's tokenizer caps
        long inputs anyway, but passing whole ExpertPack chunks makes tokenization
        and inference much slower. Keep the front matter/lead content where the
        answer signal usually lives, and trim at a word boundary when possible.
        """
        if self._max_chars <= 0 or len(text) <= self._max_chars:
            return text
        clipped = text[: self._max_chars]
        boundary = max(clipped.rfind("\n"), clipped.rfind(". "), clipped.rfind(" "))
        if boundary >= max(120, int(self._max_chars * 0.6)):
            clipped = clipped[: boundary + 1]
        return clipped
