"""ep_search MCP tool implementation."""

from __future__ import annotations

import logging
import time

from ..retrieval.engine import RetrievalEngine
from ..retrieval.models import SearchRequest, SearchResult

logger = logging.getLogger(__name__)


async def ep_search(
    engine: RetrievalEngine,
    query: str,
    type: str | None = None,
    tags: list[str] | None = None,
    max_results: int = 10,
) -> list[dict]:
    """Search the ExpertPack for relevant domain expertise.

    Args:
        engine: Configured RetrievalEngine for the target pack
        query: Natural language search query
        type: Filter by content type (concept, workflow, reference, etc.)
        tags: Filter by content tags (results match at least one)
        max_results: Maximum results to return (1-50, default 10)

    Returns:
        List of result dicts matching the MCP output schema

    Raises:
        ToolError: On retrieval failure (logged with context)
    """
    t0 = time.monotonic()
    request = SearchRequest(
        query=query,
        type=type,
        tags=tags,
        max_results=min(max(max_results, 1), 50),
    )

    try:
        results = await engine.search(request)
    except Exception:
        logger.exception(
            "ep_search failed | pack=%s query=%r type=%s tags=%s",
            engine.pack.slug, query, type, tags,
        )
        raise

    elapsed_ms = (time.monotonic() - t0) * 1000
    logger.info(
        "ep_search | pack=%s query=%r type=%s tags=%s "
        "results=%d top_score=%.4f elapsed=%.0fms",
        engine.pack.slug,
        query,
        type,
        tags,
        len(results),
        results[0].score if results else 0.0,
        elapsed_ms,
    )

    return [
        {
            "text": r.text,
            "source_file": r.source_file,
            "id": r.id,
            "content_hash": r.content_hash,
            "verified_at": r.verified_at,
            "score": r.score,
            "type": r.type,
            "tags": r.tags,
            "title": r.title,
        }
        for r in results
    ]
