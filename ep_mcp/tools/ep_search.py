"""ep_search MCP tool implementation."""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

from ..retrieval.engine import RetrievalEngine
from ..retrieval.models import SearchRequest, SearchResult

logger = logging.getLogger(__name__)


async def ep_search(
    engine: RetrievalEngine,
    query: str,
    type: str | None = None,
    tags: list[str] | None = None,
    max_results: int = 10,
    query_log_path: str | None = None,
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

    # Detect embed cache hit from cache wrapper (if applicable)
    embed_cached: bool | None = None
    provider = engine.provider
    if hasattr(provider, "last_cache_hit"):
        embed_cached = provider.last_cache_hit

    top_score = results[0].score if results else 0.0
    logger.info(
        "ep_search | pack=%s query=%r type=%s tags=%s "
        "results=%d top_score=%.4f elapsed=%.0fms embed_cached=%s",
        engine.pack.slug,
        query,
        type,
        tags,
        len(results),
        top_score,
        elapsed_ms,
        embed_cached,
    )

    # Structured query log (JSONL) — written when query_log_path is configured
    if query_log_path:
        _append_query_log(
            path=query_log_path,
            pack=engine.pack.slug,
            query=query,
            type=type,
            tags=tags,
            results=results,
            top_score=top_score,
            elapsed_ms=elapsed_ms,
            embed_cached=embed_cached,
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


def log_query(
    query_log_path: str,
    pack: str,
    query: str,
    results: list[SearchResult],
    elapsed_ms: float,
    embed_cached: bool | None,
    type: str | None = None,
    tags: list[str] | None = None,
) -> None:
    """Public helper: write one JSONL record for any search path (REST or MCP tool)."""
    _append_query_log(
        path=query_log_path,
        pack=pack,
        query=query,
        type=type,
        tags=tags,
        results=results,
        top_score=results[0].score if results else 0.0,
        elapsed_ms=elapsed_ms,
        embed_cached=embed_cached,
    )


def _append_query_log(
    path: str,
    pack: str,
    query: str,
    type: str | None,
    tags: list[str] | None,
    results: list[SearchResult],
    top_score: float,
    elapsed_ms: float,
    embed_cached: bool | None,
) -> None:
    """Append a single JSONL record to the query log file.

    Each record contains:
      ts          ISO-8601 UTC timestamp
      pack        pack slug
      query       raw query string
      type        type filter (or null)
      tags        tag filter list (or null)
      result_count  number of results returned
      chunks      list of source_file paths for returned results
      scores      parallel list of scores for returned results
      top_score   score of the first result (0.0 when empty)
      elapsed_ms  total ep_search latency in milliseconds
      embed_cached  true/false/null (null when provider has no cache)
    """
    record = {
        "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "pack": pack,
        "query": query,
        "type": type,
        "tags": tags,
        "result_count": len(results),
        "chunks": [r.source_file for r in results],
        "scores": [r.score for r in results],
        "top_score": top_score,
        "elapsed_ms": round(elapsed_ms, 1),
        "embed_cached": embed_cached,
    }
    try:
        log_path = Path(path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
    except Exception:
        logger.exception("query_log write failed (path=%r)", path)
