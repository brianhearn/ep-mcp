"""ep_list_topics MCP tool implementation."""

from __future__ import annotations

import logging

from ..pack.models import Pack

logger = logging.getLogger(__name__)


def ep_list_topics(pack: Pack, type: str | None = None) -> dict:
    """List available topics and content structure in the ExpertPack.

    Args:
        pack: Loaded Pack object
        type: Filter by content type. If omitted, returns all types.

    Returns:
        Dict with pack metadata and grouped file listing
    """
    # Group files by type
    topics: dict[str, list[dict]] = {}
    for path, file in sorted(pack.files.items()):
        file_type = file.type or "uncategorized"

        if type and file_type != type:
            continue

        if file_type not in topics:
            topics[file_type] = []

        topics[file_type].append({
            "path": path,
            "title": file.title,
        })

    # Pack metadata
    pack_info = {
        "name": pack.name,
        "slug": pack.slug,
        "type": pack.type,
        "version": pack.version,
        "description": pack.description,
        "file_count": len(pack.files),
    }

    if pack.freshness:
        pack_info["freshness"] = {
            "coverage_pct": pack.freshness.coverage_pct,
            "last_full_review": pack.freshness.last_full_review,
        }

    result = {
        "pack": pack_info,
        "topics": topics,
    }
    total_topics = sum(len(v) for v in topics.values())
    logger.info(
        "ep_list_topics | pack=%s type_filter=%s types=%d files=%d",
        pack.slug, type, len(topics), total_topics,
    )
    return result
