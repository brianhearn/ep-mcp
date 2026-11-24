"""ep_list_packs and ep_describe_pack MCP tools (discoverability, M3)."""

from __future__ import annotations

from collections import Counter
from typing import Any

from ..pack.models import Pack


def ep_describe_pack(pack: Pack, chunk_count: int | None = None) -> dict[str, Any]:
    """Return a rich structured description of a single ExpertPack."""
    type_counts = Counter(f.type or "untyped" for f in pack.files.values())
    indexable_count = sum(1 for f in pack.files.values() if f.is_indexable)

    boundary = None
    if pack.manifest.authority_boundary is not None:
        ab = pack.manifest.authority_boundary
        boundary = {
            "in_scope": ab.in_scope,
            "out_of_scope": list(ab.out_of_scope),
            "refuse_when": list(ab.refuse_when),
            "no_source_no_claim": ab.no_source_no_claim,
        }

    freshness = None
    if pack.freshness is not None:
        freshness = {
            "refresh_cycle": pack.freshness.refresh_cycle,
            "last_full_review": pack.freshness.last_full_review,
            "coverage_pct": pack.freshness.coverage_pct,
        }

    return {
        "slug": pack.slug,
        "name": pack.name,
        "type": pack.type,
        "version": pack.version,
        "schema_version": pack.manifest.schema_version,
        "description": pack.description,
        "entry_point": pack.entry_point,
        "license": pack.manifest.license,
        "canonical_url": pack.manifest.canonical_url,
        "applies_to": pack.manifest.applies_to,
        "integrity_status": pack.integrity_status,
        "digest_verified": pack.digest_verified,
        "security_blocked_files": list(pack.security_blocked_files),
        "file_count": len(pack.files),
        "indexable_file_count": indexable_count,
        "chunk_count": chunk_count,
        "content_types": dict(sorted(type_counts.items())),
        "has_graph": pack.graph is not None,
        "freshness": freshness,
        "authority_boundary": boundary,
    }


def ep_list_packs(packs: dict[str, tuple[Pack, int | None]]) -> dict[str, Any]:
    """Return compact metadata for all configured packs.

    Args:
        packs: Mapping of slug -> (Pack, chunk_count_or_None)
    """
    items = []
    for slug in sorted(packs.keys()):
        pack, chunk_count = packs[slug]
        items.append({
            "slug": slug,
            "name": pack.name,
            "type": pack.type,
            "version": pack.version,
            "description": pack.description,
            "license": pack.manifest.license,
            "applies_to": pack.manifest.applies_to,
            "integrity_status": pack.integrity_status,
            "digest_verified": pack.digest_verified,
            "file_count": len(pack.files),
            "chunk_count": chunk_count,
        })
    return {
        "count": len(items),
        "packs": items,
    }
