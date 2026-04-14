"""MCP Resource registration for ExpertPack packs.

Exposes three categories of resources:
  1. Always-tier files — foundational context the agent reads at registration
  2. Additional declared resources — files from mcp.resources.additional
  3. On-demand file access — any pack file by URI template
"""

from __future__ import annotations

import json
import logging

from ..pack.models import Pack

logger = logging.getLogger(__name__)


def register_resources(mcp, pack: Pack) -> None:
    """Register MCP Resources for a pack.

    Args:
        mcp: FastMCP server instance
        pack: Loaded Pack object
    """
    slug = pack.slug

    # --- 1. Manifest (always exposed) ---
    @mcp.resource(f"ep://{slug}/manifest")
    async def get_manifest() -> str:
        """Pack identity, version, coverage, and freshness metadata."""
        return json.dumps(pack.manifest.raw, indent=2, default=str)

    # --- 2. Overview / entry point (always exposed) ---
    entry = pack.entry_point
    if entry in pack.files:
        @mcp.resource(f"ep://{slug}/overview")
        async def get_overview() -> str:
            """Pack overview — what this pack covers and how to navigate it."""
            return pack.files[entry].raw_content
    else:
        logger.warning(
            "Pack '%s': entry_point '%s' not found in files — overview resource skipped",
            slug, entry,
        )

    # --- 3. Always-tier files ---
    if pack.mcp_config.resources.include_always_tier:
        registered_always = 0
        for path in pack.always_tier:
            if path not in pack.files:
                continue
            if path == entry:
                continue  # already registered as /overview

            # Capture path in closure
            def _make_always_handler(p: str):
                async def get_always_file() -> str:
                    f"""Always-tier file: {p}"""
                    return pack.files[p].raw_content
                get_always_file.__doc__ = f"Always-tier context file: {p}"
                return get_always_file

            uri = f"ep://{slug}/always/{path}"
            mcp.resource(uri)(_make_always_handler(path))
            registered_always += 1

        logger.info(
            "Pack '%s': registered %d always-tier resources",
            slug, registered_always,
        )

    # --- 4. Additional declared resources ---
    for path in pack.mcp_config.resources.additional:
        if path not in pack.files:
            logger.warning(
                "Pack '%s': mcp.resources.additional path '%s' not found — skipped",
                slug, path,
            )
            continue
        if path == entry:
            continue  # already registered as /overview

        def _make_additional_handler(p: str):
            async def get_additional_file() -> str:
                return pack.files[p].raw_content
            get_additional_file.__doc__ = f"Declared additional resource: {p}"
            return get_additional_file

        uri = f"ep://{slug}/file/{path}"
        mcp.resource(uri)(_make_additional_handler(path))
        logger.debug("Pack '%s': registered additional resource: %s", slug, path)

    # --- 5. Legacy file listing (kept for backward compatibility) ---
    @mcp.resource(f"ep://{slug}/files")
    async def get_file_listing() -> str:
        """Full file listing with metadata for all pack files."""
        files = []
        for path, f in sorted(pack.files.items()):
            files.append({
                "path": path,
                "title": f.title,
                "type": f.type,
                "tags": f.tags,
                "tokens": f.size_tokens,
                "has_provenance": f.provenance.id is not None,
                "in_always_tier": path in pack.always_tier,
            })
        return json.dumps(files, indent=2)
