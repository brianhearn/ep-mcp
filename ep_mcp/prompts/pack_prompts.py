"""MCP Prompt registration for ExpertPack packs.

Prompts deliver complete workflow methodology to connecting agents.
Each prompt maps to a type:workflow file in the pack.

Resolution order:
  1. Explicit mcp.prompts declarations in manifest.yaml (preferred)
  2. Auto-discovery: all files with type: workflow (fallback)
"""

from __future__ import annotations

import logging
import re

from ..pack.models import MCPPromptDeclaration, Pack

logger = logging.getLogger(__name__)


def register_prompts(mcp, pack: Pack) -> None:
    """Register MCP Prompts for a pack.

    Args:
        mcp: FastMCP server instance
        pack: Loaded Pack object
    """
    declarations = _resolve_prompt_declarations(pack)

    if not declarations:
        logger.info("Pack '%s': no prompts to register (no workflows found)", pack.slug)
        return

    registered = 0
    for decl in declarations:
        source_file = pack.files.get(decl.source)

        # Validate: source file must exist
        if source_file is None:
            logger.error(
                "E-MCP-01: Prompt '%s' source file not found | pack=%s source=%s — skipped",
                decl.name, pack.slug, decl.source,
            )
            continue

        # Validate: should be type:workflow
        if source_file.type != "workflow":
            logger.warning(
                "W-MCP-01: Prompt '%s' source file is type '%s', expected 'workflow' | pack=%s source=%s",
                decl.name, source_file.type, pack.slug, decl.source,
            )

        # Validate: should be retrieval_strategy:atomic
        if source_file.retrieval_strategy not in ("atomic", "always"):
            logger.warning(
                "W-MCP-02: Prompt '%s' source file retrieval_strategy is '%s', expected 'atomic' | pack=%s source=%s",
                decl.name, source_file.retrieval_strategy, pack.slug, decl.source,
            )

        # Register the prompt
        def _make_prompt_handler(content: str, name: str):
            async def get_prompt() -> str:
                """Return the full workflow content for this prompt."""
                return content
            get_prompt.__doc__ = f"Workflow prompt: {name}"
            return get_prompt

        try:
            mcp.prompt(name=decl.name, description=decl.description)(
                _make_prompt_handler(source_file.content, decl.name)
            )
            registered += 1
            logger.debug(
                "Registered prompt '%s' from '%s' | pack=%s",
                decl.name, decl.source, pack.slug,
            )
        except Exception:
            logger.exception(
                "Failed to register prompt '%s' | pack=%s", decl.name, pack.slug
            )

    logger.info("Pack '%s': registered %d prompts", pack.slug, registered)


def _resolve_prompt_declarations(pack: Pack) -> list[MCPPromptDeclaration]:
    """Return explicit manifest declarations, or auto-discover from workflow files.

    Explicit declarations in mcp.prompts take priority.
    Auto-discovery scans for all files with type: workflow.
    """
    if pack.mcp_config.prompts:
        logger.debug(
            "Pack '%s': using %d explicit prompt declarations from manifest",
            pack.slug, len(pack.mcp_config.prompts),
        )
        return pack.mcp_config.prompts

    # Auto-discovery fallback
    discovered: list[MCPPromptDeclaration] = []
    for path, f in pack.files.items():
        if f.type == "workflow":
            name = _filename_to_prompt_name(path)
            description = f.title or name.replace("_", " ").title()
            discovered.append(MCPPromptDeclaration(
                name=name,
                description=description,
                source=path,
            ))

    if discovered:
        logger.info(
            "Pack '%s': auto-discovered %d workflow prompts (no mcp.prompts in manifest)",
            pack.slug, len(discovered),
        )
    
    return sorted(discovered, key=lambda d: d.name)


def _filename_to_prompt_name(path: str) -> str:
    """Convert a file path to a snake_case prompt name.

    Examples:
        workflows/wf-build-territories.md  -> build_territories
        workflows/wf-analyze-balance.md    -> analyze_balance
        processes/proc-onboarding.md       -> onboarding
    """
    # Take just the filename stem
    stem = path.split("/")[-1]
    if stem.endswith(".md"):
        stem = stem[:-3]

    # Strip common file prefixes (wf-, proc-, workflow-, process-)
    stem = re.sub(r"^(wf|proc|workflow|process)-", "", stem)

    # kebab-case to snake_case
    return stem.replace("-", "_")
