"""Parse and validate ExpertPack manifest.yaml files."""

from __future__ import annotations

from pathlib import Path

import yaml

from .models import ContextTiers, FreshnessMetadata, Manifest, MCPConfig, MCPPromptDeclaration, MCPResourcesConfig


class ManifestError(Exception):
    """Raised when manifest.yaml is invalid or missing required fields."""


def parse_manifest(manifest_path: Path) -> Manifest:
    """Parse a manifest.yaml file into a Manifest model.

    Args:
        manifest_path: Path to manifest.yaml

    Returns:
        Parsed Manifest object

    Raises:
        ManifestError: If file is missing, unreadable, or invalid
    """
    if not manifest_path.exists():
        raise ManifestError(f"manifest.yaml not found: {manifest_path}")

    try:
        raw = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as e:
        raise ManifestError(f"Invalid YAML in {manifest_path}: {e}") from e

    if not isinstance(raw, dict):
        raise ManifestError(f"manifest.yaml must be a YAML mapping, got {type(raw).__name__}")

    # Required fields
    for field in ("slug", "name", "type"):
        if field not in raw:
            raise ManifestError(f"Missing required field '{field}' in manifest.yaml")

    pack_type = raw["type"]
    if pack_type not in ("person", "product", "process", "composite"):
        raise ManifestError(
            f"Invalid pack type '{pack_type}' — must be person, product, process, or composite"
        )

    # Parse context tiers
    context_raw = raw.get("context", {})
    context = ContextTiers(
        always=_ensure_list(context_raw.get("always", [])),
        on_demand=_ensure_list(context_raw.get("on_demand", [])),
    )

    # Parse freshness metadata
    freshness_raw = raw.get("freshness", {})
    freshness = FreshnessMetadata(
        refresh_cycle=freshness_raw.get("refresh_cycle"),
        coverage_pct=freshness_raw.get("coverage_pct"),
        last_full_review=freshness_raw.get("last_full_review"),
        decay_rate=freshness_raw.get("decay_rate"),
    )

    # Parse mcp block
    mcp_raw = raw.get("mcp", {})
    mcp_config = _parse_mcp_config(mcp_raw)

    return Manifest(
        slug=raw["slug"],
        name=raw["name"],
        type=pack_type,
        version=str(raw.get("version", "1.0.0")),
        description=raw.get("description", ""),
        entry_point=raw.get("entry_point", "overview.md"),
        schema_version=str(raw.get("schema_version", "")),
        context=context,
        freshness=freshness,
        mcp=mcp_config,
        raw=raw,
    )


def _parse_mcp_config(mcp_raw: dict) -> MCPConfig:
    """Parse the mcp block from manifest.yaml into an MCPConfig model."""
    if not isinstance(mcp_raw, dict):
        return MCPConfig()

    # Parse instructions
    instructions = mcp_raw.get("instructions")
    if instructions is not None:
        instructions = str(instructions).strip()
        if len(instructions) > 500:
            import logging
            logging.getLogger(__name__).warning(
                "W-MCP-03: mcp.instructions exceeds 500 characters (%d chars) — consider trimming",
                len(instructions),
            )

    # Parse prompts
    prompts_raw = mcp_raw.get("prompts", [])
    prompts = []
    if isinstance(prompts_raw, list):
        for p in prompts_raw:
            if not isinstance(p, dict):
                continue
            if not p.get("name") or not p.get("source"):
                continue
            prompts.append(MCPPromptDeclaration(
                name=str(p["name"]),
                description=str(p.get("description", "")),
                source=str(p["source"]),
            ))

    # Parse resources
    resources_raw = mcp_raw.get("resources", {})
    if not isinstance(resources_raw, dict):
        resources_raw = {}
    resources = MCPResourcesConfig(
        include_always_tier=bool(resources_raw.get("include_always_tier", True)),
        additional=_ensure_list(resources_raw.get("additional", [])),
    )

    return MCPConfig(
        instructions=instructions,
        prompts=prompts,
        resources=resources,
    )


def _ensure_list(value: object) -> list[str]:
    """Ensure a value is a list of strings."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(v) for v in value]
    return []
