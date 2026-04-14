"""Load and validate an ExpertPack from disk."""

from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path

import yaml

from .manifest import ManifestError, parse_manifest
from .models import GraphEdge, Pack, PackFile, PackGraph, Provenance

logger = logging.getLogger(__name__)

# Regex to match YAML frontmatter block
_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n?", re.DOTALL)

# Directories/files to skip when inventorying
_SKIP_DIRS = {".git", ".obsidian", ".ep-mcp", "node_modules", "__pycache__", ".venv"}
_SKIP_FILES = {"manifest.yaml", "_graph.yaml", "_access.json", "entities.json"}


class PackLoadError(Exception):
    """Raised when a pack cannot be loaded."""


def load_pack(pack_dir: str | Path, slug_override: str | None = None) -> Pack:
    """Load an ExpertPack from a directory.

    Args:
        pack_dir: Path to the ExpertPack root directory
        slug_override: Override the slug from manifest (for multi-pack routing)

    Returns:
        Fully loaded Pack object with all files parsed

    Raises:
        PackLoadError: If the directory is invalid or manifest is broken
    """
    pack_path = Path(pack_dir).resolve()

    if not pack_path.is_dir():
        raise PackLoadError(f"Pack directory does not exist: {pack_path}")

    manifest_path = pack_path / "manifest.yaml"
    if not manifest_path.exists():
        raise PackLoadError(f"No manifest.yaml found in {pack_path}")

    # Parse manifest
    try:
        manifest = parse_manifest(manifest_path)
    except ManifestError as e:
        raise PackLoadError(f"Failed to parse manifest: {e}") from e

    slug = slug_override or manifest.slug

    # Inventory and parse all .md files
    files = _inventory_files(pack_path, manifest)
    logger.info(
        "Loaded pack '%s': %d files, %d tokens total",
        slug,
        len(files),
        sum(f.size_tokens for f in files.values()),
    )

    # Load knowledge graph if present
    graph = _load_graph(pack_path)
    if graph:
        logger.info(
            "Loaded graph for '%s': %d nodes, %d edges",
            slug,
            len(graph.nodes),
            len(graph.edges),
        )

    # Build index path
    index_dir = pack_path / ".ep-mcp"
    index_path = str(index_dir / "index.db")

    # Resolve always-tier file paths
    always_tier = _resolve_always_tier(manifest.context.always, files, pack_path)

    return Pack(
        slug=slug,
        name=manifest.name,
        type=manifest.type,
        version=manifest.version,
        description=manifest.description,
        entry_point=manifest.entry_point,
        files=files,
        manifest=manifest,
        graph=graph,
        freshness=manifest.freshness,
        mcp_config=manifest.mcp,
        always_tier=always_tier,
        index_path=index_path,
        pack_dir=str(pack_path),
    )


def _inventory_files(pack_path: Path, manifest) -> dict[str, PackFile]:
    """Walk the pack directory and parse all .md files."""
    files: dict[str, PackFile] = {}
    always_set = set(manifest.context.always)
    on_demand_set = set(manifest.context.on_demand)

    for md_file in sorted(pack_path.rglob("*.md")):
        # Skip hidden/infrastructure directories
        rel_path = md_file.relative_to(pack_path)
        if any(part in _SKIP_DIRS for part in rel_path.parts):
            continue

        rel_str = str(rel_path)
        raw_content = md_file.read_text(encoding="utf-8", errors="replace")

        # Parse frontmatter
        frontmatter, content = _split_frontmatter(raw_content)

        # Extract title: frontmatter title > first H1 > filename
        title = (
            frontmatter.get("title")
            or _extract_h1(content)
            or rel_path.stem.replace("-", " ").replace("_", " ").title()
        )

        # Provenance metadata
        provenance = Provenance(
            id=frontmatter.get("id"),
            content_hash=frontmatter.get("content_hash"),
            verified_at=str(frontmatter["verified_at"]) if "verified_at" in frontmatter else None,
            verified_by=frontmatter.get("verified_by"),
        )

        # Determine retrieval strategy from context tiers
        if rel_str in always_set:
            retrieval_strategy = "always"
        elif rel_str in on_demand_set:
            retrieval_strategy = "on_demand"
        else:
            retrieval_strategy = frontmatter.get("retrieval_strategy", "standard")

        # Compute content hash (of stripped content) for index staleness checks
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()

        # Approximate token count
        size_tokens = _estimate_tokens(content)

        files[rel_str] = PackFile(
            path=rel_str,
            title=title,
            type=frontmatter.get("type"),
            tags=_ensure_list(frontmatter.get("tags", [])),
            provenance=provenance,
            retrieval_strategy=retrieval_strategy,
            content=content,
            raw_content=raw_content,
            size_tokens=size_tokens,
        )

        # Override content_hash on provenance with our computed one
        # (the frontmatter hash may be stale; the computed one is authoritative for indexing)
        files[rel_str].provenance.content_hash = content_hash

    return files


def _split_frontmatter(raw: str) -> tuple[dict, str]:
    """Split YAML frontmatter from markdown content.

    Returns:
        (frontmatter_dict, content_without_frontmatter)
    """
    match = _FRONTMATTER_RE.match(raw)
    if not match:
        return {}, raw

    try:
        fm = yaml.safe_load(match.group(1))
        if not isinstance(fm, dict):
            return {}, raw
    except yaml.YAMLError:
        return {}, raw

    content = raw[match.end():]
    return fm, content


def _extract_h1(content: str) -> str | None:
    """Extract the first H1 heading from markdown content."""
    for line in content.split("\n"):
        line = line.strip()
        if line.startswith("# ") and not line.startswith("##"):
            return line[2:].strip()
    return None


def _estimate_tokens(text: str) -> int:
    """Fast token count approximation: word count × 1.3."""
    words = len(text.split())
    return int(words * 1.3)


def _ensure_list(value: object) -> list[str]:
    """Ensure a value is a list of strings."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(v) for v in value]
    return []


def _resolve_always_tier(always_paths: list[str], files: dict, pack_path: Path) -> list[str]:
    """Resolve always-tier path declarations to actual file paths.

    Supports both direct file paths and directory paths (trailing /).
    Returns sorted list of resolved file paths present in the pack.
    """
    resolved = []
    for declared in always_paths:
        if declared.endswith("/"):
            # Directory glob — include all files under this directory
            prefix = declared.rstrip("/")
            for file_path in files:
                if file_path.startswith(prefix + "/"):
                    resolved.append(file_path)
        else:
            if declared in files:
                resolved.append(declared)
            else:
                logger.debug("always-tier path not found in pack: %s", declared)
    return sorted(set(resolved))


def _load_graph(pack_path: Path) -> PackGraph | None:
    """Load _graph.yaml if present."""
    graph_path = pack_path / "_graph.yaml"
    if not graph_path.exists():
        return None

    try:
        raw = yaml.safe_load(graph_path.read_text(encoding="utf-8"))
    except yaml.YAMLError:
        logger.warning("Failed to parse _graph.yaml, skipping graph")
        return None

    if not isinstance(raw, dict):
        return None

    nodes = raw.get("nodes", [])
    if not isinstance(nodes, list):
        nodes = []

    edges_raw = raw.get("edges", [])
    edges = []
    if isinstance(edges_raw, list):
        for e in edges_raw:
            if isinstance(e, dict) and "source" in e and "target" in e:
                edges.append(
                    GraphEdge(
                        source=e["source"],
                        target=e["target"],
                        kind=e.get("kind", "wikilink"),
                        weight=float(e.get("weight", 1.0)),
                    )
                )

    return PackGraph(nodes=[str(n) for n in nodes], edges=edges)
