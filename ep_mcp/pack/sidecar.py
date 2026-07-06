"""Load RFC-004 chunk metadata sidecars (*.chunks.yaml)."""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

_FRONTMATTER_RE = re.compile(r"^---\s*\n.*?\n---\s*\n?", re.DOTALL)


@dataclass
class SidecarChunk:
    """A single chunk boundary from a sidecar file."""

    chunk_id: str
    chunk_order: int
    section: str | None
    line_range: tuple[int, int]
    chunk_summary: str | None = None


@dataclass
class ChunkSidecar:
    """Parsed `<name>.chunks.yaml` sidecar."""

    schema_version: str
    source_id: str
    content_hash: str
    chunks: list[SidecarChunk]


def sidecar_path_for_md(pack_dir: Path, md_rel_path: str, frontmatter: dict | None = None) -> Path | None:
    """Resolve the sidecar path for a markdown file, if one exists."""
    pack_dir = Path(pack_dir)
    rel = Path(md_rel_path)
    declared = (frontmatter or {}).get("chunks_sidecar")
    if declared:
        candidate = pack_dir / declared
        if candidate.is_file():
            return candidate
    default = pack_dir / rel.with_suffix(".chunks.yaml")
    return default if default.is_file() else None


def load_sidecar(path: Path) -> ChunkSidecar | None:
    """Parse a chunk sidecar YAML file."""
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (yaml.YAMLError, OSError) as exc:
        logger.warning("Failed to parse chunk sidecar %s: %s", path, exc)
        return None

    if not isinstance(raw, dict):
        return None

    chunks_raw = raw.get("chunks", [])
    if not isinstance(chunks_raw, list):
        return None

    chunks: list[SidecarChunk] = []
    for item in chunks_raw:
        if not isinstance(item, dict):
            continue
        line_range = item.get("line_range")
        if not isinstance(line_range, (list, tuple)) or len(line_range) != 2:
            continue
        chunks.append(
            SidecarChunk(
                chunk_id=str(item.get("chunk_id", "")),
                chunk_order=int(item.get("chunk_order", len(chunks))),
                section=item.get("section"),
                line_range=(int(line_range[0]), int(line_range[1])),
                chunk_summary=item.get("chunk_summary") or None,
            )
        )

    if not chunks:
        return None

    chunks.sort(key=lambda c: c.chunk_order)
    return ChunkSidecar(
        schema_version=str(raw.get("schema_version", "1.0")),
        source_id=str(raw.get("source_id", "")),
        content_hash=str(raw.get("content_hash", "")),
        chunks=chunks,
    )


def section_slug_from_chunk(sidecar_chunk: SidecarChunk) -> str:
    """Derive the RFC-003 section slug from a sidecar chunk record."""
    if "--" in sidecar_chunk.chunk_id:
        slug = sidecar_chunk.chunk_id.split("--", 1)[1]
        if slug:
            return slug
    if sidecar_chunk.section:
        return _slugify(sidecar_chunk.section)
    return "opening"


def extract_line_range_text(raw_content: str, line_range: tuple[int, int]) -> str:
    """Extract text for a 1-indexed inclusive line range from raw markdown."""
    lines = raw_content.splitlines(keepends=True)
    start, end = line_range
    if start < 1 or end < start:
        return ""
    segment = "".join(lines[start - 1 : end])
    return segment


def embeddable_span(raw_span: str) -> str:
    """Return the embeddable body text for a raw-file span."""
    match = _FRONTMATTER_RE.match(raw_span)
    if match:
        return raw_span[match.end() :]
    return raw_span


def span_sha256(text: str) -> str:
    """SHA-256 hex digest of a span (no prefix)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug or "section"
