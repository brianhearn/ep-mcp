"""EP-native chunking: file = chunk, with oversized splitting and RFC-004 sidecars."""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..pack.sidecar import (
    ChunkSidecar,
    embeddable_span,
    extract_line_range_text,
    section_slug_from_chunk,
    span_sha256,
)

@dataclass
class Chunk:
    """A single content chunk ready for indexing."""

    file_path: str
    chunk_index: int
    content: str
    title: str | None
    token_count: int
    line_range: tuple[int, int] | None = None
    section_slug: str | None = None
    sidecar_chunk_id: str | None = None
    span_hash: str | None = None


# Default token threshold for splitting (from ARCHITECTURE.md §4.2)
DEFAULT_MAX_TOKENS = 1000

# Heading patterns for split points
_HEADING_RE = re.compile(r"^(#{2,3})\s+(.+)$", re.MULTILINE)


def chunk_file(
    file_path: str,
    content: str,
    title: str | None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    raw_content: str | None = None,
    sidecar: ChunkSidecar | None = None,
) -> list[Chunk]:
    """Chunk a single file for indexing.

    EP schema-as-chunker: most files pass through as a single chunk.
    When an RFC-004 sidecar is present, its boundaries are authoritative.
    Otherwise oversized files (>max_tokens) are split at heading boundaries.

    Args:
        file_path: Relative path within pack
        content: Markdown content (frontmatter already stripped)
        title: File title for context prefix on splits
        max_tokens: Token threshold for splitting
        raw_content: Full markdown with frontmatter (needed for sidecar line ranges)
        sidecar: Optional RFC-004 chunk metadata sidecar

    Returns:
        List of Chunk objects (usually just one)
    """
    if sidecar is not None:
        return _chunk_from_sidecar(file_path, raw_content or content, title, sidecar)

    tokens = estimate_tokens(content)

    # Most EP files are under the threshold — pass through intact
    if tokens <= max_tokens:
        return [
            Chunk(
                file_path=file_path,
                chunk_index=0,
                content=content,
                title=title,
                token_count=tokens,
                line_range=(1, max(1, len((raw_content or content).splitlines()))),
                section_slug="opening",
                span_hash=span_sha256(content),
            )
        ]

    # Oversized: split at ## or ### headings
    sections = _split_at_headings(content)

    if len(sections) <= 1:
        # No headings to split on — return as single chunk anyway
        return [
            Chunk(
                file_path=file_path,
                chunk_index=0,
                content=content,
                title=title,
                token_count=tokens,
                line_range=(1, max(1, len((raw_content or content).splitlines()))),
                section_slug="opening",
                span_hash=span_sha256(content),
            )
        ]

    # Build chunks, prefixing each with the file title for context
    chunks = []
    for i, section in enumerate(sections):
        section_content = section.strip()
        if not section_content:
            continue

        # Prefix with file title if this isn't the first section
        if i > 0 and title:
            section_content = f"# {title}\n\n{section_content}"

        section_slug = _section_slug_from_content(section_content, i)
        chunks.append(
            Chunk(
                file_path=file_path,
                chunk_index=len(chunks),
                content=section_content,
                title=title,
                token_count=estimate_tokens(section_content),
                section_slug=section_slug,
                span_hash=span_sha256(section_content),
            )
        )

    return chunks if chunks else [
        Chunk(
            file_path=file_path,
            chunk_index=0,
            content=content,
            title=title,
            token_count=tokens,
            line_range=(1, max(1, len((raw_content or content).splitlines()))),
            section_slug="opening",
            span_hash=span_sha256(content),
        )
    ]


def _chunk_from_sidecar(
    file_path: str,
    raw_content: str,
    title: str | None,
    sidecar: ChunkSidecar,
) -> list[Chunk]:
    """Build index chunks from an RFC-004 sidecar."""
    chunks: list[Chunk] = []
    for sidecar_chunk in sidecar.chunks:
        raw_span = extract_line_range_text(raw_content, sidecar_chunk.line_range)
        content = embeddable_span(raw_span).strip("\n")
        if not content:
            continue
        chunks.append(
            Chunk(
                file_path=file_path,
                chunk_index=sidecar_chunk.chunk_order,
                content=content,
                title=title,
                token_count=estimate_tokens(content),
                line_range=sidecar_chunk.line_range,
                section_slug=section_slug_from_chunk(sidecar_chunk),
                sidecar_chunk_id=sidecar_chunk.chunk_id or None,
                span_hash=span_sha256(content),
            )
        )
    return chunks


def _section_slug_from_content(section_content: str, index: int) -> str:
    if index == 0:
        return "opening"
    match = _HEADING_RE.search(section_content)
    if match:
        return _slugify(match.group(2))
    return f"section-{index}"


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug or "section"


def _split_at_headings(content: str) -> list[str]:
    """Split markdown content at ## or ### headings.

    Returns a list of sections. The first section is everything
    before the first heading (may be empty).
    """
    # Find all heading positions
    splits = []
    for match in _HEADING_RE.finditer(content):
        splits.append(match.start())

    if not splits:
        return [content]

    sections = []
    # Content before first heading
    if splits[0] > 0:
        sections.append(content[: splits[0]])

    # Each heading section
    for i, start in enumerate(splits):
        end = splits[i + 1] if i + 1 < len(splits) else len(content)
        sections.append(content[start:end])

    return sections


def estimate_tokens(text: str) -> int:
    """Fast token count approximation.

    Uses word count × 1.3 as specified in ARCHITECTURE.md §4.2.
    """
    words = len(text.split())
    return int(words * 1.3)
