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
    indexed_content: str | None = None
    heading_path: list[str] | None = None

    def text_for_index(self) -> str:
        """Text used for embed + BM25 (may include context_prefix)."""
        return self.indexed_content if self.indexed_content is not None else self.content


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
    prefix_heading_path: bool = False,
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
        prefix_heading_path: Put the full heading path ("# Title > H2 > H3") in
            the index-side prefix of split chunks instead of just "# Title".
            Off by default: on the 2026-10-20 golden sets it was neutral to
            slightly negative (see CHANGELOG 0.7.0). heading_path is always
            stored and returned with hits either way.
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
                heading_path=[title] if title else [],
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
                heading_path=[title] if title else [],
            )
        ]

    # Build chunks. The returned content / span_hash are the ORIGINAL span; the
    # title + heading-path context prefix goes into indexed_content only (1.3).
    raw = raw_content or content
    body_line_offset = _body_line_offset(raw, content)
    chunks = []
    pos = 0
    for i, section in enumerate(sections):
        sec_start = pos
        pos += len(section)
        section_content = section.strip()
        if not section_content:
            continue
        lead = section[: len(section) - len(section.lstrip())]
        start_line = body_line_offset + content.count("\n", 0, sec_start) + lead.count("\n") + 1
        end_line = start_line + section_content.count("\n")
        hpath = heading_path_at(raw, start_line, title)
        prefix = _context_prefix(title, hpath if prefix_heading_path else []) if i > 0 else None
        section_slug = _section_slug_from_content(section_content, i)
        chunks.append(
            Chunk(
                file_path=file_path,
                chunk_index=len(chunks),
                content=section_content,
                title=title,
                token_count=estimate_tokens(prefix + "\n\n" + section_content if prefix else section_content),
                line_range=(start_line, end_line),
                section_slug=section_slug,
                span_hash=span_sha256(section_content),
                indexed_content=f"{prefix}\n\n{section_content}" if prefix else None,
                heading_path=hpath,
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
            heading_path=[title] if title else [],
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
        prefix = (sidecar_chunk.context_prefix or "").strip()
        hpath = list(sidecar_chunk.heading_path) if sidecar_chunk.heading_path else heading_path_at(
            raw_content, sidecar_chunk.line_range[0], title
        )
        indexed = f"{prefix}\n\n{content}" if prefix else content
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
                indexed_content=indexed if prefix else None,
                heading_path=hpath,
            )
        )
    return chunks


_ANY_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_FENCE_RE = re.compile(r"^\s*(```|~~~)")


def heading_path_at(raw: str, line_no: int, title: str | None = None) -> list[str]:
    """Ancestor heading path for the 1-indexed line ``line_no`` (inclusive).

    Walks headings outside code fences and frontmatter. The H1 slot is the file
    title when the file has no explicit H1. Example: ``["Routing", "Setup", "Ports"]``.
    """
    lines = raw.splitlines()
    stack: list[tuple[int, str]] = []
    in_fence = False
    start = 0
    if lines and lines[0].strip() == "---":
        for j in range(1, len(lines)):
            if lines[j].strip() == "---":
                start = j + 1
                break
    for idx in range(start, min(line_no, len(lines))):
        line = lines[idx]
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        m = _ANY_HEADING_RE.match(line)
        if not m:
            continue
        level = len(m.group(1))
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, m.group(2).strip()))
    path = [t for _, t in stack]
    if title and (not stack or stack[0][0] != 1):
        path.insert(0, title)
    return path


def _context_prefix(title: str | None, hpath: list[str]) -> str | None:
    if not hpath:
        return f"# {title}" if title else None
    return "# " + " > ".join(hpath)


def _body_line_offset(raw: str, content: str) -> int:
    """Number of raw lines preceding the frontmatter-stripped body."""
    if raw is content or not raw.endswith(content):
        idx = raw.find(content) if content else -1
        return raw.count("\n", 0, idx) if idx > 0 else 0
    return raw.count("\n", 0, len(raw) - len(content))


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
