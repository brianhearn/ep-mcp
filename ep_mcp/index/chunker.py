"""EP-native chunking: file = chunk, with oversized splitting."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class Chunk:
    """A single content chunk ready for indexing."""

    file_path: str
    chunk_index: int
    content: str
    title: str | None
    token_count: int


# Default token threshold for splitting (from ARCHITECTURE.md §4.2)
DEFAULT_MAX_TOKENS = 1000

# Heading patterns for split points
_HEADING_RE = re.compile(r"^(#{2,3})\s+(.+)$", re.MULTILINE)


def chunk_file(
    file_path: str,
    content: str,
    title: str | None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> list[Chunk]:
    """Chunk a single file for indexing.

    EP schema-as-chunker: most files pass through as a single chunk.
    Oversized files (>max_tokens) are split at heading boundaries.

    Args:
        file_path: Relative path within pack
        content: Markdown content (frontmatter already stripped)
        title: File title for context prefix on splits
        max_tokens: Token threshold for splitting

    Returns:
        List of Chunk objects (usually just one)
    """
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

        chunks.append(
            Chunk(
                file_path=file_path,
                chunk_index=len(chunks),
                content=section_content,
                title=title,
                token_count=estimate_tokens(section_content),
            )
        )

    return chunks if chunks else [
        Chunk(
            file_path=file_path,
            chunk_index=0,
            content=content,
            title=title,
            token_count=tokens,
        )
    ]


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
