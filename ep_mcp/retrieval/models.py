"""Pydantic models for search requests and results."""

from __future__ import annotations

from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    """A search query with optional filters.

    When ``vector`` is supplied, the retrieval engine skips the internal
    ``embed_query`` call and uses the caller-supplied embedding directly.
    This lets clients that have already embedded the query upstream (e.g.
    OpenClaw's memory plugin) avoid a redundant round-trip to the embedding
    provider.

    The vector dimension MUST match the pack's configured embedding provider
    (e.g. 3072 for ``gemini-embedding-001``). Mismatched dimensions raise a
    validation error at the HTTP layer before the engine is called.
    """

    query: str
    type: str | None = None
    tags: list[str] | None = None
    max_results: int = Field(10, ge=1, le=50)
    vector: list[float] | None = Field(
        default=None,
        description=(
            "Optional pre-computed query embedding. When provided, the engine "
            "skips its own embed_query() call. Dimension must match the pack's "
            "embedding provider."
        ),
    )
    reconstruct: bool = Field(
        default=False,
        description=(
            "When true, enrich results with original markdown spans and a "
            "structured provenance block for verification."
        ),
    )


class SearchResult(BaseModel):
    """A single search result with provenance metadata."""

    text: str = Field(description="Content (frontmatter stripped)")
    source_file: str = Field(description="Path within pack")
    id: str | None = Field(None, description="Provenance ID")
    content_hash: str | None = Field(None, description="SHA-256 for verification")
    verified_at: str | None = Field(None, description="ISO 8601 verification date")
    score: float = Field(description="Relevance score (0-1)")
    type: str | None = Field(None, description="Content type")
    tags: list[str] = Field(default_factory=list)
    chunk_index: int = Field(0, description="0 for whole-file chunks")
    title: str | None = Field(None, description="Content title")
    graph_expanded: bool = Field(False, description="Whether this result was added via post-top-K graph expansion")
    requires_expanded: bool = Field(False, description="Whether this result was added via post-top-K requires: expansion (atomic-conceptual dependencies)")
    original_span: str | None = Field(
        None,
        description="Original markdown span for verification/reconstruction",
    )
    provenance_block: dict | None = Field(
        None,
        description="Structured provenance metadata for the returned span",
    )
    byte_offset: int | list[int] | None = Field(
        None,
        description="UTF-8 byte offset(s) of the span within the raw markdown file",
    )
    fragment_id: str | None = Field(
        None,
        description="RFC-003 content-addressed span ID",
    )
    line_range: tuple[int, int] | None = Field(
        None,
        description="1-indexed inclusive line range in the source markdown file",
    )
    section_slug: str | None = Field(
        None,
        description="Kebab-case section slug used in fragment_id",
    )
    span_hash: str | None = Field(
        None,
        description="SHA-256 hex digest of the embeddable span at index time",
    )
    confidence: str | None = Field(
        None,
        description="Provenance grade from frontmatter",
    )
    excerpt: str | None = Field(
        None,
        description="Matched passage (same as text in reconstruct mode)",
    )
    original_markdown: str | None = Field(
        None,
        description="Full source span from the raw markdown file (RFC-003)",
    )
    stale: bool | None = Field(
        None,
        description="True when the span hash no longer matches the indexed hash",
    )
