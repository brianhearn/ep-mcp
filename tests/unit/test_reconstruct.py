"""Tests for reconstruct-mode search result enrichment."""

import pytest

from ep_mcp.pack.models import Manifest, Pack, PackFile, Provenance
from ep_mcp.retrieval.engine import RetrievalEngine
from ep_mcp.retrieval.models import SearchResult


class DummyStore:
    pass


class DummyProvider:
    pass


def _engine() -> RetrievalEngine:
    raw = """---
id: test-pack/concepts/alpha
verified_at: 2026-05-05
verified_by: test-suite
confidence: expert-verified
---
# Alpha

Alpha is the first concept.
"""
    content = "# Alpha\n\nAlpha is the first concept.\n"
    pack = Pack(
        slug="test-pack",
        name="Test Pack",
        type="product",
        version="1.0.0",
        manifest=Manifest(
            slug="test-pack",
            name="Test Pack",
            type="product",
            version="1.0.0",
            description="Test pack",
            entry_point="overview.md",
        ),
        files={
            "concepts/alpha.md": PackFile(
                path="concepts/alpha.md",
                title="Alpha",
                type="concept",
                tags=["alpha"],
                provenance=Provenance(
                    id="test-pack/concepts/alpha",
                    content_hash="sha256:abc123",
                    verified_at="2026-05-05",
                    verified_by="test-suite",
                    confidence="expert-verified",
                ),
                content=content,
                raw_content=raw,
                size_tokens=10,
            )
        },
    )
    return RetrievalEngine(pack, DummyStore(), DummyProvider())


def test_reconstruct_enriches_whole_file_result():
    engine = _engine()
    content = "# Alpha\n\nAlpha is the first concept.\n"
    result = SearchResult(
        text=content,
        source_file="concepts/alpha.md",
        id="test-pack/concepts/alpha",
        content_hash="sha256:abc123",
        verified_at="2026-05-05",
        score=0.9,
        type="concept",
        tags=["alpha"],
        chunk_index=0,
        title="Alpha",
        section_slug="opening",
        span_hash=__import__("hashlib").sha256(content.encode()).hexdigest(),
        confidence="expert-verified",
    )

    engine._enrich_with_reconstruct([result])

    assert result.original_span.startswith("---\nid: test-pack/concepts/alpha")
    assert result.original_markdown == result.original_span
    assert result.byte_offset[0] == 0
    assert result.byte_offset[1] == len(result.original_span.encode("utf-8"))
    assert result.fragment_id.startswith("test-pack/concepts/alpha#opening:")
    assert result.line_range[0] == 1
    assert result.stale is False
    assert result.excerpt == content
    assert result.content_hash.startswith("sha256:")
    assert result.provenance_block["fragment_id"] == result.fragment_id
    assert result.provenance_block["confidence"] == "expert-verified"
    assert result.provenance_block["span_sha256"].startswith("sha256:")
    assert result.provenance_block["file_sha256"] == "sha256:abc123"
