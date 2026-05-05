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
---
# Alpha

Alpha is the first concept.
"""
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
                    content_hash="abc123",
                    verified_at="2026-05-05",
                    verified_by="test-suite",
                ),
                content="# Alpha\n\nAlpha is the first concept.\n",
                raw_content=raw,
                size_tokens=10,
            )
        },
    )
    return RetrievalEngine(pack, DummyStore(), DummyProvider())


def test_reconstruct_enriches_whole_file_result():
    engine = _engine()
    result = SearchResult(
        text="# Alpha\n\nAlpha is the first concept.\n",
        source_file="concepts/alpha.md",
        id="test-pack/concepts/alpha",
        content_hash="abc123",
        verified_at="2026-05-05",
        score=0.9,
        type="concept",
        tags=["alpha"],
        chunk_index=0,
        title="Alpha",
    )

    engine._enrich_with_reconstruct([result])

    assert result.original_span.startswith("---\nid: test-pack/concepts/alpha")
    assert result.byte_offset == 0
    assert result.provenance_block["id"] == "test-pack/concepts/alpha"
    assert result.provenance_block["source_file"] == "concepts/alpha.md"
    assert result.provenance_block["chunk_index"] == 0
    assert result.provenance_block["verified_by"] == "test-suite"
    assert result.provenance_block["span_sha256"].startswith("sha256:")
    assert result.provenance_block["file_sha256"].startswith("sha256:")
