"""Tests for ep_read and indexability helpers."""

from pathlib import Path

import pytest

from ep_mcp.pack.loader import load_pack
from ep_mcp.pack.models import PackFile
from ep_mcp.tools.ep_read import ep_read


@pytest.fixture
def consume_pack(tmp_path: Path):
    (tmp_path / "manifest.yaml").write_text(
        """
slug: consume-pack
name: Consume Pack
type: product
version: "1.0.0"
entry_point: overview.md
authority_boundary:
  in_scope: Product knowledge
  out_of_scope:
    - Legal advice
  refuse_when:
    - No supporting atom
  no_source_no_claim: true
context:
  always:
    - overview.md
  on_demand:
    - extras/hidden.md
""",
        encoding="utf-8",
    )
    (tmp_path / "overview.md").write_text(
        """---
id: consume-pack/overview
type: concept
---
# Overview
Always-tier overview.
""",
        encoding="utf-8",
    )
    concepts = tmp_path / "concepts"
    concepts.mkdir()
    (concepts / "topic-a.md").write_text(
        """---
id: consume-pack/concepts/topic-a
type: concept
requires:
  - topic-b.md
---
# Topic A
Atom A body.
""",
        encoding="utf-8",
    )
    (concepts / "topic-b.md").write_text(
        """---
id: consume-pack/concepts/topic-b
type: concept
---
# Topic B
Atom B body.
""",
        encoding="utf-8",
    )
    (concepts / "_index.md").write_text(
        """---
title: Concepts index
type: concept
retrieval_strategy: navigation
---
# Index
Links only.
""",
        encoding="utf-8",
    )
    extras = tmp_path / "extras"
    extras.mkdir()
    (extras / "hidden.md").write_text(
        """---
id: consume-pack/extras/hidden
type: reference
---
# Hidden
On-demand only.
""",
        encoding="utf-8",
    )
    workflows = tmp_path / "workflows"
    workflows.mkdir()
    (workflows / "do-thing.md").write_text(
        """---
id: consume-pack/workflows/do-thing
type: workflow
retrieval_strategy: atomic
activation:
  tools:
    - ep_read
  constraints:
    - Stay in scope
  next:
    - concepts/topic-a.md
---
# Do Thing
Workflow body.
""",
        encoding="utf-8",
    )
    return load_pack(tmp_path)


class TestEpRead:
    def test_read_by_path(self, consume_pack):
        result = ep_read(consume_pack, path="concepts/topic-a.md")
        assert result["title"] == "Topic A"
        assert "Atom A body" in result["content"]
        assert result["requires"] == ["topic-b.md"]
        assert result["id"] == "consume-pack/concepts/topic-a"

    def test_read_by_id(self, consume_pack):
        result = ep_read(consume_pack, id="consume-pack/concepts/topic-b")
        assert result["path"] == "concepts/topic-b.md"
        assert "Atom B body" in result["content"]

    def test_missing_file(self, consume_pack):
        result = ep_read(consume_pack, path="nope.md")
        assert "error" in result

    def test_requires_path_or_id(self, consume_pack):
        result = ep_read(consume_pack)
        assert result["error"] == "Provide path or id"

    def test_activation_on_workflow(self, consume_pack):
        result = ep_read(consume_pack, path="workflows/do-thing.md")
        assert result["activation"]["tools"] == ["ep_read"]
        assert result["activation"]["next"] == ["concepts/topic-a.md"]

    def test_reconstruct(self, consume_pack):
        result = ep_read(consume_pack, path="concepts/topic-a.md", reconstruct=True)
        assert "---" in result["original_markdown"]
        assert result["provenance_block"]["id"] == "consume-pack/concepts/topic-a"

    def test_reads_on_demand_and_navigation(self, consume_pack):
        hidden = ep_read(consume_pack, path="extras/hidden.md")
        assert "On-demand only" in hidden["content"]
        index = ep_read(consume_pack, path="concepts/_index.md")
        assert index["retrieval_strategy"] == "navigation"


class TestIndexManagerFilter:
    @pytest.mark.asyncio
    async def test_skips_navigation_and_on_demand(self, consume_pack, tmp_path):
        from ep_mcp.embeddings.base import EmbeddingProvider
        from ep_mcp.index.manager import INDEX_FEATURES, IndexManager
        from ep_mcp.index.sqlite_store import SQLiteStore

        class FakeProvider(EmbeddingProvider):
            @property
            def model_name(self) -> str:
                return "fake/test"

            @property
            def dimension(self) -> int:
                return 4

            async def embed(self, texts: list[str]) -> list[list[float]]:
                return [[0.1, 0.2, 0.3, 0.4] for _ in texts]

        consume_pack.index_path = str(tmp_path / "idx" / "index.db")
        store = SQLiteStore(consume_pack.index_path, embedding_dimension=4)
        store.open()
        try:
            stats = await IndexManager(consume_pack, store, FakeProvider()).build_index()
            indexed = store.get_indexed_files()
            assert "concepts/_index.md" not in indexed
            assert "extras/hidden.md" not in indexed
            assert "concepts/topic-a.md" in indexed
            assert store.get_meta("index_features") == INDEX_FEATURES
            assert stats.total_chunks >= 1
        finally:
            store.close()


class TestIndexability:
    def test_navigation_file_not_indexable(self, consume_pack):
        assert consume_pack.files["concepts/_index.md"].is_indexable is False

    def test_on_demand_not_indexable(self, consume_pack):
        assert consume_pack.files["extras/hidden.md"].is_indexable is False

    def test_standard_is_indexable(self, consume_pack):
        assert consume_pack.files["concepts/topic-a.md"].is_indexable is True

    def test_index_basename_excluded_even_if_standard(self):
        f = PackFile(
            path="concepts/_index.md",
            retrieval_strategy="standard",
            content="hub",
        )
        assert f.is_indexable is False
