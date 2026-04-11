"""Tests for pack loading."""

import tempfile
from pathlib import Path

import pytest

from ep_mcp.pack.loader import PackLoadError, load_pack


@pytest.fixture
def pack_dir():
    """Create a minimal valid pack directory."""
    with tempfile.TemporaryDirectory() as d:
        p = Path(d)

        # manifest.yaml
        (p / "manifest.yaml").write_text("""
slug: test-pack
name: Test Pack
type: product
version: "1.0.0"
description: A test pack
entry_point: overview.md
context:
  always:
    - overview.md
""", encoding="utf-8")

        # overview.md with frontmatter
        (p / "overview.md").write_text("""---
id: test-pack/overview
content_hash: abc123
verified_at: 2026-04-10
verified_by: human
type: concept
tags:
  - core
  - overview
---

# Test Pack Overview

This is the overview of the test pack. It contains important information
about the pack structure and purpose.
""", encoding="utf-8")

        # concepts dir
        concepts = p / "concepts"
        concepts.mkdir()
        (concepts / "topic-a.md").write_text("""---
id: test-pack/concepts/topic-a
type: concept
tags:
  - alpha
---

# Topic A

Content about topic A with enough words to be meaningful for testing
token estimation and chunking behavior.
""", encoding="utf-8")

        # File without frontmatter
        (concepts / "topic-b.md").write_text("""# Topic B

A file with no frontmatter at all. Just plain markdown content.
""", encoding="utf-8")

        yield p


@pytest.fixture
def pack_with_graph(pack_dir):
    """Add a _graph.yaml to the pack."""
    (pack_dir / "_graph.yaml").write_text("""
nodes:
  - overview.md
  - concepts/topic-a.md
  - concepts/topic-b.md
edges:
  - source: overview.md
    target: concepts/topic-a.md
    kind: wikilink
  - source: concepts/topic-a.md
    target: concepts/topic-b.md
    kind: related
    weight: 0.8
""", encoding="utf-8")
    return pack_dir


class TestLoadPack:
    def test_load_valid_pack(self, pack_dir):
        pack = load_pack(pack_dir)
        assert pack.slug == "test-pack"
        assert pack.name == "Test Pack"
        assert pack.type == "product"
        assert pack.version == "1.0.0"
        assert len(pack.files) == 3

    def test_files_parsed(self, pack_dir):
        pack = load_pack(pack_dir)
        assert "overview.md" in pack.files
        assert "concepts/topic-a.md" in pack.files
        assert "concepts/topic-b.md" in pack.files

    def test_frontmatter_stripped(self, pack_dir):
        pack = load_pack(pack_dir)
        overview = pack.files["overview.md"]
        assert "---" not in overview.content
        assert "# Test Pack Overview" in overview.content
        assert "id: test-pack/overview" not in overview.content

    def test_frontmatter_preserved_in_raw(self, pack_dir):
        pack = load_pack(pack_dir)
        overview = pack.files["overview.md"]
        assert "---" in overview.raw_content
        assert "id: test-pack/overview" in overview.raw_content

    def test_provenance_extracted(self, pack_dir):
        pack = load_pack(pack_dir)
        overview = pack.files["overview.md"]
        assert overview.provenance.id == "test-pack/overview"
        assert overview.provenance.verified_at == "2026-04-10"
        assert overview.provenance.verified_by == "human"
        # content_hash is recomputed from stripped content
        assert overview.provenance.content_hash is not None
        assert len(overview.provenance.content_hash) == 64  # SHA-256 hex

    def test_no_frontmatter_file(self, pack_dir):
        pack = load_pack(pack_dir)
        topic_b = pack.files["concepts/topic-b.md"]
        assert topic_b.provenance.id is None
        assert topic_b.title == "Topic B"
        assert "# Topic B" in topic_b.content

    def test_title_extraction(self, pack_dir):
        pack = load_pack(pack_dir)
        assert pack.files["overview.md"].title == "Test Pack Overview"
        assert pack.files["concepts/topic-a.md"].title == "Topic A"
        assert pack.files["concepts/topic-b.md"].title == "Topic B"

    def test_type_extraction(self, pack_dir):
        pack = load_pack(pack_dir)
        assert pack.files["overview.md"].type == "concept"
        assert pack.files["concepts/topic-a.md"].type == "concept"
        assert pack.files["concepts/topic-b.md"].type is None

    def test_tags_extraction(self, pack_dir):
        pack = load_pack(pack_dir)
        assert pack.files["overview.md"].tags == ["core", "overview"]
        assert pack.files["concepts/topic-a.md"].tags == ["alpha"]
        assert pack.files["concepts/topic-b.md"].tags == []

    def test_retrieval_strategy(self, pack_dir):
        pack = load_pack(pack_dir)
        assert pack.files["overview.md"].retrieval_strategy == "always"
        assert pack.files["concepts/topic-a.md"].retrieval_strategy == "standard"

    def test_token_count(self, pack_dir):
        pack = load_pack(pack_dir)
        for f in pack.files.values():
            assert f.size_tokens > 0

    def test_slug_override(self, pack_dir):
        pack = load_pack(pack_dir, slug_override="custom-slug")
        assert pack.slug == "custom-slug"

    def test_index_path(self, pack_dir):
        pack = load_pack(pack_dir)
        assert pack.index_path.endswith(".ep-mcp/index.db")

    def test_graph_loaded(self, pack_with_graph):
        pack = load_pack(pack_with_graph)
        assert pack.graph is not None
        assert len(pack.graph.nodes) == 3
        assert len(pack.graph.edges) == 2

    def test_graph_neighbors(self, pack_with_graph):
        pack = load_pack(pack_with_graph)
        neighbors = pack.graph.neighbors("overview.md")
        assert "concepts/topic-a.md" in neighbors

    def test_graph_edge_kinds(self, pack_with_graph):
        pack = load_pack(pack_with_graph)
        neighbors = pack.graph.neighbors("concepts/topic-a.md", edge_kinds=["related"])
        assert "concepts/topic-b.md" in neighbors
        assert "overview.md" not in neighbors

    def test_no_graph(self, pack_dir):
        pack = load_pack(pack_dir)
        assert pack.graph is None

    def test_missing_directory(self):
        with pytest.raises(PackLoadError, match="does not exist"):
            load_pack("/nonexistent/path")

    def test_missing_manifest(self, pack_dir):
        (pack_dir / "manifest.yaml").unlink()
        with pytest.raises(PackLoadError, match="No manifest.yaml"):
            load_pack(pack_dir)

    def test_skips_hidden_dirs(self, pack_dir):
        git_dir = pack_dir / ".git"
        git_dir.mkdir()
        (git_dir / "config.md").write_text("# Git config")
        pack = load_pack(pack_dir)
        assert ".git/config.md" not in pack.files

    def test_skips_obsidian_dir(self, pack_dir):
        obs_dir = pack_dir / ".obsidian"
        obs_dir.mkdir()
        (obs_dir / "settings.md").write_text("# Settings")
        pack = load_pack(pack_dir)
        assert ".obsidian/settings.md" not in pack.files
