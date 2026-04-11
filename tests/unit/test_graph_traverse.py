"""Tests for the ep_graph_traverse tool."""

from __future__ import annotations

import pytest

from ep_mcp.pack.models import (
    ContextTiers,
    GraphEdge,
    Manifest,
    Pack,
    PackGraph,
)
from ep_mcp.retrieval.graph_helpers import GraphLookup
from ep_mcp.tools.ep_graph_traverse import ep_graph_traverse


NODES_RAW = [
    {"id": "pack/concepts/alpha", "title": "Alpha Concept", "type": "concept", "file": "concepts/alpha.md"},
    {"id": "pack/concepts/beta", "title": "Beta Concept", "type": "concept", "file": "concepts/beta.md"},
    {"id": "pack/workflows/wf-alpha", "title": "Alpha Workflow", "type": "workflow", "file": "workflows/wf-alpha.md"},
    {"id": "pack/reference/ref-config", "title": "Config Reference", "type": "reference", "file": "reference/ref-config.md"},
    {"id": "pack/concepts/gamma", "title": "Gamma Concept", "type": "concept", "file": "concepts/gamma.md"},
]

EDGES = [
    GraphEdge(source="pack/concepts/alpha", target="pack/workflows/wf-alpha", kind="wikilink"),
    GraphEdge(source="pack/concepts/alpha", target="pack/reference/ref-config", kind="related"),
    GraphEdge(source="pack/workflows/wf-alpha", target="pack/concepts/beta", kind="wikilink"),
    GraphEdge(source="pack/concepts/beta", target="pack/concepts/gamma", kind="wikilink"),
]


def _build_pack(edges: list[GraphEdge] | None = None) -> Pack:
    """Build a minimal Pack with a graph."""
    e = edges if edges is not None else EDGES
    graph = PackGraph(
        nodes=[str(n) for n in NODES_RAW],
        edges=e,
    )
    return Pack(
        slug="pack",
        name="Test Pack",
        type="product",
        version="1.0.0",
        files={},
        manifest=Manifest(slug="pack", name="Test Pack", type="product", context=ContextTiers()),
        graph=graph,
        pack_dir="",
    )


def _build_lookup() -> GraphLookup:
    """Build a GraphLookup from test nodes."""
    return GraphLookup.from_raw_nodes(NODES_RAW)


class TestGraphTraverse:
    """Tests for the ep_graph_traverse function."""

    def test_basic_1hop_traversal(self):
        """1-hop traversal from alpha should find wf-alpha and ref-config."""
        pack = _build_pack()
        lookup = _build_lookup()

        result = ep_graph_traverse(
            pack=pack, graph_lookup=lookup,
            file_path="concepts/alpha.md", depth=1,
        )

        assert result["start_node"]["file"] == "concepts/alpha.md"
        assert result["start_node"]["title"] == "Alpha Concept"
        assert result["start_node"]["type"] == "concept"

        connected_files = {c["file"] for c in result["connected"]}
        assert "workflows/wf-alpha.md" in connected_files
        assert "reference/ref-config.md" in connected_files
        assert len(result["connected"]) == 2

        # All should be depth 1
        for c in result["connected"]:
            assert c["depth"] == 1

    def test_depth2_traversal(self):
        """Depth=2 from alpha should reach beta via wf-alpha."""
        pack = _build_pack()
        lookup = _build_lookup()

        result = ep_graph_traverse(
            pack=pack, graph_lookup=lookup,
            file_path="concepts/alpha.md", depth=2,
        )

        connected_files = {c["file"] for c in result["connected"]}
        # Depth 1: wf-alpha, ref-config
        # Depth 2: beta (via wf-alpha)
        assert "workflows/wf-alpha.md" in connected_files
        assert "reference/ref-config.md" in connected_files
        assert "concepts/beta.md" in connected_files

        # Check depth annotations
        depth_map = {c["file"]: c["depth"] for c in result["connected"]}
        assert depth_map["workflows/wf-alpha.md"] == 1
        assert depth_map["reference/ref-config.md"] == 1
        assert depth_map["concepts/beta.md"] == 2

    def test_edge_kind_filtering(self):
        """Filtering by edge_kind should only return matching edges."""
        pack = _build_pack()
        lookup = _build_lookup()

        # Only wikilinks — should exclude ref-config (which is "related")
        result = ep_graph_traverse(
            pack=pack, graph_lookup=lookup,
            file_path="concepts/alpha.md", depth=1,
            edge_kinds=["wikilink"],
        )

        connected_files = {c["file"] for c in result["connected"]}
        assert "workflows/wf-alpha.md" in connected_files
        assert "reference/ref-config.md" not in connected_files

        # Only related — should only find ref-config
        result2 = ep_graph_traverse(
            pack=pack, graph_lookup=lookup,
            file_path="concepts/alpha.md", depth=1,
            edge_kinds=["related"],
        )

        connected_files2 = {c["file"] for c in result2["connected"]}
        assert "reference/ref-config.md" in connected_files2
        assert "workflows/wf-alpha.md" not in connected_files2

    def test_file_not_found_returns_empty(self):
        """A file not in the graph should return empty connected list."""
        pack = _build_pack()
        lookup = _build_lookup()

        result = ep_graph_traverse(
            pack=pack, graph_lookup=lookup,
            file_path="nonexistent/file.md", depth=1,
        )

        assert result["start_node"]["file"] == "nonexistent/file.md"
        assert result["start_node"]["title"] is None
        assert result["connected"] == []
        assert result["total_edges_traversed"] == 0

    def test_depth_clamped_to_1_3(self):
        """Depth should be clamped: 0→1, 5→3."""
        pack = _build_pack()
        lookup = _build_lookup()

        # Depth 0 should become 1
        result_low = ep_graph_traverse(
            pack=pack, graph_lookup=lookup,
            file_path="concepts/alpha.md", depth=0,
        )
        # Should have depth-1 results (same as depth=1)
        connected_files = {c["file"] for c in result_low["connected"]}
        assert len(connected_files) > 0
        for c in result_low["connected"]:
            assert c["depth"] == 1

        # Depth 5 should be clamped to 3
        result_high = ep_graph_traverse(
            pack=pack, graph_lookup=lookup,
            file_path="concepts/alpha.md", depth=5,
        )
        # Should find nodes at depth 3 (gamma via alpha->wf-alpha->beta->gamma)
        connected_files_high = {c["file"] for c in result_high["connected"]}
        assert "concepts/gamma.md" in connected_files_high

        # Verify max depth is 3
        max_depth = max(c["depth"] for c in result_high["connected"])
        assert max_depth == 3

    def test_no_graph_returns_empty(self):
        """When pack has no graph, should return empty connected list."""
        pack = Pack(
            slug="pack",
            name="Test Pack",
            type="product",
            version="1.0.0",
            files={},
            manifest=Manifest(slug="pack", name="Test Pack", type="product", context=ContextTiers()),
            graph=None,
            pack_dir="",
        )

        result = ep_graph_traverse(
            pack=pack, graph_lookup=None,
            file_path="concepts/alpha.md", depth=1,
        )

        assert result["connected"] == []
        assert result["total_edges_traversed"] == 0

    def test_no_lookup_returns_empty(self):
        """When graph_lookup is None, should return empty connected list."""
        pack = _build_pack()

        result = ep_graph_traverse(
            pack=pack, graph_lookup=None,
            file_path="concepts/alpha.md", depth=1,
        )

        assert result["connected"] == []
        assert result["total_edges_traversed"] == 0
