"""Tests for graph expansion in the retrieval engine."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from ep_mcp.config import RetrievalConfig
from ep_mcp.index.sqlite_store import SQLiteStore
from ep_mcp.pack.models import (
    ContextTiers,
    GraphEdge,
    Manifest,
    Pack,
    PackGraph,
)
from ep_mcp.retrieval.engine import RetrievalEngine
from ep_mcp.retrieval.graph_helpers import GraphLookup
from ep_mcp.retrieval.models import SearchRequest


def _fake_embedding(dim: int = 4) -> list[float]:
    """Generate a simple test embedding."""
    return [0.1 * i for i in range(dim)]


def _build_pack_with_graph(
    nodes_raw: list[dict],
    edges: list[GraphEdge],
) -> Pack:
    """Build a minimal Pack with a graph."""
    graph = PackGraph(
        nodes=[str(n) for n in nodes_raw],
        edges=edges,
    )
    return Pack(
        slug="test-pack",
        name="Test Pack",
        type="product",
        version="1.0.0",
        files={},
        manifest=Manifest(
            slug="test-pack",
            name="Test Pack",
            type="product",
            context=ContextTiers(),
        ),
        graph=graph,
        pack_dir="",
    )


def _build_pack_no_graph() -> Pack:
    """Build a minimal Pack without a graph."""
    return Pack(
        slug="test-pack",
        name="Test Pack",
        type="product",
        version="1.0.0",
        files={},
        manifest=Manifest(
            slug="test-pack",
            name="Test Pack",
            type="product",
            context=ContextTiers(),
        ),
        graph=None,
        pack_dir="",
    )


@pytest.fixture
def store():
    """Create a temporary SQLite store with test data."""
    with tempfile.TemporaryDirectory() as d:
        db_path = str(Path(d) / "test.db")
        s = SQLiteStore(db_path, embedding_dimension=4)
        s.open()
        yield s
        s.close()


def _seed_chunks(store: SQLiteStore, pack_slug: str = "test-pack") -> dict[str, int]:
    """Seed the store with test chunks. Returns {file_path: chunk_id}."""
    files = {
        "concepts/auto-build.md": ("Auto-Build Territories", "concept", "Auto build content"),
        "workflows/wf-auto-build.md": ("Auto-Build Workflow", "workflow", "Workflow content"),
        "reference/ref-settings.md": ("Settings Reference", "reference", "Settings content"),
        "concepts/partitioning.md": ("Partitioning", "concept", "Partitioning content"),
        "workflows/wf-partitioning.md": ("Partitioning Workflow", "workflow", "Partitioning workflow"),
    }
    result = {}
    for fp, (title, type_, content) in files.items():
        cid = store.upsert_chunk(
            file_path=fp,
            chunk_index=0,
            content=content,
            title=title,
            type_=type_,
            tags=[],
            pack_slug=pack_slug,
            prov_id=None,
            content_hash=None,
            verified_at=None,
            verified_by=None,
            token_count=10,
            embedding=_fake_embedding(),
        )
        result[fp] = cid
    store.commit()
    return result


NODES_RAW = [
    {"id": "test-pack/concepts/auto-build", "title": "Auto-Build", "type": "concept", "file": "concepts/auto-build.md"},
    {"id": "test-pack/workflows/wf-auto-build", "title": "Auto-Build Workflow", "type": "workflow", "file": "workflows/wf-auto-build.md"},
    {"id": "test-pack/reference/ref-settings", "title": "Settings Reference", "type": "reference", "file": "reference/ref-settings.md"},
    {"id": "test-pack/concepts/partitioning", "title": "Partitioning", "type": "concept", "file": "concepts/partitioning.md"},
    {"id": "test-pack/workflows/wf-partitioning", "title": "Partitioning Workflow", "type": "workflow", "file": "workflows/wf-partitioning.md"},
]

EDGES = [
    GraphEdge(source="test-pack/concepts/auto-build", target="test-pack/workflows/wf-auto-build", kind="wikilink"),
    GraphEdge(source="test-pack/concepts/auto-build", target="test-pack/reference/ref-settings", kind="wikilink"),
    GraphEdge(source="test-pack/concepts/partitioning", target="test-pack/workflows/wf-partitioning", kind="wikilink"),
    # Chain: auto-build -> partitioning (for multi-hop test)
    GraphEdge(source="test-pack/workflows/wf-auto-build", target="test-pack/concepts/partitioning", kind="wikilink"),
]


class TestGraphExpansion:
    """Tests for graph expansion in the retrieval pipeline."""

    def _make_engine(
        self,
        store: SQLiteStore,
        pack: Pack,
        graph_lookup: GraphLookup | None = None,
        config: RetrievalConfig | None = None,
    ) -> RetrievalEngine:
        """Build a RetrievalEngine with a mock embedding provider."""
        provider = AsyncMock()
        provider.embed_query = AsyncMock(return_value=_fake_embedding())
        provider.dimension = 4
        return RetrievalEngine(
            pack=pack,
            store=store,
            embedding_provider=provider,
            config=config or RetrievalConfig(),
            graph_lookup=graph_lookup,
        )

    def test_graph_expansion_adds_neighbor_chunks(self, store):
        """Graph expansion should add neighbor chunks to the candidate set."""
        chunk_ids = _seed_chunks(store)
        pack = _build_pack_with_graph(NODES_RAW, EDGES)
        lookup = GraphLookup.from_raw_nodes(NODES_RAW)

        # Simulate fused scores: only auto-build.md is in the initial set
        fused = {chunk_ids["concepts/auto-build.md"]: 0.8}
        chunks = store.get_chunks_by_ids(list(fused.keys()))

        config = RetrievalConfig(
            graph_expansion_enabled=True,
            graph_expansion_depth=1,
            graph_expansion_discount=0.6,
            min_score=0.1,
        )
        engine = self._make_engine(store, pack, lookup, config)

        new_fused, new_chunks = engine._apply_graph_expansion(
            fused, chunks, max_results=10,
        )

        # Should have added neighbors: wf-auto-build.md and ref-settings.md
        assert len(new_fused) == 3
        assert chunk_ids["workflows/wf-auto-build.md"] in new_fused
        assert chunk_ids["reference/ref-settings.md"] in new_fused

    def test_graph_expansion_respects_discount(self, store):
        """Neighbor scores should be parent_score * discount."""
        chunk_ids = _seed_chunks(store)
        pack = _build_pack_with_graph(NODES_RAW, EDGES)
        lookup = GraphLookup.from_raw_nodes(NODES_RAW)

        parent_score = 0.8
        discount = 0.5
        fused = {chunk_ids["concepts/auto-build.md"]: parent_score}
        chunks = store.get_chunks_by_ids(list(fused.keys()))

        config = RetrievalConfig(
            graph_expansion_enabled=True,
            graph_expansion_depth=1,
            graph_expansion_discount=discount,
            min_score=0.1,
        )
        engine = self._make_engine(store, pack, lookup, config)

        new_fused, _ = engine._apply_graph_expansion(fused, chunks, max_results=10)

        # Neighbor score should be 0.8 * 0.5 = 0.4
        for cid, score in new_fused.items():
            if cid != chunk_ids["concepts/auto-build.md"]:
                assert score == pytest.approx(parent_score * discount)

    def test_graph_expansion_respects_min_score(self, store):
        """Neighbors below min_score after discounting should be excluded."""
        chunk_ids = _seed_chunks(store)
        pack = _build_pack_with_graph(NODES_RAW, EDGES)
        lookup = GraphLookup.from_raw_nodes(NODES_RAW)

        # parent_score=0.5, discount=0.6 → neighbor_score=0.3 which is < min_score=0.35
        fused = {chunk_ids["concepts/auto-build.md"]: 0.5}
        chunks = store.get_chunks_by_ids(list(fused.keys()))

        config = RetrievalConfig(
            graph_expansion_enabled=True,
            graph_expansion_depth=1,
            graph_expansion_discount=0.6,
            min_score=0.35,
        )
        engine = self._make_engine(store, pack, lookup, config)

        new_fused, _ = engine._apply_graph_expansion(fused, chunks, max_results=10)

        # No neighbors should be added (0.5 * 0.6 = 0.3 < 0.35)
        assert len(new_fused) == 1

    def test_graph_expansion_noops_when_graph_is_none(self, store):
        """Graph expansion should no-op when pack.graph is None."""
        chunk_ids = _seed_chunks(store)
        pack = _build_pack_no_graph()

        fused = {chunk_ids["concepts/auto-build.md"]: 0.8}
        chunks = store.get_chunks_by_ids(list(fused.keys()))

        config = RetrievalConfig(graph_expansion_enabled=True)
        engine = self._make_engine(store, pack, None, config)

        new_fused, _ = engine._apply_graph_expansion(fused, chunks, max_results=10)

        assert len(new_fused) == 1

    def test_graph_expansion_noops_when_disabled(self, store):
        """Graph expansion should no-op when config disables it."""
        chunk_ids = _seed_chunks(store)
        pack = _build_pack_with_graph(NODES_RAW, EDGES)
        lookup = GraphLookup.from_raw_nodes(NODES_RAW)

        fused = {chunk_ids["concepts/auto-build.md"]: 0.8}
        chunks = store.get_chunks_by_ids(list(fused.keys()))

        config = RetrievalConfig(
            graph_expansion_enabled=False,
            min_score=0.1,
        )
        engine = self._make_engine(store, pack, lookup, config)

        new_fused, _ = engine._apply_graph_expansion(fused, chunks, max_results=10)

        assert len(new_fused) == 1

    def test_graph_expansion_multi_hop(self, store):
        """Depth=2 should follow chains: auto-build -> wf-auto-build -> partitioning."""
        chunk_ids = _seed_chunks(store)
        pack = _build_pack_with_graph(NODES_RAW, EDGES)
        lookup = GraphLookup.from_raw_nodes(NODES_RAW)

        fused = {chunk_ids["concepts/auto-build.md"]: 0.9}
        chunks = store.get_chunks_by_ids(list(fused.keys()))

        config = RetrievalConfig(
            graph_expansion_enabled=True,
            graph_expansion_depth=2,
            graph_expansion_discount=0.7,
            min_score=0.1,
        )
        engine = self._make_engine(store, pack, lookup, config)

        new_fused, _ = engine._apply_graph_expansion(fused, chunks, max_results=20)

        # Depth 1: wf-auto-build, ref-settings
        # Depth 2: partitioning (via wf-auto-build -> partitioning edge)
        assert chunk_ids["workflows/wf-auto-build.md"] in new_fused
        assert chunk_ids["reference/ref-settings.md"] in new_fused
        assert chunk_ids["concepts/partitioning.md"] in new_fused

        # Depth-2 neighbor should have discount^2 score: 0.9 * 0.7 * 0.7 = 0.441
        part_score = new_fused[chunk_ids["concepts/partitioning.md"]]
        assert part_score == pytest.approx(0.9 * 0.7 * 0.7)

    def test_graph_expansion_duplicate_suppression(self, store):
        """Neighbors already in the candidate set should not be re-added."""
        chunk_ids = _seed_chunks(store)
        pack = _build_pack_with_graph(NODES_RAW, EDGES)
        lookup = GraphLookup.from_raw_nodes(NODES_RAW)

        # Both auto-build and its neighbor wf-auto-build are already in fused
        fused = {
            chunk_ids["concepts/auto-build.md"]: 0.8,
            chunk_ids["workflows/wf-auto-build.md"]: 0.7,
        }
        chunks = store.get_chunks_by_ids(list(fused.keys()))

        config = RetrievalConfig(
            graph_expansion_enabled=True,
            graph_expansion_depth=1,
            graph_expansion_discount=0.6,
            min_score=0.1,
        )
        engine = self._make_engine(store, pack, lookup, config)

        new_fused, _ = engine._apply_graph_expansion(fused, chunks, max_results=10)

        # wf-auto-build should keep its original score of 0.7, not be overwritten
        assert new_fused[chunk_ids["workflows/wf-auto-build.md"]] == 0.7

        # ref-settings should be added as neighbor of auto-build
        assert chunk_ids["reference/ref-settings.md"] in new_fused
