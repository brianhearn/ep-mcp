"""Tests for the new post-top-K additive graph expansion design."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
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
from ep_mcp.retrieval.models import SearchRequest, SearchResult


def _fake_embedding(dim: int = 4) -> list[float]:
    """Generate a simple test embedding (normalized for cosine)."""
    vec = np.random.rand(dim).astype(np.float32)
    vec /= np.linalg.norm(vec)
    return vec.tolist()


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
    GraphEdge(source="test-pack/workflows/wf-auto-build", target="test-pack/concepts/partitioning", kind="wikilink"),
]


class TestGraphExpansion:
    """Tests for the new post-top-K additive graph expansion."""

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

    def test_graph_expansion_adds_bonus_neighbors(self, store):
        """Neighbors added as bonus results after top-K; does not displace top-K."""
        chunk_ids = _seed_chunks(store)
        pack = _build_pack_with_graph(NODES_RAW, EDGES)
        lookup = GraphLookup.from_raw_nodes(NODES_RAW)

        config = RetrievalConfig(
            graph_expansion_enabled=True,
            graph_expansion_confidence_threshold=0.5,
            graph_expansion_min_score=0.1,
            graph_expansion_structural_bonus=1.0,
        )
        engine = self._make_engine(store, pack, lookup, config)

        # Create finalized top-K results (post-MMR simulation)
        top_k = [
            SearchResult(
                text="Auto build content",
                source_file="concepts/auto-build.md",
                score=0.85,
                type="concept",
                tags=[],
                chunk_index=0,
                graph_expanded=False,
            ),
            SearchResult(
                text="Some other result",
                source_file="other.md",
                score=0.65,
                type="concept",
                tags=[],
                chunk_index=0,
                graph_expanded=False,
            ),
        ]

        with patch.object(engine, '_load_embeddings_for_chunks') as mock_load:
            mock_load.return_value = {
                chunk_ids["workflows/wf-auto-build.md"]: _fake_embedding(),
                chunk_ids["reference/ref-settings.md"]: _fake_embedding(),
            }
            bonus = engine._apply_graph_expansion(top_k, _fake_embedding(4))

        assert len(bonus) == 2
        assert all(r.graph_expanded for r in bonus)
        assert any("wf-auto-build" in r.source_file for r in bonus)
        assert any("ref-settings" in r.source_file for r in bonus)
        # Top-K not in bonus
        assert not any(r in top_k for r in bonus)  # identity not, but check files
        top_files = {r.source_file for r in top_k}
        bonus_files = {r.source_file for r in bonus}
        assert len(top_files & bonus_files) == 0

    def test_no_expansion_below_confidence_threshold(self, store):
        """No expansion when no seeds meet confidence_threshold."""
        chunk_ids = _seed_chunks(store)
        pack = _build_pack_with_graph(NODES_RAW, EDGES)
        lookup = GraphLookup.from_raw_nodes(NODES_RAW)

        config = RetrievalConfig(
            graph_expansion_enabled=True,
            graph_expansion_confidence_threshold=0.7,
            graph_expansion_min_score=0.1,
        )
        engine = self._make_engine(store, pack, lookup, config)

        top_k = [
            SearchResult(
                text="Low confidence",
                source_file="concepts/auto-build.md",
                score=0.45,  # below 0.7
                type="concept",
                tags=[],
                chunk_index=0,
                graph_expanded=False,
            ),
        ]

        bonus = engine._apply_graph_expansion(top_k, _fake_embedding(4))
        assert len(bonus) == 0

    def test_neighbor_below_min_score_excluded(self, store):
        """Neighbor with cosine_sim < graph_expansion_min_score is excluded."""
        chunk_ids = _seed_chunks(store)
        pack = _build_pack_with_graph(NODES_RAW, EDGES)
        lookup = GraphLookup.from_raw_nodes(NODES_RAW)

        config = RetrievalConfig(
            graph_expansion_enabled=True,
            graph_expansion_confidence_threshold=0.3,
            graph_expansion_min_score=0.6,
            graph_expansion_structural_bonus=1.0,
        )
        engine = self._make_engine(store, pack, lookup, config)

        top_k = [
            SearchResult(
                text="Seed",
                source_file="concepts/auto-build.md",
                score=0.75,
                type="concept",
                tags=[],
                chunk_index=0,
                graph_expanded=False,
            ),
        ]

        # Mock embeddings to force low cosine sim
        with patch.object(engine, '_load_embeddings_for_chunks') as mock_load:
            low_sim_emb = [1.0, 0.0, 0.0, 0.0]  # orthogonal to typical query
            mock_load.return_value = {999: low_sim_emb}
            bonus = engine._apply_graph_expansion(top_k, [0.0, 1.0, 0.0, 0.0])

        assert len(bonus) == 0

    def test_no_readdition_of_top_k_neighbors(self, store):
        """Neighbor already in top-K is not re-added as bonus."""
        chunk_ids = _seed_chunks(store)
        pack = _build_pack_with_graph(NODES_RAW, EDGES)
        lookup = GraphLookup.from_raw_nodes(NODES_RAW)

        config = RetrievalConfig(
            graph_expansion_enabled=True,
            graph_expansion_confidence_threshold=0.5,
            graph_expansion_min_score=0.1,
        )
        engine = self._make_engine(store, pack, lookup, config)

        top_k = [
            SearchResult(
                text="Seed",
                source_file="concepts/auto-build.md",
                score=0.8,
                type="concept",
                tags=[],
                chunk_index=0,
                graph_expanded=False,
            ),
            SearchResult(
                text="Already in top-K",
                source_file="workflows/wf-auto-build.md",
                score=0.6,
                type="workflow",
                tags=[],
                chunk_index=0,
                graph_expanded=False,
            ),
        ]

        with patch.object(engine, '_load_embeddings_for_chunks') as mock_load:
            mock_load.return_value = {chunk_ids.get("reference/ref-settings.md", 999): _fake_embedding()}
            bonus = engine._apply_graph_expansion(top_k, _fake_embedding(4))

        # Only ref-settings should be added, wf-auto-build skipped
        assert len(bonus) == 1
        assert "ref-settings" in bonus[0].source_file

    def test_graph_expanded_flag_set(self, store):
        """Bonus results must have graph_expanded=True."""
        chunk_ids = _seed_chunks(store)
        pack = _build_pack_with_graph(NODES_RAW, EDGES)
        lookup = GraphLookup.from_raw_nodes(NODES_RAW)

        config = RetrievalConfig(
            graph_expansion_enabled=True,
            graph_expansion_confidence_threshold=0.4,
            graph_expansion_min_score=0.1,
        )
        engine = self._make_engine(store, pack, lookup, config)

        top_k = [SearchResult(source_file="concepts/auto-build.md", score=0.75, text="", tags=[], graph_expanded=False)]

        with patch.object(engine, '_load_embeddings_for_chunks') as mock_load, \
             patch.object(engine.store, 'get_chunks_by_file_paths') as mock_get_chunks:
            mock_emb = _fake_embedding(4)
            mock_load.return_value = {999: mock_emb}
            # Mock chunk return for neighbor lookup
            mock_get_chunks.return_value = {
                "reference/ref-settings.md": [{
                    "id": 999,
                    "content": "Settings content",
                    "file_path": "reference/ref-settings.md",
                    "type": "reference",
                    "tags": "[]",
                    "chunk_index": 0,
                    "title": "Settings Reference",
                    "prov_id": None,
                    "content_hash": None,
                    "verified_at": None,
                }]
            }
            bonus = engine._apply_graph_expansion(top_k, _fake_embedding(4))

        assert len(bonus) == 1
        assert bonus[0].graph_expanded is True

    def test_structural_bonus_applied(self, store):
        """final_score = cosine_sim * graph_expansion_structural_bonus."""
        chunk_ids = _seed_chunks(store)
        pack = _build_pack_with_graph(NODES_RAW, EDGES)
        lookup = GraphLookup.from_raw_nodes(NODES_RAW)

        config = RetrievalConfig(
            graph_expansion_enabled=True,
            graph_expansion_confidence_threshold=0.4,
            graph_expansion_min_score=0.0,
            graph_expansion_structural_bonus=1.15,
        )
        engine = self._make_engine(store, pack, lookup, config)

        top_k = [SearchResult(source_file="concepts/auto-build.md", score=0.8, text="", tags=[], graph_expanded=False)]

        query_emb = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        neighbor_emb = np.array([0.8, 0.6, 0.0, 0.0], dtype=np.float32)
        neighbor_emb /= np.linalg.norm(neighbor_emb)
        cosine_sim = float(np.dot(query_emb, neighbor_emb))

        with patch.object(engine, '_load_embeddings_for_chunks') as mock_load, \
             patch.object(engine.store, 'get_chunks_by_file_paths') as mock_get_chunks:
            mock_load.return_value = {999: neighbor_emb.tolist()}
            mock_get_chunks.return_value = {
                "reference/ref-settings.md": [{
                    "id": 999,
                    "content": "Settings content",
                    "file_path": "reference/ref-settings.md",
                    "type": "reference",
                    "tags": "[]",
                    "chunk_index": 0,
                    "title": "Settings Reference",
                    "prov_id": None,
                    "content_hash": None,
                    "verified_at": None,
                }]
            }
            bonus = engine._apply_graph_expansion(top_k, query_emb.tolist())

        assert len(bonus) == 1
        expected_score = cosine_sim * 1.15
        assert pytest.approx(bonus[0].score, 0.01) == round(expected_score, 4)

    def test_noop_when_graph_none_or_disabled(self, store):
        """Should return empty list (no-op) when graph is None or disabled."""
        chunk_ids = _seed_chunks(store)
        pack_no_graph = _build_pack_no_graph()
        lookup = GraphLookup.from_raw_nodes(NODES_RAW)

        top_k = [SearchResult(source_file="seed.md", score=0.9, text="", tags=[], graph_expanded=False)]
        query_emb = _fake_embedding(4)

        # Disabled
        config_disabled = RetrievalConfig(graph_expansion_enabled=False)
        engine_disabled = self._make_engine(store, pack_no_graph, lookup, config_disabled)
        assert len(engine_disabled._apply_graph_expansion(top_k, query_emb)) == 0

        # No graph
        config_enabled = RetrievalConfig(graph_expansion_enabled=True)
        engine_no_graph = self._make_engine(store, pack_no_graph, None, config_enabled)
        assert len(engine_no_graph._apply_graph_expansion(top_k, query_emb)) == 0

    # Legacy test names updated to reflect new behavior; old multi-hop/discount tests removed
    # as they no longer apply to the post-top-K additive design (1-hop, independent scoring).
