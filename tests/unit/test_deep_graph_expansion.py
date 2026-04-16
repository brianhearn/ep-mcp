"""Unit tests for deep (multi-hop BFS) graph expansion."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch

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
from ep_mcp.retrieval.models import SearchResult


def _fake_embedding(dim: int = 4) -> list[float]:
    vec = np.random.rand(dim).astype(np.float32)
    vec /= np.linalg.norm(vec)
    return vec.tolist()


def _high_sim_embedding(query_emb: list[float], sim: float = 0.9, dim: int = 4) -> list[float]:
    """Create an embedding with approximately `sim` cosine similarity to query_emb."""
    q = np.array(query_emb, dtype=np.float32)
    # Random orthogonal component
    rand = np.random.rand(dim).astype(np.float32)
    rand -= np.dot(rand, q) * q / (np.dot(q, q) + 1e-8)
    rand /= (np.linalg.norm(rand) + 1e-8)
    # Blend for target similarity
    vec = sim * q + np.sqrt(1 - sim**2) * rand
    vec /= (np.linalg.norm(vec) + 1e-8)
    return vec.tolist()


def _build_chain_pack(n: int = 5) -> tuple[Pack, GraphLookup, list[str]]:
    """Build a linear chain graph: A -> B -> C -> D -> E.

    Returns (pack, lookup, file_paths).
    """
    files = [f"node-{i}.md" for i in range(n)]
    node_ids = [f"pack/{f[:-3]}" for f in files]

    nodes_raw = [
        {"id": nid, "title": f"Node {i}", "type": "concept", "file": fp}
        for i, (nid, fp) in enumerate(zip(node_ids, files))
    ]

    edges = [
        GraphEdge(source=node_ids[i], target=node_ids[i + 1], kind="related")
        for i in range(n - 1)
    ]

    graph = PackGraph(nodes=[str(n) for n in nodes_raw], edges=edges)
    pack = Pack(
        slug="pack",
        name="Test Pack",
        type="product",
        version="1.0.0",
        files={},
        manifest=Manifest(slug="pack", name="Test Pack", type="product", context=ContextTiers()),
        graph=graph,
        pack_dir="",
    )
    lookup = GraphLookup.from_raw_nodes(nodes_raw)
    return pack, lookup, files


@pytest.fixture
def store():
    with tempfile.TemporaryDirectory() as d:
        db_path = str(Path(d) / "test.db")
        s = SQLiteStore(db_path, embedding_dimension=4)
        s.open()
        yield s
        s.close()


def _seed_chain(store: SQLiteStore, files: list[str]) -> dict[str, int]:
    result = {}
    for fp in files:
        cid = store.upsert_chunk(
            file_path=fp,
            chunk_index=0,
            content=f"Content of {fp}",
            title=fp.replace(".md", ""),
            type_="concept",
            tags=[],
            pack_slug="pack",
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


def _make_engine(
    store: SQLiteStore,
    pack: Pack,
    lookup: GraphLookup,
    config: RetrievalConfig,
) -> RetrievalEngine:
    provider = AsyncMock()
    provider.embed_query = AsyncMock(return_value=_fake_embedding())
    provider.dimension = 4
    return RetrievalEngine(
        pack=pack, store=store, embedding_provider=provider,
        config=config, graph_lookup=lookup,
    )


class TestDeepGraphExpansion:
    """Tests for multi-hop BFS graph expansion."""

    def test_deep_reaches_2_hops(self, store):
        """With depth=2 on a chain A->B->C->D->E, seed=A should reach B and C."""
        pack, lookup, files = _build_chain_pack(5)
        chunk_ids = _seed_chain(store, files)

        config = RetrievalConfig(
            graph_expansion_enabled=True,
            graph_expansion_deep=True,
            graph_expansion_depth=2,
            graph_expansion_confidence_threshold=0.3,
            graph_expansion_min_score=0.0,  # accept everything
            graph_expansion_structural_bonus=1.0,
            graph_expansion_discount=0.85,
            graph_expansion_deep_max_bonus=10,
        )
        engine = _make_engine(store, pack, lookup, config)

        seed = SearchResult(
            text="Seed", source_file="node-0.md", score=0.8,
            tags=[], graph_expanded=False,
        )

        query_emb = _fake_embedding(4)
        with patch.object(engine, '_load_embeddings_for_chunks') as mock_load:
            # Return valid embeddings for any chunk
            def mock_embs(chunk_ids):
                return {cid: _high_sim_embedding(query_emb, 0.7) for cid in chunk_ids}
            mock_load.side_effect = mock_embs

            bonus = engine._apply_graph_expansion([seed], query_emb)

        bonus_files = {r.source_file for r in bonus}
        assert "node-1.md" in bonus_files, "Hop 1 neighbor missing"
        assert "node-2.md" in bonus_files, "Hop 2 neighbor missing"
        # node-3.md is 3 hops away — should NOT be reached at depth=2
        assert "node-3.md" not in bonus_files

    def test_deep_applies_per_hop_discount(self, store):
        """Hop-2 neighbors get discount^2 applied to their structural bonus."""
        pack, lookup, files = _build_chain_pack(4)
        chunk_ids = _seed_chain(store, files)

        config = RetrievalConfig(
            graph_expansion_enabled=True,
            graph_expansion_deep=True,
            graph_expansion_depth=2,
            graph_expansion_confidence_threshold=0.3,
            graph_expansion_min_score=0.0,
            graph_expansion_structural_bonus=1.0,
            graph_expansion_discount=0.80,
            graph_expansion_deep_max_bonus=10,
        )
        engine = _make_engine(store, pack, lookup, config)

        seed = SearchResult(
            text="Seed", source_file="node-0.md", score=0.8,
            tags=[], graph_expanded=False,
        )

        # Use a known embedding so we can predict cosine sim
        query_emb = [1.0, 0.0, 0.0, 0.0]
        # All neighbors get the same embedding → same base cosine sim
        neighbor_emb = np.array([0.8, 0.6, 0.0, 0.0], dtype=np.float32)
        neighbor_emb /= np.linalg.norm(neighbor_emb)
        base_cosine = float(np.dot(query_emb, neighbor_emb.tolist()))

        with patch.object(engine, '_load_embeddings_for_chunks') as mock_load:
            mock_load.return_value = {}  # will be called per-chunk

            def mock_embs(chunk_ids):
                return {cid: neighbor_emb.tolist() for cid in chunk_ids}
            mock_load.side_effect = mock_embs

            bonus = engine._apply_graph_expansion([seed], query_emb)

        hop1 = next(r for r in bonus if r.source_file == "node-1.md")
        hop2 = next(r for r in bonus if r.source_file == "node-2.md")

        expected_hop1 = round(base_cosine * 1.0 * 0.80, 4)
        expected_hop2 = round(base_cosine * 1.0 * (0.80 ** 2), 4)

        assert hop1.score == expected_hop1
        assert hop2.score == expected_hop2

    def test_deep_max_bonus_cap(self, store):
        """Deep expansion respects graph_expansion_deep_max_bonus cap."""
        pack, lookup, files = _build_chain_pack(10)
        chunk_ids = _seed_chain(store, files)

        config = RetrievalConfig(
            graph_expansion_enabled=True,
            graph_expansion_deep=True,
            graph_expansion_depth=8,
            graph_expansion_confidence_threshold=0.3,
            graph_expansion_min_score=0.0,
            graph_expansion_structural_bonus=1.0,
            graph_expansion_discount=0.95,
            graph_expansion_deep_max_bonus=3,
        )
        engine = _make_engine(store, pack, lookup, config)

        seed = SearchResult(
            text="Seed", source_file="node-0.md", score=0.8,
            tags=[], graph_expanded=False,
        )
        query_emb = _fake_embedding(4)

        with patch.object(engine, '_load_embeddings_for_chunks') as mock_load:
            def mock_embs(chunk_ids):
                return {cid: _high_sim_embedding(query_emb, 0.7) for cid in chunk_ids}
            mock_load.side_effect = mock_embs

            bonus = engine._apply_graph_expansion([seed], query_emb)

        assert len(bonus) <= 3

    def test_deep_stops_when_frontier_dries_up(self, store):
        """BFS stops early if no neighbors qualify for the next frontier."""
        pack, lookup, files = _build_chain_pack(5)
        chunk_ids = _seed_chain(store, files)

        config = RetrievalConfig(
            graph_expansion_enabled=True,
            graph_expansion_deep=True,
            graph_expansion_depth=4,
            graph_expansion_confidence_threshold=0.3,
            graph_expansion_min_score=0.6,  # high bar
            graph_expansion_structural_bonus=1.0,
            graph_expansion_discount=0.5,  # aggressive decay — hop2 = 0.25 bonus
            graph_expansion_deep_max_bonus=10,
        )
        engine = _make_engine(store, pack, lookup, config)

        seed = SearchResult(
            text="Seed", source_file="node-0.md", score=0.8,
            tags=[], graph_expanded=False,
        )
        query_emb = _fake_embedding(4)

        with patch.object(engine, '_load_embeddings_for_chunks') as mock_load:
            # Give low-ish cosine sims so that with aggressive discount,
            # hop2+ won't clear min_score
            def mock_embs(chunk_ids):
                return {cid: _high_sim_embedding(query_emb, 0.65) for cid in chunk_ids}
            mock_load.side_effect = mock_embs

            bonus = engine._apply_graph_expansion([seed], query_emb)

        # At hop1 (discount=0.5): score = 0.65 * 0.5 = 0.325 < 0.6 → rejected
        # Nothing should pass at all with these numbers
        assert len(bonus) == 0

    def test_deep_no_revisit(self, store):
        """BFS must not revisit nodes already seen (from top-K or earlier hops)."""
        pack, lookup, files = _build_chain_pack(4)
        chunk_ids = _seed_chain(store, files)

        config = RetrievalConfig(
            graph_expansion_enabled=True,
            graph_expansion_deep=True,
            graph_expansion_depth=3,
            graph_expansion_confidence_threshold=0.3,
            graph_expansion_min_score=0.0,
            graph_expansion_structural_bonus=1.0,
            graph_expansion_discount=0.9,
            graph_expansion_deep_max_bonus=10,
        )
        engine = _make_engine(store, pack, lookup, config)

        # node-0 and node-2 already in top-K
        top_k = [
            SearchResult(text="Seed", source_file="node-0.md", score=0.8,
                         tags=[], graph_expanded=False),
            SearchResult(text="Already", source_file="node-2.md", score=0.6,
                         tags=[], graph_expanded=False),
        ]
        query_emb = _fake_embedding(4)

        with patch.object(engine, '_load_embeddings_for_chunks') as mock_load:
            def mock_embs(chunk_ids):
                return {cid: _high_sim_embedding(query_emb, 0.7) for cid in chunk_ids}
            mock_load.side_effect = mock_embs

            bonus = engine._apply_graph_expansion(top_k, query_emb)

        bonus_files = {r.source_file for r in bonus}
        # node-2 is in top-K — must not appear in bonus
        assert "node-2.md" not in bonus_files
        # node-1 is 1-hop from node-0, should be found
        assert "node-1.md" in bonus_files

    def test_deep_false_uses_shallow(self, store):
        """With graph_expansion_deep=False, should use shallow (1-hop) expansion."""
        pack, lookup, files = _build_chain_pack(4)
        chunk_ids = _seed_chain(store, files)

        config = RetrievalConfig(
            graph_expansion_enabled=True,
            graph_expansion_deep=False,  # shallow
            graph_expansion_depth=3,  # should be ignored
            graph_expansion_confidence_threshold=0.3,
            graph_expansion_min_score=0.0,
            graph_expansion_structural_bonus=1.0,
            graph_expansion_deep_max_bonus=10,
        )
        engine = _make_engine(store, pack, lookup, config)

        seed = SearchResult(
            text="Seed", source_file="node-0.md", score=0.8,
            tags=[], graph_expanded=False,
        )
        query_emb = _fake_embedding(4)

        with patch.object(engine, '_load_embeddings_for_chunks') as mock_load:
            def mock_embs(chunk_ids):
                return {cid: _high_sim_embedding(query_emb, 0.7) for cid in chunk_ids}
            mock_load.side_effect = mock_embs

            bonus = engine._apply_graph_expansion([seed], query_emb)

        bonus_files = {r.source_file for r in bonus}
        # Shallow: only 1-hop neighbor
        assert "node-1.md" in bonus_files
        # 2-hop should NOT be reached in shallow mode
        assert "node-2.md" not in bonus_files

    def test_all_bonus_flagged_graph_expanded(self, store):
        """Every deep expansion result must have graph_expanded=True."""
        pack, lookup, files = _build_chain_pack(4)
        chunk_ids = _seed_chain(store, files)

        config = RetrievalConfig(
            graph_expansion_enabled=True,
            graph_expansion_deep=True,
            graph_expansion_depth=3,
            graph_expansion_confidence_threshold=0.3,
            graph_expansion_min_score=0.0,
            graph_expansion_structural_bonus=1.0,
            graph_expansion_discount=0.9,
            graph_expansion_deep_max_bonus=10,
        )
        engine = _make_engine(store, pack, lookup, config)

        seed = SearchResult(
            text="Seed", source_file="node-0.md", score=0.8,
            tags=[], graph_expanded=False,
        )
        query_emb = _fake_embedding(4)

        with patch.object(engine, '_load_embeddings_for_chunks') as mock_load:
            def mock_embs(chunk_ids):
                return {cid: _high_sim_embedding(query_emb, 0.7) for cid in chunk_ids}
            mock_load.side_effect = mock_embs

            bonus = engine._apply_graph_expansion([seed], query_emb)

        for r in bonus:
            assert r.graph_expanded is True
