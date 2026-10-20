"""Integration test: index a real ExpertPack with Gemini embeddings.

Requires:
- GEMINI_API_KEY env var set
- A real pack at the specified path

Run with:
    pytest tests/integration/test_index_real_pack.py -v --pack /path/to/pack
"""

from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path

import pytest

from ep_mcp.embeddings.gemini import GeminiEmbeddingProvider
from ep_mcp.index.manager import IndexManager
from ep_mcp.index.sqlite_store import SQLiteStore
from ep_mcp.pack.loader import load_pack



@pytest.fixture
def pack_path(request):
    path = request.config.getoption("--pack")
    if not path:
        pytest.skip("No --pack path provided")
    if not Path(path).is_dir():
        pytest.skip(f"Pack path not found: {path}")
    return path


@pytest.fixture
def gemini_provider():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        pytest.skip("GEMINI_API_KEY not set")
    return GeminiEmbeddingProvider(api_key=api_key)


class TestIndexRealPack:
    def test_full_index(self, pack_path, gemini_provider):
        """Load a real pack, index it with Gemini, verify search works."""
        pack = load_pack(pack_path)
        print(f"\nLoaded: {pack.name} ({len(pack.files)} files)")

        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "test_index.db")
            store = SQLiteStore(db_path, embedding_dimension=gemini_provider.dimension)
            store.open()

            try:
                manager = IndexManager(pack, store, gemini_provider)
                stats = asyncio.get_event_loop().run_until_complete(manager.build_index())

                print(f"Index stats: {stats}")
                assert stats.total_chunks > 0
                # 0.5.0+: navigation / on_demand files are not indexed
                assert stats.new_files == sum(1 for f in pack.files.values() if f.is_indexable)
                assert stats.cache_misses > 0

                # Verify search works
                query_emb = asyncio.get_event_loop().run_until_complete(
                    gemini_provider.embed_query("territory management")
                )

                vec_results = store.vector_search(query_emb, limit=5)
                assert len(vec_results) > 0
                print(f"Vector search returned {len(vec_results)} results")

                # Pack-agnostic lexical probe: a word from the first indexed title.
                probe = next(
                    w for f in pack.files.values() if f.is_indexable and f.title
                    for w in f.title.split() if w.isalpha() and len(w) > 3
                )
                bm25_results = store.bm25_search(probe, limit=5)
                assert len(bm25_results) > 0
                print(f"BM25 search returned {len(bm25_results)} results")

                # Verify metadata was written
                assert store.get_meta("embedding_model") == gemini_provider.model_name
                assert store.get_meta("pack_version") == pack.version

            finally:
                store.close()
