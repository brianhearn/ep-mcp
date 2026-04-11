"""Tests for SQLite FTS5 + sqlite-vec storage layer."""

import tempfile
from pathlib import Path

import pytest

from ep_mcp.index.sqlite_store import SQLiteStore


@pytest.fixture
def store():
    """Create a temporary SQLite store."""
    with tempfile.TemporaryDirectory() as d:
        db_path = str(Path(d) / "test.db")
        s = SQLiteStore(db_path, embedding_dimension=4)
        s.open()
        yield s
        s.close()


def _fake_embedding(dim: int = 4) -> list[float]:
    """Generate a simple test embedding."""
    return [0.1 * i for i in range(dim)]


class TestSQLiteStore:
    def test_create_schema(self, store):
        """Schema should be created on open."""
        assert store.chunk_count() == 0
        assert store.file_count() == 0

    def test_upsert_chunk(self, store):
        chunk_id = store.upsert_chunk(
            file_path="test.md",
            chunk_index=0,
            content="Test content",
            title="Test",
            type_="concept",
            tags=["test"],
            pack_slug="test-pack",
            prov_id="test-pack/test",
            content_hash="abc123",
            verified_at="2026-04-10",
            verified_by="human",
            token_count=10,
            embedding=_fake_embedding(),
        )
        store.commit()
        assert chunk_id > 0
        assert store.chunk_count() == 1

    def test_upsert_updates_existing(self, store):
        # Insert
        store.upsert_chunk(
            file_path="test.md", chunk_index=0, content="Original",
            title="Test", type_="concept", tags=[], pack_slug="tp",
            prov_id=None, content_hash="hash1", verified_at=None,
            verified_by=None, token_count=5, embedding=_fake_embedding(),
        )
        store.commit()
        assert store.chunk_count() == 1

        # Update same file/chunk
        store.upsert_chunk(
            file_path="test.md", chunk_index=0, content="Updated",
            title="Test", type_="concept", tags=[], pack_slug="tp",
            prov_id=None, content_hash="hash2", verified_at=None,
            verified_by=None, token_count=5, embedding=_fake_embedding(),
        )
        store.commit()
        assert store.chunk_count() == 1  # Still 1, not 2

        chunk = store.get_chunk_by_id(1)
        assert chunk["content"] == "Updated"
        assert chunk["content_hash"] == "hash2"

    def test_delete_file_chunks(self, store):
        store.upsert_chunk(
            file_path="a.md", chunk_index=0, content="A",
            title="A", type_=None, tags=[], pack_slug="tp",
            prov_id=None, content_hash="h1", verified_at=None,
            verified_by=None, token_count=1, embedding=_fake_embedding(),
        )
        store.upsert_chunk(
            file_path="b.md", chunk_index=0, content="B",
            title="B", type_=None, tags=[], pack_slug="tp",
            prov_id=None, content_hash="h2", verified_at=None,
            verified_by=None, token_count=1, embedding=_fake_embedding(),
        )
        store.commit()
        assert store.file_count() == 2

        deleted = store.delete_file_chunks("a.md")
        store.commit()
        assert deleted == 1
        assert store.file_count() == 1

    def test_get_file_hashes(self, store):
        store.upsert_chunk(
            file_path="a.md", chunk_index=0, content="A",
            title="A", type_=None, tags=[], pack_slug="tp",
            prov_id=None, content_hash="hash_a", verified_at=None,
            verified_by=None, token_count=1,
        )
        store.upsert_chunk(
            file_path="b.md", chunk_index=0, content="B",
            title="B", type_=None, tags=[], pack_slug="tp",
            prov_id=None, content_hash="hash_b", verified_at=None,
            verified_by=None, token_count=1,
        )
        store.commit()

        hashes = store.get_file_hashes()
        assert hashes == {"a.md": "hash_a", "b.md": "hash_b"}

    def test_bm25_search(self, store):
        store.upsert_chunk(
            file_path="apple.md", chunk_index=0,
            content="Apples are a delicious fruit",
            title="Apples", type_=None, tags=[], pack_slug="tp",
            prov_id=None, content_hash="h1", verified_at=None,
            verified_by=None, token_count=6,
        )
        store.upsert_chunk(
            file_path="banana.md", chunk_index=0,
            content="Bananas are yellow and tasty",
            title="Bananas", type_=None, tags=[], pack_slug="tp",
            prov_id=None, content_hash="h2", verified_at=None,
            verified_by=None, token_count=6,
        )
        store.commit()

        results = store.bm25_search("apple fruit", limit=10)
        assert len(results) >= 1
        # Apple doc should be first (matches both terms)
        first_chunk = store.get_chunk_by_id(results[0]["chunk_id"])
        assert first_chunk["file_path"] == "apple.md"

    def test_vector_search(self, store):
        store.upsert_chunk(
            file_path="a.md", chunk_index=0, content="A",
            title="A", type_=None, tags=[], pack_slug="tp",
            prov_id=None, content_hash="h1", verified_at=None,
            verified_by=None, token_count=1,
            embedding=[1.0, 0.0, 0.0, 0.0],
        )
        store.upsert_chunk(
            file_path="b.md", chunk_index=0, content="B",
            title="B", type_=None, tags=[], pack_slug="tp",
            prov_id=None, content_hash="h2", verified_at=None,
            verified_by=None, token_count=1,
            embedding=[0.0, 1.0, 0.0, 0.0],
        )
        store.commit()

        # Search for something close to A
        results = store.vector_search([0.9, 0.1, 0.0, 0.0], limit=2)
        assert len(results) == 2
        # A should be closer
        first = store.get_chunk_by_id(results[0]["chunk_id"])
        assert first["file_path"] == "a.md"

    def test_embedding_cache(self, store):
        store.cache_embedding("hash1", "gemini/test", [0.1, 0.2, 0.3, 0.4])
        store.commit()

        cached = store.get_cached_embedding("hash1", "gemini/test")
        assert cached is not None
        assert len(cached) == 4
        assert abs(cached[0] - 0.1) < 0.001

        # Different model = no cache hit
        assert store.get_cached_embedding("hash1", "openai/test") is None

    def test_invalidate_cache(self, store):
        store.cache_embedding("h1", "model-a", [0.1, 0.2, 0.3, 0.4])
        store.cache_embedding("h2", "model-a", [0.5, 0.6, 0.7, 0.8])
        store.cache_embedding("h3", "model-b", [0.1, 0.2, 0.3, 0.4])
        store.commit()

        removed = store.invalidate_cache_for_model("model-a")
        store.commit()
        assert removed == 2
        assert store.get_cached_embedding("h1", "model-a") is None
        assert store.get_cached_embedding("h3", "model-b") is not None

    def test_metadata(self, store):
        store.set_meta("version", "1.0")
        store.set_meta("model", "gemini")
        store.commit()

        assert store.get_meta("version") == "1.0"
        assert store.get_meta("model") == "gemini"
        assert store.get_meta("nonexistent") is None

        # Overwrite
        store.set_meta("version", "2.0")
        store.commit()
        assert store.get_meta("version") == "2.0"

    def test_get_chunks_by_ids(self, store):
        id1 = store.upsert_chunk(
            file_path="a.md", chunk_index=0, content="A",
            title="A", type_=None, tags=[], pack_slug="tp",
            prov_id=None, content_hash="h1", verified_at=None,
            verified_by=None, token_count=1,
        )
        id2 = store.upsert_chunk(
            file_path="b.md", chunk_index=0, content="B",
            title="B", type_=None, tags=[], pack_slug="tp",
            prov_id=None, content_hash="h2", verified_at=None,
            verified_by=None, token_count=1,
        )
        store.commit()

        chunks = store.get_chunks_by_ids([id1, id2])
        assert len(chunks) == 2
        assert chunks[id1]["content"] == "A"
        assert chunks[id2]["content"] == "B"

    def test_get_chunks_by_ids_empty(self, store):
        assert store.get_chunks_by_ids([]) == {}
