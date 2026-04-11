"""SQLite FTS5 + sqlite-vec storage layer for pack indexes."""

from __future__ import annotations

import json
import logging
import sqlite3
import struct
from pathlib import Path

import sqlite_vec

logger = logging.getLogger(__name__)


class SQLiteStore:
    """Manages a SQLite database with FTS5 and sqlite-vec for a single pack.

    Schema matches ARCHITECTURE.md §4.1:
    - chunks: content + metadata
    - chunks_fts: FTS5 virtual table for BM25
    - chunks_vec: sqlite-vec virtual table for vector search
    - embedding_cache: content_hash → embedding vector
    - index_meta: key-value metadata
    """

    def __init__(self, db_path: str, embedding_dimension: int = 768):
        self.db_path = db_path
        self.embedding_dimension = embedding_dimension
        self._conn: sqlite3.Connection | None = None

    def open(self) -> None:
        """Open the database connection and initialize schema."""
        # Ensure parent directory exists
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        self._conn = sqlite3.connect(self.db_path)
        self._conn.row_factory = sqlite3.Row

        # Load sqlite-vec extension
        self._conn.enable_load_extension(True)
        sqlite_vec.load(self._conn)
        self._conn.enable_load_extension(False)

        # Enable WAL mode for concurrent reads
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")

        self._create_schema()

    def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RuntimeError("Database not opened — call open() first")
        return self._conn

    def _create_schema(self) -> None:
        """Create tables if they don't exist."""
        c = self.conn
        c.executescript(f"""
            -- Content chunks
            CREATE TABLE IF NOT EXISTS chunks (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path   TEXT NOT NULL,
                chunk_index INTEGER NOT NULL DEFAULT 0,
                content     TEXT NOT NULL,
                title       TEXT,
                type        TEXT,
                tags        TEXT,
                pack_slug   TEXT NOT NULL,
                prov_id     TEXT,
                content_hash TEXT,
                verified_at TEXT,
                verified_by TEXT,
                token_count INTEGER,
                UNIQUE(file_path, chunk_index)
            );

            -- Index metadata
            CREATE TABLE IF NOT EXISTS index_meta (
                key   TEXT PRIMARY KEY,
                value TEXT
            );

            -- Embedding cache
            CREATE TABLE IF NOT EXISTS embedding_cache (
                content_hash TEXT PRIMARY KEY,
                model_name   TEXT NOT NULL,
                embedding    BLOB NOT NULL
            );
        """)

        # FTS5 — create only if not exists (can't use IF NOT EXISTS with virtual tables easily)
        try:
            c.execute("""
                CREATE VIRTUAL TABLE chunks_fts USING fts5(
                    content,
                    title,
                    content='chunks',
                    content_rowid='id',
                    tokenize='porter unicode61'
                )
            """)
        except sqlite3.OperationalError:
            pass  # Already exists

        # FTS sync triggers
        for trigger_sql in [
            """CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
                INSERT INTO chunks_fts(rowid, content, title)
                VALUES (new.id, new.content, new.title);
            END""",
            """CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
                INSERT INTO chunks_fts(chunks_fts, rowid, content, title)
                VALUES ('delete', old.id, old.content, old.title);
            END""",
            """CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
                INSERT INTO chunks_fts(chunks_fts, rowid, content, title)
                VALUES ('delete', old.id, old.content, old.title);
                INSERT INTO chunks_fts(rowid, content, title)
                VALUES (new.id, new.content, new.title);
            END""",
        ]:
            try:
                c.execute(trigger_sql)
            except sqlite3.OperationalError:
                pass

        # sqlite-vec virtual table
        try:
            c.execute(f"""
                CREATE VIRTUAL TABLE chunks_vec USING vec0(
                    chunk_id INTEGER PRIMARY KEY,
                    embedding FLOAT[{self.embedding_dimension}]
                )
            """)
        except sqlite3.OperationalError:
            pass  # Already exists

        c.commit()

    # ── Chunk operations ──

    def upsert_chunk(
        self,
        file_path: str,
        chunk_index: int,
        content: str,
        title: str | None,
        type_: str | None,
        tags: list[str],
        pack_slug: str,
        prov_id: str | None,
        content_hash: str | None,
        verified_at: str | None,
        verified_by: str | None,
        token_count: int,
        embedding: list[float] | None = None,
    ) -> int:
        """Insert or update a chunk. Returns the chunk row id."""
        c = self.conn
        tags_json = json.dumps(tags) if tags else "[]"

        # Check if chunk exists
        row = c.execute(
            "SELECT id FROM chunks WHERE file_path = ? AND chunk_index = ?",
            (file_path, chunk_index),
        ).fetchone()

        if row:
            chunk_id = row["id"]
            c.execute(
                """UPDATE chunks SET
                    content = ?, title = ?, type = ?, tags = ?,
                    pack_slug = ?, prov_id = ?, content_hash = ?,
                    verified_at = ?, verified_by = ?, token_count = ?
                WHERE id = ?""",
                (content, title, type_, tags_json, pack_slug, prov_id,
                 content_hash, verified_at, verified_by, token_count, chunk_id),
            )
            # Update vector
            if embedding:
                c.execute("DELETE FROM chunks_vec WHERE chunk_id = ?", (chunk_id,))
                c.execute(
                    "INSERT INTO chunks_vec (chunk_id, embedding) VALUES (?, ?)",
                    (chunk_id, _serialize_f32(embedding)),
                )
        else:
            cursor = c.execute(
                """INSERT INTO chunks
                    (file_path, chunk_index, content, title, type, tags,
                     pack_slug, prov_id, content_hash, verified_at, verified_by, token_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (file_path, chunk_index, content, title, type_, tags_json,
                 pack_slug, prov_id, content_hash, verified_at, verified_by, token_count),
            )
            chunk_id = cursor.lastrowid
            # Insert vector
            if embedding:
                c.execute(
                    "INSERT INTO chunks_vec (chunk_id, embedding) VALUES (?, ?)",
                    (chunk_id, _serialize_f32(embedding)),
                )

        return chunk_id

    def delete_file_chunks(self, file_path: str) -> int:
        """Delete all chunks for a file. Returns count deleted."""
        c = self.conn
        # Get chunk IDs for vector cleanup
        rows = c.execute(
            "SELECT id FROM chunks WHERE file_path = ?", (file_path,)
        ).fetchall()
        chunk_ids = [r["id"] for r in rows]

        if chunk_ids:
            placeholders = ",".join("?" * len(chunk_ids))
            c.execute(f"DELETE FROM chunks_vec WHERE chunk_id IN ({placeholders})", chunk_ids)
            c.execute(f"DELETE FROM chunks WHERE id IN ({placeholders})", chunk_ids)

        return len(chunk_ids)

    def get_file_hashes(self) -> dict[str, str]:
        """Get content_hash for all indexed files. Returns {file_path: content_hash}."""
        rows = self.conn.execute(
            "SELECT DISTINCT file_path, content_hash FROM chunks WHERE chunk_index = 0"
        ).fetchall()
        return {r["file_path"]: r["content_hash"] for r in rows}

    def get_indexed_files(self) -> set[str]:
        """Get set of all indexed file paths."""
        rows = self.conn.execute(
            "SELECT DISTINCT file_path FROM chunks"
        ).fetchall()
        return {r["file_path"] for r in rows}

    # ── Search operations ──

    def vector_search(
        self, query_embedding: list[float], limit: int = 40
    ) -> list[dict]:
        """Find chunks by vector similarity.

        Returns list of dicts with chunk_id, distance.
        Lower distance = more similar.
        """
        rows = self.conn.execute(
            """SELECT chunk_id, distance
            FROM chunks_vec
            WHERE embedding MATCH ?
            ORDER BY distance
            LIMIT ?""",
            (_serialize_f32(query_embedding), limit),
        ).fetchall()
        return [{"chunk_id": r["chunk_id"], "distance": r["distance"]} for r in rows]

    def bm25_search(self, query: str, limit: int = 40) -> list[dict]:
        """Find chunks by BM25 text search.

        Returns list of dicts with chunk_id, bm25_score.
        More negative = more relevant (FTS5 convention).
        """
        # Sanitize query for FTS5: quote each token to prevent syntax errors
        safe_query = _sanitize_fts5_query(query)
        if not safe_query:
            return []
        rows = self.conn.execute(
            """SELECT chunks.id as chunk_id, chunks_fts.rank as bm25_score
            FROM chunks_fts
            JOIN chunks ON chunks.id = chunks_fts.rowid
            WHERE chunks_fts MATCH ?
            ORDER BY chunks_fts.rank
            LIMIT ?""",
            (safe_query, limit),
        ).fetchall()
        return [{"chunk_id": r["chunk_id"], "bm25_score": r["bm25_score"]} for r in rows]

    def get_chunk_by_id(self, chunk_id: int) -> dict | None:
        """Get full chunk data by ID."""
        row = self.conn.execute(
            "SELECT * FROM chunks WHERE id = ?", (chunk_id,)
        ).fetchone()
        if not row:
            return None
        return dict(row)

    def get_chunks_by_ids(self, chunk_ids: list[int]) -> dict[int, dict]:
        """Get multiple chunks by ID. Returns {chunk_id: chunk_data}."""
        if not chunk_ids:
            return {}
        placeholders = ",".join("?" * len(chunk_ids))
        rows = self.conn.execute(
            f"SELECT * FROM chunks WHERE id IN ({placeholders})", chunk_ids
        ).fetchall()
        return {r["id"]: dict(r) for r in rows}

    # ── Embedding cache ──

    def get_cached_embedding(
        self, content_hash: str, model_name: str
    ) -> list[float] | None:
        """Get a cached embedding by content hash and model."""
        row = self.conn.execute(
            "SELECT embedding FROM embedding_cache WHERE content_hash = ? AND model_name = ?",
            (content_hash, model_name),
        ).fetchone()
        if not row:
            return None
        return _deserialize_f32(row["embedding"])

    def cache_embedding(
        self, content_hash: str, model_name: str, embedding: list[float]
    ) -> None:
        """Cache an embedding."""
        self.conn.execute(
            """INSERT OR REPLACE INTO embedding_cache (content_hash, model_name, embedding)
            VALUES (?, ?, ?)""",
            (content_hash, model_name, _serialize_f32(embedding)),
        )

    def invalidate_cache_for_model(self, model_name: str) -> int:
        """Remove all cached embeddings for a specific model. Returns count removed."""
        cursor = self.conn.execute(
            "DELETE FROM embedding_cache WHERE model_name = ?", (model_name,)
        )
        return cursor.rowcount

    # ── Metadata ──

    def get_meta(self, key: str) -> str | None:
        """Get a metadata value."""
        row = self.conn.execute(
            "SELECT value FROM index_meta WHERE key = ?", (key,)
        ).fetchone()
        return row["value"] if row else None

    def set_meta(self, key: str, value: str) -> None:
        """Set a metadata value."""
        self.conn.execute(
            "INSERT OR REPLACE INTO index_meta (key, value) VALUES (?, ?)",
            (key, value),
        )

    # ── Stats ──

    def chunk_count(self) -> int:
        """Total number of chunks in the index."""
        row = self.conn.execute("SELECT COUNT(*) as cnt FROM chunks").fetchone()
        return row["cnt"]

    def file_count(self) -> int:
        """Number of unique files in the index."""
        row = self.conn.execute(
            "SELECT COUNT(DISTINCT file_path) as cnt FROM chunks"
        ).fetchone()
        return row["cnt"]

    def commit(self) -> None:
        """Commit pending changes."""
        self.conn.commit()


import re as _re

# FTS5 special characters that need escaping
_FTS5_SPECIAL = _re.compile(r'[^\w\s]', _re.UNICODE)


def _sanitize_fts5_query(query: str) -> str:
    """Sanitize a natural language query for FTS5 MATCH.

    Strips punctuation and wraps each token in quotes to prevent
    FTS5 syntax errors from characters like ?, *, (, etc.
    """
    # Remove special characters
    clean = _FTS5_SPECIAL.sub(' ', query)
    # Split into tokens, quote each one
    tokens = [t.strip() for t in clean.split() if t.strip()]
    if not tokens:
        return ''
    # Join with implicit AND (space-separated quoted tokens)
    return ' '.join(f'"{t}"' for t in tokens)


def _serialize_f32(vec: list[float]) -> bytes:
    """Serialize a float list to raw bytes for sqlite-vec."""
    return struct.pack(f"{len(vec)}f", *vec)


def _deserialize_f32(blob: bytes) -> list[float]:
    """Deserialize raw bytes back to float list."""
    n = len(blob) // 4
    return list(struct.unpack(f"{n}f", blob))
