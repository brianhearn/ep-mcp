"""Index lifecycle management: build, update, staleness checks."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from ..embeddings.base import EmbeddingProvider
from ..pack.models import Pack
from .chunker import Chunk, chunk_file
from .sqlite_store import SQLiteStore

logger = logging.getLogger(__name__)


class IndexManager:
    """Manages the indexing pipeline for a single pack.

    Handles:
    - Full index builds (first run or embedding model change)
    - Incremental updates (only re-index changed/new/deleted files)
    - Embedding caching to avoid redundant API calls
    """

    def __init__(
        self,
        pack: Pack,
        store: SQLiteStore,
        embedding_provider: EmbeddingProvider,
    ):
        self.pack = pack
        self.store = store
        self.provider = embedding_provider

    async def build_index(self) -> IndexStats:
        """Build or update the pack index.

        Detects whether a full rebuild or incremental update is needed:
        - Full rebuild: first run, or embedding model changed
        - Incremental: only changed/new/deleted files

        Returns:
            IndexStats with counts of what was done
        """
        stats = IndexStats()

        # Check if we need a full rebuild
        stored_model = self.store.get_meta("embedding_model")
        current_model = self.provider.model_name

        if stored_model and stored_model != current_model:
            logger.info(
                "Embedding model changed: %s → %s — full rebuild required",
                stored_model, current_model,
            )
            stats.full_rebuild = True
            self.store.invalidate_cache_for_model(stored_model)
            # Clear all chunks to trigger full re-index
            for file_path in self.store.get_indexed_files():
                self.store.delete_file_chunks(file_path)
            self.store.commit()

        # Get current index state
        indexed_hashes = self.store.get_file_hashes()
        indexed_files = set(indexed_hashes.keys())
        pack_files = set(self.pack.files.keys())

        # Determine what needs updating
        new_files = pack_files - indexed_files
        deleted_files = indexed_files - pack_files
        existing_files = pack_files & indexed_files

        # Check for changed content in existing files
        changed_files = set()
        for fp in existing_files:
            pack_hash = self.pack.files[fp].provenance.content_hash
            if pack_hash and indexed_hashes.get(fp) != pack_hash:
                changed_files.add(fp)

        files_to_index = new_files | changed_files
        stats.new_files = len(new_files)
        stats.changed_files = len(changed_files)
        stats.deleted_files = len(deleted_files)
        stats.unchanged_files = len(existing_files) - len(changed_files)

        # Delete removed files
        for fp in deleted_files:
            count = self.store.delete_file_chunks(fp)
            logger.debug("Deleted %d chunks for removed file: %s", count, fp)

        # Index new and changed files
        if files_to_index:
            await self._index_files(files_to_index, stats)

        # Update metadata
        self.store.set_meta("embedding_model", current_model)
        self.store.set_meta("embedding_dimension", str(self.provider.dimension))
        self.store.set_meta("pack_version", self.pack.version)
        self.store.set_meta("indexed_at", datetime.now(timezone.utc).isoformat())
        self.store.set_meta("file_count", str(self.store.file_count()))
        self.store.set_meta("chunk_count", str(self.store.chunk_count()))
        self.store.commit()

        logger.info(
            "Index complete for '%s': %d new, %d changed, %d deleted, %d unchanged, %d chunks total",
            self.pack.slug,
            stats.new_files,
            stats.changed_files,
            stats.deleted_files,
            stats.unchanged_files,
            stats.total_chunks,
        )

        return stats

    async def _index_files(self, file_paths: set[str], stats: IndexStats) -> None:
        """Chunk and embed a set of files."""
        # Step 1: Chunk all files
        all_chunks: list[Chunk] = []
        for fp in sorted(file_paths):
            pack_file = self.pack.files[fp]
            # Delete existing chunks for this file (if updating)
            self.store.delete_file_chunks(fp)
            # Chunk the file
            chunks = chunk_file(fp, pack_file.content, pack_file.title)
            all_chunks.extend(chunks)

        stats.total_chunks = self.store.chunk_count()  # existing
        logger.info("Chunked %d files into %d chunks", len(file_paths), len(all_chunks))

        # Step 2: Check embedding cache and identify what needs embedding
        texts_to_embed: list[str] = []
        chunk_embed_map: list[tuple[int, list[float] | None]] = []  # (index_in_all_chunks, cached_or_None)

        for i, chunk in enumerate(all_chunks):
            # Compute content hash for this chunk's content
            import hashlib
            chunk_hash = hashlib.sha256(chunk.content.encode("utf-8")).hexdigest()

            cached = self.store.get_cached_embedding(chunk_hash, self.provider.model_name)
            if cached:
                chunk_embed_map.append((i, cached))
                stats.cache_hits += 1
            else:
                chunk_embed_map.append((i, None))
                texts_to_embed.append(chunk.content)
                stats.cache_misses += 1

        # Step 3: Embed uncached texts
        if texts_to_embed:
            logger.info(
                "Embedding %d chunks (%d from cache)",
                len(texts_to_embed), stats.cache_hits,
            )
            new_embeddings = await self.provider.embed(texts_to_embed)
        else:
            new_embeddings = []

        # Step 4: Merge cached and new embeddings, write to store
        embed_iter = iter(new_embeddings)
        for i, chunk in enumerate(all_chunks):
            _, cached_embedding = chunk_embed_map[i]
            if cached_embedding:
                embedding = cached_embedding
            else:
                embedding = next(embed_iter)
                # Cache the new embedding
                import hashlib
                chunk_hash = hashlib.sha256(chunk.content.encode("utf-8")).hexdigest()
                self.store.cache_embedding(chunk_hash, self.provider.model_name, embedding)

            pack_file = self.pack.files[chunk.file_path]
            self.store.upsert_chunk(
                file_path=chunk.file_path,
                chunk_index=chunk.chunk_index,
                content=chunk.content,
                title=chunk.title,
                type_=pack_file.type,
                tags=pack_file.tags,
                pack_slug=self.pack.slug,
                prov_id=pack_file.provenance.id,
                content_hash=pack_file.provenance.content_hash,
                verified_at=pack_file.provenance.verified_at,
                verified_by=pack_file.provenance.verified_by,
                token_count=chunk.token_count,
                embedding=embedding,
            )

        stats.total_chunks = self.store.chunk_count()
        self.store.commit()


class IndexStats:
    """Statistics from an indexing run."""

    def __init__(self):
        self.full_rebuild: bool = False
        self.new_files: int = 0
        self.changed_files: int = 0
        self.deleted_files: int = 0
        self.unchanged_files: int = 0
        self.total_chunks: int = 0
        self.cache_hits: int = 0
        self.cache_misses: int = 0

    def __repr__(self) -> str:
        return (
            f"IndexStats(new={self.new_files}, changed={self.changed_files}, "
            f"deleted={self.deleted_files}, unchanged={self.unchanged_files}, "
            f"chunks={self.total_chunks}, cache_hits={self.cache_hits}, "
            f"cache_misses={self.cache_misses}, rebuild={self.full_rebuild})"
        )
