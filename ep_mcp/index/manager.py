"""Index lifecycle management: build, update, staleness checks."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from pathlib import Path

from ..config import SecurityConfig
from ..embeddings.base import EmbeddingProvider
from ..pack.loader import _split_frontmatter
from ..pack.models import Pack
from ..pack.security import (
    PackSecurityError,
    detect_shipped_vector_files,
    scan_pack_security,
    verify_pack_integrity,
)
from ..pack.sidecar import load_sidecar, sidecar_path_for_md
from .chunker import Chunk, chunk_file
from .sqlite_store import SQLiteStore

logger = logging.getLogger(__name__)

# Bump when index-side contract changes (context_prefix, navigation filter).
# Existing indexes with a different value are fully rebuilt.
INDEX_FEATURES = "context_prefix_v1+nav_filter_v1+heading_path_v1+split_span_v1"


class IndexManager:
    """Manages the indexing pipeline for a single pack.

    Handles:
    - Pack integrity verification (W-INT-01..02) and security lint (W-SEC-01..05)
    - Always-re-embed guarantee (shipped vector artifacts are ignored)
    - Full index builds (first run, or embedding model/dimension change)
    - Incremental updates (only re-index changed/new/deleted files)
    - Embedding caching to avoid redundant API calls
    """

    def __init__(
        self,
        pack: Pack,
        store: SQLiteStore,
        embedding_provider: EmbeddingProvider,
        security_config: SecurityConfig | None = None,
    ):
        self.pack = pack
        self.store = store
        self.provider = embedding_provider
        self.security_config = security_config or SecurityConfig()

    def _check_integrity_and_security(self) -> set[str]:
        """Verify pack integrity, detect shipped vectors, and run W-SEC lint.

        Returns the set of pack-relative file paths blocked from indexing.
        Raises PackSecurityError when policy refuses the pack.
        """
        cfg = self.security_config
        blocked_files: set[str] = set()

        if self.pack.pack_dir and Path(self.pack.pack_dir).is_dir():
            # M2: Always re-embed — never trust or load shipped vector artifacts
            shipped_vecs = detect_shipped_vector_files(self.pack.pack_dir)
            if shipped_vecs:
                logger.warning(
                    "Pack '%s' contains precomputed vector/db files (%s) — ignoring them; "
                    "ep-mcp always re-embeds directly from Markdown source.",
                    self.pack.slug,
                    ", ".join(shipped_vecs[:5]),
                )

            # M1: Verify manifest.yaml integrity block
            report = verify_pack_integrity(
                self.pack.pack_dir,
                self.pack.manifest.raw,
                slug=self.pack.slug,
            )
            self.pack.integrity_status = report.status
            self.pack.digest_verified = report.digest_verified

            if report.status == "unverifiable":
                msg = (
                    f"Pack '{self.pack.slug}' failed integrity verification "
                    f"({'; '.join(report.issues) or 'unverifiable'})"
                )
                if cfg.on_integrity_failure == "refuse":
                    raise PackSecurityError(msg)
                logger.warning("%s — continuing because on_integrity_failure='warn'", msg)
            elif report.status == "unsigned":
                if report.digest_verified:
                    logger.info(
                        "Pack '%s' digest verified (unsigned, no cryptographic signature)",
                        self.pack.slug,
                    )
                else:
                    logger.warning(
                        "Pack '%s' has no integrity digest (integrity_status=unsigned) — loading anyway",
                        self.pack.slug,
                    )
            else:
                logger.info("Pack '%s' integrity verified (signed)", self.pack.slug)

            # M1: Security lint (W-SEC-01/02/03/05)
            if cfg.check_security_lint:
                findings = scan_pack_security(self.pack.pack_dir, self.pack.manifest.raw)
                blocking_codes = {"W-SEC-01", "W-SEC-03"}
                if cfg.strict_security:
                    blocking_codes |= {"W-SEC-02", "W-SEC-05"}

                blocking = [f for f in findings if f.code in blocking_codes]
                advisory = [f for f in findings if f.code not in blocking_codes]

                for f in advisory:
                    logger.warning("Security lint %s [%s/%s]: %s", f.code, self.pack.slug, f.file_path, f.message)

                if blocking:
                    summary = "; ".join(f"{f.code} in {f.file_path}: {f.message}" for f in blocking[:5])
                    if cfg.on_security_finding == "refuse":
                        raise PackSecurityError(
                            f"Pack '{self.pack.slug}' rejected by security lint: {summary}"
                        )
                    elif cfg.on_security_finding == "skip":
                        for f in blocking:
                            logger.warning(
                                "Skipping file from index due to %s [%s/%s]: %s",
                                f.code, self.pack.slug, f.file_path, f.message,
                            )
                            blocked_files.add(f.file_path)
                    else:
                        for f in blocking:
                            logger.warning(
                                "Security lint %s [%s/%s]: %s (policy=warn)",
                                f.code, self.pack.slug, f.file_path, f.message,
                            )

        self.pack.security_blocked_files = sorted(blocked_files)
        return blocked_files

    async def build_index(self) -> IndexStats:
        """Build or update the pack index.

        Detects whether a full rebuild or incremental update is needed:
        - Full rebuild: first run, or embedding model/dimension changed
        - Incremental: only changed/new/deleted files

        Returns:
            IndexStats with counts of what was done
        """
        stats = IndexStats()
        blocked_files = self._check_integrity_and_security()

        # Check if we need a full rebuild
        stored_model = self.store.get_meta("embedding_model")
        current_model = self.provider.model_name
        stored_dim = self.store.get_meta("embedding_dimension")
        current_dim = str(self.provider.dimension)

        stored_features = self.store.get_meta("index_features")
        # None + existing chunks = pre-0.5 index that must rebuild for prefixes/nav filter.
        features_changed = stored_features != INDEX_FEATURES and (
            stored_features is not None or self.store.chunk_count() > 0
        )
        model_changed = bool(stored_model and stored_model != current_model)
        dim_changed = bool(stored_dim and stored_dim != current_dim)

        if model_changed or dim_changed or features_changed:
            if model_changed:
                reason = f"embedding model {stored_model} → {current_model}"
            elif dim_changed:
                reason = f"embedding dimension {stored_dim} → {current_dim}"
            else:
                reason = f"index features {stored_features} → {INDEX_FEATURES}"
            logger.info("Full rebuild required (%s)", reason)
            stats.full_rebuild = True
            if model_changed or dim_changed:
                if stored_model:
                    self.store.invalidate_cache_for_model(stored_model)
            for file_path in self.store.get_indexed_files():
                self.store.delete_file_chunks(file_path)
            if dim_changed:
                self.store.recreate_vec_table(self.provider.dimension)
            self.store.commit()

        # Get current index state
        indexed_hashes = self.store.get_file_hashes()
        indexed_files = set(indexed_hashes.keys())
        indexable_files = {
            fp for fp, f in self.pack.files.items()
            if f.is_indexable and fp not in blocked_files
        }

        # Determine what needs updating
        new_files = indexable_files - indexed_files
        deleted_files = indexed_files - indexable_files
        existing_files = indexable_files & indexed_files

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
        self.store.set_meta("integrity_status", self.pack.integrity_status)
        self.store.set_meta("digest_verified", "true" if self.pack.digest_verified else "false")
        self.store.set_meta("indexed_at", datetime.now(timezone.utc).isoformat())
        self.store.set_meta("file_count", str(self.store.file_count()))
        self.store.set_meta("chunk_count", str(self.store.chunk_count()))
        self.store.set_meta("index_features", INDEX_FEATURES)
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

            sidecar = None
            frontmatter, _ = _split_frontmatter(pack_file.raw_content)
            sidecar_path = sidecar_path_for_md(
                Path(self.pack.pack_dir),
                fp,
                frontmatter=frontmatter,
            )
            if sidecar_path is not None:
                sidecar = load_sidecar(sidecar_path)
                if sidecar is None:
                    logger.warning("Failed to load chunk sidecar for %s", fp)

            chunks = chunk_file(
                fp,
                pack_file.content,
                pack_file.title,
                raw_content=pack_file.raw_content,
                sidecar=sidecar,
            )
            all_chunks.extend(chunks)

        stats.total_chunks = self.store.chunk_count()  # existing
        logger.info("Chunked %d files into %d chunks", len(file_paths), len(all_chunks))

        # Step 2: Check embedding cache and identify what needs embedding
        # When provider.dimension == 0 (lexical/BM25-only mode), skip embeddings completely.
        lexical_only = self.provider.dimension == 0
        texts_to_embed: list[str] = []
        chunk_embed_map: list[tuple[int, list[float] | None]] = []  # (index_in_all_chunks, cached_or_None)

        if not lexical_only:
            for i, chunk in enumerate(all_chunks):
                # Compute content hash for this chunk's index text (may include prefix)
                import hashlib
                index_text = chunk.text_for_index()
                chunk_hash = hashlib.sha256(index_text.encode("utf-8")).hexdigest()

                cached = self.store.get_cached_embedding(chunk_hash, self.provider.model_name)
                if cached:
                    chunk_embed_map.append((i, cached))
                    stats.cache_hits += 1
                else:
                    chunk_embed_map.append((i, None))
                    texts_to_embed.append(index_text)
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
            if lexical_only:
                embedding = None
            else:
                _, cached_embedding = chunk_embed_map[i]
                if cached_embedding:
                    embedding = cached_embedding
                else:
                    embedding = next(embed_iter)
                    # Cache the new embedding
                    import hashlib
                    chunk_hash = hashlib.sha256(chunk.text_for_index().encode("utf-8")).hexdigest()
                    self.store.cache_embedding(chunk_hash, self.provider.model_name, embedding)

            pack_file = self.pack.files[chunk.file_path]
            line_start = line_end = None
            if chunk.line_range is not None:
                line_start, line_end = chunk.line_range
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
                line_start=line_start,
                line_end=line_end,
                section_slug=chunk.section_slug,
                sidecar_chunk_id=chunk.sidecar_chunk_id,
                span_hash=chunk.span_hash,
                confidence=pack_file.provenance.confidence,
                indexed_content=chunk.indexed_content,
                heading_path=chunk.heading_path,
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
