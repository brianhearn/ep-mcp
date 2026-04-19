"""Tests for atomic-conceptual `requires:` expansion (schema v4.1+).

The `_apply_requires_expansion` helper appends atoms declared in a retrieved
atom's `requires:` frontmatter as additional SearchResults flagged
`requires_expanded=True`. It respects depth, count, and token-budget caps,
skips atoms already in the result set, and supports three resolution forms
for `requires:` entries (full provenance id, exact path, bare basename).
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock

import numpy as np
import pytest

from ep_mcp.config import RetrievalConfig
from ep_mcp.index.sqlite_store import SQLiteStore
from ep_mcp.pack.models import (
    ContextTiers,
    Manifest,
    Pack,
    PackFile,
    Provenance,
)
from ep_mcp.retrieval.engine import RetrievalEngine
from ep_mcp.retrieval.models import SearchResult


def _fake_embedding(dim: int = 4) -> list[float]:
    vec = np.random.rand(dim).astype(np.float32)
    vec /= np.linalg.norm(vec)
    return vec.tolist()


def _make_packfile(
    path: str,
    prov_id: str | None = None,
    requires: list[str] | None = None,
    tokens: int = 500,
) -> PackFile:
    return PackFile(
        path=path,
        title=path.rsplit("/", 1)[-1].replace(".md", "").replace("-", " ").title(),
        type="concept",
        tags=[],
        provenance=Provenance(id=prov_id),
        retrieval_strategy="atomic",
        requires=requires or [],
        content=f"Body of {path}",
        raw_content=f"Body of {path}",
        size_tokens=tokens,
    )


def _build_pack(files: dict[str, PackFile]) -> Pack:
    return Pack(
        slug="test-pack",
        name="Test Pack",
        type="product",
        version="1.0.0",
        files=files,
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
    with tempfile.TemporaryDirectory() as d:
        db_path = str(Path(d) / "test.db")
        s = SQLiteStore(db_path, embedding_dimension=4)
        s.open()
        yield s
        s.close()


def _seed_chunks(store: SQLiteStore, paths: list[str]) -> dict[str, int]:
    """Seed the SQLite store with primary chunks for the given paths."""
    out: dict[str, int] = {}
    for fp in paths:
        cid = store.upsert_chunk(
            file_path=fp,
            chunk_index=0,
            content=f"Body of {fp}",
            title=fp.rsplit("/", 1)[-1].replace(".md", ""),
            type_="concept",
            tags=[],
            pack_slug="test-pack",
            prov_id=None,
            content_hash=None,
            verified_at=None,
            verified_by=None,
            token_count=10,
            embedding=_fake_embedding(),
        )
        out[fp] = cid
    store.commit()
    return out


def _make_engine(store: SQLiteStore, pack: Pack, config: RetrievalConfig | None = None) -> RetrievalEngine:
    provider = AsyncMock()
    provider.embed_query = AsyncMock(return_value=_fake_embedding())
    provider.dimension = 4
    return RetrievalEngine(
        pack=pack,
        store=store,
        embedding_provider=provider,
        config=config or RetrievalConfig(),
    )


def _top_k_result(path: str, score: float = 0.8) -> SearchResult:
    return SearchResult(
        text=f"Body of {path}",
        source_file=path,
        score=score,
        type="concept",
        tags=[],
        chunk_index=0,
    )


# ---------------------------------------------------------------------------
# Resolver tests (id / path / basename)
# ---------------------------------------------------------------------------


class TestResolver:
    def test_resolve_by_full_provenance_id(self, store):
        files = {
            "concepts/foo.md": _make_packfile(
                "concepts/foo.md",
                prov_id="test-pack/concepts/foo",
                requires=["test-pack/concepts/bar"],
            ),
            "concepts/bar.md": _make_packfile(
                "concepts/bar.md", prov_id="test-pack/concepts/bar",
            ),
        }
        pack = _build_pack(files)
        _seed_chunks(store, list(files))
        engine = _make_engine(store, pack)

        resolved = engine._resolve_requires_for_path("concepts/foo.md")
        assert resolved == ["concepts/bar.md"]

    def test_resolve_by_exact_path(self, store):
        files = {
            "concepts/foo.md": _make_packfile(
                "concepts/foo.md", requires=["concepts/bar.md"],
            ),
            "concepts/bar.md": _make_packfile("concepts/bar.md"),
        }
        pack = _build_pack(files)
        _seed_chunks(store, list(files))
        engine = _make_engine(store, pack)

        resolved = engine._resolve_requires_for_path("concepts/foo.md")
        assert resolved == ["concepts/bar.md"]

    def test_resolve_by_bare_basename_same_dir(self, store):
        files = {
            "concepts/foo.md": _make_packfile(
                "concepts/foo.md", requires=["bar.md"],
            ),
            "concepts/bar.md": _make_packfile("concepts/bar.md"),
        }
        pack = _build_pack(files)
        _seed_chunks(store, list(files))
        engine = _make_engine(store, pack)

        resolved = engine._resolve_requires_for_path("concepts/foo.md")
        assert resolved == ["concepts/bar.md"]

    def test_resolve_bare_slug_without_md_extension(self, store):
        """Schema examples use bare slugs like 'bar' without .md"""
        files = {
            "concepts/foo.md": _make_packfile(
                "concepts/foo.md", requires=["bar"],
            ),
            "concepts/bar.md": _make_packfile("concepts/bar.md"),
        }
        pack = _build_pack(files)
        _seed_chunks(store, list(files))
        engine = _make_engine(store, pack)

        resolved = engine._resolve_requires_for_path("concepts/foo.md")
        assert resolved == ["concepts/bar.md"]

    def test_unresolved_entry_returns_empty(self, store):
        files = {
            "concepts/foo.md": _make_packfile(
                "concepts/foo.md", requires=["concepts/does-not-exist.md"],
            ),
        }
        pack = _build_pack(files)
        _seed_chunks(store, list(files))
        engine = _make_engine(store, pack)

        resolved = engine._resolve_requires_for_path("concepts/foo.md")
        assert resolved == []

    def test_self_reference_filtered(self, store):
        files = {
            "concepts/foo.md": _make_packfile(
                "concepts/foo.md",
                prov_id="test-pack/concepts/foo",
                requires=["concepts/foo.md", "test-pack/concepts/foo"],
            ),
        }
        pack = _build_pack(files)
        _seed_chunks(store, list(files))
        engine = _make_engine(store, pack)

        resolved = engine._resolve_requires_for_path("concepts/foo.md")
        assert resolved == []

    def test_no_requires_returns_empty(self, store):
        files = {"concepts/foo.md": _make_packfile("concepts/foo.md")}
        pack = _build_pack(files)
        _seed_chunks(store, list(files))
        engine = _make_engine(store, pack)

        resolved = engine._resolve_requires_for_path("concepts/foo.md")
        assert resolved == []


# ---------------------------------------------------------------------------
# Expansion tests (depth, count, token budget, already-present)
# ---------------------------------------------------------------------------


class TestExpansion:
    def test_basic_one_hop_expansion(self, store):
        """A atom with requires:[B] retrieves A + B when A is in top-K."""
        files = {
            "concepts/a.md": _make_packfile(
                "concepts/a.md", requires=["concepts/b.md"],
            ),
            "concepts/b.md": _make_packfile("concepts/b.md"),
        }
        pack = _build_pack(files)
        _seed_chunks(store, list(files))
        engine = _make_engine(store, pack)

        top_k = [_top_k_result("concepts/a.md")]
        bonus = engine._apply_requires_expansion(top_k)

        assert len(bonus) == 1
        assert bonus[0].source_file == "concepts/b.md"
        assert bonus[0].requires_expanded is True
        assert bonus[0].graph_expanded is False

    def test_disabled_returns_empty(self, store):
        files = {
            "concepts/a.md": _make_packfile("concepts/a.md", requires=["concepts/b.md"]),
            "concepts/b.md": _make_packfile("concepts/b.md"),
        }
        pack = _build_pack(files)
        _seed_chunks(store, list(files))
        config = RetrievalConfig(requires_expansion_enabled=False)
        engine = _make_engine(store, pack, config)

        # Even calling directly, zero-caps yield nothing
        config2 = RetrievalConfig(requires_expansion_max_atoms=0)
        engine2 = _make_engine(store, pack, config2)
        assert engine2._apply_requires_expansion([_top_k_result("concepts/a.md")]) == []

    def test_transitive_expansion_within_depth(self, store):
        """A -> B -> C: depth=2 pulls both B and C; depth=1 pulls only B."""
        files = {
            "concepts/a.md": _make_packfile("concepts/a.md", requires=["concepts/b.md"]),
            "concepts/b.md": _make_packfile("concepts/b.md", requires=["concepts/c.md"]),
            "concepts/c.md": _make_packfile("concepts/c.md"),
        }
        pack = _build_pack(files)
        _seed_chunks(store, list(files))

        # depth=2 — both B and C come through
        engine2 = _make_engine(store, pack, RetrievalConfig(requires_expansion_max_depth=2))
        bonus = engine2._apply_requires_expansion([_top_k_result("concepts/a.md")])
        assert sorted(b.source_file for b in bonus) == ["concepts/b.md", "concepts/c.md"]

        # depth=1 — only B
        engine1 = _make_engine(store, pack, RetrievalConfig(requires_expansion_max_depth=1))
        bonus = engine1._apply_requires_expansion([_top_k_result("concepts/a.md")])
        assert [b.source_file for b in bonus] == ["concepts/b.md"]

    def test_atom_cap_respected(self, store):
        """max_atoms caps total expansion across all top-K seeds."""
        files = {
            "concepts/a.md": _make_packfile("concepts/a.md", requires=["concepts/b.md", "concepts/c.md", "concepts/d.md", "concepts/e.md"]),
            "concepts/b.md": _make_packfile("concepts/b.md"),
            "concepts/c.md": _make_packfile("concepts/c.md"),
            "concepts/d.md": _make_packfile("concepts/d.md"),
            "concepts/e.md": _make_packfile("concepts/e.md"),
        }
        pack = _build_pack(files)
        _seed_chunks(store, list(files))
        config = RetrievalConfig(requires_expansion_max_atoms=2)
        engine = _make_engine(store, pack, config)

        bonus = engine._apply_requires_expansion([_top_k_result("concepts/a.md")])
        assert len(bonus) == 2

    def test_token_budget_stops_expansion(self, store):
        """Token budget cuts expansion when the next atom would overflow."""
        files = {
            "concepts/a.md": _make_packfile("concepts/a.md", requires=["concepts/b.md", "concepts/c.md"]),
            "concepts/b.md": _make_packfile("concepts/b.md", tokens=800),
            "concepts/c.md": _make_packfile("concepts/c.md", tokens=800),
        }
        pack = _build_pack(files)
        _seed_chunks(store, list(files))
        # Budget = 1000. First atom (800) fits; second (800) would push to 1600, over budget.
        config = RetrievalConfig(requires_expansion_token_budget=1000)
        engine = _make_engine(store, pack, config)

        bonus = engine._apply_requires_expansion([_top_k_result("concepts/a.md")])
        assert len(bonus) == 1
        assert bonus[0].source_file == "concepts/b.md"

    def test_already_in_top_k_not_duplicated(self, store):
        """If B is already in top-K, expansion must not re-add it."""
        files = {
            "concepts/a.md": _make_packfile("concepts/a.md", requires=["concepts/b.md"]),
            "concepts/b.md": _make_packfile("concepts/b.md"),
        }
        pack = _build_pack(files)
        _seed_chunks(store, list(files))
        engine = _make_engine(store, pack)

        top_k = [_top_k_result("concepts/a.md"), _top_k_result("concepts/b.md")]
        bonus = engine._apply_requires_expansion(top_k)
        assert bonus == []

    def test_top_k_carrier_propagates_requires_even_when_in_set(self, store):
        """If A in top-K requires B and B in top-K requires C, C still expands (even though B is already present)."""
        files = {
            "concepts/a.md": _make_packfile("concepts/a.md", requires=["concepts/b.md"]),
            "concepts/b.md": _make_packfile("concepts/b.md", requires=["concepts/c.md"]),
            "concepts/c.md": _make_packfile("concepts/c.md"),
        }
        pack = _build_pack(files)
        _seed_chunks(store, list(files))
        config = RetrievalConfig(requires_expansion_max_depth=2)
        engine = _make_engine(store, pack, config)

        top_k = [_top_k_result("concepts/a.md"), _top_k_result("concepts/b.md")]
        bonus = engine._apply_requires_expansion(top_k)
        assert [b.source_file for b in bonus] == ["concepts/c.md"]

    def test_directional_only(self, store):
        """Retrieving B alone does not pull A, even if A requires B."""
        files = {
            "concepts/a.md": _make_packfile("concepts/a.md", requires=["concepts/b.md"]),
            "concepts/b.md": _make_packfile("concepts/b.md"),
        }
        pack = _build_pack(files)
        _seed_chunks(store, list(files))
        engine = _make_engine(store, pack)

        bonus = engine._apply_requires_expansion([_top_k_result("concepts/b.md")])
        assert bonus == []

    def test_no_primary_chunk_skipped(self, store):
        """If the required file has no indexed chunk, it's skipped (not an error)."""
        files = {
            "concepts/a.md": _make_packfile("concepts/a.md", requires=["concepts/b.md", "concepts/c.md"]),
            "concepts/b.md": _make_packfile("concepts/b.md"),
            "concepts/c.md": _make_packfile("concepts/c.md"),
        }
        pack = _build_pack(files)
        # Only seed A and C; B is declared but not indexed.
        _seed_chunks(store, ["concepts/a.md", "concepts/c.md"])
        engine = _make_engine(store, pack)

        bonus = engine._apply_requires_expansion([_top_k_result("concepts/a.md")])
        assert [b.source_file for b in bonus] == ["concepts/c.md"]

    def test_unresolved_entry_warns_and_continues(self, store):
        """Unresolvable requires entries don't crash; other entries still resolve."""
        files = {
            "concepts/a.md": _make_packfile("concepts/a.md", requires=["concepts/missing.md", "concepts/b.md"]),
            "concepts/b.md": _make_packfile("concepts/b.md"),
        }
        pack = _build_pack(files)
        _seed_chunks(store, list(files))
        engine = _make_engine(store, pack)

        bonus = engine._apply_requires_expansion([_top_k_result("concepts/a.md")])
        assert [b.source_file for b in bonus] == ["concepts/b.md"]

    def test_loader_parses_requires_from_frontmatter(self, tmp_path):
        """End-to-end: the loader reads `requires:` into PackFile.requires."""
        from ep_mcp.pack.loader import load_pack

        (tmp_path / "manifest.yaml").write_text(
            """
slug: test-pack
name: Test Pack
type: product
version: '1.0.0'
description: t
entry_point: overview.md
""".strip(),
            encoding="utf-8",
        )
        (tmp_path / "overview.md").write_text(
            "---\nid: test-pack/overview\ntype: concept\nrequires:\n  - concepts/bar.md\n  - concepts/baz.md\n---\n\n# Overview\n\nBody.\n",
            encoding="utf-8",
        )
        concepts = tmp_path / "concepts"
        concepts.mkdir()
        (concepts / "bar.md").write_text("---\nid: test-pack/concepts/bar\n---\n\n# Bar\n", encoding="utf-8")
        (concepts / "baz.md").write_text("---\nid: test-pack/concepts/baz\n---\n\n# Baz\n", encoding="utf-8")

        pack = load_pack(str(tmp_path))
        assert pack.files["overview.md"].requires == ["concepts/bar.md", "concepts/baz.md"]
        assert pack.files["concepts/bar.md"].requires == []
