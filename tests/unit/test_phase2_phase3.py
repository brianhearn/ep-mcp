"""Tests for zero-config/zero-key lexical mode (D1), discoverability + exact-phrase +
citation tokens (M3), and ep_feedback JSONL queue (M4)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from ep_mcp.cli import _auto_embedding_config, _build_zero_config
from ep_mcp.config import EmbeddingConfig, ServerConfig
from ep_mcp.embeddings.lexical import LexicalOnlyEmbeddingProvider
from ep_mcp.index.manager import IndexManager
from ep_mcp.index.sqlite_store import SQLiteStore
from ep_mcp.pack.loader import load_pack
from ep_mcp.pack.security import compute_pack_digest
from ep_mcp.retrieval.engine import RetrievalEngine
from ep_mcp.server import create_embedding_provider, create_multi_pack_mcp, create_pack_mcp, PackInstance
from ep_mcp.tools.ep_describe import ep_describe_pack, ep_list_packs
from ep_mcp.tools.ep_feedback import ep_feedback
from ep_mcp.tools.ep_read import ep_read
from ep_mcp.tools.ep_search import ep_search


def _make_pack(tmp_path: Path, slug: str = "solar-mini") -> Path:
    pack_dir = tmp_path / slug
    (pack_dir / "concepts").mkdir(parents=True)
    (pack_dir / "overview.md").write_text(
        f"---\ntitle: Overview\ntype: reference\ntags: [solar]\npack: {slug}\n---\n"
        "# Solar Mini\n\nCompact residential solar test pack.\n",
        encoding="utf-8",
    )
    (pack_dir / "concepts" / "rapid-shutdown.md").write_text(
        f"---\ntitle: NEC Rapid Shutdown\ntype: concept\ntags: [nec, safety]\npack: {slug}\n"
        f"id: {slug}/concepts/rapid-shutdown\nverified_at: '2026-10-01'\nconfidence: expert-verified\n---\n"
        "# NEC Rapid Shutdown\n\nNEC 690.12 requires module-level rapid shutdown within 30 seconds.\n",
        encoding="utf-8",
    )
    (pack_dir / "concepts" / "string-sizing.md").write_text(
        f"---\ntitle: String Sizing\ntype: concept\ntags: [design, nec]\npack: {slug}\n"
        f"id: {slug}/concepts/string-sizing\nverified_at: '2026-10-01'\nconfidence: expert-verified\n---\n"
        "# String Sizing\n\nCompute cold-weather open-circuit voltage (Voc) using NEC Table 690.7.\n",
        encoding="utf-8",
    )
    digest = compute_pack_digest(pack_dir)
    manifest = {
        "name": "Solar Mini",
        "slug": slug,
        "type": "product",
        "version": "1.5.0",
        "schema_version": "4.2",
        "description": "Mini solar pack for unit tests.",
        "entry_point": "overview.md",
        "license": "CC-BY-4.0",
        "canonical_url": f"https://example.com/packs/{slug}",
        "applies_to": "nec@>=2020",
        "authority_boundary": {
            "in_scope": "Solar NEC rules",
            "out_of_scope": ["Tax advice"],
            "refuse_when": ["No supporting atom"],
            "no_source_no_claim": True,
        },
        "integrity": {
            "algorithm": "sha256",
            "pack_digest": digest,
            "pack_version": "1.5.0",
        },
    }
    (pack_dir / "manifest.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    return pack_dir


def test_auto_embedding_falls_back_to_zero_key_lexical(monkeypatch, tmp_path: Path):
    for var in ("GEMINI_API_KEY", "OPENAI_API_KEY", "OPENAI_BASE_URL", "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT"):
        monkeypatch.delenv(var, raising=False)

    emb = _auto_embedding_config("auto")
    assert emb.provider == "lexical"

    pack_dir = _make_pack(tmp_path)
    cfg = _build_zero_config((str(pack_dir),))
    assert len(cfg.packs) == 1
    assert cfg.packs[0].slug == "solar-mini"
    assert cfg.embedding.provider == "lexical"

    provider = create_embedding_provider(cfg)
    assert isinstance(provider, LexicalOnlyEmbeddingProvider)
    assert provider.dimension == 0


@pytest.mark.asyncio
async def test_zero_key_lexical_index_search_exact_phrase_and_provenance(tmp_path: Path):
    pack_dir = _make_pack(tmp_path)
    pack = load_pack(pack_dir)
    provider = LexicalOnlyEmbeddingProvider()
    store = SQLiteStore(pack.index_path, embedding_dimension=0)
    store.open()
    try:
        mgr = IndexManager(pack, store, provider)
        stats = await mgr.build_index()
        assert stats.total_chunks >= 2
        assert pack.digest_verified is True
        assert pack.integrity_status == "unsigned"

        engine = RetrievalEngine(pack, store, provider)

        # Lexical BM25 search works with zero API keys and attaches version/integrity/citation token
        hits = await ep_search(engine, "rapid shutdown 690.12")
        assert len(hits) >= 1
        top = hits[0]
        assert top["source_file"] == "concepts/rapid-shutdown.md"
        assert top["pack_version"] == "1.5.0"
        assert top["integrity_status"] == "unsigned"
        assert top["digest_verified"] is True
        assert top["provenance_token"].startswith("[ep:solar-mini/concepts/rapid-shutdown@v1.5.0|expert-verified|2026-10-01|unsigned]")
        assert "concepts/rapid-shutdown.md" in top["citation_markdown"]

        # Exact-phrase mode returns ONLY literal matches
        exact_hits = await ep_search(engine, "Table 690.7", exact_phrase=True)
        assert len(exact_hits) == 1
        assert exact_hits[0]["source_file"] == "concepts/string-sizing.md"

        no_hits = await ep_search(engine, "nonexistent literal phrase xyz", exact_phrase=True)
        assert no_hits == []

        # ep_read also includes pack_version, integrity_status, provenance_token
        doc = ep_read(pack, path="concepts/rapid-shutdown.md")
        assert doc["pack_version"] == "1.5.0"
        assert doc["digest_verified"] is True
        assert doc["provenance_token"].startswith("[ep:solar-mini/concepts/rapid-shutdown@v1.5.0")
    finally:
        store.close()


@pytest.mark.asyncio
async def test_describe_list_packs_and_feedback_queue(tmp_path: Path):
    pack_dir = _make_pack(tmp_path)
    pack = load_pack(pack_dir)
    provider = LexicalOnlyEmbeddingProvider()
    store = SQLiteStore(pack.index_path, embedding_dimension=0)
    store.open()
    try:
        mgr = IndexManager(pack, store, provider)
        await mgr.build_index()
        engine = RetrievalEngine(pack, store, provider)

        desc = ep_describe_pack(pack, chunk_count=store.chunk_count())
        assert desc["slug"] == "solar-mini"
        assert desc["version"] == "1.5.0"
        assert desc["license"] == "CC-BY-4.0"
        assert desc["applies_to"] == "nec@>=2020"
        assert desc["digest_verified"] is True
        assert desc["authority_boundary"]["no_source_no_claim"] is True

        listed = ep_list_packs({"solar-mini": (pack, store.chunk_count())})
        assert listed["count"] == 1
        assert listed["packs"][0]["slug"] == "solar-mini"

        # ep_feedback writes JSONL queue and never modifies pack Markdown
        orig_md = (pack_dir / "concepts" / "rapid-shutdown.md").read_text(encoding="utf-8")
        fb_dir = tmp_path / "feedback"
        res = ep_feedback(
            pack,
            kind="stale",
            detail="NEC 2026 revised rapid shutdown clearance.",
            path="concepts/rapid-shutdown.md",
            user_goal="Verify current rapid shutdown rules",
            feedback_dir=fb_dir,
        )
        assert res["status"] == "recorded"
        queue_file = Path(res["queue_file"])
        assert queue_file.is_file()
        lines = queue_file.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1
        rec = json.loads(lines[0])
        assert rec["kind"] == "stale"
        assert rec["pack"] == "solar-mini"
        assert rec["pack_version"] == "1.5.0"
        assert (pack_dir / "concepts" / "rapid-shutdown.md").read_text(encoding="utf-8") == orig_md

        # Invalid kind rejected cleanly
        bad = ep_feedback(pack, kind="bogus", detail="x", feedback_dir=fb_dir)
        assert "error" in bad

        # Verify tools are registered on both single-pack and multi-pack MCPServer instances
        single_mcp = create_pack_mcp("solar-mini", pack, engine, feedback_dir=str(fb_dir))
        inst = PackInstance(pack, store, engine, single_mcp, mgr)
        multi_mcp = create_multi_pack_mcp({"solar-mini": inst}, feedback_dir=str(fb_dir))
        for server in (single_mcp, multi_mcp):
            tool_names = {t.name for t in await server.list_tools()}
            assert "ep_list_packs_tool" in tool_names
            assert "ep_describe_pack_tool" in tool_names
            assert "ep_feedback_tool" in tool_names
    finally:
        store.close()
