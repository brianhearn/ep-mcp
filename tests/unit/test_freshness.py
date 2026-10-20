"""Age staleness / volatile expiry (Core 4.2) — 0.7.0."""

from datetime import date
from pathlib import Path

import pytest

from ep_mcp.retrieval.freshness import compute_freshness, parse_duration_days
from ep_mcp.retrieval.models import SearchResult


@pytest.mark.parametrize("value,days", [
    ("P90D", 90), ("P3M", 90), ("P1Y", 365), ("P2W", 14), ("quarterly", 91),
    ("per-release", None), (None, None), ("", None), ("P", None),
])
def test_parse_duration(value, days):
    assert parse_duration_days(value) == days


def test_age_stale_and_expiry():
    today = date(2026, 10, 20)
    fr = compute_freshness("2026-06-01", "P90D", today=today)
    assert fr["verified_age_days"] == 141 and fr["age_stale"] is True and fr["expired"] is None
    fr = compute_freshness("2026-10-01", "P90D", today=today)
    assert fr["age_stale"] is False
    fr = compute_freshness(None, "P90D", expires_at="2026-10-19", today=today)
    assert fr["age_stale"] is None and fr["expired"] is True and fr["expires_at"] == "2026-10-19"
    # per-file refresh overrides the manifest cycle
    assert compute_freshness("2026-10-01", "P90D", file_refresh="P7D", today=today)["age_stale"] is True
    assert compute_freshness("2026-10-01", "per-release", today=today)["age_stale"] is None


def _pack(tmp_path: Path):
    from ep_mcp.pack.loader import load_pack
    (tmp_path / "manifest.yaml").write_text(
        "slug: f\nname: F\ntype: product\nfreshness:\n  refresh_cycle: P30D\n", encoding="utf-8")
    (tmp_path / "old.md").write_text("---\nverified_at: 2026-01-01\n---\n# Old\nx\n", encoding="utf-8")
    (tmp_path / "new.md").write_text("---\nverified_at: 2026-10-15\n---\n# New\nx\n", encoding="utf-8")
    (tmp_path / "vol.md").write_text(
        "---\nrefresh: P7D\nfetched_at: 2026-10-01\nexpires_at: 2026-10-08\n---\n# Vol\nx\n", encoding="utf-8")
    return load_pack(tmp_path)


def _engine(pack, penalty=0.0):
    from ep_mcp.config import RetrievalConfig
    from ep_mcp.retrieval.engine import RetrievalEngine

    eng = RetrievalEngine.__new__(RetrievalEngine)
    eng.pack = pack
    eng.config = RetrievalConfig(freshness_penalty=penalty, freshness_today="2026-10-20")
    return eng


def _results():
    return [SearchResult(text="", source_file=p, score=s) for p, s in
            (("old.md", 0.9), ("vol.md", 0.85), ("new.md", 0.8))]


def test_engine_annotation_no_rerank_by_default(tmp_path):
    pack = _pack(tmp_path)
    assert pack.files["vol.md"].fetched_at == "2026-10-01"
    res = _results()
    _engine(pack)._annotate_freshness(res)
    by = {r.source_file: r for r in res}
    assert by["old.md"].age_stale is True and by["new.md"].age_stale is False
    assert by["vol.md"].expired is True and by["vol.md"].age_stale is None
    assert [r.source_file for r in res] == ["old.md", "vol.md", "new.md"]  # order unchanged


def test_engine_down_rank_opt_in(tmp_path):
    res = _results()
    _engine(_pack(tmp_path), penalty=0.5)._annotate_freshness(res)
    assert res[0].source_file == "new.md"
    assert res[1].score == 0.45


def test_ep_read_freshness_and_sections(tmp_path):
    from ep_mcp.pack.loader import load_pack
    from ep_mcp.tools.ep_read import ep_read

    (tmp_path / "manifest.yaml").write_text("slug: r\nname: R\ntype: product\n", encoding="utf-8")
    (tmp_path / "a.md").write_text(
        "---\ntitle: Guide\n---\n# Guide\nIntro.\n\n## Setup\nS.\n\n### Ports\nP.\n\n## Usage\nU.\n",
        encoding="utf-8")
    pack = load_pack(tmp_path)
    full = ep_read(pack, path="a.md")
    assert [s["slug"] for s in full["sections"]] == ["opening", "setup", "ports", "usage"]
    assert "age_stale" in full and "expired" in full
    sec = ep_read(pack, path="a.md", section="ports", neighbors=True)
    assert sec["content"] == "### Ports\nP."
    assert sec["heading_path"] == ["Guide", "Setup", "Ports"]
    assert sec["neighbors"]["previous"]["slug"] == "setup"
    assert sec["neighbors"]["next"]["slug"] == "usage"
    setup = ep_read(pack, path="a.md", section="Setup")
    assert "### Ports" in setup["content"]  # ## includes its ### children
    miss = ep_read(pack, path="a.md", section="nope")
    assert miss["error"] == "Section not found" and "usage" in miss["available_sections"]
