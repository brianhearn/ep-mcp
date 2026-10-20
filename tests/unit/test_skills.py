"""Skills over MCP (io.modelcontextprotocol/skills) — 0.7.0."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from pydantic import TypeAdapter

from ep_mcp.pack.loader import load_pack
from ep_mcp.server import create_pack_mcp
from ep_mcp.skills import EXTENSION_ID, SkillCatalog, generate_skill_md


def _write_pack(root: Path, enabled: bool = True) -> Path:
    (root / "concepts").mkdir(parents=True)
    (root / "volatile").mkdir()
    (root / "manifest.yaml").write_text(
        f"""
slug: tiny-skill
name: Tiny Skill Pack
type: product
version: "1.0.0"
description: Tiny fixture pack for skills.
entry_point: overview.md
authority_boundary:
  in_scope: Tiny widgets
  out_of_scope: [Legal advice]
context:
  always: [overview.md]
  on_demand: [concepts/secret.md]
mcp:
  skills:
    enabled: {str(enabled).lower()}
""",
        encoding="utf-8",
    )
    (root / "overview.md").write_text("---\nid: tiny/overview\n---\n# Tiny\nHello — ünïcode.\n", encoding="utf-8")
    (root / "concepts" / "widget.md").write_text("---\ntitle: Widget\ntype: concept\n---\n# Widget\nBody.\n", encoding="utf-8")
    (root / "concepts" / "secret.md").write_text("# Secret\nOn demand only.\n", encoding="utf-8")
    (root / "volatile" / "prices.md").write_text(
        "---\ntitle: Prices\ntype: volatile\nrefresh: P7D\nexpires_at: 2026-01-01\n---\n# Prices\n", encoding="utf-8"
    )
    return root


async def _rpc(client, method: str, params: dict):
    from typing import Any

    from mcp_types import Request

    class _Req(Request):
        params: dict[str, Any] | None = None

    req = _Req(method=method, params=params)
    return await client.session.send_request(req, TypeAdapter(dict))


def _payload(result):
    return json.loads(result.content[0].text) if not result.structured_content else result.structured_content


def test_entry_digests_match_served_bytes(tmp_path):
    pack = load_pack(_write_pack(tmp_path))
    cat = SkillCatalog(pack)
    entry = cat.entry()
    assert entry["uri"] == "skill://tiny-skill/SKILL.md"
    assert entry["frontmatter"]["name"] == "tiny-skill"
    uris = [r["uri"] for r in entry["resources"]]
    assert uris[0] == "skill://tiny-skill/SKILL.md"
    assert "skill://tiny-skill/concepts/secret.md" not in uris  # on_demand excluded
    for r in entry["resources"]:
        data = cat.read(r["uri"].removeprefix("skill://tiny-skill/")).encode("utf-8")
        assert r["size"] == len(data)
        assert r["digest"] == "sha256:" + hashlib.sha256(data).hexdigest()
    # overview served verbatim (frontmatter included), raw bytes on disk
    assert cat.read("overview.md").encode() == (tmp_path / "overview.md").read_bytes()
    # ttl = min(volatile refresh, 1h) -> 1h; scope public by default
    assert cat.ttl_ms == 3_600_000 and cat.cache_scope == "public"
    assert SkillCatalog(pack, private=True).cache_scope == "private"


def test_entry_stable_across_reload(tmp_path):
    root = _write_pack(tmp_path)
    a = SkillCatalog(load_pack(root)).entry()
    b = SkillCatalog(load_pack(root)).entry()
    assert a == b


def test_skill_md_frontmatter(tmp_path):
    import yaml
    pack = load_pack(_write_pack(tmp_path))
    md = generate_skill_md(pack)
    assert md.startswith("---\n")
    fm = yaml.safe_load(md.split("---")[1])
    assert fm["name"] == "tiny-skill"
    assert "Tiny widgets" in fm["description"] and len(fm["description"]) <= 1024
    assert "[Widget](concepts/widget.md)" in md
    assert "secret.md" not in md


@pytest.mark.asyncio
async def test_mcp_skills_enabled(tmp_path):
    from mcp import Client

    pack = load_pack(_write_pack(tmp_path))
    mcp = create_pack_mcp(pack.slug, pack, AsyncMock())
    async with Client(mcp) as client:
        caps = client.server_capabilities
        assert EXTENSION_ID in (caps.extensions or {})
        assert caps.resources is not None
        listed = await _rpc(client, "skills/list", {})
        assert listed["resultType"] == "complete"
        assert listed["cacheScope"] in ("public", "private") and isinstance(listed["ttlMs"], int)
        entry = listed["skills"][0]
        got = await _rpc(client, "skills/get", {"uri": entry["uri"]})
        assert got["skill"] == entry
        for item in entry["resources"]:
            res = await client.read_resource(item["uri"])
            text = res.contents[0].text
            assert "sha256:" + hashlib.sha256(text.encode()).hexdigest() == item["digest"]
            assert len(text.encode()) == item["size"]
        from mcp.shared.exceptions import MCPError
        with pytest.raises(MCPError) as ei:
            await _rpc(client, "skills/get", {"uri": "skill://nope/SKILL.md"})
        assert ei.value.error.code == -32602
        with pytest.raises(Exception):
            await client.read_resource("skill://tiny-skill/concepts/secret.md")
        # ep_* tools still present and primary
        tools = {t.name for t in (await client.list_tools()).tools}
        assert "ep_search_tool" in tools and "ep_read_tool" in tools


@pytest.mark.asyncio
async def test_mcp_skills_disabled_by_default(tmp_path):
    from mcp import Client

    pack = load_pack(_write_pack(tmp_path, enabled=False))
    mcp = create_pack_mcp(pack.slug, pack, AsyncMock())
    async with Client(mcp) as client:
        assert EXTENSION_ID not in (client.server_capabilities.extensions or {})
        with pytest.raises(Exception):
            await _rpc(client, "skills/list", {})
