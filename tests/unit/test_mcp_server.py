"""In-memory MCP SDK v2 smoke tests (legacy + 2026-07-28)."""

import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from ep_mcp.pack.loader import load_pack
from ep_mcp.server import create_pack_mcp


def _payload(result) -> dict:
    if result.structured_content:
        return result.structured_content
    text = result.content[0].text
    return json.loads(text)


@pytest.fixture
def tiny_pack(tmp_path: Path):
    (tmp_path / "manifest.yaml").write_text(
        """
slug: tiny
name: Tiny Pack
type: product
version: "1.2.0"
description: Tiny fixture pack
entry_point: overview.md
authority_boundary:
  in_scope: Tiny domain
  out_of_scope:
    - Unrelated topics
context:
  always:
    - overview.md
""",
        encoding="utf-8",
    )
    (tmp_path / "overview.md").write_text(
        """---
id: tiny/overview
type: concept
---
# Tiny
Hello.
""",
        encoding="utf-8",
    )
    return load_pack(tmp_path)


def _make_server(pack):
    engine = AsyncMock()
    return create_pack_mcp(pack.slug, pack, engine)


@pytest.mark.asyncio
async def test_list_tools_modern_and_legacy(tiny_pack):
    from mcp import Client

    mcp = _make_server(tiny_pack)
    expected = {"ep_search_tool", "ep_list_topics_tool", "ep_graph_traverse_tool", "ep_read_tool"}

    async with Client(mcp) as modern:
        modern_tools = {t.name for t in (await modern.list_tools()).tools}
        assert expected <= modern_tools
        result = await modern.call_tool("ep_list_topics_tool", {})
        payload = _payload(result)
        assert payload.get("pack", {}).get("slug") == "tiny"
        assert payload["pack"]["authority_boundary"]["in_scope"] == "Tiny domain"

    async with Client(mcp, mode="legacy") as legacy:
        legacy_tools = {t.name for t in (await legacy.list_tools()).tools}
        assert expected <= legacy_tools
        result = await legacy.call_tool("ep_read_tool", {"path": "overview.md"})
        payload = _payload(result)
        assert "Hello" in payload.get("content", "")
