"""Tests for multi-pack stdio MCP server routing."""

import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from ep_mcp.pack.loader import load_pack
from ep_mcp.server import create_multi_pack_mcp, PackInstance


def _payload(result) -> dict | list:
    """Extract payload from MCP result.
    
    The MCP SDK wraps tool returns in structured_content as {'result': <actual_value>}.
    We unwrap this to get the actual tool return value.
    """
    if result.structured_content:
        data = result.structured_content
        # Unwrap the MCP SDK's "result" wrapper
        if isinstance(data, dict) and "result" in data and len(data) == 1:
            return data["result"]
        return data
    text = result.content[0].text
    return json.loads(text)


@pytest.fixture
def pack_a(tmp_path: Path):
    pack_dir = tmp_path / "pack-a"
    pack_dir.mkdir()
    (pack_dir / "manifest.yaml").write_text(
        """
slug: pack-a
name: Pack A
type: product
version: "1.0.0"
description: Test pack A
entry_point: overview.md
context:
  always:
    - overview.md
""",
        encoding="utf-8",
    )
    (pack_dir / "overview.md").write_text(
        """---
id: pack-a/overview
type: concept
---
# Pack A Overview
Content from Pack A.
""",
        encoding="utf-8",
    )
    return load_pack(pack_dir)


@pytest.fixture
def pack_b(tmp_path: Path):
    pack_dir = tmp_path / "pack-b"
    pack_dir.mkdir()
    (pack_dir / "manifest.yaml").write_text(
        """
slug: pack-b
name: Pack B
type: product
version: "1.0.0"
description: Test pack B
entry_point: intro.md
context:
  always:
    - intro.md
""",
        encoding="utf-8",
    )
    (pack_dir / "intro.md").write_text(
        """---
id: pack-b/intro
type: concept
---
# Pack B Introduction
Content from Pack B.
""",
        encoding="utf-8",
    )
    return load_pack(pack_dir)


def _make_pack_instance(pack):
    """Create a minimal PackInstance with mocked components."""
    engine = AsyncMock()
    engine.pack = pack
    store = AsyncMock()
    mcp = AsyncMock()
    return PackInstance(pack=pack, store=store, engine=engine, mcp=mcp)


@pytest.mark.asyncio
async def test_multi_pack_search_with_pack_param(pack_a, pack_b):
    """Test ep_search_tool with explicit pack parameter."""
    from mcp import Client
    from ep_mcp.retrieval.models import SearchResult
    
    instances = {
        "pack-a": _make_pack_instance(pack_a),
        "pack-b": _make_pack_instance(pack_b),
    }
    
    # Mock search to return SearchResult objects
    async def mock_search_a(request):
        return [
            SearchResult(
                text="Result from Pack A",
                source_file="overview.md",
                score=0.9,
                id="pack-a/overview",
                content_hash="abc123",
                verified_at="2026-01-01",
                type="concept",
                tags=[],
                title="Overview",
                chunk_index=0,
            )
        ]
    
    async def mock_search_b(request):
        return [
            SearchResult(
                text="Result from Pack B",
                source_file="intro.md",
                score=0.8,
                id="pack-b/intro",
                content_hash="def456",
                verified_at="2026-01-01",
                type="concept",
                tags=[],
                title="Intro",
                chunk_index=0,
            )
        ]
    
    instances["pack-a"].engine.search = AsyncMock(side_effect=mock_search_a)
    instances["pack-b"].engine.search = AsyncMock(side_effect=mock_search_b)
    
    mcp = create_multi_pack_mcp(instances)
    
    async with Client(mcp) as client:
        # Search pack-a
        result_a = await client.call_tool("ep_search_tool", {"query": "test", "pack": "pack-a"})
        payload_a = _payload(result_a)
        assert isinstance(payload_a, list)
        assert len(payload_a) == 1
        assert payload_a[0]["text"] == "Result from Pack A"
        
        # Search pack-b
        result_b = await client.call_tool("ep_search_tool", {"query": "test", "pack": "pack-b"})
        payload_b = _payload(result_b)
        assert isinstance(payload_b, list)
        assert len(payload_b) == 1
        assert payload_b[0]["text"] == "Result from Pack B"


@pytest.mark.asyncio
async def test_multi_pack_invalid_pack_error(pack_a, pack_b):
    """Test that invalid pack slugs return clear errors."""
    from mcp import Client
    
    instances = {
        "pack-a": _make_pack_instance(pack_a),
        "pack-b": _make_pack_instance(pack_b),
    }
    
    mcp = create_multi_pack_mcp(instances)
    
    async with Client(mcp) as client:
        result = await client.call_tool("ep_search_tool", {"query": "test", "pack": "invalid-pack"})
        payload = _payload(result)
        assert "error" in payload
        assert "Unknown pack: 'invalid-pack'" in payload["error"]
        assert "pack-a" in payload["error"]
        assert "pack-b" in payload["error"]


@pytest.mark.asyncio
async def test_single_pack_default_behavior(pack_a):
    """Test that pack parameter is optional when only one pack is configured."""
    from mcp import Client
    from ep_mcp.retrieval.models import SearchResult
    
    instances = {
        "pack-a": _make_pack_instance(pack_a),
    }
    
    async def mock_search(request):
        return [
            SearchResult(
                text="Result from Pack A",
                source_file="overview.md",
                score=0.9,
                id="pack-a/overview",
                content_hash="abc123",
                verified_at="2026-01-01",
                type="concept",
                tags=[],
                title="Overview",
                chunk_index=0,
            )
        ]
    
    instances["pack-a"].engine.search = AsyncMock(side_effect=mock_search)
    
    mcp = create_multi_pack_mcp(instances)
    
    async with Client(mcp) as client:
        # Should work without pack parameter
        result = await client.call_tool("ep_search_tool", {"query": "test"})
        payload = _payload(result)
        assert isinstance(payload, list)
        assert len(payload) == 1
        assert payload[0]["text"] == "Result from Pack A"
        
        # Should also work with explicit pack parameter
        result = await client.call_tool("ep_search_tool", {"query": "test", "pack": "pack-a"})
        payload = _payload(result)
        assert isinstance(payload, list)
        assert len(payload) == 1


@pytest.mark.asyncio
async def test_multi_pack_missing_pack_error(pack_a, pack_b):
    """Test that missing pack parameter returns clear error for multi-pack configs."""
    from mcp import Client
    
    instances = {
        "pack-a": _make_pack_instance(pack_a),
        "pack-b": _make_pack_instance(pack_b),
    }
    
    mcp = create_multi_pack_mcp(instances)
    
    async with Client(mcp) as client:
        result = await client.call_tool("ep_search_tool", {"query": "test"})
        payload = _payload(result)
        assert "error" in payload
        assert "Missing required parameter 'pack'" in payload["error"]
        assert "pack-a" in payload["error"]
        assert "pack-b" in payload["error"]


@pytest.mark.asyncio
async def test_multi_pack_list_topics(pack_a, pack_b):
    """Test ep_list_topics_tool with pack routing."""
    from mcp import Client
    
    instances = {
        "pack-a": _make_pack_instance(pack_a),
        "pack-b": _make_pack_instance(pack_b),
    }
    
    mcp = create_multi_pack_mcp(instances)
    
    async with Client(mcp) as client:
        # List topics for pack-a
        result_a = await client.call_tool("ep_list_topics_tool", {"pack": "pack-a"})
        payload_a = _payload(result_a)
        assert "pack" in payload_a
        assert payload_a["pack"]["slug"] == "pack-a"
        assert payload_a["pack"]["name"] == "Pack A"
        
        # List topics for pack-b
        result_b = await client.call_tool("ep_list_topics_tool", {"pack": "pack-b"})
        payload_b = _payload(result_b)
        assert "pack" in payload_b
        assert payload_b["pack"]["slug"] == "pack-b"
        assert payload_b["pack"]["name"] == "Pack B"


@pytest.mark.asyncio
async def test_multi_pack_read(pack_a, pack_b):
    """Test ep_read_tool with pack routing."""
    from mcp import Client
    
    instances = {
        "pack-a": _make_pack_instance(pack_a),
        "pack-b": _make_pack_instance(pack_b),
    }
    
    mcp = create_multi_pack_mcp(instances)
    
    async with Client(mcp) as client:
        # Read from pack-a
        result_a = await client.call_tool("ep_read_tool", {"pack": "pack-a", "path": "overview.md"})
        payload_a = _payload(result_a)
        assert "Content from Pack A" in payload_a.get("content", "")
        
        # Read from pack-b
        result_b = await client.call_tool("ep_read_tool", {"pack": "pack-b", "path": "intro.md"})
        payload_b = _payload(result_b)
        assert "Content from Pack B" in payload_b.get("content", "")


@pytest.mark.asyncio
async def test_multi_pack_graph_traverse(pack_a, pack_b):
    """Test ep_graph_traverse_tool with pack routing."""
    from mcp import Client
    
    instances = {
        "pack-a": _make_pack_instance(pack_a),
        "pack-b": _make_pack_instance(pack_b),
    }
    
    mcp = create_multi_pack_mcp(instances)
    
    async with Client(mcp) as client:
        # Traverse in pack-a (no graph, should return empty result)
        result_a = await client.call_tool(
            "ep_graph_traverse_tool", 
            {"pack": "pack-a", "file_path": "overview.md"}
        )
        payload_a = _payload(result_a)
        assert "start_node" in payload_a
        assert payload_a["start_node"]["file"] == "overview.md"
        
        # Invalid pack
        result_invalid = await client.call_tool(
            "ep_graph_traverse_tool",
            {"pack": "invalid", "file_path": "test.md"}
        )
        payload_invalid = _payload(result_invalid)
        assert "error" in payload_invalid
        assert "Unknown pack: 'invalid'" in payload_invalid["error"]


@pytest.mark.asyncio
async def test_multi_pack_all_tools_registered(pack_a, pack_b):
    """Test that all expected tools are registered in multi-pack mode."""
    from mcp import Client
    
    instances = {
        "pack-a": _make_pack_instance(pack_a),
        "pack-b": _make_pack_instance(pack_b),
    }
    
    mcp = create_multi_pack_mcp(instances)
    
    expected_tools = {
        "ep_search_tool",
        "ep_list_topics_tool",
        "ep_graph_traverse_tool",
        "ep_read_tool",
    }
    
    async with Client(mcp) as client:
        tools = await client.list_tools()
        tool_names = {t.name for t in tools.tools}
        assert expected_tools <= tool_names
