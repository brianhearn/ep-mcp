"""FastMCP server setup, tool/resource registration, multi-pack routing."""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Mount, Route

from .auth import APIKeyAuth
from .config import ServerConfig
from .embeddings.base import EmbeddingProvider
from .embeddings.gemini import GeminiEmbeddingProvider
from .index.manager import IndexManager
from .index.sqlite_store import SQLiteStore
from .pack.loader import load_pack
from .pack.models import Pack
from .retrieval.engine import RetrievalEngine
from .tools.ep_list_topics import ep_list_topics
from .tools.ep_search import ep_search

logger = logging.getLogger(__name__)


class PackInstance:
    """A fully initialized pack with all its components."""

    def __init__(self, pack, store, engine, mcp):
        self.pack = pack
        self.store = store
        self.engine = engine
        self.mcp = mcp


def create_embedding_provider(config: ServerConfig) -> EmbeddingProvider:
    """Create the configured embedding provider."""
    emb = config.embedding
    if emb.provider == "gemini":
        return GeminiEmbeddingProvider(model=emb.model)
    raise ValueError(f"Unsupported embedding provider: {emb.provider}")


def create_pack_mcp(slug: str, pack: Pack, engine: RetrievalEngine):
    """Create a FastMCP instance with tools registered for a pack."""
    from mcp.server.fastmcp import FastMCP

    mcp = FastMCP(
        name=f"ep-mcp-{slug}",
        instructions=f"ExpertPack MCP server for {pack.name}. "
        f"Use ep_search to find domain expertise and ep_list_topics to browse pack structure.",
        stateless_http=True,
    )

    @mcp.tool(
        annotations={
            "readOnlyHint": True,
            "idempotentHint": True,
            "openWorldHint": False,
        }
    )
    async def ep_search_tool(
        query: str,
        type: str | None = None,
        tags: list[str] | None = None,
        max_results: int = 10,
    ) -> str:
        """Search the ExpertPack for relevant domain expertise.

        Args:
            query: Natural language search query.
            type: Filter by content type (concept, workflow, reference,
                  troubleshooting, faq, specification, etc.)
            tags: Filter by content tags. Results must match at least one.
            max_results: Maximum results to return (1-50, default 10).

        Returns:
            JSON array of ranked results with provenance metadata.
        """
        try:
            results = await ep_search(engine, query, type, tags, max_results)
            return json.dumps(results, indent=2)
        except Exception as e:
            logger.exception(
                "ep_search_tool error | pack=%s query=%r", slug, query,
            )
            return json.dumps({
                "error": str(e),
                "pack": slug,
                "query": query,
            })

    @mcp.tool(
        annotations={
            "readOnlyHint": True,
            "idempotentHint": True,
            "openWorldHint": False,
        }
    )
    async def ep_list_topics_tool(
        type: str | None = None,
    ) -> str:
        """List available topics and content structure in the ExpertPack.

        Args:
            type: Filter by content type. If omitted, returns all types.

        Returns:
            JSON with pack metadata and grouped file listing.
        """
        try:
            result = ep_list_topics(pack, type)
            return json.dumps(result, indent=2)
        except Exception as e:
            logger.exception(
                "ep_list_topics_tool error | pack=%s type=%s", slug, type,
            )
            return json.dumps({"error": str(e), "pack": slug})

    @mcp.resource(f"ep://{slug}/manifest")
    async def get_manifest() -> str:
        """Get the pack's manifest.yaml as JSON."""
        return json.dumps(pack.manifest.raw, indent=2, default=str)

    @mcp.resource(f"ep://{slug}/files")
    async def get_file_listing() -> str:
        """Get the pack's file listing with metadata."""
        files = []
        for path, f in sorted(pack.files.items()):
            files.append({
                "path": path,
                "title": f.title,
                "type": f.type,
                "tags": f.tags,
                "tokens": f.size_tokens,
                "has_provenance": f.provenance.id is not None,
            })
        return json.dumps(files, indent=2)

    return mcp


async def init_pack(
    slug: str,
    pack_path: str,
    provider: EmbeddingProvider,
    config: ServerConfig,
) -> PackInstance:
    """Load, index, and initialize a pack with MCP tools."""
    pack = load_pack(pack_path, slug_override=slug)
    store = SQLiteStore(pack.index_path, embedding_dimension=provider.dimension)
    store.open()

    manager = IndexManager(pack, store, provider)
    stats = await manager.build_index()
    logger.info("Pack '%s' indexed: %s", slug, stats)

    engine = RetrievalEngine(pack, store, provider, config.retrieval)
    mcp = create_pack_mcp(slug, pack, engine)

    return PackInstance(pack=pack, store=store, engine=engine, mcp=mcp)


def build_app(config: ServerConfig, pack_instances: dict[str, PackInstance]) -> Starlette:
    """Build the Starlette ASGI application with pack routing.

    Each pack's FastMCP app is mounted as a sub-application with its own
    lifespan managed through the parent app's lifespan.
    """
    auth = APIKeyAuth()
    for pack_config in config.packs:
        if pack_config.api_keys:
            auth.add_pack_keys(pack_config.slug, pack_config.api_keys)

    # Collect MCP session managers for lifespan management
    session_managers = []
    for slug, inst in pack_instances.items():
        mcp_app = inst.mcp.streamable_http_app()
        session_managers.append((slug, inst.mcp.session_manager, mcp_app))

    @asynccontextmanager
    async def lifespan(app):
        """Manage all pack MCP session managers."""
        # Build nested context managers for each session manager
        from contextlib import AsyncExitStack
        async with AsyncExitStack() as stack:
            for slug, sm, _ in session_managers:
                logger.info("Starting session manager for pack '%s'", slug)
                await stack.enter_async_context(sm.run())
            yield

    async def health(request: Request) -> JSONResponse:
        packs_info = {}
        for slug, inst in pack_instances.items():
            packs_info[slug] = {
                "name": inst.pack.name,
                "files": len(inst.pack.files),
                "chunks": inst.store.chunk_count(),
                "version": inst.pack.version,
            }
        return JSONResponse({"status": "healthy", "packs": packs_info})

    async def list_packs(request: Request) -> JSONResponse:
        packs = []
        for slug, inst in pack_instances.items():
            packs.append({
                "slug": slug,
                "name": inst.pack.name,
                "type": inst.pack.type,
                "version": inst.pack.version,
                "file_count": len(inst.pack.files),
            })
        return JSONResponse({"packs": packs})

    routes = [
        Route("/health", health),
        Route("/packs", list_packs),
    ]

    for slug, sm, mcp_app in session_managers:
        routes.append(Mount(f"/packs/{slug}", app=mcp_app))
        logger.info("Mounted MCP endpoint: /packs/%s/mcp", slug)

    return Starlette(routes=routes, lifespan=lifespan)
