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
from .embeddings.ollama import OllamaEmbeddingProvider
from .index.manager import IndexManager
from .index.sqlite_store import SQLiteStore
from .pack.loader import load_pack
from .pack.models import Pack
from .retrieval.engine import RetrievalEngine
from .retrieval.reranker import Reranker
from .retrieval.graph_helpers import GraphLookup
from .prompts.pack_prompts import register_prompts
from .resources.pack_resources import register_resources
from .tools.ep_graph_traverse import ep_graph_traverse
from .tools.ep_list_topics import ep_list_topics
from .tools.ep_search import ep_search

logger = logging.getLogger(__name__)


class PackInstance:
    """A fully initialized pack with all its components."""

    def __init__(self, pack, store, engine, mcp, index_manager=None):
        self.pack = pack
        self.store = store
        self.engine = engine
        self.mcp = mcp
        self.index_manager = index_manager  # retained for file watcher reindex


def create_embedding_provider(config: ServerConfig) -> EmbeddingProvider:
    """Create the configured embedding provider."""
    emb = config.embedding
    if emb.provider == "gemini":
        return GeminiEmbeddingProvider(model=emb.model)
    if emb.provider == "ollama":
        base_url = getattr(emb, "base_url", "http://localhost:11434")
        return OllamaEmbeddingProvider(model=emb.model, base_url=base_url)
    raise ValueError(f"Unsupported embedding provider: {emb.provider}")


def _get_server_instructions(pack: Pack) -> str:
    """Derive the MCP server instructions string for a pack.

    Priority: mcp.instructions in manifest > manifest.description > generic fallback.
    """
    if pack.mcp_config.instructions:
        return pack.mcp_config.instructions
    if pack.description:
        return pack.description
    return (
        f"{pack.name} ExpertPack knowledge service. "
        f"Use ep_search to find domain expertise, ep_list_topics to browse pack structure, "
        f"and ep_graph_traverse to explore knowledge graph connections."
    )


def create_pack_mcp(
    slug: str,
    pack: Pack,
    engine: RetrievalEngine,
    graph_lookup: GraphLookup | None = None,
):
    """Create a FastMCP instance with tools, resources, and prompts registered for a pack."""
    from mcp.server.fastmcp import FastMCP

    mcp = FastMCP(
        name=f"ep-mcp-{slug}",
        instructions=_get_server_instructions(pack),
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

    @mcp.tool(
        annotations={
            "readOnlyHint": True,
            "idempotentHint": True,
            "openWorldHint": False,
        }
    )
    async def ep_graph_traverse_tool(
        file_path: str,
        depth: int = 1,
        edge_kinds: list[str] | None = None,
    ) -> str:
        """Traverse the ExpertPack knowledge graph from a starting file.

        Explores connections between content files (concepts, workflows,
        references, etc.) through the pack's knowledge graph.

        Args:
            file_path: Starting file path (e.g. 'concepts/auto-build.md').
            depth: Number of hops to follow (1-3, default 1).
            edge_kinds: Filter by edge types (wikilink, related, context).
                       If omitted, follows all edge types.

        Returns:
            JSON with start node info, connected nodes, and traversal stats.
        """
        try:
            result = ep_graph_traverse(
                pack=pack,
                graph_lookup=graph_lookup,
                file_path=file_path,
                depth=depth,
                edge_kinds=edge_kinds,
            )
            return json.dumps(result, indent=2)
        except Exception as e:
            logger.exception(
                "ep_graph_traverse_tool error | pack=%s file_path=%r",
                slug, file_path,
            )
            return json.dumps({
                "error": str(e),
                "pack": slug,
                "file_path": file_path,
            })

    # Register resources (always-tier files, overview, manifest, additional declared)
    register_resources(mcp, pack)

    # Register prompts (from mcp.prompts manifest declarations or auto-discovered workflows)
    register_prompts(mcp, pack)

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

    graph_lookup = GraphLookup.from_pack(pack)

    # Build effective retrieval config: start with global, apply pack-level overrides
    pack_cfg = next((p for p in config.packs if p.slug == slug), None)
    retrieval_cfg = config.retrieval
    if pack_cfg is not None:
        overrides = {}
        if pack_cfg.graph_expansion_enabled is not None:
            overrides["graph_expansion_enabled"] = pack_cfg.graph_expansion_enabled
        if pack_cfg.graph_expansion_confidence_threshold is not None:
            overrides["graph_expansion_confidence_threshold"] = pack_cfg.graph_expansion_confidence_threshold
        if pack_cfg.graph_expansion_min_score is not None:
            overrides["graph_expansion_min_score"] = pack_cfg.graph_expansion_min_score
        if pack_cfg.graph_expansion_structural_bonus is not None:
            overrides["graph_expansion_structural_bonus"] = pack_cfg.graph_expansion_structural_bonus
        if overrides:
            retrieval_cfg = retrieval_cfg.model_copy(update=overrides)
            logger.info("Pack '%s' applying retrieval overrides: %s", slug, overrides)

    reranker_cfg = config.reranker
    reranker = Reranker(
        model_name=reranker_cfg.model,
        candidate_pool_size=reranker_cfg.candidate_pool_size,
        enabled=reranker_cfg.enabled,
    )
    engine = RetrievalEngine(pack, store, provider, retrieval_cfg, graph_lookup, reranker=reranker)
    mcp = create_pack_mcp(slug, pack, engine, graph_lookup)

    return PackInstance(pack=pack, store=store, engine=engine, mcp=mcp, index_manager=manager)


def build_app(
    config: ServerConfig,
    pack_instances: dict[str, PackInstance],
    dev_watch: bool = False,
) -> Starlette:
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
        """Manage all pack MCP session managers and optional file watchers."""
        from contextlib import AsyncExitStack
        async with AsyncExitStack() as stack:
            for slug, sm, _ in session_managers:
                logger.info("Starting session manager for pack '%s'", slug)
                await stack.enter_async_context(sm.run())

            # Start file watchers in dev mode
            watchers = []
            if dev_watch:
                try:
                    from .index.watcher import start_watchers
                    loop = asyncio.get_event_loop()
                    watchers = start_watchers(
                        list(pack_instances.values()), loop
                    )
                    if watchers:
                        logger.info(
                            "Dev file watch enabled for %d pack(s)", len(watchers)
                        )
                    else:
                        logger.warning(
                            "Dev file watch requested but no watchers started "
                            "(watchdog installed?)"
                        )
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Could not start file watchers: %s", exc)

            yield

            for w in watchers:
                w.stop()

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

    async def search(request: Request) -> JSONResponse:
        """GET /search?q=<query>&pack=<slug>&n=<max_results>&type=<type>&tags=<tag1,tag2>

        Lightweight HTTP search endpoint for non-MCP clients (e.g. web_fetch, curl).
        Requires Bearer token if API keys are configured for the pack.

        Tuning overrides (optional, for eval/tuning only — not for production use):
          graph_expansion_confidence_threshold=<float>
          graph_expansion_min_score=<float>
        """
        q = request.query_params.get("q", "").strip()
        slug = request.query_params.get("pack", "").strip()
        n = int(request.query_params.get("n", "10"))
        type_filter = request.query_params.get("type", None)
        tags_raw = request.query_params.get("tags", None)
        tags = [t.strip() for t in tags_raw.split(",") if t.strip()] if tags_raw else None

        # Optional tuning overrides
        conf_raw = request.query_params.get("graph_expansion_confidence_threshold", None)
        min_raw = request.query_params.get("graph_expansion_min_score", None)
        conf_override = float(conf_raw) if conf_raw is not None else None
        min_override = float(min_raw) if min_raw is not None else None

        if not q:
            return JSONResponse({"error": "Missing required parameter: q"}, status_code=400)
        if not slug:
            return JSONResponse({"error": "Missing required parameter: pack"}, status_code=400)
        if slug not in pack_instances:
            return JSONResponse({"error": f"Unknown pack: {slug}"}, status_code=404)

        # Auth check
        auth_header = request.headers.get("Authorization", "")
        if not auth.authenticate(auth_header, slug):
            return JSONResponse({"error": "Unauthorized"}, status_code=401)

        try:
            engine = pack_instances[slug].engine
            from .retrieval.models import SearchRequest
            search_req = SearchRequest(query=q, type=type_filter, tags=tags, max_results=n)
            raw_results = await engine.search(
                search_req,
                graph_expansion_confidence_threshold=conf_override,
                graph_expansion_min_score=min_override,
            )
            return JSONResponse({
                "query": q,
                "pack": slug,
                "results": [
                    {
                        "source_file": r.source_file,
                        "title": r.title,
                        "text": r.text,
                        "score": round(r.score, 4),
                        "type": r.type,
                        "tags": r.tags,
                        "graph_expanded": r.graph_expanded,
                    }
                    for r in raw_results
                ],
            })
        except Exception as exc:
            logger.exception("search endpoint error | pack=%s query=%r", slug, q)
            return JSONResponse({"error": str(exc)}, status_code=500)

    routes = [
        Route("/health", health),
        Route("/packs", list_packs),
        Route("/search", search),
    ]

    for slug, sm, mcp_app in session_managers:
        routes.append(Mount(f"/packs/{slug}", app=mcp_app))
        logger.info("Mounted MCP endpoint: /packs/%s/mcp", slug)

    return Starlette(routes=routes, lifespan=lifespan)
