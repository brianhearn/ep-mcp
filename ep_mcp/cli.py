"""CLI entry point for EP MCP server."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import click

from .config import EmbeddingConfig, PackConfig, ServerConfig, load_config
from .index.manager import IndexManager
from .index.sqlite_store import SQLiteStore
from .pack.loader import PackLoadError, load_pack
from .pack.manifest import ManifestError, parse_manifest


def _auto_embedding_config(choice: str | None) -> EmbeddingConfig:
    """Resolve an EmbeddingConfig from CLI choice or available environment keys.

    Auto order:
      1. GEMINI_API_KEY           -> gemini / gemini-embedding-001
      2. OPENAI_API_KEY/BASE_URL  -> openai / text-embedding-3-small
      3. AZURE_OPENAI_*           -> azure-openai / text-embedding-3-large
      4. None available           -> lexical (zero-key offline BM25-only)
    """
    sel = (choice or "auto").strip().lower()
    if sel == "lexical":
        return EmbeddingConfig(provider="lexical", model="bm25-only")
    if sel == "gemini":
        return EmbeddingConfig(provider="gemini", model="gemini-embedding-001")
    if sel == "openai":
        return EmbeddingConfig(provider="openai", model="text-embedding-3-small")
    if sel == "azure-openai":
        return EmbeddingConfig(provider="azure-openai", model="text-embedding-3-large")

    # auto-detect from environment
    if os.environ.get("GEMINI_API_KEY"):
        return EmbeddingConfig(provider="gemini", model="gemini-embedding-001")
    if os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_BASE_URL"):
        return EmbeddingConfig(provider="openai", model="text-embedding-3-small")
    if os.environ.get("AZURE_OPENAI_API_KEY") and os.environ.get("AZURE_OPENAI_ENDPOINT"):
        return EmbeddingConfig(provider="azure-openai", model="text-embedding-3-large")
    return EmbeddingConfig(provider="lexical", model="bm25-only")


def _build_zero_config(
    pack_paths: tuple[str, ...],
    embedding_choice: str | None = None,
    host: str = "127.0.0.1",
    port: int = 8000,
    feedback_dir: str | None = None,
) -> ServerConfig:
    """Build a ServerConfig directly from one or more pack directories (no config YAML needed)."""
    packs: list[PackConfig] = []
    seen_slugs: set[str] = set()
    for raw_path in pack_paths:
        p = Path(raw_path).resolve()
        try:
            manifest = parse_manifest(p / "manifest.yaml")
        except ManifestError as exc:
            raise click.ClickException(f"Invalid pack at {p}: {exc}") from exc
        slug = manifest.slug
        if slug in seen_slugs:
            raise click.ClickException(f"Duplicate pack slug '{slug}' across pack paths")
        seen_slugs.add(slug)
        packs.append(PackConfig(slug=slug, path=str(p)))

    return ServerConfig(
        host=host,
        port=port,
        packs=packs,
        embedding=_auto_embedding_config(embedding_choice),
        feedback_dir=feedback_dir,
    )


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable debug logging")
@click.option("--log-file", "-l", default=None, help="Write logs to file (in addition to stderr)")
def cli(verbose: bool, log_file: str | None) -> None:
    """ExpertPack MCP Server — expertise-as-a-service over MCP."""
    level = logging.DEBUG if verbose else logging.INFO
    handlers: list[logging.Handler] = [
        logging.StreamHandler(),
    ]
    if log_file:
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
    )


@cli.command()
@click.argument("pack_paths", nargs=-1, type=click.Path(exists=True, file_okay=False))
@click.option("--config", "-c", "config_path", default=None, help="Path to config YAML (optional when pack paths are given)")
@click.option(
    "--transport", "-t", default=None, type=click.Choice(["http", "stdio"]),
    help="MCP transport (default: stdio for zero-config pack paths, http with --config)",
)
@click.option(
    "--embedding", "-e", "embedding_choice", default=None,
    type=click.Choice(["auto", "lexical", "gemini", "openai", "azure-openai"]),
    help="Embedding provider override (default: auto-detect keys, fallback to zero-key lexical BM25)",
)
@click.option("--host", default=None, help="Bind host for HTTP transport (default: 127.0.0.1)")
@click.option("--port", default=None, type=int, help="Bind port for HTTP transport (default: 8000)")
@click.option("--feedback-dir", default=None, help="Directory for ep_feedback JSONL queue")
def serve(
    pack_paths: tuple[str, ...],
    config_path: str | None,
    transport: str | None,
    embedding_choice: str | None,
    host: str | None,
    port: int | None,
    feedback_dir: str | None,
) -> None:
    """Start the EP MCP server (supports zero-config `ep-mcp serve ./my-pack`)."""
    if not config_path and not pack_paths:
        raise click.UsageError("Provide at least one PACK_PATH or --config <config.yaml>.")

    if config_path:
        try:
            config = load_config(config_path)
        except (FileNotFoundError, ValueError) as e:
            click.echo(f"Error loading config: {e}", err=True)
            sys.exit(1)
        if embedding_choice:
            config.embedding = _auto_embedding_config(embedding_choice)
        if host:
            config.host = host
        if port is not None:
            config.port = port
        if feedback_dir:
            config.feedback_dir = feedback_dir
        effective_transport = transport or "http"
    else:
        config = _build_zero_config(
            pack_paths,
            embedding_choice=embedding_choice,
            host=host or "127.0.0.1",
            port=port if port is not None else 8000,
            feedback_dir=feedback_dir,
        )
        effective_transport = transport or "stdio"

    if not config.packs:
        click.echo("Error: no packs configured", err=True)
        sys.exit(1)

    # stdout is the JSON-RPC channel on stdio: status lines must go to stderr.
    _stdio = effective_transport == "stdio"

    click.echo(err=_stdio, message=f"Starting EP MCP server on {config.host}:{config.port}")
    click.echo(err=_stdio, message=f"Transport: {effective_transport}")
    click.echo(err=_stdio, message=f"Packs: {[p.slug for p in config.packs]}")
    click.echo(err=_stdio, message=f"Embedding: {config.embedding.provider}/{config.embedding.model}")

    import asyncio
    import uvicorn
    from .embeddings.cache import QueryEmbeddingCache
    from .server import build_app, create_embedding_provider, init_pack

    async def _load_instances():
        provider = create_embedding_provider(config)
        # Wrap with query embedding cache when vector embeddings are active
        if provider.dimension > 0:
            first_pack = config.packs[0]
            _cache_base = (
                Path(first_pack.index_dir)
                if first_pack.index_dir
                else Path(first_pack.path) / ".ep-mcp"
            )
            provider = QueryEmbeddingCache(provider, cache_path=_cache_base / "query_embed_cache.db")
        pack_instances = {}
        for pack_config in config.packs:
            click.echo(err=_stdio, message=f"Loading pack: {pack_config.slug} from {pack_config.path}")
            inst = await init_pack(
                slug=pack_config.slug,
                pack_path=pack_config.path,
                provider=provider,
                config=config,
                index_dir=pack_config.index_dir,
            )
            pack_instances[pack_config.slug] = inst
            click.echo(
                err=_stdio,
                message=(
                    f"  \u2705 {inst.pack.name}: {len(inst.pack.files)} files, "
                    f"{inst.store.chunk_count()} chunks "
                    f"(integrity={inst.pack.integrity_status})"
                ),
            )
        return pack_instances

    if effective_transport == "stdio":
        instances = asyncio.run(_load_instances())
        from .server import create_multi_pack_mcp

        multi_mcp = create_multi_pack_mcp(
            instances,
            query_log_path=config.query_log_path,
            feedback_dir=config.feedback_dir,
        )

        if len(config.packs) == 1:
            click.echo(
                f"Starting stdio MCP server for pack '{config.packs[0].slug}'",
                err=True,
            )
        else:
            click.echo(
                f"Starting multi-pack stdio MCP server with {len(config.packs)} packs: "
                f"{', '.join(p.slug for p in config.packs)}",
                err=True,
            )

        multi_mcp.run(transport="stdio")
        return

    async def _startup():
        instances = await _load_instances()
        return build_app(config, instances, dev_watch=config.dev_mode_watch)

    app = asyncio.run(_startup())
    click.echo(f"\nServer ready at http://{config.host}:{config.port}")
    click.echo("Endpoints:")
    click.echo("  GET  /health")
    click.echo("  GET  /packs")
    click.echo("  GET  /search?q=<query>&pack=<slug>&n=<max_results>")
    for p in config.packs:
        click.echo(f"  POST /packs/{p.slug}/mcp")

    uvicorn.run(app, host=config.host, port=config.port, log_level=config.log_level)


@cli.command()
@click.option("--pack", "-p", "pack_path", required=True, help="Path to ExpertPack directory")
def validate(pack_path: str) -> None:
    """Validate that a pack can be loaded and check integrity/security."""
    from .pack.security import scan_pack_security, verify_pack_integrity

    try:
        pack = load_pack(pack_path)
    except PackLoadError as e:
        click.echo(f"❌ Pack load failed: {e}", err=True)
        sys.exit(1)

    report = verify_pack_integrity(pack.pack_dir, pack.manifest.raw, slug=pack.slug)
    findings = scan_pack_security(pack.pack_dir, pack.manifest.raw)

    click.echo(f"✅ Pack loaded: {pack.name} ({pack.slug})")
    click.echo(f"   Type: {pack.type}")
    click.echo(f"   Version: {pack.version}")
    click.echo(f"   Integrity: {report.status} (digest_verified={report.digest_verified})")
    click.echo(f"   Security lint findings: {len(findings)}")
    click.echo(f"   Files: {len(pack.files)}")
    click.echo(f"   Total tokens: {sum(f.size_tokens for f in pack.files.values()):,}")
    if pack.graph:
        click.echo(f"   Graph: {len(pack.graph.nodes)} nodes, {len(pack.graph.edges)} edges")
    else:
        click.echo("   Graph: not present")

    with_prov = sum(1 for f in pack.files.values() if f.provenance.id)
    click.echo(f"   Provenance: {with_prov}/{len(pack.files)} files have IDs")

    always = sum(1 for f in pack.files.values() if f.retrieval_strategy == "always")
    on_demand = sum(1 for f in pack.files.values() if f.retrieval_strategy == "on_demand")
    click.echo(f"   Context tiers: {always} always, {on_demand} on_demand")

    if report.status == "unverifiable":
        for issue in report.issues:
            click.echo(f"   ❌ {issue}", err=True)
        sys.exit(1)


@cli.command()
@click.option("--pack", "-p", "pack_path", required=True, help="Path to ExpertPack directory")
@click.option("--config", "-c", "config_path", default=None, help="Path to config YAML (for embedding/security settings)")
@click.option(
    "--embedding", "-e", "embedding_choice", default=None,
    type=click.Choice(["auto", "lexical", "gemini", "openai", "azure-openai"]),
    help="Embedding provider override (default: config or auto-detect / lexical fallback)",
)
def index(pack_path: str, config_path: str | None, embedding_choice: str | None) -> None:
    """Index a pack without starting the server."""
    import asyncio
    from .server import create_embedding_provider

    try:
        pack = load_pack(pack_path)
    except PackLoadError as e:
        click.echo(f"\u274c Pack load failed: {e}", err=True)
        sys.exit(1)

    click.echo(f"Indexing: {pack.name} ({len(pack.files)} files)")

    if config_path:
        config = load_config(config_path)
        if embedding_choice:
            config.embedding = _auto_embedding_config(embedding_choice)
    else:
        config = ServerConfig(embedding=_auto_embedding_config(embedding_choice))

    provider = create_embedding_provider(config)
    click.echo(f"Embedding: {provider.model_name} ({provider.dimension}d)")

    store = SQLiteStore(pack.index_path, embedding_dimension=provider.dimension)
    store.open()

    try:
        manager = IndexManager(pack, store, provider, security_config=config.security)
        stats = asyncio.run(manager.build_index())
        click.echo(f"\u2705 Indexed: {stats} (integrity={pack.integrity_status})")
    finally:
        store.close()


@cli.command()
@click.option("--config", "-c", "config_path", required=True, help="Path to config YAML")
def info(config_path: str) -> None:
    """Show server configuration info."""
    try:
        config = load_config(config_path)
    except (FileNotFoundError, ValueError) as e:
        click.echo(f"Error loading config: {e}", err=True)
        sys.exit(1)

    click.echo("EP MCP Server Configuration")
    click.echo(f"  Host: {config.host}")
    click.echo(f"  Port: {config.port}")
    click.echo(f"  Embedding: {config.embedding.provider}/{config.embedding.model}")
    click.echo(f"  Security: integrity={config.security.on_integrity_failure}, "
               f"lint={config.security.check_security_lint} ({config.security.on_security_finding})")
    click.echo(f"  Retrieval: vector={config.retrieval.vector_weight}, "
               f"text={config.retrieval.text_weight}, "
               f"mmr={'on' if config.retrieval.mmr_enabled else 'off'}")
    click.echo(f"  Packs ({len(config.packs)}):")
    for p in config.packs:
        click.echo(f"    - {p.slug}: {p.path}")


if __name__ == "__main__":
    cli()
