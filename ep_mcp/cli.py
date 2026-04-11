"""CLI entry point for EP MCP server."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import click

from .config import load_config
from .pack.loader import PackLoadError, load_pack
from .index.sqlite_store import SQLiteStore
from .index.manager import IndexManager


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable debug logging")
def cli(verbose: bool) -> None:
    """ExpertPack MCP Server — expertise-as-a-service over MCP."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


@cli.command()
@click.option("--config", "-c", "config_path", required=True, help="Path to config YAML")
@click.option("--transport", "-t", default="http", type=click.Choice(["http", "stdio"]))
def serve(config_path: str, transport: str) -> None:
    """Start the EP MCP server."""
    try:
        config = load_config(config_path)
    except (FileNotFoundError, ValueError) as e:
        click.echo(f"Error loading config: {e}", err=True)
        sys.exit(1)

    if not config.packs:
        click.echo("Error: no packs configured", err=True)
        sys.exit(1)

    click.echo(f"Starting EP MCP server on {config.host}:{config.port}")
    click.echo(f"Transport: {transport}")
    click.echo(f"Packs: {[p.slug for p in config.packs]}")
    click.echo(f"Embedding: {config.embedding.provider}/{config.embedding.model}")

    import asyncio
    import uvicorn
    from .server import build_app, create_embedding_provider, init_pack

    async def startup():
        provider = create_embedding_provider(config)
        pack_instances = {}
        for pack_config in config.packs:
            click.echo(f"Loading pack: {pack_config.slug} from {pack_config.path}")
            inst = await init_pack(
                slug=pack_config.slug,
                pack_path=pack_config.path,
                provider=provider,
                config=config,
            )
            pack_instances[pack_config.slug] = inst
            click.echo(f"  \u2705 {inst.pack.name}: {len(inst.pack.files)} files, "
                       f"{inst.store.chunk_count()} chunks")
        return build_app(config, pack_instances)

    app = asyncio.run(startup())
    click.echo(f"\nServer ready at http://{config.host}:{config.port}")
    click.echo("Endpoints:")
    click.echo(f"  GET  /health")
    click.echo(f"  GET  /packs")
    for p in config.packs:
        click.echo(f"  POST /packs/{p.slug}/mcp")

    uvicorn.run(app, host=config.host, port=config.port, log_level=config.log_level)


@cli.command()
@click.option("--pack", "-p", "pack_path", required=True, help="Path to ExpertPack directory")
def validate(pack_path: str) -> None:
    """Validate that a pack can be loaded."""
    try:
        pack = load_pack(pack_path)
    except PackLoadError as e:
        click.echo(f"❌ Pack load failed: {e}", err=True)
        sys.exit(1)

    click.echo(f"✅ Pack loaded: {pack.name} ({pack.slug})")
    click.echo(f"   Type: {pack.type}")
    click.echo(f"   Version: {pack.version}")
    click.echo(f"   Files: {len(pack.files)}")
    click.echo(f"   Total tokens: {sum(f.size_tokens for f in pack.files.values()):,}")
    if pack.graph:
        click.echo(f"   Graph: {len(pack.graph.nodes)} nodes, {len(pack.graph.edges)} edges")
    else:
        click.echo("   Graph: not present")

    # Provenance stats
    with_prov = sum(1 for f in pack.files.values() if f.provenance.id)
    click.echo(f"   Provenance: {with_prov}/{len(pack.files)} files have IDs")

    # Context tier stats
    always = sum(1 for f in pack.files.values() if f.retrieval_strategy == "always")
    on_demand = sum(1 for f in pack.files.values() if f.retrieval_strategy == "on_demand")
    click.echo(f"   Context tiers: {always} always, {on_demand} on_demand")


@cli.command()
@click.option("--pack", "-p", "pack_path", required=True, help="Path to ExpertPack directory")
@click.option("--config", "-c", "config_path", default=None, help="Path to config YAML (for embedding settings)")
def index(pack_path: str, config_path: str | None) -> None:
    """Index a pack without starting the server."""
    import asyncio

    try:
        pack = load_pack(pack_path)
    except PackLoadError as e:
        click.echo(f"\u274c Pack load failed: {e}", err=True)
        sys.exit(1)

    click.echo(f"Indexing: {pack.name} ({len(pack.files)} files)")

    # Load embedding config
    if config_path:
        config = load_config(config_path)
        emb_config = config.embedding
    else:
        from .config import EmbeddingConfig
        emb_config = EmbeddingConfig()

    # Create embedding provider
    provider = _create_embedding_provider(emb_config)
    click.echo(f"Embedding: {provider.model_name} ({provider.dimension}d)")

    # Create store and index
    store = SQLiteStore(pack.index_path, embedding_dimension=provider.dimension)
    store.open()

    try:
        manager = IndexManager(pack, store, provider)
        stats = asyncio.run(manager.build_index())
        click.echo(f"\u2705 Indexed: {stats}")
    finally:
        store.close()


def _create_embedding_provider(emb_config):
    """Create an embedding provider from config."""
    if emb_config.provider == "gemini":
        from .embeddings.gemini import GeminiEmbeddingProvider
        return GeminiEmbeddingProvider(model=emb_config.model)
    else:
        raise ValueError(f"Unsupported embedding provider: {emb_config.provider}")


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
    click.echo(f"  Retrieval: vector={config.retrieval.vector_weight}, "
               f"text={config.retrieval.text_weight}, "
               f"mmr={'on' if config.retrieval.mmr_enabled else 'off'}")
    click.echo(f"  Packs ({len(config.packs)}):")
    for p in config.packs:
        click.echo(f"    - {p.slug}: {p.path}")


if __name__ == "__main__":
    cli()
