"""CLI entry point for EP MCP server."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import click

from .config import load_config
from .pack.loader import PackLoadError, load_pack


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

    click.echo(f"Starting EP MCP server on {config.host}:{config.port}")
    click.echo(f"Transport: {transport}")
    click.echo(f"Packs: {[p.slug for p in config.packs]}")
    click.echo(f"Embedding: {config.embedding.provider}/{config.embedding.model}")

    # TODO: Wire up server.py, index manager, transport
    click.echo("Server implementation pending — scaffold only")


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
