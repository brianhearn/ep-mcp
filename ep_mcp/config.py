"""Server configuration for EP MCP."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class PackConfig(BaseModel):
    """Configuration for a single pack."""

    slug: str
    path: str
    api_keys: list[str] = Field(default_factory=list)

    # Optional pack-level overrides for graph expansion (fall back to RetrievalConfig globals)
    graph_expansion_enabled: bool | None = None
    graph_expansion_confidence_threshold: float | None = None
    graph_expansion_min_score: float | None = None
    graph_expansion_structural_bonus: float | None = None


class EmbeddingConfig(BaseModel):
    """Embedding provider configuration."""

    provider: str = "gemini"
    model: str = "gemini-embedding-001"
    # Azure-specific
    azure_endpoint: str | None = None
    azure_api_version: str = "2024-10-21"
    azure_deployment: str | None = None


class RetrievalConfig(BaseModel):
    """Retrieval pipeline configuration."""

    vector_weight: float = 0.7
    text_weight: float = 0.3
    candidate_multiplier: int = 8
    mmr_enabled: bool = True
    mmr_lambda: float = 0.7
    min_score: float = 0.35
    default_max_results: int = 10
    type_match_boost: float = 0.05
    tag_match_boost: float = 0.03
    always_tier_boost: float = 0.02
    graph_expansion_enabled: bool = False
    graph_expansion_depth: int = 1
    graph_expansion_discount: float = 0.85
    graph_expansion_min_score: float = 0.20
    graph_expansion_confidence_threshold: float = 0.38
    graph_expansion_structural_bonus: float = 1.0


class ServerConfig(BaseModel):
    """Top-level server configuration."""

    host: str = "127.0.0.1"
    port: int = 8000
    log_level: str = "info"
    packs: list[PackConfig] = Field(default_factory=list)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)


def load_config(config_path: str | Path) -> ServerConfig:
    """Load server configuration from a YAML file.

    Args:
        config_path: Path to ep-mcp-config.yaml

    Returns:
        Parsed ServerConfig

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config is invalid
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in config: {e}") from e

    if not isinstance(raw, dict):
        raise ValueError("Config must be a YAML mapping")

    # Extract server-level config
    server_raw = raw.get("server", {})

    # Parse pack configs
    packs = []
    for pack_raw in raw.get("packs", []):
        packs.append(PackConfig(**pack_raw))

    # Parse embedding config
    embedding_raw = raw.get("embedding", {})
    embedding = EmbeddingConfig(**embedding_raw)

    # Parse retrieval config
    retrieval_raw = raw.get("retrieval", {})
    retrieval = RetrievalConfig(**retrieval_raw)

    return ServerConfig(
        host=server_raw.get("host", "127.0.0.1"),
        port=server_raw.get("port", 8000),
        log_level=server_raw.get("log_level", "info"),
        packs=packs,
        embedding=embedding,
        retrieval=retrieval,
    )
