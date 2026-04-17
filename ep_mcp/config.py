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




class RerankerConfig(BaseModel):
    """Cross-encoder reranker configuration (second-pass precision layer)."""

    enabled: bool = False
    model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    candidate_pool_size: int = 20  # how many candidates to rerank before final slice

class RetrievalConfig(BaseModel):
    """Retrieval pipeline configuration."""

    vector_weight: float = 0.7
    text_weight: float = 0.3
    candidate_multiplier: int = 8
    mmr_enabled: bool = True
    mmr_lambda: float = 0.7
    min_score: float = 0.35  # deprecated: use adaptive threshold fields below
    default_max_results: int = 10

    # Adaptive threshold filtering (replaces flat min_score)
    adaptive_threshold: bool = True
    activation_floor: float = 0.15
    score_ratio: float = 0.55
    absolute_floor: float = 0.20
    type_match_boost: float = 0.05
    tag_match_boost: float = 0.03
    always_tier_boost: float = 0.03
    graph_expansion_enabled: bool = False
    graph_expansion_depth: int = 1
    graph_expansion_discount: float = 0.85
    graph_expansion_min_score: float = 0.20
    graph_expansion_confidence_threshold: float = 0.38
    graph_expansion_structural_bonus: float = 1.0

    # Deep graph traversal — multi-hop BFS expansion (opt-in)
    # When enabled, graph expansion uses BFS up to graph_expansion_depth hops
    # with per-hop score decay (graph_expansion_discount per hop).
    # Max bonus results capped at graph_expansion_deep_max_bonus.
    graph_expansion_deep: bool = False
    graph_expansion_deep_max_bonus: int = 5

    # Length penalty — discount very short chunks (stubs, headings, nav artefacts)
    length_penalty_threshold: int = 80   # chars; chunks below this are penalized
    length_penalty_factor: float = 0.15  # multiplicative penalty (score *= 1 - factor)

    # Intent-aware routing — adjust vector/BM25 weights based on query intent
    intent_routing_enabled: bool = True

    # File-level deduplication — max chunks returned from the same source file
    # Prevents a large/chunky file from monopolizing the top-K results
    max_chunks_per_file: int = 2

    # Reserved slots for graph-expanded bonus results.
    # When > 0, the final top-K is structured as:
    #   max_results - reserved_slots  → core MMR results
    #   reserved_slots                → best graph-expanded bonuses
    # This guarantees related files (e.g. interface files for workflow queries,
    # linked concept files) get into the result set even when the structural_bonus
    # discount would make them lose a raw merge-sort against core results.
    # Set to 0 to disable (use legacy merge-sort behavior).
    graph_expansion_reserved_slots: int = 2

    # Pre-MMR graph widening — pull neighbors of high-confidence fused candidates
    # into the candidate pool BEFORE MMR re-ranking, so they compete on their own
    # merit through the full pipeline (not just as discounted post-K bonuses).
    # Disabled by default because it requires more embedding lookups per query.
    graph_widen_enabled: bool = False
    graph_widen_max_seeds: int = 5
    graph_widen_min_seed_score: float = 0.38
    graph_widen_min_neighbor_score: float = 0.25


class ServerConfig(BaseModel):
    """Top-level server configuration."""

    host: str = "127.0.0.1"
    port: int = 8000
    log_level: str = "info"
    packs: list[PackConfig] = Field(default_factory=list)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    reranker: RerankerConfig = Field(default_factory=RerankerConfig)
    # Dev mode: watch pack source directories for .md changes and trigger live reindex.
    # Requires: pip install watchdog. Not for production use.
    dev_mode_watch: bool = False


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

    # Parse reranker config
    reranker_raw = raw.get("reranker", {})
    reranker = RerankerConfig(**reranker_raw)

    return ServerConfig(
        host=server_raw.get("host", "127.0.0.1"),
        port=server_raw.get("port", 8000),
        log_level=server_raw.get("log_level", "info"),
        dev_mode_watch=server_raw.get("dev_mode_watch", False),
        packs=packs,
        embedding=embedding,
        retrieval=retrieval,
        reranker=reranker,
    )
