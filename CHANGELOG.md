# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Graph-aware retrieval** in `ep_search` — expands results by following `_graph.yaml` edges (BFS, configurable depth/discount)
- **`ep_graph_traverse` MCP tool** — standalone graph exploration (depth 1-3, edge_kind filtering)
- `GraphLookup` helper for bidirectional file_path ↔ node_id mapping
- `SQLiteStore.get_chunks_by_file_paths()` for batch chunk lookup by file path
- `RetrievalConfig` fields: `graph_expansion_enabled`, `graph_expansion_depth`, `graph_expansion_discount`
- 14 new unit tests for graph retrieval and traversal (76 total)

### Changed
### Fixed
### Removed

## [0.1.0] - 2026-04-11

### Added
- Complete implementation of ExpertPack MCP server from scratch
- **VISION.md** v0.1 through v0.3 (expertise-as-a-service thesis, MCP integration, measurable MVP criteria)
- **ARCHITECTURE.md** v0.1 (14 sections covering modular design, multi-pack routing, hybrid retrieval pipeline, indexing, embedding providers, auth, transport, and testing strategy)
- Pack loader with manifest parsing, frontmatter extraction, and provenance model
- EP-native chunker (file-as-chunk by default, heading-based split for oversized files)
- SQLite index manager with FTS5 + sqlite-vec, incremental content-hash based updates
- Hybrid retrieval engine with score fusion, metadata boosting, and MMR re-ranking
- `ep_search` and `ep_list_topics` MCP tools with structured output schemas
- MCP Resources for manifest, file listing, and raw file access
- FastMCP + Streamable HTTP transport (primary) with path-based multi-pack routing (`/packs/{slug}/mcp`)
- stdio transport support for local development
- Phase 1 API key authentication (per-pack keys)
- Configurable embedding providers (Gemini default, Azure OpenAI, OpenAI)
- CLI entrypoint (`ep-mcp serve --config config.yaml`)
- Comprehensive logging, error handling, and `--log-file` option
- Initial test scaffolding and evaluation framework

### Changed
- Bumped `candidate_multiplier` from 4 to 8 for improved recall before MMR
- Corrected embedding dimension handling (768 → 3072 for Gemini) and updated deployment target from EZT Help Bot to dedicated ExpertPack droplet (165.245.136.51)

### Fixed
- Embedding dimension mismatch between index and runtime provider
- Various request logging and error handling edge cases

### Evaluation
- Achieved **90.9% retrieval hit rate** on 22-question benchmark against `ezt-designer` pack (845 chunks)

**Initial release**: Vision → full working MCP server + production deployment in 3 days.

[0.1.0]: https://github.com/brianhearn/ExpertPack_MCP/releases/tag/v0.1.0
