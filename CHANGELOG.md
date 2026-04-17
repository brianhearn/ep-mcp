# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **BM25 fallback mode** — when the embedding provider fails (quota, outage, network error), the engine now gracefully degrades to BM25-only search instead of returning an error. Uses a richer lexical scorer (`score_bm25_fallback`) that augments the normalized BM25 score with term coverage, token density, path boost, and length boost — similar to OpenClaw's `scoreFallbackKeywordResult()`. Adaptive threshold, metadata boosts, and length penalty still apply. MMR and graph expansion are skipped (both require embeddings). Clients see results with lower confidence rather than a hard failure.
- **File watcher (dev mode)** — optional live reindex when pack source files change. Enabled via `server.dev_mode_watch: true` in config. Uses `watchdog` (install separately: `pip install watchdog`) to watch the pack directory for `.md` file changes and triggers a debounced incremental reindex (2s debounce). Designed for Obsidian authoring workflows. Not recommended for production. The watcher runs in a background thread, debounced reindex runs on the asyncio event loop. `PackInstance` now stores `index_manager` reference for watcher reuse.
- **Parallel batch embedding** — `GeminiEmbeddingProvider.embed()` now fires up to 4 concurrent `embed_content` requests during bulk index builds (controlled by `max_parallel_batches=4`). Single-query and single-batch calls take the fast path (no parallelism overhead). For packs with hundreds of chunks split across multiple batches, this reduces index build time proportionally to the batch count.
- **`dev_mode_watch` config field** — `ServerConfig.dev_mode_watch: bool = false`. Parsed from `server.dev_mode_watch` in YAML config.

### Fixed
- **BM25 score normalization** — replaced relative min-max normalization with the absolute transform `abs(rank) / (1 + abs(rank))`. Min-max inflated weak BM25 matches to 1.0 when all results scored low (e.g. no strong keyword hits), distorting fusion. The absolute transform ensures BM25 scores are comparable across queries: a 0.9 always means strong keyword overlap, regardless of what other results are in the set. Verified: monotonically increasing with relevance, no per-query variance.

### Changed
- **`absolute_floor` default raised** from 0.10 → 0.20. The previous value was too permissive — it allowed garbage results to pass when the adaptive ratio cutoff was low. OpenClaw uses a flat 0.35 hard minimum; 0.20 is a better balance between recall and precision for domain-specific packs. Overridable per deployment in `retrieval.absolute_floor`.
- **`build_app()` accepts `dev_watch: bool`** parameter (default `False`). CLI passes `config.dev_mode_watch` through.
- **`PackInstance` stores `index_manager`** — retained as `instance.index_manager` for use by the file watcher and future hot-reload scenarios.

### Added
- **Intent-aware routing** — new `IntentClassifier` in `retrieval/intent.py` classifies queries into ENTITY/HOW/WHY/WHEN/GENERAL intents using regex heuristics (zero LLM dependency). The engine adjusts vector/BM25 fusion weights per intent: ENTITY queries boost BM25 (0.45/0.55), HOW/WHY queries boost vector (0.80/0.20), WHEN queries boost BM25 (0.40/0.60), GENERAL uses configured defaults. Controlled by `retrieval.intent_routing_enabled` (default: `true`).
- **Deep graph traversal** — optional multi-hop BFS graph expansion alongside the existing 1-hop (shallow) mode. Controlled by `retrieval.graph_expansion_deep` (default: `false`) and `retrieval.graph_expansion_deep_max_bonus` (default: 5). When enabled, BFS traverses up to `graph_expansion_depth` hops with per-hop score decay (`graph_expansion_discount` per hop). Only neighbors clearing `min_score` after discount can seed the next hop, preventing drift. Capped at `graph_expansion_deep_max_bonus` bonus results.
- **Adaptive threshold filtering** — ratio-based score filtering replaces the brittle flat `min_score` cutoff. Two-step filter: activation floor (if best score is below floor, return empty) + score ratio (keep results within ratio of best score). Controlled by `retrieval.adaptive_threshold` (default: `true`), `activation_floor` (0.15), `score_ratio` (0.55), `absolute_floor` (0.10). Adapts to pack size, embedding model, and query specificity.
- **Length penalty** — multiplicative discount for very short chunks (stubs, headings, navigation artefacts) below a configurable character threshold. `retrieval.length_penalty_threshold` (default: 80 chars), `retrieval.length_penalty_factor` (default: 0.15). Short chunks can still win on pure relevance but at a discounted rate.
- **`GET /search` endpoint** — lightweight HTTP search for non-MCP clients (web_fetch, curl, etc.). Query params: `q`, `pack`, `n`, `type`, `tags`. Requires Bearer token when API keys are configured for the pack. Returns `source_file`, `title`, `text`, `score`, `type`, `tags` per result.
- **`scripts/deploy.sh`** — rsync-based deploy script that copies `ep_mcp/` source directly to both `/opt/ep-mcp/ep_mcp/` and site-packages on the ExpertPack droplet. Replaces the previous tar+pip install pattern which corrupted entrypoint shebangs (shebang was rewritten to EasyBot's local venv path). Supports `--restart-only` flag for service restarts without file copy.
- **Graph expansion logging** — INFO-level log lines per query: seeds selected (with threshold), total neighbor candidates evaluated, bonus results appended with file paths. DEBUG-level log for each rejected neighbor with cosine score. Provides full per-query visibility for threshold tuning.

### Fixed
- **BM25 stopword filtering** — `_sanitize_fts5_query()` now strips ~100 English stopwords before building the FTS5 AND query. Previously, natural language queries returned 0 BM25 results because every token (including question words like "What", "does", "how") had to match literally. Tokens shorter than 3 characters are also filtered. Falls back to the 3 longest original tokens if all tokens are stopwords.
- **MMR score preservation** — `mmr_rerank()` now uses the MMR formula only for selection order but emits the original relevance score. Previously, MMR rewrote scores using the diversity-penalized MMR value, which collapsed scores from ~0.47 to ~0.08 on domain-specific packs where all chunks have high inter-similarity.
- **Vector-anchored adaptive threshold** — `apply_adaptive_threshold()` accepts an optional `anchor_score` parameter. The engine now passes `vector_weight * best_vec_score` as the anchor, preventing BM25 keyword spikes from inflating the ratio-based cutoff and killing genuine vector-matched results.
- **Pipeline observability** — added INFO-level logging at every pipeline stage: after fusion, after boosts, after MMR (top-20 chunks with file paths and scores), threshold filter counts, and adaptive threshold parameters.

### Changed
- **min_score gating in graph expansion** — now applied to the *final* score (cosine × effective bonus) instead of raw cosine similarity alone. This ensures per-hop decay in deep mode is reflected in the filtering threshold. Shallow mode is equivalent (structural bonus defaults to 1.0).
- **Graph expansion refactored** — `_apply_graph_expansion()` delegates to `_shallow_graph_expansion()` or `_deep_graph_expansion()` based on config. Shared `_score_neighbor()` helper eliminates duplication.

### Added (previous)
- **Graph-aware retrieval** in `ep_search` — additive post-top-K expansion; neighbors scored independently against query embedding (cosine similarity), appended as bonus results with `graph_expanded=True` flag; does not interfere with core top-K selection
- **`ep_graph_traverse` MCP tool** — standalone graph exploration (depth 1-3, edge_kind filtering)
- `GraphLookup` helper for bidirectional file_path ↔ node_id mapping
- `SQLiteStore.get_chunks_by_file_paths()` for batch chunk lookup by file path
- `graph_expanded: bool` field on `SearchResult` — marks bonus graph neighbors in response
- `RetrievalConfig` fields: `graph_expansion_enabled`, `graph_expansion_depth`, `graph_expansion_discount`, `graph_expansion_min_score`, `graph_expansion_confidence_threshold`, `graph_expansion_structural_bonus`
- Unit tests for graph retrieval and traversal (76 total)

### Changed
- **Graph expansion architecture**: replaced pre-MMR BFS/discount injection with post-top-K additive expansion. Seeds = top-K results scoring above `graph_expansion_confidence_threshold` (default 0.38). Neighbors scored via independent cosine similarity — no score inheritance from parent. Eliminates displacement problem where broad files (FAQ, overview) pulled in irrelevant neighbors that crowded out correct MMR results.
- Graph expansion default: **disabled** (`graph_expansion_enabled: false`) — enable once confidence threshold is tuned to your score distribution

### Fixed
- Graph neighbor dedup: only add primary chunk (chunk_index=0) per neighbor file, preventing oversized split files from flooding results

### Removed
- Pre-MMR graph injection (replaced by post-top-K additive design)

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
- Corrected embedding dimension handling (768 → 3072 for Gemini) and updated deployment target from help bot to dedicated ExpertPack droplet (production server)

### Fixed
- Embedding dimension mismatch between index and runtime provider
- Various request logging and error handling edge cases

### Evaluation
- Achieved **90.9% retrieval hit rate** on 22-question benchmark against `my-pack` pack (845 chunks)

**Initial release**: Vision → full working MCP server + production deployment in 3 days.

[0.1.0]: https://github.com/brianhearn/ep-mcp/releases/tag/v0.1.0
