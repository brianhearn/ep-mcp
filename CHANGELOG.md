# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **`requires:` expansion (schema v4.1 atomic-conceptual)** — new `retrieval.requires_expansion_*` config block (enabled by default). When a top-K result's source file declares `requires: [...]` in frontmatter, the listed atoms are appended to the result set as additional `SearchResult`s flagged `requires_expanded=True`. Expansion is transitive up to `requires_expansion_max_depth` hops (default 2), capped at `requires_expansion_max_atoms` atoms (default 3) and `requires_expansion_token_budget` cumulative tokens (default 3500). Directional: retrieving A with `requires: [B]` pulls B; retrieving B alone does NOT pull A. Atoms already in the top-K are never duplicated, but their own `requires:` still propagates. Resolves `requires:` entries against: (1) full provenance id, (2) exact pack-relative path, (3) bare basename in the origin atom's directory, (4) unique basename anywhere in the pack. Unresolvable entries log a warning and are skipped. Expansion is appended AFTER the `max_results` slice — additive only, does not displace top-K. New `SearchResult.requires_expanded: bool = False` field. Parser: `PackFile.requires: list[str]` parsed from frontmatter by `loader.py`.
- **BM25 K-of-N token matching** — `_sanitize_fts5_query()` now accepts a `min_token_match_ratio` parameter (0.0–1.0) and builds the FTS5 MATCH expression accordingly. Default `bm25_min_token_match_ratio: float = 1.0` in `RetrievalConfig` preserves strict AND behavior (all tokens must appear). Lower ratios emit OR-of-AND combinations to allow a legitimately relevant file to match even when one query token is missing. For example, ratio=0.67 with 3 tokens yields `(t1 AND t2) OR (t1 AND t3) OR (t2 AND t3)` — at least 2 of 3 must match. Formula: `k = max(1, int(n * ratio))`, clamped to `n-1` when ratio < 1.0. Token count capped at 6 via `_MAX_BM25_TOKENS` to bound combinatorial blowup (C(6,4)=15 clauses). `bm25_search()` accepts the ratio and passes it through to the sanitizer. Both hybrid and BM25-fallback paths in `retrieval/engine.py` thread `config.bm25_min_token_match_ratio` to the store. **Dormant by default** — see Known Limitations below for why, and when to flip the ratio.
- **Query embedding cache** (`embeddings/cache.py`) — SQLite-backed `QueryEmbeddingCache` wrapper class. Caches `embed_query()` results by `(model_name, dimension, sha256(query))` key. Warm latency drops from ~4s (live Gemini API) to ~1ms. Bulk `embed()` calls (index builds) always pass through uncached. Cache stored at `<first_pack_path>/.ep-mcp/query_embed_cache.db`. Wired in `cli.py` — wraps the provider transparently after creation. Graceful: if cache DB fails to open, provider still works normally.
- **MRL dimension support in `GeminiEmbeddingProvider`** — optional `output_dimensionality` param (default `None` = full 3072d). Passes `config={"output_dimensionality": N}` to the Gemini API. Exposed in `EmbeddingConfig.output_dimensionality`. Note: 768d showed regression on this domain-specific pack (69.8% vs 83.3% at 3072d) despite MTEB benchmark showing only 0.26% loss — general benchmarks don't reflect domain packs with high inter-chunk similarity. Default kept at 3072d.

### Removed
- **Ollama embedding provider** (`embeddings/ollama.py`) — CPU-only performance is impractical (~45s/32-chunk batch). Removed provider file, import, and dead code branch in `server.py`. Ollama service also removed from ExpertPack droplet.
- **Reserved-slot graph expansion merge** — new `retrieval.graph_expansion_reserved_slots` config (default: 2). When > 0, the final top-K is structured as `(max_results - reserved_slots)` core MMR results plus `reserved_slots` best graph-expanded bonuses. Previously, post-K graph expansion merged bonus results into the same pool via merge-sort and almost always lost to core results due to the `structural_bonus` discount (typically 0.5). Reserved slots guarantee related files surface (e.g. interface files for workflow queries, linked concept files) by giving them protected slots competed-for only among themselves. Set to 0 to restore legacy merge-sort behavior.
- **Pre-MMR graph widening** (opt-in) — new `retrieval.graph_widen_enabled` config (default: `false`). When enabled, the engine identifies the top-N fused candidates (post-boost, `graph_widen_max_seeds`, default 5) that clear `graph_widen_min_seed_score` (default 0.38), pulls their 1-hop graph neighbors, scores each neighbor's primary chunk via cosine similarity against the query embedding, and injects qualifying neighbors (clearing `graph_widen_min_neighbor_score`, default 0.25) as regular candidates BEFORE MMR runs. Unlike post-K graph expansion (which appends structurally-discounted bonuses), widen candidates compete on raw cosine score through the full pipeline. Targets the secondary-file miss pattern where a linked file exists in the graph but never enters the vector candidate pool.
- **Cross-encoder reranker** — new `reranker` config section (disabled by default). When enabled, a second-pass cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2` by default) rescores the top `candidate_pool_size` candidates (default 20) before the final top-K slice. Model is lazy-loaded on first query (~80MB download, cached). Adds ~70ms warm latency. Graceful degradation if model fails to load.
- **Ollama embedding provider** — alternative to Gemini for local embeddings. Set `embedding.provider: "ollama"` in config. Not currently recommended for indexing (CPU-only is ~45s/batch) but available for experimentation. Query and index embeddings must use the same model; don't mix providers.
- **BM25 fallback mode** — when the embedding provider fails (quota, outage, network error), the engine now gracefully degrades to BM25-only search instead of returning an error. Uses a richer lexical scorer (`score_bm25_fallback`) that augments the normalized BM25 score with term coverage, token density, path boost, and length boost — similar to OpenClaw's `scoreFallbackKeywordResult()`. Adaptive threshold, metadata boosts, and length penalty still apply. MMR and graph expansion are skipped (both require embeddings). Clients see results with lower confidence rather than a hard failure.
- **File watcher (dev mode)** — optional live reindex when pack source files change. Enabled via `server.dev_mode_watch: true` in config. Uses `watchdog` (install separately: `pip install watchdog`) to watch the pack directory for `.md` file changes and triggers a debounced incremental reindex (2s debounce). Designed for Obsidian authoring workflows. Not recommended for production. The watcher runs in a background thread, debounced reindex runs on the asyncio event loop. `PackInstance` now stores `index_manager` reference for watcher reuse.
- **Parallel batch embedding** — `GeminiEmbeddingProvider.embed()` now fires up to 4 concurrent `embed_content` requests during bulk index builds (controlled by `max_parallel_batches=4`). Single-query and single-batch calls take the fast path (no parallelism overhead). For packs with hundreds of chunks split across multiple batches, this reduces index build time proportionally to the batch count.
- **`dev_mode_watch` config field** — `ServerConfig.dev_mode_watch: bool = false`. Parsed from `server.dev_mode_watch` in YAML config.

### Fixed
- **Stale site-packages `scorer.py` on ExpertPack droplet** — the deployed site-packages copy of `ep_mcp/retrieval/scorer.py` was out of date and missing the `bm25_cap` kwarg on `normalize_bm25_scores()`, so the `bm25_cap: 0.7` setting in `config.yaml` was effectively a no-op in production since the cap was introduced. Re-synced from `/opt/ep-mcp/ep_mcp/` to `/opt/ep-mcp/.venv/lib/python3.12/site-packages/ep_mcp/` and restarted the service. With the cap now actually applied, the top BM25 fused scores dropped from ~0.77 to ~0.69 — the expected behavior. Eval hit rate unchanged at 83.7%.
- **BM25 score normalization** — replaced relative min-max normalization with the absolute transform `abs(rank) / (1 + abs(rank))`. Min-max inflated weak BM25 matches to 1.0 when all results scored low (e.g. no strong keyword hits), distorting fusion. The absolute transform ensures BM25 scores are comparable across queries: a 0.9 always means strong keyword overlap, regardless of what other results are in the set. Verified: monotonically increasing with relevance, no per-query variance.

### Known Limitations
- **K-of-N BM25 matching causes regression when paired with current MMR.** Running the ezt-designer 43-source eval with `bm25_min_token_match_ratio: 0.67` improved BM25 recall as designed — the target file for the canonical motivating case (q007 `con-managed-projects-common-scenarios.md`) moved from "not in BM25 candidates at all" to rank 8 after score fusion. But the enlarged candidate pool altered the downstream score distribution such that MMR, during its diversity-driven selection, dropped the sibling concept file out of the top-20 post-MMR (it has high cosine similarity with `con-managed-projects-constraint-model.md`, which was picked first). Net eval regressed 83.7% → 79.1%. The K-of-N code is committed and reversible via a single config flip, but stays dormant (ratio=1.0) until one of: (a) MMR is replaced with a sibling-aware diversity policy for domain packs, (b) MMR is moved to run after the cross-encoder reranker has finalized precision ordering, or (c) the reranker candidate pool is enlarged so sibling files survive MMR pruning. See `memory/session-state.md` for the full trace.
- **OR-mode BM25 regresses more broadly.** A pure OR implementation (tried as run24) regressed eval to 79.1% by flooding the candidate pool with single-token matches that crowded the reranker's top-20 pool. K-of-N is the principled middle ground when the downstream pipeline is ready for it.

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
