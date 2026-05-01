# ep-mcp

**Expertise-as-a-Service over the Model Context Protocol.**

An MCP server that turns any ExpertPack into a live, queryable knowledge service. Any MCP-compatible agent (Claude Desktop, Cursor, Windsurf, Claude.ai, custom hosts) can connect and retrieve high-quality, provenance-rich domain expertise via standardized tools.

**[🌐 expertpack.ai](https://expertpack.ai)** · **[📦 ExpertPack Framework](https://github.com/brianhearn/expert-pack)** · **[📖 Architecture](ARCHITECTURE.md)**

97.6% retrieval hit rate / 0 full misses on a 22-question benchmark against a production v4.1 atomic-conceptual pack (658 chunks). Prior v4.0 baseline on the same pack and eval set: 65.1%.

## Features

- **Schema-aware hybrid retrieval**: BM25 + vector search (sqlite-vec), metadata boosting, MMR re-ranking, adaptive threshold filtering, length penalty
- **`requires:` expansion (schema v4.1 atomic-conceptual)**: When a top-K atom's frontmatter declares `requires: [B, C]`, the referenced atoms are auto-appended to results as additional `SearchResult`s flagged `requires_expanded=True`. Directional (A→B does not imply B→A), transitive up to 2 hops, capped at 3 atoms and 3,500 cumulative tokens, appended after the `max_results` slice so it never displaces top-K.
- **Intent-aware routing**: Automatic query classification (ENTITY/HOW/WHY/WHEN/GENERAL) adjusts vector/BM25 fusion weights per query — no LLM required
- **Query embedding cache**: SQLite-backed cache keyed by `(model, dimension, sha256(query))` — warm query latency drops from ~4s (live Gemini API) to ~1ms
- **Deep graph traversal**: Optional multi-hop BFS expansion over knowledge graph edges with per-hop score decay and drift prevention
- **Reserved-slot graph merge**: Configurable protected slots in the final top-K reserved for graph-expanded neighbors so related files surface even against dominant core results
- **BM25 fallback mode**: Embedding provider outage degrades gracefully to lexical-only search (coverage + density + path + length scoring) instead of a hard error
- **Cross-encoder reranker**: Optional second-pass precision reranker (disabled by default, ~70ms warm latency when enabled)
- **Provenance-first**: Every result includes `id`, `content_hash`, `verified_at`, `source_file`
- **EP-native chunking**: Files are treated as atomic retrieval units (split at `##` only for oversized content)
- **Frontmatter aware**: Strips metadata for embedding, extracts `type`, `tags`, `requires`, etc. for boosting/expansion/filtering
- **Incremental indexing**: Content-hash based staleness detection — only re-embeds changed files
- **Multi-pack support**: Path-based routing at `/packs/{slug}/mcp`
- **Graph-aware retrieval**: optional post-retrieval expansion via `_graph.yaml` adjacency
- **Tools**:
  - `ep_search` — primary hybrid retrieval tool
  - `ep_list_topics` — browse pack structure and available content types
  - `ep_graph_traverse`
- **Resources**:
  - `ep://{slug}/manifest`
  - `ep://{slug}/files`
  - `ep://{slug}/file/{path}` (raw content with frontmatter)
- **Transports**: Streamable HTTP (primary, cloud-ready), stdio (local dev)
- **HTTP endpoint**: `GET /search` (for non-MCP HTTP clients)
- **Auth**: API key (Phase 1), designed for OAuth 2.1 (Phase 2)

## Quick Start

### Installation

```bash
git clone https://github.com/brianhearn/ep-mcp.git
cd ep-mcp
pip install -e .[dev]
```

### Configuration

Create `config.yaml` with at minimum:

```yaml
server:
  host: "127.0.0.1"
  port: 8000
  log_level: "info"

packs:
  - slug: "my-pack"
    path: "/path/to/your/expertpack"
    api_keys:
      - "your-secret-key-here"

embedding:
  provider: "gemini"           # "gemini" (default) or "azure-openai"
  model: "gemini-embedding-001"
```

For the full configuration reference — embedding providers, retrieval tuning (`mmr_lambda`, `min_score`, `default_max_results`, graph expansion, reranker, `requires:` expansion, etc.), production systemd setup, and deployment gotchas — see **[DEPLOYMENT.md](DEPLOYMENT.md)**.

### Running

```bash
ep-mcp serve --config config.yaml
```

The server builds/updates the SQLite index (FTS5 + sqlite-vec) on startup.

The GET `/search` HTTP endpoint is available at:

`GET /search?q=<query>&pack=<slug>&n=10`

with header `Authorization: Bearer <key>`.

## Environment Variables

Only the key for your configured embedding provider is required.

| Variable                  | Required | Description |
|---------------------------|----------|-------------|
| `GEMINI_API_KEY`          | If `embedding.provider: gemini` | Gemini embedding API key |
| `AZURE_OPENAI_ENDPOINT`   | If `embedding.provider: azure-openai` | Azure OpenAI resource endpoint |
| `AZURE_OPENAI_API_KEY`    | If `embedding.provider: azure-openai` | Azure OpenAI API key |
| `EP_MCP_KEY_{SLUG}`       | Optional | Per-pack API key override (uppercase slug, hyphens→underscores) |
| `EP_MCP_REMOTE_HOST`      | Optional | Deploy target for `scripts/deploy.sh` |
| `EP_MCP_REMOTE_SRC`       | Optional | Remote source path for `deploy.sh` |
| `EP_MCP_REMOTE_SITE_PKG`  | Optional | Remote site-packages path for `deploy.sh` |

## Tech Stack

- **Language**: Python 3.12+
- **MCP Framework**: FastMCP (from the official `mcp` SDK)
- **Storage**: SQLite with FTS5 + [sqlite-vec](https://github.com/asg017/sqlite-vec)
- **Embeddings**: Gemini (default), configurable Azure OpenAI / OpenAI providers
- **Web**: Starlette + Uvicorn (for Streamable HTTP)
- **CLI**: Click

See `pyproject.toml` for exact dependencies.

## Project Structure

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed module breakdown, retrieval pipeline, indexing strategy, and design decisions.

## Related

- **[expertpack.ai](https://expertpack.ai)** — Framework website, community packs, and documentation
- **[ExpertPack](https://github.com/brianhearn/expert-pack)** — Schema framework, validation tools, Obsidian compatibility, community packs
- **[ExpertPacks](https://github.com/brianhearn/expert-packs)** — Published knowledge packs (private)
- **EasyTerritory MCP** (future) — Domain-specific layer built on top of this repo + `ezt-designer` pack

## License

Apache 2.0 © 2026 Brian Hearn

## Author

**Brian Hearn** — Built in 3 days (April 10–11, 2026).
