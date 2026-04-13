# ep-mcp

**Expertise-as-a-Service over the Model Context Protocol.**

An MCP server that turns any ExpertPack into a live, queryable knowledge service. Any MCP-compatible agent (Claude Desktop, Cursor, Windsurf, Claude.ai, custom hosts) can connect and retrieve high-quality, provenance-rich domain expertise via standardized tools.

**[🌐 expertpack.ai](https://expertpack.ai)** · **[📦 ExpertPack Framework](https://github.com/brianhearn/ExpertPack)** · **[📖 Architecture](ARCHITECTURE.md)**

>95.5% retrieval hit rate on a 22-question benchmark against a real production pack (826 chunks).

## Features

- **Schema-aware hybrid retrieval**: BM25 + vector search (sqlite-vec), metadata boosting, MMR re-ranking
- **Provenance-first**: Every result includes `id`, `content_hash`, `verified_at`, `source_file`
- **EP-native chunking**: Files are treated as atomic retrieval units (split at `##` only for oversized content)
- **Frontmatter aware**: Strips metadata for embedding, extracts `type`, `tags`, etc. for boosting/filtering
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

Create `config.yaml`:

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
  provider: "gemini"
  model: "gemini-embedding-001"
  azure_endpoint: null
  azure_api_version: "2024-10-21"
  azure_deployment: null

retrieval:
  vector_weight: 0.7
  text_weight: 0.3
  candidate_multiplier: 8
  mmr_enabled: true
  mmr_lambda: 0.7
  min_score: 0.35
  default_max_results: 10
  type_match_boost: 0.05
  tag_match_boost: 0.03
  always_tier_boost: 0.02
  graph_expansion_enabled: false
  graph_expansion_depth: 1
  graph_expansion_discount: 0.85
  graph_expansion_min_score: 0.20
  graph_expansion_confidence_threshold: 0.38
  graph_expansion_structural_bonus: 1.0
```

**Note:** All `retrieval:` fields are optional — defaults shown above.

### Running

```bash
ep-mcp serve --config config.yaml
```

The server builds/updates the SQLite index (FTS5 + sqlite-vec) on startup.

The GET `/search` HTTP endpoint is available at:

`GET /search?q=<query>&pack=<slug>&n=10`

with header `Authorization: Bearer <key>`.

## Environment Variables

| Variable                  | Required          | Description |
|---------------------------|-------------------|-----------|
| `GEMINI_API_KEY`          | Yes (Gemini)      | Gemini embedding API key |
| `AZURE_OPENAI_API_KEY`    | Yes (Azure)       | Azure OpenAI API key |
| `EP_MCP_KEY_{SLUG}`       | Optional          | Per-pack API key override (uppercase slug, hyphens→underscores) |
| `EP_MCP_REMOTE_HOST`      | Optional          | Deploy target for `scripts/deploy.sh` |
| `EP_MCP_REMOTE_SRC`       | Optional          | Remote source path for `deploy.sh` |
| `EP_MCP_REMOTE_SITE_PKG`  | Optional          | Remote site-packages path for `deploy.sh` |

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
- **[ExpertPack](https://github.com/brianhearn/ExpertPack)** — Schema framework, validation tools, Obsidian compatibility, community packs
- **[ExpertPacks](https://github.com/brianhearn/ExpertPacks)** — Published knowledge packs (private)
- **EasyTerritory MCP** (future) — Domain-specific layer built on top of this repo + `ezt-designer` pack

## License

Apache 2.0 © 2026 Brian Hearn

## Author

**Brian Hearn** — Built in 3 days (April 10–11, 2026).
