# ExpertPack MCP — Architecture

**Version:** 0.1 (Draft)
**Date:** 2026-04-11
**Authors:** Brian Hearn, EasyBot
**Prerequisite:** [VISION.md](VISION.md) v0.3

---

## Decision Log

Decisions made before drafting, confirmed by Brian:

| # | Decision | Choice | Rationale |
|---|----------|--------|-----------|
| 1 | SDK / Language | Python (FastMCP) | Retrieval stack is Python-native (embeddings, sqlite-vec, BM25). MCP Python SDK's FastMCP is mature with native Streamable HTTP support. |
| 2 | Retrieval engine | Embedded SQLite (FTS5 + sqlite-vec) | No external database dependency. Same hybrid approach proven in help bot evals (84.8% correctness). Single-file, portable. |
| 3 | Embedding provider | Configurable interface, Gemini default | Azure OpenAI as first-class provider for EZT production (Microsoft shop). Clean provider abstraction supports any backend. |
| 4 | Server structure | Modular (FastMCP + pack loader + index manager + retrieval engine) | Separation of concerns. Each module testable independently. |
| 5 | Multi-pack routing | Path-based (`/packs/{slug}/mcp`) | Each pack gets its own Streamable HTTP endpoint, index, and resource namespace. Clean URL semantics. |

---

## 1. System Overview

```
┌─────────────────────────────────────────────────────────┐
│                    MCP Clients                          │
│  (Claude Desktop, Cursor, Windsurf, custom agents)      │
└──────────────┬──────────────────────────┬───────────────┘
               │ Streamable HTTP          │ stdio
               │ (cloud/remote)           │ (local dev)
               ▼                          ▼
┌─────────────────────────────────────────────────────────┐
│                  EP MCP Server                          │
│  ┌───────────────────────────────────────────────────┐  │
│  │              Transport Layer                      │  │
│  │   Streamable HTTP (primary) / stdio (secondary)   │  │
│  └───────────────────┬───────────────────────────────┘  │
│                      │                                  │
│  ┌───────────────────▼───────────────────────────────┐  │
│  │              FastMCP Server                       │  │
│  │   Tool registration, resource registration,       │  │
│  │   capability negotiation, session management      │  │
│  └───────────────────┬───────────────────────────────┘  │
│                      │                                  │
│  ┌───────────────────▼───────────────────────────────┐  │
│  │            Pack Router                            │  │
│  │   /packs/{slug}/mcp → pack-specific handler       │  │
│  └──────┬────────────────────────────┬───────────────┘  │
│         │                            │                  │
│  ┌──────▼──────┐              ┌──────▼──────┐          │
│  │  Pack A     │              │  Pack B     │          │
│  │  ┌────────┐ │              │  ┌────────┐ │          │
│  │  │Loader  │ │              │  │Loader  │ │          │
│  │  │Index   │ │              │  │Index   │ │          │
│  │  │Search  │ │              │  │Search  │ │          │
│  │  └────────┘ │              │  └────────┘ │          │
│  └─────────────┘              └─────────────┘          │
│                                                         │
│  ┌─────────────────────────────────────────────────────┐│
│  │           Embedding Provider Interface              ││
│  │  Gemini │ Azure OpenAI │ OpenAI │ Local (future)    ││
│  └─────────────────────────────────────────────────────┘│
│                                                         │
│  ┌─────────────────────────────────────────────────────┐│
│  │              Auth Layer                             ││
│  │  API Key (phase 1) → OAuth 2.1 (phase 2)           ││
│  └─────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────┘
```


---

## 2. Module Structure

```
ep_mcp/
├── __init__.py
├── server.py              # FastMCP server setup, tool/resource registration
├── config.py              # Server configuration (packs, auth, embedding provider)
├── auth.py                # API key validation (phase 1), OAuth 2.1 (phase 2)
├── router.py              # Multi-pack routing: path → pack instance
├── pack/
│   ├── __init__.py
│   ├── loader.py          # Load & validate ExpertPack from disk
│   ├── manifest.py        # Parse manifest.yaml, extract pack metadata
│   └── models.py          # Pydantic models: Pack, PackFile, Manifest, Provenance
├── index/
│   ├── __init__.py
│   ├── manager.py         # Index lifecycle: build, update, check staleness
│   ├── sqlite_store.py    # SQLite FTS5 + sqlite-vec storage layer
│   └── chunker.py         # EP-native chunking (file = chunk, oversized splitting)
├── retrieval/
│   ├── __init__.py
│   ├── engine.py          # Hybrid search orchestration (vector + BM25 + MMR)
│   ├── scorer.py          # Score fusion, MMR re-ranking, metadata boosting
│   └── models.py          # Pydantic models: SearchResult, SearchRequest
├── embeddings/
│   ├── __init__.py
│   ├── base.py            # Abstract EmbeddingProvider interface
│   ├── gemini.py          # Google Gemini (gemini-embedding-001)
│   ├── azure_openai.py    # Azure OpenAI (text-embedding-3-small/large)
│   ├── openai.py          # Direct OpenAI API
│   └── cache.py           # Embedding cache (content_hash → vector)
├── tools/
│   ├── __init__.py
│   ├── ep_search.py       # ep_search tool implementation
│   ├── ep_list_topics.py  # ep_list_topics tool implementation
│   └── ep_graph_traverse.py  # ep_graph_traverse (post-MVP)
├── resources/
│   ├── __init__.py
│   └── pack_resources.py  # MCP resource registration (manifest, file listing)
└── cli.py                 # Entry point: `python -m ep_mcp` or `ep-mcp serve`
```

### Dependency Map

```
cli.py
  → config.py (load server config)
  → server.py (create FastMCP instance)
    → auth.py (middleware)
    → router.py (multi-pack dispatch)
      → pack/loader.py (load each configured pack)
        → pack/manifest.py (parse manifest.yaml)
        → pack/models.py (structured pack data)
      → index/manager.py (build/check indexes)
        → index/chunker.py (prepare content)
        → index/sqlite_store.py (write to SQLite)
        → embeddings/*.py (generate vectors)
      → tools/ep_search.py
        → retrieval/engine.py (query orchestration)
          → index/sqlite_store.py (FTS5 + vec queries)
          → retrieval/scorer.py (fusion + re-rank)
      → tools/ep_list_topics.py
        → pack/models.py (pack structure data)
      → resources/pack_resources.py
        → pack/models.py (file metadata)
```


---

## 3. Pack Loading

### 3.1 Pack Discovery

Packs are declared in server config, not auto-discovered:

```yaml
# ep-mcp-config.yaml
server:
  host: "0.0.0.0"
  port: 8000

packs:
  - slug: "ezt-designer"
    path: "/data/packs/ezt-designer"
    api_keys: ["key_abc123"]          # Per-pack auth (phase 1)
  - slug: "blender-3d"
    path: "/data/packs/blender-3d"
    api_keys: ["key_def456"]

embedding:
  provider: "gemini"                   # or "azure_openai", "openai"
  model: "gemini-embedding-001"        # provider-specific model name
  # Azure-specific (when provider = azure_openai):
  # endpoint: "https://<resource>.openai.azure.com"
  # api_version: "2024-10-21"
  # deployment: "text-embedding-3-small"
```

### 3.2 Load Sequence

On server startup, for each configured pack:

1. **Validate path** — confirm directory exists, contains `manifest.yaml`
2. **Parse manifest** — extract slug, type, version, entry_point, context tiers, freshness metadata
3. **Validate manifest** — check required fields, confirm `slug` matches config
4. **Inventory files** — walk directory tree, collect all `.md` files, parse YAML frontmatter from each
5. **Build pack model** — structured `Pack` object with file inventory, metadata, provenance stats
6. **Check index** — compare pack file inventory + content hashes against existing SQLite index
7. **Index if needed** — full rebuild on first run, incremental on content changes (see §4)

### 3.3 Pack Model

```python
@dataclass
class PackFile:
    """A single content file within an ExpertPack."""
    path: str                  # Relative path within pack (e.g., "concepts/auto-build.md")
    title: str | None          # From frontmatter or first H1
    type: str | None           # From frontmatter: concept, workflow, reference, etc.
    tags: list[str]            # From frontmatter
    id: str | None             # Provenance ID (e.g., "ezt-designer/concepts/auto-build")
    content_hash: str | None   # SHA-256 of content (frontmatter stripped)
    verified_at: str | None    # ISO 8601 verification date
    verified_by: str | None    # "human" or "agent"
    retrieval_strategy: str    # "standard" (default), "always", "on_demand"
    content: str               # Full markdown content (frontmatter stripped)
    raw_content: str           # Full markdown content (frontmatter intact, for resource serving)
    size_tokens: int           # Approximate token count

@dataclass
class Pack:
    """A loaded ExpertPack ready for indexing and serving."""
    slug: str
    name: str
    type: str                  # "person", "product", "process"
    version: str
    description: str
    entry_point: str
    files: dict[str, PackFile] # path → PackFile
    manifest: dict             # Raw parsed manifest.yaml
    graph: dict | None         # Parsed _graph.yaml (if present)
    freshness: dict | None     # Freshness metadata from manifest
    index_path: str            # Path to SQLite index file
```

### 3.4 Frontmatter Handling

**Critical design rule:** Frontmatter serves two different purposes and must be handled accordingly.

- **For indexing/retrieval:** Frontmatter is **stripped** before embedding. Provenance metadata (`id`, `content_hash`, `verified_at`) is noise for semantic search. This was proven in eval Runs 11-12: stripping frontmatter improved correctness by +3.3%.
- **For metadata/filtering:** Frontmatter fields (`type`, `tags`, `pack`, `retrieval_strategy`) are extracted and stored as structured columns in SQLite for filtering and scoring.
- **For resource serving:** Raw content (with frontmatter) is available via MCP Resources for clients that want full file access.


---

## 4. Indexing Pipeline

### 4.1 SQLite Schema

Each pack gets its own SQLite database file: `{pack_slug}.index.db`

```sql
-- Content chunks (EP-native: one file ≈ one chunk, unless oversized)
CREATE TABLE chunks (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    file_path   TEXT NOT NULL,              -- relative path within pack
    chunk_index INTEGER NOT NULL DEFAULT 0, -- 0 for single-chunk files
    content     TEXT NOT NULL,              -- markdown content (frontmatter stripped)
    title       TEXT,
    type        TEXT,                       -- concept, workflow, reference, etc.
    tags        TEXT,                       -- JSON array
    pack_slug   TEXT NOT NULL,
    prov_id     TEXT,                       -- provenance ID
    content_hash TEXT,                      -- SHA-256 for cache/staleness
    verified_at TEXT,                       -- ISO 8601
    verified_by TEXT,
    token_count INTEGER,
    UNIQUE(file_path, chunk_index)
);

-- FTS5 index for BM25 keyword search
CREATE VIRTUAL TABLE chunks_fts USING fts5(
    content,
    title,
    content='chunks',
    content_rowid='id',
    tokenize='porter unicode61'
);

-- Triggers to keep FTS5 in sync
CREATE TRIGGER chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, content, title)
    VALUES (new.id, new.content, new.title);
END;
CREATE TRIGGER chunks_ad AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, content, title)
    VALUES ('delete', old.id, old.content, old.title);
END;
CREATE TRIGGER chunks_au AFTER UPDATE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, content, title)
    VALUES ('delete', old.id, old.content, old.title);
    INSERT INTO chunks_fts(rowid, content, title)
    VALUES (new.id, new.content, new.title);
END;

-- Vector embeddings (sqlite-vec)
-- Dimension depends on embedding provider (e.g., 3072 for Gemini, 1536/3072 for OpenAI)
CREATE VIRTUAL TABLE chunks_vec USING vec0(
    chunk_id INTEGER PRIMARY KEY,
    embedding FLOAT[{dimension}]
);

-- Index metadata
CREATE TABLE index_meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);
-- Stores: pack_version, embedding_model, embedding_dimension,
--         indexed_at, file_count, chunk_count
```

### 4.2 Chunking Strategy

ExpertPack uses **schema-as-chunker**: files are authored as retrieval units. The chunking strategy reflects this:

1. **Default:** One file = one chunk. Most EP files are 200-800 tokens and pass through intact.
2. **Oversized files (>1000 tokens):** Split at markdown heading boundaries (## or ###). Each split retains the file's title as a prefix for context.
3. **Always-load files** (from manifest `context.always`): Indexed but also flagged for guaranteed inclusion in search results when relevant.
4. **On-demand files** (from manifest `context.on_demand`): Indexed normally. The "on demand" designation is informational for the consuming agent, not a retrieval filter.

Token counting uses a fast approximation (whitespace split × 0.75) for indexing decisions. Exact counts are not needed — the 1000-token threshold is a guideline, not a hard boundary.

### 4.3 Incremental Indexing

On startup and optionally on a configurable interval:

1. **Hash check:** For each pack file, compute SHA-256 of content (frontmatter stripped). Compare against `content_hash` stored in the `chunks` table.
2. **Changed files:** Re-chunk, re-embed, upsert into all three tables (chunks, FTS5, vec).
3. **Deleted files:** Remove from all three tables.
4. **New files:** Insert into all three tables.
5. **Embedding model change:** If `index_meta.embedding_model` doesn't match the configured provider/model, trigger a full re-embed (content stays, vectors regenerated).

This means the server can restart with an updated pack directory and only re-index what changed — no full rebuild required unless the embedding model changes.

### 4.4 Embedding Cache

Embeddings are expensive (API calls, latency). The cache layer avoids redundant calls:

```python
class EmbeddingCache:
    """Content-hash → embedding vector cache.
    
    Persisted in the same SQLite database as the index.
    Cache key is (content_hash, model_name, dimension).
    """
    
    def get(self, content_hash: str) -> list[float] | None: ...
    def put(self, content_hash: str, embedding: list[float]) -> None: ...
    def invalidate_model(self, model_name: str) -> None: ...
```

Cache is invalidated when the embedding model changes. Individual entries are invalidated when content changes (new hash).


---

## 5. Retrieval Engine

### 5.1 Query Pipeline

```
Query (string)
    │
    ▼
┌─────────────────┐     ┌─────────────────┐
│  Vector Search   │     │   BM25 Search   │
│  (sqlite-vec)    │     │   (FTS5)        │
│                  │     │                  │
│  query → embed   │     │  query → tokens  │
│  → ANN search    │     │  → BM25 rank     │
│  → top K×M       │     │  → top K×M       │
└────────┬─────────┘     └────────┬─────────┘
         │                        │
         ▼                        ▼
    ┌─────────────────────────────────┐
    │        Score Fusion             │
    │                                 │
    │  hybrid_score =                 │
    │    (vector_weight × vec_score)  │
    │  + (text_weight × bm25_score)  │
    │                                 │
    │  Default: 0.7 vec + 0.3 bm25   │
    └────────────────┬────────────────┘
                     │
                     ▼
    ┌─────────────────────────────────┐
    │     Metadata Boosting           │
    │                                 │
    │  +boost if type matches filter  │
    │  +boost if tags match filter    │
    │  +boost if in context.always    │
    └────────────────┬────────────────┘
                     │
                     ▼
    ┌─────────────────────────────────┐
    │    MMR Re-ranking               │
    │                                 │
    │  Maximal Marginal Relevance     │
    │  λ=0.7 (relevance vs diversity) │
    │  Reduces redundant chunks       │
    └────────────────┬────────────────┘
                     │
                     ▼
    Top K results (default K=10)
```

### 5.2 Search Configuration

Retrieval parameters are set at the server level (not per-query). These match the proven config from help bot eval Run 12:

```python
@dataclass
class RetrievalConfig:
    """Retrieval pipeline configuration."""
    
    # Hybrid search weights
    vector_weight: float = 0.7
    text_weight: float = 0.3
    
    # Candidate multiplier: fetch N × max_results candidates before fusion
    candidate_multiplier: int = 4
    
    # MMR
    mmr_enabled: bool = True
    mmr_lambda: float = 0.7        # Balance relevance vs diversity
    
    # Scoring
    min_score: float = 0.35        # Minimum hybrid score to include
    default_max_results: int = 10  # Default K if not specified in query
    
    # Metadata boosting
    type_match_boost: float = 0.05  # Boost when type filter matches
    tag_match_boost: float = 0.03   # Boost per matching tag
    always_tier_boost: float = 0.02 # Boost for context.always files
```

### 5.3 Search Result Model

```python
@dataclass
class SearchResult:
    """A single search result returned by ep_search."""
    text: str               # Content (frontmatter stripped)
    source_file: str        # Path within pack
    id: str | None          # Provenance ID
    content_hash: str | None
    verified_at: str | None # ISO 8601 freshness signal
    score: float            # Hybrid relevance score (0-1)
    type: str | None        # File type (concept, workflow, etc.)
    tags: list[str]         # File tags
    chunk_index: int        # 0 for whole-file chunks
    title: str | None       # File title
```

### 5.4 Graph-Aware Retrieval (Post-MVP)

When `_graph.yaml` is present, the retrieval engine can optionally expand results by following graph edges:

1. Run standard hybrid search → get top K results
2. For each result, look up adjacent nodes in the graph (wikilink, related, context edges)
3. If adjacent nodes aren't already in results and meet a relevance threshold, include them as "related" results with a lower score
4. MMR re-rank the combined set

This is architecturally supported from day one (graph is loaded in the Pack model) but not wired into the search pipeline for MVP.


---

## 6. Embedding Provider Interface

### 6.1 Abstract Interface

```python
from abc import ABC, abstractmethod

class EmbeddingProvider(ABC):
    """Abstract interface for text embedding providers."""
    
    @property
    @abstractmethod
    def model_name(self) -> str:
        """Unique identifier for cache keying (e.g., 'gemini/gemini-embedding-001')."""
        ...
    
    @property
    @abstractmethod
    def dimension(self) -> int:
        """Output embedding dimension."""
        ...
    
    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a batch of texts.
        
        Implementations must handle:
        - Rate limiting / retry with backoff
        - Batch size limits (split large batches internally)
        - Error propagation with meaningful messages
        """
        ...
    
    async def embed_query(self, query: str) -> list[float]:
        """Embed a single query string. Default calls embed([query])[0].
        
        Override if the provider distinguishes query vs document embeddings.
        """
        result = await self.embed([query])
        return result[0]
```

### 6.2 Provider Implementations

| Provider | Class | Model | Dimensions | Notes |
|----------|-------|-------|------------|-------|
| Gemini | `GeminiEmbeddingProvider` | `gemini-embedding-001` | 3072 | Default for dev/testing. Same provider used in OpenClaw's RAG. |
| Azure OpenAI | `AzureOpenAIEmbeddingProvider` | `text-embedding-3-small` | 1536 | EZT production path. Requires Azure endpoint + API key or Entra ID. Supports adjustable dimensions. |
| Azure OpenAI | `AzureOpenAIEmbeddingProvider` | `text-embedding-3-large` | 3072 | Higher quality, higher cost. Adjustable down to 256 dims. |
| OpenAI | `OpenAIEmbeddingProvider` | `text-embedding-3-small` | 1536 | Direct OpenAI API. Same models as Azure, different auth. |

### 6.3 Configuration

```yaml
# Gemini (default)
embedding:
  provider: "gemini"
  model: "gemini-embedding-001"
  # api_key from GEMINI_API_KEY env var

# Azure OpenAI
embedding:
  provider: "azure_openai"
  model: "text-embedding-3-small"
  azure_endpoint: "https://myresource.openai.azure.com"
  azure_api_version: "2024-10-21"
  azure_deployment: "text-embedding-3-small"
  # api_key from AZURE_OPENAI_API_KEY env var
  # OR use Azure Entra ID (DefaultAzureCredential) when no key is set

# Direct OpenAI
embedding:
  provider: "openai"
  model: "text-embedding-3-small"
  # api_key from OPENAI_API_KEY env var
```

### 6.4 Provider Selection Logic

1. Read `embedding.provider` from config
2. Instantiate the matching provider class
3. Provider reads its own config fields + environment variables for secrets
4. On startup, provider validates connectivity (embed a test string)
5. Provider reports `model_name` and `dimension` — used for index creation and cache keying

**Index portability rule:** An index built with one provider/model cannot be queried with a different one. If `embedding.provider` or `embedding.model` changes, the index must be rebuilt. The `index_meta` table tracks this and triggers automatic re-indexing on mismatch.


---

## 7. MCP Tools

### 7.1 `ep_search` (MVP)

The primary tool. Performs hybrid search against the pack's index.

**Registration:**

```python
@mcp.tool(annotations={"readOnlyHint": True})
async def ep_search(
    query: str,
    type: str | None = None,
    tags: list[str] | None = None,
    max_results: int = 10,
) -> list[SearchResult]:
    """Search the ExpertPack for relevant domain expertise.
    
    Args:
        query: Natural language search query.
        type: Filter by content type (concept, workflow, reference, 
              troubleshooting, faq, specification, fact, etc.)
        tags: Filter by content tags. Results must match at least one.
        max_results: Maximum results to return (1-50, default 10).
    
    Returns:
        Ranked list of content chunks with provenance metadata.
    """
```

**Output schema** (MCP `structuredContent`):

```json
{
  "type": "array",
  "items": {
    "type": "object",
    "properties": {
      "text": {"type": "string", "description": "Content (frontmatter stripped)"},
      "source_file": {"type": "string", "description": "Path within pack"},
      "id": {"type": "string", "description": "Provenance ID"},
      "content_hash": {"type": "string", "description": "SHA-256 for verification"},
      "verified_at": {"type": "string", "description": "ISO 8601 verification date"},
      "score": {"type": "number", "description": "Relevance score (0-1)"},
      "type": {"type": "string", "description": "Content type"},
      "tags": {"type": "array", "items": {"type": "string"}},
      "title": {"type": "string", "description": "Content title"}
    },
    "required": ["text", "source_file", "score"]
  }
}
```

### 7.2 `ep_list_topics` (MVP)

Pack structure discovery. Returns the pack's content hierarchy without performing a search.

```python
@mcp.tool(annotations={"readOnlyHint": True})
async def ep_list_topics(
    type: str | None = None,
) -> dict:
    """List available topics and content structure in the ExpertPack.
    
    Args:
        type: Filter by content type. If omitted, returns all types.
    
    Returns:
        Pack metadata and grouped file listing.
    """
```

**Output:**

```json
{
  "pack": {
    "name": "EasyTerritory Designer",
    "slug": "ezt-designer",
    "type": "product",
    "version": "2.0.0",
    "description": "...",
    "file_count": 296,
    "freshness": {
      "coverage_pct": 100,
      "last_full_review": "2026-04-10"
    }
  },
  "topics": {
    "concept": [
      {"path": "concepts/auto-build.md", "title": "Auto-Build Territories"},
      ...
    ],
    "workflow": [...],
    "troubleshooting": [...],
    ...
  }
}
```

### 7.3 `ep_graph_traverse` (Post-MVP)

Follow relationships in the pack's knowledge graph.

```python
@mcp.tool(annotations={"readOnlyHint": True})
async def ep_graph_traverse(
    file_path: str,
    depth: int = 1,
    edge_kinds: list[str] | None = None,
) -> dict:
    """Traverse the ExpertPack knowledge graph from a starting file.
    
    Args:
        file_path: Starting file path within the pack.
        depth: How many hops to follow (1-3, default 1).
        edge_kinds: Filter by edge type (wikilink, related, context).
                    If omitted, follows all edge types.
    
    Returns:
        Starting node and connected nodes with relationship metadata.
    """
```

### 7.4 Tool Annotations

All EP MCP tools are annotated per MCP spec:

- `readOnlyHint: true` — tools don't modify pack content
- `idempotentHint: true` — same query, same results (within index state)
- `openWorldHint: false` — tools only access the loaded pack, no external calls

---

## 8. MCP Resources

### 8.1 Resource URI Scheme

```
ep://{pack_slug}/manifest          → Pack manifest (YAML)
ep://{pack_slug}/files             → File listing with metadata
ep://{pack_slug}/file/{path}       → Individual file content (raw, with frontmatter)
ep://{pack_slug}/graph             → Knowledge graph (_graph.yaml)
```

### 8.2 Resource Annotations

MCP resource annotations map to EP metadata:

| MCP Annotation | EP Source | Example |
|----------------|-----------|---------|
| `name` | File title (frontmatter or H1) | "Auto-Build Territories" |
| `description` | First paragraph or frontmatter description | "How the auto-build algorithm..." |
| `mimeType` | Always `text/markdown` for content files | `text/markdown` |
| `audience` | EP `retrieval_strategy` | "internal" for on_demand, "public" for standard |
| `lastModified` | `verified_at` from provenance | "2026-04-10T00:00:00Z" |

### 8.3 Resource Templates

Dynamic resources use URI templates per MCP spec:

```python
@mcp.resource("ep://{pack_slug}/file/{path}")
async def get_file(pack_slug: str, path: str) -> str:
    """Get a specific file from the ExpertPack.
    
    Returns raw markdown content including frontmatter.
    Use ep_search for retrieval-optimized content (frontmatter stripped).
    """
```

### 8.4 Resource vs Tool: When to Use Which

| Use Case | Primitive | Why |
|----------|-----------|-----|
| "Search for information about X" | Tool (`ep_search`) | Agent needs ranked, relevant content |
| "Show me the pack structure" | Tool (`ep_list_topics`) | Agent needs structured metadata |
| "Read this specific file" | Resource (`ep://slug/file/path`) | Client/agent knows exactly which file |
| "What does this pack cover?" | Resource (`ep://slug/manifest`) | Direct metadata access |
| "What's connected to this file?" | Tool (`ep_graph_traverse`) | Agent needs computed relationships |


---

## 9. Transport & Authentication

### 9.1 Streamable HTTP (Primary)

The server uses FastMCP's built-in Streamable HTTP transport. Single endpoint per pack:

```
POST /packs/{slug}/mcp     → JSON-RPC requests (tool calls, resource reads)
GET  /packs/{slug}/mcp     → SSE stream (optional, for server-initiated notifications)
DELETE /packs/{slug}/mcp   → Session termination
```

**Session management:** Streamable HTTP sessions are stateless from the server's perspective — no session affinity required. Each request carries its own auth context and pack routing. MCP session IDs (`Mcp-Session-Id` header) are accepted but not required for the MVP tools (all read-only, no multi-turn state).

**ASGI mounting:** The FastMCP app mounts onto a Starlette/Uvicorn ASGI server. This allows:
- Multiple pack endpoints on the same server process
- Health check endpoint at `/health`
- Pack listing endpoint at `/packs` (returns configured pack slugs)
- Standard ASGI middleware for logging, CORS, rate limiting

```python
from starlette.applications import Starlette
from starlette.routing import Mount

app = Starlette(routes=[
    # Health check
    Route("/health", health_endpoint),
    # Pack listing
    Route("/packs", list_packs_endpoint),
    # Per-pack MCP endpoints
    Mount("/packs/ezt-designer", app=ezt_mcp.streamable_http_app()),
    Mount("/packs/blender-3d", app=blender_mcp.streamable_http_app()),
])
```

### 9.2 stdio (Secondary)

For local development and testing. Launched as:

```bash
# Single pack, stdio mode
ep-mcp serve --pack /path/to/ezt-designer --transport stdio

# Used in MCP client config:
{
  "mcpServers": {
    "ezt-designer": {
      "command": "ep-mcp",
      "args": ["serve", "--pack", "/path/to/ezt-designer", "--transport", "stdio"]
    }
  }
}
```

stdio mode serves a single pack (no routing needed). Same server logic, different transport.

### 9.3 Authentication — Phase 1 (API Key)

Simple API key authentication sufficient for internal use and early external deployments.

**Request flow:**

1. Client sends `Authorization: Bearer <api_key>` header with every request
2. Auth middleware extracts the key
3. Key is validated against the pack's configured `api_keys` list
4. If valid, request proceeds with the pack context
5. If invalid or missing, return HTTP 401

**Key properties:**
- Keys are per-pack (different packs can have different keys)
- Keys are configured in the server config file (not hardcoded)
- Keys can also be set via environment variables: `EP_MCP_KEY_{SLUG}=<key>`
- Multiple keys per pack (for key rotation)

```python
class APIKeyAuth:
    """Phase 1 authentication: API key validation."""
    
    async def authenticate(self, request: Request, pack_slug: str) -> bool:
        """Validate API key for the requested pack."""
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return False
        key = auth_header[7:]
        return key in self.pack_keys.get(pack_slug, set())
```

### 9.4 Authentication — Phase 2 (OAuth 2.1, Future)

Aligns with MCP spec's OAuth 2.1 with PKCE support. Enables:
- SSO integration (Azure AD / Entra ID — natural for EZT customers)
- Per-pack scopes (`ep:ezt-designer:read`, `ep:blender-3d:read`)
- Token-based access with expiry and refresh
- Enterprise-managed auth policies

Phase 2 auth is architecturally supported (the auth middleware is a pluggable interface) but not implemented in MVP.

### 9.5 CORS

For browser-based MCP clients (e.g., Claude.ai web):

```python
cors_config = {
    "allow_origins": ["https://claude.ai", "https://*.anthropic.com"],
    "allow_methods": ["GET", "POST", "DELETE", "OPTIONS"],
    "allow_headers": ["Authorization", "Content-Type", "Mcp-Session-Id"],
}
```

Configurable in server config. Default: restrictive (no open CORS).


---

## 10. Deployment

### 10.1 MVP Deployment Target

ExpertPack droplet (`165.245.136.51`):
- Already hosts the `ezt-designer` pack (stripped, provenance backfilled)
- Nginx reverse proxy available for TLS termination
- Python 3.12+ available

**Process architecture:**

```
Internet
    │
    ▼
┌─────────┐
│  Nginx   │  TLS termination, reverse proxy
│  :443    │
└────┬─────┘
     │
     ▼
┌────────────────┐
│  EP MCP Server  │  Uvicorn + Starlette + FastMCP
│  :8000          │
│                 │
│  /health        │  → health check
│  /packs         │  → pack listing
│  /packs/ezt-designer/mcp  → MCP endpoint
└────────────────┘
```

Managed via systemd unit. Single process, single pack for MVP.

### 10.2 CLI Entry Point

```bash
# Start server (Streamable HTTP, production)
ep-mcp serve --config ep-mcp-config.yaml

# Start server (stdio, local dev)
ep-mcp serve --pack /path/to/pack --transport stdio

# Index a pack without starting the server
ep-mcp index --pack /path/to/pack --config ep-mcp-config.yaml

# Validate a pack can be loaded
ep-mcp validate --pack /path/to/pack

# Show server info
ep-mcp info --config ep-mcp-config.yaml
```

### 10.3 Environment Variables

```bash
# Embedding provider API keys (one required, depends on config)
GEMINI_API_KEY=...
AZURE_OPENAI_API_KEY=...
OPENAI_API_KEY=...

# Azure-specific (when using azure_openai provider)
AZURE_OPENAI_ENDPOINT=https://myresource.openai.azure.com

# Pack API keys (alternative to config file)
EP_MCP_KEY_EZT_DESIGNER=key_abc123

# Server
EP_MCP_HOST=0.0.0.0
EP_MCP_PORT=8000
EP_MCP_LOG_LEVEL=info
```

---

## 11. Testing Strategy

### 11.1 Unit Tests

Each module tested independently:

| Module | Test Focus |
|--------|------------|
| `pack/loader.py` | Load valid/invalid packs, frontmatter parsing, manifest validation |
| `pack/manifest.py` | Manifest field extraction, defaults, required field enforcement |
| `index/chunker.py` | Single-file chunks, oversized splitting, token counting |
| `index/sqlite_store.py` | FTS5 + vec table creation, insert, query, incremental update |
| `retrieval/engine.py` | Hybrid search, score fusion, MMR, metadata boosting |
| `retrieval/scorer.py` | Score normalization, fusion weights, boost calculations |
| `embeddings/*.py` | Mock embedding providers, batch handling, error paths |
| `auth.py` | Key validation, missing header, invalid key, multi-key rotation |

### 11.2 Integration Tests

End-to-end tests using the `ezt-designer` pack:

1. **Load + index:** Load pack from disk, build full index, verify chunk/FTS/vec counts
2. **Search quality:** Run known queries, verify expected files appear in top results
3. **Provenance:** Verify `id`, `content_hash`, `verified_at` present in all results
4. **Incremental index:** Modify a file, re-index, verify only changed file was re-embedded
5. **MCP protocol:** Full JSON-RPC roundtrip — initialize → list tools → call ep_search → verify response schema
6. **Auth:** Authenticated request succeeds, unauthenticated request rejected

### 11.3 Eval Alignment

Search quality tests should align with the help bot eval baseline:
- Same 20-question benchmark set (tagged `benchmark: true` in `questions.yaml`)
- Same scoring methodology (Sonnet judge)
- Target: ≥84.8% correctness (Run 12 baseline)

This ensures the MCP retrieval pipeline matches or exceeds the existing help bot.

### 11.4 Test Execution

```bash
# Unit tests
pytest tests/unit/ -v

# Integration tests (requires ezt-designer pack + embedding API key)
pytest tests/integration/ -v --pack /path/to/ezt-designer

# MCP protocol compliance
pytest tests/mcp/ -v
```

---

## 12. Dependencies

### 12.1 Python Packages

```
# Core
mcp[cli]>=1.0               # MCP Python SDK (FastMCP, transports)
uvicorn>=0.30                # ASGI server
starlette>=0.40              # ASGI framework (routing, middleware)
pyyaml>=6.0                  # manifest.yaml parsing
pydantic>=2.0                # Data models, validation

# Retrieval
sqlite-vec>=0.1              # Vector similarity search extension
                             # (FTS5 is built into Python's sqlite3)

# Embeddings
google-genai>=0.8            # Gemini embedding API
openai>=1.50                 # OpenAI + Azure OpenAI embedding API
                             # (azure_openai uses the same package with azure config)

# Utilities
httpx>=0.27                  # Async HTTP client (for embedding API calls)
python-dotenv>=1.0           # Environment variable loading
click>=8.0                   # CLI framework

# Testing
pytest>=8.0
pytest-asyncio>=0.23
```

### 12.2 System Dependencies

- **Python 3.11+** (for modern async, typing, sqlite3 improvements)
- **SQLite 3.41+** (for FTS5 and sqlite-vec compatibility — Python 3.11+ bundles this)
- **sqlite-vec extension** (loaded at runtime, installed via pip)

### 12.3 No Heavy Dependencies

Intentionally excluded:
- **No LangChain / LlamaIndex** — direct SQLite queries are simpler, faster, and fully controllable
- **No Chroma / Qdrant / Pinecone** — SQLite + sqlite-vec handles our scale (hundreds to low thousands of files per pack)
- **No FAISS** — sqlite-vec provides sufficient ANN performance for EP-scale indexes
- **No sentence-transformers** — we use API-based embeddings, not local models (for now)

The dependency footprint is deliberately small. EP MCP should install and run with `pip install ep-mcp` — no binary dependencies beyond sqlite-vec.

---

## 13. Resolved Questions

| # | Question | Decision | Notes |
|---|----------|----------|-------|
| 1 | Pack hot-reload | **Restart** | Simplest for MVP. No file watcher, no reload API. Restart the server to pick up pack changes. |
| 2 | Concurrent pack queries | **Per-pack connection pool** | Must support concurrent queries. SQLite WAL mode + read-only connection pool per pack. |
| 3 | Large result truncation | **Return full, add param later** | Full content for MVP. Add optional `max_chunk_tokens` param in a future release if needed. |
| 4 | Embedding batch size | **Provider-specific** | Handled internally by each provider implementation (Gemini: 100/batch, OpenAI: 2048/batch). |
| 5 | Multi-pack composite queries | **Not implemented** | Out of scope. Keep it simple — each pack has its own endpoint. No cross-pack search. |
| 6 | Index storage location | **Same directory as pack** | Index lives at `{pack_path}/.ep-mcp/index.db`. Add `.ep-mcp/` to `.gitignore`. |
| 7 | Startup indexing timeout | **Block startup, background later** | Block for MVP (EP-scale indexing takes seconds). Add background indexing in a future release for large packs. |

---

## 14. Relationship to Domain MCP Servers

ExpertPack MCP is **Layer 1** — the generic knowledge serving layer. Domain MCP servers (Layer 2) extend it:

```
┌─────────────────────────────────────────────┐
│         EasyTerritory MCP (Layer 2)         │
│                                             │
│  ┌───────────────────┐  ┌────────────────┐  │
│  │  EP MCP (Layer 1) │  │  EZT Tools     │  │
│  │  ezt-designer     │  │                │  │
│  │  pack loaded      │  │  auto_build    │  │
│  │                   │  │  create_terr   │  │
│  │  ep_search        │  │  geocode       │  │
│  │  ep_list_topics   │  │  cluster       │  │
│  └───────────────────┘  └────────────────┘  │
└─────────────────────────────────────────────┘
```

**Integration pattern:** The EZT MCP server imports `ep_mcp` as a library, loads the ezt-designer pack, and registers additional domain-specific tools alongside the EP tools. Same FastMCP instance, same transport, unified tool surface.

```python
from ep_mcp import create_pack_server
from ezt_mcp.tools import register_ezt_tools

# Create EP MCP server with ezt-designer pack
server = create_pack_server(pack_path="/data/packs/ezt-designer")

# Register domain-specific tools on the same server
register_ezt_tools(server, api_config=ezt_cloud_config)

# Serve — agents see both EP tools and EZT tools
server.run(transport="streamable-http")
```

This is the architectural boundary: EP MCP provides knowledge, domain MCP provides actions. The agent uses knowledge to reason, then tools to execute.

---

## What's Next

1. ✅ Vision (VISION.md v0.3)
2. ✅ Architecture (this document)
3. ⬜ Implementation — start with pack loader + indexer + ep_search, then transport + auth
4. ⬜ Validation — integration tests against ezt-designer, eval alignment
5. ⬜ Deploy — ExpertPack droplet, nginx reverse proxy, systemd
6. ⬜ EasyTerritory MCP — domain tools layer

