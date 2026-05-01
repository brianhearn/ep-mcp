# DEPLOYMENT.md

Practical guide for deploying and operating **ep-mcp** in production.

## 1. Prerequisites

- Python 3.12+
- A valid ExpertPack directory (with `_graph.yaml` optional)
- Gemini API key (or Azure OpenAI credentials)
- A server with systemd (for production)

## 2. Installation

```bash
git clone https://github.com/brianhearn/ep-mcp.git
cd ep-mcp
pip install -e .
```

## 3. Configuration

- Create `config.yaml` (see [README.md](README.md) for the full schema).
- For external access set `host: "0.0.0.0"` (or keep the default `127.0.0.1` if running behind a reverse proxy — **recommended**).
- Set `api_keys` per pack. Treat these as passwords. Use the environment variable `EP_MCP_KEY_{SLUG}` to avoid committing real keys.
- Copy `.env.example` to `.env` and populate the API keys.

### Embedding provider

The default is Gemini. To use Azure OpenAI instead:

```yaml
embedding:
  provider: "azure-openai"
  model: "text-embedding-3-large"   # or text-embedding-3-small, text-embedding-ada-002
  azure_endpoint: "https://your-resource.openai.azure.com"  # or set AZURE_OPENAI_ENDPOINT
  azure_api_key: "your-key"          # or set AZURE_OPENAI_API_KEY
  azure_api_version: "2024-10-21"    # default
  azure_deployment: "my-deployment"  # defaults to model name if omitted
  output_dimensionality: null        # MRL shortening: null=full dim, e.g. 768=4× smaller
```

⚠️ **Embedding dimensions must match at index time.** Switching providers on an existing index requires a full reindex — `rm -rf <pack>/.ep-mcp/` then restart.

### Retrieval tuning

All retrieval options live under the `retrieval:` block in `config.yaml`. Key options:

```yaml
retrieval:
  # Results returned per query — keep <= reranker.candidate_pool_size for best reranker coverage
  default_max_results: 20

  # Hybrid search weights (defaults; intent routing overrides per-query)
  vector_weight: 0.7
  text_weight: 0.3
  candidate_multiplier: 8

  # MMR (Maximal Marginal Relevance) — diversity re-ranking after fusion
  # lambda: 1.0 = pure relevance, 0.0 = pure diversity. 0.95 recommended for domain packs.
  mmr_enabled: true
  mmr_lambda: 0.95

  # Adaptive threshold filtering (replaces flat min_score)
  adaptive_threshold: true
  activation_floor: 0.15
  score_ratio: 0.55
  absolute_floor: 0.10

  # Length penalty — discount very short chunks
  length_penalty_threshold: 80   # chars
  length_penalty_factor: 0.15

  # Intent-aware routing — adjust vector/BM25 weights per query intent
  intent_routing_enabled: true

  # BM25 saturation cap — prevent keyword-dense files from dominating fusion
  # 1.0 = disabled (default), 0.7 = recommended to prevent saturation
  bm25_cap: 1.0

  # BM25 K-of-N token matching — require only a fraction of query tokens (not strict AND)
  # 1.0 = strict AND (default), 0.67 = 2-of-3 tokens required
  bm25_min_token_match_ratio: 1.0

  # File-level deduplication — max chunks returned from the same source file
  max_chunks_per_file: 2

  # Graph expansion (shallow, 1-hop)
  graph_expansion_enabled: false
  graph_expansion_confidence_threshold: 0.38
  graph_expansion_min_score: 0.20
  graph_expansion_structural_bonus: 1.0

  # Deep graph traversal (multi-hop BFS, opt-in)
  graph_expansion_deep: false
  graph_expansion_depth: 2           # max hops (meaningful when deep=true)
  graph_expansion_discount: 0.85     # per-hop score decay
  graph_expansion_deep_max_bonus: 5

  # Reserved slots — guarantee graph-expanded results appear in final top-K
  # 0 = legacy merge-sort behavior
  graph_expansion_reserved_slots: 2

  # Pre-MMR graph widening — pull neighbors into candidate pool before MMR
  graph_widen_enabled: false
  graph_widen_max_seeds: 5
  graph_widen_min_seed_score: 0.38
  graph_widen_min_neighbor_score: 0.25

  # `requires:` frontmatter expansion (schema v4.1+)
  # Appends atoms declared as dependencies of top-K results (after final slice)
  requires_expansion_enabled: true
  requires_expansion_max_depth: 2       # transitive hop cap
  requires_expansion_max_atoms: 3       # max extra atoms appended per query
  requires_expansion_token_budget: 3500 # cumulative token budget for appended atoms
  requires_expansion_score: 0.30        # displayed score for appended atoms
```

See [ARCHITECTURE.md §5](ARCHITECTURE.md#5-retrieval-engine) for full pipeline documentation.

### Reranker (optional second-pass precision layer)

Adds a cross-encoder re-ranking pass after hybrid fusion. Requires `pip install sentence-transformers`. **Recommended: enable in production** — adds ~70–300ms warm latency but measurably improves precision.

```yaml
reranker:
  enabled: true
  model: "cross-encoder/ms-marco-MiniLM-L-6-v2"
  candidate_pool_size: 20  # candidates to rerank before slicing to max_results
  max_chars: 512           # truncate document text before cross-encoder scoring
  batch_size: 32
```

> **Note:** `candidate_pool_size` sets how many candidates the cross-encoder sees. For best results, ensure the retrieval `n` requested by the client (or `default_max_results`) is ≤ `candidate_pool_size`. If `n > candidate_pool_size`, the reranker only sees the first `n` results and the pool size has no effect.

### Pack-level index directory (`index_dir`)

By default the SQLite index and query-embedding cache are written to `<pack_path>/.ep-mcp/` (co-located with pack files). Set `index_dir` when pack files live on a non-persistent path (e.g. Azure Container Apps with `/tmp` pack staging) and you need the index to survive container restarts on a separately-mounted volume:

```yaml
packs:
  - slug: ezt-designer
    path: /tmp/packs/ezt-designer       # pack files (may be ephemeral)
    index_dir: /mnt/cache/ezt-designer  # persistent volume for index + cache
    api_keys:
      - your-key
```

## 4. Timing logs

Retrievals automatically emit `[TIMING]` lines at INFO level — no config required. Each line covers one pipeline stage (embed, search+fusion, threshold+boosts, MMR, dedup, graph_expansion, requires_expansion, total). Use these to identify latency bottlenecks without any instrumentation overhead.

```
INFO  [TIMING] embed=4123ms dual_search+fusion=28ms threshold+boosts+penalty=2ms mmr=5ms dedup+build=1ms graph_expansion=0ms requires_expansion=12ms total=4171ms
```

`embed_cached: true` queries skip the Gemini API call and show ~1ms embed latency.

## 5. Running (dev)

```bash
ep-mcp serve --config config.yaml
```

The server builds the SQLite index (`.ep-mcp/index.db` inside the pack directory) on first run. Subsequent starts use incremental indexing based on content hashes.

## 6. Running (production — systemd)

Create `/etc/systemd/system/ep-mcp.service`:

```ini
[Unit]
Description=ExpertPack MCP Server
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/ep-mcp
EnvironmentFile=/opt/ep-mcp/.env
ExecStart=/opt/ep-mcp/.venv/bin/ep-mcp serve --config /opt/ep-mcp/config.yaml
Restart=on-failure
RestartSec=5
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

Commands:

```bash
systemctl enable ep-mcp
systemctl start ep-mcp
journalctl -u ep-mcp -f
```

## 7. Reverse proxy (nginx example)

```nginx
location /mcp {
    proxy_pass http://127.0.0.1:8100;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_http_version 1.1;
    proxy_set_header Connection "";
}
```

**Note:** Streamable HTTP requires HTTP/1.1 keep-alive. Terminate SSL at nginx.

## 8. Re-indexing

For **content-only changes** (editing existing atom files), the incremental indexer detects changed files by content hash — just restart the service without wiping the index:

```bash
systemctl restart ep-mcp
```

Only wipe the index when the **schema or chunking logic changes** (i.e. code updates that affect how chunks are generated):

```bash
rm -rf /path/to/pack/.ep-mcp/
systemctl restart ep-mcp
```

The server rebuilds the full index on next startup. For large packs (~650 chunks) expect 30–60 seconds (Gemini embedding API). **Avoid multiple full reindexes in quick succession** — the Gemini free tier has hourly embedding quotas; repeated cold starts will exhaust them and cause startup failures (RESOURCE_EXHAUSTED 429).

## 9. Updating the server

Use the rsync-based `scripts/deploy.sh` (avoids pip shebang issues):

```bash
EP_MCP_REMOTE_HOST=root@your-server \
EP_MCP_REMOTE_SRC=/opt/ep-mcp/ep_mcp \
EP_MCP_REMOTE_SITE_PKG=/opt/ep-mcp/.venv/lib/python3.12/site-packages/ep_mcp \
./scripts/deploy.sh
```

For service restart only:

```bash
./scripts/deploy.sh --restart-only
```

## 10. Verifying the deployment

```bash
# Health check (no auth required)
curl https://your-domain/mcp/health

# Search endpoint
curl -H "Authorization: Bearer your-key" \
  "https://your-domain/mcp/search?q=your+query&pack=my-pack&n=5"
```

Expected response: JSON with a `results` array. Each result contains `text`, `source_file`, `score`, `id`, `content_hash`, `verified_at`.

## 11. Firewall / security notes

- Prefer binding to `127.0.0.1` (not `0.0.0.0`) when behind a reverse proxy.
- If you must bind to `0.0.0.0`, restrict the port with `iptables`/`ufw` to trusted IPs only.
- API keys are **per-pack**. Use different keys for different clients.
- Never commit `.env` or `config.yaml` containing real keys.

## 12. Query logging

EP MCP can write a structured JSONL log of every search call — useful for monitoring production queries, measuring retrieval quality, and building crowdsourced FAQs from real user questions.

### Enable

Add one line to the `server:` block in `config.yaml`:

```yaml
server:
  host: "127.0.0.1"
  port: 8100
  log_level: "info"
  query_log_path: "/var/log/ep-mcp-queries.jsonl"
```

The directory is created automatically if it doesn't exist. Restart the service after changing config.

### Log record format

Each line is a JSON object:

```json
{
  "ts": "2026-04-23T13:54:45Z",
  "pack": "ezt-designer",
  "query": "how do I create a territory",
  "type": null,
  "tags": null,
  "result_count": 3,
  "chunks": [
    "workflows/wf-create-territory-basic-boundary.md",
    "workflows/wf-create-territory-drawing-custom.md",
    "faq/faq-general.md"
  ],
  "scores": [0.6625, 0.3377, 0.3246],
  "top_score": 0.6625,
  "elapsed_ms": 5157.9,
  "embed_cached": false
}
```

| Field | Description |
|---|---|
| `ts` | UTC timestamp (ISO-8601) |
| `pack` | Pack slug |
| `query` | Raw query string |
| `type` / `tags` | Filter params passed by the client (null if not used) |
| `result_count` | Number of chunks returned |
| `chunks` | Source file paths for returned results (pack-relative) |
| `scores` | Relevance scores parallel to `chunks` |
| `top_score` | Score of the first result (convenience field) |
| `elapsed_ms` | Total search latency in milliseconds |
| `embed_cached` | `true` = embedding served from cache (warm); `false` = live Gemini API call (cold); `null` = no cache wrapper |

### Notes

- All three search paths write to the log: `GET /search`, `POST /search`, and the MCP `ep_search` tool.
- The log is append-only. Rotate with `logrotate` or a cron job if needed.
- `embed_cached: false` queries show true end-to-end latency; `embed_cached: true` queries reflect the warm path (embedding served from SQLite cache, ~1ms vs ~4s).
- When `query_log_path` is not set (default), no log file is created and no performance overhead is added.

---
