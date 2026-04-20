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

### Retrieval tuning

All retrieval options live under the `retrieval:` block in `config.yaml`. Key options:

```yaml
retrieval:
  # Hybrid search weights (defaults; intent routing overrides per-query)
  vector_weight: 0.7
  text_weight: 0.3
  candidate_multiplier: 8

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

  # Graph expansion (shallow, 1-hop)
  graph_expansion_enabled: false
  graph_expansion_confidence_threshold: 0.38
  graph_expansion_min_score: 0.20
  graph_expansion_structural_bonus: 1.0

  # Deep graph traversal (multi-hop BFS, opt-in)
  graph_expansion_deep: false
  graph_expansion_depth: 2       # max hops (meaningful when deep=true)
  graph_expansion_discount: 0.85 # per-hop score decay
  graph_expansion_deep_max_bonus: 5
```

See [ARCHITECTURE.md §5](ARCHITECTURE.md#5-retrieval-engine) for full pipeline documentation.

## 4. Running (dev)

```bash
ep-mcp serve --config config.yaml
```

The server builds the SQLite index (`.ep-mcp/index.db` inside the pack directory) on first run. Subsequent starts use incremental indexing based on content hashes.

## 5. Running (production — systemd)

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

## 6. Reverse proxy (nginx example)

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

## 7. Re-indexing

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

## 8. Updating the server

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

## 9. Verifying the deployment

```bash
# Health check (no auth required)
curl https://your-domain/mcp/health

# Search endpoint
curl -H "Authorization: Bearer your-key" \
  "https://your-domain/mcp/search?q=your+query&pack=my-pack&n=5"
```

Expected response: JSON with a `results` array. Each result contains `text`, `source_file`, `score`, `id`, `content_hash`, `verified_at`.

## 10. Firewall / security notes

- Prefer binding to `127.0.0.1` (not `0.0.0.0`) when behind a reverse proxy.
- If you must bind to `0.0.0.0`, restrict the port with `iptables`/`ufw` to trusted IPs only.
- API keys are **per-pack**. Use different keys for different clients.
- Never commit `.env` or `config.yaml` containing real keys.

---
