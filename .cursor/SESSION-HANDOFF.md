# Session Handoff — 2026-08-17

*ep-mcp **0.5.0** is implemented locally: MCP Python SDK v2 + ExpertPack consumer contract. **Do not deploy or reindex production** unless the user explicitly asks. Retrieval quality on real packs has **not** been measured yet.*

Prior chat: [EP MCP dual upgrade](51392480-b715-4672-a415-3710101b66aa)

---

## First thing next session

The user asked whether sample-question retrieval was run. **It was not.** Only `tests/unit` (166 passed).

To judge 0.5.0 retrieval you must:

1. Point at a real pack (canonical: **ezt-designer**, 22-question eval in the **pack** repo under `eval/` / `questions.yaml` — not in ep-mcp).
2. Rebuild the SQLite index (`index_features=context_prefix_v1+nav_filter_v1`). Startup rebuilds older indexes automatically, or `rm -rf <pack>/.ep-mcp/`.
3. Run the pack’s sample questions against local `/search` (HTTP payload shape is unchanged). Sidecar-heavy packs (blender-3d, home-assistant) need the rebuild before `context_prefix` / nav filter can be judged.

This repo’s only live-pack test is `tests/integration/test_index_real_pack.py` (`--pack` + embedding key). It is **not** the 22-question benchmark.

---

## Git / package

| Item | Value |
|------|--------|
| **Branch** | `main` (commit + push requested this session) |
| **Package** | `0.5.0` (`pyproject.toml`, `ep_mcp.__version__`) |
| **MCP SDK** | `mcp` **2.0.0** (`mcp[cli]>=2,<3`); keep first-party `httpx>=0.27` |
| **pydantic** | `>=2.12` (v2 transitively needs it) |
| **Do not pin** | `mcp-types` (SDK exact-pins it) |
| **Deploy** | **Not done** |
| **Plan** | `c:\Users\BrianHearn\.cursor\plans\ep_mcp_dual_upgrade_1a8df9f9.plan.md` (do not edit) |
| **ExpertPack sibling** | `C:\Users\BrianHearn\source\repos\expert-pack` `main` @ `a590366` (2026-08-17) |

Do **not** commit `.cursor/mcp.json` (local Cursor MCP; uses `EZT_MCP_API_KEY`) or `ep-mcp.code-workspace`.

---

## What landed

### A — MCP Python SDK v2 (2026-07-28, dual-era)

- `from mcp.server.mcpserver import MCPServer` — `mcp.server.fastmcp` is gone.
- Constructor: `MCPServer(f"ep-mcp-{slug}", instructions=..., version=pack.version)`. **No** `stateless_http` on the constructor (`TypeError`).
- `stateless_http=True` + `transport_security=TransportSecuritySettings(...)` go on `streamable_http_app()`.
- v2 defaults `host="127.0.0.1"` and **auto-enables DNS-rebinding protection**. Public `Host: expertpack.ai` → **421** unless allowlisted.
- Config: `server.mcp_allowed_hosts` / `server.mcp_allowed_origins`. Include `host:*` forms — Host headers often have a port.
- Tools return **lists/dicts**, not `json.dumps(...)`. v2 often puts JSON in `result.content[0].text`; `structured_content` may be `None`. Tests parse via `_payload()` in `tests/unit/test_mcp_server.py`.
- Registered tool **function** names: `ep_search_tool`, `ep_list_topics_tool`, `ep_graph_traverse_tool`, `ep_read_tool`.
- `ep://{slug}/file/{+path}` (RFC 6570 reserved expansion) and `ep://{slug}/authority`.
- `ep-mcp serve --transport stdio` → `mcp.run(transport="stdio")` for the **first** configured pack only. HTTP still uvicorn.
- Dual-era: same process serves 2026-07-28 and 2025-era (`Client(mcp, mode="legacy")`).
- Mount pattern unchanged: parent Starlette lifespan still enters `mcp.session_manager.run()`.
- **Do not** wrap Streamable HTTP mounts with `APIKeyAuth`. Only `/search` is Bearer-gated; MCP auth stays on the proxy.

### B — ExpertPack consumer contract (schema family **4.1**, additive only)

- **`context_prefix`**: parsed on `SidecarChunk`. Embed/FTS text = `f"{prefix}\n\n{content}"` via `Chunk.indexed_content` / `text_for_index()`. `chunks.content` and reconstruct spans stay the original body. `span_hash` is hash-of-body only (RFC-003 stale detection must not include the prefix). Heuristic `# {title}` heading prefix is unchanged.
- **`INDEX_FEATURES = "context_prefix_v1+nav_filter_v1"`** in `IndexManager`. Rebuild if flag differs **or** old index has chunks but no flag. First-run empty DB (`index_features is None` and `chunk_count==0`) must **not** spuriously full-rebuild.
- **SQLite**: `indexed_content` column; FTS triggers use `COALESCE(indexed_content, content)`. Install triggers **after** schema migrate (`_install_fts_triggers`).
- **Not indexed** (`PackFile.is_indexable`): `retrieval_strategy`/`concept_scope` navigation; `_index.md`; `meta/source-coverage.md`; `context.on_demand`. Incremental update **deletes** previously indexed rows for those files (they stay in `pack.files`, so they would not look “deleted” otherwise). Always-tier + navigation: still MCP Resources, not in the vector/FTS pool.
- **`ep_read`**: path or provenance `id` → whole atom (`content` frontmatter-stripped, `requires`, `activation`, `retrieval_strategy`, optional `reconstruct`). Search hits are locators. Navigation/on_demand are readable here.
- **Do not change** `ep_search` default text dump or HTTP `GET`/`POST /search` payload (OpenClaw + eval harnesses).
- Consume loop in `instructions=`: search → `ep_read` → `requires:` expand → stop (step budget **3** / hard cap **7**). “Chunk contradiction” is adaptive second-pass via `ep_read` + `requires:`, not a new tool.
- **`authority_boundary`**: `in_scope`, `out_of_scope`, `refuse_when`, `no_source_no_claim` on Manifest; resource + `ep_list_topics` + instructions.
- **`activation:`** (`tools`, `constraints`, `next`) on workflow / decision / gotcha / phase only — ignore on concepts. Surfaced on `ep_read`, `ep_list_topics`, prompt descriptions.
- Loader inventory keys: `rel_path.as_posix()` (fixes Windows `concepts\topic-a.md`). Navigation wins over always-tier override. `index_path` remains a real OS path (tests must accept `\`).

---

## Tests

```powershell
cd C:\Users\BrianHearn\source\repos\ep-mcp
py -3.12 -c "from importlib.metadata import version; print('mcp', version('mcp'))"
py -3.12 -m pytest tests/unit -q --tb=short
```

Expected: `mcp 2.0.0`, **166 passed**.

Covered: sidecar prefix (embed has prefix; body/`span_hash` do not); nav/on_demand skipped by `IndexManager` but `ep_read` serves them; `ep_read` by path/id + unknown-path error; authority + activation parse; posix loader keys; in-memory `Client(mcp)` + `mode="legacy"`.

**Not run:** pack `eval/` sample questions, production reindex, `tests/integration/test_index_real_pack.py`.

---

## Constraints / out of scope (still)

- Stay on schema family **4.1**. No core 4.2.
- No TAC, no composite `fail_closed` resolver (pack tooling), no OAuth 2.1.
- No production droplet deploy unless asked.
- Do not add stub `router.py`, `embeddings/openai.py`, or `tests/mcp/`.
- `create_pack_server` does **not** exist. Use `create_pack_mcp` / `init_pack` in `ep_mcp.server`.

---

## Key files

```
ep_mcp/server.py                 # MCPServer + ep_read + allowlists + instructions
ep_mcp/cli.py                    # stdio transport
ep_mcp/config.py                 # mcp_allowed_hosts / origins
ep_mcp/tools/ep_read.py          # whole-atom read
ep_mcp/tools/ep_list_topics.py   # authority + activation summary
ep_mcp/pack/sidecar.py           # context_prefix
ep_mcp/pack/models.py            # AuthorityBoundary, Activation, is_indexable
ep_mcp/pack/manifest.py          # authority_boundary parse
ep_mcp/pack/loader.py            # posix keys, activation, navigation
ep_mcp/index/chunker.py          # indexed_content
ep_mcp/index/manager.py          # INDEX_FEATURES + nav filter + purge
ep_mcp/index/sqlite_store.py     # indexed_content + FTS triggers
ep_mcp/resources/pack_resources.py
ep_mcp/prompts/pack_prompts.py
tests/unit/test_mcp_server.py
tests/unit/test_ep_read.py
```

---

*End handoff.*
