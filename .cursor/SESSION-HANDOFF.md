# Session Handoff — 2026-07-07

*Context for the next agent session. Top-5 (RFC-003/004) runtime shipped 2026-07-06; this session verified end-to-end and synced framework docs.*

---

## Git state (after push)

| Item | Value |
|------|--------|
| **Branch** | `main` — pushed to `origin/main` |
| **Runtime commit** | `a9d1639` — RFC-003/004 sidecars + reconstruct envelope |
| **Doc sync commit** | (this session) — `ARCHITECTURE.md` §5.9, `README.md`, handoff |
| **Untracked (local only)** | `ep-mcp.code-workspace` — IDE workspace; exclude from commits |
| **Deploy** | **Not done** — user explicitly deferred deploy this session |

---

## Verification summary (2026-07-07)

Cross-checked **expert-pack** (`355257b`, `ede5d2c`) against **ep-mcp** (`a9d1639`). All RFC-003/004 runtime checklist items are implemented.

| Area | Status |
|------|--------|
| RFC-004 sidecar load + authoritative boundaries | ✅ `ep_mcp/pack/sidecar.py`, `chunker.py`, `manager.py` |
| RFC-003 reconstruct envelope (`fragment_id`, `line_range`, `stale`, etc.) | ✅ `retrieval/engine.py` `_enrich_with_reconstruct` |
| Provenance `confidence` grade surfacing | ✅ loader → SQLite → results |
| MCP + HTTP (`GET`/`POST /search`) `reconstruct` flag | ✅ `server.py`, `ep_search.py` |
| Back-compat (`reconstruct=false` unchanged) | ✅ |
| TAC v1 (agent response layer) | expert-pack only — consumes `fragment_id` from reconstruct |
| Strict gate / ingest-gate / pre-commit | expert-pack only |

**Tests (Windows, py 3.12):**
- RFC tests: `test_sidecar.py` + `test_reconstruct.py` — **4/4 pass**
- Full unit suite: **140/148 pass** — 8 failures are pre-existing Windows path-key flakes (`concepts/topic-a.md` vs `concepts\topic-a.md` in `test_loader.py`, `test_requires_expansion.py`)

**Sidecar smoke (local expert-pack `blender-3d`):**
```
character-animation → 26 chunks, first line_range (15, 23)
```

**FastMCP / `mcp` SDK:**
- Latest PyPI: `mcp` **1.28.1**
- Installed: **1.28.1** — no upgrade needed
- FastMCP is `from mcp.server.fastmcp import FastMCP` (no separate package)
- `pyproject.toml` pins `mcp[cli]>=1.0` (loose; resolves to latest on install)

---

## What was accomplished (prior session — `a9d1639`)

### RFC-004 — chunk sidecar consumption

- `ep_mcp/pack/sidecar.py` — loads `<name>.chunks.yaml` (or `chunks_sidecar:` frontmatter pointer)
- `chunk_file()` uses sidecar `line_range` / `chunk_order` when present (authoritative over header split)
- Index stores `line_start`, `line_end`, `section_slug`, `sidecar_chunk_id`, `span_hash` in SQLite (auto-migrated columns)

### RFC-003 — Reconstruct Mode envelope

- `reconstruct=true` returns: `fragment_id`, `line_range`, `original_markdown`, span `content_hash`, `excerpt`, `stale`, `byte_offset` `[start, end]`
- Legacy `original_span` / `provenance_block` retained
- Staleness: indexed `span_hash` vs live span hash at query time
- Highlight rendering: **not implemented** (RFC-003 host-dependent; ep-mcp returns raw span)

### Provenance confidence grade

- Optional frontmatter `confidence` (`expert-verified` | `crawled` | `inferred`) parsed, stored, surfaced

---

## What was accomplished (this session — doc sync only)

| Doc | Change |
|-----|--------|
| `ARCHITECTURE.md` §5.9 | Full RFC-003 field list (was legacy-only) |
| `README.md` | Reconstruct bullet lists `fragment_id` / `stale` |
| expert-pack `README.md` | Implementation status: RFC-003/004 → ✅ Full (was "EP MCP external") |
| expert-pack `ROADMAP.md` | Top-5 bullets note ep-mcp shipped |
| expert-pack `CHANGELOG.md` | EP MCP runtime alignment entry |
| expert-pack RFC-003/004 | Status headers + checklist closure note |
| expert-pack `.cursor/SESSION-HANDOFF.md` | External deps marked shipped |

---

## Key design decisions (carry forward)

1. **Sidecar wins over heuristic split** — boundaries match `ep-chunk-annotate.py` exactly
2. **Reconstruct is post-retrieval** — enrichment only; no scoring/MMR/expansion changes
3. **Reindex required before production** — existing indexes lack `line_start/end` / `span_hash`; rebuild after deploy for accurate `fragment_id` / `stale`
4. **Schema family 4.1** — additive only; no 4.2 bump unless user requests
5. **Do not deploy without explicit user request** — code is on main; production droplet not updated this session

---

## Verification commands (Windows)

```powershell
cd C:\Users\BrianHearn\source\repos\ep-mcp
py -3.12 -m pip install -e ".[dev]" numpy -q
py -3.12 -m pytest tests/unit/test_sidecar.py tests/unit/test_reconstruct.py -v
py -3.12 -m pytest tests/unit -v --tb=line
py -3.12 -c "from importlib.metadata import version; print('mcp', version('mcp'))"
```

Sidecar smoke:

```powershell
py -3.12 -c "
from pathlib import Path
from ep_mcp.pack.loader import load_pack
from ep_mcp.pack.sidecar import load_sidecar, sidecar_path_for_md
from ep_mcp.index.chunker import chunk_file
pack = load_pack(r'C:\Users\BrianHearn\source\repos\expert-pack\packs\blender-3d')
rel = [k for k in pack.files if 'character-animation' in k][0]
pf = pack.files[rel]
norm = rel.replace('\\\\', '/')
sidecar = load_sidecar(sidecar_path_for_md(Path(pack.pack_dir), norm))
chunks = chunk_file(norm, pf.content, pf.title, raw_content=pf.raw_content, sidecar=sidecar)
print(len(chunks), chunks[0].line_range)
"
```

Expected: `26 (15, 23)`.

---

## Deploy checklist (production — NOT DONE)

1. Deploy updated `ep_mcp/` to droplet (`scripts/deploy.sh` or rsync in `DEPLOYMENT.md`)
2. **Reindex packs** with `.chunks.yaml` sidecars (blender-3d, home-assistant, solar-diy workflows/phases)
3. Smoke POST `/search` with `"reconstruct": true` — confirm `fragment_id` + `line_range`
4. OpenClaw memory plugin `reconstruct: true` → full RFC-003 envelope end-to-end
5. Bump version / tag release after live smoke passes

---

## Known backlog (non-blocking)

- **Windows path keys in loader** — `pack.files` keys use OS separators on Windows; 8 unit tests fail; normalize to forward slashes or fix test assertions
- **Highlight rendering** — ep-mcp returns raw span; host renders highlights
- **W-V41-01** (expert-pack) — oversized standard concepts in demo packs; CI uses `--ignore`
- **Plan file** (expert-pack) — todos still `pending` in YAML; work complete; plan intentionally not edited

---

## Likely next-session work

1. **Deploy + reindex + live smoke** on expertpack.ai/mcp (user deferred this session)
2. Normalize loader file paths to forward slashes (fix Windows test flake)
3. Version tag + move `[Unreleased]` → versioned CHANGELOG sections (both repos)
4. Eval: sidecar-boundary indexing vs heuristic split on demo packs

---

## Key file index

```
ep_mcp/pack/sidecar.py              # RFC-004 sidecar loader
ep_mcp/index/chunker.py             # sidecar-driven chunking + span metadata
ep_mcp/index/manager.py             # load sidecar at index time
ep_mcp/index/sqlite_store.py        # migrated columns + upsert fields
ep_mcp/retrieval/engine.py          # _enrich_with_reconstruct (RFC-003)
ep_mcp/retrieval/models.py          # SearchResult RFC-003 fields
ep_mcp/server.py                    # HTTP reconstruct + fragment output
ep_mcp/tools/ep_search.py           # MCP reconstruct + fragment output
tests/unit/test_sidecar.py
tests/unit/test_reconstruct.py
ARCHITECTURE.md                     # §5.9 Reconstruct Mode (updated)
```

---

## Related repos

| Repo | Path | Key commits |
|------|------|-------------|
| expert-pack | `C:\Users\BrianHearn\source\repos\expert-pack` | `355257b`, `ede5d2c` — Top 5 specs/tooling; doc sync marks ep-mcp shipped |
| ep-mcp | `C:\Users\BrianHearn\source\repos\ep-mcp` | `a9d1639` — runtime; this session — doc sync |

---

*End handoff.*
