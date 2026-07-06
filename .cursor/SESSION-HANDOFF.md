# Session Handoff — 2026-07-06

*Context for the next agent session. Aligns ep-mcp with expert-pack Top 5 (RFC-003/004) landed today in `../expert-pack`.*

---

## Git state (after push)

| Item | Value |
|------|--------|
| **Branch** | `main` — pushed to `origin/main` |
| **Commit** | `feat: align ep-mcp with expert-pack RFC-003/004 sidecars and reconstruct` |
| **Untracked (local only)** | `ep-mcp.code-workspace` — IDE workspace; exclude from commits |
| **Working tree** | Clean (except workspace file) |

---

## What was accomplished

Compared **expert-pack** commits `355257b` + `ede5d2c` (2026-07-06) against ep-mcp and closed the runtime gaps called out in RFC-003/004 handoff checklists.

### RFC-004 — chunk sidecar consumption

- New `ep_mcp/pack/sidecar.py` — loads `<name>.chunks.yaml` (or `chunks_sidecar:` frontmatter pointer)
- `chunk_file()` uses sidecar `line_range` / `chunk_order` when present (authoritative over header split)
- Index stores `line_start`, `line_end`, `section_slug`, `sidecar_chunk_id`, `span_hash` in SQLite (auto-migrated columns)
- Verified on real pack: `blender-3d/workflows/character-animation.chunks.yaml` → **26 chunks**, opening `(15, 23)`

### RFC-003 — Reconstruct Mode envelope

- `reconstruct=true` (MCP + GET/POST `/search`) now returns OpenClaw-compatible fields:
  - `fragment_id` — `{id}#{section-slug}:{sha256-prefix-12}`
  - `line_range`, `original_markdown`, span-level `content_hash`, `excerpt`, `stale`
  - `byte_offset` as `[start, end]` UTF-8 tuple
- Legacy `original_span` / `provenance_block` retained for backward compatibility
- Staleness: compares indexed `span_hash` to live span hash at query time

### Provenance confidence grade

- Optional frontmatter `confidence` (`expert-verified` | `crawled` | `inferred`) parsed in loader, stored on chunks, surfaced on results

### Already on main before this session

- Partial reconstruct mode (`2a87e66`, `1b27c4e`)
- Ontology/entity graph traversal (`d4d6023`)

### Not ep-mcp scope (authoring repo)

- TAC v1 — agent *response* envelope; consumes `fragment_id` from reconstruct results
- Strict validation gate, ingest-gate, pre-commit — expert-pack CI only

---

## Key design decisions (carry forward)

1. **Sidecar wins over heuristic split** — when `.chunks.yaml` exists, boundaries match `ep-chunk-annotate.py` exactly
2. **Reconstruct is post-retrieval** — no scoring/MMR/expansion changes; enrichment only after final result list
3. **Reindex required** — existing indexes lack `line_start/end` / `span_hash`; rebuild after deploy for accurate `fragment_id`
4. **Schema family 4.1** — additive only; no breaking manifest/frontmatter changes in ep-mcp

---

## Verification commands (Windows)

```powershell
cd C:\Users\BrianHearn\source\repos\ep-mcp
py -3.12 -m pip install -e ".[dev]" numpy -q
py -3.12 -m pytest tests/unit/test_sidecar.py tests/unit/test_reconstruct.py -v
py -3.12 -m pytest tests/unit -v --tb=line
```

Sidecar smoke against local expert-pack demo:

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

## Deploy checklist (production)

1. Deploy updated `ep_mcp/` to droplet (`scripts/deploy.sh` or rsync pattern in DEPLOYMENT.md)
2. **Reindex packs** that gained `.chunks.yaml` sidecars (blender-3d, home-assistant, solar-diy workflows/phases)
3. Smoke POST `/search` with `"reconstruct": true` — confirm `fragment_id` + `line_range` present
4. OpenClaw memory plugin `reconstruct: true` should now get full RFC-003 envelope end-to-end

---

## Known backlog (non-blocking)

- **Windows path keys in loader tests** — `test_loader.py` uses forward-slash keys; fails on Windows (`concepts/topic-a.md` vs `concepts\topic-a.md`); pre-existing, not introduced this session
- **ARCHITECTURE.md §5.9** — still describes pre-RFC-003 reconstruct fields; could be updated to list `fragment_id` / `line_range` / `stale`
- **Highlight rendering** — RFC-003 allows host-dependent highlighting; ep-mcp returns raw span only (no inline highlight markers)

---

## Likely next-session work

- Production reindex + live reconstruct smoke on expertpack.ai/mcp
- Update ARCHITECTURE.md reconstruct section to match RFC-003 envelope
- Optional: normalize pack file paths to forward slashes in loader (fix Windows test flake)
- Bump ep-mcp version / tag release when deploy verified
- Eval: measure retrieval impact of sidecar-boundary indexing vs heuristic split on demo packs

---

## Key file index (this session)

```
ep_mcp/pack/sidecar.py              # NEW — RFC-004 sidecar loader
ep_mcp/index/chunker.py             # sidecar-driven chunking + span metadata
ep_mcp/index/manager.py             # load sidecar at index time
ep_mcp/index/sqlite_store.py        # migrated columns + upsert fields
ep_mcp/pack/loader.py               # confidence grade
ep_mcp/pack/models.py               # Provenance.confidence
ep_mcp/retrieval/models.py          # RFC-003 SearchResult fields
ep_mcp/retrieval/engine.py          # fragment envelope + _search_result_from_chunk
ep_mcp/tools/ep_search.py           # MCP output fields
ep_mcp/server.py                    # HTTP output fields
tests/unit/test_sidecar.py          # NEW
tests/unit/test_reconstruct.py      # RFC-003 assertions
CHANGELOG.md                        # [Unreleased] entries
```

---

## Related repos

| Repo | Path | Today's commits |
|------|------|-----------------|
| expert-pack | `C:\Users\BrianHearn\source\repos\expert-pack` | `355257b`, `ede5d2c` — RFC-003/004 specs, sidecars, TAC, strict gate |
| ep-mcp | `C:\Users\BrianHearn\source\repos\ep-mcp` | this session — runtime consumption |

---

*End handoff.*
