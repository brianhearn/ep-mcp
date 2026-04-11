# ExpertPack MCP — Vision

**Version:** 0.1 (Draft)
**Date:** 2026-04-11
**Authors:** Brian Hearn, EasyBot

---

## Thesis

ExpertPack MCP is an expertise-as-a-service layer built on the Model Context Protocol. It makes Esoteric Knowledge (EK) — domain expertise that lives outside LLM weights and isn't easily searchable on the internet — available to any MCP-compatible agent over a standard protocol.

An ExpertPack is a structured, schema-validated knowledge pack with provenance metadata, graph relationships, and freshness signals. ExpertPack MCP turns any EP into a live service that agents can connect to for retrieval-augmented generation and, in domain-specific extensions, tool execution.

**One pack. Any agent. Expert answers.**

---

## The Problem

LLMs are generalists. They know a lot about everything, but they lack the deep, current, verified domain expertise that professionals rely on. This expertise:

- Lives in senior engineers' heads, internal wikis, tribal knowledge
- Changes faster than model training cycles
- Is too specialized for general web search to surface reliably
- Is often proprietary or commercially valuable

Today, the options for giving an agent domain expertise are:
1. **Fine-tuning** — expensive, slow, doesn't handle freshness
2. **Naive RAG** — dump docs into a vector store, hope the right chunks surface
3. **Manual context stuffing** — copy-paste into prompts, doesn't scale

None of these treat expertise as a first-class, structured, portable artifact.

---

## The Solution

ExpertPack MCP serves structured domain expertise over the Model Context Protocol.

### Core Loop

1. **User prompts their agent** (Claude Desktop, Cursor, Windsurf, custom app — any MCP host)
2. **Agent queries the EP MCP server** with the user's question
3. **EP MCP performs schema-aware retrieval** — hybrid search (vector + keyword), graph-aware traversal, EP-native chunking
4. **Returns expert content** — with provenance (source file, content hash, verification date), citation metadata, and freshness signals
5. **Agent responds** with grounded, expert-level answers that cite their sources

### What Makes This Different From "Just RAG"

- **Schema-validated knowledge** — not a pile of scraped docs. Typed files with known structure, relationships, and freshness guarantees.
- **Provenance** — every chunk carries `id`, `content_hash`, `verified_at`, `verified_by`. Agents can cite sources properly and consumers can verify claims.
- **Graph-aware retrieval** — EP's `_graph.yaml` adjacency layer means related content follows wikilinks and structural relationships, not just embedding similarity.
- **EP-native chunking** — files are authored as retrieval units (schema-as-chunker). No lossy post-hoc splitting that breaks context.
- **Pack portability** — same server, different pack, different domain. Swap `ezt-designer` for `blender-3d` and you have a different expertise service running.
- **Freshness as a feature** — provenance metadata and manifest-level freshness SLAs mean the agent (and the user) know how current the knowledge is.

---

## Architecture: Two Layers

### Layer 1: ExpertPack MCP (Generic)

The open-source core. Loads any valid ExpertPack and exposes it over MCP.

**Capabilities:**
- **Tools** — `ep_search` (hybrid retrieval with provenance), `ep_graph_traverse` (follow relationships), `ep_list_topics` (pack structure discovery)
- **Resources** — EP files as browsable resources with URI scheme (`ep://{pack}/{path}`), manifest as root resource, frontmatter as resource annotations
- **Prompts** — Pack-aware prompt templates (optional, pack-defined)

**Does not contain:** Any domain-specific logic or tools. Knows about ExpertPacks, not about territories or 3D modeling or solar panels.

### Layer 2: Domain MCP Servers (Domain-Specific)

Built on top of ExpertPack MCP. Loads a specific pack and adds domain tools.

**Example: EasyTerritory MCP**
- Loads the `ezt-designer` ExpertPack for RAG (territory planning expertise)
- Adds EZT Cloud API tools: `auto_build_territories`, `create_territory`, `geocode_accounts`, `cluster_locations`, etc.
- The agent uses RAG to reason about *what* to do, then tools to *do* it

**The conversation arc:**

```
User: "Build 10 territories for my Southeast sales team"

Agent → EP MCP (RAG): retrieves territory planning concepts
Agent → User: "Should these be polygon-based (ZIP/county) or account clusters?
               Here's when you'd use each..." [cites EP sources]

User: "Clusters, balanced by revenue"

Agent → EP MCP (RAG): retrieves auto-build best practices, balance metrics
Agent → EZT MCP (Tool): auto_build_territories({
    method: "cluster", count: 10, balance: "revenue", ...
})

Agent → EP MCP (RAG): validates results against planning heuristics
Agent → User: "Done. Territory 3 is 40% above average revenue —
               typically indicates a coverage gap. Want me to rebalance?"
```

This is what separates an expert agent from a dumb API client. The EK is what makes the tool calls intelligent.

---

## User Tiers

### Tier 1 — Internal Agents (Primary, MVP)
Companies building agents powered by their own proprietary expertise. EasyTerritory building an agent that plans territories using the ezt-designer EP. A medical device company building an agent that configures equipment using their internal EP.

### Tier 2 — Customer-Facing Expertise Services
Companies exposing their domain expertise as a service to their customers' agents. EasyTerritory customers connect their planning agents to the EZT EP MCP endpoint. The expertise becomes part of the product offering.

### Tier 3 — Expertise Marketplace (Future)
A registry of EP MCP servers — discoverable expertise endpoints. Developers publish packs, they become live services. Agents discover and connect to the expertise they need dynamically.

- Discoverable via MCP Server Cards (`.well-known` metadata — on the 2026 MCP roadmap)
- Monetization: per-query, subscription, bundled with product
- Quality signals: EP validation scores, freshness SLAs, provenance coverage

---

## MVP Scope

**Goal:** A single EP MCP server that loads one pack and enables an agent to get expert answers. Prove the pattern works end-to-end.

### In Scope (MVP)
- Load a valid ExpertPack from a local path
- Expose `ep_search` tool — hybrid retrieval returning content with provenance
- Expose pack manifest and file listing as MCP Resources
- stdio transport (works with Claude Desktop, Cursor, etc. out of the box)
- Works with any MCP-compatible host — no host-specific code

### Out of Scope (MVP)
- Streamable HTTP transport (needed for cloud/Tier 2+, but not for proving the pattern)
- Domain-specific tools (that's the EZT MCP layer, not this repo)
- Multi-pack serving / multi-tenancy
- Authentication / access control
- Prompts primitive
- Marketplace / registry integration
- Graph traversal tool (valuable but not MVP-critical)

### MVP Success Criteria
1. Connect Claude Desktop (or equivalent MCP host) to the EP MCP server
2. Load the `ezt-designer` pack
3. Ask territory planning questions
4. Get accurate, sourced answers that an agent without the EP couldn't produce
5. Provenance metadata included in responses

---

## Technology Considerations

### MCP Spec Version
Target: **2025-11-25** (current stable). Next spec release expected ~June 2026.

### Transport
- **MVP:** stdio (simplest, broadest host compatibility)
- **Tier 2+:** Streamable HTTP (required for cloud deployment, horizontal scaling)

### Retrieval
- Hybrid search: vector embeddings + BM25 keyword matching
- EP frontmatter awareness: use `type`, `tags`, `pack`, `retrieval_strategy` for scoring
- Graph-aware: optionally follow `_graph.yaml` edges for related content
- Frontmatter stripped from returned content (retrieval tax — proven in eval Runs 11-12)

### SDK
TBD — Python (`mcp` SDK) or TypeScript. Decision deferred to Architecture phase.

### Key MCP Features to Leverage
- **Resource annotations** (`audience`, `priority`, `lastModified`) — map directly to EP frontmatter
- **Structured tool output** (`outputSchema` + `structuredContent`) — typed retrieval results
- **Tool annotations** — mark all EP MCP tools as read-only
- **Tasks** (experimental) — future use for expensive graph traversal or cross-pack queries

### MCP Roadmap Items to Watch
- **Server Cards** (`.well-known` discoverability) — enables Tier 3 marketplace
- **Reference-based results** — return URIs instead of inlining large content (huge for EPs)
- **Skills primitive** — composed capabilities, directly relevant
- **Triggers/events** — server-initiated notifications when pack content updates

---

## Relationship to Existing Work

| Component | Role |
|-----------|------|
| ExpertPack schema (`ExpertPack` repo) | Defines the knowledge pack format — what gets served |
| ExpertPack packs (`ExpertPacks` repo) | The actual knowledge — what powers specific expertise services |
| EP tooling (`ep-validate`, `ep-graph-export`, etc.) | Pack authoring & validation — ensures quality of what gets served |
| **ExpertPack MCP (this repo)** | **The serving layer — turns a pack into a live expertise service** |
| EasyTerritory MCP (future) | Domain extension — adds EZT tools on top of EP MCP + ezt-designer pack |
| ClawHub `expertpack` skill | OpenClaw-specific skill for working with EPs — complementary, not replaced |

---

## What's Next

1. ✅ Vision (this document)
2. ⬜ Architecture — server structure, SDK choice, retrieval pipeline design, resource/tool schemas, transport details
3. ⬜ Implementation — MVP build
4. ⬜ Validation — end-to-end test with ezt-designer pack in Claude Desktop
5. ⬜ EasyTerritory MCP — domain layer with EZT Cloud API tools
