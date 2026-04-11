# ExpertPack MCP — Vision

**Version:** 0.3 (Draft)
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

## Why ExpertPack + MCP

MCP matters because it makes the expertise service **agent-native and portable** — any MCP-compatible host (Claude, Cursor, Windsurf, custom apps) can connect without custom integration. The agent doesn't need to know how retrieval works; it just calls a tool and gets expert content back.

ExpertPack matters because it makes the knowledge itself **structured, verifiable, and maintainable** — not a pile of documents in a vector store, but a schema-validated pack with provenance, freshness guarantees, and graph relationships.

Together: ExpertPack structures expertise. ExpertPack MCP serves it to any agent. The result is an expertise-as-a-service layer that is portable across hosts, trustworthy in its sourcing, and maintainable over time.

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
- **Prompts** (roadmap) — Pack-aware prompt templates for common domain workflows (e.g., `/plan_territories`, `/compare_methods`). Not in first release; will be informed by real usage patterns.

**Does not contain:** Any domain-specific logic or tools. Knows about ExpertPacks, not about territories or 3D modeling or solar panels.

**Boundary rule:** ExpertPack MCP answers questions and exposes knowledge. Domain MCP servers perform domain actions. This separation is architectural, not negotiable.

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

**Goal:** A cloud-hosted EP MCP server that loads a pack and enables any MCP-compatible agent to get expert answers. Prove the pattern works end-to-end.

### In Scope (MVP)
- Load a valid ExpertPack from a configurable path
- Expose `ep_search` tool — hybrid retrieval returning content with provenance
- Expose pack manifest and file listing as MCP Resources
- **Streamable HTTP transport** (primary — cloud-hosted, any client can connect)
- stdio transport (secondary — for local dev/testing convenience)
- **Authentication** — API key auth (phase 1), OAuth 2.1 (phase 2, aligns with MCP spec)
- **Multi-pack architecture** — designed from day one, even if MVP exercises single-pack. Pack is a configuration parameter; routing, indexing, and resource namespaces are pack-scoped.
- Works with any MCP-compatible host — no host-specific code

### `ep_search` Tool — MVP Specification

**Input:**
- `query` (string, required) — natural language search query
- `type` (string, optional) — filter by EP file type (concept, workflow, reference, etc.)
- `tags` (string[], optional) — filter by EP tags
- `max_results` (integer, optional, default 10) — maximum results to return

**Output:** Ranked list of content chunks, each containing:
- `text` — content with frontmatter stripped (no retrieval tax)
- `source_file` — path within the pack
- `id` — provenance ID from frontmatter
- `content_hash` — for citation verification
- `verified_at` — freshness signal (ISO 8601)
- `score` — relevance score
- `type` / `tags` — file metadata for agent to assess relevance

**Retrieval pipeline:** Hybrid (vector embeddings + BM25 keyword), same proven approach as help bot eval (84.8% correctness, Run 12 baseline).

### Out of Scope (MVP)
- Domain-specific tools (that's the EZT MCP layer, not this repo)
- Prompts primitive (roadmap — informed by real usage patterns)
- Marketplace / registry integration
- Graph traversal tool (valuable but not MVP-critical)
- Composite pack support (designed for in multi-pack architecture, not exercised in MVP)

### MVP Success Criteria
1. Server runs as a cloud-hosted Streamable HTTP endpoint
2. Load the `ezt-designer` pack
3. Send MCP-compliant search queries over HTTP and receive expert answers
4. **Retrieval quality matches or exceeds help bot eval baselines** (84.8% correctness, ≤13% hallucination — Run 12 baseline, Sonnet judge)
5. Provenance metadata (id, content_hash, verified_at) included in all responses
6. Authentication enforced — unauthenticated requests rejected
7. Testable directly via HTTP (curl, Python, MCP client SDK) without requiring a UI host

---

## Technology Considerations

### MCP Spec Version
Target: **2025-11-25** (current stable). Next spec release expected ~June 2026.

### Transport
- **Primary:** Streamable HTTP — cloud-hosted, works behind reverse proxies and load balancers. Single HTTP endpoint handling POST and GET, optional SSE for streaming.
- **Secondary:** stdio — for local development and testing convenience. Same server logic, different transport layer.
- Both transports share the same server implementation; the MCP SDKs abstract the transport.

### Retrieval
- Hybrid search: vector embeddings + BM25 keyword matching
- EP frontmatter awareness: use `type`, `tags`, `pack`, `retrieval_strategy` for scoring
- Graph-aware: optionally follow `_graph.yaml` edges for related content
- Frontmatter stripped from returned content (retrieval tax — proven in eval Runs 11-12)

### SDK
TBD — Python (`mcp` SDK) or TypeScript. Decision deferred to Architecture phase.

### Authentication (Phase 1+)
- **Phase 1:** API key authentication (Authorization header). Simple, sufficient to prove the pattern and protect proprietary packs.
- **Phase 2:** OAuth 2.1 with PKCE (aligns with MCP 2025-11-25 spec). Supports SSO integration, per-pack scopes, enterprise-managed auth.
- Per-pack access control: different keys/scopes for different packs in a multi-pack deployment.

### Multi-Pack Architecture
Designed from day one:
- Pack is a configuration parameter, not hardcoded
- Each pack has its own retrieval index, resource namespace, and search scope
- Routing options: path-based (`/packs/{pack-name}/`), connection parameter, or subdomain
- Composite packs (multiple EPs combined into a single expertise surface) supported by the architecture, exercised in a future release
- Shared infrastructure (embedding engine, transport, auth) is pack-agnostic

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

## Testing Strategy

### Deployment Target
EZT Help Bot droplet (`64.225.0.26`) — already hosts the ezt-designer pack and has the infrastructure in place. EP MCP server will run alongside (or replace) the existing help bot.

### Primary: Direct HTTP Testing
Streamable HTTP is standard HTTP — testable directly without a UI host:
- **curl / Python requests** — raw JSON-RPC calls to validate protocol compliance
- **MCP Python client SDK** — full protocol simulation (capability negotiation, initialization handshake, tool calls)
- **Automated integration tests** — load pack, run search queries, validate retrieval quality against known-good answers
- Tests run from EasyBot droplet (`129.212.189.28`) against the help bot droplet

### Secondary: MCP Host Testing
- **Claude.ai (web)** — supports connecting to remote MCP servers over Streamable HTTP. Visual demo surface for human testing.
- **Claude Desktop** — stdio transport for local dev/testing.
- **Cursor / Windsurf** — IDE integration testing.

### MCP Host Landscape (as of April 2026)

| Host | Transport | Notes |
|------|-----------|-------|
| Claude.ai (web) | Streamable HTTP | Anthropic web client — connects to remote MCP servers |
| Claude Desktop | stdio | Anthropic desktop app — most popular MCP host |
| Cursor | stdio | IDE with MCP support |
| Windsurf (Codeium) | stdio | IDE with MCP support |
| Continue.dev | stdio | Open-source IDE extension |
| Custom apps | Both | Anyone building with MCP client SDKs |

---

## What's Next

1. ✅ Vision (this document)
2. ⬜ Architecture — server structure, SDK choice, retrieval pipeline design, resource/tool schemas, transport details
3. ⬜ Implementation — MVP build
4. ⬜ Validation — end-to-end HTTP testing with ezt-designer pack, then Claude.ai demo
5. ⬜ EasyTerritory MCP — domain layer with EZT Cloud API tools
