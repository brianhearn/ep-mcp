"""Pydantic models for ExpertPack data structures."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Provenance(BaseModel):
    """Provenance metadata from file frontmatter."""

    id: str | None = None
    content_hash: str | None = None
    verified_at: str | None = None
    verified_by: str | None = None
    confidence: str | None = Field(
        None,
        description="Provenance grade: expert-verified | crawled | inferred",
    )


class Activation(BaseModel):
    """Optional knowledge-activation block on operational atoms."""

    tools: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    next: list[str] = Field(default_factory=list)


class AuthorityBoundary(BaseModel):
    """Pack-level refusal contract from manifest.yaml."""

    in_scope: str = ""
    out_of_scope: list[str] = Field(default_factory=list)
    refuse_when: list[str] = Field(default_factory=list)
    no_source_no_claim: bool = False


_NAV_BASENAMES = frozenset({"_index.md", "source-coverage.md"})
_OPERATIONAL_TYPES = frozenset({"workflow", "decision", "gotcha", "phase"})


class PackFile(BaseModel):
    """A single content file within an ExpertPack."""

    path: str = Field(description="Relative path within pack")
    title: str | None = None
    type: str | None = Field(
        None, description="Content type: concept, workflow, reference, etc."
    )
    tags: list[str] = Field(default_factory=list)
    provenance: Provenance = Field(default_factory=Provenance)
    retrieval_strategy: str = Field(
        "standard",
        description="standard | always | on_demand | atomic | navigation",
    )
    concept_scope: str | None = Field(
        None, description="single | reference | multi | navigation"
    )
    requires: list[str] = Field(
        default_factory=list,
        description="Atomic-conceptual: paths of atoms that should auto-expand when this atom is retrieved. Directional (A requires B does not imply B requires A).",
    )
    activation: Activation | None = Field(
        None,
        description="Optional tools/constraints/next on operational atoms",
    )
    content: str = Field("", description="Markdown content, frontmatter stripped")
    raw_content: str = Field("", description="Full markdown content with frontmatter")
    size_tokens: int = Field(0, description="Approximate token count")

    @property
    def is_indexable(self) -> bool:
        """Whether this file belongs in the RAG (vector + FTS) pool."""
        posix = self.path.replace("\\", "/")
        basename = posix.rsplit("/", 1)[-1]
        if basename in _NAV_BASENAMES or posix.endswith("meta/source-coverage.md"):
            return False
        if self.retrieval_strategy in {"navigation", "on_demand"}:
            return False
        if self.concept_scope == "navigation":
            return False
        return True

    @property
    def is_operational(self) -> bool:
        return (self.type or "") in _OPERATIONAL_TYPES


class FreshnessMetadata(BaseModel):
    """Pack-level freshness metadata from manifest.yaml."""

    refresh_cycle: str | None = None
    coverage_pct: float | None = None
    last_full_review: str | None = None
    decay_rate: str | None = None


class ContextTiers(BaseModel):
    """Context loading tiers from manifest.yaml."""

    always: list[str] = Field(default_factory=list)
    on_demand: list[str] = Field(default_factory=list)


# --- MCP configuration models (defined before Manifest so Manifest can reference them) ---

class MCPPromptDeclaration(BaseModel):
    """A single prompt declared in manifest mcp.prompts."""

    name: str = Field(description="Snake_case prompt name exposed to MCP clients")
    description: str = Field(description="One-line description shown during registration")
    source: str = Field(description="Relative path to the workflow file")


class MCPResourcesConfig(BaseModel):
    """Resources sub-config from manifest mcp.resources."""

    include_always_tier: bool = Field(
        True, description="Expose context.always files as MCP Resources"
    )
    additional: list[str] = Field(
        default_factory=list, description="Extra files to expose beyond always tier"
    )


class MCPConfig(BaseModel):
    """Parsed mcp block from manifest.yaml."""

    instructions: str | None = Field(
        None, description="Server instructions= string for MCP registration"
    )
    prompts: list[MCPPromptDeclaration] = Field(
        default_factory=list, description="Explicit prompt declarations"
    )
    resources: MCPResourcesConfig = Field(default_factory=MCPResourcesConfig)


# --- Manifest ---

class Manifest(BaseModel):
    """Parsed manifest.yaml for an ExpertPack."""

    slug: str
    name: str
    type: str = Field(description="person | product | process | composite")
    version: str = "1.0.0"
    description: str = ""
    entry_point: str = "overview.md"
    schema_version: str = ""
    context: ContextTiers = Field(default_factory=ContextTiers)
    freshness: FreshnessMetadata = Field(default_factory=FreshnessMetadata)
    mcp: MCPConfig = Field(default_factory=MCPConfig)
    authority_boundary: AuthorityBoundary | None = None
    raw: dict = Field(default_factory=dict, description="Full parsed YAML for passthrough")


class GraphEdge(BaseModel):
    """A single edge in the knowledge graph."""

    source: str
    target: str
    kind: str = Field(description="wikilink | related | context")
    weight: float = 1.0


class PackGraph(BaseModel):
    """Knowledge graph parsed from _graph.yaml."""

    nodes: list[str] = Field(default_factory=list)
    edges: list[GraphEdge] = Field(default_factory=list)

    def neighbors(self, file_path: str, edge_kinds: list[str] | None = None) -> list[str]:
        """Get neighboring nodes for a given file path."""
        results = []
        for edge in self.edges:
            if edge_kinds and edge.kind not in edge_kinds:
                continue
            if edge.source == file_path:
                results.append(edge.target)
            elif edge.target == file_path:
                results.append(edge.source)
        return results


class Pack(BaseModel):
    """A loaded ExpertPack ready for indexing and serving."""

    slug: str
    name: str
    type: str = Field(description="person | product | process | composite")
    version: str
    description: str = ""
    entry_point: str = "overview.md"
    files: dict[str, PackFile] = Field(default_factory=dict, description="path -> PackFile")
    manifest: Manifest
    graph: PackGraph | None = None
    freshness: FreshnessMetadata | None = None
    mcp_config: MCPConfig = Field(default_factory=MCPConfig)
    always_tier: list[str] = Field(
        default_factory=list, description="Resolved file paths in context.always tier"
    )
    index_path: str = Field("", description="Path to SQLite index file")
    pack_dir: str = Field("", description="Absolute path to pack directory")
