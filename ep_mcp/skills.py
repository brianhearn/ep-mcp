"""Skills over MCP (extension ``io.modelcontextprotocol/skills``, SEP-2640 Final).

Opt-in per pack via ``mcp.skills.enabled: true`` in manifest.yaml (default off).
Each pack is served as ONE skill named ``<slug>``:

* ``skill://<slug>/SKILL.md`` — generated: frontmatter ``name`` + ``description``,
  body = consume loop, authority/refusal rules and a path map.
* ``skill://<slug>/<relative-path>.md`` — every non-on_demand pack markdown file,
  served verbatim (frontmatter included).

``skills/list`` / ``skills/get`` return entries shaped per the published spec
(``uri``, ``frontmatter``, ``resources: [{uri, digest: "sha256:<hex>", size}]``)
with result-level ``resultType: "complete"``, ``ttlMs`` and ``cacheScope``
(``"public"`` | ``"private"``). Digests and sizes are computed over the exact
UTF-8 bytes served by ``resources/read``. The ep_* tools remain the primary path.
"""

from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel

from .pack.models import Pack
from .retrieval.freshness import parse_duration_days

logger = logging.getLogger(__name__)

EXTENSION_ID = "io.modelcontextprotocol/skills"
DEFAULT_TTL_MS = 3_600_000
MAX_SKILL_FILES = 512          # spec: servers SHOULD NOT exceed 512 files / 16 MiB per skill
MAX_SKILL_BYTES = 16 * 1024 * 1024
MAX_DESCRIPTION = 1000         # Agent Skills caps description at 1024 chars
_NAME_RE = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")


def skills_enabled(pack: Pack) -> bool:
    return bool(pack.mcp_config.skills.enabled)


def _skill_name(slug: str) -> str:
    name = re.sub(r"[^a-z0-9-]+", "-", slug.lower()).strip("-")
    name = re.sub(r"-{2,}", "-", name)[:64].strip("-")
    return name or "pack"


def skill_description(pack: Pack) -> str:
    cfg = pack.mcp_config.skills
    if cfg.description:
        desc = cfg.description
    else:
        parts = [pack.description.strip() or f"{pack.name} ExpertPack."]
        b = pack.manifest.authority_boundary
        if b is not None and b.in_scope:
            parts.append(f"Use for: {b.in_scope.strip()}")
        intended = pack.manifest.raw.get("intended_use")
        if isinstance(intended, str) and intended.strip():
            parts.append(intended.strip())
        desc = " ".join(" ".join(parts).split())
    if len(desc) > MAX_DESCRIPTION:
        desc = desc[: MAX_DESCRIPTION - 1].rstrip() + "…"
    return desc


def skill_file_paths(pack: Pack) -> list[str]:
    """Pack-relative paths served as supporting files (on_demand excluded)."""
    paths = [
        p for p, f in sorted(pack.files.items())
        if f.retrieval_strategy != "on_demand" and p != "SKILL.md"
    ]
    if len(paths) > MAX_SKILL_FILES - 1:
        logger.warning(
            "Pack '%s': %d skill files exceed the %d-file limit; truncating (always-tier first)",
            pack.slug, len(paths), MAX_SKILL_FILES,
        )
        always = [p for p in paths if p in pack.always_tier]
        rest = [p for p in paths if p not in pack.always_tier]
        paths = sorted((always + rest)[: MAX_SKILL_FILES - 1])
    return paths


def generate_skill_md(pack: Pack, paths: list[str] | None = None) -> str:
    name = _skill_name(pack.slug)
    fm = {"name": name, "description": skill_description(pack)}
    paths = skill_file_paths(pack) if paths is None else paths
    lines = [
        "---",
        yaml.safe_dump(fm, sort_keys=False, allow_unicode=True, width=10_000).strip(),
        "---",
        "",
        f"# {pack.name}",
        "",
        "## How to use this pack",
        "",
        (
            "1. Prefer the server's tools when available: `ep_search` for candidate atoms, then "
            "`ep_read` (optionally `section=`) to load the whole atom."
        ),
        "2. Without tools, read the files below relative to this skill (e.g. `overview.md`).",
        "3. Stop when an atom answers the question, or after 3 steps (hard cap 7).",
        "4. Caveat answers from atoms flagged `age_stale` or `expired`.",
        "",
    ]
    b = pack.manifest.authority_boundary
    if b is not None:
        lines += ["## Authority", ""]
        if b.in_scope:
            lines.append(f"- In scope: {b.in_scope.strip()}")
        for item in b.out_of_scope:
            lines.append(f"- Refuse: {item}")
        for item in b.refuse_when:
            lines.append(f"- Refuse when: {item}")
        if b.no_source_no_claim:
            lines.append("- Do not assert a claim unless a pack file supports it.")
        lines.append("")
    lines += ["## Path map", ""]
    for p in paths:
        f = pack.files[p]
        title = (f.title or p).replace("\n", " ")
        typ = f" ({f.type})" if f.type else ""
        lines.append(f"- [{title}]({p}){typ}")
    lines.append("")
    return "\n".join(lines)


class SkillCatalog:
    """Builds the skill entry and serves skill:// bytes for one pack."""

    def __init__(self, pack: Pack, private: bool = False):
        self.pack = pack
        self.name = _skill_name(pack.slug)
        self.root = f"skill://{self.name}"
        self.private = private

    @property
    def skill_uri(self) -> str:
        return f"{self.root}/SKILL.md"

    @property
    def cache_scope(self) -> str:
        return "private" if self.private else "public"

    @property
    def ttl_ms(self) -> int:
        cfg = self.pack.mcp_config.skills
        if cfg.ttl_ms is not None:
            return max(0, int(cfg.ttl_ms))
        ttls = [
            d * 86_400_000 for f in self.pack.files.values()
            if f.refresh and (d := parse_duration_days(f.refresh))
        ]
        return min([DEFAULT_TTL_MS, *ttls]) if ttls else DEFAULT_TTL_MS

    def _file_text(self, path: str) -> str:
        disk = Path(self.pack.pack_dir) / path
        try:
            return disk.read_bytes().decode("utf-8", errors="replace")
        except OSError:
            return self.pack.files[path].raw_content

    def read(self, rel: str) -> str:
        """Text served for skill://<name>/<rel>; raises KeyError for unknown files."""
        rel = rel.replace("\\", "/").lstrip("/")
        paths = skill_file_paths(self.pack)
        if rel == "SKILL.md":
            return generate_skill_md(self.pack, paths)
        if rel not in paths:
            raise KeyError(rel)
        return self._file_text(rel)

    def entry(self) -> dict[str, Any]:
        paths = skill_file_paths(self.pack)
        skill_md = generate_skill_md(self.pack, paths)
        resources = [_manifest_item(self.skill_uri, skill_md)]
        total = resources[0]["size"]
        for p in paths:
            item = _manifest_item(f"{self.root}/{p}", self._file_text(p))
            if total + item["size"] > MAX_SKILL_BYTES:
                logger.warning("Pack '%s': skill exceeds 16 MiB; truncating manifest", self.pack.slug)
                break
            total += item["size"]
            resources.append(item)
        return {
            "uri": self.skill_uri,
            "frontmatter": {"name": self.name, "description": skill_description(self.pack)},
            "resources": resources,
        }


def _manifest_item(uri: str, text: str) -> dict[str, Any]:
    data = text.encode("utf-8")
    return {"uri": uri, "digest": "sha256:" + hashlib.sha256(data).hexdigest(), "size": len(data)}


# ── MCP wiring ────────────────────────────────────────────────────────────


class _ListParams(BaseModel):
    cursor: str | None = None
    model_config = {"extra": "allow"}


class _GetParams(BaseModel):
    uri: str
    model_config = {"extra": "allow"}


def build_skills_extension(catalogs: dict[str, SkillCatalog]):
    """Return an SDK Extension serving skills/list and skills/get for the catalogs."""
    from mcp.server.extension import Extension, MethodBinding
    from mcp.shared.exceptions import MCPError

    def _scope() -> str:
        return "private" if any(c.private for c in catalogs.values()) else "public"

    def _ttl() -> int:
        return min((c.ttl_ms for c in catalogs.values()), default=DEFAULT_TTL_MS)

    async def list_skills(ctx, params) -> dict:
        return {
            "resultType": "complete",
            "skills": [catalogs[k].entry() for k in sorted(catalogs)],
            "ttlMs": _ttl(),
            "cacheScope": _scope(),
        }

    async def get_skill(ctx, params) -> dict:
        for cat in catalogs.values():
            if params.uri == cat.skill_uri:
                return {
                    "resultType": "complete",
                    "skill": cat.entry(),
                    "ttlMs": cat.ttl_ms,
                    "cacheScope": cat.cache_scope,
                }
        raise MCPError(-32602, f"Unknown skill: {params.uri}")

    class SkillsExtension(Extension):
        identifier = EXTENSION_ID

        def settings(self) -> dict[str, Any]:
            return {}  # directoryRead not supported (defaults to false)

        def methods(self):
            return (
                MethodBinding("skills/list", _ListParams, list_skills),
                MethodBinding("skills/get", _GetParams, get_skill),
            )

    return SkillsExtension()


def register_skill_resources(mcp, catalog: SkillCatalog) -> None:
    """Register skill://<name>/{+path} on the MCPServer (same files ep_read serves)."""

    @mcp.resource(f"{catalog.root}/{{+path}}", mime_type="text/markdown")
    async def read_skill_file(path: str) -> str:
        """ExpertPack skill file (SKILL.md or a pack markdown file, verbatim)."""
        try:
            return catalog.read(path)
        except KeyError:
            raise ValueError(f"Unknown skill file: {catalog.root}/{path}") from None
