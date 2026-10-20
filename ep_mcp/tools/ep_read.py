"""ep_read MCP tool — load a whole atom by path or provenance id."""

from __future__ import annotations

import logging
import re

from ..index.chunker import heading_path_at
from ..pack.models import Pack, PackFile
from ..retrieval.freshness import freshness_for_file

logger = logging.getLogger(__name__)


def ep_read(
    pack: Pack,
    path: str | None = None,
    id: str | None = None,
    reconstruct: bool = False,
    section: str | None = None,
    neighbors: bool = False,
) -> dict:
    """Read a whole ExpertPack atom.

    Args:
        pack: Loaded pack
        path: Pack-relative file path (forward slashes)
        id: Provenance id (e.g. pack/concepts/topic)
        reconstruct: Include raw markdown and provenance envelope
        section: Optional section slug (e.g. 'setup') or heading text; returns
            only that ## / ### section (with its sub-sections) as ``content``.
        neighbors: With ``section``, also return the previous/next sibling
            sections and the parent heading path (for hits that need context).

    Returns:
        Atom payload, or an error dict if the file cannot be resolved.
    """
    if not path and not id:
        return {"error": "Provide path or id"}

    pack_file = None
    if path:
        pack_file = _resolve_by_path(pack, path)
    if pack_file is None and id:
        pack_file = _resolve_by_id(pack, id)

    if pack_file is None:
        logger.info("ep_read miss | pack=%s path=%r id=%r", pack.slug, path, id)
        return {"error": "File not found in pack", "path": path, "id": id}

    payload = {
        "path": pack_file.path,
        "title": pack_file.title,
        "type": pack_file.type,
        "tags": pack_file.tags,
        "content": pack_file.content,
        "id": pack_file.provenance.id,
        "content_hash": pack_file.provenance.content_hash,
        "verified_at": pack_file.provenance.verified_at,
        "verified_by": pack_file.provenance.verified_by,
        "confidence": pack_file.provenance.confidence,
        "retrieval_strategy": pack_file.retrieval_strategy,
        "concept_scope": pack_file.concept_scope,
        "requires": pack_file.requires,
        "activation": _activation_dict(pack_file),
        "sections": [
            {"slug": sec["slug"], "heading": sec["heading"], "level": sec["level"],
             "heading_path": sec["heading_path"], "line_range": sec["line_range"]}
            for sec in _sections(pack_file)
        ],
        **freshness_for_file(pack, pack_file),
    }
    if section:
        secs = _sections(pack_file)
        idx = _find_section(secs, section)
        if idx is None:
            return {
                "error": "Section not found",
                "path": pack_file.path,
                "section": section,
                "available_sections": [sec["slug"] for sec in secs],
            }
        sec = secs[idx]
        payload["content"] = sec["text"]
        payload["section"] = sec["slug"]
        payload["heading_path"] = sec["heading_path"]
        payload["line_range"] = sec["line_range"]
        if neighbors:
            payload["neighbors"] = {
                "previous": _section_ref(secs[idx - 1]) if idx > 0 else None,
                "next": _section_ref(secs[idx + 1]) if idx + 1 < len(secs) else None,
                "previous_text": secs[idx - 1]["text"] if idx > 0 else None,
                "next_text": secs[idx + 1]["text"] if idx + 1 < len(secs) else None,
            }
    if reconstruct:
        payload["original_markdown"] = pack_file.raw_content
        payload["original_span"] = pack_file.raw_content
        payload["provenance_block"] = {
            "id": pack_file.provenance.id,
            "source_file": pack_file.path,
            "content_hash": pack_file.provenance.content_hash,
            "verified_at": pack_file.provenance.verified_at,
            "verified_by": pack_file.provenance.verified_by,
            "confidence": pack_file.provenance.confidence,
        }

    logger.info(
        "ep_read | pack=%s path=%s reconstruct=%s",
        pack.slug, pack_file.path, reconstruct,
    )
    return payload


def _resolve_by_path(pack: Pack, path: str) -> PackFile | None:
    key = path.replace("\\", "/").lstrip("/")
    if key in pack.files:
        return pack.files[key]
    lower = key.lower()
    for fp, f in pack.files.items():
        if fp.lower() == lower:
            return f
    return None


def _resolve_by_id(pack: Pack, prov_id: str) -> PackFile | None:
    for f in pack.files.values():
        if f.provenance.id == prov_id:
            return f
    return None


def _activation_dict(pack_file: PackFile) -> dict | None:
    if pack_file.activation is None:
        return None
    return {
        "tools": pack_file.activation.tools,
        "constraints": pack_file.activation.constraints,
        "next": pack_file.activation.next,
    }


_SEC_RE = re.compile(r"^(#{2,3})\s+(.+?)\s*#*\s*$")
_FENCE = re.compile(r"^\s*(```|~~~)")


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "section"


def _sections(pack_file: PackFile) -> list[dict]:
    """Split a file into ## / ### sections (line numbers relative to raw markdown).

    A ## section's text includes its ### children; each ### is also its own entry.
    Index 0 is the opening (text before the first ## / ###) when non-empty.
    """
    raw = pack_file.raw_content or pack_file.content
    lines = raw.splitlines()
    body_start = 0
    if lines and lines[0].strip() == "---":
        for j in range(1, len(lines)):
            if lines[j].strip() == "---":
                body_start = j + 1
                break
    heads: list[tuple[int, int, str]] = []  # (line_idx, level, text)
    in_fence = False
    for i in range(body_start, len(lines)):
        if _FENCE.match(lines[i]):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        m = _SEC_RE.match(lines[i])
        if m:
            heads.append((i, len(m.group(1)), m.group(2).strip()))
    out: list[dict] = []
    first = heads[0][0] if heads else len(lines)
    opening = "\n".join(lines[body_start:first]).strip()
    if opening:
        out.append({
            "slug": "opening", "heading": pack_file.title, "level": 1,
            "heading_path": [pack_file.title] if pack_file.title else [],
            "line_range": [body_start + 1, first], "text": opening,
        })
    seen: dict[str, int] = {}
    for k, (li, level, text) in enumerate(heads):
        end = len(lines)
        for lj, lvl2, _ in heads[k + 1:]:
            if lvl2 <= level:
                end = lj
                break
        slug = _slug(text)
        if slug in seen:
            seen[slug] += 1
            slug = f"{slug}-{seen[slug]}"
        else:
            seen[slug] = 1
        out.append({
            "slug": slug, "heading": text, "level": level,
            "heading_path": heading_path_at(raw, li + 1, pack_file.title),
            "line_range": [li + 1, end],
            "text": "\n".join(lines[li:end]).strip(),
        })
    return out


def _find_section(secs: list[dict], wanted: str) -> int | None:
    w = wanted.strip().lstrip("#").strip()
    for i, sec in enumerate(secs):
        if sec["slug"] == w or (sec["heading"] or "").lower() == w.lower():
            return i
    ws = _slug(w)
    for i, sec in enumerate(secs):
        if sec["slug"] == ws:
            return i
    return None


def _section_ref(sec: dict) -> dict:
    return {"slug": sec["slug"], "heading": sec["heading"], "line_range": sec["line_range"]}
